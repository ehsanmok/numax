"""Solves against triangular and tridiagonal matrices.
`scipy.linalg`'s `solve_triangular` and the banded solvers.

**Tier 1** for the `Array` substitutions: fixed trip count, no per-lane
branching, so all of it launches inside a GPU thread at any conformer.
These are the primitives the factorizations in `cholesky`, `lu` and `qr`
finish with.

MAX has no triangular solve at any size -- no `trsm` outside the private
cuBLAS/rocBLAS FFI -- so nothing here delegates. The blocked, device-
resident `_trsv` pair is numax's too: each step solves one `block x block`
diagonal system with a `numax.linalg.panel` kernel and then updates the
rest of the vector with a `gemv_sub`, which is where the `O(n^2)` is and
which MAX's `elementwise` parallelizes. `TensorLU.solve` is their caller.

`tridiagonal_solve` is Thomas, `O(n)` rather than the `O(n^3)` a general
solve costs, which is what makes cubic splines and implicit 1-D PDE steps
tractable. It will not gain a blocked `Tensor` form: Thomas is already
linear and has nothing to hand a GEMM.
"""

from layout import Coord, TileTensor
from layout.tile_layout import TensorLayout, row_major
from linalg.matmul import matmul as _max_matmul
from max.gpu.host import DeviceContext
from std.collections import Array
from std.sys.info import align_of
from std.utils import IndexList

from ..core.array import Shaped, zeros_dyn
from ..core.numeric import FloatLike, guard_nonzero

from .blas import _target
from .common import _PIVOT_FLOOR, _Dense, _zeros
from .panel import (
    _View,
    gemv_sub,
    pack_block,
    pack_vector,
    trsm_diag,
    trsv_diag,
)


def forward_substitution[
    T: FloatLike, n: Int, unit_diagonal: Bool = False
](lower: Array[T, n * n], b: Array[T, n]) -> Array[T, n]:
    """Solve `L @ x = b` for lower-triangular `L`.

    `unit_diagonal=True` treats `L`'s diagonal as an implicit `1` without
    reading it, which is what the packed output of `lu` needs.

    No MAX equivalent at any size: MAX ships no triangular solve (no
    `trsm`, no BLAS-1 at all). This loop is the whole algorithm.
    """
    var x = _zeros[T, n]()
    for i in range(n):
        var total = b[i].copy()
        for j in range(i):
            total = total - (lower[i * n + j] * x[j])
        comptime if unit_diagonal:
            x[i] = total^
        else:
            x[i] = total / guard_nonzero(
                lower[i * n + i], T.constant(_PIVOT_FLOOR)
            )
    return x^


def back_substitution[
    T: FloatLike, n: Int
](upper: Array[T, n * n], b: Array[T, n]) -> Array[T, n]:
    """Solve `U @ x = b` for upper-triangular `U`.

    No MAX equivalent at any size, same as `forward_substitution`.
    """
    var x = _zeros[T, n]()
    for step in range(n):
        var i = n - 1 - step
        var total = b[i].copy()
        for j in range(i + 1, n):
            total = total - (upper[i * n + j] * x[j])
        x[i] = total / guard_nonzero(upper[i * n + i], T.constant(_PIVOT_FLOOR))
    return x^


def _trsv[
    dtype: DType,
    ALayout: TensorLayout,
    XLayout: TensorLayout,
    upper: Bool,
    unit: Bool,
    trans: Bool = False,
    gpu: Bool = False,
](
    a: _View[dtype, ALayout],
    x: _View[dtype, XLayout],
    n: Int,
    block: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """Solve a triangular system against a vector, in place, blocked and
    device-resident.

    `upper` picks back substitution over forward; `unit` says the stored
    diagonal is not the triangle's, which is what the packed `L` of an LU
    needs; `trans` reads the triangle transposed, which is what a Cholesky
    solve's second half needs. The triangle is read where it lies -- a strided block is fine
    here, because both steps address it themselves rather than handing it
    to `matmul`.

    Two launches per block step: `trsv_diag` for the diagonal system,
    which is sequential and small, and `gemv_sub` for the update to the
    rest of the vector, which is `O(n^2)` overall and parallel over rows.
    Nothing crosses to the host.
    """
    var steps = (n + block - 1) // block
    for step in range(steps):
        var k = (n - (step + 1) * block) if upper else (step * block)
        var nb = block
        if upper:
            if k < 0:
                nb = block + k
                k = 0
        else:
            nb = min(block, n - k)

        comptime if gpu:
            ctx.enqueue_function[
                trsv_diag[
                    dtype,
                    ALayout=ALayout,
                    XLayout=XLayout,
                    upper=upper,
                    unit=unit,
                    trans=trans,
                    gpu=True,
                ]
            ](a, x, Int32(k), Int32(nb), grid_dim=1, block_dim=1)
        else:
            trsv_diag[dtype, ALayout, XLayout, upper, unit, trans](
                a, x, Int32(k), Int32(nb)
            )

        # Everything not yet solved, updated by the block just solved.
        comptime if upper:
            gemv_sub[trans=trans, target=_target[gpu]()](a, x, 0, k, k, nb, ctx)
        else:
            gemv_sub[trans=trans, target=_target[gpu]()](
                a, x, k + nb, k, n - k - nb, nb, ctx
            )


def _trsm[
    dtype: DType,
    ALayout: TensorLayout,
    upper: Bool,
    unit: Bool,
    trans: Bool = False,
    gpu: Bool = False,
](
    a: _View[dtype, ALayout],
    b: _Dense[dtype],
    n: Int,
    rhs: Int,
    block: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """Solve a triangular system against a matrix, in place, blocked and
    device-resident. BLAS `trsm`, left side.

    `_trsv` with `rhs` right-hand sides at once, and the reason it is a
    separate routine rather than a loop over that one: with a matrix on
    the right the update between diagonal steps is a *matrix* product, so
    it goes to `linalg.matmul` and the whole solve is `O(n^2 * rhs)` of
    GEMM instead of `O(n^2)` of `gemv` done `rhs` times.

    `b` is a `_Dense` view, so dense and row-major by its type rather than
    by a promise in prose. That is not generality lost -- every caller here allocates it -- and it is what lets each
    block row of `b` be handed to `matmul` as a contiguous operand with no
    packing at all. The triangle is read where it lies and its block is
    packed, because `matmul` ignores row stride.

    `upper`, `unit` and `trans` mean what they mean in `_trsv`.
    """
    if n <= 0 or rhs <= 0:
        return

    # The triangle's block, made dense for the GEMM, and the GEMM's own
    # output, which nothing reads: the epilogue subtracts each tile into
    # `b` as it lands. Both allocated once for the whole solve.
    var operand = zeros_dyn[dtype, 2](n, block, ctx=ctx)
    var scratch = zeros_dyn[dtype, 2](n, rhs, ctx=ctx)
    var ov = operand.view()
    var sv = scratch.view()

    var steps = (n + block - 1) // block
    for step in range(steps):
        var k = (n - (step + 1) * block) if upper else (step * block)
        var nb = block
        if upper:
            if k < 0:
                nb = block + k
                k = 0
        else:
            nb = min(block, n - k)

        trsm_diag[upper=upper, unit=unit, trans=trans, target=_target[gpu]()](
            a, b, k, nb, rhs, ctx
        )

        # The rows still unsolved, updated by the block just solved.
        var row0 = 0 if upper else k + nb
        var rows = k if upper else n - k - nb
        if rows <= 0:
            continue

        var left: _Dense[dtype] = TileTensor(
            ov.ptr_at_offset(Coord(0, 0)), row_major(Coord(rows, nb))
        )
        pack_block[trans=trans, target=_target[gpu]()](
            a, left, row0, k, rows, nb, ctx
        )

        # `b`'s own rows `k..k+nb`: dense already, because `b` is.
        var right: _Dense[dtype] = TileTensor(
            b.ptr_at_offset(Coord(k, 0)), row_major(Coord(nb, rhs))
        )
        var product: _Dense[dtype] = TileTensor(
            sv.ptr_at_offset(Coord(0, 0)), row_major(Coord(rows, rhs))
        )

        @parameter
        @always_inline
        @__copy_capture(b, row0)
        def subtract[
            _dtype: DType,
            width: SIMDLength,
            *,
            alignment: Int = align_of[SIMD[_dtype, width]](),
        ](idx: IndexList[2], value: SIMD[_dtype, width]) capturing -> None:
            var at = Coord(row0 + idx[0], idx[1])
            b.store[width](
                at, b.load[width](at) - rebind[SIMD[dtype, width]](value)
            )

        _max_matmul[elementwise_lambda_fn=subtract, target=_target[gpu]()](
            product, left, right, ctx
        )

    ctx.synchronize()


def solve_triangular[
    dtype: DType,
    n: Int,
    upper: Bool = False,
    unit: Bool = False,
    trans: Bool = False,
    gpu: Bool = False,
    block: Int = 16,
](mut a: Shaped[dtype, n, n], mut b: Shaped[dtype, n]) raises -> Shaped[
    dtype, n
] where dtype.is_floating_point():
    """**Tier 2.** Solve `A @ x = b` for triangular `A`.
    `scipy.linalg.solve_triangular`.

    The `Tensor`-tier sibling of `forward_substitution` and
    `back_substitution`, which are one function here because `upper`
    already distinguishes them and a caller reaching for a triangular
    solve should not have to know which of two names its triangle wants.
    `unit=True` treats the diagonal as an implicit `1` without reading it,
    which is what an LU's packed `L` needs; `trans=True` solves against
    `A^T` without transposing `A`.

    MAX ships no triangular solve at any size -- `trsm` exists only inside
    its private cuBLAS/rocBLAS bindings, which are per-vendor and so out
    of bounds here -- so this is numax's, blocked and device-resident: one
    `numax.linalg.panel` kernel per diagonal block and one `gemv_sub`
    between them, nothing crossing to the host.

    `a` is taken mutably because `view()` hands back a writable
    `TileTensor`; neither operand is modified. `b` is copied, so the
    caller's vector survives.
    """
    var ctx = a.context()
    var x = Shaped[dtype, n](ctx)
    var xv = x.view()
    pack_vector[target=_target[gpu]()](b.view(), xv, 0, n, ctx)
    _trsv[upper=upper, unit=unit, trans=trans, gpu=gpu](
        a.view(), xv, n, block, ctx
    )
    ctx.synchronize()
    return x^


def solve_triangular[
    dtype: DType,
    n: Int,
    rhs: Int,
    upper: Bool = False,
    unit: Bool = False,
    trans: Bool = False,
    gpu: Bool = False,
    block: Int = 16,
](mut a: Shaped[dtype, n, n], mut b: Shaped[dtype, n, rhs]) raises -> Shaped[
    dtype, n, rhs
] where dtype.is_floating_point():
    """**Tier 2.** Solve `A @ X = B` for triangular `A` and a matrix `B`.
    `scipy.linalg.solve_triangular` with a two-dimensional right-hand
    side.

    Not a loop over the vector overload: with several right-hand sides the
    update between diagonal blocks is a matrix product, so it goes to
    MAX's `matmul` and the cubic term is a GEMM. That is the whole reason
    to have both spellings, and it is what `inverse` is built on.

    Parameters are the vector overload's. Device-resident throughout.
    """
    var ctx = a.context()
    var x = Shaped[dtype, n, rhs](ctx)
    var xd: _Dense[dtype] = TileTensor(
        x.view().ptr_at_offset(Coord(0, 0)), row_major(Coord(n, rhs))
    )
    pack_block[target=_target[gpu]()](b.view(), xd, 0, 0, n, rhs, ctx)
    _trsm[upper=upper, unit=unit, trans=trans, gpu=gpu](
        a.view(), xd, n, rhs, block, ctx
    )
    return x^


def tridiagonal_solve[
    T: FloatLike, n: Int
](
    sub: Array[T, n],
    diag: Array[T, n],
    sup: Array[T, n],
    rhs: Array[T, n],
) -> Array[T, n]:
    """Solve a tridiagonal system by the Thomas algorithm.

    `sub[i]` is the entry below the diagonal in row `i` (so `sub[0]` is
    unused) and `sup[i]` is the one above it (so `sup[n-1]` is unused);
    both are passed full-length rather than short by one so the indexing
    matches the row it belongs to.

    No MAX equivalent at any size, and unlike the dense routines here
    there is nothing to gain from one: Thomas is already linear, so
    MAX's `matmul` has nothing to improve on and the blocked treatment
    the dense factorizations get would be pure overhead.

    `O(n)` rather than the `O(n^3)` a general solve would cost, which is
    why cubic splines and implicit one-dimensional PDE steps are tractable
    at all. Also unpivoted -- Thomas is stable without pivoting for
    diagonally dominant or symmetric positive definite systems, which
    covers both of those uses.
    """
    var c_prime = _zeros[T, n]()
    var d_prime = _zeros[T, n]()

    var first = guard_nonzero(diag[0], T.constant(_PIVOT_FLOOR))
    c_prime[0] = sup[0] / first
    d_prime[0] = rhs[0] / first

    for i in range(1, n):
        var denominator = guard_nonzero(
            diag[i] - (sub[i] * c_prime[i - 1]),
            T.constant(_PIVOT_FLOOR),
        )
        c_prime[i] = sup[i] / denominator
        d_prime[i] = (rhs[i] - (sub[i] * d_prime[i - 1])) / denominator

    var x = _zeros[T, n]()
    x[n - 1] = d_prime[n - 1].copy()
    for step in range(1, n):
        var i = n - 1 - step
        x[i] = d_prime[i] - (c_prime[i] * x[i + 1])

    return x^
