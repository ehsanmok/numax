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

from layout.tile_layout import TensorLayout
from max.gpu.host import DeviceContext
from std.collections import Array

from ..core.numeric import FloatLike, guard_nonzero

from .blas import _target
from .common import _PIVOT_FLOOR, _zeros
from .panel import _View, gemv_sub, trsv_diag


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
    needs. The triangle is read where it lies -- a strided block is fine
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
                    gpu=True,
                ]
            ](a, x, Int32(k), Int32(nb), grid_dim=1, block_dim=1)
        else:
            trsv_diag[dtype, ALayout, XLayout, upper, unit](
                a, x, Int32(k), Int32(nb)
            )

        # Everything not yet solved, updated by the block just solved.
        comptime if upper:
            gemv_sub[target=_target[gpu]()](a, x, 0, k, k, nb, ctx)
        else:
            gemv_sub[target=_target[gpu]()](
                a, x, k + nb, k, n - k - nb, nb, ctx
            )


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
