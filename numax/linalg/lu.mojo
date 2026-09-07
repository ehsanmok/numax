"""LU factorization, pivoted and not, and the determinant.
`scipy.linalg`'s `lu`/`lu_factor`/`lu_solve` and `det`.

MAX ships no LU at any size, so everything here is numax's.

**Three things, because pivoting is a branch on data.** `lu` is tier 1 and
unpivoted, so it runs at any conformer inside a GPU thread and fails on a
matrix with a zero pivot even when that matrix is perfectly well
conditioned -- `[[0, 1], [1, 0]]` is the standard example. `PivotedLU`
(from `lu_factor` over `Array`) pivots properly and gives that up: it is
`Plain[dtype, 1]` at width 1, because a SIMD `T` holds several matrices
whose lanes would want different pivot orders and there is no single order
to pick. `TensorLU` (from `lu_factor` over `Tensor`) pivots too and is
blocked and device-resident: the factors and the pivot vector stay in
device memory, the panel and the two substitutions are `panel.mojo`
kernels, and the trailing `L21 @ U12` update is fused into
`linalg.matmul`'s epilogue.

`TensorLU` carries `gpu` in its type, so a factorization built on the
accelerator cannot be solved against by host code -- the mismatch is a
compile error rather than a device-pointer read.

Both factorization objects carry the solves that reuse them, which is the
point of returning a factorization rather than a solution: one
factorization, several right-hand sides.

Rank deficiency is not detected at any tier. A singular matrix factors to a
zero pivot; `cond` on the original matrix is the check.
"""

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul as _max_matmul
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.collections import Array
from std.sys.info import align_of
from std.utils import IndexList

from ..core.array import Shaped, zeros, zeros_dyn
from ..core.numeric import FloatLike, guard_nonzero
from ..core.plain import Plain

from .blas import _target
from .cholesky import _Dense
from .common import _PIVOT_FLOOR, _zeros
from .panel import (
    _PANEL_THREADS,
    getrf_panel,
    laswp,
    pack_block,
    pack_vector,
    trsm_left_lower_unit,
)
from .triangular import _trsv, back_substitution, forward_substitution


def lu[T: FloatLike, n: Int](a: Array[T, n * n]) -> Array[T, n * n]:
    """Doolittle `LU` without pivoting, packed into one matrix.

    The strict lower triangle holds `L` (whose diagonal is an implicit
    `1`), and the upper triangle including the diagonal holds `U`. Packing
    them avoids returning two matrices where the two halves never overlap.

    See this module's docstring for what "without pivoting" costs.

    MAX ships no `lu` at any size. Past the crossover, use `lu_factor` over
    `Tensor`: blocked, partially pivoted, trailing update in MAX's
    `matmul`. It pivots, so it also factors matrices this cannot start on.
    """
    var out = _zeros[T, n * n]()
    for i in range(n * n):
        out[i] = a[i].copy()

    for k in range(n):
        var pivot = guard_nonzero(out[k * n + k], T.constant(_PIVOT_FLOOR))
        for i in range(k + 1, n):
            var factor = out[i * n + k] / pivot
            out[i * n + k] = factor.copy()
            for j in range(k + 1, n):
                out[i * n + j] = out[i * n + j] - (factor * out[k * n + j])

    return out^


@fieldwise_init
struct PivotedLU[dtype: DType, n: Int](Movable where dtype.is_floating_point()):
    """**Tier 2.** An `LU` factorization with partial pivoting, and the
    solves that reuse it. `scipy.linalg`'s `lu_factor`/`lu_solve` pair.

    The answer to this module's "Scope: no pivoting" limitation, and the
    reason it is separate from `lu` rather than an improvement to it:
    choosing a row by the magnitude of a value is a data-dependent branch,
    so this cannot be tier 1 and cannot run inside a GPU thread. It is
    `Plain[dtype, 1]` at width 1 for the same reason -- a SIMD `T` holds
    several matrices whose lanes would want different pivot orders, and
    there is no single order to pick.

    What it buys is correctness where `lu` has none. The exchange matrix
    `[[0, 1], [1, 0]]` is perfectly well conditioned with a determinant of
    `-1`, and unpivoted `lu` cannot start on it at all; this factors it.
    Accuracy on a small-pivot matrix improves for the same reason.

    Rank deficiency is still not detected: a singular matrix factors to a
    zero pivot, which is floored rather than reported, so the result is
    finite and wrong. `cond` on the original matrix is the check.
    """

    var factored: Array[Plain[Self.dtype, 1], Self.n * Self.n]
    """`L` below the diagonal (its own diagonal an implicit `1`) and `U` on
    and above it, packed the way `lu` packs them."""

    var permutation: Array[Int, Self.n]
    """`permutation[i]` is the row of `A` that became row `i`."""

    var sign: Int
    """`+1` or `-1`, by the parity of the row swaps. `det` needs it; a
    determinant read off the diagonal alone would be wrong by that factor.
    """

    def solve(
        self, b: Array[Plain[Self.dtype, 1], Self.n]
    ) -> Array[
        Plain[Self.dtype, 1], Self.n
    ] where Self.dtype.is_floating_point():
        """Solve `A @ x = b`, reusing this factorization.

        Permutes `b` the way the factorization permuted `A`'s rows, then
        runs the same two substitutions the unpivoted `solve` does. Several
        right-hand sides against one matrix is what the split into a
        factorization and a solve is for.
        """
        var permuted = _zeros[Plain[Self.dtype, 1], Self.n]()
        for i in range(Self.n):
            permuted[i] = b[self.permutation[i]].copy()
        var y = forward_substitution[
            Plain[Self.dtype, 1], Self.n, unit_diagonal=True
        ](self.factored, permuted)
        return back_substitution[Plain[Self.dtype, 1], Self.n](self.factored, y)

    def det(self) -> Plain[Self.dtype, 1] where Self.dtype.is_floating_point():
        """The determinant: the product of `U`'s diagonal, signed by the
        permutation.

        The pivoted counterpart of `det`, and it gets the answers `det`
        cannot -- the exchange matrix comes out as `-1` where the unpivoted
        route returns approximately zero.
        """
        var product = Plain[Self.dtype, 1].constant(Float64(self.sign))
        for i in range(Self.n):
            product = product * self.factored[i * Self.n + i]
        return product^


struct TensorLU[dtype: DType, n: Int, gpu: Bool = False](
    Movable where dtype.is_floating_point()
):
    """**Tier 2.** A blocked `LU` factorization of a `Tensor`, with partial
    pivoting, and the solves that reuse it.

    The `Tensor` counterpart of `PivotedLU`, and separate from it for the
    same reason `matmul` has two overloads: this one is `dtype`-monomorphic
    and its matrix has its own storage, that one is `FloatLike`-generic and
    its matrix lives in registers.

    MAX has no LU, so the factorization is numax's -- but blocked, so its
    cubic term is a matrix product and goes to MAX's `matmul`. Each step
    factors a `block`-wide panel with partial pivoting, solves the block
    row to its right, and subtracts `L21 @ U12` from what remains.

    Pivoting is why this is host-side and cannot be tier 1: choosing a row
    by the magnitude of a value is a branch on data. It is also what makes
    it correct where the unpivoted `lu` is not -- the exchange matrix
    `[[0, 1], [1, 0]]` has a zero leading pivot and factors fine here.

    Rank deficiency is still not reported: a singular matrix factors to a
    zero pivot and `solve` divides by it, so the result is not finite
    rather than being flagged. `cond` on the original matrix is the check.
    """

    var factored: Shaped[Self.dtype, Self.n, Self.n]
    """`L` below the diagonal (its own diagonal an implicit `1`) and `U` on
    and above it, row-major, packed the way `PivotedLU` packs them.

    A `Tensor`, on whichever device the factorization ran on, and it never
    leaves it. An earlier version kept this as a host `List` on the
    argument that a substitution walks elements in an order fixed by the
    previous one -- true of an element, false of a *block*: `solve` is
    blocked exactly the way the factorization is, so the sequential part
    is one `block x block` system per step and the `O(n^2)` between them
    is parallel.
    """

    var pivots: Shaped[DType.int32, Self.n]
    """`pivots[j]` is the row column `j` interchanged with, LAPACK's `ipiv`
    convention.

    On the device beside `factored`, because `solve` applies these to the
    right-hand side there. Composed in order rather than resolved into a
    permutation vector, which is what lets `laswp` replay them with one
    pass and no allocation.
    """

    var sign: Int
    """`+1` or `-1`, the parity of the row swaps; `det`'s sign.

    A host `Int`, computed once from `pivots` when the factorization
    finishes: it is one number, and reading it off the device per `det`
    call would cost a synchronization for a parity.
    """

    def __init__(
        out self,
        var factored: Shaped[Self.dtype, Self.n, Self.n],
        var pivots: Shaped[DType.int32, Self.n],
        sign: Int,
    ):
        self.factored = factored^
        self.pivots = pivots^
        self.sign = sign

    def to_tensor_factored(
        mut self, ctx: Optional[DeviceContext] = None
    ) raises -> Shaped[Self.dtype, Self.n, Self.n]:
        """The packed `L`/`U` as a tensor, for inspection or reuse.

        A copy, so the caller cannot invalidate this factorization by
        writing through it. `ctx` is accepted for signature compatibility
        and ignored -- the copy stays on the factorization's own device.
        """
        var out = Shaped[Self.dtype, Self.n, Self.n](self.factored.context())
        pack_block[target=_target[Self.gpu]()](
            self.factored.view(),
            out.view(),
            0,
            0,
            Self.n,
            Self.n,
            self.factored.context(),
        )
        self.factored.context().synchronize()
        return out^

    def solve[
        block: Int = 64
    ](mut self, mut b: Shaped[Self.dtype, Self.n]) raises -> Shaped[
        Self.dtype, Self.n
    ] where Self.dtype.is_floating_point():
        """`x` with `A @ x == b`, reusing this factorization.

        `scipy.linalg.lu_solve`. Three steps, all on the factorization's
        own device: replay the interchanges onto `b` with `laswp`,
        forward-substitute through `L` (unit diagonal, so no division),
        back-substitute through `U`.

        Blocked like the factorization, so "a substitution is sequential"
        is true only of the `block x block` diagonal system each step
        solves; the `O(n^2)` update between them is one row per thread. The
        factor stays where the factorization left it, which is the point of
        `lu_factor` returning this object rather than a matrix.
        """
        var ctx = self.factored.context()
        var x = Shaped[Self.dtype, Self.n](ctx)
        var fv = self.factored.view()
        var xv = x.view()
        var pv = self.pivots.view()

        pack_vector[target=_target[Self.gpu]()](b.view(), xv, 0, Self.n, ctx)

        comptime if Self.gpu:
            ctx.enqueue_function[
                laswp[
                    Self.dtype,
                    XLayout=type_of(xv).LayoutType,
                    PLayout=type_of(pv).LayoutType,
                    gpu=True,
                ]
            ](xv, pv, Int32(Self.n), grid_dim=1, block_dim=1)
        else:
            laswp(xv, pv, Int32(Self.n))

        _trsv[
            Self.dtype,
            type_of(fv).LayoutType,
            type_of(xv).LayoutType,
            upper=False,
            unit=True,
            gpu=Self.gpu,
        ](fv, xv, Self.n, block, ctx)
        _trsv[
            Self.dtype,
            type_of(fv).LayoutType,
            type_of(xv).LayoutType,
            upper=True,
            unit=False,
            gpu=Self.gpu,
        ](fv, xv, Self.n, block, ctx)

        ctx.synchronize()
        return x^

    def det(
        mut self,
    ) raises -> Scalar[Self.dtype] where Self.dtype.is_floating_point():
        """`det(A)`: the product of `U`'s diagonal, times the swap parity.

        The diagonal is gathered on the device into an `n`-vector and only
        that is read back, so the cost is `O(n)` of transfer rather than
        the `O(n^2)` a copy of the factor would be. The product itself is
        `n` multiplications and stays on the host, where an overflow is at
        least visible.
        """
        var ctx = self.factored.context()
        var diagonal = Shaped[Self.dtype, Self.n](ctx)
        var fv = self.factored.view()
        var dv = diagonal.view()

        @always_inline
        def gather[w: Int, alignment: Int = 1](coord: Coord) {var fv, var dv}:
            var i = coord[0]
            dv.store[1](Coord(i), fv[Coord(i, i)])

        elementwise[simd_width=1, target=_target[Self.gpu]()](
            gather, Coord(Self.n), ctx
        )
        ctx.synchronize()

        var values = diagonal.to_host()
        var product = Scalar[Self.dtype](self.sign)
        for i in range(Self.n):
            product *= values[i]
        return product


def lu_factor[
    dtype: DType, n: Int
](a: Array[Plain[dtype, 1], n * n]) -> PivotedLU[
    dtype, n
] where dtype.is_floating_point():
    """**Tier 2.** Factor `A` into `L @ U` with partial pivoting.
    `scipy.linalg.lu_factor`.

    See `PivotedLU` for what pivoting costs and what it buys.
    """
    var out = _zeros[Plain[dtype, 1], n * n]()
    for i in range(n * n):
        out[i] = a[i].copy()

    var permutation = Array[Int, n](fill=0)
    for i in range(n):
        permutation[i] = i
    var sign = 1

    for k in range(n):
        var best = k
        var best_magnitude = abs(out[k * n + k].v)
        for i in range(k + 1, n):
            var magnitude = abs(out[i * n + k].v)
            if magnitude > best_magnitude:
                best = i
                best_magnitude = magnitude

        if best != k:
            for j in range(n):
                var swap = out[k * n + j].copy()
                out[k * n + j] = out[best * n + j].copy()
                out[best * n + j] = swap^
            var swap_index = permutation[k]
            permutation[k] = permutation[best]
            permutation[best] = swap_index
            sign = -sign

        var pivot = guard_nonzero(
            out[k * n + k], Plain[dtype, 1].constant(_PIVOT_FLOOR)
        )
        for i in range(k + 1, n):
            var factor = out[i * n + k] / pivot
            out[i * n + k] = factor.copy()
            for j in range(k + 1, n):
                out[i * n + j] = out[i * n + j] - (factor * out[k * n + j])

    return PivotedLU[dtype, n](out^, permutation^, sign)


def lu_factor[
    dtype: DType, n: Int, gpu: Bool = False, block: Int = 64
](mut a: Shaped[dtype, n, n]) raises -> TensorLU[
    dtype, n, gpu
] where dtype.is_floating_point():
    """**Tier 2.** Factor `a` into `P @ L @ U`, blocked.
    `scipy.linalg.lu_factor`.

    Right-looking and blocked: factor a `block`-wide panel with partial
    pivoting, solve the block row to its right against `L11`, then subtract
    `L21 @ U12` from the trailing submatrix. That subtraction is the cubic
    term and it is a matrix product, so it is MAX's `matmul`; the panel and
    the block-row solve are `O(n * block^2)`.

    Row interchanges are applied across the full width of the matrix as
    they are chosen, rather than being recorded and replayed over the
    columns outside the panel afterwards. Same result, one less pass.

    See `TensorLU` for what pivoting costs and what it buys, and
    `cholesky` for the same blocking on a symmetric matrix, where pivoting
    is unnecessary. The `Array[T, n*n]` sibling `lu_factor` is the one to
    call at a conformer other than a raw `dtype`.

    **Nothing leaves the device.** As in `cholesky`, the matrix crosses to
    the host once on the way in and once on the way out: the panel is a
    `numax.linalg.panel` kernel addressing the matrix in place, the block
    row is an `elementwise` solve, and the trailing update's result goes
    back into the strided trailing block through `matmul`'s epilogue. Only
    the GEMM's two operands are packed dense, because MAX's `matmul`
    ignores their row stride.

    The cost is a `n x n` workspace MAX writes the GEMM into that nothing
    reads -- `matmul` stores to `c` whether an epilogue is given or not --
    plus an `n x block` and a `block x n` for the operands.

    **Ceiling.** `getrf_panel` runs on one thread block: its panel is the
    full remaining height, so it is `O(n^2 * block / 2)` of work on a
    single SM while the rest of the accelerator waits. `cholesky` has no
    equivalent because its panel is only `block x block`. The upgrade is a
    recursive panel (LAPACK's `getrf2`); its docstring in `panel.mojo` has
    the detail.
    """
    var ctx = a.context()
    var work = Shaped[dtype, n, n](ctx)
    var pivots = zeros[DType.int32, n + _PANEL_THREADS](ctx)
    var info = zeros[DType.int32, 1](ctx)
    # `L21` and `U12` made dense for the GEMM, and the GEMM's own output,
    # which nothing reads. All three are allocated once.
    var left_operand = zeros_dyn[dtype, 2](n, block, ctx=ctx)
    var right_operand = zeros_dyn[dtype, 2](block, n, ctx=ctx)
    var scratch = zeros_dyn[dtype, 2](n, n, ctx=ctx)

    var wv = work.view()
    var pv = pivots.view()
    var iv = info.view()
    var lv = left_operand.view()
    var rv = right_operand.view()
    var sv = scratch.view()

    pack_block[target=_target[gpu]()](a.view(), wv, 0, 0, n, n, ctx)

    var k = 0
    while k < n:
        var nb = min(block, n - k)

        comptime if gpu:
            ctx.enqueue_function[
                getrf_panel[
                    dtype,
                    ALayout=type_of(wv).LayoutType,
                    PLayout=type_of(pv).LayoutType,
                    ILayout=type_of(iv).LayoutType,
                    gpu=True,
                ]
            ](
                wv,
                pv,
                iv,
                Int32(k),
                Int32(nb),
                Int32(n),
                grid_dim=1,
                block_dim=_PANEL_THREADS,
            )
        else:
            getrf_panel(wv, pv, iv, Int32(k), Int32(nb), Int32(n))

        trsm_left_lower_unit[target=_target[gpu]()](wv, k, nb, n, ctx)

        var m = n - k - nb
        if m > 0:
            var base = k + nb

            var left: _Dense[dtype] = TileTensor(
                lv.ptr_at_offset(Coord(0, 0)), row_major(Coord(m, nb))
            )
            var right: _Dense[dtype] = TileTensor(
                rv.ptr_at_offset(Coord(0, 0)), row_major(Coord(nb, m))
            )
            pack_block[target=_target[gpu]()](wv, left, base, k, m, nb, ctx)
            pack_block[target=_target[gpu]()](wv, right, k, base, nb, m, ctx)

            var product: _Dense[dtype] = TileTensor(
                sv.ptr_at_offset(Coord(0, 0)), row_major(Coord(m, m))
            )

            @parameter
            @always_inline
            @__copy_capture(wv, base)
            def subtract[
                _dtype: DType,
                width: SIMDLength,
                *,
                alignment: Int = align_of[SIMD[_dtype, width]](),
            ](idx: IndexList[2], value: SIMD[_dtype, width]) capturing -> None:
                var at = Coord(base + idx[0], base + idx[1])
                wv.store[width](
                    at,
                    wv.load[width](at) - rebind[SIMD[dtype, width]](value),
                )

            _max_matmul[elementwise_lambda_fn=subtract, target=_target[gpu]()](
                product, left, right, ctx
            )

        k += nb

    ctx.synchronize()

    # The interchange list is `n` int32s, so reading it back to count the
    # parity is `O(n)` of transfer against the factorization's `O(n^3)`.
    # `pivots` is over-allocated by `_PANEL_THREADS` because
    # `getrf_panel`'s single block parks its candidates in the tail; only
    # the head travels with the factorization.
    var recorded = pivots.to_host()
    var sign = 1
    var trimmed = zeros[DType.int32, n](ctx)
    var head = List[Scalar[DType.int32]](capacity=n)
    for j in range(n):
        head.append(recorded[j])
        if Int(recorded[j]) != j:
            sign = -sign
    trimmed.copy_from_host(head^)

    return TensorLU[dtype, n, gpu](work^, trimmed^, sign)


def det[T: FloatLike, n: Int](a: Array[T, n * n]) -> T:
    """The determinant, as the product of the unpivoted LU's diagonal.

    Unpivoted, so the sign is always the product's own -- there is no row
    swap count to correct for.

    MAX ships no `det` at any size. Past the crossover, `lu_factor` over
    `Tensor` and take `TensorLU.det`, which corrects for the swap parity
    this one has no swaps to correct for.
    """
    var factored = lu[T, n](a)
    var product = T.one()
    for i in range(n):
        product = product * factored[i * n + i]
    return product^
