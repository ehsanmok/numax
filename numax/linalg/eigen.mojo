"""Spectral factorizations over `Tensor`: `scipy.linalg`'s `_decomp`.

**This module is tier 2.** Every routine here is two phases: a reduction
to a condensed form, which is Householder work whose cubic term goes to
`linalg.matmul`, then a sweep on the condensed band, which loops to a
tolerance and deflates on a test of the data. The band sweep itself has no
GEMM to hand anything to -- that is a property of the algorithm, not of
this implementation -- so it runs on the host over `O(n)` numbers and each
function says so where it happens.

The *transformations* that sweep produces are a different matter, and they
are no longer host work: `_RotationBatch` holds a batch of sweeps' worth,
reorders it into windows of mutually commuting entries and applies each
window as one `matmul`. `svd` shares that machinery through two batches
rather than one: `_bdsqr` chases the bidiagonal itself and pushes a right
rotation into one and a left rotation into the other, so `U_B^T` and
`V_B^T` are built device-resident at width `n`. The `2n` Golub-Kahan
doubling that route replaced is gone, and `_golub_kahan` is kept only as
its test oracle. `schur` shares it too, with the batch at
reach 2 for the Francis chase's order-three reflectors; what is left on
its host side is the quasi-triangular `T`, which only multishift QR
moves, and `schur`'s docstring carries the numbers.

MAX ships nothing to delegate to here. There is no eigensolver, no SVD, no
`sytrd`, no Jacobi or Givens helper anywhere in `linalg`, `nn`,
`algorithm` or `layout`; the one decomposition in the tree,
`linalg.qr_factorization`, is `LayoutTensor`-only and a CPU scalar loop,
denied twice over. So everything here is an **extend**, written in MAX's
idiom.

`numax.linalg.eigen` is the `FloatLike`-generic sibling, one import
away: cyclic Jacobi at a fixed sweep count, square matrices small enough to
live in registers, differentiable and GPU-launchable in a way this tier
cannot be. This tier is for the sizes that one cannot reach.
"""

from layout import Coord, TileTensor, coord_to_index_list
from layout.tile_layout import TensorLayout, row_major
from linalg.matmul import matmul as _max_matmul
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.builtin.sort import sort as _std_sort
from std.math import copysign as _copysign, hypot as _hypot, sqrt as _sqrt
from std.sys.info import align_of, simd_width_of
from std.utils import IndexList

from ._twostage import _two_stage_bidiagonal, _two_stage_tridiagonal
from ._multishift import _NMIN, _hseqr
from ..core.tensorlike import TensorLike, TensorView, dim, is_row_major
from ..core.tensor import (
    Dynamic,
    Static,
    _dyn_shape,
    _same_order,
    copy as _copy,
    transpose,
    zeros,
    zeros_dyn,
)
from ..core.ops import add, multiply
from .cholesky import cholesky
from .triangular import solve_triangular
from .blas import _target, dot, inner, matmul, matvec
from .common import (
    _mut_view,
    _mut_view_as,
    _Dense,
    _device_identity,
    _vector_aligned,
)
from .panel import (
    _PANEL_THREADS,
    gebd2_col,
    gebd2_row,
    _View,
    labrd_column,
    labrd_row,
    labrd_x,
    labrd_y,
    lahr2_column,
    lahr2_y,
    latrd_column,
    latrd_w,
    pack_block,
    sytd2_column,
)
from .qr import _apply_block_reflector, _MIN_GEMM_COLS, _ReflectorWork
from std.collections import Array
from ..core.numeric import FloatLike
from ._array.eigen import eigh as _array_eigh
from ..core.complex import Complex
from ._array.eigen import eigvals as _array_eigvals
from ._array.eigen import eigvalsh as _array_eigvalsh
from ._array.eigen import hessenberg as _array_hessenberg
from ._array.eigen import matrix_rank as _array_matrix_rank
from ._array.eigen import svd as _array_svd
from ._array.eigen import svdvals as _array_svdvals


struct Tridiagonal[dtype: DType, n: Int, gpu: Bool = False](
    Movable where dtype.is_floating_point() and n >= 1
):
    """The symmetric tridiagonal reduction `A = Q T Q^T`, device-resident.

    `T` is the tridiagonal matrix `diag(d)` with `e` on both off-diagonals;
    `Q` is orthogonal and is the product of the Householder reflections the
    reduction applied, held packed rather than formed. `.q()` materializes
    it when a caller wants the vectors; `eigvalsh` never asks.

    `e` is `n` long where only `e[0 .. n-2]` carries a subdiagonal entry,
    rather than the `n - 1` LAPACK uses: an `n == 1` matrix would otherwise
    need a zero-length field, and the last entry is documented as unused
    rather than made unrepresentable.
    """

    var d: Static[Self.dtype, Self.n]
    """`T`'s diagonal."""

    var e: Static[Self.dtype, Self.n]
    """`T`'s subdiagonal in `e[0 .. n-2]`; `e[n-1]` is unused and zero."""

    var reflectors: Static[Self.dtype, Self.n, Self.n]
    """The reduced matrix, holding `v` for column `k` below row `k + 1`
    with its leading `1` implicit -- LAPACK's packed form."""

    var taus: Static[Self.dtype, Self.n]
    """One Householder scale per column; `taus[k] == 0` where the column
    was already reduced and the reflection is the identity."""

    var block: Int
    """The panel width `.q()` forms `Q` in, carried from `sytrd`."""

    def __init__(
        out self,
        var d: Static[Self.dtype, Self.n],
        var e: Static[Self.dtype, Self.n],
        var reflectors: Static[Self.dtype, Self.n, Self.n],
        var taus: Static[Self.dtype, Self.n],
        block: Int,
    ):
        self.d = d^
        self.e = e^
        self.reflectors = reflectors^
        self.taus = taus^
        self.block = block

    def q(
        mut self,
    ) raises -> Static[
        Self.dtype, Self.n, Self.n
    ] where Self.dtype.is_floating_point():
        """Materialize `Q`, the orthogonal matrix of the reduction.
        LAPACK's `orgtr`.

        `Q = H_0 H_1 ... H_{n-3}` applied to the identity, panels of
        `block` reflectors in reverse order, each panel one block
        reflector -- `larft` then `C := (I - V T V^T) C` as three matrix
        products. That is `_accumulate_reflectors`, and it is `QR`'s
        `orgqr` walk over the same kernels.

        Still `O(n^3)`, but the cubic term is `linalg.matmul`'s rather
        than `n` matrix-vector launches. A caller who only wants
        eigenvalues should not call this; `eigvalsh` does not.
        """
        return _accumulate_reflectors[gpu=Self.gpu](
            self.reflectors, self.taus, Self.n - 2, self.block
        )


def _sytrd_run[
    T: TensorLike, gpu: Bool
](
    a: T,
    n: Int,
    block: Int,
    mut work: Dynamic[T.dtype, 2],
    mut taus: Dynamic[T.dtype, 1],
    mut d: Dynamic[T.dtype, 1],
    mut e: Dynamic[T.dtype, 1],
) raises where (T.dtype.is_floating_point() and T.LayoutType.rank == 2):
    """`sytrd`'s blocked reduction of the `n x n` symmetric `a`, at an
    order known only at run time: the static overload and the run-time
    spectral routines both run this loop. Every kernel it launches
    already took its extents as values; only the allocations and the
    `p = A v` product needed the order in the type, and `matvec`'s
    run-time overload pads exactly as the static one did.

    Writes into the caller's `n x n` `work` (the packed reflectors), and
    `n`-vectors `taus` (zeroed), `d` and `e`: out-parameters rather than a
    result struct, since Mojo cannot move one field out of a struct that
    still owns another (`findings.mdc`)."""
    var width = min(block, n)
    var ctx = a.context()
    var vpad = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var scratch = zeros_dyn[T.dtype, 1](_PANEL_THREADS + 1, ctx=ctx)
    var left = zeros_dyn[T.dtype, 2](n, 2 * width, ctx=ctx)
    var right = zeros_dyn[T.dtype, 2](n, 2 * width, ctx=ctx)
    # `latrd_w`'s reductions and the scalar it folds out of them.
    var red = zeros_dyn[T.dtype, 1](2 * width + 2, ctx=ctx)
    var product = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)

    var wv = work.tile()
    var tv = taus.tile()
    var vv = vpad.tile()
    var sv = scratch.tile()
    var lv = left.tile()
    var rv = right.tile()
    var redv = red.tile()
    var pv = product.tile()

    pack_block[target=_target[gpu]()](_mut_view(a), wv, 0, 0, n, n, ctx)

    var k0 = 0
    while k0 < n - 2:
        var nb = min(width, n - 2 - k0)

        # A narrower panel writes only `2 * nb` of the operands' `2 *
        # width` columns, and the GEMM below reads all of them. Clearing
        # the rest is this factorization's instance of the scratch-viewed-
        # at-two-widths trap: only the last panel can be narrow, so this
        # runs at most once.
        if nb < width:

            @always_inline
            def clear[
                w: Int, alignment: Int = 1
            ](coord: Coord) {var lv, var rv}:
                lv.store[1](coord, Scalar[T.dtype](0))
                rv.store[1](coord, Scalar[T.dtype](0))

            elementwise[simd_width=1, target=_target[gpu]()](
                clear, Coord(n, 2 * width), ctx
            )

        for j in range(nb):
            latrd_column[target=_target[gpu]()](wv, lv, k0, j, width, n, ctx)

            comptime if gpu:
                ctx.enqueue_function[
                    sytd2_column[
                        T.dtype,
                        ALayout=type_of(wv).LayoutType,
                        VLayout=type_of(vv).LayoutType,
                        TauLayout=type_of(tv).LayoutType,
                        SLayout=type_of(sv).LayoutType,
                        gpu=True,
                    ]
                ](
                    wv,
                    vv,
                    tv,
                    sv,
                    Int32(k0 + j),
                    Int32(n),
                    grid_dim=1,
                    block_dim=_PANEL_THREADS,
                )
                ctx.synchronize()
            else:
                sytd2_column(wv, vv, tv, sv, Int32(k0 + j), Int32(n))

            # `p = A v` over the whole matrix: `vpad` is zero at and above
            # `k0 + j`, so the panel's own columns -- which hold packed
            # reflectors, not matrix entries -- are never read, and the
            # leading block needs no staging. The panel's deferred updates
            # reach `p` inside `latrd_w` instead of through `work`.
            var p = matvec[gpu=gpu](work, vpad)

            latrd_w[target=_target[gpu]()](
                vv, p.tile(), lv, rv, redv, tv, k0, j, width, n, ctx
            )
            # `p`'s last mention is `.tile()`, and a view erases the
            # origin; see `findings.mdc` on the queued free.
            _ = p^

        _subtract_panel[gpu=gpu](
            wv,
            lv,
            rv,
            pv,
            k0 + nb,
            k0 + nb,
            n - k0 - nb,
            n - k0 - nb,
            2 * width,
            ctx,
        )
        k0 += nb

    # `sv`, `lv`, `rv`, `redv` and `pv` are read by the launches above and
    # their owners are named nowhere else, so without these Mojo would
    # destroy them after `.tile()`; see `findings.mdc` on the origin-erased
    # view and the queued free.
    _ = scratch^
    _ = left^
    _ = right^
    _ = red^
    _ = product^

    var dv = d.tile()
    var ev = e.tile()

    # The band, on the device: `work`'s diagonal and first subdiagonal.
    # `e[n-1]` is the documented unused entry and is written zero.
    @always_inline
    def band[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var wv, var dv, var ev, var n}:
        var at = coord_to_index_list(coord)[0]
        dv.store[1](coord, wv[Coord(at, at)])
        var below = Scalar[T.dtype](0)
        if at + 1 < n:
            below = wv[Coord(at + 1, at)]
        ev.store[1](coord, below)

    elementwise[simd_width=1, target=_target[gpu]()](band, Coord(n), ctx)
    ctx.synchronize()


def sytrd[
    T: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: T) raises -> Tridiagonal[T.dtype, dim[T, 0], gpu] where (
    (T.dtype.is_floating_point() and block >= 1)
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
    and dim[T, 1] == dim[T, 0]
):
    """**Tier 2.** Reduce a symmetric `a` to tridiagonal form by
    Householder reflections, device-resident and blocked. LAPACK's
    `sytrd` over `latrd` panels.

    **`gpu=True` is the whole reduction on the device**: the panel, the
    `p = A v` product and the rank-`2 * block` update all launch there, and
    the band comes back as two `n`-vectors. The spectral routines built on
    this one keep only their band iteration on the host -- sequential and
    `O(n^2)` -- with its rotations applied to the vectors as device GEMMs.
    `tests_gpu/linalg/test_spectral_gpu.mojo` holds every one of them to
    the host answer at `float32`. The panel kernels capture by value: a
    by-reference capture forwarded into a device launch reads zeros on
    Metal, which is what kept this path wrong through 0.2.

    `a` is read as symmetric and is **not checked** -- checking costs a
    full pass and the reduction is meaningless on a matrix that is not,
    in a way the caller is better placed to notice. Only the lower triangle
    and the diagonal are read.

    This is the half of an eigendecomposition that has a GEMM in it, and
    it is over half the arithmetic. A panel of `block` columns is reduced
    one column at a time without touching the trailing block -- each
    column is brought up to date with the panel's own reflectors by
    `latrd_column`, its reflector formed by `sytd2_column`, and its `w`
    built by `latrd_w` out of `p = A v` and the `2j + 1` reductions the
    panel needs -- and then the **whole panel** goes out as one symmetric
    rank-`2 * block` update,

        A[k0+nb:, k0+nb:] -= [V | W] @ [W | V]^T

    a single `matmul` with `transpose_b=True` and the subtraction fused
    into its epilogue. That identity is what lets numax skip the `syr2k`
    MAX does not ship, and `block` is what turns it from the rank-2 GEMM
    of an unblocked reduction into a rank-64 one: on the M3 Pro the
    rank-`k` table in `docs/performance.md` runs at 470 GFLOP/s at `k =
    64` and a small fraction of that at `k = 2`.

    `block` is the panel width, and it is also the width `.q()` forms `Q`
    in. **`block == 1` is the unblocked algorithm exactly** -- no
    pre-update, one rank-2 GEMM per column -- and `block == n` is one
    panel in which every column is pre-updated and the trailing GEMM never
    runs; a test pins every width against `block == n`.

    The trailing update is **restricted to rows and columns `k0 + nb`
    onward** rather than applied to the whole matrix. That restriction is
    load-bearing above `block == 1`: rows `k0+1 .. k0+nb-1` are where the
    panel packed its own reflectors, and a whole-matrix update would write
    over them. At `block == 1` the restriction coincides with the masking
    `latrd_w` already does, which is why the two agree there.

    **No `O(n * block)` term runs on one thread.** `latrd_w`'s `2j + 1`
    reductions and its `w` build are both `O(n j)`, and both are
    independent across the axis they are launched over, so each is one
    `parallelize` on the host and one `elementwise` on the accelerator
    with only the scalar between them on a single thread. Before that
    split they sat inside a single-block kernel and `sytrd` got *slower*
    as `block` grew -- 53, 66, 80, 109 ms at block 8, 16, 32, 64 and
    n = 1024 -- which is the shape a serial term makes in a width sweep.

    `ponytail:` what is left is the matrix-vector product. `p = A v` is
    one `matvec` over the whole matrix per column -- `2n^3` flops in
    total, bandwidth-bound rather than GEMM-bound, and taken over the
    whole matrix rather than the shrinking trailing block because a
    strided sub-block cannot be handed to `matmul` without staging it
    dense, and staging it per column would be `O(n^3)` of memcpy. LAPACK
    is in the same position -- half of `dsytrd`'s flops are BLAS-2 for
    exactly this reason -- so closing it is not a matter of blocking
    harder. The upgrade is a **two-stage reduction**: dense to banded,
    which is all GEMM, then banded to tridiagonal by a bulge chase. That
    is a different algorithm and it is not in 0.2.

    `numax.linalg.eigh` is the small-matrix route and needs none of
    this: cyclic Jacobi at a fixed sweep count, differentiable, and
    launchable inside a GPU thread.

    Parameters:
        T: The `TensorLike` type of `a`; a square matrix of floating dtype.
        gpu: Run the reduction and every `O(n^3)` product on the device of `a`.
        block: The reduction's panel width, also the width `.q()` forms `Q` in.

    Args:
        a: The `n x n` symmetric matrix; only its lower triangle and
            diagonal are read.

    Returns:
        A `Tridiagonal` holding the diagonal `d`, the subdiagonal `e` and
        the packed Householder reflectors that form `Q`.

    Raises:
        If a device allocation, copy or kernel launch fails.
    """
    comptime n = dim[T, 0]
    var ctx = a.context()
    var work = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    var taus = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var d = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var e = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    _sytrd_run[gpu=gpu](a, n, block, work, taus, d, e)
    return Tridiagonal[T.dtype, n, gpu](
        d^.as_static[n](),
        e^.as_static[n](),
        work^.as_static[n, n](),
        taus^.as_static[n](),
        block,
    )


def _subtract_panel[
    dtype: DType,
    ALayout: TensorLayout,
    LLayout: TensorLayout,
    RLayout: TensorLayout,
    PLayout: TensorLayout,
    gpu: Bool = False,
](
    target: _View[dtype, ALayout],
    left: _View[dtype, LLayout],
    right: _View[dtype, RLayout],
    product: _View[dtype, PLayout],
    row_base: Int,
    col_base: Int,
    rows: Int,
    cols: Int,
    width: Int,
    ctx: DeviceContext,
) raises where (
    dtype.is_floating_point()
    and _View[dtype, ALayout].flat_rank == 2
    and _View[dtype, LLayout].flat_rank == 2
    and _View[dtype, RLayout].flat_rank == 2
    and _View[dtype, PLayout].flat_rank == 2
):
    """`target[row_base:, col_base:] -= left[row_base:, :] @
    right[col_base:, :]^T`, in one GEMM with the subtraction fused into
    the epilogue.

    With `left = [V | W]` and `right = [W | V]` over one square matrix
    this is the symmetric rank-`width` update `sum_c (v_c w_c^T + w_c
    v_c^T)`, which is the whole trailing update of a `latrd` panel. With
    the two bases apart and the operands rectangular it is half of a
    `labrd` panel's `A -= V Y^T + X U`, called once per half.

    Both operands are **row ranges** of dense buffers, which are
    themselves dense, so `matmul` -- which ignores the row stride of its
    arguments -- reads them correctly. `target`'s trailing block is not
    dense, which is why the offsets live in the epilogue's store rather
    than in a view.
    """
    if rows <= 0 or cols <= 0:
        return

    # See `_MIN_GEMM_COLS`: a one-column product takes MAX's GEMV path,
    # which hands the epilogue coordinates past the block. A `labrd` panel
    # ending one column short of `n` is that shape, and the update is
    # `O(rows * width)` there, small enough to write directly.
    if cols < _MIN_GEMM_COLS:

        @always_inline
        def narrow[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var target,
            var left,
            var right,
            var row_base,
            var col_base,
            var width,
        }:
            var at = coord_to_index_list(coord)
            var to = Coord(row_base + at[0], col_base + at[1])
            var total = target[to]
            for t in range(width):
                total -= (
                    left[Coord(row_base + at[0], t)]
                    * right[Coord(col_base + at[1], t)]
                )
            target.store[1](to, total)

        elementwise[simd_width=1, target=_target[gpu]()](
            narrow, Coord(rows, cols), ctx
        )
        ctx.synchronize()
        return

    @__parameter
    @always_inline
    @__copy_capture(target, row_base, col_base)
    def subtract[
        _dtype: DType,
        lanes: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, lanes]](),
    ](idx: IndexList[2], value: SIMD[_dtype, lanes]) capturing -> None:
        var at = Coord(row_base + idx[0], col_base + idx[1])
        if _vector_aligned[_dtype, lanes](
            Int(target.ptr), Int(target.layout.stride[0]().value()), at
        ):
            target.store[lanes](
                at, target.load[lanes](at) - rebind[SIMD[dtype, lanes]](value)
            )
        else:
            target.store[lanes, alignment=align_of[Scalar[_dtype]]()](
                at,
                target.load[lanes, alignment=align_of[Scalar[_dtype]]()](at)
                - rebind[SIMD[dtype, lanes]](value),
            )

    var la: _Dense[dtype] = TileTensor(
        left.ptr_at_offset(Coord(row_base, 0)), row_major(Coord(rows, width))
    )
    var ra: _Dense[dtype] = TileTensor(
        right.ptr_at_offset(Coord(col_base, 0)), row_major(Coord(cols, width))
    )
    var out: _Dense[dtype] = TileTensor(
        product.ptr_at_offset(Coord(0, 0)), row_major(Coord(rows, cols))
    )

    _max_matmul[
        transpose_b=True,
        elementwise_lambda_fn=subtract,
        target=_target[gpu](),
    ](out, la, ra, ctx)
    ctx.synchronize()


@always_inline
def _eps[dtype: DType]() -> Scalar[dtype]:
    """Machine epsilon for `dtype`, the deflation threshold the sweeps use."""
    comptime if dtype == DType.float32:
        return Scalar[dtype](1.1920929e-07)
    else:
        return Scalar[dtype](2.220446049250313e-16)


comptime _MAX_QL_SWEEPS = 60
"""Sweeps allowed per eigenvalue before `eigvalsh` gives up.

LAPACK's `sterf` uses `30 * n` in total; per eigenvalue that is a bound
nothing well-formed approaches, since a Wilkinson-shifted QL step converges
cubically once the off-diagonal is small. Reaching it means `e` carries a
NaN or an infinity, and the raise says so rather than looping forever.
"""


struct _RotationBatch[
    dtype: DType,
    gpu: Bool,
    vectors: Bool,
    reach: Int = 1,
    ascending: Bool = False,
](Movable where dtype.is_floating_point() and reach >= 1):
    """The plane transformations of up to one batch of consecutive sweeps,
    held until enough of them have accumulated to go out as GEMMs.

    **Lang's windowing (1998), which is what makes this a GEMM at all.**
    Two transformations commute when the column ranges they touch do not
    meet. `reach` is how far past its own index one entry reaches: `1` for
    a Givens rotation on the adjacent pair `(i, i+1)`, `2` for a Francis
    order-three reflector on `i .. i+2`. So entries at `i` and `j` commute
    when `|i - j| > reach`.

    The two chases this serves run in opposite directions, and each gets
    the tag whose order is safe to apply in.

    **Descending, reach 1** -- `_tql`'s QL chase, where `i` falls within a
    sweep. Tag `group(s, i) = (i - s + B) // B` with `B = block` and
    `K = B` sweeps per batch; apply the groups in *decreasing* order, each
    group's entries in stream order. The only pairs that reordering swaps
    are at least two apart: within a sweep `i` descends so the tag never
    rises, and across sweeps `group(s1, i1) < group(s2, i2)` with `s1 < s2`
    forces `i2 - i1 >= 1 + (s2 - s1) >= 2`. The `+ B` shifts every tag by
    exactly one, so it is the same partition in the same order and the
    numerator never goes negative.

    **Ascending, reach 2** -- `_hqr`'s Francis bulge chase, where `k` rises
    within a sweep and a later sweep may reach at most two columns further
    back than the one before it, which is what the `+ 2 s` pays for. Tag
    `group(s, k) = (k + 2 s) // B` with `K = B // 2` sweeps per batch;
    apply the groups in *increasing* order. Every dependency edge points at
    a tag no smaller than its source, which is what makes the reordering
    exact: within a sweep `k` rises so the tag never falls, and for
    `s1 < s2` a *non*-commuting pair has `k2 - k1 >= -2` by definition, so
    `(k2 + 2 s2) - (k1 + 2 s1) >= -2 + 2 (s2 - s1) >= 0`. A tie lands in
    the same group, where the bucketing is stable and stream order
    survives. `k + 2 s` is never negative, so no shift is needed here.

    A group's entries then span at most `B + reach * K` columns either way
    -- `2 * block` -- and that is the window.

    So one group is a `w x w` orthogonal factor `U` -- the product of its
    transformations -- applied to a `w`-column stripe of `Z`. `Z` is held
    **transposed** as `zt`, because a column stripe of `Z` is a contiguous
    *row* block of `Z^T`, and a row block is what `matmul` will read
    without a stride: `zt[c_lo:c_lo+w, :] <- U^T @ staged`, where `staged`
    is a dense copy of that row block (`matmul` will not alias its result
    with an operand).

    `U^T` is built on the host, packed into the first `w*w` entries of
    `ut_host` so the claimed row stride and the memory agree, by the same
    lines the unblocked sweep ran on `Z`'s columns -- on `U^T`'s rows
    instead. Column `j` of `Z` and row `j - c_lo` of `U^T` take identical
    coefficients, since a factor acting on `Z`'s columns as `Z <- Z G`
    acts on `U^T`'s rows as `U^T <- G^T U^T`. That is `O(w)` per entry
    against `O(N)`, so the host term falls from `O(N^3)` to
    `O(block * N^2)` and the `O(N^3)` lands in `linalg.matmul`.

    At `vectors=False` nothing is ever pushed, so every buffer here is a
    one-element allocation and `eigvalsh`, `svdvals` and `eigvals` share
    the sweep bodies without paying for it.
    """

    var zt: Dynamic[Self.dtype, 2]
    """`Z^T`, `N x N` and device-resident. Row `j` is eigenvector `j`."""

    var order: Int
    """`N`, the order of the matrix whose vectors are accumulated -- a
    run-time value, so one compiled batch serves every size."""

    var staged: Dynamic[Self.dtype, 2]
    """A dense `2*block x N` copy of the row block under update."""

    var u_dev: Dynamic[Self.dtype, 2]
    """The uploaded `U^T`, `2*block x 2*block` and read packed."""

    var ut_host: List[Scalar[Self.dtype]]
    """Where `U^T` is built before it is uploaded."""

    var rot_index: List[Int]
    """`i` of each logged entry, an absolute column index."""

    var rot_sweep: List[Int]
    """Which sweep of the batch each logged entry came from."""

    var rot_kind: List[Int]
    """`0` a rotation on `(i, i+1)`, `2` an order-three reflector on
    `i .. i+2`, `1` its order-two form. Left empty at `reach == 1`, where
    every entry is a rotation by construction."""

    var rot_cos: List[Scalar[Self.dtype]]
    """`c` of a rotation, `x` of a reflector."""

    var rot_sin: List[Scalar[Self.dtype]]
    """`s` of a rotation, `y` of a reflector."""

    var rot_z: List[Scalar[Self.dtype]]
    """`zz` of a reflector, zero for a rotation. Empty at `reach == 1`."""

    var rot_q: List[Scalar[Self.dtype]]
    """`q` of a reflector. Empty at `reach == 1`."""

    var rot_r: List[Scalar[Self.dtype]]
    """`r` of a reflector. Empty at `reach == 1`."""

    var sweep: Int
    """Sweeps closed since the last flush; the batch goes out at `sweeps`."""

    var block: Int
    """`B`, the group stride."""

    var sweeps: Int
    """`K`, sweeps per batch: `B` at reach 1, `B // 2` at reach 2."""

    var window: Int
    """`B + reach * K`, the widest group the tag admits -- and the row
    count `staged`, `u_dev` and `ut_host` were sized for."""

    def __init__(out self, order: Int, block: Int, ctx: DeviceContext) raises:
        self.order = order
        self.block = max(block, 1)
        self.sweeps = max(self.block // Self.reach, 1)
        self.sweep = 0
        var side = order if Self.vectors else 1
        self.window = min(self.block + Self.reach * self.sweeps, side)
        var wide = self.window
        self.zt = zeros_dyn[Self.dtype, 2](side, side, ctx=ctx)
        self.staged = zeros_dyn[Self.dtype, 2](wide, side, ctx=ctx)
        self.u_dev = zeros_dyn[Self.dtype, 2](wide, wide, ctx=ctx)
        self.ut_host = List[Scalar[Self.dtype]](length=wide * wide, fill=0)
        self.rot_index = List[Int]()
        self.rot_sweep = List[Int]()
        self.rot_kind = List[Int]()
        self.rot_cos = List[Scalar[Self.dtype]]()
        self.rot_sin = List[Scalar[Self.dtype]]()
        self.rot_z = List[Scalar[Self.dtype]]()
        self.rot_q = List[Scalar[Self.dtype]]()
        self.rot_r = List[Scalar[Self.dtype]]()
        comptime if Self.vectors:
            var seed: _Dense[Self.dtype] = TileTensor(
                self.zt.tile().ptr_at_offset(Coord(0, 0)),
                row_major(Coord(side, side)),
            )
            _device_identity[Self.dtype, Self.gpu](seed, side, side, ctx)
            ctx.synchronize()

    def push_rotation(
        mut self, i: Int, c: Scalar[Self.dtype], s: Scalar[Self.dtype]
    ):
        """Log a rotation on columns `(i, i+1)`.

        `Z <- Z G` with `G = [[c, s], [-s, c]]`: the new column `i` is
        `c z_i - s z_{i+1}` and the new column `i+1` is `s z_i + c z_{i+1}`,
        which is what the unblocked bodies wrote out by hand. `_hqr`'s
        `2 x 2` split rotation is this one at `c = q`, `s = -p`.
        """
        self.rot_index.append(i)
        self.rot_sweep.append(self.sweep)
        self.rot_cos.append(c)
        self.rot_sin.append(s)
        comptime if Self.reach > 1:
            self.rot_kind.append(0)
            self.rot_z.append(Scalar[Self.dtype](0))
            self.rot_q.append(Scalar[Self.dtype](0))
            self.rot_r.append(Scalar[Self.dtype](0))

    def push_reflector(
        mut self,
        k: Int,
        x: Scalar[Self.dtype],
        y: Scalar[Self.dtype],
        zz: Scalar[Self.dtype],
        q: Scalar[Self.dtype],
        r: Scalar[Self.dtype],
        notlast: Bool,
    ):
        """Log a Francis reflector on columns `k .. k+2`, or on `k, k+1`
        when `notlast` is false and the bulge has run off the end.

        The column update is `_hqr`'s own, with `p = x z_k + y z_{k+1}
        (+ zz z_{k+2})` then `z_k -= p`, `z_{k+1} -= q p`,
        `z_{k+2} -= r p`. Only a `reach >= 2` batch is ever handed one.
        """
        self.rot_index.append(k)
        self.rot_sweep.append(self.sweep)
        self.rot_kind.append(2 if notlast else 1)
        self.rot_cos.append(x)
        self.rot_sin.append(y)
        self.rot_z.append(zz)
        self.rot_q.append(q)
        self.rot_r.append(r)

    def end_sweep(mut self, ctx: DeviceContext) raises:
        """Close a sweep, and flush once `sweeps` of them have closed."""
        self.sweep += 1
        if self.sweep >= self.sweeps:
            self._flush(ctx)

    def finish(mut self, ctx: DeviceContext) raises:
        """Flush the partial batch the last sweep left and wait for it."""
        self._flush(ctx)
        ctx.synchronize()

    @always_inline
    def _group(self, k: Int) -> Int:
        """The window tag of logged entry `k`; see the struct docstring for
        why each direction's tag is the one that reorders exactly."""
        comptime if Self.ascending:
            return (self.rot_index[k] + Self.reach * self.rot_sweep[k]) // (
                self.block
            )
        else:
            return (
                self.rot_index[k] - self.rot_sweep[k] + self.block
            ) // self.block

    @always_inline
    def _span(self, k: Int) -> Int:
        """How far past its index entry `k` reaches: `1` for a rotation or
        an order-two reflector, `2` for an order-three one."""
        comptime if Self.reach == 1:
            return 1
        else:
            return 2 if self.rot_kind[k] == 2 else 1

    def _flush(mut self, ctx: DeviceContext) raises:
        self.sweep = 0
        var count = len(self.rot_index)
        if count == 0:
            return
        var cols = self.order

        # Bucket the batch by group, stably, so each group's entries keep
        # the order the sweeps emitted them in. A linear rescan per group
        # would cost `O(count * N / block)`, which is the term this whole
        # machinery is removing.
        var lowest = self._group(0)
        var highest = lowest
        for k in range(1, count):
            var g = self._group(k)
            lowest = min(lowest, g)
            highest = max(highest, g)
        var groups = highest - lowest + 1
        var start = List[Int](length=groups + 1, fill=0)
        for k in range(count):
            start[self._group(k) - lowest + 1] += 1
        for g in range(groups):
            start[g + 1] += start[g]
        var cursor = start.copy()
        var ordered = List[Int](length=count, fill=0)
        for k in range(count):
            var g = self._group(k) - lowest
            ordered[cursor[g]] = k
            cursor[g] += 1

        var zv = self.zt.tile()
        for step in range(groups):
            var g = step if Self.ascending else groups - 1 - step
            var first = start[g]
            var last = start[g + 1]
            if first == last:
                continue

            # The exact span the group touches, rather than the formula's
            # bound: tight, never wider than `window`, and never 1, since
            # the narrowest entry still owns two columns.
            var c_lo = self.rot_index[ordered[first]]
            var c_hi = c_lo + self._span(ordered[first])
            for p in range(first + 1, last):
                var e = ordered[p]
                c_lo = min(c_lo, self.rot_index[e])
                c_hi = max(c_hi, self.rot_index[e] + self._span(e))
            var w = c_hi - c_lo + 1
            if w > self.window:
                raise Error(
                    "_RotationBatch: a window of ",
                    w,
                    " columns overran the ",
                    self.window,
                    " the tag admits",
                )

            self._build_ut(w, c_lo, ordered, first, last)
            self.u_dev.copy_from_host(self.ut_host)

            var staged: _Dense[Self.dtype] = TileTensor(
                self.staged.tile().ptr_at_offset(Coord(0, 0)),
                row_major(Coord(w, cols)),
            )
            pack_block[target=_target[Self.gpu]()](
                zv, staged, c_lo, 0, w, cols, ctx
            )
            var ut: _Dense[Self.dtype] = TileTensor(
                self.u_dev.tile().ptr_at_offset(Coord(0, 0)),
                row_major(Coord(w, w)),
            )
            var out: _Dense[Self.dtype] = TileTensor(
                zv.ptr_at_offset(Coord(c_lo, 0)), row_major(Coord(w, cols))
            )
            _max_matmul[target=_target[Self.gpu]()](out, ut, staged, ctx)
            ctx.synchronize()

        self.rot_index.clear()
        self.rot_sweep.clear()
        self.rot_kind.clear()
        self.rot_cos.clear()
        self.rot_sin.clear()
        self.rot_z.clear()
        self.rot_q.clear()
        self.rot_r.clear()

    def _build_ut(
        mut self,
        w: Int,
        c_lo: Int,
        ordered: List[Int],
        first: Int,
        last: Int,
    ):
        """`U^T` for one group, packed `w x w` into `ut_host`.

        `U = G_1 G_2 ... G_p` in stream order, so `U^T = G_p^T ... G_1^T`
        is built by left-multiplying by each `G_k^T` in turn -- which is
        the unblocked body's lines run on rows `a ..` of `U^T` instead of
        on columns `i ..` of `Z`, with `a = i - c_lo`.
        """
        comptime lanes = simd_width_of[Self.dtype]()
        for k in range(w * w):
            self.ut_host[k] = Scalar[Self.dtype](0)
        for k in range(w):
            self.ut_host[k * w + k] = Scalar[Self.dtype](1)

        var p = self.ut_host.unsafe_ptr()
        for entry in range(first, last):
            var k = ordered[entry]
            var ra = (self.rot_index[k] - c_lo) * w
            var rb = ra + w
            comptime if Self.reach > 1:
                if self.rot_kind[k] != 0:
                    var order3 = self.rot_kind[k] == 2
                    var x = self.rot_cos[k]
                    var y = self.rot_sin[k]
                    var zz = self.rot_z[k]
                    var q = self.rot_q[k]
                    var r = self.rot_r[k]
                    var rc = rb + w
                    var col = 0
                    while col + lanes <= w:
                        var ua = p.unsafe_load[width=lanes](ra + col)
                        var ub = p.unsafe_load[width=lanes](rb + col)
                        var pr = ua + q * ub
                        if order3:
                            var uc = p.unsafe_load[width=lanes](rc + col)
                            pr = ua + q * ub + r * uc
                            p.unsafe_store(rc + col, uc - pr * zz)
                        p.unsafe_store(rb + col, ub - pr * y)
                        p.unsafe_store(ra + col, ua - pr * x)
                        col += lanes
                    while col < w:
                        var ua = p[unsafe_offset=ra + col]
                        var ub = p[unsafe_offset=rb + col]
                        var pr = ua + q * ub
                        if order3:
                            var uc = p[unsafe_offset=rc + col]
                            pr = ua + q * ub + r * uc
                            p[unsafe_offset=rc + col] = uc - pr * zz
                        p[unsafe_offset=rb + col] = ub - pr * y
                        p[unsafe_offset=ra + col] = ua - pr * x
                        col += 1
                    continue

            var c = self.rot_cos[k]
            var s = self.rot_sin[k]
            var col = 0
            while col + lanes <= w:
                var ua = p.unsafe_load[width=lanes](ra + col)
                var ub = p.unsafe_load[width=lanes](rb + col)
                p.unsafe_store(rb + col, s * ua + c * ub)
                p.unsafe_store(ra + col, c * ua - s * ub)
                col += lanes
            while col < w:
                var ua = p[unsafe_offset=ra + col]
                var ub = p[unsafe_offset=rb + col]
                p[unsafe_offset=rb + col] = s * ua + c * ub
                p[unsafe_offset=ra + col] = c * ua - s * ub
                col += 1


def _tql[
    dtype: DType, gpu: Bool, vectors: Bool
](
    mut d: List[Scalar[dtype]],
    mut e: List[Scalar[dtype]],
    mut acc: _RotationBatch[dtype, gpu, vectors],
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """Implicit QL with Wilkinson shifts on the symmetric tridiagonal
    `(d, e)`, in place into `d`. LAPACK's `sterf` at `vectors=False` and
    `steqr` at `vectors=True`, where the rotations are also pushed into
    `acc`, which accumulates them into `Z^T` device-resident.

    **The band iteration is tier 2, host-side.** Each sweep is a chain of
    Givens rotations down the band that touches two entries at a time and
    stops at the first negligible subdiagonal; there is no GEMM to hand
    *that* to, and the deflation test branches on the data. It is `O(n^2)`
    and negligible beside the reduction.

    **The accumulation is not.** Applying every rotation to a full column
    pair is `O(n)` each and `O(n^3)` overall, and it used to run as scalar
    host code -- 5.7 of `eigh`'s 6.1 s at `n = 1024`. `_RotationBatch`
    holds `block` sweeps' worth, reorders them into commuting windows
    (Lang 1998; see its docstring for why the reorder is exact), and sends
    each window out as one `linalg.matmul` of a `w x w` factor against a
    `w x n` stripe. The host keeps `O(block * n^2)` of work building those
    factors on contiguous rows, and the cubic term is MAX's.

    `block == 1` recovers the unblocked algorithm exactly: one rotation per
    window, applied in stream order, each a `2 x 2` GEMM. Larger `block`
    only changes how many commuting rotations ride in one product, which is
    why the blocking tests pin every size against every other.

    `e` is read as the `n` entries `Tridiagonal` carries, with
    `e[n-1]` unused and treated as zero. Numerical Recipes' `tqli`, which
    is also Golub and Van Loan's Algorithm 8.3.3 read left to right.
    """
    var n = len(d)
    if n <= 1:
        return
    e[n - 1] = Scalar[dtype](0)
    var eps = _eps[dtype]()

    for l in range(n):
        var sweeps = 0
        while True:
            # Find the first negligible subdiagonal at or past `l`; the
            # block `l..m` is what this sweep works on.
            var m = l
            while m < n - 1:
                var scale = abs(d[m]) + abs(d[m + 1])
                if abs(e[m]) <= eps * scale:
                    break
                m += 1
            if m == l:
                break

            sweeps += 1
            if sweeps > _MAX_QL_SWEEPS:
                raise Error(
                    "eigvalsh: eigenvalue ",
                    l,
                    " did not converge in ",
                    _MAX_QL_SWEEPS,
                    " sweeps -- the matrix holds a NaN or an infinity",
                )

            # Wilkinson shift from the leading 2x2 of the block.
            var g = (d[l + 1] - d[l]) / (Scalar[dtype](2) * e[l])
            var r = _hypot(g, Scalar[dtype](1))
            var signed_r = r if g >= 0 else -r
            g = d[m] - d[l] + e[l] / (g + signed_r)

            var s_ = Scalar[dtype](1)
            var c = Scalar[dtype](1)
            var p = Scalar[dtype](0)
            var i = m - 1
            var underflowed = False
            while i >= l:
                var f = s_ * e[i]
                var b = c * e[i]
                r = _hypot(f, g)
                e[i + 1] = r
                if r == 0:
                    # The chain broke early: a zero rotation splits the
                    # block here and the next pass restarts from `l`.
                    d[i + 1] -= p
                    e[m] = Scalar[dtype](0)
                    underflowed = True
                    break
                s_ = f / r
                c = g / r
                g = d[i + 1] - p
                r = (d[i] - g) * s_ + Scalar[dtype](2) * c * b
                p = s_ * r
                d[i + 1] = g + p
                g = c * r - b
                comptime if vectors:
                    acc.push_rotation(i, c, s_)
                i -= 1
            if underflowed:
                acc.end_sweep(ctx)
                continue
            d[l] -= p
            e[l] = g
            e[m] = Scalar[dtype](0)
            acc.end_sweep(ctx)


comptime _TWO_STAGE_BAND = 16
"""The band width the two-stage reductions pass through: measured best on
an M3 Pro at `n = 1024`, where a wider band makes the Givens chase of
stage 2 dearer faster than it makes stage 1's products cheaper."""

comptime _TWO_STAGE_EIGVALSH = 256
"""Above this order `eigvalsh` takes the two-stage reduction; at or below
it the one-stage `sytrd` is faster (both measured at `float32`)."""

comptime _TWO_STAGE_SVDVALS = 32
"""From this width `svdvals` takes the two-stage reduction, which is ahead
of the one-stage `gebrd` at every size measured."""


def _symmetric_from_lower[
    gpu: Bool, T: TensorLike
](a: T, n: Int) raises -> Dynamic[T.dtype, 2] where T.LayoutType.rank == 2:
    """`a`'s lower triangle mirrored into a full symmetric row-major copy:
    `eigvalsh` reads only the lower triangle, and the two-stage reduction's
    trailing products want both."""
    var full = rebind_var[Dynamic[T.dtype, 2]](
        _same_order(a, row_major(_dyn_shape[2](n, n)))
    )
    var fp = full.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

    @always_inline
    def mirror[w: Int, alignment: Int = 1](coord: Coord) {var fp, var n}:
        var e = coord_to_index_list(coord)[0]
        var i = e // n
        var j = e % n
        if j > i:
            fp[unsafe_offset=e] = fp[unsafe_offset=j * n + i]

    elementwise[simd_width=1, target=_target[gpu]()](
        mirror, Coord(n * n), full.context()
    )
    full.context().synchronize()
    return full^


def eigvalsh[
    T: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: T) raises -> Static[T.dtype, dim[T, 0]] where (
    (T.dtype.is_floating_point() and block >= 1)
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
    and dim[T, 1] == dim[T, 0]
):
    """**Tier 2.** The eigenvalues of a symmetric `a`, ascending, without
    the eigenvectors. `numpy.linalg.eigvalsh` / `scipy.linalg.eigvalsh`.

    **`gpu=True`** runs the reduction and every `O(n^3)` product on the
    device; `sytrd` says what stays on the host and why.

    Two phases, and the split is the whole story. `sytrd` reduces `a` to
    tridiagonal form device-resident -- `O(4n^3/3)` with the cubic term in
    `linalg.matmul` -- and then `_tql` runs implicit QL sweeps on the
    two diagonals on the host. That sweep is `O(n^2)`, sequential, and
    branches on the data, which is tier 2 by numax's definition; beside the
    reduction it is genuinely negligible, and the eigenvalues never asked
    for a vector to be accumulated, which is where the `O(n^3)` of a
    host-side `eigh` would hide.

    Above `n = 256` the reduction is **two-stage**
    (`numax/linalg/_twostage.mojo`): block-reflector panels take the
    matrix to a band of half-width 16 with the trailing update a product
    on the device, then LAPACK's `dsbtrd` chases the band to tridiagonal
    form on the host. That removes the whole-matrix `matvec` per column the
    one-stage `sytrd` pays; at `n = 1024`, `float32`, on an M3 Pro it is
    about 94 ms against 142 for `sytrd` at the default `block`. At or below
    `n = 256`, `sytrd` is faster and runs instead.

    `block` is `sytrd`'s `latrd` panel width there, forwarded -- there is
    one `block` across this subsystem and it means "panel and window
    width" everywhere. Only the panel half of that is live, since no
    vectors are accumulated, and the panel's per-column arithmetic runs on
    the single thread block `numax.linalg.panel`'s kernels launch with, so
    a narrow `block` is fastest for this routine alone; the default is 32
    because `eigh`, which pays for the window and for `q()` as well, is
    fastest there. The two-stage path's band width is fixed at the
    measured best.

    Ascending, as SciPy returns them. The `Array` tier's `eigvalsh` is
    cyclic Jacobi at a fixed sweep count and returns its values unsorted --
    the two agree as multisets, and a test pins that.

    `a` is read as symmetric and not checked; see `sytrd`.

    Parameters:
        T: The `TensorLike` type of `a`; a square matrix of floating dtype.
        gpu: Run the reduction and every `O(n^3)` product on the device of `a`.
        block: The `sytrd` panel width.

    Args:
        a: The `n x n` symmetric matrix; only its lower triangle and
            diagonal are read.

    Returns:
        The `n` eigenvalues, ascending.

    Raises:
        If an eigenvalue fails to converge within the QL sweep budget,
        which means `a` holds a NaN or an infinity, or if a device operation
        fails.
    """
    comptime n = dim[T, 0]
    return Static[T.dtype, n](_eigvalsh_run[gpu=gpu](a, n, block), a.context())


def _eigvalsh_run[
    T: TensorLike, gpu: Bool
](a: T, n: Int, block: Int) raises -> List[Scalar[T.dtype]] where (
    T.dtype.is_floating_point() and T.LayoutType.rank == 2
):
    """`eigvalsh`'s body at an order known only at run time, the one both
    overloads run: the reduction (`sytrd`, or the two stages above
    `_TWO_STAGE_EIGVALSH`), the QL sweep, the sort."""
    var ctx = a.context()
    var d: List[Scalar[T.dtype]]
    var e: List[Scalar[T.dtype]]
    if n > _TWO_STAGE_EIGVALSH:
        var full = _symmetric_from_lower[gpu](a, n)
        var tri = _two_stage_tridiagonal[T.dtype, gpu](
            full^, n, _TWO_STAGE_BAND
        )
        d = tri[0].copy()
        e = tri[1].copy()
    else:
        var work = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
        var taus = zeros_dyn[T.dtype, 1](n, ctx=ctx)
        var dd = zeros_dyn[T.dtype, 1](n, ctx=ctx)
        var de = zeros_dyn[T.dtype, 1](n, ctx=ctx)
        _sytrd_run[gpu=gpu](a, n, block, work, taus, dd, de)
        d = dd.to_host()
        e = de.to_host()
    # `vectors=False` pushes nothing, so the batch is three one-element
    # allocations and its `block` never decides anything.
    var acc = _RotationBatch[T.dtype, gpu, False](n, 1, ctx)
    _tql[gpu=gpu, vectors=False](d, e, acc, ctx)
    _std_sort(d)
    return d^


struct Eigh[dtype: DType, n: Int](
    Movable where dtype.is_floating_point() and n >= 1
):
    """`eigh`'s result: eigenvalues ascending, eigenvectors as columns.

    A struct rather than the `(w, v)` tuple SciPy returns, for the reason
    `qr_factor` returns a `QR`: a `Tuple` of two `Tensor`s cannot be
    destructured in Mojo 1.0, so a tuple-shaped return would hand back a
    pair no caller could take apart. `numax.linalg.eigh` returns the
    tuple because `Array` copies.
    """

    var values: Static[Self.dtype, Self.n]
    """The eigenvalues, ascending."""

    var vectors: Static[Self.dtype, Self.n, Self.n]
    """The eigenvectors, one per column, in the order of `values`, so
    `a @ vectors[:, j] == values[j] * vectors[:, j]`. Orthonormal."""

    def __init__(
        out self,
        var values: Static[Self.dtype, Self.n],
        var vectors: Static[Self.dtype, Self.n, Self.n],
    ):
        self.values = values^
        self.vectors = vectors^


struct DynamicEigh[dtype: DType](Movable where dtype.is_floating_point()):
    """`eigh`'s result at a run-time order: `Eigh` with the order in the
    value rather than the type, as `DynamicLU` is to `LU`."""

    var values: Dynamic[Self.dtype, 1]
    """The eigenvalues, ascending."""

    var vectors: Dynamic[Self.dtype, 2]
    """The eigenvectors, one per column, in the order of `values`.
    Orthonormal."""

    def __init__(
        out self,
        var values: Dynamic[Self.dtype, 1],
        var vectors: Dynamic[Self.dtype, 2],
    ):
        self.values = values^
        self.vectors = vectors^


def eigh[
    T: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: T) raises -> Eigh[T.dtype, dim[T, 0]] where (
    (T.dtype.is_floating_point() and block >= 1)
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
    and dim[T, 1] == dim[T, 0]
):
    """**Tier 2.** The eigendecomposition of a symmetric `a`: eigenvalues
    ascending and orthonormal eigenvectors as columns.
    `numpy.linalg.eigh` / `scipy.linalg.eigh`.

    **`gpu=True`** runs the reduction and every `O(n^3)` product on the
    device; `sytrd` says what stays on the host and why.

    Three steps, all three GEMM-shaped. `sytrd` reduces `a` to
    tridiagonal form device-resident, `O(4n^3/3)` through `linalg.matmul`.
    Implicit QL then diagonalizes the tridiagonal, its rotations
    accumulated into `Z^T` device-resident in windows of commuting
    rotations that each go out as one `matmul` -- `_RotationBatch` above
    carries the argument. And the eigenvectors of `a` are `Q Z`, where `Q`
    is the reduction's orthogonal factor -- LAPACK's `ormtr` -- which is
    `inner(q, zt)` under `transpose_b=True`, so `Z` is never transposed
    back.

    `block` is all three widths at once, and that is deliberate: it is
    `sytrd`'s `latrd` panel, it is the rotation window -- `block`
    consecutive sweeps batch together and a window spans `2 * block`
    columns, so the host builds `w x w` rotation products at
    `O(block * n^2)` and MAX does the `O(n^3)` -- and it is the panel
    `q()` forms `Q` in. `block == 1` recovers the unblocked algorithm
    exactly in all three, one reflector and one rotation per product, and
    a test pins every size against every other.

    What is left on the host: the `O(n^2)` band sweep itself, which is
    sequential and deflates on a test of the data, and the `O(n^2)`
    insertion sort of the eigenvalue order -- the permutation is then
    applied to `Z^T`'s rows in one launch on the device.

    The `Array` tier's `eigh` is cyclic Jacobi at a fixed sweep count,
    unsorted, differentiable and launchable inside a GPU thread. For a
    matrix small enough to live in registers it is the better tool; this
    is for the sizes it cannot reach.

    `a` is read as symmetric and not checked; see `sytrd`.

    Parameters:
        T: The `TensorLike` type of `a`; a square matrix of floating dtype.
        gpu: Run the reduction and every `O(n^3)` product on the device of `a`.
        block: The `sytrd` panel width, the rotation window and the `.q()`
            panel width at once.

    Args:
        a: The `n x n` symmetric matrix; only its lower triangle and
            diagonal are read.

    Returns:
        An `Eigh` with the eigenvalues ascending in `values` and the
        orthonormal eigenvectors as the columns of `vectors`.

    Raises:
        If an eigenvalue fails to converge within the QL sweep budget,
        which means `a` holds a NaN or an infinity, or if a device operation
        fails.
    """
    comptime n = dim[T, 0]
    var ctx = a.context()
    var values = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var vectors = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    _eigh_run[gpu=gpu](a, n, block, values, vectors)
    return Eigh[T.dtype, n](values^.as_static[n](), vectors^.as_static[n, n]())


def _eigh_run[
    T: TensorLike, gpu: Bool
](
    a: T,
    n: Int,
    block: Int,
    mut values: Dynamic[T.dtype, 1],
    mut vectors: Dynamic[T.dtype, 2],
) raises where (T.dtype.is_floating_point() and T.LayoutType.rank == 2):
    """`eigh`'s body at an order known only at run time, the one both
    overloads run: `sytrd`, the QL sweep with its rotations batched into
    device GEMMs, the sort as a row gather, and `Q Z` as one product.
    Writes the caller's `n`-vector `values` and `n x n` `vectors`, the way
    `_sytrd_run` writes its outputs."""
    var ctx = a.context()
    var reflectors = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    var taus = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var diag = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var off = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    _sytrd_run[gpu=gpu](a, n, block, reflectors, taus, diag, off)
    var d = diag.to_host()
    var e = off.to_host()

    var acc = _RotationBatch[T.dtype, gpu, True](n, block, ctx)
    _tql[gpu=gpu, vectors=True](d, e, acc, ctx)
    acc.finish(ctx)

    # Sort ascending. `O(n^2)` scalar host work on `n` numbers, beside the
    # `O(n^3)` above; only the permutation reaches the device.
    var order = List[Int](capacity=n)
    for i in range(n):
        order.append(i)
    for i in range(1, n):
        var j = i
        while j > 0 and d[order[j]] < d[order[j - 1]]:
            var tmp = order[j]
            order[j] = order[j - 1]
            order[j - 1] = tmp
            j -= 1

    var sorted_values = List[Scalar[T.dtype]](capacity=n)
    var perm_host = List[Scalar[DType.int64]](capacity=n)
    for j in range(n):
        sorted_values.append(d[order[j]])
        perm_host.append(Scalar[DType.int64](order[j]))
    values.copy_from_host(sorted_values)
    var perm = Dynamic[DType.int64, 1](
        row_major(_dyn_shape[1](n)), perm_host^, ctx
    )

    # Row `j` of `zt` is eigenvector `j`, so the sort is a row gather --
    # source and destination are the same shape, which is what keeps a
    # cross-shape `elementwise` read out of it.
    var zt_sorted = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    var src = acc.zt.tile()
    var dst = zt_sorted.tile()
    var pv = perm.tile()

    @always_inline
    def gather[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var src, var dst, var pv}:
        var at = coord_to_index_list(coord)
        dst.store[1](coord, src[Coord(Int(pv[Coord(at[0])]), at[1])])

    elementwise[simd_width=1, target=_target[gpu]()](gather, Coord(n, n), ctx)
    ctx.synchronize()
    # Both owners' last mention is `.tile()` above, and a view erases the
    # origin, so without these Mojo frees them while `gather` still reads
    # through `src` and `pv`.
    _ = acc^
    _ = perm^

    var q = _accumulate_reflectors_run[gpu=gpu](
        reflectors, taus, n, n - 2, block
    )
    _max_matmul[transpose_b=True, target=_target[gpu]()](
        vectors.tile(), q.tile(), zt_sorted.tile(), ctx
    )
    ctx.synchronize()
    _ = q^
    _ = zt_sorted^


def _runtime_order[
    T: TensorLike
](a: T, name: StaticString) raises -> Int where T.LayoutType.rank == 2:
    """The order of the run-time-shaped square `a`, raising unless it is
    square and non-empty."""
    var n = a.dim_at(0)
    if a.dim_at(1) != n:
        raise Error(
            name, ": the matrix must be square, got ", n, " x ", a.dim_at(1)
        )
    if n < 1:
        raise Error(name, ": the matrix is empty")
    return n


def eigvalsh[
    T: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: T) raises -> Dynamic[T.dtype, 1] where (
    (T.dtype.is_floating_point() and block >= 1)
    and T.LayoutType.rank == 2
    and not T.LayoutType.all_dims_known
):
    """**Tier 2.** The run-time-shape overload: the eigenvalues of a
    symmetric `a` whose order is data, ascending.
    `scipy.linalg.eigvalsh`.

    The static overload's body, run at the order `a` carries: the same
    reduction (`sytrd` at or below `n = 256`, the two-stage band above
    it), the same QL sweep and the same sort, so one compiled program
    takes any size and the two overloads agree to the bit.

    Parameters:
        T: The `TensorLike` type of `a`, a rank-2 run-time-shaped matrix.
        gpu: Run the reduction and every `O(n^3)` product on `a`'s device.
        block: The `sytrd` panel width.

    Args:
        a: The `n x n` symmetric matrix; only its lower triangle and
            diagonal are read.

    Returns:
        A run-time-shaped vector of the `n` eigenvalues, ascending.

    Raises:
        If `a` is not square or is empty, if an eigenvalue fails to
        converge within the QL sweep budget (`a` holds a NaN or an
        infinity), or if a device operation fails.
    """
    var n = _runtime_order(a, "eigvalsh")
    return Dynamic[T.dtype, 1](
        row_major(_dyn_shape[1](n)),
        _eigvalsh_run[gpu=gpu](a, n, block),
        a.context(),
    )


def eigh[
    T: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: T) raises -> DynamicEigh[T.dtype] where (
    (T.dtype.is_floating_point() and block >= 1)
    and T.LayoutType.rank == 2
    and not T.LayoutType.all_dims_known
):
    """**Tier 2.** The run-time-shape overload: the eigendecomposition of
    a symmetric `a` whose order is data. `scipy.linalg.eigh`.

    The static overload's body at the order `a` carries -- `sytrd`, the
    QL sweep with its rotations batched into device GEMMs, the sort, `Q Z`
    -- returning a `DynamicEigh`, so one compiled program takes any size
    and the two overloads agree to the bit.

    Parameters:
        T: The `TensorLike` type of `a`, a rank-2 run-time-shaped matrix.
        gpu: Run the reduction and every `O(n^3)` product on `a`'s device.
        block: The `sytrd` panel width, the rotation window and the width
            `Q` is formed in.

    Args:
        a: The `n x n` symmetric matrix; only its lower triangle and
            diagonal are read.

    Returns:
        A `DynamicEigh` with the eigenvalues ascending in `values` and the
        orthonormal eigenvectors as the columns of `vectors`.

    Raises:
        If `a` is not square or is empty, if an eigenvalue fails to
        converge within the QL sweep budget (`a` holds a NaN or an
        infinity), or if a device operation fails.
    """
    var n = _runtime_order(a, "eigh")
    var ctx = a.context()
    var values = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var vectors = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    _eigh_run[gpu=gpu](a, n, block, values, vectors)
    return DynamicEigh[T.dtype](values^, vectors^)


def _standard_form[
    dtype: DType, n: Int, gpu: Bool
](a: Static[dtype, n, n], b: Static[dtype, n, n]) raises -> Tuple[
    Static[dtype, n, n], Static[dtype, n, n]
] where dtype.is_floating_point():
    """`(C, L)` with `b = L L^T` and `C = L^-1 a L^-T`, the standard form
    of the pencil `(a, b)`: Cholesky, then two triangular solves, then the
    average of `C` and its transpose, since the two solves leave it
    symmetric only to rounding and `sytrd` reads one triangle."""
    var l = cholesky[gpu=gpu](b)
    var y = solve_triangular[gpu=gpu](l, a)
    var c = solve_triangular[gpu=gpu](l, transpose[gpu=gpu](y))
    var sym = multiply[gpu=gpu](
        add[gpu=gpu](c, transpose[gpu=gpu](c)), Scalar[dtype](0.5)
    )
    return (sym^, l^)


def _pencil[
    A: TensorLike, B: TensorLike
](a: A, b: B) raises -> Tuple[
    Static[A.dtype, dim[A, 0], dim[A, 0]], Static[A.dtype, dim[A, 0], dim[A, 0]]
] where (
    A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 2
    and B.LayoutType.all_dims_known
):
    """`a` and `b` copied to one concrete square type on their device, so
    the solves between them see a single `dtype` and extent -- the layout
    prover does not carry `B.dtype == A.dtype` through a callee's result."""
    comptime n = dim[A, 0]
    var ac = _same_order(a, Static[A.dtype, n, n]._static_layout())
    var bc = rebind_var[Static[A.dtype, n, n]](
        _same_order(b, Static[B.dtype, n, n]._static_layout())
    )
    return (ac^, bc^)


def eigvalsh[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: A, b: B) raises -> Static[A.dtype, dim[A, 0]] where (
    (A.dtype.is_floating_point() and block >= 1)
    and A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and dim[A, 1] == dim[A, 0]
    and B.dtype == A.dtype
    and B.dtype.is_floating_point()
    and B.LayoutType.rank == 2
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 0]
    and dim[B, 1] == dim[A, 0]
    and dim[B, 1] == dim[B, 0]
):
    """**Tier 2.** The eigenvalues of the symmetric-definite pencil `a x =
    lambda b x`, ascending. `scipy.linalg.eigvalsh(a, b)`.

    `eigvalsh` of the standard form `L^-1 a L^-T`, `b = L L^T`, as
    `eigh(a, b)` computes it; `gpu=True` runs all of it on the device.

    Parameters:
        A: The tensor type of `a`, square.
        B: The tensor type of `b`, the same shape and `dtype`.
        gpu: Whether to run on the inputs' device.
        block: The reduction's panel width, as `eigvalsh`'s.

    Args:
        a: The symmetric matrix.
        b: The symmetric positive definite matrix.

    Returns:
        The generalized eigenvalues, ascending.

    Raises:
        If `b` is not positive definite, or a device operation fails.
    """
    var pair = _pencil(a, b)
    var standard = _standard_form[gpu=gpu](pair[0], pair[1])
    return eigvalsh[gpu=gpu, block=block](standard[0])


def eigh[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: A, b: B) raises -> Eigh[A.dtype, dim[A, 0]] where (
    (A.dtype.is_floating_point() and block >= 1)
    and A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and dim[A, 1] == dim[A, 0]
    and B.dtype == A.dtype
    and B.dtype.is_floating_point()
    and B.LayoutType.rank == 2
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 0]
    and dim[B, 1] == dim[A, 0]
    and dim[B, 1] == dim[B, 0]
):
    """**Tier 2.** The eigendecomposition of the symmetric-definite pencil
    `a x = lambda b x`: eigenvalues ascending and eigenvectors as columns,
    normalized so `X^T b X = I`. `scipy.linalg.eigh(a, b)`.

    The standard reduction, LAPACK's `sygv` with `itype = 1`: `b = L L^T`
    by `cholesky`, `C = L^-1 a L^-T` by two `solve_triangular`s, `C = V
    Lambda V^T` by `eigh`, and `X = L^-T V` by a third, transposed, solve.
    Each step has its device path, so `gpu=True` keeps everything on the
    device. The eigenvalues are `C`'s, so they carry `eigh`'s accuracy
    times `b`'s conditioning, the bound `sygv` has too; eigenvectors are
    determined up to sign, as `eigh`'s are.

    Parameters:
        A: The tensor type of `a`, square.
        B: The tensor type of `b`, the same shape and `dtype`.
        gpu: Whether to run on the inputs' device.
        block: The reduction's panel width, as `eigh`'s.

    Args:
        a: The symmetric matrix.
        b: The symmetric positive definite matrix.

    Returns:
        An `Eigh` with the generalized eigenvalues and `b`-orthonormal
        eigenvectors.

    Raises:
        If `b` is not positive definite, or a device operation fails.
    """
    comptime n = dim[A, 0]
    var pair = _pencil(a, b)
    var standard = _standard_form[gpu=gpu](pair[0], pair[1])
    var decomposed = eigh[gpu=gpu, block=block](standard[0])
    var vectors = solve_triangular[trans=True, gpu=gpu](
        standard[1], decomposed.vectors
    )
    return Eigh[A.dtype, n](_copy(decomposed.values), vectors^)


# --------------------------------------------------- Hessenberg and Schur


def _accumulate_reflectors[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](reflectors: A, taus: B, count: Int, block: Int) raises -> Static[
    A.dtype, dim[A, 0], dim[A, 0]
] where (
    A.dtype.is_floating_point()
    and A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and dim[A, 1] == dim[A, 0]
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 0]
):
    """`Q = H_0 H_1 ... H_{count-1}` from `count` reflectors held in LAPACK's
    packed form -- column `k` carries `v` below row `k + 1` with its leading
    `1` implicit -- applied to the identity in panels of `block`, last panel
    first. LAPACK's `orgtr` and `orghr` are both this.

    The panel walk is `QR.q()`'s and so are the kernels:
    `larft_panel` builds the panel's `T`, `pack_reflectors` materializes
    `V` and `V^T`, and `_apply_block_reflector` applies `I - V T V^T` as
    `W = V^T C`, `Y = T W`, `C -= V Y`. Three GEMMs and two small launches
    per panel, against the `3 * count` launches and `2 * count` device
    synchronizations the per-reflector walk needed.

    **The one adapter.** A reduction to tridiagonal or Hessenberg form
    puts the implicit unit at row `k + 1`, one below the row a QR puts it
    on, so `reflectors` is handed to those kernels through a view shifted
    one row down -- `n - 1` rows over the same memory, no copy. `taus`
    indexes unshifted, which is what both kernels already do. In the
    shifted frame the panel at `k` is exactly QR's, and the block it
    updates is the `(n - 1 - k)` square of `Q` at `(k + 1, k + 1)`: the
    columns to its left are untouched because `V^T e_j` is zero there.

    A caller who only wants eigenvalues never calls this.
    """
    comptime n = dim[A, 0]
    return _accumulate_reflectors_run[gpu=gpu](
        reflectors, taus, n, count, block
    ).as_static[n, n]()


def _accumulate_reflectors_run[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](reflectors: A, taus: B, n: Int, count: Int, block: Int) raises -> Dynamic[
    A.dtype, 2
] where (
    A.dtype.is_floating_point()
    and A.LayoutType.rank == 2
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
):
    """`_accumulate_reflectors` at an order known only at run time, the
    loop the static overload runs: `Q` from `count` packed reflectors of
    the `n x n` row-major `reflectors`, in panels of `block`."""
    var ctx = reflectors.context()
    var result = zeros_dyn[A.dtype, 2](n, n, ctx=ctx)
    var out: _Dense[A.dtype] = TileTensor(
        result.tile().ptr, row_major(Coord(n, n))
    )
    _device_identity[gpu=gpu](out, n, n, ctx)
    if count <= 0:
        ctx.synchronize()
        return result^

    var shifted: _Dense[A.dtype] = TileTensor(
        _mut_view(reflectors).ptr + n,
        row_major(Coord(n - 1, n)),
    )
    var tv = _mut_view_as[A.dtype](taus)
    var work = _ReflectorWork[A.dtype](n - 1, n, block, ctx)
    var steps = (count + block - 1) // block
    for step in range(steps):
        var k = (steps - 1 - step) * block
        var nb = min(block, count - k)
        _apply_block_reflector[transposed=False, gpu=gpu](
            shifted,
            tv,
            out,
            k,
            nb,
            n - 1,
            n - 1 - k,
            n - 1 - k,
            k + 1,
            k + 1,
            work,
            ctx,
        )

    ctx.synchronize()
    return result^


def _store_column[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](dst: A, v: B, k: Int, n: Int, ctx: DeviceContext) raises where (
    A.LayoutType.rank == 2 and B.dtype == A.dtype and B.LayoutType.rank == 1
):
    """`dst[:, k] = v` for the `n x n` `dst`, on the device."""
    var dv = _mut_view(dst)
    var vv = _mut_view_as[A.dtype](v)

    @always_inline
    def fill[w: Int, alignment: Int = 1](coord: Coord) {var dv, var vv, var k}:
        var i = coord_to_index_list(coord)[0]
        dv.store[1](Coord(i, k), vv[Coord(i)])

    elementwise[simd_width=1, target=_target[gpu]()](fill, Coord(n), ctx)


def _zero_column_below[
    T: TensorLike,
    gpu: Bool = False,
](a: T, k: Int, first: Int, n: Int, ctx: DeviceContext) raises where (
    T.LayoutType.rank == 2
):
    """`a[first.., k] = 0` for the `n x n` `a`, on the device: the entries
    a reflector annihilates, written as the exact zeros they are rather
    than left to the update that never touches its own column."""
    var av = _mut_view(a)

    @always_inline
    def fill[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var av, var k, var first}:
        var i = coord_to_index_list(coord)[0]
        if i >= first:
            av.store[1](Coord(i, k), Scalar[T.dtype](0))

    elementwise[simd_width=1, target=_target[gpu]()](fill, Coord(n), ctx)


struct Hessenberg[dtype: DType, n: Int, gpu: Bool = False](
    Movable where dtype.is_floating_point() and n >= 1
):
    """The Hessenberg reduction `A = Q H Q^T`, device-resident: `H` upper
    Hessenberg (zero below the first subdiagonal), `Q` orthogonal and held
    as packed reflectors until `.q()` materializes it. `scipy.linalg.hessenberg`
    with `calc_q=True` returns the pair; here `.h` is the matrix and `.q()`
    is the factor, so `eigvals`, which never needs it, never forms it.
    """

    var h: Static[Self.dtype, Self.n, Self.n]
    """The Hessenberg matrix, exact zeros below the subdiagonal."""

    var reflectors: Static[Self.dtype, Self.n, Self.n]
    """Column `k` holds `v` for reflector `k` below row `k + 1`, its
    leading `1` implicit -- LAPACK's packed form, `Tridiagonal`'s."""

    var taus: Static[Self.dtype, Self.n]
    """One Householder scale per column; zero where the column was already
    in Hessenberg form."""

    var block: Int
    """The panel width `.q()` forms `Q` in, carried from `hessenberg`."""

    def __init__(
        out self,
        var h: Static[Self.dtype, Self.n, Self.n],
        var reflectors: Static[Self.dtype, Self.n, Self.n],
        var taus: Static[Self.dtype, Self.n],
        block: Int,
    ):
        self.h = h^
        self.reflectors = reflectors^
        self.taus = taus^
        self.block = block

    def q(
        mut self,
    ) raises -> Static[
        Self.dtype, Self.n, Self.n
    ] where Self.dtype.is_floating_point():
        """Materialize `Q`, LAPACK's `orghr`: the reflectors applied to
        the identity in panels of `block`, three GEMMs each. See
        `_accumulate_reflectors`.

        A column whose reflection was the identity carries the bare unit
        vector `e_{k+1}` here rather than the packed form the reduction
        left in `sytrd`'s work matrix; both read the same, because
        `pack_reflectors` writes the unit itself and reads only below it,
        and `larft` gives a zero `tau` a zero column of `T`, so the panel
        never sees that reflector again.
        """
        return _accumulate_reflectors[gpu=Self.gpu](
            self.reflectors, self.taus, Self.n - 2, self.block
        )


def hessenberg[
    T: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: T) raises -> Hessenberg[T.dtype, dim[T, 0], gpu] where (
    (T.dtype.is_floating_point() and block >= 1)
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
    and dim[T, 1] == dim[T, 0]
):
    """**Tier 2.** Reduce a general square `a` to upper Hessenberg form by
    Householder reflections, device-resident and blocked. LAPACK's
    `gehrd` over `lahr2` panels, `scipy.linalg.hessenberg`.

    **`gpu=True`** runs the reduction and every `O(n^3)` product on the
    device; `sytrd` says what stays on the host and why.

    The shape is `sytrd`'s, minus the symmetry: a panel of `block` columns
    is reduced one column at a time without touching the trailing block,
    and then the whole panel goes out as a two-sided update in GEMMs. What
    the panel accumulates is `V`, its triangular factor `T` -- so that
    `Q = I - V T V^T` is the product of the panel's reflections -- and
    `Y = A V T`, which is the right factor's whole effect:

        A[:, k0+nb:]   -= Y V[k0+nb:, :]^T
        A[k0+1:, k0+nb:] = (I - V T^T V^T) A[k0+1:, k0+nb:]

    the first one `matmul` with the subtraction fused into its epilogue,
    the second `larfb`'s three through `_apply_block_reflector`. The right
    update reaches every row, including those above `k0`, because `A Q` is
    a column operation; the left one starts at `k0 + 1`, where the
    reflectors do.

    **That takes the per-column whole-matrix traffic from four passes to
    one.** The unblocked reduction did `v^T A`, a rank-one subtract, `A v`
    and a second rank-one subtract for every column, all of them BLAS-2
    over the whole matrix; this does `p = A v` and nothing else, with the
    two rank-one updates deferred into the panel's GEMMs. `lahr2_column`
    brings the next column up to date with the panel's own reflectors and
    `lahr2_y` builds the panel's `Y` and `T`, both of them out of `O(n j)`
    arithmetic that is threaded the way `latrd_w`'s is.

    `block == 1` is the unblocked algorithm -- one column per panel, the
    two updates restricted to exactly the rows and columns the masked
    rank-one updates reached -- and `block == n` is one panel whose
    trailing updates never run; a test pins every width against
    `block == n`. `block` is also the panel width `.q()` forms `Q` in.

    The trailing updates are **restricted to columns `k0 + nb` onward**,
    and the panel's own columns are finished inside the loop instead:
    `lahr2_column` writes the fully updated column back before the
    reflector is formed from it, so rows above `k0` are current there too.
    A whole-matrix update would write over the band entries and zeros the
    panel just wrote. The reduction runs `n - 2` columns, so the panel
    ends at `n - 2` at the latest and the trailing GEMM always has at
    least two output columns -- the `_MIN_GEMM_COLS` shape a `labrd` panel
    can reach and this one cannot.

    **No host read per column.** `tau` is read on the device by
    `lahr2_y`, so the `taus.to_host()[k]` synchronization the unblocked
    reduction paid once per column is gone, and with it the branch that
    skipped an identity reflection: a zero `tau` scales its column of `Y`
    and of `T` to zero and the panel's updates then add nothing.

    `ponytail:` what is left is the matrix-vector product, the ceiling
    `sytrd` states at length. `p = A v` is one `matvec` over the whole
    matrix per column -- taken over the whole matrix rather than the
    trailing block because a strided sub-block cannot be handed to
    `matmul` without staging it dense -- and LAPACK's `dgehrd` is half
    BLAS-2 for the same reason. Closing it needs a two-stage reduction,
    not a wider panel.

    `numax.linalg.hessenberg` is the `FloatLike`-generic sibling for
    matrices small enough to live in registers.

    Parameters:
        T: The `TensorLike` type of `a`; a square matrix of floating dtype.
        gpu: Run the reduction and every `O(n^3)` product on the device of `a`.
        block: The `lahr2` panel width, also the width `.q()` forms `Q` in.

    Args:
        a: The general `n x n` matrix to reduce.

    Returns:
        A `Hessenberg` holding `H`, the packed reflectors and their scales,
        from which `.q()` forms `Q`.

    Raises:
        If a device allocation, copy or kernel launch fails.
    """
    comptime n = dim[T, 0]
    var ctx = a.context()
    var h = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    var reflectors = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    var taus = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    _gehrd_run[gpu=gpu](a, n, block, h, reflectors, taus)
    return Hessenberg[T.dtype, n, gpu](
        h^.as_static[n, n](),
        reflectors^.as_static[n, n](),
        taus^.as_static[n](),
        block,
    )


def _gehrd_run[
    T: TensorLike, gpu: Bool
](
    a: T,
    n: Int,
    block: Int,
    mut work: Dynamic[T.dtype, 2],
    mut reflectors: Dynamic[T.dtype, 2],
    mut taus: Dynamic[T.dtype, 1],
) raises where (T.dtype.is_floating_point() and T.LayoutType.rank == 2):
    """`hessenberg`'s blocked reduction of the `n x n` `a` at an order
    known only at run time, the loop the static overload and the run-time
    `eigvals`/`schur` run. Writes the caller's zeroed `work` (becoming
    `H`), `reflectors` and `taus`, as `_sytrd_run` writes its outputs."""
    var width = min(block, n)
    var ctx = a.context()
    var vpad = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var scratch = zeros_dyn[T.dtype, 1](_PANEL_THREADS + 1, ctx=ctx)
    # The panel's `Y`, its dense `V` for the trailing GEMM, its triangular
    # factor, and the reductions `lahr2_column`/`lahr2_y` share.
    var yy = zeros_dyn[T.dtype, 2](n, width, ctx=ctx)
    var vp = zeros_dyn[T.dtype, 2](n, width, ctx=ctx)
    var tt = zeros_dyn[T.dtype, 2](width, width, ctx=ctx)
    var red = zeros_dyn[T.dtype, 1](2 * width + 2, ctx=ctx)
    var product = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)

    var wv = work.tile()
    var rfv = reflectors.tile()
    var tv = taus.tile()
    var vv = vpad.tile()
    var sv = scratch.tile()
    var yv = yy.tile()
    var vpv = vp.tile()
    var ttv = tt.tile()
    var redv = red.tile()
    var pv = product.tile()

    pack_block[target=_target[gpu]()](_mut_view(a), wv, 0, 0, n, n, ctx)

    var lwork = _ReflectorWork[T.dtype](max(n - 1, 1), n, width, ctx)

    var k0 = 0
    while k0 < n - 2:
        var nb = min(width, n - 2 - k0)

        # A narrower panel writes only `nb` of the `width` columns the
        # trailing GEMM reads. Clearing `Y` is enough -- it is one of the
        # two operands, so a zero column there kills the stale column of
        # `V` beside it -- and only the last panel can be narrow, so this
        # runs at most once.
        if nb < width:

            @always_inline
            def clear[w: Int, alignment: Int = 1](coord: Coord) {var yv}:
                yv.store[1](coord, Scalar[T.dtype](0))

            elementwise[simd_width=1, target=_target[gpu]()](
                clear, Coord(n, width), ctx
            )

        for j in range(nb):
            var i = k0 + j

            lahr2_column[target=_target[gpu]()](
                wv, rfv, yv, ttv, redv, k0, j, width, n, ctx
            )

            comptime if gpu:
                ctx.enqueue_function[
                    sytd2_column[
                        T.dtype,
                        ALayout=type_of(wv).LayoutType,
                        VLayout=type_of(vv).LayoutType,
                        TauLayout=type_of(tv).LayoutType,
                        SLayout=type_of(sv).LayoutType,
                        gpu=True,
                    ]
                ](
                    wv,
                    vv,
                    tv,
                    sv,
                    Int32(i),
                    Int32(n),
                    grid_dim=1,
                    block_dim=_PANEL_THREADS,
                )
                ctx.synchronize()
            else:
                sytd2_column(wv, vv, tv, sv, Int32(i), Int32(n))

            _store_column[gpu=gpu](reflectors, vpad, i, n, ctx)
            _zero_column_below[gpu=gpu](work, i, i + 2, n, ctx)

            # `p = A v` over the whole matrix: `vpad` is zero at and above
            # `i`, so the columns the panel has already finished are never
            # read and the ones it has not are still as the panel found
            # them. The deferred update reaches `Y` inside `lahr2_y`.
            var p = matvec[gpu=gpu](work, vpad)
            lahr2_y[target=_target[gpu]()](
                p.tile(), rfv, yv, ttv, redv, tv, k0, j, n, ctx
            )
            # `p`'s last mention is `.tile()`, and a view erases the
            # origin; see `findings.mdc` on the queued free.
            _ = p^

        # `V` is a column range of `reflectors`, not dense, so it is
        # packed once per panel rather than once per column.
        pack_block[target=_target[gpu]()](rfv, vpv, 0, k0, n, nb, ctx)

        var base = k0 + nb
        # Right first, then left: `Q^T A Q` is `Q^T (A Q)`, and `Y` was
        # built against the matrix as the panel found it.
        _subtract_panel[gpu=gpu](
            wv, yv, vpv, pv, 0, base, n, n - base, width, ctx
        )
        # The unit sits at row `k + 1` here and at row `k` in QR's frame,
        # so the reflectors reach `larfb` through a view shifted one row
        # down -- `_accumulate_reflectors`' adapter, the same one.
        var shifted: _Dense[T.dtype] = TileTensor(
            rfv.ptr_at_offset(Coord(1, 0)), row_major(Coord(n - 1, n))
        )
        _apply_block_reflector[transposed=True, gpu=gpu](
            shifted,
            tv,
            wv,
            k0,
            nb,
            n - 1,
            n - 1 - k0,
            n - base,
            k0 + 1,
            base,
            lwork,
            ctx,
        )
        k0 += nb

    ctx.synchronize()
    # Read by the launches above through origin-erased views whose owners
    # are named nowhere else; see `findings.mdc` on the queued free.
    _ = scratch^
    _ = vpad^
    _ = yy^
    _ = vp^
    _ = tt^
    _ = red^
    _ = product^
    _ = lwork^


comptime _MAX_QR_SWEEPS_PER_N = 30
"""Francis sweeps allowed per deflation, per matrix dimension -- LAPACK
`dlahqr`'s `30 max(10, n)`, a budget for each eigenvalue rather than one
pool for the whole matrix. Reaching it means the Hessenberg matrix
carries a NaN or an infinity, and the raise says so rather than looping
forever."""

comptime _EXCEPTIONAL_EVERY = 10
"""`dlahqr`'s `KEXSH`: sweeps without a deflation between exceptional
shifts."""


@always_inline
def _chase_rows[
    dtype: DType, order3: Bool
](
    mut h: List[Scalar[dtype]],
    k: Int,
    n: Int,
    j0: Int,
    j1: Int,
    x: Scalar[dtype],
    y: Scalar[dtype],
    zz: Scalar[dtype],
    q: Scalar[dtype],
    r: Scalar[dtype],
):
    """`_hqr`'s row update for one bulge reflector, over columns
    `j0 .. j1`.

    The three rows the reflector touches are contiguous runs of `h`, so
    the scalar loop this replaces is one SIMD walk down them -- the only
    part of the `wantt` work that vectorizes, since the matching column
    update strides by `n`. The arithmetic is the scalar body's, term for
    term and in the same association.
    """
    comptime lanes = simd_width_of[dtype]()
    var p = h.unsafe_ptr()
    var a0 = k * n
    var a1 = a0 + n
    var a2 = a1 + n
    var j = j0
    while j + lanes <= j1:
        var va = p.unsafe_load[width=lanes](a0 + j)
        var vb = p.unsafe_load[width=lanes](a1 + j)
        comptime if order3:
            var vc = p.unsafe_load[width=lanes](a2 + j)
            var pv = va + q * vb + r * vc
            p.unsafe_store(a2 + j, vc - pv * zz)
            p.unsafe_store(a1 + j, vb - pv * y)
            p.unsafe_store(a0 + j, va - pv * x)
        else:
            var pv = va + q * vb
            p.unsafe_store(a1 + j, vb - pv * y)
            p.unsafe_store(a0 + j, va - pv * x)
        j += lanes
    while j < j1:
        var va = p[unsafe_offset=a0 + j]
        var vb = p[unsafe_offset=a1 + j]
        comptime if order3:
            var vc = p[unsafe_offset=a2 + j]
            var pv = va + q * vb + r * vc
            p[unsafe_offset=a2 + j] = vc - pv * zz
            p[unsafe_offset=a1 + j] = vb - pv * y
            p[unsafe_offset=a0 + j] = va - pv * x
        else:
            var pv = va + q * vb
            p[unsafe_offset=a1 + j] = vb - pv * y
            p[unsafe_offset=a0 + j] = va - pv * x
        j += 1


def _hqr[
    dtype: DType, wantt: Bool, wantz: Bool, gpu: Bool
](
    mut h: List[Scalar[dtype]],
    mut acc: _RotationBatch[dtype, gpu, wantz, 2, True],
    n: Int,
    ctx: DeviceContext,
) raises -> Tuple[
    List[Scalar[dtype]], List[Scalar[dtype]]
] where dtype.is_floating_point():
    """Francis double-shift QR on the upper Hessenberg `h` (row-major
    `n x n`, in place). EISPACK's `hqr2`, which is LAPACK's `dlahqr`.

    **Tier 2, host-side.** Each sweep chases a `3 x 3` bulge down the
    active block with Householder reflectors of order three, deflates on a
    test of the data, and takes an exceptional shift every tenth sweep
    without a deflation, alternating between the block's two ends --
    none of which has a GEMM to hand anything to. Returns the eigenvalues
    as `(real, imaginary)` in LAPACK's order; a complex pair sits in
    consecutive slots with the positive imaginary part first.

    With `wantt` the whole quasi-triangular `T` is kept -- the reflectors
    are applied to every column to the right and every row above -- and
    the strict lower band the chase never writes (the two entries each
    reflector annihilates in the column to its left, mathematically zero)
    is cleared at the end, so `T` is the Schur form and not the Schur form
    plus stale bulge entries. Without it only the active block is updated,
    which is all the eigenvalues need.

    **With `wantz` the transformations go into `acc`, not into a host
    matrix.** Every reflector the chase forms and every rotation the
    `2 x 2` real split forms is pushed there; `_RotationBatch` reorders a
    batch of sweeps into windows of commuting entries and applies each
    window to `Z^T` as one `matmul`. The chase ascends in `k` and a
    reflector reaches two columns past its index, so the batch is the
    `reach = 2`, `ascending` one -- its docstring carries the argument for
    why the reorder is exact. The split rotation is pushed as a sweep of
    its own, closed on both sides, so it needs no dependency special case.

    A converged `2 x 2` block with real eigenvalues is split by that one
    rotation, so every surviving `2 x 2` block on `T`'s diagonal has a
    complex conjugate pair -- the standard real Schur form.

    `block == 1` recovers the unblocked accumulation exactly: one entry
    per window, applied in stream order. Larger `block` only changes how
    many commuting entries ride in one product, which is why the blocking
    tests pin every size against every other.
    """
    var wr = List[Scalar[dtype]](length=n, fill=0)
    var wi = List[Scalar[dtype]](length=n, fill=0)
    if n == 0:
        return (wr^, wi^)
    var eps = _eps[dtype]()
    var zero = Scalar[dtype](0)

    var anorm = zero
    for i in range(n):
        var j0 = i - 1 if i > 0 else 0
        for j in range(j0, n):
            anorm += abs(h[i * n + j])

    var en = n - 1
    var itmax = _MAX_QR_SWEEPS_PER_N * max(10, n)
    var x: Scalar[dtype]
    var y: Scalar[dtype]
    var w: Scalar[dtype]
    var p = zero
    var q = zero
    var r = zero
    var zz: Scalar[dtype]
    var s: Scalar[dtype]

    while en >= 0:
        var its = 0
        while True:
            # A negligible subdiagonal splits off the block `l .. en`.
            var l = en
            while l > 0:
                s = abs(h[(l - 1) * n + (l - 1)]) + abs(h[l * n + l])
                if s == zero:
                    s = anorm
                if abs(h[l * n + (l - 1)]) <= eps * s:
                    h[l * n + (l - 1)] = zero
                    break
                l -= 1
            x = h[en * n + en]
            if l == en:
                # One root.
                wr[en] = x
                wi[en] = zero
                en -= 1
                break
            y = h[(en - 1) * n + (en - 1)]
            w = h[en * n + (en - 1)] * h[(en - 1) * n + en]
            if l == en - 1:
                # Two roots.
                p = Scalar[dtype](0.5) * (y - x)
                q = p * p + w
                zz = _sqrt(abs(q))
                if q >= zero:
                    # A real pair: split the block with one rotation.
                    zz = p + _copysign(zz, p)
                    wr[en - 1] = x + zz
                    wr[en] = wr[en - 1]
                    if zz != zero:
                        wr[en] = x - w / zz
                    wi[en - 1] = zero
                    wi[en] = zero
                    var xx = h[en * n + (en - 1)]
                    s = abs(xx) + abs(zz)
                    p = xx / s
                    q = zz / s
                    r = _sqrt(p * p + q * q)
                    p = p / r
                    q = q / r
                    var j_first = en - 1
                    for j in range(j_first, n if wantt else en + 1):
                        var top = h[(en - 1) * n + j]
                        h[(en - 1) * n + j] = q * top + p * h[en * n + j]
                        h[en * n + j] = q * h[en * n + j] - p * top
                    var i_first = 0 if wantt else l
                    for i in range(i_first, en + 1):
                        var left = h[i * n + (en - 1)]
                        h[i * n + (en - 1)] = q * left + p * h[i * n + en]
                        h[i * n + en] = q * h[i * n + en] - p * left
                    comptime if wantz:
                        # `Z <- Z G` with the new column `en - 1` equal to
                        # `q z_{en-1} + p z_en`, which is `push_rotation`'s
                        # `(c, s)` convention at `c = q`, `s = -p`. A sweep
                        # of its own on both sides, so the window tag needs
                        # no case for a rotation between two chases.
                        acc.end_sweep(ctx)
                        acc.push_rotation(en - 1, q, -p)
                        acc.end_sweep(ctx)
                    # The rotation annihilates this entry; write the zero it
                    # is, as `dlanv2` does, so the block partition downstream
                    # reads two real eigenvalues and not a complex block.
                    h[en * n + (en - 1)] = zero
                else:
                    wr[en - 1] = x + p
                    wr[en] = x + p
                    wi[en - 1] = zz
                    wi[en] = -zz
                en -= 2
                break

            if its >= itmax:
                raise Error(
                    "eigvals/schur: the QR iteration did not converge in ",
                    itmax,
                    (
                        " sweeps on one eigenvalue; the matrix likely holds a"
                        " NaN or an infinity"
                    ),
                )
            its += 1
            # LAPACK's `dlahqr` shift schedule. Every tenth sweep without a
            # deflation takes an exceptional shift, alternating between the
            # top of the active block and its bottom, so a cycle the
            # Wilkinson shifts fall into is broken wherever it sits and
            # broken again if it re-forms; the other sweeps use the
            # trailing `2 x 2`'s eigenvalues, and when those are real, the
            # one nearer `h[en, en]` twice.
            if its % (2 * _EXCEPTIONAL_EVERY) == 0:
                s = abs(h[en * n + (en - 1)]) + abs(h[(en - 1) * n + (en - 2)])
                x = Scalar[dtype](0.75) * s + h[en * n + en]
                y = x
                w = Scalar[dtype](-0.4375) * s * s
            elif its % _EXCEPTIONAL_EVERY == 0:
                s = abs(h[(l + 1) * n + l]) + abs(h[(l + 2) * n + (l + 1)])
                x = Scalar[dtype](0.75) * s + h[l * n + l]
                y = x
                w = Scalar[dtype](-0.4375) * s * s
            else:
                var half = Scalar[dtype](0.5) * (y - x)
                var disc = half * half + w
                if disc > zero:
                    var root = _sqrt(disc)
                    var mid = Scalar[dtype](0.5) * (x + y)
                    var near = mid + root
                    var far = mid - root
                    if abs(far - x) < abs(near - x):
                        near = far
                    x = near
                    y = near
                    w = zero

            # Two consecutive small subdiagonals let the sweep start below
            # `l`: the first column of the shift polynomial, and where it is
            # negligible against its neighbors.
            var m = en - 2
            while m >= l:
                zz = h[m * n + m]
                r = x - zz
                s = y - zz
                p = (r * s - w) / h[(m + 1) * n + m] + h[m * n + (m + 1)]
                q = h[(m + 1) * n + (m + 1)] - zz - r - s
                r = h[(m + 2) * n + (m + 1)]
                s = abs(p) + abs(q) + abs(r)
                p = p / s
                q = q / s
                r = r / s
                if m == l:
                    break
                var tst1 = abs(p) * (
                    abs(h[(m - 1) * n + (m - 1)])
                    + abs(zz)
                    + abs(h[(m + 1) * n + (m + 1)])
                )
                var tst2 = abs(h[m * n + (m - 1)]) * (abs(q) + abs(r))
                if tst2 <= eps * tst1:
                    break
                m -= 1
            for i in range(m + 2, en + 1):
                h[i * n + (i - 2)] = zero
                if i != m + 2:
                    h[i * n + (i - 3)] = zero

            # The double step: chase the bulge from `m` to `en`.
            for k in range(m, en):
                var notlast = k != en - 1
                if k != m:
                    p = h[k * n + (k - 1)]
                    q = h[(k + 1) * n + (k - 1)]
                    r = h[(k + 2) * n + (k - 1)] if notlast else zero
                    x = abs(p) + abs(q) + abs(r)
                    if x == zero:
                        continue
                    p = p / x
                    q = q / x
                    r = r / x
                s = _copysign(_sqrt(p * p + q * q + r * r), p)
                if k != m:
                    h[k * n + (k - 1)] = -s * x
                elif l != m:
                    h[k * n + (k - 1)] = -h[k * n + (k - 1)]
                p = p + s
                x = p / s
                y = q / s
                zz = r / s
                q = q / p
                r = r / p
                var j_last = n if wantt else en + 1
                if notlast:
                    _chase_rows[order3=True](h, k, n, k, j_last, x, y, zz, q, r)
                else:
                    _chase_rows[order3=False](
                        h, k, n, k, j_last, x, y, zz, q, r
                    )
                var i_last = en if en < k + 3 else k + 3
                for i in range(0 if wantt else l, i_last + 1):
                    p = x * h[i * n + k] + y * h[i * n + (k + 1)]
                    if notlast:
                        p = p + zz * h[i * n + (k + 2)]
                        h[i * n + (k + 2)] = h[i * n + (k + 2)] - p * r
                    h[i * n + (k + 1)] = h[i * n + (k + 1)] - p * q
                    h[i * n + k] = h[i * n + k] - p
                comptime if wantz:
                    acc.push_reflector(k, x, y, zz, q, r, notlast)
            comptime if wantz:
                acc.end_sweep(ctx)

    comptime if wantt:
        for i in range(n):
            for j in range(i - 1):
                h[i * n + j] = zero
    return (wr^, wi^)


struct Eigenvalues[dtype: DType, n: Int](
    Movable where dtype.is_floating_point() and n >= 1
):
    """`eigvals`'s result: the spectrum as a real and an imaginary tensor.

    A `dtype`-monomorphic `Tensor` cannot hold a `Complex`, the constraint
    `numax.fft`'s `Spectrum` answers the same way; `numax.linalg.eigvals`
    returns `Array[Complex[T], n]` because `Array` can. A complex pair sits
    in consecutive slots, positive imaginary part first, and a real
    eigenvalue has `im == 0` exactly.
    """

    var re: Static[Self.dtype, Self.n]
    """Real parts, in LAPACK's deflation order -- no particular order."""

    var im: Static[Self.dtype, Self.n]
    """Imaginary parts; exactly zero for a real eigenvalue."""

    def __init__(
        out self,
        var re: Static[Self.dtype, Self.n],
        var im: Static[Self.dtype, Self.n],
    ):
        self.re = re^
        self.im = im^


struct DynamicEigenvalues[dtype: DType](
    Movable where dtype.is_floating_point()
):
    """`eigvals`'s result at a run-time order: `Eigenvalues` with the
    order in the value rather than the type."""

    var re: Dynamic[Self.dtype, 1]
    """Real parts, in LAPACK's deflation order -- no particular order."""

    var im: Dynamic[Self.dtype, 1]
    """Imaginary parts; exactly zero for a real eigenvalue."""

    def __init__(
        out self,
        var re: Dynamic[Self.dtype, 1],
        var im: Dynamic[Self.dtype, 1],
    ):
        self.re = re^
        self.im = im^


def eigvals[
    T: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: T) raises -> Eigenvalues[T.dtype, dim[T, 0]] where (
    (T.dtype.is_floating_point() and block >= 1)
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
    and dim[T, 1] == dim[T, 0]
):
    """**Tier 2.** The eigenvalues of a general square `a`, real or
    complex, as a `(re, im)` pair. `numpy.linalg.eigvals`,
    `scipy.linalg.eigvals`.

    **`gpu=True`** runs the reduction and every `O(n^3)` product on the
    device; `sytrd` says what stays on the host and why.

    `hessenberg` reduces `a` device-resident with the cubic term in
    `linalg.matmul`, then the QR iteration runs on the Hessenberg matrix
    on the host, tier 2 by numax's definition. Above `n = 75` it is
    LAPACK's small-bulge multishift QR with aggressive early deflation
    (`numax/linalg/_multishift.mojo`, `dhseqr`'s path), whose sweep
    updates are window-sized products; at or below it, the Francis
    double-shift iteration in `_hqr`. Nothing is accumulated, so this is
    the cheap half of `schur`. `eigvalsh` is the symmetric route, with real output
    and a better algorithm; use it when the matrix is symmetric.

    Eigenvalues come out in LAPACK's deflation order, not sorted; a
    complex pair is adjacent with the positive imaginary part first.
    `numax.linalg.eigvals` is the fixed-sweep, differentiable
    sibling for matrices small enough to live in registers.

    `block` is the reduction's `lahr2` panel width, the one thing here it
    tunes -- nothing is accumulated, so the rotation batch is a single
    entry whatever it says. The deflation order is not continuous in `H`,
    and a different panel width moves `H` in the last bits, so two widths
    may report the same spectrum in a different order.

    Parameters:
        T: The `TensorLike` type of `a`; a square matrix of floating dtype.
        gpu: Run the reduction and every `O(n^3)` product on the device of `a`.
        block: The `hessenberg` panel width.

    Args:
        a: The general `n x n` matrix.

    Returns:
        An `Eigenvalues` holding the `n` real parts in `re` and imaginary
        parts in `im`, in deflation order.

    Raises:
        If the QR iteration does not converge within its iteration budget,
        which means `a` holds a NaN or an infinity, or if a device
        operation fails.
    """
    comptime n = dim[T, 0]
    var ctx = a.context()
    var re = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var im = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    _eigvals_run[gpu=gpu](a, n, block, re, im)
    return Eigenvalues[T.dtype, n](re^.as_static[n](), im^.as_static[n]())


def _eigvals_run[
    T: TensorLike, gpu: Bool
](
    a: T,
    n: Int,
    block: Int,
    mut re: Dynamic[T.dtype, 1],
    mut im: Dynamic[T.dtype, 1],
) raises where (T.dtype.is_floating_point() and T.LayoutType.rank == 2):
    """`eigvals`'s body at an order known only at run time, the one both
    overloads run: the Hessenberg reduction, then the multishift QR above
    `_NMIN` or `_hqr` at or below it. Writes the caller's `re` and `im`."""
    var ctx = a.context()
    var reduced = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    var reflectors = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    var taus = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    _gehrd_run[gpu=gpu](a, n, block, reduced, reflectors, taus)
    var h = reduced.to_host()
    # `wantz=False` makes the batch's `vectors` false, which shrinks every
    # buffer in it to one element; the values-only route pushes nothing and
    # allocates nothing for vectors it never forms.
    if n > _NMIN:
        var wr = List[Scalar[T.dtype]](length=n, fill=0)
        var wi = List[Scalar[T.dtype]](length=n, fill=0)
        var unused = List[Scalar[T.dtype]](length=1, fill=0)
        var info = _hseqr[T.dtype](False, False, n, h, wr, wi, unused)
        if info != 0:
            raise Error(
                "eigvals: the multishift QR iteration did not converge;"
                " the matrix likely holds a NaN or an infinity"
            )
        re.copy_from_host(wr)
        im.copy_from_host(wi)
        return
    var acc = _RotationBatch[T.dtype, gpu, False, 2, True](n, 1, ctx)
    var values = _hqr[wantt=False, wantz=False, gpu=gpu](h, acc, n, ctx)
    re.copy_from_host(values[0])
    im.copy_from_host(values[1])


struct Schur[dtype: DType, n: Int](
    Movable where dtype.is_floating_point() and n >= 1
):
    """`schur`'s result: `a = z t z^T` with `t` real quasi-triangular and
    `z` orthogonal. A struct rather than SciPy's `(T, Z)` tuple, for the
    reason `qr_factor` returns a `QR`.
    """

    var t: Static[Self.dtype, Self.n, Self.n]
    """The real Schur form: upper triangular except for `2 x 2` diagonal
    blocks, each holding one complex conjugate pair of eigenvalues, exact
    zeros below the band."""

    var z: Static[Self.dtype, Self.n, Self.n]
    """The Schur vectors, orthogonal: `z^T a z == t`."""

    def __init__(
        out self,
        var t: Static[Self.dtype, Self.n, Self.n],
        var z: Static[Self.dtype, Self.n, Self.n],
    ):
        self.t = t^
        self.z = z^


struct DynamicSchur[dtype: DType](Movable where dtype.is_floating_point()):
    """`schur`'s result at a run-time order: `Schur` with the order in the
    value rather than the type."""

    var t: Dynamic[Self.dtype, 2]
    """The real Schur form, quasi-triangular, exact zeros below the band."""

    var z: Dynamic[Self.dtype, 2]
    """The Schur vectors, orthogonal: `z^T a z == t`."""

    def __init__(
        out self,
        var t: Dynamic[Self.dtype, 2],
        var z: Dynamic[Self.dtype, 2],
    ):
        self.t = t^
        self.z = z^


def schur[
    T: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: T) raises -> Schur[T.dtype, dim[T, 0]] where (
    (T.dtype.is_floating_point() and block >= 1)
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
    and dim[T, 1] == dim[T, 0]
):
    """**Tier 2.** The real Schur decomposition `a = Z T Z^T`.
    `scipy.linalg.schur(a, output="real")`.

    **`gpu=True`** runs the reduction and every `O(n^3)` product on the
    device; `sytrd` says what stays on the host and why.

    Three steps. `hessenberg` reduces `a` device-resident. Above `n = 75`
    the multishift QR with aggressive early deflation
    (`numax/linalg/_multishift.mojo`) triangularizes the Hessenberg matrix
    on the host with the Hessenberg-level Schur vectors `Z_h` accumulated
    there too, every window's transformation applied to `T` and `Z_h` as
    one product, and the Schur vectors of `a` are `Q Z_h`, one `matmul`
    on the tensor's device. At or below it the Francis iteration runs
    instead, its order-three reflectors and `2 x 2` split rotations pushed
    into a `_RotationBatch` that applies each window of commuting entries
    to `Z^T` as one `matmul`, so `Z` stays on the device throughout; the
    Schur vectors are then `inner`'s `transpose_b=True` product of `Q`
    against that `Z^T`.

    `block` names three knobs, as `eigh`'s and `svd`'s do. `hessenberg`
    takes it as the **`lahr2` panel width** of the reduction and as the
    panel width `.q()` forms `Q` in. The Francis sweep takes it as the
    **rotation window**: `block // 2` consecutive sweeps batch together
    and a window spans at most `2 * block` columns of `Z^T`, the halving
    because a Francis reflector reaches two columns past its index where a
    Givens rotation reaches one. `block == 1` recovers all three unblocked
    algorithms exactly.

    A test pins every width against every other, and it starts from an
    already-Hessenberg matrix, which is load-bearing: the reduction's
    panel width moves `H` in the last bits, and the Francis chase's
    deflation order is **not** continuous in `H`. Two widths can therefore
    return different, equally valid real Schur forms of the same matrix --
    the same eigenvalues in a different order along the diagonal. A caller
    who needs a reproducible ordering fixes `block`.

    **`ponytail:` the QR iteration is host work.** At `n = 1024`,
    `float32`, on an M3 Pro, `schur` is about 250 ms, of which the
    `lahr2`-blocked `hessenberg` and `.q()` are most; the multishift
    iteration moved the host term from about 520 ms (the double-shift
    Francis loop carrying the full `T`) to about 100, because its
    far-from-diagonal updates are window-sized products rather than
    rank-3 updates per sweep. What stays host-side is the chase inside
    each window and the early-deflation Schur decompositions, which are
    sequential by construction.

    This is the form every matrix function in `numax.linalg.matfuncs`
    beyond `expm` is built on.

    Parameters:
        T: The `TensorLike` type of `a`; a square matrix of floating dtype.
        gpu: Run the reduction and every `O(n^3)` product on the device of `a`.
        block: The `hessenberg` panel width, the `.q()` panel width and
            twice the number of Francis sweeps batched per rotation window.

    Args:
        a: The general `n x n` matrix.

    Returns:
        A `Schur` holding the real quasi-triangular form `t` and the
        orthogonal Schur vectors `z`, with `a == z t z^T`.

    Raises:
        If the QR iteration does not converge within its iteration budget,
        which means `a` holds a NaN or an infinity, or if a device
        operation fails.
    """
    comptime n = dim[T, 0]
    var ctx = a.context()
    var t = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    var z = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    _schur_run[gpu=gpu](a, n, block, t, z)
    return Schur[T.dtype, n](t^.as_static[n, n](), z^.as_static[n, n]())


def eigvals[
    T: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: T) raises -> DynamicEigenvalues[T.dtype] where (
    (T.dtype.is_floating_point() and block >= 1)
    and T.LayoutType.rank == 2
    and not T.LayoutType.all_dims_known
):
    """**Tier 2.** The run-time-shape overload: the eigenvalues of a
    general square `a` whose order is data, as a `(re, im)` pair.
    `scipy.linalg.eigvals`.

    The static overload's body at the order `a` carries -- the Hessenberg
    reduction, then the multishift QR above `n = 75` or the double-shift
    iteration at or below it -- so one compiled program takes any size
    and the two overloads agree to the bit.

    Parameters:
        T: The `TensorLike` type of `a`, a rank-2 run-time-shaped matrix.
        gpu: Run the reduction and every `O(n^3)` product on `a`'s device.
        block: The `lahr2` panel width.

    Args:
        a: The `n x n` matrix.

    Returns:
        A `DynamicEigenvalues` with the real parts in `re` and the
        imaginary parts in `im`, a complex pair in consecutive slots.

    Raises:
        If `a` is not square or is empty, if the QR iteration does not
        converge (`a` holds a NaN or an infinity), or if a device
        operation fails.
    """
    var n = _runtime_order(a, "eigvals")
    var ctx = a.context()
    var re = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var im = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    _eigvals_run[gpu=gpu](a, n, block, re, im)
    return DynamicEigenvalues[T.dtype](re^, im^)


def schur[
    T: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: T) raises -> DynamicSchur[T.dtype] where (
    (T.dtype.is_floating_point() and block >= 1)
    and T.LayoutType.rank == 2
    and not T.LayoutType.all_dims_known
):
    """**Tier 2.** The run-time-shape overload: the real Schur
    decomposition `a = Z T Z^T` of a square `a` whose order is data.
    `scipy.linalg.schur(a, output="real")`.

    The static overload's body at the order `a` carries, returning a
    `DynamicSchur`, so one compiled program takes any size and the two
    overloads agree to the bit.

    Parameters:
        T: The `TensorLike` type of `a`, a rank-2 run-time-shaped matrix.
        gpu: Run the reduction and every `O(n^3)` product on `a`'s device.
        block: The `lahr2` panel width, the rotation window and the width
            `Q` is formed in.

    Args:
        a: The `n x n` matrix.

    Returns:
        A `DynamicSchur` with the quasi-triangular `t` and the orthogonal
        Schur vectors `z`.

    Raises:
        If `a` is not square or is empty, if the QR iteration does not
        converge (`a` holds a NaN or an infinity), or if a device
        operation fails.
    """
    var n = _runtime_order(a, "schur")
    var ctx = a.context()
    var t = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    var z = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    _schur_run[gpu=gpu](a, n, block, t, z)
    return DynamicSchur[T.dtype](t^, z^)


def _schur_run[
    T: TensorLike, gpu: Bool
](
    a: T,
    n: Int,
    block: Int,
    mut t: Dynamic[T.dtype, 2],
    mut vectors: Dynamic[T.dtype, 2],
) raises where (T.dtype.is_floating_point() and T.LayoutType.rank == 2):
    """`schur`'s body at an order known only at run time, the one both
    overloads run: the Hessenberg reduction, the QR iteration with its
    Schur vectors (multishift above `_NMIN`, `_hqr` with batched device
    rotations at or below it), and `Z = Q Z_h` as one product. Writes the
    caller's `t` and `vectors`."""
    var ctx = a.context()
    var reduced = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    var reflectors = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    var taus = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    _gehrd_run[gpu=gpu](a, n, block, reduced, reflectors, taus)
    var h = reduced.to_host()
    var q = _accumulate_reflectors_run[gpu=gpu](
        reflectors, taus, n, n - 2, block
    )
    if n > _NMIN:
        var wr = List[Scalar[T.dtype]](length=n, fill=0)
        var wi = List[Scalar[T.dtype]](length=n, fill=0)
        var z = List[Scalar[T.dtype]](length=n * n, fill=0)
        for i in range(n):
            z[i * n + i] = 1
        var info = _hseqr[T.dtype](True, True, n, h, wr, wi, z)
        if info != 0:
            raise Error(
                "schur: the multishift QR iteration did not converge;"
                " the matrix likely holds a NaN or an infinity"
            )
        t.copy_from_host(h)
        # `_hseqr` leaves `Z_h` itself, row-major, so this is a plain
        # product rather than `_hqr`'s transposed batch.
        var zh = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
        zh.copy_from_host(z)
        _max_matmul[target=_target[gpu]()](
            vectors.tile(), q.tile(), zh.tile(), ctx
        )
        ctx.synchronize()
        _ = zh^
        _ = q^
        return
    var acc = _RotationBatch[T.dtype, gpu, True, 2, True](n, block, ctx)
    _ = _hqr[wantt=True, wantz=True, gpu=gpu](h, acc, n, ctx)
    acc.finish(ctx)
    t.copy_from_host(h)
    # `acc.zt` holds `Z_h^T`: `inner`'s `transpose_b=True`, so it is never
    # transposed back.
    _max_matmul[transpose_b=True, target=_target[gpu]()](
        vectors.tile(), q.tile(), acc.zt.tile(), ctx
    )
    ctx.synchronize()
    _ = acc^
    _ = q^


# ----------------------------------------------------------------- SVD


def _left_products[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, v: B, m: Int, n: Int) raises -> Dynamic[A.dtype, 1] where (
    A.LayoutType.rank == 2 and B.dtype == A.dtype and B.LayoutType.rank == 1
):
    """`v^T A` as an `n`-vector: `v` read as a `1 x m` row against the
    `m x n` `A`."""
    var ctx = a.context()
    var result = zeros_dyn[A.dtype, 1](n, ctx=ctx)
    var row: _Dense[A.dtype] = TileTensor(
        _mut_view_as[A.dtype](v).ptr_at_offset(Coord(0)), row_major(Coord(1, m))
    )
    var out: _Dense[A.dtype] = TileTensor(
        result.tile().ptr_at_offset(Coord(0)), row_major(Coord(1, n))
    )
    _max_matmul[target=_target[gpu]()](out, row, _mut_view(a), ctx)
    ctx.synchronize()
    return result^


def _column_of[
    T: TensorLike,
    gpu: Bool = False,
](dense: T, k: Int, m: Int) raises -> Dynamic[T.dtype, 1] where (
    T.LayoutType.rank == 2
):
    """Column `k` of a dense `m x n` as its own vector, gathered on the
    device."""
    var ctx = dense.context()
    var out = zeros_dyn[T.dtype, 1](m, ctx=ctx)
    var src = _mut_view(dense)
    var dst = out.tile()

    @always_inline
    def gather[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var src, var dst, var k}:
        var i = coord_to_index_list(coord)[0]
        dst.store[1](coord, src[Coord(i, k)])

    elementwise[simd_width=1, target=_target[gpu]()](gather, Coord(m), ctx)
    return out^


def _row_of[
    T: TensorLike,
    gpu: Bool = False,
](dense: T, k: Int, n: Int) raises -> Dynamic[T.dtype, 1] where (
    T.LayoutType.rank == 2
):
    """Row `k` of a dense `m x n` as its own vector, on the device -- a
    contiguous copy, so a `pack_block` of one row."""
    var ctx = dense.context()
    var out = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var dst: _Dense[T.dtype] = TileTensor(
        out.tile().ptr_at_offset(Coord(0)), row_major(Coord(1, n))
    )
    pack_block[target=_target[gpu]()](_mut_view(dense), dst, k, 0, 1, n, ctx)
    ctx.synchronize()
    return out^


struct Bidiagonal[dtype: DType, m: Int, n: Int, gpu: Bool = False](
    Movable where dtype.is_floating_point() and m >= n and n >= 1
):
    """The bidiagonal reduction `A = Q B P^T` of an `m x n` matrix with
    `m >= n`, device-resident. LAPACK's `gebrd` output.

    `B` is upper bidiagonal: `d` on the diagonal, `e[0 .. n-2]` on the
    superdiagonal (`e[n-1]` is unused and zero, for the reason
    `Tridiagonal` gives). `Q` (`m x n`, the thin form) and `P`
    (`n x n`) are orthogonal and held as their Householder vectors -- one
    full-length vector per column of `left` and per row of `right` -- so
    `.q()` and `.p()` materialize them and `svdvals` never asks.
    """

    var d: Static[Self.dtype, Self.n]
    """`B`'s diagonal."""

    var e: Static[Self.dtype, Self.n]
    """`B`'s superdiagonal in `e[0 .. n-2]`; `e[n-1]` unused and zero."""

    var left: Static[Self.dtype, Self.m, Self.n]
    """Column `k` is the left reflector `v_k`, full length, `1` at `k`."""

    var right: Static[Self.dtype, Self.n, Self.n]
    """Row `k` is the right reflector `u_k`, full length, `1` at `k + 1`;
    rows `n-2` and `n-1` are unused."""

    var taus_left: Static[Self.dtype, Self.n]
    var taus_right: Static[Self.dtype, Self.n]

    var block: Int
    """The `labrd` panel width the reduction ran at, which is also the
    width `.q()` and `.p()` form their factors in. Carried from `gebrd`."""

    def __init__(
        out self,
        var d: Static[Self.dtype, Self.n],
        var e: Static[Self.dtype, Self.n],
        var left: Static[Self.dtype, Self.m, Self.n],
        var right: Static[Self.dtype, Self.n, Self.n],
        var taus_left: Static[Self.dtype, Self.n],
        var taus_right: Static[Self.dtype, Self.n],
        block: Int,
    ):
        self.d = d^
        self.e = e^
        self.left = left^
        self.right = right^
        self.taus_left = taus_left^
        self.taus_right = taus_right^
        self.block = block

    def q(
        mut self,
    ) raises -> Static[
        Self.dtype, Self.m, Self.n
    ] where Self.dtype.is_floating_point():
        """The thin `Q`, `m x n`: `H_0 H_1 ... H_{n-1}` applied to the
        first `n` columns of the identity, panels of `block` reflectors in
        reverse order. LAPACK's `orgbr` with `vect='Q'`.

        This is `QR.q()`'s walk unchanged, because `left` is already
        in QR's packed form: `gebd2_col` writes the reflector's unit at row
        `k` -- the row a QR puts it on -- so no row-shifted view is needed
        here, unlike `orgtr`'s. `pack_reflectors` forces the diagonal to
        `1` and reads only the strict lower trapezoid, so the explicit unit
        and the explicit zeros above it are ignored either way.

        Three matrix products per panel against the `3n` launches and `2n`
        device synchronizations the per-reflector walk needed. `svdvals`
        never calls this.
        """
        return _orgbr_q_run[gpu=Self.gpu](
            self.left, self.taus_left, Self.m, Self.n, self.block
        ).as_static[Self.m, Self.n]()

    def p(
        mut self,
    ) raises -> Static[
        Self.dtype, Self.n, Self.n
    ] where Self.dtype.is_floating_point():
        """`P`, `n x n`: `G_0 G_1 ... G_{n-3}` applied to the identity in
        panels of `block`, the same walk `q()` runs. LAPACK's `orgbr` with
        `vect='P'`.

        One adapter, and it is a copy rather than a view: the right
        reflectors are held as *rows* of `right`, so one
        `pack_block[trans=True]` turns them into columns. That transpose
        lands them in `sytrd`'s packed form -- the unit at row `k + 1`, one
        below a QR's -- so `_accumulate_reflectors` takes it from there
        through the row-shifted view it already uses for `orgtr`.

        `right`'s last two rows carry no reflector (`gebd2_row` at
        `k = n - 2` sees a one-element vector and returns `tau = 0`), which
        is why the walk is over `n - 2` of them.
        """
        return _orgbr_p_run[gpu=Self.gpu](
            self.right, self.taus_right, Self.n, self.block
        ).as_static[Self.n, Self.n]()


def _orgbr_q_run[
    L: TensorLike,
    B: TensorLike,
    gpu: Bool,
](left: L, taus: B, m: Int, n: Int, block: Int) raises -> Dynamic[
    L.dtype, 2
] where (
    L.dtype.is_floating_point()
    and L.LayoutType.rank == 2
    and B.dtype == L.dtype
    and B.LayoutType.rank == 1
):
    """`Bidiagonal.q()`'s walk at run-time extents: the thin `m x n` `Q`
    from the left reflectors packed in `left`, panels of `block`."""
    var ctx = left.context()
    var result = zeros_dyn[L.dtype, 2](m, n, ctx=ctx)
    var out: _Dense[L.dtype] = TileTensor(
        result.tile().ptr_at_offset(Coord(0, 0)), row_major(Coord(m, n))
    )
    _device_identity[L.dtype, gpu](out, m, n, ctx)

    var lv = _mut_view(left)
    var tv = _mut_view_as[L.dtype](taus)
    var work = _ReflectorWork[L.dtype](m, n, block, ctx)
    var steps = (n + block - 1) // block
    for step in range(steps):
        var k = (steps - 1 - step) * block
        var nb = min(block, n - k)
        _apply_block_reflector[transposed=False, gpu=gpu](
            lv, tv, out, k, nb, m, m - k, n - k, k, k, work, ctx
        )

    ctx.synchronize()
    return result^


def _orgbr_p_run[
    R: TensorLike,
    B: TensorLike,
    gpu: Bool,
](right: R, taus: B, n: Int, block: Int) raises -> Dynamic[R.dtype, 2] where (
    R.dtype.is_floating_point()
    and R.LayoutType.rank == 2
    and B.dtype == R.dtype
    and B.LayoutType.rank == 1
):
    """`Bidiagonal.p()`'s walk at a run-time order: the right reflectors,
    rows of `right`, transposed into `sytrd`'s packed form and handed to
    `_accumulate_reflectors_run`."""
    var ctx = right.context()
    var packed = zeros_dyn[R.dtype, 2](n, n, ctx=ctx)
    var pv = packed.tile()
    pack_block[trans=True, target=_target[gpu]()](
        _mut_view(right), pv, 0, 0, n, n, ctx
    )
    return _accumulate_reflectors_run[gpu=gpu](packed, taus, n, n - 2, block)


def gebrd[
    T: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: T) raises -> Bidiagonal[T.dtype, dim[T, 0], dim[T, 1], gpu] where (
    (
        T.dtype.is_floating_point()
        and dim[T, 0] >= dim[T, 1]
        and dim[T, 1] >= 1
        and block >= 1
    )
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
):
    """**Tier 2.** Reduce an `m x n` matrix, `m >= n`, to upper bidiagonal
    form by alternating left and right Householder reflections,
    device-resident and blocked. LAPACK's `gebrd` over `labrd` panels.

    **`gpu=True`** runs the reduction and every `O(n^3)` product on the
    device; `sytrd` says what stays on the host and why.

    Column `k` takes a left reflector from `a[k.., k]` and row `k` a right
    one from `a[k, k+1..]`. What blocking changes is when the rest of the
    matrix hears about them: a panel of `block` columns is reduced without
    touching the trailing block, and the whole panel then goes out as two
    GEMMs,

        A[k0+nb:, k0+nb:] -= V Y^T + X U

    with the subtraction fused into each one's epilogue. `V` is the
    panel's left reflectors as columns and `U` its right reflectors as
    rows -- the two the factorization keeps anyway -- while `Y` and `X`
    are the corrections the panel accumulates, `Y[:, j] = tauq (A^T v)`
    and `X[:, j] = taup (A u)` each brought up to date with the panel's
    own pairs. `labrd_column` and `labrd_row` apply those pairs to the one
    column and the one row the next reflector is taken from, `labrd_y` and
    `labrd_x` build the corrections, and every `O((m + n) j)` term among
    them is threaded the way `latrd_w`'s is.

    **That halves the whole-matrix traffic.** The unblocked reduction made
    four passes over the matrix per column -- `v^T A`, a rank-one
    subtract, `A u`, another rank-one subtract -- and this makes two, the
    two products, with the subtracts deferred into one rank-`block` GEMM
    per panel per side. `block == 1` is the unblocked algorithm (one
    column per panel, the update restricted to exactly the rows and
    columns the masked rank-one updates reached) and `block == n` is one
    panel whose trailing GEMMs never run; a test pins every width against
    `block == n`.

    The trailing update is **restricted to rows and columns `k0 + nb`
    onward**. Everything the panel finished -- the band entries, the zeros
    `gebd2_col` and `gebd2_row` wrote, and the rows and columns they wrote
    them in -- is outside it, which is what the masks on `Y` and `X`
    already say and what makes the restriction free rather than load-
    bearing here.

    `block` is also the panel width `.q()` and `.p()` form `Q` and `P` in.

    `ponytail:` the two matrix-vector products stay, for the reason
    `sytrd` gives about `p = A v` -- they are taken over the whole matrix
    because a strided sub-block cannot be handed to `matmul` without
    staging it dense, and LAPACK's `dgebrd` is half BLAS-2 for the same
    reason. Closing that needs a two-stage reduction, not a wider panel.
    """
    comptime m = dim[T, 0]
    comptime n = dim[T, 1]
    var ctx = a.context()
    var left = zeros_dyn[T.dtype, 2](m, n, ctx=ctx)
    var right = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    var taus_left = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var taus_right = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var d = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var e = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    _gebrd_run[gpu=gpu](
        a, m, n, block, left, right, taus_left, taus_right, d, e
    )
    return Bidiagonal[T.dtype, m, n, gpu](
        d^.as_static[n](),
        e^.as_static[n](),
        left^.as_static[m, n](),
        right^.as_static[n, n](),
        taus_left^.as_static[n](),
        taus_right^.as_static[n](),
        block,
    )


def _gebrd_run[
    T: TensorLike, gpu: Bool
](
    a: T,
    m: Int,
    n: Int,
    block: Int,
    mut left: Dynamic[T.dtype, 2],
    mut right: Dynamic[T.dtype, 2],
    mut taus_left: Dynamic[T.dtype, 1],
    mut taus_right: Dynamic[T.dtype, 1],
    mut d: Dynamic[T.dtype, 1],
    mut e: Dynamic[T.dtype, 1],
) raises where (T.dtype.is_floating_point() and T.LayoutType.rank == 2):
    """`gebrd`'s blocked reduction of the `m x n` `a`, `m >= n`, at
    extents known only at run time: the loop both the static overload and
    the run-time `svd`/`svdvals` run. Writes the caller's zeroed `left`
    (`m x n`), `right` (`n x n`), `taus_left`, `taus_right`, `d` and `e`
    (`n` each), as `_sytrd_run` writes its outputs."""
    var width = min(block, n)
    var ctx = a.context()
    var work = zeros_dyn[T.dtype, 2](m, n, ctx=ctx)
    var scratch = zeros_dyn[T.dtype, 1](_PANEL_THREADS + 1, ctx=ctx)
    # The panel's two corrections, the two operands the trailing GEMMs
    # read them against, and the reductions `labrd_y`/`labrd_x` share.
    var yy = zeros_dyn[T.dtype, 2](n, width, ctx=ctx)
    var xx = zeros_dyn[T.dtype, 2](m, width, ctx=ctx)
    var vp = zeros_dyn[T.dtype, 2](m, width, ctx=ctx)
    var ut = zeros_dyn[T.dtype, 2](n, width, ctx=ctx)
    var red = zeros_dyn[T.dtype, 1](2 * width + 2, ctx=ctx)
    var product = zeros_dyn[T.dtype, 2](m, n, ctx=ctx)

    var wv = work.tile()
    var lv = left.tile()
    var rv = right.tile()
    var tlv = taus_left.tile()
    var trv = taus_right.tile()
    var sv = scratch.tile()
    var yv = yy.tile()
    var xv = xx.tile()
    var vpv = vp.tile()
    var utv = ut.tile()
    var redv = red.tile()
    var pv = product.tile()

    pack_block[target=_target[gpu]()](_mut_view(a), wv, 0, 0, m, n, ctx)

    var k0 = 0
    while k0 < n:
        var nb = min(width, n - k0)

        # A ragged panel writes only `nb` of the `width` columns the
        # trailing GEMMs read. Clearing the two corrections is enough --
        # each GEMM has one of them as an operand, so a zero column there
        # kills the stale column beside it -- and only the last panel can
        # be narrow, so this runs at most once.
        if nb < width:

            @always_inline
            def clear_y[w: Int, alignment: Int = 1](coord: Coord) {var yv}:
                yv.store[1](coord, Scalar[T.dtype](0))

            @always_inline
            def clear_x[w: Int, alignment: Int = 1](coord: Coord) {var xv}:
                xv.store[1](coord, Scalar[T.dtype](0))

            elementwise[simd_width=1, target=_target[gpu]()](
                clear_y, Coord(n, width), ctx
            )
            elementwise[simd_width=1, target=_target[gpu]()](
                clear_x, Coord(m, width), ctx
            )

        for j in range(nb):
            var i = k0 + j

            labrd_column[target=_target[gpu]()](
                wv, lv, xv, yv, rv, k0, j, m, ctx
            )

            comptime if gpu:
                ctx.enqueue_function[
                    gebd2_col[
                        T.dtype,
                        ALayout=type_of(wv).LayoutType,
                        VLayout=type_of(lv).LayoutType,
                        TauLayout=type_of(tlv).LayoutType,
                        SLayout=type_of(sv).LayoutType,
                        gpu=True,
                    ]
                ](
                    wv,
                    lv,
                    tlv,
                    sv,
                    Int32(i),
                    Int32(m),
                    grid_dim=1,
                    block_dim=_PANEL_THREADS,
                )
                ctx.synchronize()
            else:
                gebd2_col(wv, lv, tlv, sv, Int32(i), Int32(m))

            # `v^T A` over the whole matrix. The panel's own columns hold
            # only the band entries and zeros `gebd2_col` wrote, and `v`
            # vanishes above row `i`, so nothing needs staging; the
            # panel's deferred pairs reach the product inside `labrd_y`.
            var v = _column_of[gpu=gpu](left, i, m)
            var t1 = _left_products[gpu=gpu](work, v, m, n)
            labrd_y[target=_target[gpu]()](
                t1.tile(), lv, xv, yv, rv, redv, tlv, k0, j, m, n, ctx
            )
            # `v` and `t1` are last mentioned through a view, and a view
            # erases the origin; see `findings.mdc` on the queued free.
            _ = v^
            _ = t1^

            if i + 1 < n:
                labrd_row[target=_target[gpu]()](
                    wv, lv, xv, yv, rv, k0, j, n, ctx
                )

                comptime if gpu:
                    ctx.enqueue_function[
                        gebd2_row[
                            T.dtype,
                            ALayout=type_of(wv).LayoutType,
                            ULayout=type_of(rv).LayoutType,
                            TauLayout=type_of(trv).LayoutType,
                            SLayout=type_of(sv).LayoutType,
                            gpu=True,
                        ]
                    ](
                        wv,
                        rv,
                        trv,
                        sv,
                        Int32(i),
                        Int32(n),
                        grid_dim=1,
                        block_dim=_PANEL_THREADS,
                    )
                    ctx.synchronize()
                else:
                    gebd2_row(wv, rv, trv, sv, Int32(i), Int32(n))

                var u = _row_of[gpu=gpu](right, i, n)
                var t2 = matvec[gpu=gpu](work, u)
                labrd_x[target=_target[gpu]()](
                    t2.tile(), lv, xv, yv, rv, redv, trv, k0, j, m, n, ctx
                )
                _ = u^
                _ = t2^

        # `V` is a column range of `left` and `U^T` a transposed row range
        # of `right`, neither dense, so both are packed before the GEMMs
        # -- once per panel, not once per column.
        pack_block[target=_target[gpu]()](lv, vpv, 0, k0, m, nb, ctx)
        pack_block[trans=True, target=_target[gpu]()](
            rv, utv, 0, k0, n, nb, ctx
        )

        var base = k0 + nb
        _subtract_panel[gpu=gpu](
            wv, vpv, yv, pv, base, base, m - base, n - base, width, ctx
        )
        _subtract_panel[gpu=gpu](
            wv, xv, utv, pv, base, base, m - base, n - base, width, ctx
        )
        k0 += nb

    # Read by the launches above through origin-erased views whose owners
    # are named nowhere else; see `findings.mdc` on the queued free.
    _ = scratch^
    _ = yy^
    _ = xx^
    _ = vp^
    _ = ut^
    _ = red^
    _ = product^

    var dv = d.tile()
    var ev = e.tile()

    # The band, on the device: `work`'s diagonal and first superdiagonal.
    # `e[n-1]` is the documented unused entry and is written zero.
    @always_inline
    def band[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var wv, var dv, var ev, var n}:
        var at = coord_to_index_list(coord)[0]
        dv.store[1](coord, wv[Coord(at, at)])
        var above = Scalar[T.dtype](0)
        if at + 1 < n:
            above = wv[Coord(at, at + 1)]
        ev.store[1](coord, above)

    elementwise[simd_width=1, target=_target[gpu]()](band, Coord(n), ctx)
    ctx.synchronize()
    # `work` is not returned, and after the panel loop it is read only
    # through `wv`, which erases the origin -- so without this Mojo frees
    # it before `band` runs and the band comes back as heap garbage. See
    # `findings.mdc` on the `.tile()` lifetime bug that `print` hides.
    _ = work^


def _golub_kahan[
    dtype: DType,
    n: Int,
    gpu: Bool = False,
    vectors: Bool = False,
](
    d: List[Scalar[dtype]],
    e: List[Scalar[dtype]],
    mut acc: _RotationBatch[dtype, gpu, vectors],
    ctx: DeviceContext,
) raises -> List[Scalar[dtype]] where dtype.is_floating_point():
    """The eigenvalues of the Golub-Kahan tridiagonal of the upper
    bidiagonal `(d, e)`, unsorted; at `vectors=True` its eigenvector matrix
    is left in `acc.zt`, **transposed** and device-resident, so row `j` is
    eigenvector `j`.

    **Nothing in the library calls this. It is `_bdsqr`'s test oracle**,
    kept because the two routes share no arithmetic: this one diagonalizes
    a `2n x 2n` symmetric tridiagonal by implicit QL, and `_bdsqr` chases
    a bulge down the `n`-long bidiagonal, so agreement between them is
    evidence rather than a tautology.
    `test_bdsqr_matches_the_golub_kahan_oracle` is the pin.

    The Golub-Kahan matrix is the `2n x 2n` symmetric tridiagonal with zero
    diagonal and off-diagonal `d_1, e_1, d_2, e_2, ..., d_n`. Its
    eigenvalues are `+-sigma_i`, and the eigenvector for `+sigma_i`
    interleaves the singular vectors: even entries are `v_i / sqrt(2)`, odd
    entries `u_i / sqrt(2)`. So the sweep `eigh` runs gives the SVD of `B`
    with no new numerics -- LAPACK's `dbdsvdx` takes this route, and `svd`
    did until `_bdsqr` landed.

    The accumulator is the caller's, the way `_tql`'s is: `acc.zt` is the
    `2n x 2n` device tensor the eigenvectors are left in, and `block` is
    the window it was built with rather than a second parameter here.
    Singular vectors for an *exactly* zero singular value are not
    guaranteed orthonormal on this route, since the two zero eigenvalues'
    vectors may mix; the values are exact.
    """
    comptime size = 2 * n
    var gd = List[Scalar[dtype]](length=size, fill=0)
    var ge = List[Scalar[dtype]](length=size, fill=0)
    for i in range(n):
        ge[2 * i] = d[i]
        if i + 1 < n:
            ge[2 * i + 1] = e[i]
    _tql[gpu=gpu, vectors=vectors](gd, ge, acc, ctx)
    acc.finish(ctx)
    return gd^


@always_inline
def _tiny[dtype: DType]() -> Scalar[dtype]:
    """The smallest normal of `dtype` -- `dbdsqr`'s `UNFL`, the floor under
    the deflation threshold so an all-but-zero band still terminates."""
    comptime if dtype == DType.float32:
        return Scalar[dtype](1.1754944e-38)
    else:
        return Scalar[dtype](2.2250738585072014e-308)


@always_inline
def _bdsqr_tol[dtype: DType]() -> Scalar[dtype]:
    """`dbdsqr`'s relative tolerance, `max(10, min(100, eps^(-1/8))) * eps`,
    evaluated per `dtype` the way `_eps` is rather than through a `pow`.

    `float32`: `eps^(-1/8)` is 7.34, so the `10` wins and the tolerance is
    `10 eps`. `float64`: it is 90.5, so that wins and the tolerance is
    `90.5 eps = 2.01e-14`. Asking for relative rather than absolute
    accuracy is what makes the zero-shift branch worth having.
    """
    comptime if dtype == DType.float32:
        return Scalar[dtype](1.1920929e-06)
    else:
        return Scalar[dtype](2.0097183471152322e-14)


@always_inline
def _lartg[
    dtype: DType
](f: Scalar[dtype], g: Scalar[dtype]) -> Tuple[
    Scalar[dtype], Scalar[dtype], Scalar[dtype]
] where dtype.is_floating_point():
    """LAPACK's `lartg`: `(c, s, r)` with `c f + s g = r` and
    `-s f + c g = 0`, normalized so `c > 0` whenever `|f| > |g|` -- which
    is what makes the factor unique and two runs of the same sweep agree
    sign for sign."""
    if g == 0:
        return (Scalar[dtype](1), Scalar[dtype](0), f)
    if f == 0:
        return (Scalar[dtype](0), Scalar[dtype](1), g)
    var r = _hypot(f, g)
    var c = f / r
    var s = g / r
    if abs(f) > abs(g) and c < 0:
        return (-c, -s, -r)
    return (c, s, r)


@always_inline
def _las2[
    dtype: DType
](f: Scalar[dtype], g: Scalar[dtype], h: Scalar[dtype]) -> Tuple[
    Scalar[dtype], Scalar[dtype]
] where dtype.is_floating_point():
    """The singular values `(smin, smax)` of the upper triangular `2 x 2`
    `[[f, g], [0, h]]`. LAPACK's `las2`, transcribed for its scaling: the
    obvious `sqrt` of the eigenvalues of the `2 x 2` normal equations
    loses half the digits of a small singular value."""
    var fa = abs(f)
    var ga = abs(g)
    var ha = abs(h)
    var fhmn = min(fa, ha)
    var fhmx = max(fa, ha)
    var one = Scalar[dtype](1)
    var two = Scalar[dtype](2)
    if fhmn == 0:
        if fhmx == 0:
            return (Scalar[dtype](0), ga)
        var big = max(fhmx, ga)
        var small = min(fhmx, ga)
        var ratio = small / big
        return (Scalar[dtype](0), big * _sqrt(one + ratio * ratio))
    if ga < fhmx:
        var a_s = one + fhmn / fhmx
        var at = (fhmx - fhmn) / fhmx
        var au = (ga / fhmx) * (ga / fhmx)
        var c = two / (_sqrt(a_s * a_s + au) + _sqrt(at * at + au))
        return (fhmn * c, fhmx / c)
    var au = fhmx / ga
    if au == 0:
        return ((fhmn * fhmx) / ga, ga)
    var a_s = one + fhmn / fhmx
    var at = (fhmx - fhmn) / fhmx
    var c = one / (
        _sqrt(one + (a_s * au) * (a_s * au))
        + _sqrt(one + (at * au) * (at * au))
    )
    var smin = (fhmn * c) * au
    return (smin + smin, ga / (c + c))


comptime _MAX_BDSQR_ITER_PER_N = 6
"""Bulge-chase steps allowed in total, per `n^2` -- LAPACK's `MAXITR` in
`dbdsqr`, where the budget is `6 n^2` and each step spends the length of
the block it chased. Reaching it means the bidiagonal carries a NaN or an
infinity, and the raise says so rather than looping forever."""


def _bdsqr[
    dtype: DType, gpu: Bool, vectors: Bool
](
    mut d: List[Scalar[dtype]],
    mut e: List[Scalar[dtype]],
    mut uacc: _RotationBatch[dtype, gpu, vectors, 1, True],
    mut vacc: _RotationBatch[dtype, gpu, vectors, 1, True],
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """The implicit-shift QR iteration on the upper bidiagonal `(d, e)`,
    in place: on return `d` holds the singular values -- unsorted and of
    either sign -- and `e` is zero. LAPACK's `dbdsqr`, Demmel and Kahan
    1990.

    At `vectors=True` the two rotation streams are pushed into `uacc` and
    `vacc`, so `uacc.zt` is `U_B^T` and `vacc.zt` is `V_B^T` for
    `B = U_B diag(d) V_B^T`. The accumulators are the caller's, the way
    `_tql`'s is: `svd` reads them after this returns.

    **This is what replaces the Golub-Kahan doubling.** `_golub_kahan`
    embeds `B` in the `2n x 2n` symmetric tridiagonal whose eigenvalues
    are `+-sigma` and runs `_tql` there, which is about `4n^2` rotations
    at width `2n` and then a de-interleave. The sweep here rotates `U` and
    `V` directly at width `n` -- about `2n^2` rotations against stripes a
    quarter the area -- and the singular values come out of the band
    itself rather than as half of a symmetric spectrum. `_golub_kahan`
    stays as the test oracle and nothing in the library calls it.

    **Tier 2, host-side**, for `_tql`'s reason: the chase is sequential,
    the deflation test branches on the data, and there is no GEMM to hand
    `O(n^2)` of band arithmetic to. The *transformations* are the cubic
    term and they go out as GEMMs through the batches.

    **One direction.** `dbdsqr` chases top-down or bottom-up per block,
    choosing by `|d[ll]| >= |d[m]|` so the criterion that preserves
    relative accuracy runs along the grading. This chases **top-down**
    only, and the reason is the window tag: a top-down chase has its index
    *rising* within a sweep, which is the `ascending` batch (`reach = 1`,
    tag `(k + s) // block` applied in increasing order), while a bottom-up
    one has it falling and needs the descending tag `_tql` uses. One
    accumulator cannot be both, and two would double every buffer. The
    cost is that a matrix graded the wrong way takes more sweeps; the
    absolute threshold still terminates it. A `dbdsqr` that picks the
    direction per block is the upgrade, and it needs a batch whose tag is
    chosen per flush.

    The rest is `dbdsqr`'s: split the band at a negligible `e`, deflate
    from the bottom, take the zero shift when a nonzero one would cost
    relative accuracy (`n tol (sminl / smax) <= max(eps, tol/100)`, plus
    the `(shift/sll)^2 < eps` test), otherwise the shift is the smallest
    singular value of the trailing `2 x 2` from `_las2`, and chase the
    bulge with one rotation on the right and one on the left per column.

    Both streams push `(i, cos, -sin)`: `lartg`'s pair applies as
    `[[c, -s], [s, c]]` on the column pair where `push_rotation` applies
    `[[c, s], [-s, c]]`, and `B <- G_L^T B` on the left means
    `U_B <- U_B G_L`, so the left stream takes the same negation as the
    right one.
    """
    var count = len(d)
    if count <= 1:
        return
    e[count - 1] = Scalar[dtype](0)

    var eps = _eps[dtype]()
    var tol = _bdsqr_tol[dtype]()
    var zero = Scalar[dtype](0)
    var one = Scalar[dtype](1)

    # A lower bound on the smallest singular value, `dbdsqr`'s `sminoa`,
    # and the absolute floor it puts under the deflation test.
    var sminoa = abs(d[0])
    if sminoa != 0:
        var mu = sminoa
        for i in range(1, count):
            mu = abs(d[i]) * (mu / (mu + abs(e[i - 1])))
            sminoa = min(sminoa, mu)
            if sminoa == 0:
                break
    sminoa = sminoa / _sqrt(Scalar[dtype](count))
    var thresh = max(
        tol * sminoa,
        Scalar[dtype](_MAX_BDSQR_ITER_PER_N * count * count) * _tiny[dtype](),
    )

    var m = count - 1
    var spent = 0
    var budget = _MAX_BDSQR_ITER_PER_N * count * count

    while m > 0:
        if spent > budget:
            raise Error(
                "svd: the bidiagonal QR iteration did not converge in ",
                budget,
                " steps; the matrix likely holds a NaN or an infinity",
            )

        # The active block `ll .. m`: scan up from `m` to the first
        # negligible off-diagonal, collecting the block's scale on the way.
        var smax = abs(d[m])
        var ll = 0
        var split = False
        var probe = m - 1
        while probe >= 0:
            var abse = abs(e[probe])
            if abse <= thresh:
                e[probe] = zero
                ll = probe
                split = True
                break
            smax = max(smax, max(abs(d[probe]), abse))
            probe -= 1
        if split:
            if ll == m - 1:
                # The bottom singular value has converged.
                m -= 1
                continue
            ll += 1

        # Convergence at the bottom of the block, absolute and relative.
        if abs(e[m - 1]) <= tol * abs(d[m]):
            e[m - 1] = zero
            continue
        var mu = abs(d[ll])
        var sminl = mu
        var deflated = False
        for i in range(ll, m):
            if abs(e[i]) <= tol * mu:
                e[i] = zero
                deflated = True
                break
            mu = abs(d[i + 1]) * (mu / (mu + abs(e[i])))
            sminl = min(sminl, mu)
        if deflated:
            continue

        # The shift, and the two tests that say to drop it. `d[ll] == 0`
        # is covered by `sminl`, which starts there, but the shifted chase
        # divides by `d[ll]` so the guard is written out rather than
        # inferred.
        var shift = zero
        if (
            Scalar[dtype](count) * tol * (sminl / smax)
            > max(eps, Scalar[dtype](0.01) * tol)
            and d[ll] != 0
        ):
            var pair = _las2(d[m - 1], e[m - 1], d[m])
            shift = pair[0]
            var sll = abs(d[ll])
            if sll > 0:
                var ratio = shift / sll
                if ratio * ratio < eps:
                    shift = zero
        spent += m - ll

        if shift == 0:
            # Demmel and Kahan's zero-shift sweep, which computes the
            # small singular values to high relative accuracy because no
            # subtraction of nearly equal quantities happens in it.
            var cs = one
            var oldcs = one
            var oldsn = zero
            for i in range(ll, m):
                var right = _lartg(d[i] * cs, e[i])
                cs = right[0]
                var sn = right[1]
                var r = right[2]
                if i > ll:
                    e[i - 1] = oldsn * r
                var left = _lartg(oldcs * r, d[i + 1] * sn)
                oldcs = left[0]
                oldsn = left[1]
                d[i] = left[2]
                comptime if vectors:
                    vacc.push_rotation(i, cs, -sn)
                    uacc.push_rotation(i, oldcs, -oldsn)
            var h = d[m] * cs
            d[m] = h * oldcs
            e[m - 1] = h * oldsn
        else:
            # The shifted sweep. `f` is the first entry of the first
            # column of `B^T B - shift^2 I`, written so the difference of
            # squares never forms.
            var f = (abs(d[ll]) - shift) * (
                _copysign(one, d[ll]) + shift / d[ll]
            )
            var g = e[ll]
            for i in range(ll, m):
                var right = _lartg(f, g)
                var cosr = right[0]
                var sinr = right[1]
                if i > ll:
                    e[i - 1] = right[2]
                f = cosr * d[i] + sinr * e[i]
                e[i] = cosr * e[i] - sinr * d[i]
                g = sinr * d[i + 1]
                d[i + 1] = cosr * d[i + 1]
                var left = _lartg(f, g)
                var cosl = left[0]
                var sinl = left[1]
                d[i] = left[2]
                f = cosl * e[i] + sinl * d[i + 1]
                d[i + 1] = cosl * d[i + 1] - sinl * e[i]
                if i < m - 1:
                    g = sinl * e[i + 1]
                    e[i + 1] = cosl * e[i + 1]
                comptime if vectors:
                    vacc.push_rotation(i, cosr, -sinr)
                    uacc.push_rotation(i, cosl, -sinl)
            e[m - 1] = f

        uacc.end_sweep(ctx)
        vacc.end_sweep(ctx)


def _top_n_descending[
    dtype: DType
](values: List[Scalar[dtype]], n: Int) -> List[Int]:
    """Indices of the `n` largest entries of `values`, descending.

    `svd` and `svdvals` hand it the `n` magnitudes `_bdsqr` left, so for
    them it is a descending sort. The oracle test hands it `_golub_kahan`'s
    `2n` eigenvalues, where taking the largest `n` is taking the positive
    half of a symmetric spectrum -- the case the name was written for."""
    var order = List[Int](capacity=len(values))
    for i in range(len(values)):
        order.append(i)
    for i in range(1, len(values)):
        var j = i
        while j > 0 and values[order[j]] > values[order[j - 1]]:
            var tmp = order[j]
            order[j] = order[j - 1]
            order[j - 1] = tmp
            j -= 1
    var top = List[Int](capacity=n)
    for i in range(n):
        top.append(order[i])
    return top^


def svdvals[
    T: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: T) raises -> Static[T.dtype, dim[T, 1]] where (
    (
        T.dtype.is_floating_point()
        and dim[T, 0] >= dim[T, 1]
        and dim[T, 1] >= 1
        and block >= 1
    )
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
):
    """**Tier 2.** The singular values of an `m x n` matrix, `m >= n`,
    descending. `scipy.linalg.svdvals`.

    **`gpu=True`** runs the reduction and every `O(n^3)` product on the
    device; `sytrd` says what stays on the host and why.

    From `n = 32` the reduction to bidiagonal form is **two-stage**
    (`numax/linalg/_twostage.mojo`): alternating QR and LQ panels take `a`
    to an upper band of width 16 with every trailing update a product on
    the device, then LAPACK's `dgbbrd` chases the band to bidiagonal form
    on the host. That removes the two whole-matrix products per column the
    one-stage `gebrd` pays, and measured on an M3 Pro at `n = 1024`,
    `float32`, it took the call from about 1,460 ms to about 150. Below
    `n = 32`, `gebrd` reduces device-resident and blocked. Either way
    `_bdsqr` then runs the implicit-shift QR iteration on the two
    diagonals -- `O(n^2)` on the host, no vectors pushed, so the two
    rotation batches it is handed are one-element allocations. Descending,
    as SciPy returns them; the `Array` tier's `svdvals` is one-sided
    Jacobi and comes back unsorted, and a test pins the two as multisets.

    `block` is `gebrd`'s `labrd` panel width below `n = 32`; the two-stage
    path's band width is fixed at the measured best.

    Parameters:
        T: The `TensorLike` type of `a`; an `m x n` matrix, `m >= n >= 1`.
        gpu: Run the reduction and every `O(n^3)` product on the device of `a`.
        block: The `gebrd` panel width.

    Args:
        a: The `m x n` matrix, `m >= n`.

    Returns:
        The `n` singular values, descending.

    Raises:
        If the bidiagonal QR iteration exceeds its `6 n^2` step budget,
        which means `a` holds a NaN or an infinity, or if a device operation
        fails.
    """
    comptime m = dim[T, 0]
    comptime n = dim[T, 1]
    return Static[T.dtype, n](
        _svdvals_run[gpu=gpu](a, m, n, block), a.context()
    )


def _svdvals_run[
    T: TensorLike, gpu: Bool
](a: T, m: Int, n: Int, block: Int) raises -> List[Scalar[T.dtype]] where (
    T.dtype.is_floating_point() and T.LayoutType.rank == 2
):
    """`svdvals`'s body at extents known only at run time, the one both
    overloads run: the reduction (`gebrd` below `_TWO_STAGE_SVDVALS`
    columns, the two stages from it), the bidiagonal QR iteration and the
    descending sort of the magnitudes."""
    var ctx = a.context()
    var d: List[Scalar[T.dtype]]
    var e: List[Scalar[T.dtype]]
    if n >= _TWO_STAGE_SVDVALS:
        var dense = _same_order(a, row_major(_dyn_shape[2](m, n)))
        var bd = _two_stage_bidiagonal[T.dtype, gpu](
            rebind_var[Dynamic[T.dtype, 2]](dense^), m, n, _TWO_STAGE_BAND
        )
        d = bd[0].copy()
        e = bd[1].copy()
    else:
        var left = zeros_dyn[T.dtype, 2](m, n, ctx=ctx)
        var right = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
        var taus_left = zeros_dyn[T.dtype, 1](n, ctx=ctx)
        var taus_right = zeros_dyn[T.dtype, 1](n, ctx=ctx)
        var dd = zeros_dyn[T.dtype, 1](n, ctx=ctx)
        var de = zeros_dyn[T.dtype, 1](n, ctx=ctx)
        _gebrd_run[gpu=gpu](
            a, m, n, block, left, right, taus_left, taus_right, dd, de
        )
        d = dd.to_host()
        e = de.to_host()
    var uacc = _RotationBatch[T.dtype, gpu, False, 1, True](n, block, ctx)
    var vacc = _RotationBatch[T.dtype, gpu, False, 1, True](n, block, ctx)
    _bdsqr[gpu=gpu, vectors=False](d, e, uacc, vacc, ctx)
    uacc.finish(ctx)
    vacc.finish(ctx)

    # `_bdsqr` leaves the values unsorted and of either sign; a singular
    # value is the magnitude.
    var mags = List[Scalar[T.dtype]](capacity=n)
    for i in range(n):
        mags.append(abs(d[i]))
    var order = _top_n_descending(mags, n)
    var out = List[Scalar[T.dtype]](capacity=n)
    for i in range(n):
        out.append(mags[order[i]])
    return out^


struct SVD[dtype: DType, m: Int, n: Int](
    Movable where dtype.is_floating_point() and m >= n and n >= 1
):
    """`svd`'s result: `A = U diag(s) V^T`, singular values descending.

    A struct rather than SciPy's `(U, s, Vh)` tuple, for the reason
    `qr_factor` returns a `QR`. The fields follow the `Array` tier's
    `svd` -- `v` holds the right singular vectors as *columns*, so SciPy's
    `Vh` is `transpose(v)`, and `A == U @ diag(s) @ V^T` reads the way the
    reconstruction is written.
    """

    var u: Static[Self.dtype, Self.m, Self.n]
    """The left singular vectors as columns; `m x n`, the thin form."""

    var s: Static[Self.dtype, Self.n]
    """The singular values, descending."""

    var v: Static[Self.dtype, Self.n, Self.n]
    """The right singular vectors as columns."""

    def __init__(
        out self,
        var u: Static[Self.dtype, Self.m, Self.n],
        var s: Static[Self.dtype, Self.n],
        var v: Static[Self.dtype, Self.n, Self.n],
    ):
        self.u = u^
        self.s = s^
        self.v = v^


struct DynamicSVD[dtype: DType](Movable where dtype.is_floating_point()):
    """`svd`'s result at run-time extents: `SVD` with `m` and `n` in the
    values rather than the type, as `DynamicLU` is to `LU`."""

    var u: Dynamic[Self.dtype, 2]
    """The left singular vectors as columns; `m x n`, the thin form."""

    var s: Dynamic[Self.dtype, 1]
    """The singular values, descending."""

    var v: Dynamic[Self.dtype, 2]
    """The right singular vectors as columns; `n x n`."""

    def __init__(
        out self,
        var u: Dynamic[Self.dtype, 2],
        var s: Dynamic[Self.dtype, 1],
        var v: Dynamic[Self.dtype, 2],
    ):
        self.u = u^
        self.s = s^
        self.v = v^


def svd[
    T: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: T) raises -> SVD[T.dtype, dim[T, 0], dim[T, 1]] where (
    (
        T.dtype.is_floating_point()
        and dim[T, 0] >= dim[T, 1]
        and dim[T, 1] >= 1
        and block >= 1
    )
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
):
    """**Tier 2.** The thin singular value decomposition of an `m x n`
    matrix, `m >= n`: `A = U diag(s) V^T` with `s` descending.
    `scipy.linalg.svd(a, full_matrices=False)`.

    **`gpu=True`** runs the reduction and every `O(n^3)` product on the
    device; `sytrd` says what stays on the host and why.

    Three steps, and the vectors never touch the host in any of them.
    `gebrd` reduces `A` to bidiagonal `B = Q^T A P` device-resident.
    `_bdsqr` diagonalizes `B` by the implicit-shift QR iteration --
    `O(n^2)` of band arithmetic on the host, its two rotation streams
    pushed into two batches that hold `U_B^T` and `V_B^T` device-resident
    and send each window of commuting rotations out as one `matmul`. Then
    the descending order and the sign of each value are applied as one
    `elementwise` gather over `(n, n)`, and `U = Q U_B`, `V = P V_B` come
    back through `inner` under `transpose_b=True` because the batches hold
    both factors transposed. Only the `n` values and the order they sort
    into cross to the host.

    Two knobs share the name `block` and they mean different things, as
    `eigh`'s do. `gebrd` takes it as the **panel width** -- its own
    `labrd` panel, and the one `.q()` and `.p()` form `Q` and `P` in. The
    sweep takes it as the **rotation window**: `block` consecutive sweeps
    batch together and a window spans at most `2 * block` columns.
    `block == 1` recovers both unblocked algorithms exactly, and a test
    pins every width against every other.

    `m >= n` is a constraint, not a convention: there is no transposing
    route here, so a wide matrix is transposed by the caller and the
    factors swapped.

    `ponytail:` the band sweep is no longer the ceiling -- the reduction
    is. At `n = 1024`, `float32`, on an M3 Pro, `svd` is 1,768 ms at the
    default `block` and `svdvals` 1,648, so everything the vectors cost is
    around a tenth of the call and `gebrd` is the rest. Dropping the
    doubling took `svd` from 1,949 ms; it did not move `svdvals`, which
    pushes no rotation either way. What is left of the sweep is the
    sequential `O(n^2)` chase, which divide and conquer (`dbdsdc`)
    replaces rather than reschedules, and what is left of the reduction is
    the pair of matrix-vector products `gebrd`'s docstring names.

    Rectangular, unlike the `Array` tier's square-only one-sided Jacobi,
    and descending where that one is unsorted; both docstrings say so.

    Parameters:
        T: The `TensorLike` type of `a`; an `m x n` matrix, `m >= n >= 1`.
        gpu: Run the reduction and every `O(n^3)` product on the device of `a`.
        block: The `gebrd` panel width, the rotation window and the width
            `U` and `V` are formed in.

    Args:
        a: The `m x n` matrix, `m >= n`.

    Returns:
        An `SVD` holding the `m x n` thin `u`, the `n` singular values `s`
        descending and the `n x n` `v`, with `a == u diag(s) v^T`.

    Raises:
        If the bidiagonal QR iteration exceeds its `6 n^2` step budget,
        which means `a` holds a NaN or an infinity, or if a device operation
        fails.
    """
    comptime m = dim[T, 0]
    comptime n = dim[T, 1]
    var ctx = a.context()
    var u = zeros_dyn[T.dtype, 2](m, n, ctx=ctx)
    var s = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var v = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    _svd_run[gpu=gpu](a, m, n, block, u, s, v)
    return SVD[T.dtype, m, n](
        u^.as_static[m, n](), s^.as_static[n](), v^.as_static[n, n]()
    )


def _svd_run[
    T: TensorLike, gpu: Bool
](
    a: T,
    m: Int,
    n: Int,
    block: Int,
    mut u: Dynamic[T.dtype, 2],
    mut s: Dynamic[T.dtype, 1],
    mut v: Dynamic[T.dtype, 2],
) raises where (T.dtype.is_floating_point() and T.LayoutType.rank == 2):
    """`svd`'s body at extents known only at run time, the one both
    overloads run: `gebrd`, the bidiagonal QR iteration with both rotation
    streams batched into device GEMMs, the descending sort and sign fix as
    one gather, and `U = Q U_B`, `V = P V_B` as two products. Writes the
    caller's `u` (`m x n`), `s` (`n`) and `v` (`n x n`)."""
    var ctx = a.context()
    var left = zeros_dyn[T.dtype, 2](m, n, ctx=ctx)
    var right = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    var taus_left = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var taus_right = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var diag = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var off = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    _gebrd_run[gpu=gpu](
        a, m, n, block, left, right, taus_left, taus_right, diag, off
    )
    var d = diag.to_host()
    var e = off.to_host()
    var uacc = _RotationBatch[T.dtype, gpu, True, 1, True](n, block, ctx)
    var vacc = _RotationBatch[T.dtype, gpu, True, 1, True](n, block, ctx)
    _bdsqr[gpu=gpu, vectors=True](d, e, uacc, vacc, ctx)
    uacc.finish(ctx)
    vacc.finish(ctx)

    var mags = List[Scalar[T.dtype]](capacity=n)
    for i in range(n):
        mags.append(abs(d[i]))
    var order = _top_n_descending(mags, n)

    var s_host = List[Scalar[T.dtype]](capacity=n)
    var row_host = List[Scalar[DType.int64]](capacity=n)
    var sign_host = List[Scalar[T.dtype]](capacity=n)
    for j in range(n):
        var at = order[j]
        s_host.append(mags[at])
        row_host.append(Scalar[DType.int64](at))
        # A negative value is made positive by flipping its right singular
        # vector, which is `dbdsqr`'s own sign fix.
        sign_host.append(
            Scalar[T.dtype](-1) if d[at] < 0 else Scalar[T.dtype](1)
        )
    var rows = Dynamic[DType.int64, 1](
        row_major(_dyn_shape[1](n)), row_host^, ctx
    )
    var signs = Dynamic[T.dtype, 1](
        row_major(_dyn_shape[1](n)), sign_host^, ctx
    )

    # The gather is `eigh`'s: source and destination are both `(n, n)` and
    # only the index tensors are rank 1, which is the cross-shape case
    # `findings.mdc` records as sound.
    var ub_t = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    var vb_t = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    var ut = uacc.zt.tile()
    var vt = vacc.zt.tile()
    var ub = ub_t.tile()
    var vb = vb_t.tile()
    var rv = rows.tile()
    var sv = signs.tile()

    @always_inline
    def gather[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var ut, var vt, var ub, var vb, var rv, var sv}:
        var at = coord_to_index_list(coord)
        var row = Int(rv[Coord(at[0])])
        ub.store[1](coord, ut[Coord(row, at[1])])
        vb.store[1](coord, vt[Coord(row, at[1])] * sv[Coord(at[0])])

    elementwise[simd_width=1, target=_target[gpu]()](gather, Coord(n, n), ctx)
    ctx.synchronize()
    # The batches and the two index tensors are named nowhere past the
    # `.tile()` a view erases the origin of; see `findings.mdc` on the
    # queued free.
    _ = uacc^
    _ = vacc^
    _ = rows^
    _ = signs^

    var q = _orgbr_q_run[gpu=gpu](left, taus_left, m, n, block)
    var p = _orgbr_p_run[gpu=gpu](right, taus_right, n, block)
    _max_matmul[transpose_b=True, target=_target[gpu]()](
        u.tile(), q.tile(), ub_t.tile(), ctx
    )
    _max_matmul[transpose_b=True, target=_target[gpu]()](
        v.tile(), p.tile(), vb_t.tile(), ctx
    )
    ctx.synchronize()
    s.copy_from_host(s_host)
    _ = q^
    _ = p^
    _ = ub_t^
    _ = vb_t^


def _runtime_tall[
    T: TensorLike
](a: T, name: StaticString) raises -> Tuple[Int, Int] where (
    T.LayoutType.rank == 2
):
    """The extents of the run-time-shaped `a`, raising unless `m >= n >= 1`,
    the shape the static overloads require in their `where` clause."""
    var m = a.dim_at(0)
    var n = a.dim_at(1)
    if m < n:
        raise Error(
            name,
            ": the matrix must have at least as many rows as columns, got ",
            m,
            " x ",
            n,
            "; take the transpose's",
        )
    if n < 1:
        raise Error(name, ": the matrix is empty")
    return (m, n)


def svdvals[
    T: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: T) raises -> Dynamic[T.dtype, 1] where (
    (T.dtype.is_floating_point() and block >= 1)
    and T.LayoutType.rank == 2
    and not T.LayoutType.all_dims_known
):
    """**Tier 2.** The run-time-shape overload: the singular values of an
    `m x n` matrix, `m >= n`, whose extents are data, descending.
    `scipy.linalg.svdvals`.

    The static overload's body at the extents `a` carries -- `gebrd` below
    32 columns and the two-stage band from there, then the bidiagonal QR
    iteration -- so one compiled program takes any shape and the two
    overloads agree to the bit.

    Parameters:
        T: The `TensorLike` type of `a`, a rank-2 run-time-shaped matrix.
        gpu: Run the reduction and every `O(n^3)` product on `a`'s device.
        block: The `gebrd` panel width.

    Args:
        a: The `m x n` matrix, `m >= n`.

    Returns:
        A run-time-shaped vector of the `n` singular values, descending.

    Raises:
        If `a` has fewer rows than columns or is empty, if the bidiagonal
        QR iteration exceeds its `6 n^2` step budget (`a` holds a NaN or an
        infinity), or if a device operation fails.
    """
    var shape = _runtime_tall(a, "svdvals")
    var n = shape[1]
    return Dynamic[T.dtype, 1](
        row_major(_dyn_shape[1](n)),
        _svdvals_run[gpu=gpu](a, shape[0], n, block),
        a.context(),
    )


def svd[
    T: TensorLike,
    gpu: Bool = False,
    block: Int = 32,
](a: T) raises -> DynamicSVD[T.dtype] where (
    (T.dtype.is_floating_point() and block >= 1)
    and T.LayoutType.rank == 2
    and not T.LayoutType.all_dims_known
):
    """**Tier 2.** The run-time-shape overload: the thin SVD
    `A = U diag(s) V^T` of an `m x n` matrix, `m >= n`, whose extents are
    data. `scipy.linalg.svd(full_matrices=False)`.

    The static overload's body at the extents `a` carries, returning a
    `DynamicSVD`, so one compiled program takes any shape and the two
    overloads agree to the bit.

    Parameters:
        T: The `TensorLike` type of `a`, a rank-2 run-time-shaped matrix.
        gpu: Run the reduction and every `O(n^3)` product on `a`'s device.
        block: The `gebrd` panel width, the rotation window and the width
            `Q` and `P` are formed in.

    Args:
        a: The `m x n` matrix, `m >= n`.

    Returns:
        A `DynamicSVD` with the `m x n` left vectors `u`, the singular
        values `s` descending and the `n x n` right vectors `v` as columns.

    Raises:
        If `a` has fewer rows than columns or is empty, if the bidiagonal
        QR iteration exceeds its `6 n^2` step budget (`a` holds a NaN or an
        infinity), or if a device operation fails.
    """
    var shape = _runtime_tall(a, "svd")
    var m = shape[0]
    var n = shape[1]
    var ctx = a.context()
    var u = zeros_dyn[T.dtype, 2](m, n, ctx=ctx)
    var s = zeros_dyn[T.dtype, 1](n, ctx=ctx)
    var v = zeros_dyn[T.dtype, 2](n, n, ctx=ctx)
    _svd_run[gpu=gpu](a, m, n, block, u, s, v)
    return DynamicSVD[T.dtype](u^, s^, v^)


def matrix_rank[
    T: TensorLike,
    gpu: Bool = False,
](a: T, tol: Optional[Float64] = None) raises -> Int where (
    (T.dtype.is_floating_point() and dim[T, 0] >= dim[T, 1] and dim[T, 1] >= 1)
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
):
    """**Tier 2.** How many singular values exceed `tol`.
    `numpy.linalg.matrix_rank`.

    `tol` defaults to NumPy's: the largest singular value times
    `max(m, n)` times the machine epsilon of `T.dtype`, which is the noise
    floor an SVD of that size can be expected to carry. Pass an explicit
    `tol` to ask a different question -- "how many directions carry more
    than one part in a thousand" is `tol = 1e-3 * s_max`.

    An `Int`, where the `Array` tier returns `T`: one `Tensor` is one
    matrix, so there is one rank, and nothing here needs to stay branchless.

    Parameters:
        T: The `TensorLike` type of `a`; an `m x n` matrix, `m >= n >= 1`.
        gpu: Run the `svdvals` call on the device of `a`.

    Args:
        a: The `m x n` matrix, `m >= n`.
        tol: The singular-value threshold; `None` uses `s_max * max(m, n)`
            times the machine epsilon of `T.dtype`.

    Returns:
        The number of singular values strictly greater than the threshold.

    Raises:
        If `svdvals` fails to converge or a device operation fails.
    """
    comptime m = dim[T, 0]
    comptime n = dim[T, 1]
    var s = svdvals[gpu=gpu](a).to_host()
    var eps: Float64
    comptime if T.dtype == DType.float32:
        eps = 1.1920929e-07
    else:
        eps = 2.220446049250313e-16
    var threshold = (
        tol.value() if tol else Float64(s[0]) * Float64(max(m, n)) * eps
    )
    var rank = 0
    for i in range(n):
        if Float64(s[i]) > threshold:
            rank += 1
    return rank


def eigh[
    T: FloatLike, n: Int, sweeps: Int = 12
](a: Array[T, n * n]) -> Tuple[Array[T, n], Array[T, n * n]]:
    """The `Array`-tier overload: one problem in registers, generic over
    the `FloatLike` conformer. The algorithm and its bound are documented
    at `numax.linalg._array.eigen.eigh`.

    Parameters:
        T: The `FloatLike` conformer of the entries.
        n: The matrix order.
        sweeps: The fixed number of Jacobi sweeps.

    Args:
        a: The `n x n` matrix, row-major.

    Returns:
        The `(eigenvalues, eigenvectors)` pair, eigenvalues unordered and
        eigenvectors as the columns of the `n x n` matrix.
    """
    return _array_eigh[T=T, n=n, sweeps=sweeps](a)


def eigvals[
    T: FloatLike, n: Int, sweeps: Int = 100, tol: Float64 = 1e-8
](a: Array[T, n * n]) -> Array[Complex[T], n]:
    """The `Array`-tier overload: one problem in registers, generic over
    the `FloatLike` conformer. The algorithm and its bound are documented
    at `numax.linalg._array.eigen.eigvals`.

    Parameters:
        T: The `FloatLike` conformer of the entries.
        n: The matrix order.
        sweeps: The fixed number of shifted QR steps.
        tol: The relative threshold below which a subdiagonal entry counts
            as converged to zero rather than part of a `2 x 2` block.

    Args:
        a: The `n x n` matrix, row-major.

    Returns:
        The `n` eigenvalues as `Complex[T]`.
    """
    return _array_eigvals[T=T, n=n, sweeps=sweeps, tol=tol](a)


def eigvalsh[
    T: FloatLike, n: Int, sweeps: Int = 12
](a: Array[T, n * n]) -> Array[T, n]:
    """The `Array`-tier overload: one problem in registers, generic over
    the `FloatLike` conformer. The algorithm and its bound are documented
    at `numax.linalg._array.eigen.eigvalsh`.

    Parameters:
        T: The `FloatLike` conformer of the entries.
        n: The matrix order.
        sweeps: The fixed number of Jacobi sweeps.

    Args:
        a: The `n x n` symmetric matrix, row-major.

    Returns:
        The `n` eigenvalues, unordered.
    """
    return _array_eigvalsh[T=T, n=n, sweeps=sweeps](a)


def hessenberg[T: FloatLike, n: Int](a: Array[T, n * n]) -> Array[T, n * n]:
    """The `Array`-tier overload: one problem in registers, generic over
    the `FloatLike` conformer. The algorithm and its bound are documented
    at `numax.linalg._array.eigen.hessenberg`.

    Parameters:
        T: The `FloatLike` conformer of the entries.
        n: The matrix order.

    Args:
        a: The `n x n` matrix, row-major.

    Returns:
        The upper Hessenberg matrix `H = Q^T A Q`, row-major, without `Q`.
    """
    return _array_hessenberg[T=T, n=n](a)


def matrix_rank[
    T: FloatLike, n: Int, sweeps: Int = 12, rcond: Float64 = 1e-12
](a: Array[T, n * n]) -> T:
    """The `Array`-tier overload: one problem in registers, generic over
    the `FloatLike` conformer. The algorithm and its bound are documented
    at `numax.linalg._array.eigen.matrix_rank`.

    Parameters:
        T: The `FloatLike` conformer of the entries.
        n: The matrix order.
        sweeps: The fixed number of Jacobi sweeps.
        rcond: The threshold relative to the largest singular value.

    Args:
        a: The `n x n` matrix, row-major.

    Returns:
        The count of singular values above `rcond * s_max`, per lane, as
        a `T`.
    """
    return _array_matrix_rank[T=T, n=n, sweeps=sweeps, rcond=rcond](a)


def svd[
    T: FloatLike, n: Int, sweeps: Int = 12
](a: Array[T, n * n]) -> Tuple[Array[T, n * n], Array[T, n], Array[T, n * n]]:
    """The `Array`-tier overload: one problem in registers, generic over
    the `FloatLike` conformer. The algorithm and its bound are documented
    at `numax.linalg._array.eigen.svd`.

    Parameters:
        T: The `FloatLike` conformer of the entries.
        n: The matrix order.
        sweeps: The fixed number of Jacobi sweeps.

    Args:
        a: The `n x n` matrix, row-major.

    Returns:
        The `(U, s, V)` triple with `a == U diag(s) V^T`, `s` unordered and
        non-negative.
    """
    return _array_svd[T=T, n=n, sweeps=sweeps](a)


def svdvals[
    T: FloatLike, n: Int, sweeps: Int = 12
](a: Array[T, n * n]) -> Array[T, n]:
    """The `Array`-tier overload: one problem in registers, generic over
    the `FloatLike` conformer. The algorithm and its bound are documented
    at `numax.linalg._array.eigen.svdvals`.

    Parameters:
        T: The `FloatLike` conformer of the entries.
        n: The matrix order.
        sweeps: The fixed number of Jacobi sweeps.

    Args:
        a: The `n x n` matrix, row-major.

    Returns:
        The `n` singular values, unordered and non-negative.
    """
    return _array_svdvals[T=T, n=n, sweeps=sweeps](a)
