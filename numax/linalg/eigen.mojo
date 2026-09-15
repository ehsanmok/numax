"""Spectral factorizations over `Tensor`: `scipy.linalg`'s `_decomp`.

**This module is tier 2.** Every routine here is two phases: a reduction
to a condensed form, which is Householder work whose cubic term goes to
`linalg.matmul`, then a sweep on the condensed band, which loops to a
tolerance and deflates on a test of the data. The band sweep itself has no
GEMM to hand anything to -- that is a property of the algorithm, not of
this implementation -- so it runs on the host over `O(n)` numbers and each
function says so where it happens.

The *rotations* that sweep produces are a different matter, and for `eigh`
and `svd` they are no longer host work: `_RotationBatch` holds a
`block`-sweep batch of them, reorders it into windows of mutually
commuting rotations and applies each window as one `matmul`. `svd` shares
that machinery through the same sweep, runs it at `2n` -- the Golub-Kahan
doubling, which stays until `bdsqr` -- and de-interleaves `U` and `V` out
of the result in one `elementwise` on the device. `schur`'s Francis sweep
still accumulates its rotations one at a time on the host.

MAX ships nothing to delegate to here. There is no eigensolver, no SVD, no
`sytrd`, no Jacobi or Givens helper anywhere in `linalg`, `nn`,
`algorithm` or `layout`; the one decomposition in the tree,
`linalg.qr_factorization`, is `LayoutTensor`-only and a CPU scalar loop,
denied twice over. So everything here is an **extend**, written in MAX's
idiom.

`numax.linalg.array.eigen` is the `FloatLike`-generic sibling, one import
away: cyclic Jacobi at a fixed sweep count, square matrices small enough to
live in registers, differentiable and GPU-launchable in a way this tier
cannot be. This tier is for the sizes that one cannot reach.
"""

from layout import Coord, TileTensor, coord_to_index_list
from layout.tile_layout import row_major
from linalg.matmul import matmul as _max_matmul
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.builtin.sort import sort as _std_sort
from std.math import copysign as _copysign, hypot as _hypot, sqrt as _sqrt
from std.sys.info import align_of, simd_width_of
from std.utils import IndexList

from ..core.array import Dynamic, Static, zeros, zeros_dyn
from .blas import _target, dot, inner, matmul, matvec
from .common import _Dense, _device_identity
from .panel import (
    _PANEL_THREADS,
    gebd2_col,
    gebd2_row,
    pack_block,
    sytd2_column,
    sytd2_rank_two,
)
from .qr import _apply_block_reflector, _ReflectorWork


struct TensorTridiagonal[dtype: DType, n: Int, gpu: Bool = False](
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
        products. That is `_accumulate_reflectors`, and it is `TensorQR`'s
        `orgqr` walk over the same kernels.

        Still `O(n^3)`, but the cubic term is `linalg.matmul`'s rather
        than `n` matrix-vector launches. A caller who only wants
        eigenvalues should not call this; `eigvalsh` does not.
        """
        return _accumulate_reflectors[Self.dtype, Self.n, Self.gpu](
            self.reflectors, self.taus, Self.n - 2, self.block
        )


def _row_combination[
    dtype: DType, n: Int, gpu: Bool = False
](mut q: Static[dtype, n, n], mut v: Static[dtype, n]) raises -> Static[
    dtype, n
]:
    """`Q^T v`, as the matrix-vector product `linalg.matmul` reaches
    through its own `transpose_b`-free path: `v^T Q` read as a row."""
    var ctx = q.context()
    var result = zeros[dtype, n](ctx)
    var row: _Dense[dtype] = TileTensor(
        v.view().ptr_at_offset(Coord(0)), row_major(Coord(1, n))
    )
    var out: _Dense[dtype] = TileTensor(
        result.view().ptr_at_offset(Coord(0)), row_major(Coord(1, n))
    )
    _max_matmul[target=_target[gpu]()](out, row, q.view(), ctx)
    ctx.synchronize()
    return result^


def _rank_one_subtract[
    dtype: DType, n: Int, gpu: Bool = False
](
    mut q: Static[dtype, n, n],
    mut v: Static[dtype, n],
    mut y: Static[dtype, n],
    scale: Scalar[dtype],
    mut product: Static[dtype, n, n],
    ctx: DeviceContext,
) raises:
    """`Q -= scale * v y^T`, through `matmul`'s epilogue so the scattered
    store is the only pass over `Q`."""
    var target = q.view()

    @parameter
    @always_inline
    @__copy_capture(target, scale)
    def subtract[
        _dtype: DType,
        lanes: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, lanes]](),
    ](idx: IndexList[2], value: SIMD[_dtype, lanes]) capturing -> None:
        var at = Coord(idx[0], idx[1])
        target.store[lanes](
            at,
            target.load[lanes](at) - scale * rebind[SIMD[dtype, lanes]](value),
        )

    var col: _Dense[dtype] = TileTensor(
        v.view().ptr_at_offset(Coord(0)), row_major(Coord(n, 1))
    )
    var row: _Dense[dtype] = TileTensor(
        y.view().ptr_at_offset(Coord(0)), row_major(Coord(1, n))
    )
    _max_matmul[elementwise_lambda_fn=subtract, target=_target[gpu]()](
        product.view(), col, row, ctx
    )
    ctx.synchronize()


def sytrd[
    dtype: DType, n: Int, gpu: Bool = False, block: Int = 32
](mut a: Static[dtype, n, n]) raises -> TensorTridiagonal[dtype, n, gpu] where (
    dtype.is_floating_point() and block >= 1
):
    """**Tier 2.** Reduce a symmetric `a` to tridiagonal form by
    Householder reflections, device-resident. LAPACK's `sytrd`.

    `a` is read as symmetric and is **not checked** -- checking costs a
    full pass and the reduction is meaningless on a matrix that is not,
    in a way the caller is better placed to notice. Only the lower triangle
    and the diagonal are read.

    This is the half of an eigendecomposition that has a GEMM in it, and
    it is over half the arithmetic. Each column is four launches and no
    host round trip for the matrix: a single-block kernel forms the
    reflector, `matvec` multiplies the *whole* matrix by the padded
    reflector, `dot` reduces one scalar, and one `matmul` applies the
    symmetric rank-two update `a -= v w^T + w v^T` as a single product
    `[v | w] @ [w | v]^T` with `transpose_b=True`. That last identity is
    what lets numax skip the `syr2k` MAX does not ship.

    `ponytail:` this is the unblocked reduction, and it has two ceilings,
    both closed by the same upgrade. It multiplies the **whole** matrix by
    each reflector rather than the shrinking trailing block, because a
    strided sub-block cannot be handed to `matmul` without staging it dense
    and staging it per column would be `O(n^3)` of memcpy -- so it does
    about `3n^3` flops against LAPACK's `4n^3/3`. And its products are
    matrix-vector shaped, which is bandwidth-bound rather than
    GEMM-bound. The upgrade is LAPACK's blocked `latrd`: accumulate `V` and
    `W` over a panel of `block` columns and apply
    `A -= [V | W] @ [W | V]^T` once per panel with `K = 2 * block`, which is
    the same single-GEMM identity this already uses at `block == 1`. The
    `dot` per column is also a device-to-host synchronization; a blocked
    panel amortizes those too.

    `block` is that panel width. The reduction itself does not read it
    yet; it is carried on the result so `.q()` forms `Q` in panels of
    `block` reflectors, and `latrd` will take the same number.

    `numax.linalg.array.eigh` is the small-matrix route and needs none of
    this: cyclic Jacobi at a fixed sweep count, differentiable, and
    launchable inside a GPU thread.
    """
    var ctx = a.context()
    var work = zeros[dtype, n, n](ctx)
    var taus = zeros[dtype, n](ctx)
    var vpad = zeros[dtype, n](ctx)
    var scratch = zeros[dtype, _PANEL_THREADS + 1](ctx)
    var left = zeros[dtype, n, 2](ctx)
    var right = zeros[dtype, n, 2](ctx)
    var product = zeros[dtype, n, n](ctx)

    var wv = work.view()
    var tv = taus.view()
    var vv = vpad.view()
    var sv = scratch.view()

    pack_block[target=_target[gpu]()](a.view(), wv, 0, 0, n, n, ctx)

    for k in range(n - 2):
        comptime if gpu:
            ctx.enqueue_function[
                sytd2_column[
                    dtype,
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
                Int32(k),
                Int32(n),
                grid_dim=1,
                block_dim=_PANEL_THREADS,
            )
            ctx.synchronize()
        else:
            sytd2_column(wv, vv, tv, sv, Int32(k), Int32(n))

        var this_tau = taus.to_host()[k]
        if this_tau == 0:
            continue

        # `p = A v` over the whole matrix: `vpad` is zero at and above `k`,
        # so the leading block contributes nothing and needs no staging.
        var p = matvec[gpu=gpu](work, vpad)
        var kappa = dot[gpu=gpu](p, vpad)

        var lv = left.view()
        var rv = right.view()
        comptime if gpu:
            ctx.enqueue_function[
                sytd2_rank_two[
                    dtype,
                    VLayout=type_of(vv).LayoutType,
                    PLayout=type_of(p.view()).LayoutType,
                    LLayout=type_of(lv).LayoutType,
                    gpu=True,
                ]
            ](
                vv,
                p.view(),
                lv,
                rv,
                this_tau,
                kappa,
                Int32(k),
                Int32(n),
                grid_dim=1,
                block_dim=_PANEL_THREADS,
            )
            ctx.synchronize()
        else:
            sytd2_rank_two(
                vv, p.view(), lv, rv, this_tau, kappa, Int32(k), Int32(n)
            )

        _subtract_rank_two[gpu=gpu](work, left, right, product, ctx)

    # `sv` is read by every launch above and `scratch` is named nowhere
    # else, so without this Mojo would destroy it after `.view()`; see
    # `findings.mdc` on the origin-erased view and the queued free.
    _ = scratch^
    var reduced = work.to_host()
    var d = zeros[dtype, n](ctx)
    var e = zeros[dtype, n](ctx)
    var d_host = d.to_host()
    var e_host = e.to_host()
    for i in range(n):
        d_host[i] = reduced[i * n + i]
        if i + 1 < n:
            e_host[i] = reduced[(i + 1) * n + i]
    d.copy_from_host(d_host)
    e.copy_from_host(e_host)

    return TensorTridiagonal[dtype, n, gpu](d^, e^, work^, taus^, block)


def _subtract_rank_two[
    dtype: DType, n: Int, gpu: Bool = False
](
    mut target: Static[dtype, n, n],
    mut left: Static[dtype, n, 2],
    mut right: Static[dtype, n, 2],
    mut product: Static[dtype, n, n],
    ctx: DeviceContext,
) raises:
    """`target -= left @ right^T`, in one GEMM with the subtraction fused
    into the epilogue.

    With `left = [v | w]` and `right = [w | v]` this is the symmetric
    rank-two update `v w^T + w v^T`, which is the whole trailing update of
    a tridiagonal reduction step.
    """
    var into = target.view()

    @parameter
    @always_inline
    @__copy_capture(into)
    def subtract[
        _dtype: DType,
        lanes: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, lanes]](),
    ](idx: IndexList[2], value: SIMD[_dtype, lanes]) capturing -> None:
        var at = Coord(idx[0], idx[1])
        into.store[lanes](
            at, into.load[lanes](at) - rebind[SIMD[dtype, lanes]](value)
        )

    _max_matmul[
        transpose_b=True,
        elementwise_lambda_fn=subtract,
        target=_target[gpu](),
    ](product.view(), left.view(), right.view(), ctx)
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


struct _RotationBatch[dtype: DType, N: Int, gpu: Bool, vectors: Bool](
    Movable where dtype.is_floating_point() and N >= 1
):
    """The Givens rotations of up to `block` consecutive QL sweeps, held
    until enough of them have accumulated to go out as GEMMs.

    **Lang's windowing (1998), which is what makes this a GEMM at all.**
    Two rotations on adjacent index pairs `(i, i+1)` and `(j, j+1)`
    commute when `|i - j| >= 2`. So within a batch of `K = block`
    consecutive sweeps, tag rotation `(sweep s, index i)` with
    `group(s, i) = (i - s + block) // block` and apply the groups in
    *decreasing* order, each group's rotations in stream order. The only
    pairs that reordering swaps are ones whose indices are at least two
    apart: within a sweep `i` descends so the tag never rises, and across
    sweeps `group(s1, i1) < group(s2, i2)` with `s1 < s2` forces
    `i2 - i1 >= 1 + (s2 - s1) >= 2`. A group's indices then span at most
    `block + K = 2 * block` columns, which is the window.

    So one group is a `w x w` orthogonal factor `U` -- the product of its
    rotations -- applied to a `w`-column stripe of `Z`. `Z` is held
    **transposed** as `zt`, because a column stripe of `Z` is a contiguous
    *row* block of `Z^T`, and a row block is what `matmul` will read
    without a stride: `zt[c_lo:c_lo+w, :] <- U^T @ staged`, where `staged`
    is a dense copy of that row block (`matmul` will not alias its result
    with an operand).

    `U^T` is built on the host, packed into the first `w*w` entries of
    `ut_host` so the claimed row stride and the memory agree, by the same
    four lines the unblocked sweep ran on `Z`'s columns -- on `U^T`'s rows
    instead. That is `O(w)` per rotation against `O(N)`, so the host term
    falls from `O(N^3)` to `O(block * N^2)` and the `O(N^3)` lands in
    `linalg.matmul`.

    At `vectors=False` nothing is ever pushed, so every buffer here is a
    one-element allocation and `eigvalsh` and `svdvals` share the sweep
    body without paying for it.
    """

    var zt: Dynamic[Self.dtype, 2]
    """`Z^T`, `N x N` and device-resident. Row `j` is eigenvector `j`."""

    var staged: Dynamic[Self.dtype, 2]
    """A dense `2*block x N` copy of the row block under update."""

    var u_dev: Dynamic[Self.dtype, 2]
    """The uploaded `U^T`, `2*block x 2*block` and read packed."""

    var ut_host: List[Scalar[Self.dtype]]
    """Where `U^T` is built before it is uploaded."""

    var rot_index: List[Int]
    """`i` of each logged rotation, an absolute column index."""

    var rot_sweep: List[Int]
    """Which sweep of the batch each logged rotation came from."""

    var rot_cos: List[Scalar[Self.dtype]]
    var rot_sin: List[Scalar[Self.dtype]]

    var sweep: Int
    """Sweeps closed since the last flush; the batch goes out at `block`."""

    var block: Int
    """`B` and `K` both: the window is `2 * block` columns wide."""

    def __init__(out self, block: Int, ctx: DeviceContext) raises:
        self.block = max(block, 1)
        self.sweep = 0
        var side = Self.N if Self.vectors else 1
        var wide = min(2 * self.block, side)
        self.zt = zeros_dyn[Self.dtype, 2](side, side, ctx=ctx)
        self.staged = zeros_dyn[Self.dtype, 2](wide, side, ctx=ctx)
        self.u_dev = zeros_dyn[Self.dtype, 2](wide, wide, ctx=ctx)
        self.ut_host = List[Scalar[Self.dtype]](length=wide * wide, fill=0)
        self.rot_index = List[Int]()
        self.rot_sweep = List[Int]()
        self.rot_cos = List[Scalar[Self.dtype]]()
        self.rot_sin = List[Scalar[Self.dtype]]()
        comptime if Self.vectors:
            var seed: _Dense[Self.dtype] = TileTensor(
                self.zt.view().ptr_at_offset(Coord(0, 0)),
                row_major(Coord(side, side)),
            )
            _device_identity[Self.dtype, Self.gpu](seed, side, side, ctx)
            ctx.synchronize()

    def push_rotation(
        mut self, i: Int, c: Scalar[Self.dtype], s: Scalar[Self.dtype]
    ):
        """Log the rotation the sweep just applied to `(d, e)` at index `i`.

        `Z <- Z G` with `G = [[c, s], [-s, c]]` on rows `i, i+1`, which is
        the rotation the unblocked body wrote out by hand.
        """
        self.rot_index.append(i)
        self.rot_sweep.append(self.sweep)
        self.rot_cos.append(c)
        self.rot_sin.append(s)

    def end_sweep(mut self, ctx: DeviceContext) raises:
        """Close a sweep, and flush once `block` of them have closed."""
        self.sweep += 1
        if self.sweep >= self.block:
            self._flush(ctx)

    def finish(mut self, ctx: DeviceContext) raises:
        """Flush the partial batch the last sweep left and wait for it."""
        self._flush(ctx)
        ctx.synchronize()

    def _flush(mut self, ctx: DeviceContext) raises:
        self.sweep = 0
        var count = len(self.rot_index)
        if count == 0:
            return
        var b = self.block
        var cols = Self.N

        # Bucket the batch by group, stably, so each group's rotations
        # keep the order the sweeps emitted them in. A linear rescan per
        # group would cost `O(count * N / block)`, which is the term this
        # whole commit is removing.
        var lowest = (self.rot_index[0] - self.rot_sweep[0] + b) // b
        var highest = lowest
        for k in range(1, count):
            var g = (self.rot_index[k] - self.rot_sweep[k] + b) // b
            lowest = min(lowest, g)
            highest = max(highest, g)
        var groups = highest - lowest + 1
        var start = List[Int](length=groups + 1, fill=0)
        for k in range(count):
            var g = (self.rot_index[k] - self.rot_sweep[k] + b) // b - lowest
            start[g + 1] += 1
        for g in range(groups):
            start[g + 1] += start[g]
        var cursor = start.copy()
        var ordered = List[Int](length=count, fill=0)
        for k in range(count):
            var g = (self.rot_index[k] - self.rot_sweep[k] + b) // b - lowest
            ordered[cursor[g]] = k
            cursor[g] += 1

        var zv = self.zt.view()
        for step in range(groups):
            var g = groups - 1 - step
            var first = start[g]
            var last = start[g + 1]
            if first == last:
                continue

            # The exact span the group touches, rather than the formula's
            # bound: tight, never wider than `2 * block`, and never 1,
            # since a rotation owns two columns.
            var c_lo = self.rot_index[ordered[first]]
            var c_hi = c_lo + 1
            for p in range(first + 1, last):
                var i = self.rot_index[ordered[p]]
                c_lo = min(c_lo, i)
                c_hi = max(c_hi, i + 1)
            var w = c_hi - c_lo + 1

            self._build_ut(w, c_lo, ordered, first, last)
            self.u_dev.copy_from_host(self.ut_host)

            var staged: _Dense[Self.dtype] = TileTensor(
                self.staged.view().ptr_at_offset(Coord(0, 0)),
                row_major(Coord(w, cols)),
            )
            pack_block[target=_target[Self.gpu]()](
                zv, staged, c_lo, 0, w, cols, ctx
            )
            var ut: _Dense[Self.dtype] = TileTensor(
                self.u_dev.view().ptr_at_offset(Coord(0, 0)),
                row_major(Coord(w, w)),
            )
            var out: _Dense[Self.dtype] = TileTensor(
                zv.ptr_at_offset(Coord(c_lo, 0)), row_major(Coord(w, cols))
            )
            _max_matmul[target=_target[Self.gpu]()](out, ut, staged, ctx)
            ctx.synchronize()

        self.rot_index.clear()
        self.rot_sweep.clear()
        self.rot_cos.clear()
        self.rot_sin.clear()

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
        the unblocked body's four lines run on rows `a, a+1` of `U^T`
        instead of on columns `i, i+1` of `Z`, with `a = i - c_lo`.
        """
        comptime lanes = simd_width_of[Self.dtype]()
        for k in range(w * w):
            self.ut_host[k] = Scalar[Self.dtype](0)
        for k in range(w):
            self.ut_host[k * w + k] = Scalar[Self.dtype](1)

        var p = self.ut_host.unsafe_ptr()
        for entry in range(first, last):
            var k = ordered[entry]
            var c = self.rot_cos[k]
            var s = self.rot_sin[k]
            var ra = (self.rot_index[k] - c_lo) * w
            var rb = ra + w
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
    dtype: DType, N: Int, gpu: Bool, vectors: Bool
](
    mut d: List[Scalar[dtype]],
    mut e: List[Scalar[dtype]],
    mut acc: _RotationBatch[dtype, N, gpu, vectors],
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

    `e` is read as the `n` entries `TensorTridiagonal` carries, with
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


def eigvalsh[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n, n]) raises -> Static[
    dtype, n
] where dtype.is_floating_point():
    """**Tier 2.** The eigenvalues of a symmetric `a`, ascending, without
    the eigenvectors. `numpy.linalg.eigvalsh` / `scipy.linalg.eigvalsh`.

    Two phases, and the split is the whole story. `sytrd` reduces `a` to
    tridiagonal form device-resident -- `O(4n^3/3)` with the cubic term in
    `linalg.matmul` -- and then `_tql` runs implicit QL sweeps on the
    two diagonals on the host. That sweep is `O(n^2)`, sequential, and
    branches on the data, which is tier 2 by numax's definition; beside the
    reduction it is genuinely negligible, and the eigenvalues never asked
    for a vector to be accumulated, which is where the `O(n^3)` of a
    host-side `eigh` would hide.

    Ascending, as SciPy returns them. The `Array` tier's `eigvalsh` is
    cyclic Jacobi at a fixed sweep count and returns its values unsorted --
    the two agree as multisets, and a test pins that.

    `a` is read as symmetric and not checked; see `sytrd`.
    """
    var ctx = a.context()
    var reduced = sytrd[gpu=gpu](a)
    var d = reduced.d.to_host()
    var e = reduced.e.to_host()
    # `vectors=False` pushes nothing, so the batch is three one-element
    # allocations and its `block` never decides anything.
    var acc = _RotationBatch[dtype, n, gpu, False](1, ctx)
    _tql[dtype, n, gpu, False](d, e, acc, ctx)
    _std_sort(d)
    return Static[dtype, n](ctx, d^)


struct TensorEigh[dtype: DType, n: Int](
    Movable where dtype.is_floating_point() and n >= 1
):
    """`eigh`'s result: eigenvalues ascending, eigenvectors as columns.

    A struct rather than the `(w, v)` tuple SciPy returns, for the reason
    `qr_factor` returns a `TensorQR`: a `Tuple` of two `Tensor`s cannot be
    destructured in Mojo 1.0, so a tuple-shaped return would hand back a
    pair no caller could take apart. `numax.linalg.array.eigh` returns the
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


def eigh[
    dtype: DType, n: Int, gpu: Bool = False, block: Int = 32
](mut a: Static[dtype, n, n]) raises -> TensorEigh[dtype, n] where (
    dtype.is_floating_point() and block >= 1
):
    """**Tier 2.** The eigendecomposition of a symmetric `a`: eigenvalues
    ascending and orthonormal eigenvectors as columns.
    `numpy.linalg.eigh` / `scipy.linalg.eigh`.

    Three steps, all three GEMM-shaped. `sytrd` reduces `a` to
    tridiagonal form device-resident, `O(4n^3/3)` through `linalg.matmul`.
    Implicit QL then diagonalizes the tridiagonal, its rotations
    accumulated into `Z^T` device-resident in windows of commuting
    rotations that each go out as one `matmul` -- `_RotationBatch` above
    carries the argument. And the eigenvectors of `a` are `Q Z`, where `Q`
    is the reduction's orthogonal factor -- LAPACK's `ormtr` -- which is
    `inner(q, zt)` under `transpose_b=True`, so `Z` is never transposed
    back.

    `block` is the window: `block` consecutive sweeps batch together and a
    window spans `2 * block` columns, so the host builds `w x w` rotation
    products at `O(block * n^2)` and MAX does the `O(n^3)`. `block == 1`
    recovers the unblocked algorithm exactly, one rotation per product,
    and a test pins every size against every other.

    What is left on the host: the `O(n^2)` band sweep itself, which is
    sequential and deflates on a test of the data, and the `O(n^2)`
    insertion sort of the eigenvalue order -- the permutation is then
    applied to `Z^T`'s rows in one launch on the device.

    The `Array` tier's `eigh` is cyclic Jacobi at a fixed sweep count,
    unsorted, differentiable and launchable inside a GPU thread. For a
    matrix small enough to live in registers it is the better tool; this
    is for the sizes it cannot reach.

    `a` is read as symmetric and not checked; see `sytrd`.
    """
    var ctx = a.context()
    var reduced = sytrd[gpu=gpu](a)
    var d = reduced.d.to_host()
    var e = reduced.e.to_host()

    var acc = _RotationBatch[dtype, n, gpu, True](block, ctx)
    _tql[dtype, n, gpu, True](d, e, acc, ctx)
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

    var sorted_values = List[Scalar[dtype]](capacity=n)
    var perm_host = List[Scalar[DType.int64]](capacity=n)
    for j in range(n):
        sorted_values.append(d[order[j]])
        perm_host.append(Scalar[DType.int64](order[j]))
    var values = Static[dtype, n](ctx, sorted_values^)
    var perm = Static[DType.int64, n](ctx, perm_host^)

    # Row `j` of `zt` is eigenvector `j`, so the sort is a row gather --
    # source and destination are the same shape, which is what keeps a
    # cross-shape `elementwise` read out of it.
    var zt_sorted = Static[dtype, n, n]._uninitialized(ctx)
    var src = acc.zt.view()
    var dst = zt_sorted.view()
    var pv = perm.view()

    @always_inline
    def gather[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var src, var dst, var pv}:
        var at = coord_to_index_list(coord)
        dst.store[1](coord, src[Coord(Int(pv[Coord(at[0])]), at[1])])

    elementwise[simd_width=1, target=_target[gpu]()](gather, Coord(n, n), ctx)
    ctx.synchronize()
    # Both owners' last mention is `.view()` above, and a view erases the
    # origin, so without these Mojo frees them while `gather` still reads
    # through `src` and `pv`.
    _ = acc^
    _ = perm^

    var q = reduced.q()
    var vectors = inner[gpu=gpu](q, zt_sorted)
    return TensorEigh[dtype, n](values^, vectors^)


# --------------------------------------------------- Hessenberg and Schur


def _accumulate_reflectors[
    dtype: DType, n: Int, gpu: Bool = False
](
    mut reflectors: Static[dtype, n, n],
    mut taus: Static[dtype, n],
    count: Int,
    block: Int,
) raises -> Static[dtype, n, n] where dtype.is_floating_point():
    """`Q = H_0 H_1 ... H_{count-1}` from `count` reflectors held in LAPACK's
    packed form -- column `k` carries `v` below row `k + 1` with its leading
    `1` implicit -- applied to the identity in panels of `block`, last panel
    first. LAPACK's `orgtr` and `orghr` are both this.

    The panel walk is `TensorQR.q()`'s and so are the kernels:
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
    var ctx = reflectors.context()
    var result = Static[dtype, n, n]._uninitialized(ctx)
    var out: _Dense[dtype] = TileTensor(
        result.view().ptr_at_offset(Coord(0, 0)), row_major(Coord(n, n))
    )
    _device_identity[dtype, gpu](out, n, n, ctx)
    if count <= 0:
        ctx.synchronize()
        return result^

    var shifted: _Dense[dtype] = TileTensor(
        reflectors.view().ptr_at_offset(Coord(1, 0)),
        row_major(Coord(n - 1, n)),
    )
    var tv = taus.view()
    var work = _ReflectorWork[dtype](n - 1, n, block, ctx)
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
    dtype: DType, n: Int, gpu: Bool = False
](
    mut dst: Static[dtype, n, n],
    mut v: Static[dtype, n],
    k: Int,
    ctx: DeviceContext,
) raises:
    """`dst[:, k] = v`, on the device."""
    var dv = dst.view()
    var vv = v.view()

    @always_inline
    def fill[w: Int, alignment: Int = 1](coord: Coord) {var dv, var vv, var k}:
        var i = coord_to_index_list(coord)[0]
        dv.store[1](Coord(i, k), vv[Coord(i)])

    elementwise[simd_width=1, target=_target[gpu]()](fill, Coord(n), ctx)


def _zero_column_below[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n, n], k: Int, first: Int, ctx: DeviceContext) raises:
    """`a[first.., k] = 0`, on the device: the entries a reflector
    annihilates, written as the exact zeros they are rather than left to
    the update that never touches its own column."""
    var av = a.view()

    @always_inline
    def fill[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var av, var k, var first}:
        var i = coord_to_index_list(coord)[0]
        if i >= first:
            av.store[1](Coord(i, k), Scalar[dtype](0))

    elementwise[simd_width=1, target=_target[gpu]()](fill, Coord(n), ctx)


struct TensorHessenberg[dtype: DType, n: Int, gpu: Bool = False](
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
    leading `1` implicit -- LAPACK's packed form, `TensorTridiagonal`'s."""

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

        A column whose reflection was the identity carries an unwritten
        `v` of zeros here rather than the packed form the reduction left
        in `sytrd`'s work matrix; both read the same, because `larft`
        gives a zero `tau` a zero column of `T` and the panel never sees
        that reflector again.
        """
        return _accumulate_reflectors[Self.dtype, Self.n, Self.gpu](
            self.reflectors, self.taus, Self.n - 2, self.block
        )


def hessenberg[
    dtype: DType, n: Int, gpu: Bool = False, block: Int = 32
](mut a: Static[dtype, n, n]) raises -> TensorHessenberg[dtype, n, gpu] where (
    dtype.is_floating_point() and block >= 1
):
    """**Tier 2.** Reduce a general square `a` to upper Hessenberg form by
    Householder reflections, device-resident. LAPACK's `gehrd`,
    `scipy.linalg.hessenberg`.

    The shape is `sytrd`'s, minus the symmetry: the same single-block
    kernel forms each column's reflector, and then `A <- (I - tau v v^T) A
    (I - tau v v^T)` is two matrix-vector products and two rank-one
    updates through `linalg.matmul`, each over the whole matrix with the
    padded reflector -- `vpad` is zero at and above `k`, so the leading
    block is untouched for free and nothing is staged. The row `v^T A` has
    its first `k + 1` entries cleared before the left update so column `k`,
    already written as `(beta, 0, ...)`, is not disturbed; the right update
    never reaches it, since `vpad` is zero there.

    `ponytail:` unblocked, and the same two ceilings as `sytrd` -- the
    whole matrix per reflector (about `5n^3` flops against LAPACK's
    `10n^3/3`) and matrix-vector shaped products. The upgrade is the blocked
    `gehrd` with `lahr2`, which accumulates a panel's reflectors and applies
    them as GEMMs.

    `block` is that panel width, carried on the result and read by `.q()`
    alone until `lahr2` lands.

    `numax.linalg.array.hessenberg` is the `FloatLike`-generic sibling for
    matrices small enough to live in registers.
    """
    var ctx = a.context()
    var work = zeros[dtype, n, n](ctx)
    var reflectors = zeros[dtype, n, n](ctx)
    var taus = zeros[dtype, n](ctx)
    var vpad = zeros[dtype, n](ctx)
    var scratch = zeros[dtype, _PANEL_THREADS + 1](ctx)
    var product = zeros[dtype, n, n](ctx)

    var wv = work.view()
    var tv = taus.view()
    var vv = vpad.view()
    var sv = scratch.view()

    pack_block[target=_target[gpu]()](a.view(), wv, 0, 0, n, n, ctx)

    for k in range(n - 2):
        comptime if gpu:
            ctx.enqueue_function[
                sytd2_column[
                    dtype,
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
                Int32(k),
                Int32(n),
                grid_dim=1,
                block_dim=_PANEL_THREADS,
            )
            ctx.synchronize()
        else:
            sytd2_column(wv, vv, tv, sv, Int32(k), Int32(n))

        var this_tau = taus.to_host()[k]
        if this_tau == 0:
            continue

        _store_column[gpu=gpu](reflectors, vpad, k, ctx)
        _zero_column_below[gpu=gpu](work, k, k + 2, ctx)

        # Left: `A -= tau v (v^T A)`, with the row cleared over the columns
        # already finished.
        var y = _row_combination[gpu=gpu](work, vpad)
        _zero_prefix[gpu=gpu](y, k + 1, ctx)
        _rank_one_subtract[gpu=gpu](work, vpad, y, this_tau, product, ctx)

        # Right: `A -= tau (A v) v^T`, on the left-updated matrix.
        var u = matvec[gpu=gpu](work, vpad)
        _rank_one_subtract[gpu=gpu](work, u, vpad, this_tau, product, ctx)

    _ = scratch^
    return TensorHessenberg[dtype, n, gpu](work^, reflectors^, taus^, block)


comptime _MAX_QR_SWEEPS_PER_N = 30
"""Francis sweeps allowed in total, per matrix dimension -- EISPACK's and
LAPACK's `30 n`. Reaching it means the Hessenberg matrix carries a NaN or
an infinity, and the raise says so rather than looping forever."""


def _hqr[
    dtype: DType, wantt: Bool, wantz: Bool
](
    mut h: List[Scalar[dtype]], mut z: List[Scalar[dtype]], n: Int
) raises -> Tuple[
    List[Scalar[dtype]], List[Scalar[dtype]]
] where dtype.is_floating_point():
    """Francis double-shift QR on the upper Hessenberg `h` (row-major
    `n x n`, in place). EISPACK's `hqr2`, which is LAPACK's `dlahqr`.

    **Tier 2, host-side.** Each sweep chases a `3 x 3` bulge down the
    active block with Householder reflectors of order three, deflates on a
    test of the data, and takes an exceptional shift every tenth sweep --
    none of which has a GEMM to hand anything to. Returns the eigenvalues
    as `(real, imaginary)` in LAPACK's order; a complex pair sits in
    consecutive slots with the positive imaginary part first.

    With `wantt` the whole quasi-triangular `T` is kept -- the reflectors
    are applied to every column to the right and every row above -- and
    the strict lower band the chase never writes (the two entries each
    reflector annihilates in the column to its left, mathematically zero)
    is cleared at the end, so `T` is the Schur form and not the Schur form
    plus stale bulge entries. Without it only the active block is updated,
    which is all the eigenvalues need. With `wantz` the same reflectors are
    accumulated into `z`, which the caller passes as the identity.

    A converged `2 x 2` block with real eigenvalues is split by one
    rotation, so every surviving `2 x 2` block on `T`'s diagonal has a
    complex conjugate pair -- the standard real Schur form.
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
    var t = zero
    var itn = _MAX_QR_SWEEPS_PER_N * n
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
                wr[en] = x + t
                wi[en] = zero
                h[en * n + en] = x + t
                en -= 1
                break
            y = h[(en - 1) * n + (en - 1)]
            w = h[en * n + (en - 1)] * h[(en - 1) * n + en]
            if l == en - 1:
                # Two roots.
                p = Scalar[dtype](0.5) * (y - x)
                q = p * p + w
                zz = _sqrt(abs(q))
                h[en * n + en] = x + t
                x = x + t
                h[(en - 1) * n + (en - 1)] = y + t
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
                        for i in range(n):
                            var left = z[i * n + (en - 1)]
                            z[i * n + (en - 1)] = q * left + p * z[i * n + en]
                            z[i * n + en] = q * z[i * n + en] - p * left
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

            if itn == 0:
                raise Error(
                    "eigvals/schur: the QR iteration did not converge in ",
                    _MAX_QR_SWEEPS_PER_N * n,
                    " sweeps; the matrix likely holds a NaN or an infinity",
                )
            if its == 10 or its == 20:
                # Exceptional shift.
                t += x
                for i in range(en + 1):
                    h[i * n + i] = h[i * n + i] - x
                s = abs(h[en * n + (en - 1)]) + abs(h[(en - 1) * n + (en - 2)])
                x = Scalar[dtype](0.75) * s
                y = x
                w = Scalar[dtype](-0.4375) * s * s
            its += 1
            itn -= 1

            # Two consecutive small subdiagonals let the sweep start below
            # `l`: the first column of the shift polynomial, and where it is
            # negligible against its neighbours.
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
                for j in range(k, n if wantt else en + 1):
                    p = h[k * n + j] + q * h[(k + 1) * n + j]
                    if notlast:
                        p = p + r * h[(k + 2) * n + j]
                        h[(k + 2) * n + j] = h[(k + 2) * n + j] - p * zz
                    h[(k + 1) * n + j] = h[(k + 1) * n + j] - p * y
                    h[k * n + j] = h[k * n + j] - p * x
                var i_last = en if en < k + 3 else k + 3
                for i in range(0 if wantt else l, i_last + 1):
                    p = x * h[i * n + k] + y * h[i * n + (k + 1)]
                    if notlast:
                        p = p + zz * h[i * n + (k + 2)]
                        h[i * n + (k + 2)] = h[i * n + (k + 2)] - p * r
                    h[i * n + (k + 1)] = h[i * n + (k + 1)] - p * q
                    h[i * n + k] = h[i * n + k] - p
                comptime if wantz:
                    for i in range(n):
                        p = x * z[i * n + k] + y * z[i * n + (k + 1)]
                        if notlast:
                            p = p + zz * z[i * n + (k + 2)]
                            z[i * n + (k + 2)] = z[i * n + (k + 2)] - p * r
                        z[i * n + (k + 1)] = z[i * n + (k + 1)] - p * q
                        z[i * n + k] = z[i * n + k] - p

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
    `numax.fft`'s `Spectrum` answers the same way; `numax.linalg.array.eigvals`
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


def eigvals[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n, n]) raises -> Eigenvalues[
    dtype, n
] where dtype.is_floating_point():
    """**Tier 2.** The eigenvalues of a general square `a`, real or
    complex, as a `(re, im)` pair. `numpy.linalg.eigvals`,
    `scipy.linalg.eigvals`.

    `hessenberg` reduces `a` device-resident with the cubic term in
    `linalg.matmul`, then the Francis double-shift QR iteration runs on
    the Hessenberg matrix on the host -- `O(n^2)` per sweep and a few
    sweeps per eigenvalue, tier 2 by numax's definition and declared so in
    `_hqr` where it happens. Nothing is accumulated, so this is the cheap
    half of `schur`. `eigvalsh` is the symmetric route, with real output
    and a better algorithm; use it when the matrix is symmetric.

    Eigenvalues come out in LAPACK's deflation order, not sorted; a
    complex pair is adjacent with the positive imaginary part first.
    `numax.linalg.array.eigvals` is the fixed-sweep, differentiable
    sibling for matrices small enough to live in registers.
    """
    var ctx = a.context()
    var reduced = hessenberg[gpu=gpu](a)
    var h = reduced.h.to_host()
    var z = List[Scalar[dtype]]()
    var values = _hqr[dtype, False, False](h, z, n)
    var re = Static[dtype, n](ctx, values[0].copy())
    var im = Static[dtype, n](ctx, values[1].copy())
    return Eigenvalues[dtype, n](re^, im^)


struct TensorSchur[dtype: DType, n: Int](
    Movable where dtype.is_floating_point() and n >= 1
):
    """`schur`'s result: `a = z t z^T` with `t` real quasi-triangular and
    `z` orthogonal. A struct rather than SciPy's `(T, Z)` tuple, for the
    reason `qr_factor` returns a `TensorQR`.
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


def schur[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n, n]) raises -> TensorSchur[
    dtype, n
] where dtype.is_floating_point():
    """**Tier 2.** The real Schur decomposition `a = Z T Z^T`.
    `scipy.linalg.schur(a, output="real")`.

    Three steps, two of them GEMM-shaped: `hessenberg` reduces `a`
    device-resident, the Francis iteration triangularizes the Hessenberg
    matrix on the host while accumulating its reflectors into `Z_h`, and
    the Schur vectors of `a` are `Q Z_h` with `Q` the reduction's factor
    -- one `matmul`.

    `ponytail:` the middle step is the ceiling, and the same one `eigh`
    names: applying each order-three reflector to every column of `T` and
    every column of `Z_h` is `O(n^3)` of scalar host work at a small
    constant, the one term with no MAX in it. The upgrade is LAPACK's
    multishift QR with aggressive early deflation (`dhseqr`), whose
    reflector products are GEMM-shaped.

    This is the form every matrix function in `numax.linalg.matfuncs`
    beyond `expm` is built on.
    """
    var ctx = a.context()
    var reduced = hessenberg[gpu=gpu](a)
    var h = reduced.h.to_host()
    var z = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        z[i * n + i] = Scalar[dtype](1)
    _ = _hqr[dtype, True, True](h, z, n)
    var t = Static[dtype, n, n](ctx, h^)
    var z_h = Static[dtype, n, n](ctx, z^)
    var q = reduced.q()
    var vectors = matmul[gpu=gpu](q, z_h)
    return TensorSchur[dtype, n](t^, vectors^)


# ----------------------------------------------------------------- SVD


def _zero_prefix[
    dtype: DType, n: Int, gpu: Bool = False
](mut v: Static[dtype, n], count: Int, ctx: DeviceContext) raises:
    """Zero `v[0 .. count)` on `v`'s device.

    What confines a reflector's update to the trailing block: the product
    `v^T A` or `A u` is taken over the whole matrix, and the entries that
    would write the row or column just finished are cleared before the
    rank-one update reads them -- the same load-bearing detail `sytrd`
    records for its `w`.
    """
    var view = v.view()

    @always_inline
    def fill[w: Int, alignment: Int = 1](coord: Coord) {var view, var count}:
        var i = coord_to_index_list(coord)[0]
        if i < count:
            view.store[1](coord, Scalar[dtype](0))

    elementwise[simd_width=1, target=_target[gpu]()](fill, Coord(n), ctx)


def _left_products[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](mut a: Static[dtype, m, n], mut v: Static[dtype, m]) raises -> Static[
    dtype, n
]:
    """`v^T A` as an `n`-vector: `v` read as a `1 x m` row against `A`."""
    var ctx = a.context()
    var result = zeros[dtype, n](ctx)
    var row: _Dense[dtype] = TileTensor(
        v.view().ptr_at_offset(Coord(0)), row_major(Coord(1, m))
    )
    var out: _Dense[dtype] = TileTensor(
        result.view().ptr_at_offset(Coord(0)), row_major(Coord(1, n))
    )
    _max_matmul[target=_target[gpu]()](out, row, a.view(), ctx)
    ctx.synchronize()
    return result^


def _rank_one_subtract_rect[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](
    mut a: Static[dtype, m, n],
    mut col: Static[dtype, m],
    mut row: Static[dtype, n],
    scale: Scalar[dtype],
    mut product: Static[dtype, m, n],
    ctx: DeviceContext,
) raises:
    """`A -= scale * col row^T` for a rectangular `A`, through `matmul`'s
    epilogue -- `_rank_one_subtract` at `m x n`."""
    var target = a.view()

    @parameter
    @always_inline
    @__copy_capture(target, scale)
    def subtract[
        _dtype: DType,
        lanes: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, lanes]](),
    ](idx: IndexList[2], value: SIMD[_dtype, lanes]) capturing -> None:
        var at = Coord(idx[0], idx[1])
        target.store[lanes](
            at,
            target.load[lanes](at) - scale * rebind[SIMD[dtype, lanes]](value),
        )

    var c: _Dense[dtype] = TileTensor(
        col.view().ptr_at_offset(Coord(0)), row_major(Coord(m, 1))
    )
    var r: _Dense[dtype] = TileTensor(
        row.view().ptr_at_offset(Coord(0)), row_major(Coord(1, n))
    )
    _max_matmul[elementwise_lambda_fn=subtract, target=_target[gpu]()](
        product.view(), c, r, ctx
    )
    ctx.synchronize()


def _column_of[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](mut dense: Static[dtype, m, n], k: Int) raises -> Static[dtype, m]:
    """Column `k` of a dense `m x n` as its own vector, gathered on the
    device."""
    var ctx = dense.context()
    var out = zeros[dtype, m](ctx)
    var src = dense.view()
    var dst = out.view()

    @always_inline
    def gather[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var src, var dst, var k}:
        var i = coord_to_index_list(coord)[0]
        dst.store[1](coord, src[Coord(i, k)])

    elementwise[simd_width=1, target=_target[gpu]()](gather, Coord(m), ctx)
    return out^


def _row_of[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](mut dense: Static[dtype, m, n], k: Int) raises -> Static[dtype, n]:
    """Row `k` of a dense `m x n` as its own vector, on the device -- a
    contiguous copy, so a `pack_block` of one row."""
    var ctx = dense.context()
    var out = zeros[dtype, n](ctx)
    var dst: _Dense[dtype] = TileTensor(
        out.view().ptr_at_offset(Coord(0)), row_major(Coord(1, n))
    )
    pack_block[target=_target[gpu]()](dense.view(), dst, k, 0, 1, n, ctx)
    ctx.synchronize()
    return out^


struct TensorBidiagonal[dtype: DType, m: Int, n: Int, gpu: Bool = False](
    Movable where dtype.is_floating_point() and m >= n and n >= 1
):
    """The bidiagonal reduction `A = Q B P^T` of an `m x n` matrix with
    `m >= n`, device-resident. LAPACK's `gebrd` output.

    `B` is upper bidiagonal: `d` on the diagonal, `e[0 .. n-2]` on the
    superdiagonal (`e[n-1]` is unused and zero, for the reason
    `TensorTridiagonal` gives). `Q` (`m x n`, the thin form) and `P`
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
    """The panel width `.q()` and `.p()` form their factors in, carried
    from `gebrd`."""

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

        This is `TensorQR.q()`'s walk unchanged, because `left` is already
        in QR's packed form: `gebd2_col` writes the reflector's unit at row
        `k` -- the row a QR puts it on -- so no row-shifted view is needed
        here, unlike `orgtr`'s. `pack_reflectors` forces the diagonal to
        `1` and reads only the strict lower trapezoid, so the explicit unit
        and the explicit zeros above it are ignored either way.

        Three matrix products per panel against the `3n` launches and `2n`
        device synchronizations the per-reflector walk needed. `svdvals`
        never calls this.
        """
        var ctx = self.left.context()
        var result = Static[Self.dtype, Self.m, Self.n]._uninitialized(ctx)
        var out: _Dense[Self.dtype] = TileTensor(
            result.view().ptr_at_offset(Coord(0, 0)),
            row_major(Coord(Self.m, Self.n)),
        )
        _device_identity[Self.dtype, Self.gpu](out, Self.m, Self.n, ctx)

        var tv = self.taus_left.view()
        var work = _ReflectorWork[Self.dtype](Self.m, Self.n, self.block, ctx)
        var steps = (Self.n + self.block - 1) // self.block
        for step in range(steps):
            var k = (steps - 1 - step) * self.block
            var nb = min(self.block, Self.n - k)
            _apply_block_reflector[transposed=False, gpu=Self.gpu](
                self.left.view(),
                tv,
                out,
                k,
                nb,
                Self.m,
                Self.m - k,
                Self.n - k,
                k,
                k,
                work,
                ctx,
            )

        ctx.synchronize()
        return result^

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
        var ctx = self.right.context()
        var packed = Static[Self.dtype, Self.n, Self.n]._uninitialized(ctx)
        var pv = packed.view()
        pack_block[trans=True, target=_target[Self.gpu]()](
            self.right.view(), pv, 0, 0, Self.n, Self.n, ctx
        )
        return _accumulate_reflectors[Self.dtype, Self.n, Self.gpu](
            packed, self.taus_right, Self.n - 2, self.block
        )


def gebrd[
    dtype: DType, m: Int, n: Int, gpu: Bool = False, block: Int = 32
](mut a: Static[dtype, m, n]) raises -> TensorBidiagonal[
    dtype, m, n, gpu
] where (dtype.is_floating_point() and m >= n and n >= 1 and block >= 1):
    """**Tier 2.** Reduce an `m x n` matrix, `m >= n`, to upper bidiagonal
    form by alternating left and right Householder reflections,
    device-resident. LAPACK's `gebrd`, unblocked.

    Column `k` takes a left reflector from `a[k.., k]` and row `k` a right
    one from `a[k, k+1..]`; each is applied to the rest of the matrix as a
    matrix-vector product and a rank-one update through `linalg.matmul`,
    with the product's entry for the row or column just finished cleared
    first so the update never touches it. That is the whole of the
    arithmetic, `O(4 m n^2)` at matrix-vector shapes.

    `block` is the panel width `.q()` and `.p()` form `Q` and `P` in. The
    reduction itself does not read it yet; it is carried on the result, and
    `labrd` will take the same number.

    `ponytail:` unblocked, like the first `sytrd`, and with the same
    ceiling: matrix-vector shapes are bandwidth-bound where LAPACK's
    blocked `labrd` accumulates a panel and applies it as GEMMs. The
    upgrade is that panel; nothing above changes when it lands.
    """
    var ctx = a.context()
    var work = zeros[dtype, m, n](ctx)
    var left = zeros[dtype, m, n](ctx)
    var right = zeros[dtype, n, n](ctx)
    var taus_left = zeros[dtype, n](ctx)
    var taus_right = zeros[dtype, n](ctx)
    var scratch = zeros[dtype, _PANEL_THREADS + 1](ctx)
    var product = zeros[dtype, m, n](ctx)

    var wv = work.view()
    var lv = left.view()
    var rv = right.view()
    var tlv = taus_left.view()
    var trv = taus_right.view()
    var sv = scratch.view()
    pack_block[target=_target[gpu]()](a.view(), wv, 0, 0, m, n, ctx)

    for k in range(n):
        # Left reflector on column k.
        comptime if gpu:
            ctx.enqueue_function[
                gebd2_col[
                    dtype,
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
                Int32(k),
                Int32(m),
                grid_dim=1,
                block_dim=_PANEL_THREADS,
            )
            ctx.synchronize()
        else:
            gebd2_col(wv, lv, tlv, sv, Int32(k), Int32(m))

        var tau_l = taus_left.to_host()[k]
        if tau_l != 0:
            var v = _column_of[gpu=gpu](left, k)
            var w = _left_products[gpu=gpu](work, v)
            _zero_prefix[gpu=gpu](w, k + 1, ctx)
            _rank_one_subtract_rect[gpu=gpu](work, v, w, tau_l, product, ctx)

        # Right reflector on row k, when there is a superdiagonal to reduce.
        if k + 1 < n:
            comptime if gpu:
                ctx.enqueue_function[
                    gebd2_row[
                        dtype,
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
                    Int32(k),
                    Int32(n),
                    grid_dim=1,
                    block_dim=_PANEL_THREADS,
                )
                ctx.synchronize()
            else:
                gebd2_row(wv, rv, trv, sv, Int32(k), Int32(n))

            var tau_r = taus_right.to_host()[k]
            if tau_r != 0:
                var u = _row_of[gpu=gpu](right, k)
                var pcol = matvec[gpu=gpu](work, u)
                _zero_prefix[gpu=gpu](pcol, k + 1, ctx)
                _rank_one_subtract_rect[gpu=gpu](
                    work, pcol, u, tau_r, product, ctx
                )

    _ = scratch^  # as in `sytrd`: pin the workspace past its last launch
    var reduced = work.to_host()
    var d = zeros[dtype, n](ctx)
    var e = zeros[dtype, n](ctx)
    var d_host = d.to_host()
    var e_host = e.to_host()
    for i in range(n):
        d_host[i] = reduced[i * n + i]
        if i + 1 < n:
            e_host[i] = reduced[i * n + i + 1]
    d.copy_from_host(d_host)
    e.copy_from_host(e_host)
    return TensorBidiagonal[dtype, m, n, gpu](
        d^, e^, left^, right^, taus_left^, taus_right^, block
    )


def _golub_kahan[
    dtype: DType,
    n: Int,
    gpu: Bool = False,
    vectors: Bool = False,
](
    d: List[Scalar[dtype]],
    e: List[Scalar[dtype]],
    mut acc: _RotationBatch[dtype, 2 * n, gpu, vectors],
    ctx: DeviceContext,
) raises -> List[Scalar[dtype]] where dtype.is_floating_point():
    """The eigenvalues of the Golub-Kahan tridiagonal of the upper
    bidiagonal `(d, e)`, unsorted; at `vectors=True` its eigenvector matrix
    is left in `acc.zt`, **transposed** and device-resident, so row `j` is
    eigenvector `j`.

    The accumulator is the caller's, the way `_tql`'s is, because `acc.zt`
    is the `2n x 2n` device tensor `svd` then de-interleaves `U` and `V`
    out of -- it has to outlive this call, and `block` is the window it was
    built with rather than a second parameter here.

    The Golub-Kahan matrix is the `2n x 2n` symmetric tridiagonal with zero
    diagonal and off-diagonal `d_1, e_1, d_2, e_2, ..., d_n`. Its
    eigenvalues are `+-sigma_i`, and the eigenvector for `+sigma_i`
    interleaves the singular vectors: even entries are `v_i / sqrt(2)`, odd
    entries `u_i / sqrt(2)`. So the sweep `eigh` already runs gives the SVD
    of `B` with no new numerics -- LAPACK's `dbdsvdx` takes the same route.

    `ponytail:` at `vectors=True` this runs the sweep at `2n`, so it issues
    about eight times `eigh`'s rotations on the same `n`. They are
    accumulated in blocks and `Z^T` never leaves the device, but **the
    doubling itself stays** until a Golub-Kahan implicit QR sweep on the
    bidiagonal (`bdsqr`) replaces it -- that sweep rotates `U` and `V`
    directly at width `n`, about half the rotations and none of the
    interleaving. Singular vectors for an *exactly* zero singular value are
    not guaranteed orthonormal, since the two zero eigenvalues' vectors may
    mix; the values are exact, and `pinv`/`matrix_rank` never use those
    vectors.
    """
    comptime size = 2 * n
    var gd = List[Scalar[dtype]](length=size, fill=0)
    var ge = List[Scalar[dtype]](length=size, fill=0)
    for i in range(n):
        ge[2 * i] = d[i]
        if i + 1 < n:
            ge[2 * i + 1] = e[i]
    _tql[dtype, size, gpu, vectors](gd, ge, acc, ctx)
    acc.finish(ctx)
    return gd^


def _top_n_descending[
    dtype: DType, n: Int
](values: List[Scalar[dtype]]) -> List[Int]:
    """Indices of the `n` largest of `2n` eigenvalues, descending -- the
    positive half of the Golub-Kahan spectrum."""
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
    dtype: DType, m: Int, n: Int, gpu: Bool = False, block: Int = 32
](mut a: Static[dtype, m, n]) raises -> Static[dtype, n] where (
    dtype.is_floating_point() and m >= n and n >= 1 and block >= 1
):
    """**Tier 2.** The singular values of an `m x n` matrix, `m >= n`,
    descending. `scipy.linalg.svdvals`.

    `gebrd` device-resident, then the Golub-Kahan tridiagonal's
    eigenvalues by the `O(n^2)` implicit-QL sweep, the positive half taken.
    Descending, as SciPy returns them; the `Array` tier's `svdvals` is
    one-sided Jacobi and comes back unsorted, and a test pins the two as
    multisets.

    `block` is `gebrd`'s panel width, carried so `svdvals` and `svd` take
    the same knob. Nothing here reads it today: the reduction is still
    unblocked, and the rotation window it also names decides nothing
    without vectors, since a values-only sweep pushes no rotations.
    """
    var ctx = a.context()
    var reduced = gebrd[dtype, m, n, gpu, block](a)
    var acc = _RotationBatch[dtype, 2 * n, gpu, False](block, ctx)
    var values = _golub_kahan[dtype, n, gpu, False](
        reduced.d.to_host(), reduced.e.to_host(), acc, ctx
    )
    var top = _top_n_descending[dtype, n](values)
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        # `|.|`: an exactly singular matrix's zero pair can land with its
        # nominally positive member a rounding error below zero.
        out.append(abs(values[top[i]]))
    return Static[dtype, n](ctx, out^)


struct TensorSVD[dtype: DType, m: Int, n: Int](
    Movable where dtype.is_floating_point() and m >= n and n >= 1
):
    """`svd`'s result: `A = U diag(s) V^T`, singular values descending.

    A struct rather than SciPy's `(U, s, Vh)` tuple, for the reason
    `qr_factor` returns a `TensorQR`. The fields follow the `Array` tier's
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


def svd[
    dtype: DType, m: Int, n: Int, gpu: Bool = False, block: Int = 32
](mut a: Static[dtype, m, n]) raises -> TensorSVD[dtype, m, n] where (
    dtype.is_floating_point() and m >= n and n >= 1 and block >= 1
):
    """**Tier 2.** The thin singular value decomposition of an `m x n`
    matrix, `m >= n`: `A = U diag(s) V^T` with `s` descending.
    `scipy.linalg.svd(a, full_matrices=False)`.

    Three steps, and the vectors never touch the host in any of them.
    `gebrd` reduces `A` to bidiagonal `B = Q^T A P` device-resident. The
    Golub-Kahan tridiagonal of `B` is diagonalized by the same implicit-QL
    sweep `eigh` uses -- the band iteration on the host, its rotations
    accumulated into `Z^T` device-resident in windows that each go out as
    one `matmul`. Its eigenvectors interleave `B`'s singular vectors, and
    the de-interleave is one `elementwise` over `(n, n)`: row `top[j]` of
    `Z^T` scattered into `V_B^T` at its even entries and `U_B^T` at its odd
    ones, both scaled by `sqrt(2)`, so the two come out already transposed
    and `U = Q U_B`, `V = P V_B` are `inner(q, ub_t)` and `inner(p, vb_t)`
    under `transpose_b=True`. Only the `2n` eigenvalues and the order they
    sort into cross to the host.

    Two knobs share the name `block` and they mean different things, as
    `eigh`'s do. `gebrd` takes it as the **panel width** `.q()` and `.p()`
    form `Q` and `P` in -- `block` reflectors per block reflector, three
    GEMMs each. The sweep takes it as the **rotation window**: `block`
    consecutive sweeps batch together and a window spans `2 * block`
    columns of `Z^T`. `block == 1` recovers both unblocked algorithms
    exactly, and a test pins every width against every other.

    `ponytail:` the middle step is the ceiling, and it is larger than
    `eigh`'s by the factor the doubling costs -- the sweep runs at `2n`, so
    about eight times the rotations of an `eigh` on the same `n`. The
    upgrade is a bidiagonal QR sweep (`bdsqr`) rotating `U` and `V`
    directly at width `n`, or divide and conquer; nothing above or below
    changes when it lands. `svdvals` skips the vectors and is `O(n^2)` past
    the reduction.

    Rectangular, unlike the `Array` tier's square-only one-sided Jacobi,
    and descending where that one is unsorted; both docstrings say so.
    """
    var ctx = a.context()
    var reduced = gebrd[dtype, m, n, gpu, block](a)
    comptime size = 2 * n
    var acc = _RotationBatch[dtype, size, gpu, True](block, ctx)
    var values = _golub_kahan[dtype, n, gpu, True](
        reduced.d.to_host(), reduced.e.to_host(), acc, ctx
    )
    var order = _top_n_descending[dtype, n](values)

    var s_host = List[Scalar[dtype]](capacity=n)
    var top_host = List[Scalar[DType.int64]](capacity=n)
    for j in range(n):
        s_host.append(abs(values[order[j]]))
        top_host.append(Scalar[DType.int64](order[j]))
    var top = Static[DType.int64, n](ctx, top_host^)

    # `Z^T` is `2n x 2n` and the destinations are `n x n`, so the gather
    # reads a source of different extents -- the pattern `findings.mdc`
    # records as unreliable for *run-time* layouts. Here the source is
    # viewed at a compile-time `size x size` layout over the same pointer
    # and indexed by a freshly built `Coord`, which is the case that holds;
    # `n` in {1, 2, 3} is pinned by its own test.
    var ub_t = Static[dtype, n, n]._uninitialized(ctx)
    var vb_t = Static[dtype, n, n]._uninitialized(ctx)
    var zt = TileTensor(
        acc.zt.view().ptr_at_offset(Coord(0, 0)), row_major[size, size]()
    )
    var ub = ub_t.view()
    var vb = vb_t.view()
    var tv = top.view()
    var root_two = Scalar[dtype](1.4142135623730951)

    @always_inline
    def split[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var zt, var ub, var vb, var tv, var root_two}:
        var at = coord_to_index_list(coord)
        var row = Int(tv[Coord(at[0])])
        var col = 2 * at[1]
        vb.store[1](coord, zt[Coord(row, col)] * root_two)
        ub.store[1](coord, zt[Coord(row, col + 1)] * root_two)

    elementwise[simd_width=1, target=_target[gpu]()](split, Coord(n, n), ctx)
    ctx.synchronize()
    # `acc` and `top` are named nowhere past the `.view()` a view erases
    # the origin of, so without these Mojo frees them while `split` still
    # reads through `zt` and `tv`; see `eigh` and `findings.mdc`.
    _ = acc^
    _ = top^

    var q = reduced.q()
    var p = reduced.p()
    var u = inner[gpu=gpu](q, ub_t)
    var v = inner[gpu=gpu](p, vb_t)
    return TensorSVD[dtype, m, n](u^, Static[dtype, n](ctx, s_host^), v^)


def matrix_rank[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](
    mut a: Static[dtype, m, n], tol: Optional[Float64] = None
) raises -> Int where (dtype.is_floating_point() and m >= n and n >= 1):
    """**Tier 2.** How many singular values exceed `tol`.
    `numpy.linalg.matrix_rank`.

    `tol` defaults to NumPy's: the largest singular value times
    `max(m, n)` times the machine epsilon of `dtype`, which is the noise
    floor an SVD of that size can be expected to carry. Pass an explicit
    `tol` to ask a different question -- "how many directions carry more
    than one part in a thousand" is `tol = 1e-3 * s_max`.

    An `Int`, where the `Array` tier returns `T`: one `Tensor` is one
    matrix, so there is one rank, and nothing here needs to stay branchless.
    """
    var s = svdvals[gpu=gpu](a).to_host()
    var eps: Float64
    comptime if dtype == DType.float32:
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
