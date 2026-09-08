"""Householder QR and the least-squares solve built from it.
`scipy.linalg`'s `qr` and `lstsq`.

**The `Tensor` tier**, and **tier 2**: `qr_factor` is blocked, so its
panel is a `panel.mojo` kernel and its trailing update is three products
through `linalg.matmul`. `numax.linalg.array.qr` is the tier-1 tier, where
a QR differentiates and `qr` returns `(R, Q)` outright.

**MAX's counterpart is denied**, and it is MAX's only factorization.
`linalg.qr_factorization` is a LAPACK-style in-place Householder
factorization over the older `LayoutTensor`, and numax's interop is
`TileTensor` only, so that the library has exactly one owning tensor type
and one view type. Three other things would argue against it regardless:
it is monomorphic in `dtype`, it is a CPU-only scalar-loop reference
rather than a tuned kernel, and its `apply_q`/`form_q` companions are
`LayoutTensor` too.

The least-squares solve lives here rather than in `basic` (where SciPy
puts `lstsq`) because it *is* this algorithm: `TensorQR.solve` applies the
reflectors to `b` and back-substitutes, never forming `Q`, which is both
cheaper and better conditioned than the normal equations
`A^T A x = A^T b`. Keeping them together means the Householder code is in
one file.
"""

from layout import Coord, TileTensor, coord_to_index_list
from layout.tile_layout import TensorLayout, row_major
from linalg.matmul import matmul as _max_matmul
from std.sys.info import align_of
from std.utils import IndexList

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core.array import Dynamic, Static, zeros, zeros_dyn

from .blas import _target
from .common import _Dense
from .panel import (
    _PANEL_THREADS,
    _View,
    geqr2_panel,
    larft_panel,
    pack_block,
    pack_reflectors,
)
from .triangular import solve_triangular


comptime _MIN_GEMM_COLS = 2
"""The narrowest right-hand side a block reflector's products are allowed
to have, and the reason a one-column update is padded to two.

Working around a MAX defect, recorded in `.cursor/rules/max-feedback.mdc`:
`linalg.matmul` at `float64` with a single output column takes its GEMV
path, which reads past the end of the vector when the inner dimension is
not a multiple of four and segfaults. A blocked QR hits it constantly --
the last panel of a matrix whose width is one past a block boundary
updates exactly one column, and `apply_q_transpose` of a single vector is
one column by definition.

Two columns instead of one keeps every product off that path, at the cost
of one column of zeros through three GEMMs. Drop this to 1 when the
upstream fix lands; nothing else changes.
"""


struct _ReflectorWork[dtype: DType](Movable):
    """The dense scratch a block reflector needs, allocated once for a
    whole factorization rather than per panel step.

    Every block here is forced by one MAX constraint or another.
    `linalg.matmul` ignores its arguments' row stride, so an operand that
    is a sub-block of a larger matrix has to be copied dense before it can
    be multiplied -- that is `staged`, and it is also why `V` is
    materialized into `v`/`v_t` rather than read where `geqr2_panel` left
    it. `matmul` will not read and write one buffer, so `w` and `y` are
    separate. And it stores to `c` whether an epilogue is supplied or not,
    so `product` exists to be thrown away.

    Sized for the widest step of a whole factorization, so later steps use
    a prefix.
    """

    var v: Dynamic[Self.dtype, 2]
    var v_t: Dynamic[Self.dtype, 2]
    var t_block: Dynamic[Self.dtype, 2]
    var t_t: Dynamic[Self.dtype, 2]
    var w: Dynamic[Self.dtype, 2]
    var y: Dynamic[Self.dtype, 2]
    var product: Dynamic[Self.dtype, 2]
    var staged: Dynamic[Self.dtype, 2]

    def __init__(
        out self, rows: Int, cols: Int, block: Int, ctx: DeviceContext
    ) raises:
        var wide = max(cols, _MIN_GEMM_COLS)
        self.v = zeros_dyn[Self.dtype, 2](rows, block, ctx=ctx)
        self.v_t = zeros_dyn[Self.dtype, 2](block, rows, ctx=ctx)
        self.t_block = zeros_dyn[Self.dtype, 2](block, block, ctx=ctx)
        self.t_t = zeros_dyn[Self.dtype, 2](block, block, ctx=ctx)
        self.w = zeros_dyn[Self.dtype, 2](block, wide, ctx=ctx)
        self.y = zeros_dyn[Self.dtype, 2](block, wide, ctx=ctx)
        self.product = zeros_dyn[Self.dtype, 2](rows, wide, ctx=ctx)
        self.staged = zeros_dyn[Self.dtype, 2](rows, wide, ctx=ctx)


def _apply_block_reflector[
    dtype: DType,
    ALayout: TensorLayout,
    TauLayout: TensorLayout,
    CLayout: TensorLayout,
    transposed: Bool,
    gpu: Bool = False,
](
    a: _View[dtype, ALayout],
    taus: _View[dtype, TauLayout],
    c: _View[dtype, CLayout],
    k: Int,
    nb: Int,
    m: Int,
    rows: Int,
    width: Int,
    row0: Int,
    col0: Int,
    mut work: _ReflectorWork[dtype],
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """`C := (I - V T V^T) C` for the panel at `k`, or with `T^T` at
    `transposed=True`, where `C` is the `rows x width` block of `c` at
    `(row0, col0)`. LAPACK's `larfb`, side left.

    Three matrix products and nothing else: `W = V^T C`, `Y = T W`,
    `C -= V Y`. Every one is `linalg.matmul`, and that is what makes a
    blocked QR's `O(n^3)` -- the factorization's trailing update and the
    formation of `Q` alike -- MAX's GEMM rather than numax's loop. The
    `nb` rank-one updates this replaces are the difference between a QR
    that reaches GEMM throughput and one that does not.

    `C` is staged dense on the way in because `matmul` reads a strided
    operand as if it were contiguous, and comes back through `matmul`'s
    epilogue on the way out because the epilogue does its own addressing
    and so can scatter into a strided block. That asymmetry is why the
    block being updated is named by view and offsets rather than passed as
    a dense operand.

    `transposed` picks which side of the identity the block reflector sits
    on. The factorization applies `Q_block^T` to the trailing columns; the
    walk that forms `Q` applies `Q_block`, panels in reverse.

    `T` is rebuilt per call rather than cached. It costs `nb^3 / 3`
    against the `O(rows * nb * width)` of the products it enables, and
    caching it would mean carrying an `n x nb` band on the factorization
    for a term no measurement shows.
    """
    if rows <= 0 or width <= 0 or nb <= 0:
        return

    # See `_MIN_GEMM_COLS`: a one-column update would segfault inside MAX.
    var padded = max(width, _MIN_GEMM_COLS)

    # Every one of these is built at the shape this step needs, and the
    # kernel that fills it is handed the same view. A buffer sized for the
    # widest step has that step's row stride, so writing through the wide
    # view and reading through a narrow one would disagree about where row
    # `i` starts -- which is silent, and wrong only on the ragged last
    # panel.
    var t_block: _Dense[dtype] = TileTensor(
        work.t_block.view().ptr_at_offset(Coord(0, 0)),
        row_major(Coord(nb, nb)),
    )
    comptime if gpu:
        ctx.enqueue_function[
            larft_panel[
                dtype,
                ALayout=ALayout,
                TauLayout=TauLayout,
                TLayout=type_of(t_block).LayoutType,
                gpu=True,
            ]
        ](
            a,
            taus,
            t_block,
            Int32(k),
            Int32(nb),
            Int32(m),
            grid_dim=1,
            block_dim=_PANEL_THREADS,
        )
    else:
        larft_panel(a, taus, t_block, Int32(k), Int32(nb), Int32(m))

    var v: _Dense[dtype] = TileTensor(
        work.v.view().ptr_at_offset(Coord(0, 0)), row_major(Coord(rows, nb))
    )
    var v_t: _Dense[dtype] = TileTensor(
        work.v_t.view().ptr_at_offset(Coord(0, 0)), row_major(Coord(nb, rows))
    )
    pack_reflectors[target=_target[gpu]()](a, v, k, nb, rows, False, ctx)
    pack_reflectors[target=_target[gpu]()](a, v_t, k, nb, rows, True, ctx)

    var staged: _Dense[dtype] = TileTensor(
        work.staged.view().ptr_at_offset(Coord(0, 0)),
        row_major(Coord(rows, padded)),
    )

    @always_inline
    def stage[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var c, var staged, var row0, var col0, var width}:
        var at = coord_to_index_list(coord)
        var value = Scalar[dtype](0)
        if at[1] < width:
            value = c[Coord(row0 + at[0], col0 + at[1])]
        staged.store[1](coord, value)

    elementwise[simd_width=1, target=_target[gpu]()](
        stage, Coord(rows, padded), ctx
    )

    # `W = V^T C`.
    var w: _Dense[dtype] = TileTensor(
        work.w.view().ptr_at_offset(Coord(0, 0)), row_major(Coord(nb, padded))
    )
    _max_matmul[target=_target[gpu]()](w, v_t, staged, ctx)

    # `Y = T W`, with `T` transposed into its own block first when asked:
    # `matmul` transposes `b`, never `a`.
    var factor: _Dense[dtype]
    comptime if transposed:
        var t_t: _Dense[dtype] = TileTensor(
            work.t_t.view().ptr_at_offset(Coord(0, 0)),
            row_major(Coord(nb, nb)),
        )
        pack_block[trans=True, target=_target[gpu]()](
            t_block, t_t, 0, 0, nb, nb, ctx
        )
        factor = t_t
    else:
        factor = t_block

    var y: _Dense[dtype] = TileTensor(
        work.y.view().ptr_at_offset(Coord(0, 0)), row_major(Coord(nb, padded))
    )
    _max_matmul[target=_target[gpu]()](y, factor, w, ctx)

    var product: _Dense[dtype] = TileTensor(
        work.product.view().ptr_at_offset(Coord(0, 0)),
        row_major(Coord(rows, padded)),
    )

    if padded == width:
        # `C -= V Y`, scattered into `c` by the epilogue: one kernel, and
        # `product`'s store is the wasted half of it.
        @parameter
        @always_inline
        @__copy_capture(c, row0, col0)
        def subtract[
            _dtype: DType,
            lanes: SIMDLength,
            *,
            alignment: Int = align_of[SIMD[_dtype, lanes]](),
        ](idx: IndexList[2], value: SIMD[_dtype, lanes]) capturing -> None:
            var at = Coord(row0 + idx[0], col0 + idx[1])
            c.store[lanes](
                at, c.load[lanes](at) - rebind[SIMD[dtype, lanes]](value)
            )

        _max_matmul[elementwise_lambda_fn=subtract, target=_target[gpu]()](
            product, v, y, ctx
        )
    else:
        # The padded case cannot fuse: an epilogue store is as wide as the
        # SIMD lane it is given, so a two-column product would write the
        # pad column into whatever sits beside `C`. Subtract the real
        # columns afterwards instead -- one extra kernel, still no host
        # round trip.
        _max_matmul[target=_target[gpu]()](product, v, y, ctx)

        @always_inline
        def apply[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var c, var product, var row0, var col0}:
            var at = coord_to_index_list(coord)
            var target = Coord(row0 + at[0], col0 + at[1])
            c.store[1](target, c[target] - product[coord])

        elementwise[simd_width=1, target=_target[gpu]()](
            apply, Coord(rows, width), ctx
        )


struct TensorQR[dtype: DType, m: Int, n: Int, gpu: Bool = False](
    Movable where dtype.is_floating_point() and m >= n
):
    """A blocked Householder QR of an `m x n` matrix, held on its device.

    **Tier 2.** What `qr_factor` returns, and the reason it returns
    something rather than a pair of matrices: `Q` costs about as much to
    write down as the factorization did, and the three things a caller
    usually wants from a QR -- `R`, `Q^T b`, a least-squares solution --
    need only the reflectors. LAPACK splits `geqrf` from `orgqr` for the
    same reason, and `numpy.linalg.qr` is the outlier in always forming
    `Q`.

    `factored` holds `R` in its upper triangle and the reflectors' `V`
    below it, unit diagonal implicit, as `geqr2_panel` left it. `taus`
    holds one scale per column. Both stay in device memory, and `gpu` is
    part of the type, so a factorization built on the accelerator cannot
    be read by host code.
    """

    var factored: Static[Self.dtype, Self.m, Self.n]
    var taus: Static[Self.dtype, Self.n]
    var block: Int

    def __init__(
        out self,
        var factored: Static[Self.dtype, Self.m, Self.n],
        var taus: Static[Self.dtype, Self.n],
        block: Int,
    ):
        self.factored = factored^
        self.taus = taus^
        self.block = block

    def r(
        mut self,
    ) raises -> Static[
        Self.dtype, Self.n, Self.n
    ] where Self.dtype.is_floating_point():
        """`R`: the leading `n x n` upper triangle, zeros below it.

        A copy rather than a view, because the storage it comes from also
        holds the reflectors and handing out a view would let a caller
        destroy them.
        """
        var ctx = self.factored.context()
        var out = zeros[Self.dtype, Self.n, Self.n](ctx)
        var fv = self.factored.view()
        var ov = out.view()

        @always_inline
        def upper[w: Int, alignment: Int = 1](coord: Coord) {var fv, var ov}:
            var at = coord_to_index_list(coord)
            if at[0] <= at[1]:
                ov.store[1](coord, fv[Coord(at[0], at[1])])

        elementwise[simd_width=1, target=_target[Self.gpu]()](
            upper, Coord(Self.n, Self.n), ctx
        )
        ctx.synchronize()
        return out^

    def q(
        mut self,
    ) raises -> Static[
        Self.dtype, Self.m, Self.n
    ] where Self.dtype.is_floating_point():
        """The thin `Q`: `m x n` with orthonormal columns, formed
        explicitly. LAPACK's `orgqr`.

        The reflectors are applied to the first `n` columns of the
        identity, last panel first, through the same block reflector the
        factorization's trailing update uses -- so this is three matrix
        products per panel rather than `n` rank-one updates.

        Columns to the left of the panel being applied are skipped, which
        is not just an optimization: `V^T e_j` is zero for `j < k`, so
        they would not change anyway.

        Costs roughly what the factorization did. `apply_q_transpose` is
        the answer when `Q` is wanted only for its action, and `solve`
        never forms it at all.
        """
        var ctx = self.factored.context()
        var out = zeros[Self.dtype, Self.m, Self.n](ctx)
        var ov = out.view()

        @always_inline
        def identity[w: Int, alignment: Int = 1](coord: Coord) {var ov}:
            var at = coord_to_index_list(coord)
            if at[0] == at[1]:
                ov.store[1](coord, Scalar[Self.dtype](1))

        elementwise[simd_width=1, target=_target[Self.gpu]()](
            identity, Coord(Self.m, Self.n), ctx
        )

        var work = _ReflectorWork[Self.dtype](Self.m, Self.n, self.block, ctx)
        var steps = (Self.n + self.block - 1) // self.block
        for step in range(steps):
            var k = (steps - 1 - step) * self.block
            var nb = min(self.block, Self.n - k)
            _apply_block_reflector[transposed=False, gpu=Self.gpu](
                self.factored.view(),
                self.taus.view(),
                ov,
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
        return out^

    def apply_q_transpose[
        rhs: Int
    ](mut self, mut b: Static[Self.dtype, Self.m, rhs]) raises -> Static[
        Self.dtype, Self.m, rhs
    ] where Self.dtype.is_floating_point():
        """`Q^T @ B`, without forming `Q`.

        The reflectors in factorization order, each panel through the
        block reflector. `O(m n rhs)` against the `O(m n^2)` that writing
        `Q` down would cost before the product even started, which is why
        a least-squares solve goes through here.

        The result is `m x rhs`, of which the leading `n` rows are what a
        thin `Q^T B` means; the rows below them are the part of `B` that
        the discarded columns of the full `Q` see.
        """
        var ctx = self.factored.context()
        var out = Static[Self.dtype, Self.m, rhs](ctx)
        var ov = out.view()
        pack_block[target=_target[Self.gpu]()](
            b.view(), ov, 0, 0, Self.m, rhs, ctx
        )

        var work = _ReflectorWork[Self.dtype](Self.m, rhs, self.block, ctx)
        var steps = (Self.n + self.block - 1) // self.block
        for step in range(steps):
            var k = step * self.block
            var nb = min(self.block, Self.n - k)
            _apply_block_reflector[transposed=True, gpu=Self.gpu](
                self.factored.view(),
                self.taus.view(),
                ov,
                k,
                nb,
                Self.m,
                Self.m - k,
                rhs,
                k,
                0,
                work,
                ctx,
            )

        ctx.synchronize()
        return out^

    def solve[
        block: Int = 16
    ](mut self, mut b: Static[Self.dtype, Self.m]) raises -> Static[
        Self.dtype, Self.n
    ] where Self.dtype.is_floating_point():
        """The least-squares solution of `A @ x ~= b`, reusing this
        factorization. `scipy.linalg.lstsq`'s first return value.

        `Q^T b` and then a back substitution against `R`, so `Q` is never
        formed -- the same algorithm the `Array` overload of `lstsq` runs,
        with the reflectors applied a panel at a time through MAX's GEMM
        instead of one at a time by hand.

        Rank deficiency is not detected, the same limit `lstsq` documents:
        a dependent column leaves a near-zero diagonal in `R`, and the
        substitution's floor keeps the answer finite rather than correct.
        """
        var ctx = self.factored.context()
        var wide = zeros[Self.dtype, Self.m, 1](ctx)
        var wv = wide.view()
        var bv = b.view()

        @always_inline
        def widen[w: Int, alignment: Int = 1](coord: Coord) {var bv, var wv}:
            var i = coord_to_index_list(coord)[0]
            wv.store[1](Coord(i, 0), bv[Coord(i)])

        elementwise[simd_width=1, target=_target[Self.gpu]()](
            widen, Coord(Self.m), ctx
        )

        var projected = self.apply_q_transpose[1](wide)
        var head = zeros[Self.dtype, Self.n](ctx)
        var hv = head.view()
        var pv = projected.view()

        @always_inline
        def narrow[w: Int, alignment: Int = 1](coord: Coord) {var hv, var pv}:
            var i = coord_to_index_list(coord)[0]
            hv.store[1](Coord(i), pv[Coord(i, 0)])

        elementwise[simd_width=1, target=_target[Self.gpu]()](
            narrow, Coord(Self.n), ctx
        )
        ctx.synchronize()

        var upper = self.r()
        return solve_triangular[
            Self.dtype, Self.n, True, False, False, Self.gpu, block
        ](upper, head)


def qr_factor[
    dtype: DType, m: Int, n: Int, gpu: Bool = False, block: Int = 16
](mut a: Static[dtype, m, n]) raises -> TensorQR[dtype, m, n, gpu] where (
    dtype.is_floating_point() and m >= n
):
    """**Tier 2.** Blocked Householder QR of an `m x n` matrix with
    `m >= n`, device-resident. LAPACK's `geqrf`.

    Right-looking and blocked, the shape `cholesky` and `lu_factor`
    already have: `geqr2_panel` factors a `block`-wide panel of reflectors
    in place, then the panel's whole block reflector is applied to
    everything to its right at once as `C -= V (T^T (V^T C))`. That is
    three matrix products, so the `O(m n^2)` is MAX's `matmul` and the
    only thing numax runs by hand is the `O(m * block^2)` panel.

    Returns a factorization rather than `(R, Q)`, which is what
    `TensorQR` is for: `.r()` and `.q()` materialize either factor, and
    `.solve` and `.apply_q_transpose` skip forming `Q` altogether. There
    is no `Tensor` overload of `qr` returning both factors as a tuple, and
    that is a Mojo limit rather than a choice -- a `Tuple` of two
    `Tensor`s cannot be destructured, because `Tensor` is `Movable` and
    not `Copyable` while tuple unpacking demands `ImplicitlyCopyable`, so
    a caller would receive a pair it could not take apart. The `Array`
    overload of `qr` returns a tuple because `Array` copies.

    **MAX's own QR is denied, and it is the only factorization MAX has.**
    `linalg.qr_factorization` is `LayoutTensor`-only, and numax's interop
    is `TileTensor`-only so that the library has one owning tensor type
    and one view type. Three other things would argue against it anyway:
    it is monomorphic in `dtype`, it is a CPU-only scalar-loop reference
    rather than a tuned kernel, and its `apply_q`/`form_q` companions are
    `LayoutTensor` too.

    **Ceiling.** The panel is one thread block, as in `lu_factor`, and
    here it bites harder: a QR panel is `O(m * block^2)` because every
    reflector touches every row below it, so `block` wants to be small
    while the trailing update wants it large. The upgrade is the same one
    -- a recursive panel -- and nothing above it changes when it lands.

    Rank deficiency is not detected. A dependent column gives a zero
    reflector (`tau = 0`, correctly the identity) and a zero on `R`'s
    diagonal; `cond` on the original matrix is the check for it.
    """
    var ctx = a.context()
    var factored = Static[dtype, m, n](ctx)
    var taus = zeros[dtype, n](ctx)
    var scratch = zeros[dtype, _PANEL_THREADS + 1](ctx)

    var fv = factored.view()
    var tv = taus.view()
    var sv = scratch.view()

    pack_block[target=_target[gpu]()](a.view(), fv, 0, 0, m, n, ctx)

    var work = _ReflectorWork[dtype](m, n, block, ctx)
    var k = 0
    while k < n:
        var nb = min(block, n - k)

        comptime if gpu:
            ctx.enqueue_function[
                geqr2_panel[
                    dtype,
                    ALayout=type_of(fv).LayoutType,
                    TauLayout=type_of(tv).LayoutType,
                    SLayout=type_of(sv).LayoutType,
                    gpu=True,
                ]
            ](
                fv,
                tv,
                sv,
                Int32(k),
                Int32(nb),
                Int32(m),
                grid_dim=1,
                block_dim=_PANEL_THREADS,
            )
        else:
            geqr2_panel(fv, tv, sv, Int32(k), Int32(nb), Int32(m))

        _apply_block_reflector[transposed=True, gpu=gpu](
            fv,
            tv,
            fv,
            k,
            nb,
            m,
            m - k,
            n - k - nb,
            k,
            k + nb,
            work,
            ctx,
        )

        k += nb

    ctx.synchronize()
    return TensorQR[dtype, m, n, gpu](factored^, taus^, block)
