"""Axis reductions delegated to MAX's `algorithm.rowwise` scaffolder.

**Tier 2 in shape, both targets in fact.** These are `Plain`-only --
`TileTensor` holds raw `dtype` lanes -- but unlike the rest of tier 2 they
are not host-only: one body serves CPU and GPU, chosen by `target`.

MAX's top-level `algorithm` root is a reduction library: `reduce_op` supplies
the monoids (`ReduceSum`, `ReduceMax`, `ReduceMin`, `ReduceProduct`,
`MinMax`, `ArgMax`, `ArgMin`, `Welford`, `OnlineLogSumExp`) and `rowwise`
supplies the scaffolder that drives one across an axis. The scaffolder's
contract is precisely the problem `numax.core.tensor` solves by hand with its
`gpu: Bool` parameter, solved upstream: the body never branches on `target`,
because `rowwise.reduce`/`pjoin`/`once`/`simd` comptime-dispatch on
`params.target`, so GPU primitives never appear in CPU codegen and vice
versa. On CPU the backend parallelizes over output rows; on GPU it picks a
warp, block, tiled or split-K tier from the shape.

**How this relates to `numax.core.tensor.reduce_axis`.** That one takes an
arbitrary `combine` function and folds scalar, left to right. This one takes
one of MAX's monoids. Both ship, because they answer different questions:

- Every reduction numax actually performs -- sums, maxima, minima, products
  -- is one of MAX's monoids, and wants MAX's threading, SIMD width and GPU
  tiering. That is this module.
- An arbitrary `combine` cannot be routed here at all. MAX's monoid is a
  struct instantiated at the tier's SIMD width inside the body, so the
  monoid *family* would have to be the parameter, and Mojo has no way to pass
  one. MAX has the same constraint and answers it the same way: one entry
  point per monoid, each naming its own. So `reduce_axis[combine]` stays for
  folds outside the closed set, at the cost of being serial and scalar.

The two agree on output layout: `dst` is indexed through its coalesced view,
holding the surviving axes in row-major order, so a call here can be checked
directly against the same reduction spelled with `combine`.

A monoid reduction is reassociated by construction -- MAX folds SIMD tiles
and joins partials across threads or lanes -- so a floating-point sum here
will not match `reduce_axis[add_combine]`'s strict left-to-right total in the
last bits. That is the usual accuracy-for-bandwidth trade and the reason
`reduce_axis` keeps its documented left-to-right order rather than being
rewritten in terms of this.
"""

from algorithm import rowwise
from algorithm.reduce_op import ReduceMax, ReduceMin, ReduceProduct, ReduceSum
from layout import Coord, TileTensor
from layout.tile_layout import TensorLayout
from layout.tile_tensor import PointerStorage
from max.gpu.host import DeviceContext
from std.utils import IndexList


def sum_axis[
    dtype: DType,
    XsLayout: TensorLayout,
    OutLayout: TensorLayout,
    axis: Int,
    target: StaticString = "cpu",
](
    xs: TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ],
    dst: TileTensor[
        dtype, OutLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ],
    ctx: Optional[DeviceContext] = None,
) raises where (
    TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and TileTensor[
        dtype, OutLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, OutLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and axis >= 0
    and axis
    < TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].rank
):
    """Sum `xs` along `axis` into `dst`, on either target.

    `numpy.sum(a, axis=k)`, and `numax.core.tensor.reduce_axis` with
    `add_combine` -- but through MAX's `ReduceSum` monoid and `rowwise`
    scaffolder rather than a scalar loop, so it is threaded on CPU and
    tiered on GPU. `ctx` is required when `target="gpu"` and unused
    otherwise.

    The sum is reassociated, so it will differ from
    `reduce_axis[add_combine]` in the last bits. See the module docstring.
    """
    comptime rank = type_of(xs).rank
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[dtype, 1], target, 64, dtype
    ]()

    var dims = IndexList[rank]()
    comptime for d in range(rank):
        dims[d] = Int(xs.dim[d]())
    var axis_size = Int(xs.dim[axis]())
    var src = xs
    var out_flat = dst.coalesce()

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut c: rowwise.Context[params]) {
        var axis_size, var src, var out_flat, var dims
    }:
        @always_inline
        def load[
            width: Int, alignment: Int, coord_rank: Int
        ](idx: IndexList[coord_rank]) {var src} -> SIMD[dtype, width]:
            return src.load[width](Coord(idx))

        var row = rowwise.Row[
            params, dtype, dtype, axis, rank, is_cached=False
        ](row_coords, axis_size, c, load)

        @always_inline
        def contribute[
            w: Int
        ](tile: SIMD[dtype, w], idx: IndexList[rank]) {} -> SIMD[dtype, w]:
            return tile

        var acc = row.reduce[ReduceSum[dtype, params.simd_width]](
            contribute, load
        ).acc

        @always_inline
        def write(oc: IndexList[rank]) {var acc, var out_flat, var dims}:
            out_flat.store[params.emit_tile_width](
                Coord(_collapsed[rank, axis](oc, dims)),
                acc.slice[params.emit_tile_width](),
            )

        row.emit(write)

    rowwise.launch[
        axis=axis,
        simd_width=simd_width,
        target=target,
        num_phases=1,
        associative=True,
    ](body, Coord(dims), ctx)


def max_axis[
    dtype: DType,
    XsLayout: TensorLayout,
    OutLayout: TensorLayout,
    axis: Int,
    target: StaticString = "cpu",
](
    xs: TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ],
    dst: TileTensor[
        dtype, OutLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ],
    ctx: Optional[DeviceContext] = None,
) raises where (
    TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and TileTensor[
        dtype, OutLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, OutLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and axis >= 0
    and axis
    < TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].rank
):
    """The largest element of `xs` along `axis`, into `dst`, on either target.

    `numpy.max(a, axis=k)`. MAX's `ReduceMax` monoid under the same
    scaffolder as `sum_axis`; unlike the sum, a maximum is exact whatever
    order it is folded in, so this agrees with
    `reduce_axis[max_combine]` bit for bit.
    """
    comptime rank = type_of(xs).rank
    comptime simd_width = rowwise.pick_simd_width[
        ReduceMax[dtype, 1], target, 64, dtype
    ]()

    var dims = IndexList[rank]()
    comptime for d in range(rank):
        dims[d] = Int(xs.dim[d]())
    var axis_size = Int(xs.dim[axis]())
    var src = xs
    var out_flat = dst.coalesce()

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut c: rowwise.Context[params]) {
        var axis_size, var src, var out_flat, var dims
    }:
        @always_inline
        def load[
            width: Int, alignment: Int, coord_rank: Int
        ](idx: IndexList[coord_rank]) {var src} -> SIMD[dtype, width]:
            return src.load[width](Coord(idx))

        var row = rowwise.Row[
            params, dtype, dtype, axis, rank, is_cached=False
        ](row_coords, axis_size, c, load)

        @always_inline
        def contribute[
            w: Int
        ](tile: SIMD[dtype, w], idx: IndexList[rank]) {} -> SIMD[dtype, w]:
            return tile

        var best = row.reduce[ReduceMax[dtype, params.simd_width]](
            contribute, load
        ).acc

        @always_inline
        def write(oc: IndexList[rank]) {var best, var out_flat, var dims}:
            out_flat.store[params.emit_tile_width](
                Coord(_collapsed[rank, axis](oc, dims)),
                best.slice[params.emit_tile_width](),
            )

        row.emit(write)

    rowwise.launch[
        axis=axis,
        simd_width=simd_width,
        target=target,
        num_phases=1,
        associative=False,
    ](body, Coord(dims), ctx)


@always_inline
def _collapsed[
    rank: Int, axis: Int
](oc: IndexList[rank], dims: IndexList[rank]) -> Int:
    """The flat index of `oc` with `axis` dropped, row-major.

    `rowwise.emit` hands out the full-rank coordinate with the reduced axis
    pinned to `0`; `dst` holds only the surviving axes. This is the same
    `o * inner + i` position `numax.core.tensor.reduce_axis` writes, which
    is what lets the two be checked against each other.
    """
    var flat = 0
    comptime for d in range(rank):
        comptime if d != axis:
            flat = flat * dims[d] + oc[d]
    return flat
