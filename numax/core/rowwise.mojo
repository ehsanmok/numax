"""Whole-tensor and axis reductions delegated to MAX's `algorithm.rowwise`
scaffolder.

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

**What is here.** `reduce_all` folds a rank-1 view of a whole tensor to one
value under a named monoid, with `rowwise`'s per-tile transform in between
the load and the fold -- which is where `dot`'s multiply, `nrm2`'s square
and `asum`'s magnitude belong, so a fused reduction reads its input once.
`argmax_all`/`argmin_all` are the same traversal through `ArgMax`/`ArgMin`,
returning the winning *index*. `sum_axis`, `prod_axis`, `max_axis` and
`min_axis` fold one axis and are wrappers over a single `_fold_axis`.

The monoid is a `StaticString` parameter rather than a type, because MAX's
monoid is a struct instantiated at the tier's SIMD width *inside* the body:
the monoid family would have to be the parameter and Mojo has no way to
pass one. A `StaticString` also cannot be constrained in a `where` clause,
so every dispatch here is a `comptime if` chain whose `else` is
`comptime assert False` -- an unknown monoid is a compile-time error, not a
quietly chosen default.

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

The two agree on output layout: `dst` receives the surviving axes in row-major
order, so a call here can be checked directly against the same reduction
spelled with `combine`.

That is a **contract on the destination, not a constraint on its layout**.
These write `dst` through a rank-1 view built over its own pointer, at a flat
row-major index, so its layout is consulted for nothing but its element count.
Two reasons, and the second is why there is no `is_row_major` clause on the
output the way `reduce_axis` has one: `coalesce()`, which would respect an
arbitrary layout, requires every extent at compile time, and these accept a
runtime-shaped destination -- `numax.stats` reduces into a `Dynamic` tensor
whose extents are read from the input rather than named, and a predicate over
a layout built at run time has nothing to prove itself against.
`numax.core.tensor`'s runtime `map` overload flattens the same way for the
same reason. Pass a row-major destination; a strided one is filled densely.

A monoid reduction is reassociated by construction -- MAX folds SIMD tiles
and joins partials across threads or lanes -- so a floating-point sum or
product here will not match `reduce_axis[add_combine]`'s strict
left-to-right total in the last bits. That is the usual
accuracy-for-bandwidth trade and the reason `reduce_axis` keeps its
documented left-to-right order rather than being rewritten in terms of
this. `max`, `min`, `argmax` and `argmin` are order-independent and agree
bit for bit; only `sum` and `prod` move. Callers that used to fold on the
host -- `numax.stats`' `sum`/`prod`/`min`/`max`/`argmax`/`argmin`,
`numax.linalg`' `dot`/`nrm2`/`asum` -- inherit that shift, and their own
docstrings say so.
"""

from algorithm import rowwise
from algorithm.reduce_op import (
    ArgMax,
    ArgMin,
    ReduceMax,
    ReduceMin,
    ReduceProduct,
    ReduceSum,
    Welford,
)
from layout import Coord, TileTensor
from layout.tile_layout import row_major, TensorLayout
from layout.tile_tensor import PointerStorage
from max.gpu.host import DeviceContext
from std.utils import IndexList

from .array import Static


@always_inline
def _monoid_width[
    dtype: DType, monoid: StaticString, target: StaticString
]() -> Int:
    """`pick_simd_width` for the monoid `monoid` names.

    The width is the monoid's to choose -- `ArgMax` wants a narrow one
    because it carries an int64 index beside every value, `ReduceSum` wants
    the target's native one -- so the choice cannot be hoisted out of the
    dispatch. Instantiated at width 1, which is what `pick_simd_width`
    inspects.
    """
    comptime if monoid == "sum":
        return rowwise.pick_simd_width[ReduceSum[dtype, 1], target, 64, dtype]()
    elif monoid == "prod":
        return rowwise.pick_simd_width[
            ReduceProduct[dtype, 1], target, 64, dtype
        ]()
    elif monoid == "max":
        return rowwise.pick_simd_width[ReduceMax[dtype, 1], target, 64, dtype]()
    elif monoid == "min":
        return rowwise.pick_simd_width[ReduceMin[dtype, 1], target, 64, dtype]()
    else:
        comptime assert (
            False
        ), "numax.core.rowwise: unknown monoid; expected sum, prod, max or min"


comptime _ASSOCIATIVE[monoid: StaticString] = monoid == "sum"
"""Whether `rowwise.launch` may widen the CPU accumulator on a long row.

Only the sum. MAX's own rule: the widening reorders the accumulation, which
is free for an additive fold and not for `prod`'s rounding, so `prod`, `max`
and `min` run the narrow chain. It is ignored on GPU.
"""


def reduce_all[
    dtype: DType,
    Contribute: (def[w: Int](SIMD[dtype, w], IndexList[1]) -> SIMD[dtype, w])
    & RegisterPassable
    & ImplicitlyCopyable,
    //,
    monoid: StaticString,
    target: StaticString = "cpu",
](
    xs: TileTensor[
        dtype,
        _,
        _,
        Storage=PointerStorage[element_width=1],
        linear_idx_type=_,
    ],
    dst: TileTensor[
        dtype,
        _,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
        linear_idx_type=_,
    ],
    contribute: Contribute,
    n: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Fold a rank-1 `xs` of `n` elements to `dst[0]` under `monoid`.

    The whole-tensor reduction, on either target: `monoid` is `"sum"`,
    `"prod"`, `"max"` or `"min"`. A `StaticString` parameter cannot be
    constrained in a `where` clause, so the dispatch is a `comptime if`
    chain ending in `comptime assert False` -- an unrecognized monoid is a
    compile-time error, never a silently chosen default. `contribute` is `rowwise`'s per-tile transform,
    which sits between the load and the fold -- the identity for a plain
    `numpy.sum(a)`, `dot`'s multiply, `nrm2`'s square, `asum`'s magnitude --
    so a fused reduction reads its row once and never materializes the
    transformed vector.

    `n` is a run-time argument rather than a parameter, so one instantiation
    serves a comptime-shaped and a run-time-shaped input alike; `xs` is
    whatever rank-1 view the caller flattened to. `ctx` is required when
    `target="gpu"` and unused otherwise. The destination is read at
    `Coord(0)`, so a one-element rank-1 tensor is what it wants.

    `"sum"` and `"prod"` are reassociated -- MAX folds SIMD tiles and joins
    partials across threads or lanes -- so they will differ in the last bits
    from a strict left-to-right loop such as `numax.core.tensor.reduce`.
    `"max"` and `"min"` are exact in any order and agree bit for bit.

    `dtype` and `Contribute` are inferred from the arguments, so a call
    names only what it chooses: `reduce_all[monoid="sum"](...)`, with
    `target=` added for a device.
    """
    comptime simd_width = _monoid_width[dtype, monoid, target]()
    var src = xs
    var out = dst

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut c: rowwise.Context[params]) {
        var src, var out, var contribute, var n
    }:
        @always_inline
        def load[
            width: Int, alignment: Int, coord_rank: Int
        ](idx: IndexList[coord_rank]) {var src} -> SIMD[dtype, width]:
            return src.load[width](Coord(idx))

        var row = rowwise.Row[params, dtype, dtype, 0, 1, is_cached=False](
            row_coords, n, c, load
        )

        var acc: SIMD[dtype, params.simd_width]
        comptime if monoid == "sum":
            acc = row.reduce[ReduceSum[dtype, params.simd_width]](
                contribute, load
            ).acc
        elif monoid == "prod":
            acc = row.reduce[ReduceProduct[dtype, params.simd_width]](
                contribute, load
            ).acc
        elif monoid == "max":
            acc = row.reduce[ReduceMax[dtype, params.simd_width]](
                contribute, load
            ).acc
        elif monoid == "min":
            acc = row.reduce[ReduceMin[dtype, params.simd_width]](
                contribute, load
            ).acc
        else:
            comptime assert False, (
                "numax.core.rowwise: unknown monoid; expected sum, prod, max"
                " or min"
            )

        @always_inline
        def write(oc: IndexList[1]) {var acc, var out}:
            out.store[params.emit_tile_width](
                Coord(0), acc.slice[params.emit_tile_width]()
            )

        row.emit(write)

    rowwise.launch[
        axis=0,
        simd_width=simd_width,
        target=target,
        num_phases=1,
        associative=_ASSOCIATIVE[monoid],
    ](body, Coord(IndexList[1](n)), ctx)


def _argn_all[
    dtype: DType, largest: Bool, target: StaticString
](
    xs: TileTensor[
        dtype,
        _,
        _,
        Storage=PointerStorage[element_width=1],
        linear_idx_type=_,
    ],
    ctx: DeviceContext,
) raises -> Int:
    """The index of the extremum of a rank-1 `xs`, on either target.

    MAX's `ArgMax`/`ArgMin` monoids, which carry the winning index beside
    the winning value so the position comes back from the same traversal
    the comparison already does. Ties break to the lower index, NumPy's
    rule.

    The winning index is read from **`acc_indices`**, not the state's scalar
    `best_idx`: `join_parallel` leaves it in `acc_indices[0]` precisely so a
    body's `emit` reads it the same way on the cooperative tier (lane 0) and
    the tiled tier (one lane per output column, where `join_parallel` is
    skipped and `best_idx` is never filled). That is what MAX's own
    `algorithm.reductions.reduce_argmax` reads.
    """
    comptime simd_width = rowwise.pick_simd_width[
        ArgMax[dtype, 1], target, 64, dtype
    ]() if largest else rowwise.pick_simd_width[
        ArgMin[dtype, 1], target, 64, dtype
    ]()
    var n = Int(xs.dim[0]())
    var src = xs
    var winner = Static[DType.int64, 1](ctx)
    var out = winner.view()

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut c: rowwise.Context[params]) {
        var src, var out, var n
    }:
        @always_inline
        def load[
            width: Int, alignment: Int, coord_rank: Int
        ](idx: IndexList[coord_rank]) {var src} -> SIMD[dtype, width]:
            return src.load[width](Coord(idx))

        var row = rowwise.Row[params, dtype, dtype, 0, 1, is_cached=False](
            row_coords, n, c, load
        )

        @always_inline
        def value[
            w: Int
        ](tile: SIMD[dtype, w], idx: IndexList[1]) {} -> SIMD[dtype, w]:
            return tile

        var indices: SIMD[DType.int64, params.simd_width]
        comptime if largest:
            indices = row.reduce[ArgMax[dtype, params.simd_width]](
                value, load
            ).acc_indices
        else:
            indices = row.reduce[ArgMin[dtype, params.simd_width]](
                value, load
            ).acc_indices

        @always_inline
        def write(oc: IndexList[1]) {var indices, var out}:
            out.store[params.emit_tile_width](
                Coord(0), indices.slice[params.emit_tile_width]()
            )

        row.emit(write)

    rowwise.launch[
        axis=0,
        simd_width=simd_width,
        target=target,
        num_phases=1,
        associative=False,
    ](body, Coord(IndexList[1](n)), Optional(ctx))
    return Int(winner.to_host()[0])


def argmax_all[
    dtype: DType, target: StaticString = "cpu"
](
    xs: TileTensor[
        dtype,
        _,
        _,
        Storage=PointerStorage[element_width=1],
        linear_idx_type=_,
    ],
    ctx: DeviceContext,
) raises -> Int:
    """The index of the largest element of a rank-1 `xs`. `numpy.argmax(a)`.

    MAX's `ArgMax` monoid under the same scaffolder as `reduce_all`, so the
    input is never downloaded -- only the index crosses back. The first
    index wins a tie, and a NaN candidate is skipped rather than taken,
    which is what `nn.argmaxmin` does too. `ctx` allocates the one-element
    index destination, so it is required on both targets.
    """
    return _argn_all[dtype, True, target](xs, ctx)


def argmin_all[
    dtype: DType, target: StaticString = "cpu"
](
    xs: TileTensor[
        dtype,
        _,
        _,
        Storage=PointerStorage[element_width=1],
        linear_idx_type=_,
    ],
    ctx: DeviceContext,
) raises -> Int:
    """The index of the smallest element of a rank-1 `xs`.
    `numpy.argmin(a)`, `argmax_all`'s mirror through `ArgMin`."""
    return _argn_all[dtype, False, target](xs, ctx)


def _fold_axis[
    dtype: DType,
    XsLayout: TensorLayout,
    OutLayout: TensorLayout,
    axis: Int,
    monoid: StaticString,
    target: StaticString,
](
    xs: TileTensor[dtype, XsLayout, _, Storage=PointerStorage[element_width=1]],
    dst: TileTensor[
        dtype, OutLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ],
    ctx: Optional[DeviceContext] = None,
) raises where (
    TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and axis >= 0
    and axis
    < TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].rank
):
    """Fold `xs` along `axis` into `dst` under `monoid`, on either target.

    The body the four public axis reductions share: `sum_axis`,
    `prod_axis`, `max_axis` and `min_axis` differ only in which of MAX's
    monoids they name and whether the CPU accumulator may be widened, so
    they are wrappers and this is the one place the scaffolding is written.
    An unrecognized `monoid` is a compile-time error, as in `reduce_all`.
    """
    comptime rank = type_of(xs).rank
    comptime simd_width = _monoid_width[dtype, monoid, target]()

    var dims = IndexList[rank]()
    comptime for d in range(rank):
        dims[d] = Int(xs.dim[d]())
    var axis_size = Int(xs.dim[axis]())
    var src = xs
    var out_flat = TileTensor(
        dst.ptr_at_offset(Coord(0)), row_major(Coord(dst.num_elements()))
    )

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

        var acc: SIMD[dtype, params.simd_width]
        comptime if monoid == "sum":
            acc = row.reduce[ReduceSum[dtype, params.simd_width]](
                contribute, load
            ).acc
        elif monoid == "prod":
            acc = row.reduce[ReduceProduct[dtype, params.simd_width]](
                contribute, load
            ).acc
        elif monoid == "max":
            acc = row.reduce[ReduceMax[dtype, params.simd_width]](
                contribute, load
            ).acc
        elif monoid == "min":
            acc = row.reduce[ReduceMin[dtype, params.simd_width]](
                contribute, load
            ).acc
        else:
            comptime assert False, (
                "numax.core.rowwise: unknown monoid; expected sum, prod, max"
                " or min"
            )

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
        associative=_ASSOCIATIVE[monoid],
    ](body, Coord(dims), ctx)


def sum_axis[
    dtype: DType,
    XsLayout: TensorLayout,
    OutLayout: TensorLayout,
    axis: Int,
    target: StaticString = "cpu",
](
    xs: TileTensor[dtype, XsLayout, _, Storage=PointerStorage[element_width=1]],
    dst: TileTensor[
        dtype, OutLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ],
    ctx: Optional[DeviceContext] = None,
) raises where (
    TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
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
    _fold_axis[axis=axis, monoid="sum", target=target](xs, dst, ctx)


def prod_axis[
    dtype: DType,
    XsLayout: TensorLayout,
    OutLayout: TensorLayout,
    axis: Int,
    target: StaticString = "cpu",
](
    xs: TileTensor[dtype, XsLayout, _, Storage=PointerStorage[element_width=1]],
    dst: TileTensor[
        dtype, OutLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ],
    ctx: Optional[DeviceContext] = None,
) raises where (
    TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and axis >= 0
    and axis
    < TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].rank
):
    """Multiply `xs` along `axis` into `dst`, on either target.

    `numpy.prod(a, axis=k)`, through MAX's `ReduceProduct` monoid. Like
    the sum it is reassociated and will differ from
    `reduce_axis[mul_combine]` in the last bits; unlike the sum the CPU
    accumulator is not widened, because a product's rounding depends on
    the order the factors arrive in.
    """
    _fold_axis[axis=axis, monoid="prod", target=target](xs, dst, ctx)


def max_axis[
    dtype: DType,
    XsLayout: TensorLayout,
    OutLayout: TensorLayout,
    axis: Int,
    target: StaticString = "cpu",
](
    xs: TileTensor[dtype, XsLayout, _, Storage=PointerStorage[element_width=1]],
    dst: TileTensor[
        dtype, OutLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ],
    ctx: Optional[DeviceContext] = None,
) raises where (
    TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
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
    _fold_axis[axis=axis, monoid="max", target=target](xs, dst, ctx)


def min_axis[
    dtype: DType,
    XsLayout: TensorLayout,
    OutLayout: TensorLayout,
    axis: Int,
    target: StaticString = "cpu",
](
    xs: TileTensor[dtype, XsLayout, _, Storage=PointerStorage[element_width=1]],
    dst: TileTensor[
        dtype, OutLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ],
    ctx: Optional[DeviceContext] = None,
) raises where (
    TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and axis >= 0
    and axis
    < TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].rank
):
    """The smallest element of `xs` along `axis`, into `dst`, on either
    target. `numpy.min(a, axis=k)`, `max_axis`'s mirror through
    `ReduceMin`, and exact for the same reason."""
    _fold_axis[axis=axis, monoid="min", target=target](xs, dst, ctx)


def mean_variance_axis[
    dtype: DType,
    XsLayout: TensorLayout,
    OutLayout: TensorLayout,
    axis: Int,
    target: StaticString = "cpu",
](
    xs: TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ],
    means: TileTensor[
        dtype, OutLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ],
    variances: TileTensor[
        dtype, OutLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ],
    ddof: Int = 0,
    ctx: Optional[DeviceContext] = None,
) raises where (
    dtype.is_floating_point()
    and TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and axis >= 0
    and axis
    < TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].rank
):
    """Mean and variance along `axis` in one pass, on either target.

    `numpy.mean(a, axis=k)` and `numpy.var(a, axis=k, ddof=ddof)` together,
    through MAX's `Welford` monoid: a single traversal carrying
    `{count, mean, M2}`, combined across SIMD lanes and threads by Chan's
    formula. `ddof` is subtracted from the divisor, so `ddof=1` gives the
    sample variance.

    Both statistics come back from one call because the monoid computes
    both and separating them would mean reducing twice. A caller wanting
    only the mean still pays one pass; it just also gets the variance.

    Numerically this is the point of the monoid rather than an incidental
    benefit. The textbook two-pass form needs the mean before it can
    accumulate deviations, so the naive one-pass alternative --
    `E[x^2] - E[x]^2` -- cancels catastrophically when the mean is large
    relative to the spread. Welford never forms `E[x^2]`, so a variance of
    order one is recoverable from values of order `1e8`, where the
    subtract-of-squares form returns noise or a negative number. That is
    also why the result will not match a two-pass host computation in the
    last bits: this one is the more accurate of the two.
    """
    comptime rank = type_of(xs).rank
    comptime simd_width = rowwise.pick_simd_width[
        Welford[dtype, 1], target, 64, dtype
    ]()

    var dims = IndexList[rank]()
    comptime for d in range(rank):
        dims[d] = Int(xs.dim[d]())
    var axis_size = Int(xs.dim[axis]())
    var src = xs
    var mean_flat = TileTensor(
        means.ptr_at_offset(Coord(0)), row_major(Coord(means.num_elements()))
    )
    var var_flat = TileTensor(
        variances.ptr_at_offset(Coord(0)),
        row_major(Coord(variances.num_elements())),
    )
    var divisor = Scalar[dtype](axis_size - ddof)

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut c: rowwise.Context[params]) {
        var axis_size,
        var src,
        var mean_flat,
        var var_flat,
        var dims,
        var divisor,
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

        var state = row.reduce[Welford[dtype, params.simd_width]](
            contribute, load
        )

        @always_inline
        def write(
            oc: IndexList[rank],
        ) {var state, var mean_flat, var var_flat, var dims, var divisor}:
            comptime w = params.emit_tile_width
            var at = Coord(_collapsed[rank, axis](oc, dims))
            mean_flat.store[w](at, state.mean.slice[w]())
            var_flat.store[w](at, state.M2.slice[w]() / divisor)

        row.emit(write)

    rowwise.launch[
        axis=axis,
        simd_width=simd_width,
        target=target,
        num_phases=1,
        associative=True,
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
