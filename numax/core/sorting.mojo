"""Sorting, searching and counting over `numax.core.array.Tensor`.

**This module is tier 2.** A comparison sort runs a data-dependent number
of comparisons and branches per element; `searchsorted` halves an interval
based on a comparison; `unique` produces an output whose *length* depends on
the input values. None of that can appear in a `FloatLike`-generic kernel --
a `Self` may hold a SIMD vector whose lanes disagree about which branch they
want, and there is no per-lane `select` on the trait. So everything here is
`Plain`-only, and none of it is launchable *inside* a kernel body the way
the `FloatLike` tier is. See `docs/architecture.md`'s "Two tiers".

That is a statement about numax's own comparison walks, which run on a host
copy. It is not a claim that nothing here reaches a device: `top_k` is a
whole delegation to `nn.top_k`, which ships a real GPU kernel, so it takes
the `gpu: Bool` parameter the rest of the delegating surface takes and
leaves the data where it lives.

That restriction was previously stated as a blanket exclusion: sorting was
"not absorbed", full stop, and `numax.stats.median` reached
`List.sort()` internally with a note that this was stdlib machinery rather
than a numax API. With the tiers written down, the honest position is
narrower and more useful. numax does not write comparison logic *inside the
trait*; a NumPy caller still gets `sort`, `argsort`, `searchsorted` and
`unique` as ordinary tier-2 names.

## MAX-first, and where it runs out

`nn.argsort` exists and is rank-1, index-returning, CPU + GPU. It is the
right thing for a caller already holding a `TileTensor` on a device.
`nn.top_k` exists too, and is the better-shaped of the two: any axis, both
targets, values and indices out together, so `top_k` below is pure
delegation. What neither gives is a *value* sort, an n-dimensional sort,
`searchsorted`, or `unique` -- and `argsort` returns indices into a tensor
rather than a sorted copy.
`std.builtin.sort` (stable, comparator-driven, over a `Span`) is what the
functions here are built on, since these walks run on a host copy of the
tensor's elements (`Tensor.to_host`) and that is
exactly what `sort` wants.

## Results whose length the data decides

`unique`, `extract` and `take` return a run-time-shaped rank-1 tensor, sized
to what they actually produced. That is the whole reason a `Tensor`'s extents
need not be compile-time: these three have no length until the values are
read. `sort` and `select` keep their input's shape, since theirs does not
depend on the values at all.

## Flat, not axis-wise

Every function numax *writes* here treats its input as flat row-major,
matching `numpy.sort(a, axis=None)` rather than the default `axis=-1`.
Axis-wise sorting would need the same `outer`/`length`/`inner` decomposition
`numax.core.tensor.reduce_axis` uses; it is a straightforward extension and is
not written yet, so the flat behavior is stated rather than implied.

`top_k` is the exception, and deliberately: `nn.top_k` takes an axis, so its
rank-2 form works row-wise like `torch.topk(a, k, dim=-1)`. Flattening it
would be numax discarding a capability MAX already has.
"""

from std.builtin.sort import sort as _std_sort

from nn.argsort import argsort as _nn_argsort
from nn.gather_scatter import (
    gather as _nn_gather,
    gather_elements as _nn_gather_elements,
)
from nn.topk import top_k as _max_top_k
from std.collections import Array

from layout import Coord, TileTensor
from layout.tile_layout import TensorLayout, row_major
from .array import (
    Dynamic,
    Static,
    Tensor,
    asarray,
    _dyn_shape,
    _dyn_shape_from,
    _extents_of,
    _product,
    _stretch_strides,
    _strides_of,
    broadcast_shapes,
)


def sort[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Static[
    dtype, LayoutType.static_product
] where LayoutType.all_dims_known:
    """A sorted rank-1 copy of `a`, ascending. `numpy.sort(a, axis=None)`.

    Stable, because `std.builtin.sort` is; for a plain numeric sort that
    is unobservable, but it costs nothing and makes the behavior
    predictable if this later grows a key argument.

    Returns rank-1 regardless of the input's rank, which is what
    `axis=None` means. `numax.core.array.reshape` puts a shape back on if one
    is wanted.

    The overload below takes a tensor whose extents are run-time values
    and returns one, so `sort(extract(mask, a))` works.
    """
    comptime n = LayoutType.static_product
    var values = a.to_host()
    _std_sort(values)
    return Static[dtype, n](a.context(), values^)


def sort[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Dynamic[
    dtype, 1
] where not LayoutType.all_dims_known:
    """A sorted rank-1 copy of `a`, ascending, for a run-time shape.

    Same sort as the overload above; the result's length is a run-time
    value because the input's is.
    """
    var values = a.to_host()
    _std_sort(values)
    return asarray(values^, a.context())


def argsort[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> List[Int]:
    """The flat indices that would sort `a`, ascending.
    `numpy.argsort(a, axis=None)`.

    Routed straight to `nn.argsort`, which is MAX's own sort: rank-1,
    ascending or descending, with a CPU and a GPU implementation behind one
    name. numax has no business writing a second one -- an earlier version
    here open-coded an O(n^2) insertion sort over an index list, which this
    replaces outright.

    Returned as a `List[Int]` rather than a `Tensor`, because an index array
    is not a numeric tensor: nothing downstream wants to run a `FloatLike`
    kernel over it, and giving it a `Static[int64, ...]` would invite exactly
    that. The flattening is the `axis=None` contract every other routine in
    this module follows, and it is also what makes the input rank-1 the way
    `nn.argsort` requires.

    The scratch tensors are built at the input's run-time length rather
    than from its type, so a tensor whose extents are run-time values
    sorts the same way a compile-time-shaped one does.
    """
    var n = a.size()
    var ctx = a.context()
    var flat = asarray(a.to_host(), ctx)
    var indices = Dynamic[DType.int64, 1](ctx, row_major(_dyn_shape[1](n)))
    var flat_view = flat.view()
    var indices_view = indices.view()
    _nn_argsort(indices_view, flat_view)

    # `view()` erases the origin, so `flat` is not kept alive by
    # `flat_view` and destruction is ASAP. See `numax.linalg.qr`.
    _ = flat^

    var order = List[Int](capacity=n)
    var raw = indices.to_host()
    for i in range(n):
        order.append(Int(raw[i]))
    return order^


def searchsorted[
    dtype: DType, n: Int
](sorted_values: Static[dtype, n], value: Scalar[dtype]) raises -> Int:
    """The index where `value` would be inserted to keep `sorted_values`
    ascending. `numpy.searchsorted(a, v, side="left")`.

    Left side: for a `value` equal to an existing element, the index of
    the *first* such element is returned, so inserting there puts the new
    value before its equals. That is NumPy's default and the convention
    that makes `searchsorted` usable for bucketing.

    `sorted_values` is assumed sorted and not checked -- checking would
    cost a full pass, and the function is meaningless on unsorted input in
    a way the caller is better placed to notice.

    Binary search, so `O(log n)` comparisons -- but a data-dependent
    number of them, which is what makes this tier 2 rather than something
    that could live in a kernel.
    """
    var values = sorted_values.to_host()
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = (lo + hi) // 2
        if values[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def searchsorted[
    dtype: DType,
    SortedLayout: TensorLayout,
    QueryLayout: TensorLayout,
    right: Bool = False,
](
    sorted_values: Tensor[dtype, SortedLayout],
    values: Tensor[dtype, QueryLayout],
) raises -> Dynamic[DType.int64, 1]:
    """One insertion index per element of `values`.
    `numpy.searchsorted(a, v)`.

    The vectorized form of the overload above, and the one every bucketing
    caller actually wants: a linear interpolation needs the bracketing
    interval for each of its query points, and a histogram needs the bin
    for each sample. Calling the scalar form in a loop re-copies
    `sorted_values` to the host on every query, which is what this exists
    to stop.

    `right=True` is `numpy.searchsorted(a, v, side="right")`: an element
    equal to an existing one lands *after* its equals rather than before.
    A `Bool` rather than NumPy's `side` string, because a `StaticString`
    parameter cannot be constrained in Mojo 1.0 and a typo would then be a
    run-time error -- `top_k`'s `largest` makes the same trade.

    `sorted_values` is assumed sorted and not checked, as in the scalar
    overload. MAX ships no `searchsorted`, so this is numax's own.
    """
    var haystack = sorted_values.to_host()
    var needles = values.to_host()
    var n = len(haystack)
    var out = List[Scalar[DType.int64]](length=len(needles), fill=0)
    for q in range(len(needles)):
        var value = needles[q]
        var lo = 0
        var hi = n
        while lo < hi:
            var mid = (lo + hi) // 2
            comptime if right:
                if haystack[mid] <= value:
                    lo = mid + 1
                else:
                    hi = mid
            else:
                if haystack[mid] < value:
                    lo = mid + 1
                else:
                    hi = mid
        out[q] = Scalar[DType.int64](lo)
    return Dynamic[DType.int64, 1](
        sorted_values.context(), row_major(_dyn_shape[1](len(needles))), out^
    )


def take[
    dtype: DType,
    LayoutType: TensorLayout,
    IndexLayout: TensorLayout,
    axis: Int,
    gpu: Bool = False,
](
    a: Tensor[dtype, LayoutType], indices: Tensor[DType.int64, IndexLayout]
) raises -> Dynamic[dtype, LayoutType.rank] where (
    axis >= 0 and axis < LayoutType.rank and IndexLayout.rank == 1
):
    """The slices of `a` at `indices` along `axis`.
    `numpy.take(a, indices, axis=k)`.

    `take(a, [2, 0])` at `axis=0` on a `(3, 2)` gives a `(2, 2)` holding
    rows 2 and 0 -- so this reorders, selects and duplicates rows or
    columns, which is the operation `argsort`'s output was missing a
    consumer for at rank > 1.

    Routed to `nn.gather`, which is ONNX `Gather` and takes `axis` as a
    compile-time parameter, a `target` and a `DeviceContext`. That is the
    **tensor** overload of `nn.gather`, not the closure one --
    `docs/parity.md` records why the closure form is unreachable, and this
    one sidesteps it, so `take` carries a real `gpu` parameter.

    Rank-1 `indices` only, which is ONNX `Gather`'s own restriction on the
    shape numax passes; the flat `List[Int]` overload above is the one for
    an already-flat selection.
    """
    comptime rank = LayoutType.rank
    var count = indices.size()
    var length = a.dim_at(axis)
    var index_values = indices.to_host()
    for q in range(count):
        var at = Int(index_values[q])
        if at < 0 or at >= length:
            raise Error(
                "take: index ",
                at,
                " is out of range for an axis of extent ",
                length,
            )

    var in_extents = List[Int](capacity=rank)
    var out_extents = List[Int](capacity=rank)
    var total = 1
    for d in range(rank):
        in_extents.append(a.dim_at(d))
        out_extents.append(count if d == axis else a.dim_at(d))
        total *= out_extents[d]

    var values = a.to_host()
    var out = List[Scalar[dtype]](length=total, fill=0)
    var ctx = a.context()
    _nn_gather[axis=axis, target="gpu" if gpu else "cpu"](
        TileTensor(out, row_major(_dyn_shape_from[rank](out_extents))),
        TileTensor(values, row_major(_dyn_shape_from[rank](in_extents))),
        TileTensor(index_values, row_major(Coord(count))),
        context=ctx,
    )
    return Dynamic[dtype, rank](
        ctx, row_major(_dyn_shape_from[rank](out_extents)), out^
    )


def take_along_axis[
    dtype: DType, LayoutType: TensorLayout, IndexLayout: TensorLayout, axis: Int
](
    a: Tensor[dtype, LayoutType], indices: Tensor[DType.int64, IndexLayout]
) raises -> Dynamic[dtype, LayoutType.rank] where (
    axis >= 0 and axis < LayoutType.rank and IndexLayout.rank == LayoutType.rank
):
    """One element of `a` per entry of `indices`, indexed along `axis`.
    `numpy.take_along_axis`.

    Not `take`: `take` picks whole slices with one index list shared by
    every position, while this picks an element per position, so `indices`
    has `a`'s rank rather than rank 1. That is what makes `argsort`'s
    per-row output usable -- `take_along_axis(a, argsort_rows, axis=1)` is
    each row of `a` sorted.

    Routed to `nn.gather_elements`, which is ONNX `GatherElements` (Torch's
    `gather`) and takes a `DeviceContext`. The result has `indices`'s
    shape, which is that operator's contract and NumPy's too.
    """
    comptime rank = LayoutType.rank
    var length = a.dim_at(axis)
    var index_values = indices.to_host()
    for q in range(len(index_values)):
        var at = Int(index_values[q])
        if at < 0 or at >= length:
            raise Error(
                "take_along_axis: index ",
                at,
                " is out of range for an axis of extent ",
                length,
            )

    var in_extents = List[Int](capacity=rank)
    var out_extents = List[Int](capacity=rank)
    var total = 1
    for d in range(rank):
        in_extents.append(a.dim_at(d))
        out_extents.append(indices.dim_at(d))
        if d != axis and a.dim_at(d) != indices.dim_at(d):
            raise Error(
                "take_along_axis: extents ",
                a.dim_at(d),
                " and ",
                indices.dim_at(d),
                " differ on axis ",
                d,
            )
        total *= out_extents[d]

    var values = a.to_host()
    var out = List[Scalar[dtype]](length=total, fill=0)
    var ctx = a.context()
    _nn_gather_elements(
        TileTensor(values, row_major(_dyn_shape_from[rank](in_extents))),
        TileTensor(index_values, row_major(_dyn_shape_from[rank](out_extents))),
        axis,
        TileTensor(out, row_major(_dyn_shape_from[rank](out_extents))),
        ctx,
    )
    return Dynamic[dtype, rank](
        ctx, row_major(_dyn_shape_from[rank](out_extents)), out^
    )


def unique[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Dynamic[dtype, 1]:
    """The sorted distinct values of `a`. `numpy.unique`.

    Right-sized: the result holds exactly as many elements as there are
    distinct values, which is a count only the data knows. That is what a
    run-time-shaped tensor is for, and `unique(a).size()` is the answer to
    "how many" rather than a second return value the caller has to carry.
    """
    var n = a.size()
    var values = a.to_host()
    _std_sort(values)

    var count = 0
    for i in range(n):
        if count == 0 or values[i] != values[count - 1]:
            values[count] = values[i]
            count += 1
    values.resize(count, fill=0)
    return asarray(values^, a.context())


def count_nonzero[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Int:
    """How many elements of `a` are not zero. `numpy.count_nonzero`.

    `-0.0` counts as zero (it compares equal to `0.0`), matching NumPy.
    NaN counts as nonzero, also matching NumPy, since `nan != 0`.
    """
    var n = a.size()
    var values = a.to_host()
    var total = 0
    for i in range(n):
        if values[i] != 0:
            total += 1
    return total


def any_nonzero[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Bool:
    """Whether any element is nonzero. `numpy.any`.

    Named `any_nonzero` rather than `any` because `any` is a Mojo builtin;
    the same kind of collision that made `numpy.var` into
    `numax.stats.variance`.

    Short-circuits, which is the point of having it rather than
    `count_nonzero(a) > 0`.
    """
    var n = a.size()
    var values = a.to_host()
    for i in range(n):
        if values[i] != 0:
            return True
    return False


def all_nonzero[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Bool:
    """Whether every element is nonzero. `numpy.all`, named for the same
    reason as `any_nonzero`. Short-circuits on the first zero."""
    var n = a.size()
    var values = a.to_host()
    for i in range(n):
        if values[i] == 0:
            return False
    return True


def nonzero[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> List[Int]:
    """The flat indices of the nonzero elements, ascending.
    `numpy.flatnonzero`.

    A `List[Int]` for the same reason `argsort` returns one: these are
    indices, not numbers to compute with. Right-sized, since a `List` can
    be -- which is exactly the freedom `unique`'s `Tensor` return does not
    have.
    """
    var n = a.size()
    var values = a.to_host()
    var indices = List[Int](capacity=n)
    for i in range(n):
        if values[i] != 0:
            indices.append(i)
    return indices^


def argwhere[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Dynamic[DType.int64, 2]:
    """The coordinates of the nonzero elements, one row each.
    `numpy.argwhere`.

    Where `nonzero` returns *flat* indices as a `List[Int]`, this returns a
    `(count, rank)` tensor whose row `i` is the full coordinate of the
    `i`-th nonzero element. That is the form a rank-2 caller needs -- a
    flat index into a `(rows, cols)` has to be divided back out, and doing
    it at the call site is where the stride convention gets mistaken.

    Right-sized: the row count depends on the data, which is what a
    run-time-shaped tensor is for.
    """
    comptime rank = LayoutType.rank
    var extents = List[Int](capacity=rank)
    for d in range(rank):
        extents.append(a.dim_at(d))

    var values = a.to_host()
    var n = len(values)
    var coords = List[Scalar[DType.int64]](capacity=n * rank)
    var count = 0
    for flat in range(n):
        if values[flat] != 0:
            count += 1
            var rem = flat
            var digits = List[Int](length=rank, fill=0)
            for k in range(rank):
                var d = rank - 1 - k
                digits[d] = rem % extents[d]
                rem //= extents[d]
            for d in range(rank):
                coords.append(Scalar[DType.int64](digits[d]))

    var shape = List[Int](capacity=2)
    shape.append(count)
    shape.append(rank)
    return Dynamic[DType.int64, 2](
        a.context(), row_major(_dyn_shape_from[2](shape)), coords^
    )


def put[
    dtype: DType, LayoutType: TensorLayout
](
    mut a: Tensor[dtype, LayoutType],
    indices: List[Int],
    values: List[Scalar[dtype]],
) raises:
    """Write `values` into `a` at the flat `indices`. `numpy.put`.

    In place and returning nothing, unlike everything else in this module,
    because that is what `numpy.put` does and because the alternative -- a
    copy with a few entries changed -- is the expensive spelling of a
    scatter. The consumer for `nonzero`'s and `argsort`'s index lists on
    the writing side, as `take` is on the reading side.

    `values` must be as long as `indices`, or one element long, which
    broadcasts to every index the way `numpy.put` does. Indices are flat
    and row-major, matching `numpy.put` with no `mode`; an out-of-range one
    raises rather than wrapping, since `mode="raise"` is NumPy's default.

    MAX's `nn.scatter_elements` and `nn.scatter_nd` were searched for this
    and neither fits: both want the indices as a tensor shaped like the
    output slice rather than a flat list, so numax would build the very
    thing the caller is trying to avoid.
    """
    var n = a.size()
    if len(values) != len(indices) and len(values) != 1:
        raise Error(
            "put: ",
            len(values),
            " values for ",
            len(indices),
            " indices -- give one value or one per index",
        )

    var current = a.to_host()
    for q in range(len(indices)):
        var at = indices[q]
        if at < 0 or at >= n:
            raise Error(
                "put: index ", at, " is out of range for ", n, " elements"
            )
        current[at] = values[0] if len(values) == 1 else values[q]
    a.copy_from_host(current)


def extract[
    dtype: DType, LayoutType: TensorLayout
](
    condition: Tensor[DType.bool, LayoutType], a: Tensor[dtype, LayoutType]
) raises -> Dynamic[dtype, 1]:
    """The elements of `a` where `condition` is nonzero. `numpy.extract`,
    which is what `a[mask]` means in NumPy.

    Boolean masking: the result's length depends on the mask's *values*, so
    it comes back run-time-shaped and right-sized.

    `condition` is a bool tensor over the same layout -- the type every
    comparison in `numax.core.logic` already returns, so `extract(a > 0, a)`
    composes without a conversion in between. A tensor of values becomes a
    mask with `numax.core.ops.astype[DType.bool]`, which is
    nonzero-means-true and is the one place that rule now lives.
    """
    var n = a.size()
    var mask = condition.to_host()
    var values = a.to_host()
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        if mask[i]:
            out.append(values[i])
    return asarray(out^, a.context())


def take[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], indices: List[Int]) raises -> Dynamic[dtype, 1]:
    """The elements of `a` at `indices`, in the order given. `numpy.take`.

    The consumer for the index lists `nonzero` and `argsort` return, which
    otherwise had nothing to feed: `take(a, nonzero(a))` is the nonzero
    values and `take(a, argsort(a))` is the sorted copy, both right-sized
    without the caller reassembling a tensor by hand.

    Indices are flat and row-major, matching `numpy.take` with no `axis`.
    Out of range raises rather than wrapping, since a silent wrap turns an
    indexing bug into wrong numbers.
    """
    var n = a.size()
    var values = a.to_host()
    var out = List[Scalar[dtype]](capacity=len(indices))
    for i in range(len(indices)):
        var idx = indices[i]
        if idx < 0 or idx >= n:
            raise Error(
                "take: index ", idx, " is outside a tensor of ", n, " elements"
            )
        out.append(values[idx])
    return asarray(out^, a.context())


def select[
    dtype: DType, LayoutType: TensorLayout
](
    condition: Tensor[DType.bool, LayoutType],
    x: Tensor[dtype, LayoutType],
    y: Tensor[dtype, LayoutType],
) raises -> Tensor[dtype, LayoutType]:
    """Elementwise select: `x` where `condition` is true, `y` elsewhere.
    `numpy.where(cond, x, y)`.

    Named `select` because `where` is a Mojo keyword -- it introduces the
    constraint clauses this library uses throughout (`numax.core.tensor`'s
    `all_dims_known` checks, `numax.core.array.reshape`'s element-count check).
    Not merely a style collision: `mojo format` cannot parse `where` as an
    identifier at all. The third such rename in the parity surface, after
    `variance` and `stddev`.

    Shape-preserving, unlike everything else in this module, which is why
    it keeps the input's rank instead of flattening: the output length is
    the input length regardless of the condition's values, so there is
    nothing data-dependent about the *shape*. The overload below takes
    three shapes NumPy would broadcast, which is what
    `where(a > 0, a, 0.0)` needs.

    That also means the three-argument select could have been written as a
    tier-1 `FloatLike` kernel using the branchless `blend` in
    `numax.core.numeric`. It lives here because a NumPy caller looks for
    `numpy.where` next to `nonzero` and `extract`, and because the branching
    version reads more clearly at `Plain`. Reach for
    `numax.core.numeric.blend` when the selection has to happen inside a kernel.
    """
    var n = x.size()
    var mask = condition.to_host()
    var x_values = x.to_host()
    var y_values = y.to_host()
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = x_values[i] if mask[i] else y_values[i]
    return Tensor[dtype, LayoutType](x.context(), condition.layout, out^)


def select[
    dtype: DType,
    CLayout: TensorLayout,
    XLayout: TensorLayout,
    YLayout: TensorLayout,
](
    condition: Tensor[DType.bool, CLayout],
    x: Tensor[dtype, XLayout],
    y: Tensor[dtype, YLayout],
) raises -> Dynamic[
    dtype,
    CLayout.rank if (
        CLayout.rank > XLayout.rank and CLayout.rank > YLayout.rank
    ) else (XLayout.rank if XLayout.rank > YLayout.rank else YLayout.rank),
]:
    """`select` over three shapes NumPy would broadcast.

    `numpy.where` broadcasts all three of its arguments, which is what makes
    `where(a > 0, a, 0.0)` and `where(mask_row, matrix, fallback_column)`
    the ordinary spellings. The overload above needs one layout type for
    all three, so neither compiled.

    All three stretch: the result shape is `broadcast_shapes` applied
    twice, and the result rank is the largest of the three input ranks.
    Like every broadcasting routine in `numax.core.ops`,
    `numax.core.elementwise` and `numax.core.logic`, the walk reads through
    zero strides rather than materializing any operand, and the result is a
    `Dynamic` because the extents are computed at run time.
    """
    comptime rank = CLayout.rank if (
        CLayout.rank > XLayout.rank and CLayout.rank > YLayout.rank
    ) else (XLayout.rank if XLayout.rank > YLayout.rank else YLayout.rank)
    var c_extents = _extents_of(condition)
    var x_extents = _extents_of(x)
    var y_extents = _extents_of(y)
    var extents = broadcast_shapes(
        broadcast_shapes(c_extents, x_extents), y_extents
    )
    var c_strides = _stretch_strides(c_extents, _strides_of(condition), rank)
    var x_strides = _stretch_strides(x_extents, _strides_of(x), rank)
    var y_strides = _stretch_strides(y_extents, _strides_of(y), rank)

    var count = 1
    for d in range(rank):
        count *= extents[d]

    var mask = condition.to_host()
    var x_values = x.to_host()
    var y_values = y.to_host()
    var out = List[Scalar[dtype]](length=count, fill=0)
    for flat in range(count):
        var rem = flat
        var ci = 0
        var xi = 0
        var yi = 0
        for k in range(rank):
            var d = rank - 1 - k
            var at = rem % extents[d]
            rem //= extents[d]
            ci += at * c_strides[d]
            xi += at * x_strides[d]
            yi += at * y_strides[d]
        out[flat] = x_values[xi] if mask[ci] else y_values[yi]

    return Dynamic[dtype, rank](
        x.context(), row_major(_dyn_shape_from[rank](extents)), out^
    )


def _top_k_into[
    dtype: DType,
    SrcLayout: TensorLayout,
    DstLayout: TensorLayout,
    IdxLayout: TensorLayout,
    largest: Bool,
    gpu: Bool,
](
    mut a: Tensor[dtype, SrcLayout],
    mut values: Tensor[dtype, DstLayout],
    mut indices: Tensor[DType.int64, IdxLayout],
    k: Int,
    axis: Int,
    sorted: Bool,
) raises:
    """`nn.top_k` with numax's tensors passed straight through.

    The whole delegation: MAX takes the input, the two destinations, the
    axis, a `sorted` flag and a `DeviceContext`, and picks its own CPU or
    GPU implementation from `target`. Unlike `argsort` above there is no
    host round trip here -- `nn.top_k`'s device path is a real kernel, not a
    host fallback, so `gpu=True` keeps the data where it already is.
    """
    var ctx = a.context()
    var source = a.view()
    var out_vals = values.view()
    var out_idxs = indices.view()
    _max_top_k[largest=largest, target="gpu" if gpu else "cpu"](
        source, k, axis, out_vals, out_idxs, sorted, ctx
    )
    ctx.synchronize()


def top_k[
    dtype: DType,
    n: Int,
    k: Int,
    largest: Bool = True,
    gpu: Bool = False,
](mut a: Static[dtype, n], sorted: Bool = True) raises -> Tuple[
    Static[dtype, k], Static[DType.int64, k]
] where (k > 0 and k <= n):
    """The `k` largest elements of `a` and where they came from.
    `numpy.argpartition` paired with its values, or `torch.topk`.

    Returns `(values, indices)`; `largest=False` gives the `k` smallest.
    `sorted` orders the result by value, which is MAX's own flag and is
    stable when it is set.

    A whole delegation to `nn.top_k`, so unlike `argsort` this one has a
    real device path: `gpu=True` runs MAX's GPU kernel over the tensor where
    it already lives, with no host copy in either direction.
    """
    var ctx = a.context()
    var values = Static[dtype, k]._uninitialized(ctx)
    var indices = Static[DType.int64, k]._uninitialized(ctx)
    _top_k_into[dtype, _, _, _, largest, gpu](a, values, indices, k, 0, sorted)
    return (values^, indices^)


def top_k[
    dtype: DType,
    rows: Int,
    cols: Int,
    k: Int,
    largest: Bool = True,
    gpu: Bool = False,
](mut a: Static[dtype, rows, cols], sorted: Bool = True) raises -> Tuple[
    Static[dtype, rows, k], Static[DType.int64, rows, k]
] where (k > 0 and k <= cols):
    """The `k` largest elements of each **row** of `a`, and their columns.
    `torch.topk(a, k, dim=-1)`.

    The rank-1 form above documents the flags. This one takes the last axis
    rather than treating the matrix as flat, which is the one place this
    module departs from its own "flat, not axis-wise" rule -- `nn.top_k`
    takes an axis, so routing it flat would be numax throwing away a
    capability MAX already has.
    """
    var ctx = a.context()
    var values = Static[dtype, rows, k]._uninitialized(ctx)
    var indices = Static[DType.int64, rows, k]._uninitialized(ctx)
    _top_k_into[dtype, _, _, _, largest, gpu](a, values, indices, k, 1, sorted)
    return (values^, indices^)
