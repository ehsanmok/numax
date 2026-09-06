"""Sorting, searching and counting over `numax.core.array.Tensor`.

**This module is tier 2.** A comparison sort runs a data-dependent number
of comparisons and branches per element; `searchsorted` halves an interval
based on a comparison; `unique` produces an output whose *length* depends on
the input values. None of that can appear in a `FloatLike`-generic kernel --
a `Self` may hold a SIMD vector whose lanes disagree about which branch they
want, and there is no per-lane `select` on the trait. So everything here is
`Plain`-only, host-side, and not GPU-launchable. See
`docs/architecture.md`'s "Two tiers".

That restriction was previously stated as a blanket exclusion: sorting was
"not absorbed", full stop, and `numax.stats.median` reached
`List.sort()` internally with a note that this was stdlib machinery rather
than a numax API. With the tiers written down, the honest position is
narrower and more useful. numax does not write comparison logic *inside the
trait*; a NumPy caller still gets `sort`, `argsort`, `searchsorted` and
`unique` as ordinary tier-2 names.

## MAX-first, and where it runs out

`nn.argsort` exists and is rank-1, index-returning, CPU + GPU. It is the
right thing for a caller already holding a `TileTensor` on a device. What it
does not give is a *value* sort, an n-dimensional sort, `searchsorted`, or
`unique` -- and it returns indices into a tensor rather than a sorted copy.
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

Every function here treats its input as flat row-major, matching
`numpy.sort(a, axis=None)` rather than the default `axis=-1`. Axis-wise
sorting would need the same `outer`/`length`/`inner` decomposition
`numax.core.tensor.reduce_axis` uses; it is a straightforward extension and is
not written yet, so the flat behavior is stated rather than implied.
"""

from std.builtin.sort import sort as _std_sort

from nn.argsort import argsort as _nn_argsort
from std.collections import Array

from layout.tile_layout import TensorLayout, row_major
from .array import Dynamic, Shaped, Tensor, asarray, _dyn_shape, _product


def sort[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Shaped[
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
    return Shaped[dtype, n](a.context(), values^)


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
    kernel over it, and giving it a `Shaped[int64, ...]` would invite exactly
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

    var order = List[Int](capacity=n)
    var raw = indices.to_host()
    for i in range(n):
        order.append(Int(raw[i]))
    return order^


def searchsorted[
    dtype: DType, n: Int
](sorted_values: Shaped[dtype, n], value: Scalar[dtype]) raises -> Int:
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
    nothing data-dependent about the *shape*.

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
