"""Sorting, searching and counting over `numax.array.Tensor`.

**This module is tier 2.** A comparison sort runs a data-dependent number
of comparisons and branches per element; `searchsorted` halves an interval
based on a comparison; `unique` produces an output whose *length* depends on
the input values. None of that can appear in a `FloatLike`-generic kernel --
a `Self` may hold a SIMD vector whose lanes disagree about which branch they
want, and there is no per-lane `select` on the trait. So everything here is
`Plain`-only, host-side, and not GPU-launchable. See
`docs/architecture.md`'s "Two tiers".

That restriction was previously stated as a blanket exclusion: sorting was
"not absorbed", full stop, and `numax.statistics.median` reached
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
functions here are built on, since a `Tensor` owns a `List` and that is
exactly what `sort` wants.

## Flat, not axis-wise

Every function here treats its input as flat row-major, matching
`numpy.sort(a, axis=None)` rather than the default `axis=-1`. Axis-wise
sorting would need the same `outer`/`length`/`inner` decomposition
`numax.tensor.reduce_axis` uses; it is a straightforward extension and is
not written yet, so the flat behavior is stated rather than implied.
"""

from std.builtin.sort import sort as _std_sort
from std.collections import Array

from .array import Tensor, _product


def sort[
    dtype: DType, *dims: Int
](mut a: Tensor[dtype, *dims]) -> Tensor[dtype, _product[*dims]()]:
    """A sorted rank-1 copy of `a`, ascending. `numpy.sort(a, axis=None)`.

    Stable, because `std.builtin.sort` is; for a plain numeric sort that
    is unobservable, but it costs nothing and makes the behavior
    predictable if this later grows a key argument.

    Returns rank-1 regardless of the input's rank, which is what
    `axis=None` means. `numax.array.reshape` puts a shape back on if one
    is wanted.
    """
    comptime n = _product[*dims]()
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(a[i])
    _std_sort(values)
    return Tensor[dtype, n](values^)


def argsort[dtype: DType, *dims: Int](mut a: Tensor[dtype, *dims]) -> List[Int]:
    """The flat indices that would sort `a`, ascending.
    `numpy.argsort(a, axis=None)`.

    Returned as a `List[Int]` rather than a `Tensor`, because an index
    array is not a numeric tensor: nothing downstream wants to run a
    `FloatLike` kernel over it, and giving it a `Tensor[int64, ...]` would
    invite exactly that.

    Implemented as a selection over a value/index pair list rather than by
    calling `nn.argsort`. `nn.argsort` would be the better choice for a
    caller already holding a device `TileTensor`; here the input is a
    host-owned `Tensor`, and going through MAX would mean building a
    tensor, launching, and reading back an int64 buffer to produce a
    `List[Int]`.
    """
    comptime n = _product[*dims]()
    var order = List[Int](capacity=n)
    for i in range(n):
        order.append(i)

    # Insertion sort on the index list, comparing through `a`. O(n^2), and
    # deliberately so: `Tensor` is comptime-sized and these are small, and
    # a comparator-driven `std.builtin.sort` over indices would need a
    # capturing closure over `a`, which is the thing `thin` function
    # parameters exist to forbid elsewhere in this library.
    for i in range(1, n):
        var key = order[i]
        var key_value = a[key]
        var j = i - 1
        while j >= 0 and a[order[j]] > key_value:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key
    return order^


def searchsorted[
    dtype: DType, n: Int
](mut sorted_values: Tensor[dtype, n], value: Scalar[dtype]) -> Int:
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
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = (lo + hi) // 2
        if sorted_values[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def unique[
    dtype: DType, *dims: Int
](mut a: Tensor[dtype, *dims]) -> Tuple[Tensor[dtype, _product[*dims]()], Int]:
    """The sorted distinct values of `a`, and how many there are.

    `numpy.unique` returns a right-sized array; a comptime-shaped `Tensor`
    cannot, since the count depends on the values. So the tensor comes back
    at full length with the distinct values packed into its first `count`
    entries and the tail left as whatever the duplicates were. **Read only
    the first `count`.**

    That is the honest shape for this constraint rather than the pretty
    one. The alternative -- returning a `List[Scalar[dtype]]` -- would be
    right-sized but would drop out of the `Tensor` surface everything else
    in `numax.array` speaks.
    """
    comptime n = _product[*dims]()
    var sorted_copy = sort[dtype, *dims](a)
    if n == 0:
        return (sorted_copy^, 0)

    var count = 1
    for i in range(1, n):
        if sorted_copy[i] != sorted_copy[count - 1]:
            sorted_copy[count] = sorted_copy[i]
            count += 1
    return (sorted_copy^, count)


def count_nonzero[dtype: DType, *dims: Int](mut a: Tensor[dtype, *dims]) -> Int:
    """How many elements of `a` are not zero. `numpy.count_nonzero`.

    `-0.0` counts as zero (it compares equal to `0.0`), matching NumPy.
    NaN counts as nonzero, also matching NumPy, since `nan != 0`.
    """
    comptime n = _product[*dims]()
    var total = 0
    for i in range(n):
        if a[i] != 0:
            total += 1
    return total


def any_nonzero[dtype: DType, *dims: Int](mut a: Tensor[dtype, *dims]) -> Bool:
    """Whether any element is nonzero. `numpy.any`.

    Named `any_nonzero` rather than `any` because `any` is a Mojo builtin;
    the same kind of collision that made `numpy.var` into
    `numax.statistics.variance`.

    Short-circuits, which is the point of having it rather than
    `count_nonzero(a) > 0`.
    """
    comptime n = _product[*dims]()
    for i in range(n):
        if a[i] != 0:
            return True
    return False


def all_nonzero[dtype: DType, *dims: Int](mut a: Tensor[dtype, *dims]) -> Bool:
    """Whether every element is nonzero. `numpy.all`, named for the same
    reason as `any_nonzero`. Short-circuits on the first zero."""
    comptime n = _product[*dims]()
    for i in range(n):
        if a[i] == 0:
            return False
    return True


def nonzero[dtype: DType, *dims: Int](mut a: Tensor[dtype, *dims]) -> List[Int]:
    """The flat indices of the nonzero elements, ascending.
    `numpy.flatnonzero`.

    A `List[Int]` for the same reason `argsort` returns one: these are
    indices, not numbers to compute with. Right-sized, since a `List` can
    be -- which is exactly the freedom `unique`'s `Tensor` return does not
    have.
    """
    comptime n = _product[*dims]()
    var indices = List[Int](capacity=n)
    for i in range(n):
        if a[i] != 0:
            indices.append(i)
    return indices^


def extract[
    dtype: DType, *dims: Int
](mut condition: Tensor[dtype, *dims], mut a: Tensor[dtype, *dims]) -> Tuple[
    Tensor[dtype, _product[*dims]()], Int
]:
    """The elements of `a` where `condition` is nonzero, and how many.
    `numpy.extract` / `a[mask]`.

    Boolean masking -- the operation nothing in `numax` could express
    before, because its output length depends on the mask's *values*.
    Packed into the first `count` entries of a full-length tensor, on the
    same terms as `unique`: **read only the first `count`**.

    `condition` is a tensor of the same dtype rather than a separate
    boolean type, and nonzero means true. That matches how the rest of
    this module reads truthiness and avoids introducing a `Tensor[bool]`
    that no `FloatLike` kernel could consume anyway.
    """
    comptime n = _product[*dims]()
    var out = List[Scalar[dtype]](length=n, fill=0)
    var count = 0
    for i in range(n):
        if condition[i] != 0:
            out[count] = a[i]
            count += 1
    return (Tensor[dtype, n](out^), count)


def select[
    dtype: DType, *dims: Int
](
    mut condition: Tensor[dtype, *dims],
    mut x: Tensor[dtype, *dims],
    mut y: Tensor[dtype, *dims],
) -> Tensor[dtype, *dims]:
    """Elementwise select: `x` where `condition` is nonzero, `y` elsewhere.
    `numpy.where(cond, x, y)`.

    Named `select` because `where` is a Mojo keyword -- it introduces the
    constraint clauses this library uses throughout (`numax.tensor`'s
    `all_dims_known` checks, `numax.array.reshape`'s element-count check).
    Not merely a style collision: `mojo format` cannot parse `where` as an
    identifier at all. The third such rename in the parity surface, after
    `variance` and `stddev`.

    Shape-preserving, unlike everything else in this module, which is why
    it keeps the input's rank instead of flattening: the output length is
    the input length regardless of the condition's values, so there is
    nothing data-dependent about the *shape*.

    That also means the three-argument select could have been written as a
    tier-1 `FloatLike` kernel using the branchless `blend` in
    `numax.numeric`. It lives here because a NumPy caller looks for
    `numpy.where` next to `nonzero` and `extract`, and because the branching
    version reads more clearly at `Plain`. Reach for
    `numax.numeric.blend` when the selection has to happen inside a kernel.
    """
    comptime n = _product[*dims]()
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = x[i] if condition[i] != 0 else y[i]
    return Tensor[dtype, *dims](out^)
