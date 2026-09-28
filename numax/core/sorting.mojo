"""Sorting, searching and counting over `numax.core.tensor.Tensor`.

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
rather than a sorted copy. So `sort`, `partition` and `argpartition` at
`gpu=True` are `nn.argsort` on the device plus a gather by its indices.
`std.builtin.sort` (stable, comparator-driven, over a `Span`) is what the
host walks here are built on, over a host copy of the tensor's elements
(`Tensor.to_host`).

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
`numax.core.functional.reduce_axis` uses; it is a straightforward extension and is
not written yet, so the flat behavior is stated rather than implied.

`top_k` is the exception, and deliberately: `nn.top_k` takes an axis, so its
rank-2 form works row-wise like `torch.topk(a, k, dim=-1)`. Flattening it
would be numax discarding a capability MAX already has.
"""

from std.builtin.sort import sort as _std_sort
from std.utils.numerics import nan as _nan

from nn.argsort import argsort as _nn_argsort
from nn.gather_scatter import (
    gather as _nn_gather,
    gather_elements as _nn_gather_elements,
)
from nn.topk import top_k as _max_top_k
from std.collections import Array

from algorithm.rowwise_types import RowCoord
from layout import Coord, TileTensor, coord_to_index_list
from max.algorithm.functional import elementwise
from std.utils import IndexList
from layout.tile_layout import TensorLayout, row_major
from .tensorlike import TensorLike, dim, is_row_major
from ._drive import (
    _check_device,
    _flat_out,
    _flat_unchecked,
    _notice,
    _require_contiguous,
)
from .logic import _count_nonzero_device
from .rowwise import reduce_all
from .tensor import (
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
    _same_order,
    _axis_gather,
    _scan_device,
)


def _nan_last_less[dtype: DType](x: Scalar[dtype], y: Scalar[dtype]) -> Bool:
    """`x < y` with NaN above every number, NumPy's sort order."""
    if y != y:
        return x == x
    return x < y


def sort[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Static[
    T.dtype, T.LayoutType.static_product
] where T.LayoutType.all_dims_known:
    """A sorted rank-1 copy of `a`, ascending. `numpy.sort(a, axis=None)`.

    Stable, because `std.builtin.sort` is; for a plain numeric sort that
    is unobservable, but it costs nothing and makes the behavior
    predictable if this later grows a key argument.

    Returns rank-1 regardless of the input's rank, which is what
    `axis=None` means. `numax.core.tensor.reshape` puts a shape back on if one
    is wanted.

    The overload below takes a tensor whose extents are run-time values
    and returns one, so `sort(extract(mask, a))` works.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    comptime n = LayoutType.static_product
    if _check_device[T, gpu](a):
        comptime if gpu:
            return _sort_device(a)^.as_static[n]()
    else:
        _notice[gpu]("sort")
    var values = a.to_host()
    comptime if dtype.is_floating_point():
        _std_sort(values, _nan_last_less[dtype])
    else:
        _std_sort(values)
    return Static[dtype, n](values^, a.context())


def sort[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Dynamic[T.dtype, 1] where not T.LayoutType.all_dims_known:
    """A sorted rank-1 copy of `a`, ascending, for a run-time shape.

    Same sort as the overload above; the result's length is a run-time
    value because the input's is.
    """
    if _check_device[T, gpu](a):
        comptime if gpu:
            return _sort_device(a)
    else:
        _notice[gpu]("sort")
    var values = a.to_host()
    comptime if T.dtype.is_floating_point():
        _std_sort(values, _nan_last_less[T.dtype])
    else:
        _std_sort(values)
    return asarray(values^, a.context())


def argsort[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Dynamic[DType.int64, 1]:
    """The flat indices that would sort `a`, ascending, as an `int64`
    tensor on `a`'s device. `numpy.argsort(a, axis=None)`.

    On the host, routed to `nn.argsort`, MAX's CPU sort. **At `gpu=True`
    numax sorts itself**, because MAX's GPU `argsort` is wrong past 256
    elements on Metal -- its single-block local sort is right and its
    cross-block merge is not (`findings.mdc`) -- and numax names no
    architecture to route around it on one vendor only. The device path is
    `_argsort_device`, a bitonic sort that orders `(value, index)` pairs,
    so ties keep their input order exactly as the host's stable sort does.

    A tensor rather than the `List[Int]` it used to be, as NumPy's is an
    array: a host list could not stay on the device, and
    `take[axis=0](a, argsort(a))` is then the sorted copy on either target.
    The flattening is the `axis=None` contract every other routine in this
    module follows, and it is what makes the input rank-1 the way
    `nn.argsort` requires. NaN sorts last, as in NumPy.
    """
    var n = a.size()
    var ctx = a.context()
    var indices = Dynamic[DType.int64, 1](row_major(_dyn_shape[1](n)), ctx)
    if _check_device[T, gpu](a):
        comptime if gpu:
            return _argsort_device(a)
    else:
        _notice[gpu]("argsort")
    var host_values = a.to_host()
    comptime if T.dtype.is_floating_point():
        # MAX's CPU sort does not order NaN (it answers close to the
        # identity once one is present), so a NaN sends the call to a
        # stable comparator sort that puts NaN last, as NumPy does.
        var has_nan = False
        for i in range(n):
            if host_values[i] != host_values[i]:
                has_nan = True
        if has_nan:
            var order = List[Int](capacity=n)
            for i in range(n):
                order.append(i)

            def nan_last(p: Int, q: Int) {imm} -> Bool:
                var x = host_values[p]
                var y = host_values[q]
                var xn = x != x
                var yn = y != y
                if xn or yn:
                    return (yn and not xn) or (xn and yn and p < q)
                return x < y or (x == y and p < q)

            _std_sort(order, nan_last)
            var out = List[Scalar[DType.int64]](capacity=n)
            for i in range(n):
                out.append(Int64(order[i]))
            indices.copy_from_host(out)
            return indices^
    var flat = asarray(host_values^, ctx)
    var flat_view = flat.tile()
    var indices_view = indices.tile()
    _nn_argsort(indices_view, flat_view)
    # `tile()` erases the origin, so `flat` is not kept alive by
    # `flat_view` and destruction is ASAP. See `numax.linalg.qr`.
    _ = flat^
    return indices^


def _argsort_device[T: TensorLike](a: T) raises -> Dynamic[DType.int64, 1]:
    """`argsort` on the device: a bitonic sort of `(value, index)` pairs.

    The keys are copied into a power-of-two buffer whose padding holds the
    largest value, each paired with its position. Every compare-exchange
    step is one `elementwise` launch over the buffer -- `log2(m) *
    (log2(m) + 1) / 2` of them for a buffer of `m` -- and a pair compares
    by value, then by index, with NaN above every number, which makes the
    order total: ties keep their input order, NaN lands after every number
    including `+inf`, and the padding (NaN, or the largest value for an
    integer dtype, at indices past `n`) lands last. The first `n`
    indices are the answer. `ponytail:` a global-memory bitonic network,
    `O(n log^2 n)` work; a shared-memory local stage, or MAX's own kernel
    once its merge is right on every target, is the upgrade.
    """
    comptime dtype = T.dtype
    var ctx = a.context()
    var n = a.size()
    if n == 0:
        return Dynamic[DType.int64, 1](row_major(_dyn_shape[1](0)), ctx)
    var m = 1
    while m < n:
        m *= 2
    var keys = Dynamic[dtype, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](m))
    )
    var order = Dynamic[DType.int64, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](m))
    )
    var src = _flat_unchecked(a)
    var kv = _flat_out(keys)
    var ov = _flat_out(order)
    var top = Scalar[dtype].MAX_FINITE
    comptime if dtype.is_floating_point():
        top = _nan[dtype]()

    @always_inline
    def seed[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var src, var kv, var ov, var n, var top}:
        var i = coord_to_index_list(coord)[0]
        var key = top
        if i < n:
            key = src[coord][0]
        kv.store[1](coord, key)
        ov.store[1](coord, Int64(i))

    elementwise[simd_width=1, target="gpu"](seed, Coord(m), ctx)
    var k = 2
    while k <= m:
        var j = k // 2
        while j >= 1:

            @always_inline
            def step[
                width: Int, alignment: Int = 1
            ](coord: Coord) {var kv, var ov, var j, var k}:
                var i = coord_to_index_list(coord)[0]
                var l = i ^ j
                if l > i:
                    var ki = kv[Coord(i)][0]
                    var kl = kv[Coord(l)][0]
                    var ii = ov[Coord(i)][0]
                    var il = ov[Coord(l)][0]
                    var greater = ki > kl
                    var same = ki == kl
                    comptime if dtype.is_floating_point():
                        var ni = ki != ki
                        var nl = kl != kl
                        greater = (ni and not nl) or (
                            not ni and not nl and ki > kl
                        )
                        same = (ni and nl) or ki == kl
                    var above = greater or (same and ii > il)
                    if above == ((i & k) == 0):
                        kv.store[1](Coord(i), kl)
                        kv.store[1](Coord(l), ki)
                        ov.store[1](Coord(i), il)
                        ov.store[1](Coord(l), ii)

            elementwise[simd_width=1, target="gpu"](step, Coord(m), ctx)
            j //= 2
        k *= 2
    ctx.synchronize()
    _ = keys^
    if m == n:
        return order^
    return _axis_gather["offset"](order, row_major(_dyn_shape[1](n)), 0, 0)


def _sort_device[T: TensorLike](a: T) raises -> Dynamic[T.dtype, 1]:
    """`sort` on the device: `argsort` there, then a gather by the
    indices, on a flat copy so the gather is along axis 0."""
    var flat = _same_order(a, row_major(_dyn_shape[1](a.size())))
    var order = argsort[gpu=True](flat)
    return take[axis=0, gpu=True](flat, order)


def searchsorted[
    T: TensorLike,
](sorted_values: T, value: Scalar[T.dtype]) raises -> Int where (
    T.LayoutType.rank == 1 and T.LayoutType.all_dims_known
):
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
    comptime n = dim[T, 0]
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


def _searchsorted_device[
    A: TensorLike, B: TensorLike, //, right: Bool
](sorted_values: A, values: B) raises -> Dynamic[DType.int64, 1] where (
    B.dtype == A.dtype
):
    """`searchsorted` on the device: one lane per query, each a binary
    search of `sorted_values` -- `O(log n)` reads, all lanes independent."""
    var ctx = values.context()
    var n = sorted_values.size()
    var m = values.size()
    var out = Dynamic[DType.int64, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](m))
    )
    if m == 0:
        return out^
    var hv = _flat_unchecked(sorted_values)
    var nv = _flat_unchecked(values)
    var ov = _flat_out(out)

    @always_inline
    def search[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var hv, var nv, var ov, var n}:
        var value = rebind[Scalar[A.dtype]](nv[coord][0])
        var lo = 0
        var hi = n
        while lo < hi:
            var mid = (lo + hi) // 2
            var here = hv[Coord(mid)][0]
            comptime if right:
                if here <= value:
                    lo = mid + 1
                else:
                    hi = mid
            else:
                if here < value:
                    lo = mid + 1
                else:
                    hi = mid
        ov.store[1](coord, Int64(lo))

    elementwise[simd_width=1, target="gpu"](search, Coord(m), ctx)
    ctx.synchronize()
    return out^


def searchsorted[
    A: TensorLike,
    B: TensorLike,
    right: Bool = False,
    gpu: Bool = False,
](sorted_values: A, values: B) raises -> Dynamic[DType.int64, 1] where (
    B.dtype == A.dtype
):
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
    overload. MAX ships no `searchsorted`, so this is numax's own. At
    `gpu=True`, with both tensors on a device, one lane per query does its
    own binary search there and the indices stay on the device; both must
    be contiguous. A residency mismatch takes the host loop with the
    `_drive` notice.
    """
    if _check_device[A, gpu](sorted_values) and _check_device[B, gpu](values):
        comptime if gpu:
            _require_contiguous(sorted_values)
            _require_contiguous(values)
            return _searchsorted_device[right=right](sorted_values, values)
    else:
        _notice[gpu]("searchsorted")
    var haystack = sorted_values.to_host()
    var needles = values.to_host[A.dtype]()
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
        row_major(_dyn_shape[1](len(needles))), out^, sorted_values.context()
    )


def _check_index_bounds_device[
    L: TensorLayout
](name: StaticString, indices: Tensor[DType.int64, L], length: Int) raises:
    """Raise, naming `name`, unless every one of the device `indices` lies
    in `[0, length)`: a device `min` and `max`, two scalars read back."""
    var count = indices.size()
    if count == 0:
        return
    var ctx = indices.context()
    var lo = Static[DType.int64, 1](ctx)
    var hi = Static[DType.int64, 1](ctx)

    @always_inline
    def identity[
        w: Int
    ](tile: SIMD[DType.int64, w], idx: RowCoord[1]) {} -> SIMD[DType.int64, w]:
        return tile

    reduce_all[monoid="min", gpu=True](
        _flat_unchecked(indices), lo.tile(), identity, count, Optional(ctx)
    )
    reduce_all[monoid="max", gpu=True](
        _flat_unchecked(indices), hi.tile(), identity, count, Optional(ctx)
    )
    var smallest = Int(lo.to_host()[0])
    var largest = Int(hi.to_host()[0])
    if smallest < 0 or largest >= length:
        raise Error(
            name,
            ": index ",
            smallest if smallest < 0 else largest,
            " is out of range for an extent of ",
            length,
        )


def _take_device[
    T: TensorLike, IndexLayout: TensorLayout, //, axis: Int
](a: T, indices: Tensor[DType.int64, IndexLayout]) raises -> Dynamic[
    T.dtype, T.LayoutType.rank
]:
    """`take` on the device: `nn.gather` over the tensors where they live.

    The bounds check is two device reductions over `indices` -- its `min`
    and `max` -- so only two scalars come back, where the host path reads
    every index. `a` and `indices` contiguous and on a GPU context.
    """
    comptime rank = T.LayoutType.rank
    var ctx = a.context()
    var count = indices.size()
    var length = a.dim_at(axis)
    _check_index_bounds_device("take", indices, length)
    var in_extents = List[Int](capacity=rank)
    var out_extents = List[Int](capacity=rank)
    for d in range(rank):
        in_extents.append(a.dim_at(d))
        out_extents.append(count if d == axis else a.dim_at(d))
    _require_contiguous(a)
    var result = Dynamic[T.dtype, rank]._uninitialized(
        ctx, row_major(_dyn_shape_from[rank](out_extents))
    )
    if result.size() == 0:
        return result^
    _nn_gather[axis=axis, target="gpu"](
        result.tile(),
        TileTensor(
            a.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin](),
            row_major(_dyn_shape_from[rank](in_extents)),
        ),
        TileTensor(
            indices.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin](),
            row_major(Coord(count)),
        ),
        context=ctx,
    )
    ctx.synchronize()
    return result^


def take[
    T: TensorLike,
    IndexLayout: TensorLayout,
    axis: Int,
    gpu: Bool = False,
](a: T, indices: Tensor[DType.int64, IndexLayout]) raises -> Dynamic[
    T.dtype, T.LayoutType.rank
] where (axis >= 0 and axis < T.LayoutType.rank and IndexLayout.rank == 1):
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
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    comptime rank = LayoutType.rank
    var count = indices.size()
    var length = a.dim_at(axis)
    if _check_device[T, gpu](a):
        comptime if gpu:
            return _take_device[axis=axis](a, indices)
    else:
        _notice[gpu]("take")
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
    _nn_gather[axis=axis, target="cpu"](
        TileTensor(out, row_major(_dyn_shape_from[rank](out_extents))),
        TileTensor(values, row_major(_dyn_shape_from[rank](in_extents))),
        TileTensor(index_values, row_major(Coord(count))),
        context=ctx,
    )
    return Dynamic[dtype, rank](
        row_major(_dyn_shape_from[rank](out_extents)), out^, ctx
    )


def _take_along_axis_device[
    T: TensorLike, IndexLayout: TensorLayout, //, axis: Int
](a: T, indices: Tensor[DType.int64, IndexLayout]) raises -> Dynamic[
    T.dtype, T.LayoutType.rank
] where (IndexLayout.rank == T.LayoutType.rank):
    """`take_along_axis` on the device: one lane per output element
    decomposes its flat index over `indices`'s extents, substitutes its
    index on `axis`, and reads `a` there."""
    comptime rank = T.LayoutType.rank
    var ctx = a.context()
    var length = a.dim_at(axis)
    var out_extents = List[Int](capacity=rank)
    var shape = IndexList[rank]()
    var strides = IndexList[rank]()
    var stride = 1
    for k in range(rank):
        var d = rank - 1 - k
        strides[d] = stride
        stride *= a.dim_at(d)
    for d in range(rank):
        if d != axis and a.dim_at(d) != indices.dim_at(d):
            raise Error(
                "take_along_axis: extents ",
                a.dim_at(d),
                " and ",
                indices.dim_at(d),
                " differ on axis ",
                d,
            )
        shape[d] = indices.dim_at(d)
        out_extents.append(indices.dim_at(d))
    _check_index_bounds_device("take_along_axis", indices, length)
    var result = Dynamic[T.dtype, rank]._uninitialized(
        ctx, row_major(_dyn_shape_from[rank](out_extents))
    )
    var total = result.size()
    if total == 0:
        return result^
    var src = _flat_unchecked(a)
    var iv = _flat_unchecked(indices)
    var dst = _flat_out(result)

    @always_inline
    def gather[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var src, var iv, var dst, var shape, var strides}:
        var f = coord_to_index_list(coord)[0]
        var rem = f
        var at = 0
        for k in range(rank):
            var d = rank - 1 - k
            var c = rem % shape[d]
            rem //= shape[d]
            if d == axis:
                c = Int(iv[coord][0])
            at += c * strides[d]
        dst.store[1](coord, src[Coord(at)][0])

    elementwise[simd_width=1, target="gpu"](gather, Coord(total), ctx)
    ctx.synchronize()
    return result^


def take_along_axis[
    T: TensorLike, IndexLayout: TensorLayout, axis: Int, gpu: Bool = False
](a: T, indices: Tensor[DType.int64, IndexLayout]) raises -> Dynamic[
    T.dtype, T.LayoutType.rank
] where (
    axis >= 0
    and axis < T.LayoutType.rank
    and IndexLayout.rank == T.LayoutType.rank
):
    """One element of `a` per entry of `indices`, indexed along `axis`.
    `numpy.take_along_axis`.

    Not `take`: `take` picks whole slices with one index list shared by
    every position, while this picks an element per position, so `indices`
    has `a`'s rank rather than rank 1. That is what makes `argsort`'s
    per-row output usable -- `take_along_axis(a, argsort_rows, axis=1)` is
    each row of `a` sorted.

    On the host, routed to `nn.gather_elements`, which is ONNX
    `GatherElements` (Torch's `gather`). The result has `indices`'s shape,
    which is that operator's contract and NumPy's too. That operator takes
    a `DeviceContext` but launches its `elementwise` with the default CPU
    target, so it has no device path; at `gpu=True`, with both tensors on
    a device and contiguous, numax's own gather runs one lane per output
    element, and the bounds check is a device `min`/`max` of the indices.
    A residency mismatch takes the host path with the `_drive` notice.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    comptime rank = LayoutType.rank
    var length = a.dim_at(axis)
    if _check_device[T, gpu](a) and not indices.on_host():
        comptime if gpu:
            _require_contiguous(a)
            _require_contiguous(indices)
            return _take_along_axis_device[axis=axis](a, indices)
    elif gpu or not a.on_host():
        _notice[gpu]("take_along_axis")
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
        row_major(_dyn_shape_from[rank](out_extents)), out^, ctx
    )


def unique[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Dynamic[T.dtype, 1]:
    """The sorted distinct values of `a`. `numpy.unique`.

    Right-sized: the result holds exactly as many elements as there are
    distinct values, which is a count only the data knows. That is what a
    run-time-shaped tensor is for, and `unique(a).size()` is the answer to
    "how many" rather than a second return value the caller has to carry.
    Every NaN is its own value, since NaN compares unequal to itself --
    NumPy's `equal_nan=False`, not its default.

    At `gpu=True`, with `a` on a device: the device sort, one launch
    flagging each element that differs from its predecessor, and the
    compaction `nonzero` uses; only the count is read back. A residency
    mismatch takes the host path with the `_drive` notice.
    """
    var n = a.size()
    if _check_device[T, gpu](a):
        comptime if gpu:
            var sorted = _sort_device(a)
            if n == 0:
                return sorted^
            var flags = Dynamic[DType.int64, 1]._uninitialized(
                sorted.context(), row_major(_dyn_shape[1](n))
            )
            var sv = _flat_unchecked(sorted)
            var fv = _flat_out(flags)

            @always_inline
            def fresh[
                width: Int, alignment: Int = 1
            ](coord: Coord) {var sv, var fv}:
                var i = coord_to_index_list(coord)[0]
                var differs = i == 0 or sv[coord][0] != sv[Coord(i - 1)][0]
                fv.store[1](coord, Int64(1) if differs else Int64(0))

            elementwise[simd_width=1, target="gpu"](
                fresh, Coord(n), sorted.context()
            )
            return _pack_device[indices=False](flags, sorted, n)
    else:
        _notice[gpu]("unique")
    var values = a.to_host()
    comptime if T.dtype.is_floating_point():
        _std_sort(values, _nan_last_less[T.dtype])
    else:
        _std_sort(values)

    var count = 0
    for i in range(n):
        if count == 0 or values[i] != values[count - 1]:
            values[count] = values[i]
            count += 1
    values.resize(count, fill=0)
    return asarray(values^, a.context())


def count_nonzero[T: TensorLike, gpu: Bool = False](a: T) raises -> Int:
    """How many elements of `a` are not zero. `numpy.count_nonzero`.

    `-0.0` counts as zero (it compares equal to `0.0`), matching NumPy.
    NaN counts as nonzero, also matching NumPy, since `nan != 0`. At
    `gpu=True` it is a device count, one flag launch and one `ReduceSum`,
    and so are `any_nonzero` and `all_nonzero` below.
    """
    if _check_device[T, gpu](a):
        comptime if gpu:
            return _count_nonzero_device(a)
    else:
        _notice[gpu]("count_nonzero")
    var n = a.size()
    var values = a.to_host()
    var total = 0
    for i in range(n):
        if values[i] != 0:
            total += 1
    return total


def any_nonzero[T: TensorLike, gpu: Bool = False](a: T) raises -> Bool:
    """Whether any element is nonzero. `numpy.any`.

    Named `any_nonzero` rather than `any` because `any` is a Mojo builtin;
    the same kind of collision that made `numpy.var` into
    `numax.stats.variance`.

    Short-circuits, which is the point of having it rather than
    `count_nonzero(a) > 0`.
    """
    if _check_device[T, gpu](a):
        comptime if gpu:
            return _count_nonzero_device(a) > 0
    else:
        _notice[gpu]("any_nonzero")
    var n = a.size()
    var values = a.to_host()
    for i in range(n):
        if values[i] != 0:
            return True
    return False


def all_nonzero[T: TensorLike, gpu: Bool = False](a: T) raises -> Bool:
    """Whether every element is nonzero. `numpy.all`, named for the same
    reason as `any_nonzero`. Short-circuits on the first zero."""
    if _check_device[T, gpu](a):
        comptime if gpu:
            return _count_nonzero_device(a) == a.size()
    else:
        _notice[gpu]("all_nonzero")
    var n = a.size()
    var values = a.to_host()
    for i in range(n):
        if values[i] == 0:
            return False
    return True


def _selected_offsets[
    S: TensorLike
](selector: S, m: Int) raises -> Dynamic[DType.int64, 1]:
    """For the first `m` elements of a GPU-context `selector`: an inclusive
    scan of their nonzero flags, so element `i`'s slot in a packed result
    is `offsets[i] - 1` and the result holds `offsets[m - 1]` elements. One
    flag launch and the device scan; the device half of every routine here
    whose output length the data decides."""
    var ctx = selector.context()
    var flags = Dynamic[DType.int64, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](m))
    )
    var sv = _flat_unchecked(selector)
    var fv = _flat_out(flags)

    @always_inline
    def flag[width: Int, alignment: Int = 1](coord: Coord) {var sv, var fv}:
        var x = sv[coord][0]
        comptime if S.dtype == DType.bool:
            fv.store[1](coord, Int64(1) if x else Int64(0))
        else:
            fv.store[1](coord, Int64(0) if x == 0 else Int64(1))

    elementwise[simd_width=1, target="gpu"](flag, Coord(m), ctx)
    return _scan_device["sum"](flags, row_major(_dyn_shape[1](m)), m, 1)


def _pack_device[
    S: TensorLike, V: TensorLike, indices: Bool
](selector: S, source: V, m: Int) raises -> Dynamic[
    DType.int64 if indices else V.dtype, 1
]:
    """The selected elements of `source` -- or, with `indices`, their flat
    positions -- packed in order, on the device: the offsets scan, one
    scalar read for the length, and one scatter launch."""
    comptime out_dtype = DType.int64 if indices else V.dtype
    var ctx = selector.context()
    if m == 0:
        return Dynamic[out_dtype, 1](row_major(_dyn_shape[1](0)), ctx)
    var offsets = _selected_offsets(selector, m)
    var count = Int(offsets[m - 1])
    var result = Dynamic[out_dtype, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](count))
    )
    if count == 0:
        return result^
    var ov = _flat_unchecked(offsets)
    var src = source.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var rv = _flat_out(result)

    @always_inline
    def scatter[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var ov, var src, var rv}:
        var i = coord_to_index_list(coord)[0]
        var here = ov[coord][0]
        var before = ov[Coord(i - 1)][0] if i > 0 else Int64(0)
        if here != before:
            comptime if indices:
                rv.store[1](
                    Coord(Int(here) - 1), rebind[Scalar[out_dtype]](Int64(i))
                )
            else:
                rv.store[1](
                    Coord(Int(here) - 1),
                    rebind[Scalar[out_dtype]](src[unsafe_offset=i]),
                )

    elementwise[simd_width=1, target="gpu"](scatter, Coord(m), ctx)
    ctx.synchronize()
    return result^


def nonzero[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Dynamic[DType.int64, 1]:
    """The flat indices of the nonzero elements, ascending.
    `numpy.flatnonzero`.

    An `int64` tensor, right-sized to the count, on `a`'s device -- the
    shape `argsort` returns and `take` consumes. At `gpu=True` the mask,
    the offsets scan and the scatter all run on the device and only the
    count is read back (`_pack_device`).
    """
    var n = a.size()
    if _check_device[T, gpu](a):
        comptime if gpu:
            _require_contiguous(a)
            return _pack_device[indices=True](a, a, n)
    else:
        _notice[gpu]("nonzero")
    var values = a.to_host()
    var indices = List[Scalar[DType.int64]](capacity=n)
    for i in range(n):
        if values[i] != 0:
            indices.append(Int64(i))
    return asarray(indices^, a.context())


def _argwhere_device[
    T: TensorLike
](a: T, extents: List[Int]) raises -> Dynamic[DType.int64, 2]:
    """`argwhere` on the device: the packed flat positions, then one launch
    turning each into its row of coordinates."""
    comptime rank = T.LayoutType.rank
    _require_contiguous(a)
    var ctx = a.context()
    var flat = _pack_device[indices=True](a, a, a.size())
    var count = flat.size()
    var result = Dynamic[DType.int64, 2]._uninitialized(
        ctx, row_major(_dyn_shape[2](count, rank))
    )
    if count == 0:
        return result^
    var ext = IndexList[rank]()
    for d in range(rank):
        ext[d] = extents[d]
    var fv = _flat_unchecked(flat)
    var rv = _flat_out(result)

    @always_inline
    def digits[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var fv, var rv, var ext}:
        var q = coord_to_index_list(coord)[0]
        var axis = q % rank
        var rem = Int(fv[Coord(q // rank)][0])
        var digit = 0
        comptime for k in range(rank):
            comptime d = rank - 1 - k
            if d == axis:
                digit = rem % ext[d]
            rem //= ext[d]
        rv.store[1](coord, Int64(digit))

    elementwise[simd_width=1, target="gpu"](digits, Coord(count * rank), ctx)
    ctx.synchronize()
    return result^


def argwhere[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Dynamic[DType.int64, 2]:
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
    comptime LayoutType = T.LayoutType
    comptime rank = LayoutType.rank
    var extents = List[Int](capacity=rank)
    for d in range(rank):
        extents.append(a.dim_at(d))

    if _check_device[T, gpu](a):
        comptime if gpu:
            return _argwhere_device(a, extents)
    else:
        _notice[gpu]("argwhere")
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
        row_major(_dyn_shape_from[2](shape)), coords^, a.context()
    )


def _put_device[
    T: TensorLike, L: TensorLayout
](
    mut a: T, indices: Tensor[DType.int64, L], values: List[Scalar[T.dtype]]
) raises:
    """`put`'s scatter on the device: the (host) `values` uploaded once,
    the device `indices` bounds-checked by a device `min`/`max`, and one
    lane per index writing its value. Duplicate indices race, and the
    last writer is unspecified, where NumPy's last index wins."""
    var n = a.size()
    var count = indices.size()
    if len(values) != count and len(values) != 1:
        raise Error(
            "put: ",
            len(values),
            " values for ",
            count,
            " indices -- give one value or one per index",
        )
    if count == 0:
        return
    _check_index_bounds_device("put", indices, n)
    var ctx = a.context()
    var upload = Dynamic[T.dtype, 1](
        row_major(_dyn_shape[1](len(values))), values.copy(), ctx
    )
    var iv = _flat_unchecked(indices)
    var vv = _flat_unchecked(upload)
    var dst = _flat_out(a)
    var single = len(values) == 1

    @always_inline
    def scatter[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var iv, var vv, var dst, var single}:
        var q = coord_to_index_list(coord)[0]
        var value = vv[Coord(0)][0] if single else vv[coord][0]
        dst.store[1](Coord(Int(iv[coord][0])), value)

    elementwise[simd_width=1, target="gpu"](scatter, Coord(count), ctx)
    ctx.synchronize()


def put[
    T: TensorLike, I: TensorLike, gpu: Bool = False
](mut a: T, indices: I, values: List[Scalar[T.dtype]]) raises where (
    I.dtype == DType.int64
):
    """`put` with the indices as a tensor, the form `nonzero` and
    `argsort` return. On the host the indices are read there, as the list
    form reads them; at `gpu=True`, with `a` and `indices` on a device and
    contiguous, the scatter runs there (`_put_device`) and the indices
    never come back. A residency mismatch takes the host path with the
    `_drive` notice."""
    if _check_device[T, gpu](a) and _check_device[I, gpu](indices):
        comptime if gpu:
            _require_contiguous(a)
            _require_contiguous(indices)
            var count = indices.size()
            var widened = Dynamic[DType.int64, 1]._uninitialized(
                a.context(), row_major(_dyn_shape[1](count))
            )
            if count > 0:
                var sv = _flat_unchecked(indices)
                var wv = _flat_out(widened)

                @always_inline
                def widen[
                    width: Int, alignment: Int = 1
                ](coord: Coord) {var sv, var wv}:
                    wv.store[1](coord, Int64(sv[coord][0]))

                elementwise[simd_width=1, target="gpu"](
                    widen, Coord(count), a.context()
                )
            _put_device(a, widened, values)
            return
    else:
        _notice[gpu]("put")
    var raw = indices.to_host()
    var flat = List[Int](capacity=len(raw))
    for i in range(len(raw)):
        flat.append(Int(raw[i]))
    put(a, flat, values)


def put[
    T: TensorLike, gpu: Bool = False
](mut a: T, indices: List[Int], values: List[Scalar[T.dtype]],) raises:
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

    At `gpu=True`, with `a` on a device and contiguous, the indices and
    values -- the caller's host data -- are uploaded once and scattered
    there in one launch, so `a` itself never crosses; duplicate indices
    then race, where on the host the last one wins. A residency mismatch
    takes the host path with the `_drive` notice.
    """
    var n = a.size()
    if _check_device[T, gpu](a):
        comptime if gpu:
            _require_contiguous(a)
            var picks = List[Scalar[DType.int64]](capacity=len(indices))
            for i in range(len(indices)):
                picks.append(Int64(indices[i]))
            var index_tensor = Dynamic[DType.int64, 1](
                row_major(_dyn_shape[1](len(indices))), picks^, a.context()
            )
            _put_device(a, index_tensor, values)
            return
    else:
        _notice[gpu]("put")
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
    C: TensorLike, T: TensorLike, gpu: Bool = False
](condition: C, a: T) raises -> Dynamic[T.dtype, 1] where (
    C.dtype == DType.bool and C.LayoutType == T.LayoutType
):
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
    comptime dtype = T.dtype
    var n = a.size()
    if _check_device[C, gpu](condition) and _check_device[T, gpu](a):
        comptime if gpu:
            _require_contiguous(condition)
            _require_contiguous(a)
            return _pack_device[indices=False](condition, a, n)
    else:
        _notice[gpu]("extract")
    var mask = condition.to_host()
    var values = a.to_host()
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        if mask[i]:
            out.append(values[i])
    return asarray(out^, a.context())


def compress[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](condition: A, a: B) raises -> Dynamic[B.dtype, 1] where (
    A.LayoutType.rank == 1 and A.dtype == DType.bool
):
    """The flat elements of `a` where the rank-1 `condition` is true.
    `numpy.compress` with no `axis`.

    Close to `extract` and different in one way that matters: `extract`
    takes a mask over `a`'s own layout, so the two shapes must match, while
    `condition` here is rank 1 and **may be shorter than `a`**. Positions
    past its end are dropped rather than treated as false-by-default or
    raising, which is NumPy's rule and the reason both names exist. A
    condition longer than `a` is the error case and raises.
    """
    comptime dtype = B.dtype
    var n = a.size()
    var m = condition.size()
    if m > n:
        raise Error(
            "compress: a condition of ",
            m,
            " entries is longer than the ",
            n,
            " elements it selects from",
        )
    if _check_device[A, gpu](condition) and _check_device[B, gpu](a):
        comptime if gpu:
            _require_contiguous(condition)
            _require_contiguous(a)
            return _pack_device[indices=False](condition, a, m)
    else:
        _notice[gpu]("compress")
    var mask = condition.to_host()
    var values = a.to_host()
    var out = List[Scalar[dtype]](capacity=m)
    for i in range(m):
        if mask[i]:
            out.append(values[i])
    return asarray(out^, a.context())


def partition[
    T: TensorLike, gpu: Bool = False
](a: T, kth: Int) raises -> Static[
    T.dtype, T.LayoutType.static_product
] where T.LayoutType.all_dims_known:
    """A rank-1 copy of `a` with the `kth` element in its sorted position,
    everything smaller before it and everything larger after.
    `numpy.partition(a, kth, axis=None)`.

    `ponytail:` this sorts. A full sort satisfies the partition contract
    exactly -- position `kth` holds the value it would in a sorted array,
    and both sides are ordered, which is stronger than required -- so the
    answer is right and the cost is `O(n log n)` where introselect is
    `O(n)`. The upgrade is the three-way quickselect that
    `numax.stats.quantiles._select_pair` already runs: it cannot be called
    from here because `numax.core` depends on no other numax subpackage,
    so sharing it means moving it down into this package, which is a
    change to a measured hot path rather than a new name.
    """
    if kth < 0 or kth >= a.size():
        raise Error(
            "partition: kth ",
            kth,
            " is outside a tensor of ",
            a.size(),
            " elements",
        )
    return sort[gpu=gpu](a)


def partition[
    T: TensorLike, gpu: Bool = False
](a: T, kth: Int) raises -> Dynamic[
    T.dtype, 1
] where not T.LayoutType.all_dims_known:
    """`numpy.partition` for a run-time shape. See the overload above,
    including why it sorts."""
    if kth < 0 or kth >= a.size():
        raise Error(
            "partition: kth ",
            kth,
            " is outside a tensor of ",
            a.size(),
            " elements",
        )
    return sort[gpu=gpu](a)


def argpartition[
    T: TensorLike, gpu: Bool = False
](a: T, kth: Int) raises -> Dynamic[DType.int64, 1]:
    """The flat indices that would partition `a` about `kth`.
    `numpy.argpartition(a, kth, axis=None)`.

    `argsort`'s indices, which satisfy the partition contract for the same
    reason `partition` sorts -- a full ordering is a partition about every
    `kth` at once. The ceiling `partition` names applies here too.
    """
    if kth < 0 or kth >= a.size():
        raise Error(
            "argpartition: kth ",
            kth,
            " is outside a tensor of ",
            a.size(),
            " elements",
        )
    return argsort[gpu=gpu](a)


def take[
    T: TensorLike, gpu: Bool = False
](a: T, indices: List[Int]) raises -> Dynamic[T.dtype, 1]:
    """The elements of `a` at `indices`, in the order given. `numpy.take`.

    The consumer for the index lists `nonzero` and `argsort` return, which
    otherwise had nothing to feed: `take(a, nonzero(a))` is the nonzero
    values and `take(a, argsort(a))` is the sorted copy, both right-sized
    without the caller reassembling a tensor by hand.

    Indices are flat and row-major, matching `numpy.take` with no `axis`.
    Out of range raises rather than wrapping, since a silent wrap turns an
    indexing bug into wrong numbers.

    At `gpu=True`, with `a` on a device and contiguous, the list -- the
    caller's host data, checked on the host -- is uploaded once and the
    gather runs there (`take[axis=0, gpu=True]` over a flat view). A
    residency mismatch takes the host path with the `_drive` notice.
    """
    comptime dtype = T.dtype
    var n = a.size()
    for i in range(len(indices)):
        if indices[i] < 0 or indices[i] >= n:
            raise Error(
                "take: index ",
                indices[i],
                " is outside a tensor of ",
                n,
                " elements",
            )
    if _check_device[T, gpu](a):
        comptime if gpu:
            _require_contiguous(a)
            var picks = List[Scalar[DType.int64]](capacity=len(indices))
            for i in range(len(indices)):
                picks.append(Int64(indices[i]))
            var index_tensor = Dynamic[DType.int64, 1](
                row_major(_dyn_shape[1](len(indices))), picks^, a.context()
            )
            var flat = _same_order(a, row_major(_dyn_shape[1](n)))
            return take[axis=0, gpu=True](flat, index_tensor)
    else:
        _notice[gpu]("take")
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


def _select_device[
    C: TensorLike, T: TensorLike
](condition: C, x: T, y: T) raises -> Tensor[T.dtype, T.LayoutType]:
    """`select` on the device at one layout: one launch, `x` where the
    mask is true and `y` elsewhere. All three contiguous, on a GPU."""
    var ctx = x.context()
    var result = Tensor[T.dtype, T.LayoutType]._uninitialized(
        ctx, x.tile().layout
    )
    var n = x.size()
    if n == 0:
        return result^
    var cp = _flat_unchecked(condition)
    var xp = _flat_unchecked(x)
    var yp = _flat_unchecked(y)
    var dst = _flat_out(result)

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var cp, var xp, var yp, var dst}:
        var pick = xp[coord] if cp[coord][0] else yp[coord]
        dst.store[1](coord, pick[0])

    elementwise[simd_width=1, target="gpu"](body, Coord(n), ctx)
    ctx.synchronize()
    return result^


def _select_broadcast_device[
    rank: Int, A: TensorLike, B: TensorLike, C: TensorLike
](
    condition: A,
    x: B,
    y: C,
    extents: List[Int],
    c_strides: List[Int],
    x_strides: List[Int],
    y_strides: List[Int],
) raises -> Dynamic[B.dtype, rank]:
    """The broadcasting `select` on the device: one thread per result
    element, each operand read through its stretched strides (zero on a
    broadcast axis), so no operand is materialized at the result's shape.
    """
    var ctx = x.context()
    var ext = IndexList[rank]()
    var cs = IndexList[rank]()
    var xs_ = IndexList[rank]()
    var ys_ = IndexList[rank]()
    var total = 1
    for d in range(rank):
        ext[d] = extents[d]
        cs[d] = c_strides[d]
        xs_[d] = x_strides[d]
        ys_[d] = y_strides[d]
        total *= extents[d]
    var result = Dynamic[B.dtype, rank]._uninitialized(
        ctx, row_major(_dyn_shape_from[rank](extents))
    )
    if total == 0:
        return result^
    var cp = condition.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var xp = x.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var yp = (
        y.tile()
        .ptr.unsafe_bitcast[Scalar[B.dtype]]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    var dst = _flat_out(result)

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {
        var cp, var xp, var yp, var dst, var ext, var cs, var xs_, var ys_
    }:
        var rem = coord_to_index_list(coord)[0]
        var ci = 0
        var xi = 0
        var yi = 0
        comptime for k in range(rank):
            comptime d = rank - 1 - k
            var at = rem % ext[d]
            rem //= ext[d]
            ci += at * cs[d]
            xi += at * xs_[d]
            yi += at * ys_[d]
        if cp[unsafe_offset=ci]:
            dst.store[1](coord, xp[unsafe_offset=xi])
        else:
            dst.store[1](coord, yp[unsafe_offset=yi])

    elementwise[simd_width=1, target="gpu"](body, Coord(total), ctx)
    ctx.synchronize()
    return result^


def select[
    C: TensorLike, T: TensorLike, gpu: Bool = False
](condition: C, x: T, y: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    C.dtype == DType.bool and C.LayoutType == T.LayoutType
):
    """Elementwise select: `x` where `condition` is true, `y` elsewhere.
    `numpy.where(cond, x, y)`.

    Named `select` because `where` is a Mojo keyword -- it introduces the
    constraint clauses this library uses throughout (`numax.core.functional`'s
    `all_dims_known` checks, `numax.core.tensor.reshape`'s element-count check).
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
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    var n = x.size()
    if (
        _check_device[C, gpu](condition)
        and _check_device[T, gpu](x)
        and _check_device[T, gpu](y)
    ):
        comptime if gpu:
            return _select_device(condition, x, y)
    else:
        _notice[gpu]("select")
    var mask = condition.to_host()
    var x_values = x.to_host()
    var y_values = y.to_host()
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = x_values[i] if mask[i] else y_values[i]
    return Tensor[dtype, LayoutType](x.tile().layout, out^, x.context())


def select[
    A: TensorLike,
    B: TensorLike,
    C: TensorLike,
    gpu: Bool = False,
](condition: A, x: B, y: C) raises -> Dynamic[
    B.dtype,
    A.LayoutType.rank if (
        A.LayoutType.rank > B.LayoutType.rank
        and A.LayoutType.rank > C.LayoutType.rank
    ) else (
        B.LayoutType.rank if B.LayoutType.rank
        > C.LayoutType.rank else C.LayoutType.rank
    ),
] where (A.dtype == DType.bool and C.dtype == B.dtype):
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
    comptime dtype = B.dtype
    comptime CLayout = A.LayoutType
    comptime XLayout = B.LayoutType
    comptime YLayout = C.LayoutType
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

    if (
        _check_device[A, gpu](condition)
        and _check_device[B, gpu](x)
        and _check_device[C, gpu](y)
    ):
        comptime if gpu:
            return _select_broadcast_device[rank](
                condition, x, y, extents, c_strides, x_strides, y_strides
            )
    else:
        _notice[gpu]("select")

    var mask = condition.to_host()
    var x_values = x.to_host()
    var y_values = y.to_host[B.dtype]()
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
        row_major(_dyn_shape_from[rank](extents)), out^, x.context()
    )


def _top_k_into[
    T: TensorLike,
    ValuesLayout: TensorLayout,
    IndicesLayout: TensorLayout,
    largest: Bool,
    gpu: Bool,
](
    a: T,
    mut values: Tensor[T.dtype, ValuesLayout],
    mut indices: Tensor[DType.int64, IndicesLayout],
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
    var source = a.tile()
    var out_vals = values.tile()
    var out_idxs = indices.tile()
    _max_top_k[largest=largest, target="gpu" if gpu else "cpu"](
        source, k, axis, out_vals, out_idxs, sorted, ctx
    )
    ctx.synchronize()


def top_k[
    T: TensorLike,
    k: Int,
    largest: Bool = True,
    gpu: Bool = False,
](a: T, sorted: Bool = True) raises -> Tuple[
    Static[T.dtype, k], Static[DType.int64, k]
] where (
    k > 0
    and k <= dim[T, 0]
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
):
    """The `k` largest elements of `a` and where they came from.
    `numpy.argpartition` paired with its values, or `torch.topk`.

    Returns `(values, indices)`; `largest=False` gives the `k` smallest.
    `sorted` orders the result by value, which is MAX's own flag and is
    stable when it is set.

    A whole delegation to `nn.top_k`, so unlike `argsort` this one has a
    real device path: `gpu=True` runs MAX's GPU kernel over the tensor where
    it already lives, with no host copy in either direction.
    """
    comptime dtype = T.dtype
    var ctx = a.context()
    var values = Static[dtype, k]._uninitialized(ctx)
    var indices = Static[DType.int64, k]._uninitialized(ctx)
    _top_k_into[largest=largest, gpu=gpu](a, values, indices, k, 0, sorted)
    return (values^, indices^)


def top_k[
    T: TensorLike,
    k: Int,
    largest: Bool = True,
    gpu: Bool = False,
](a: T, sorted: Bool = True) raises -> Tuple[
    Static[T.dtype, dim[T, 0], k], Static[DType.int64, dim[T, 0], k]
] where (
    k > 0
    and k <= dim[T, 1]
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
):
    """The `k` largest elements of each **row** of `a`, and their columns.
    `torch.topk(a, k, dim=-1)`.

    The rank-1 form above documents the flags. This one takes the last axis
    rather than treating the matrix as flat, which is the one place this
    module departs from its own "flat, not axis-wise" rule -- `nn.top_k`
    takes an axis, so routing it flat would be numax throwing away a
    capability MAX already has.
    """
    comptime dtype = T.dtype
    comptime rows = dim[T, 0]
    var ctx = a.context()
    var values = Static[dtype, rows, k]._uninitialized(ctx)
    var indices = Static[DType.int64, rows, k]._uninitialized(ctx)
    _top_k_into[largest=largest, gpu=gpu](a, values, indices, k, 1, sorted)
    return (values^, indices^)
