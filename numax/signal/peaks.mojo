"""Peak finding over `numax.core.tensor.Tensor`: `find_peaks` with SciPy's
`height`, `threshold` and `distance` conditions.

**Tier 2.** The number of peaks is a property of the data, so the result
is a run-time-length `int64` tensor of indices, like `nonzero`'s. At
`gpu=True` one launch marks every local maximum (walking a flat top to its
midpoint) with the `height` and `threshold` conditions applied, and the
device compaction packs them; the `distance` condition is a greedy pass in
order of peak height that has no independent lanes, so it alone reads the
`p` peaks back and runs on the host -- a declared exception, `O(p log p)`.
`peak_prominences` over a tensor of peaks is one thread per peak walking
out to its saddles. MAX has nothing for this; **extend**.
"""

from std.builtin.sort import sort as _sort

from algorithm.rowwise_types import RowCoord
from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise

from ..core._drive import _check_device, _flat_out, _flat_unchecked, _notice
from ..core.rowwise import reduce_all
from ..core.sorting import _pack_device
from ..core.tensorlike import TensorLike, TensorView, dim, is_row_major
from ..core.tensor import Dynamic, Static, _dyn_shape, asarray


def _distance_filter(
    peaks: List[Int], heights: List[Float64], gap: Int
) -> List[Int]:
    """SciPy's `distance` condition: highest peak first (ties by position),
    each kept peak removing every other within `gap` samples."""
    var count = len(peaks)
    var order = List[Int](capacity=count)
    for j in range(count):
        order.append(j)

    def higher(a: Int, b: Int) {imm} -> Bool:
        return heights[a] > heights[b] or (heights[a] == heights[b] and a < b)

    _sort(order, higher)
    var removed = List[Bool](length=count, fill=False)
    for j in order:
        if removed[j]:
            continue
        for k in range(count):
            if k != j and not removed[k] and abs(peaks[k] - peaks[j]) < gap:
                removed[k] = True
    var kept = List[Int]()
    for j in range(count):
        if not removed[j]:
            kept.append(peaks[j])
    return kept^


def _find_peaks_device[
    T: TensorLike
](
    x: T,
    height: Optional[Float64],
    threshold: Optional[Float64],
    distance: Optional[Int],
) raises -> Dynamic[DType.int64, 1] where (T.LayoutType.rank == 1):
    """`find_peaks` on the device: one launch marks each local maximum that
    passes `height` and `threshold`, the device compaction packs them, and
    only `distance` reads the peaks back."""
    comptime dtype = T.dtype
    var ctx = x.context()
    var n = x.size()
    var flags = Dynamic[DType.int64, 1](row_major(_dyn_shape[1](n)), ctx)
    var xv = _flat_unchecked(x)
    var fv = _flat_out(flags)
    var use_height = Bool(height)
    var floor = Scalar[dtype](height.value() if height else 0)
    var use_threshold = Bool(threshold)
    var step = Scalar[dtype](threshold.value() if threshold else 0)

    @always_inline
    def mark[
        width: Int, alignment: Int = 1
    ](coord: Coord) {
        var xv,
        var fv,
        var n,
        var use_height,
        var floor,
        var use_threshold,
        var step,
    }:
        var i = coord_to_index_list(coord)[0]
        if i < 1 or i >= n - 1:
            return
        var v = xv[Coord(i)][0]
        if not (xv[Coord(i - 1)][0] < v):
            return
        var ahead = i + 1
        while ahead < n - 1 and xv[Coord(ahead)][0] == v:
            ahead += 1
        if not (xv[Coord(ahead)][0] < v):
            return
        var mid = (i + ahead - 1) // 2
        var top = xv[Coord(mid)][0]
        if use_height and top < floor:
            return
        if use_threshold:
            var left = top - xv[Coord(mid - 1)][0]
            var right = top - xv[Coord(mid + 1)][0]
            if min(left, right) < step:
                return
        fv.store[1](Coord(mid), Int64(1))

    elementwise[simd_width=1, target="gpu"](mark, Coord(n), ctx)
    var peaks = _pack_device[indices=True](flags, flags, n)
    if not distance or peaks.size() < 2:
        return peaks^
    var raw = peaks.to_host()
    var tops = _pack_device[indices=False](flags, x, n).to_host()
    var positions = List[Int](capacity=len(raw))
    var heights = List[Float64](capacity=len(raw))
    for i in range(len(raw)):
        positions.append(Int(raw[i]))
        heights.append(Float64(tops[i]))
    var kept = _distance_filter(positions, heights, distance.value())
    var out = List[Scalar[DType.int64]](capacity=len(kept))
    for p in kept:
        out.append(Int64(p))
    return asarray(out^, ctx)


def find_peaks[
    T: TensorLike,
    gpu: Bool = False,
](
    x: T,
    height: Optional[Float64] = None,
    threshold: Optional[Float64] = None,
    distance: Optional[Int] = None,
) raises -> Dynamic[DType.int64, 1] where (
    (T.dtype.is_floating_point() and dim[T, 0] > 0)
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
    and is_row_major[T]
):
    """The indices of the local maxima of `x`, ascending, filtered by
    SciPy's conditions. `scipy.signal.find_peaks(x, height, threshold,
    distance)`, first return value.

    A local maximum is a sample strictly greater than both neighbors; a
    flat top counts once, at its midpoint, as SciPy counts it, and the two
    end samples never do. `height` keeps peaks at least that high,
    `threshold` those that stand at least that far above *both* neighbors,
    and `distance` thins peaks closer than that many samples, keeping the
    higher one first -- SciPy's greedy order, so the same peaks survive.
    `prominence`, `width` and `plateau_size` are not provided.
    """
    comptime n = dim[T, 0]
    if _check_device[T, gpu](x):
        comptime if gpu:
            return _find_peaks_device(x, height, threshold, distance)
    else:
        _notice[gpu]("find_peaks")
    var xs = x.to_host()
    var peaks = List[Int]()
    var i = 1
    while i < n - 1:
        if xs[i - 1] < xs[i]:
            var ahead = i + 1
            while ahead < n - 1 and xs[ahead] == xs[i]:
                ahead += 1
            if xs[ahead] < xs[i]:
                peaks.append((i + ahead - 1) // 2)
                i = ahead
                continue
        i += 1

    if height:
        var floor = height.value()
        var kept = List[Int]()
        for p in peaks:
            if Float64(xs[p]) >= floor:
                kept.append(p)
        peaks = kept^

    if threshold:
        var step = threshold.value()
        var kept = List[Int]()
        for p in peaks:
            var left = Float64(xs[p]) - Float64(xs[p - 1])
            var right = Float64(xs[p]) - Float64(xs[p + 1])
            if min(left, right) >= step:
                kept.append(p)
        peaks = kept^

    if distance:
        var heights = List[Float64](capacity=len(peaks))
        for p in peaks:
            heights.append(Float64(xs[p]))
        peaks = _distance_filter(peaks, heights, distance.value())

    var out = List[Scalar[DType.int64]](capacity=len(peaks))
    for p in peaks:
        out.append(Int64(p))
    return asarray(out^, x.context())


def peak_prominences[
    T: TensorLike,
](x: T, peaks: List[Int]) raises -> List[Float64] where (
    (T.dtype.is_floating_point() and dim[T, 0] > 0)
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
):
    """The topographic prominence of each peak in `peaks`.
    `scipy.signal.peak_prominences(x, peaks)`, first return value.

    How far a peak stands above the higher of the two saddles that
    separate it from any taller ground: walk left from the peak until a
    sample at least as high as the peak is met or the signal ends, keeping
    the minimum along the way, do the same to the right, and subtract the
    larger of the two minima from the peak's height. That is SciPy's
    definition exactly, including the treatment of the ends -- a peak with
    no taller ground on one side takes that side's window out to the
    boundary.

    `peaks` is what `find_peaks` returns, and an index outside the signal
    raises rather than being skipped, since a mismatched pair of arrays is
    a caller bug rather than a sample to ignore.

    **Tier 2, host-side.** The walks are data-dependent and their combined
    length is `O(n)` per peak in the worst case; SciPy's is the same
    algorithm. The `left_bases`/`right_bases` SciPy also returns are not
    provided -- ask for them when a caller needs the saddle positions
    rather than the heights.
    """
    comptime n = dim[T, 0]
    var xs = x.to_host()
    var out = List[Float64](capacity=len(peaks))
    for p in range(len(peaks)):
        var at = peaks[p]
        if at < 0 or at >= n:
            raise Error(
                "peak_prominences: peak index ",
                at,
                " is outside a signal of ",
                n,
                " samples",
            )
        var height = Float64(xs[at])

        var left_min = height
        var i = at
        while i > 0:
            i -= 1
            if Float64(xs[i]) > height:
                break
            if Float64(xs[i]) < left_min:
                left_min = Float64(xs[i])

        var right_min = height
        var j = at
        while j < n - 1:
            j += 1
            if Float64(xs[j]) > height:
                break
            if Float64(xs[j]) < right_min:
                right_min = Float64(xs[j])

        var base = left_min if left_min > right_min else right_min
        out.append(height - base)
    return out^


def peak_prominences[
    T: TensorLike,
    P: TensorLike,
    gpu: Bool = False,
](x: T, peaks: P) raises -> Dynamic[T.dtype, 1] where (
    (T.dtype.is_floating_point() and dim[T, 0] > 0)
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
    and P.dtype == DType.int64
):
    """`peak_prominences` with the peaks as the `int64` tensor `find_peaks`
    returns, and the prominences as a tensor on `x`'s device. At `gpu=True`
    one thread per peak walks out to its saddles; the bounds check is a
    device `min` and `max` over `peaks`. The list form above is the host
    walk this matches."""
    comptime n = dim[T, 0]
    comptime dtype = T.dtype
    var count = peaks.size()
    var ctx = x.context()
    if _check_device[T, gpu](x) and _check_device[P, gpu](peaks):
        comptime if gpu:
            var result = Dynamic[dtype, 1]._uninitialized(
                ctx, row_major(_dyn_shape[1](count))
            )
            if count == 0:
                return result^
            var lo = Static[P.dtype, 1](ctx)
            var hi = Static[P.dtype, 1](ctx)

            @always_inline
            def identity[
                w: Int
            ](tile: SIMD[P.dtype, w], idx: RowCoord[1]) {} -> SIMD[P.dtype, w]:
                return tile

            reduce_all[monoid="min", gpu=True](
                _flat_unchecked(peaks),
                lo.tile(),
                identity,
                count,
                Optional(ctx),
            )
            reduce_all[monoid="max", gpu=True](
                _flat_unchecked(peaks),
                hi.tile(),
                identity,
                count,
                Optional(ctx),
            )
            var smallest = Int(lo.to_host()[0])
            var largest = Int(hi.to_host()[0])
            if smallest < 0 or largest >= n:
                raise Error(
                    "peak_prominences: peak index ",
                    smallest if smallest < 0 else largest,
                    " is outside a signal of ",
                    n,
                    " samples",
                )
            var xv = _flat_unchecked(x)
            var pv = _flat_unchecked(peaks)
            var rv = _flat_out(result)

            @always_inline
            def walk[
                width: Int, alignment: Int = 1
            ](coord: Coord) {var xv, var pv, var rv}:
                var at = Int(pv[coord][0])
                var top = xv[Coord(at)][0]
                var left_min = top
                var i = at
                while i > 0:
                    i -= 1
                    var v = xv[Coord(i)][0]
                    if v > top:
                        break
                    left_min = min(left_min, v)
                var right_min = top
                var j = at
                while j < n - 1:
                    j += 1
                    var v = xv[Coord(j)][0]
                    if v > top:
                        break
                    right_min = min(right_min, v)
                rv.store[1](coord, top - max(left_min, right_min))

            elementwise[simd_width=1, target="gpu"](walk, Coord(count), ctx)
            ctx.synchronize()
            return result^
    else:
        _notice[gpu]("peak_prominences")
    var raw = peaks.to_host()
    var listed = List[Int](capacity=count)
    for i in range(count):
        listed.append(Int(raw[i]))
    var values = peak_prominences(x, listed)
    var out = List[Scalar[dtype]](capacity=count)
    for v in values:
        out.append(Scalar[dtype](v))
    return asarray(out^, ctx)
