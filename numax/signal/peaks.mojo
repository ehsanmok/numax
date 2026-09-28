"""Peak finding over `numax.core.tensor.Tensor`: `find_peaks` with SciPy's
`height`, `threshold` and `distance` conditions, `peak_prominences`,
`peak_widths`, and the relative extrema `argrelextrema`/`argrelmax`/
`argrelmin`.

**Tier 2.** The number of peaks is a property of the data, so the result
is a run-time-length `int64` tensor of indices, like `nonzero`'s. At
`gpu=True` one launch marks every local maximum (walking a flat top to its
midpoint) with the `height` and `threshold` conditions applied, and the
device compaction packs them; the `distance` condition is a greedy pass in
order of peak height that has no independent lanes, so it alone reads the
`p` peaks back and runs on the host -- a declared exception, `O(p log p)`.
`peak_prominences` over a tensor of peaks is one thread per peak walking
out to its saddles, and `peak_widths` the same walk followed by the width
walk; `argrelextrema` is one flag per sample and the device compaction.
MAX has nothing for this; **extend**.
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

    Parameters:
        T: The `TensorLike` type of `x`, rank 1, row-major, with a static
            nonzero length.
        gpu: Whether the marking and compaction run on the GPU; a mismatch
            with `x`'s residency falls back to the host path under the
            `set_fallback` policy.

    Args:
        x: The signal to search.
        height: The minimum peak value, or `None` for no height condition.
        threshold: The minimum rise above both neighbors, or `None`.
        distance: The minimum spacing in samples between kept peaks, or
            `None`; the higher peak wins a conflict.

    Returns:
        A run-time-length `int64` tensor on `x`'s device holding the
        surviving peak indices in ascending order.

    Raises:
        If a device transfer or launch fails, or if the fallback policy is
        `"raise"` and `gpu` disagrees with `x`'s residency.
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


struct PeakWidths[dtype: DType](Movable):
    """What `peak_widths` returns, SciPy's four arrays: each peak's width,
    the height it is measured at, and the interpolated positions where the
    width line crosses the signal on either side."""

    var widths: Dynamic[Self.dtype, 1]
    """The width of each peak, in samples."""
    var width_heights: Dynamic[Self.dtype, 1]
    """The height at which each width is measured."""
    var left_ips: Dynamic[Self.dtype, 1]
    """Where each width line meets the signal on the left, interpolated."""
    var right_ips: Dynamic[Self.dtype, 1]
    """Where each width line meets the signal on the right, interpolated."""

    def __init__(
        out self,
        var widths: Dynamic[Self.dtype, 1],
        var width_heights: Dynamic[Self.dtype, 1],
        var left_ips: Dynamic[Self.dtype, 1],
        var right_ips: Dynamic[Self.dtype, 1],
    ):
        """Build from the four per-peak arrays.

        Args:
            widths: The widths.
            width_heights: The heights they are measured at.
            left_ips: The left crossings.
            right_ips: The right crossings.
        """
        self.widths = widths^
        self.width_heights = width_heights^
        self.left_ips = left_ips^
        self.right_ips = right_ips^


@always_inline
def _peak_width[
    dtype: DType
](
    xp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    peak: Int,
    rel_height: Scalar[dtype],
) -> Tuple[Scalar[dtype], Scalar[dtype], Scalar[dtype], Scalar[dtype]]:
    """One peak's width, SciPy's `_peak_prominences` then `_peak_widths`:
    walk out to the bases (the lowest sample before taller ground on each
    side, the first one met on a tie), take the prominence, and walk from
    the peak to where the signal drops below `x[peak] - rel_height *
    prominence`, interpolating the crossing linearly. Straight-line code
    over a pointer, so one device thread runs it per peak."""
    var height = xp[peak]
    var left_min = height
    var left_base = peak
    var i = peak
    while i >= 0 and xp[i] <= height:
        if xp[i] < left_min:
            left_min = xp[i]
            left_base = i
        i -= 1
    var right_min = height
    var right_base = peak
    i = peak
    while i < n and xp[i] <= height:
        if xp[i] < right_min:
            right_min = xp[i]
            right_base = i
        i += 1
    var prominence = height - (left_min if left_min > right_min else right_min)
    var level = height - prominence * rel_height
    i = peak
    while left_base < i and level < xp[i]:
        i -= 1
    var left = Scalar[dtype](i)
    if xp[i] < level:
        left += (level - xp[i]) / (xp[i + 1] - xp[i])
    i = peak
    while i < right_base and level < xp[i]:
        i += 1
    var right = Scalar[dtype](i)
    if xp[i] < level:
        right -= (level - xp[i]) / (xp[i - 1] - xp[i])
    return (right - left, level, left, right)


def peak_widths[
    T: TensorLike, P: TensorLike, gpu: Bool = False
](x: T, peaks: P, rel_height: Float64 = 0.5) raises -> PeakWidths[
    T.dtype
] where (
    (T.dtype.is_floating_point() and dim[T, 0] > 0)
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
    and P.dtype == DType.int64
):
    """The width of each peak at a relative height of its prominence.
    `scipy.signal.peak_widths(x, peaks, rel_height)`.

    `rel_height = 0.5` (the default) is the full width at half prominence;
    `1` measures at the lowest contour line, the bases. The prominence
    and its bases are computed along the way exactly as
    `peak_prominences` does, and the crossings are interpolated linearly,
    as SciPy does. At `gpu=True` one thread per peak does the whole walk.

    Parameters:
        T: The tensor type of `x`, rank 1, static length, floating-point.
        P: The tensor type of `peaks`, `int64`, as `find_peaks` returns.
        gpu: Run on `x`'s device when `True` and both tensors live there; a
            residency mismatch falls back to the host with a notice.

    Args:
        x: The signal.
        peaks: The peak indices.
        rel_height: The fraction of the prominence below the peak the width
            is measured at, at least 0.

    Returns:
        A `PeakWidths` with the widths, their heights, and the left and right
        interpolated crossings, one entry per peak.

    Raises:
        If `rel_height` is negative, a peak index is outside the signal, or
        a device operation fails.
    """
    comptime n = dim[T, 0]
    comptime dtype = T.dtype
    if rel_height < 0:
        raise Error("peak_widths: rel_height must be at least 0")
    var count = peaks.size()
    var ctx = x.context()
    var level = Scalar[dtype](rel_height)
    var host_peaks = peaks.to_host()
    for p in range(count):
        if host_peaks[p] < 0 or Int(host_peaks[p]) >= n:
            raise Error(
                "peak_widths: peak index ",
                Int(host_peaks[p]),
                " is outside a signal of ",
                n,
                " samples",
            )
    if _check_device[T, gpu](x) and _check_device[P, gpu](peaks):
        comptime if gpu:
            var shape = row_major(_dyn_shape[1](count))
            var w = Dynamic[dtype, 1]._uninitialized(ctx, shape)
            var h = Dynamic[dtype, 1]._uninitialized(ctx, shape)
            var l = Dynamic[dtype, 1]._uninitialized(ctx, shape)
            var r = Dynamic[dtype, 1]._uninitialized(ctx, shape)
            if count > 0:
                var xp = x.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
                var pp = peaks.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
                var wp = w.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
                var hp = h.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
                var lp = l.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
                var rp = r.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

                @always_inline
                def lane[
                    width: Int, alignment: Int = 1
                ](coord: Coord) {
                    var xp, var pp, var wp, var hp, var lp, var rp, var level
                }:
                    var k = coord_to_index_list(coord)[0]
                    var peak = Int(pp[unsafe_offset=k])
                    var got = _peak_width[dtype](
                        rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](xp),
                        n,
                        peak,
                        level,
                    )
                    wp[unsafe_offset=k] = got[0]
                    hp[unsafe_offset=k] = got[1]
                    lp[unsafe_offset=k] = got[2]
                    rp[unsafe_offset=k] = got[3]

                elementwise[simd_width=1, target="gpu"](lane, Coord(count), ctx)
                ctx.synchronize()
            return PeakWidths(w^, h^, l^, r^)
    else:
        _notice[gpu]("peak_widths")
    var xs = x.to_host()
    var xp = xs.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var widths = List[Scalar[dtype]](capacity=count)
    var heights = List[Scalar[dtype]](capacity=count)
    var lefts = List[Scalar[dtype]](capacity=count)
    var rights = List[Scalar[dtype]](capacity=count)
    for p in range(count):
        var got = _peak_width[dtype](xp, n, Int(host_peaks[p]), level)
        widths.append(got[0])
        heights.append(got[1])
        lefts.append(got[2])
        rights.append(got[3])
    _ = xs^
    return PeakWidths(
        asarray(widths^, ctx),
        asarray(heights^, ctx),
        asarray(lefts^, ctx),
        asarray(rights^, ctx),
    )


def argrelextrema[
    T: TensorLike,
    comparator: StaticString,
    order: Int = 1,
    mode: StaticString = "clip",
    gpu: Bool = False,
](data: T) raises -> Dynamic[DType.int64, 1] where (
    T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
    and dim[T, 0] > 0
    and order >= 1
):
    """The indices where `data` is a relative extremum over `order`
    neighbors on each side. `scipy.signal.argrelextrema(data, comparator,
    order=order, mode=mode)` at rank 1.

    Sample `i` is kept when `comparator(data[i], data[i +- s])` holds for
    every `s` in `1..order`: `"greater"` or `"less"` (strict, so a flat top
    is no extremum), `"greater_equal"` or `"less_equal"`. `mode` is how a
    neighbor past the end is read: `"clip"` (the default) repeats the end
    sample, so a strict comparison never keeps an end, and `"wrap"` goes
    round. One launch flags every sample; the device compaction packs the
    indices, of which only the count comes back.

    Parameters:
        T: The tensor type of `data`, rank 1, static length.
        comparator: `"greater"`, `"less"`, `"greater_equal"` or
            `"less_equal"`.
        order: How many neighbors on each side to compare against.
        mode: `"clip"` or `"wrap"`.
        gpu: Run on `data`'s device when `True`; a residency mismatch falls
            back to the host with a notice.

    Args:
        data: The signal.

    Returns:
        A `Dynamic` rank-1 `int64` tensor of the extrema's indices,
        ascending.

    Raises:
        If the fallback policy is `"raise"` on a residency mismatch, or a
        device operation fails.
    """
    comptime assert (
        comparator == "greater"
        or comparator == "less"
        or comparator == "greater_equal"
        or comparator == "less_equal"
    ), (
        'argrelextrema: comparator is "greater", "less", "greater_equal" or'
        ' "less_equal"'
    )
    comptime assert (
        mode == "clip" or mode == "wrap"
    ), 'argrelextrema: mode is "clip" or "wrap"'
    comptime n = dim[T, 0]
    comptime dtype = T.dtype
    var ctx = data.context()
    if _check_device[T, gpu](data):
        comptime if gpu:
            var flags = Dynamic[DType.int64, 1]._uninitialized(
                ctx, row_major(_dyn_shape[1](n))
            )
            var dp = data.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
            var fp = _flat_out(flags)

            @always_inline
            def flag[
                width: Int, alignment: Int = 1
            ](coord: Coord) {var dp, var fp}:
                var i = coord_to_index_list(coord)[0]
                var keep = _is_extremum[dtype, comparator, mode](
                    rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](dp),
                    n,
                    i,
                    order,
                )
                fp.store[1](coord, Int64(1) if keep else Int64(0))

            elementwise[simd_width=1, target="gpu"](flag, Coord(n), ctx)
            return _pack_device[indices=True](flags, flags, n)
    else:
        _notice[gpu]("argrelextrema")
    var xs = data.to_host()
    var xp = xs.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var out = List[Scalar[DType.int64]]()
    for i in range(n):
        if _is_extremum[dtype, comparator, mode](xp, n, i, order):
            out.append(Int64(i))
    _ = xs^
    return asarray(out^, ctx)


@always_inline
def _is_extremum[
    dtype: DType, comparator: StaticString, mode: StaticString
](
    xp: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int, i: Int, order: Int
) -> Bool:
    """`_boolrelextrema` at one sample: the comparison against every
    neighbor within `order`, the neighbor's index clipped or wrapped."""
    var here = xp[i]
    for s in range(1, order + 1):
        var up = i + s
        var down = i - s
        comptime if mode == "clip":
            up = up if up < n else n - 1
            down = down if down >= 0 else 0
        else:
            up = up % n
            down = ((down % n) + n) % n
        var a = xp[up]
        var b = xp[down]
        comptime if comparator == "greater":
            if not (here > a and here > b):
                return False
        elif comparator == "less":
            if not (here < a and here < b):
                return False
        elif comparator == "greater_equal":
            if not (here >= a and here >= b):
                return False
        else:
            if not (here <= a and here <= b):
                return False
    return True


def argrelmax[
    T: TensorLike,
    order: Int = 1,
    mode: StaticString = "clip",
    gpu: Bool = False,
](data: T) raises -> Dynamic[DType.int64, 1] where (
    T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
    and dim[T, 0] > 0
    and order >= 1
):
    """The strict relative maxima of `data`. `scipy.signal.argrelmax`,
    `argrelextrema` with `"greater"`.

    Parameters:
        T: The tensor type of `data`, rank 1, static length.
        order: How many neighbors on each side to compare against.
        mode: `"clip"` or `"wrap"`.
        gpu: Run on `data`'s device when `True`.

    Args:
        data: The signal.

    Returns:
        The maxima's indices, ascending.

    Raises:
        As `argrelextrema` does.
    """
    return argrelextrema[T, "greater", order, mode, gpu](data)


def argrelmin[
    T: TensorLike,
    order: Int = 1,
    mode: StaticString = "clip",
    gpu: Bool = False,
](data: T) raises -> Dynamic[DType.int64, 1] where (
    T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
    and dim[T, 0] > 0
    and order >= 1
):
    """The strict relative minima of `data`. `scipy.signal.argrelmin`,
    `argrelextrema` with `"less"`.

    Parameters:
        T: The tensor type of `data`, rank 1, static length.
        order: How many neighbors on each side to compare against.
        mode: `"clip"` or `"wrap"`.
        gpu: Run on `data`'s device when `True`.

    Args:
        data: The signal.

    Returns:
        The minima's indices, ascending.

    Raises:
        As `argrelextrema` does.
    """
    return argrelextrema[T, "less", order, mode, gpu](data)
