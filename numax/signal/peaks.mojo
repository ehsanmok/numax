"""Peak finding over `numax.core.array.Tensor`: `find_peaks` with SciPy's
`height`, `threshold` and `distance` conditions.

**Tier 2, host-side**, and honestly so: the number of peaks is a property
of the data, so the result is a `List[Int]` of indices like `nonzero`'s
rather than a tensor of compile-time length, and the `distance` condition
is a greedy pass in order of peak height that has no independent lanes. One
download of the signal, `O(n)` for the maxima and `O(p log p)` for the
distance sort. MAX has nothing for this; **extend**.
"""

from std.builtin.sort import sort as _sort

from ..core.array import Static


def find_peaks[
    dtype: DType, n: Int
](
    mut x: Static[dtype, n],
    height: Optional[Float64] = None,
    threshold: Optional[Float64] = None,
    distance: Optional[Int] = None,
) raises -> List[Int] where (dtype.is_floating_point() and n > 0):
    """The indices of the local maxima of `x`, ascending, filtered by
    SciPy's conditions. `scipy.signal.find_peaks(x, height, threshold,
    distance)`, first return value.

    A local maximum is a sample strictly greater than both neighbours; a
    flat top counts once, at its midpoint, as SciPy counts it, and the two
    end samples never do. `height` keeps peaks at least that high,
    `threshold` those that stand at least that far above *both* neighbours,
    and `distance` thins peaks closer than that many samples, keeping the
    higher one first -- SciPy's greedy order, so the same peaks survive.
    `prominence`, `width` and `plateau_size` are not provided.
    """
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
        var gap = distance.value()
        var count = len(peaks)
        # Highest first; ties by position, which is SciPy's stable order.
        var order = List[Int](capacity=count)
        for j in range(count):
            order.append(j)

        @parameter
        def higher(a: Int, b: Int) -> Bool:
            var ha = Float64(xs[peaks[a]])
            var hb = Float64(xs[peaks[b]])
            return ha > hb or (ha == hb and a < b)

        _sort[higher](order)
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
        peaks = kept^

    return peaks^


def peak_prominences[
    dtype: DType, n: Int
](mut x: Static[dtype, n], peaks: List[Int]) raises -> List[Float64] where (
    dtype.is_floating_point() and n > 0
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
