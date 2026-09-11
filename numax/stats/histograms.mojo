"""Binning over `numax.core.array.Tensor`: `histogram`, `histogram2d`,
`histogramdd`, `bincount` and `digitize`, with NumPy's edge rules.

**Tier 2, host-side**, the way `median` and `quantile` are: a histogram
is a scatter into bins -- every sample increments one counter -- and MAX
ships no scatter-add or bucketize kernel to build a device path on
(`docs/parity.md` records that `nn.gather_scatter` gathers; its scatter
is a plain store, not an accumulate). The samples come to the host once,
the bins are found by arithmetic or bisection, and the counts go back up as
a tensor. The bin *count* is a compile-time parameter because it shapes
the result.

## NumPy's rules, kept

Uniform bins are `linspace(low, high, bins + 1)` with the last edge
inclusive, so a sample exactly at `high` lands in the last bin; samples
outside `[low, high]` are dropped; the bin index is the floor of the scaled
position corrected against the actual edges, which is how NumPy keeps a
sample from landing one bin off through rounding. `density=True` divides
by the total count and the bin widths so the histogram integrates to one.
`digitize` is `searchsorted` with the side flipped, and handles decreasing
bins as NumPy does. `bincount` takes non-negative integers and returns a
tensor as long as its largest value plus one, or `minlength`.
"""

from std.math import floor as _floor

from layout.tile_layout import TensorLayout
from max.gpu.host import DeviceContext

from ..core.array import Dynamic, Static, Tensor, asarray


def _as_float64[dtype: DType](values: List[Scalar[dtype]]) -> List[Float64]:
    var out = List[Float64](capacity=len(values))
    for i in range(len(values)):
        out.append(Float64(values[i]))
    return out^


def _uniform_edges(low: Float64, high: Float64, bins: Int) -> List[Float64]:
    """`numpy.linspace(low, high, bins + 1)`: `low + i * step` with the last
    edge set to `high` exactly."""
    var edges = List[Float64](capacity=bins + 1)
    var step = (high - low) / Float64(bins)
    for i in range(bins + 1):
        edges.append(low + Float64(i) * step)
    edges[bins] = high
    return edges^


def _data_range(values: List[Float64]) raises -> Tuple[Float64, Float64]:
    """NumPy's default range: the data's `[min, max]`, widened to `[v - 0.5,
    v + 0.5]` when every sample is the same value."""
    if len(values) == 0:
        raise Error("histogram: no samples")
    var lo = values[0]
    var hi = values[0]
    for i in range(1, len(values)):
        var v = values[i]
        if v != v:
            raise Error("histogram: the samples contain NaN")
        lo = min(lo, v)
        hi = max(hi, v)
    if lo == hi:
        return (lo - 0.5, hi + 0.5)
    return (lo, hi)


def _uniform_bin(v: Float64, edges: List[Float64], bins: Int) -> Int:
    """The bin of `v` among uniform `edges`, `-1` when outside. NumPy's
    scaled floor with its two corrections against the real edges, and the
    last edge inclusive."""
    var low = edges[0]
    var high = edges[bins]
    if v < low or v > high:
        return -1
    if v == high:
        return bins - 1
    var index = Int(_floor((v - low) * Float64(bins) / (high - low)))
    if index >= bins:
        index = bins - 1
    if index < 0:
        index = 0
    if v < edges[index]:
        index -= 1
    elif index + 1 < bins and v >= edges[index + 1]:
        index += 1
    return index


def _edge_bin(v: Float64, edges: List[Float64]) -> Int:
    """The bin of `v` among arbitrary ascending `edges` by bisection, `-1`
    when outside; the last edge inclusive."""
    var bins = len(edges) - 1
    if v < edges[0] or v > edges[bins]:
        return -1
    if v == edges[bins]:
        return bins - 1
    var lo = 0
    var hi = bins
    while hi - lo > 1:
        var mid = (lo + hi) // 2
        if edges[mid] <= v:
            lo = mid
        else:
            hi = mid
    return lo


def _count[
    dtype: DType, bins: Int
](
    ctx: DeviceContext,
    values: List[Float64],
    weights: List[Float64],
    edges: List[Float64],
    uniform: Bool,
    density: Bool,
) raises -> Histogram[dtype, bins]:
    """The shared tally: one pass over the samples, then the density
    normalization and the two uploads."""
    var counts = List[Float64](length=bins, fill=0.0)
    for i in range(len(values)):
        var b = _uniform_bin(values[i], edges, bins) if uniform else _edge_bin(
            values[i], edges
        )
        if b >= 0:
            counts[b] += weights[i] if len(weights) > 0 else 1.0
    if density:
        var total = 0.0
        for b in range(bins):
            total += counts[b]
        for b in range(bins):
            var width = edges[b + 1] - edges[b]
            counts[b] = counts[b] / (total * width) if total > 0 else 0.0
    var count_values = List[Scalar[dtype]](capacity=bins)
    for b in range(bins):
        count_values.append(Scalar[dtype](counts[b]))
    var edge_values = List[Scalar[dtype]](capacity=bins + 1)
    for b in range(bins + 1):
        edge_values.append(Scalar[dtype](edges[b]))
    return Histogram[dtype, bins](
        Static[dtype, bins](ctx, count_values^),
        Static[dtype, bins + 1](ctx, edge_values^),
    )


struct Histogram[dtype: DType, bins: Int](Movable):
    """What `histogram` returns: `numpy.histogram`'s `(hist, bin_edges)`,
    the counts (or densities, or summed weights) and the `bins + 1` edges,
    as a struct since a tuple of tensors does not destructure."""

    var counts: Static[Self.dtype, Self.bins]
    var edges: Static[Self.dtype, Self.bins + 1]

    def __init__(
        out self,
        var counts: Static[Self.dtype, Self.bins],
        var edges: Static[Self.dtype, Self.bins + 1],
    ):
        self.counts = counts^
        self.edges = edges^


def histogram[
    dtype: DType, LayoutType: TensorLayout, bins: Int = 10
](
    xs: Tensor[dtype, LayoutType],
    low: Optional[Float64] = None,
    high: Optional[Float64] = None,
    density: Bool = False,
) raises -> Histogram[dtype, bins] where (
    dtype.is_floating_point() and bins > 0
):
    """The histogram of every element of `xs` over `bins` equal-width bins
    spanning `[low, high]` -- the data's range by default.
    `numpy.histogram(a, bins, range=(low, high), density)`.

    NumPy's rules to the count: the last edge is inclusive, samples outside
    the range are dropped, and the bin index is corrected against the real
    edges so rounding never moves a sample one bin over. `bins` is a
    compile-time parameter because it shapes the result; the explicit-edges
    form takes an `edges` tensor instead. Host-side, one pass.
    """
    var values = _as_float64(xs.to_host())
    var span = _data_range(values)
    var lo = low.value() if low else span[0]
    var hi = high.value() if high else span[1]
    if hi <= lo:
        raise Error("histogram: high must exceed low")
    return _count[dtype, bins](
        xs.context(),
        values,
        List[Float64](),
        _uniform_edges(lo, hi, bins),
        True,
        density,
    )


def histogram[
    dtype: DType, LayoutType: TensorLayout, bins: Int = 10
](
    xs: Tensor[dtype, LayoutType],
    weights: Tensor[dtype, LayoutType],
    low: Optional[Float64] = None,
    high: Optional[Float64] = None,
    density: Bool = False,
) raises -> Histogram[dtype, bins] where (
    dtype.is_floating_point() and bins > 0
):
    """`histogram` with each sample contributing its weight rather than
    one. `numpy.histogram(a, bins, weights=w)`; `density` normalizes the
    weighted total."""
    var values = _as_float64(xs.to_host())
    var span = _data_range(values)
    var lo = low.value() if low else span[0]
    var hi = high.value() if high else span[1]
    if hi <= lo:
        raise Error("histogram: high must exceed low")
    return _count[dtype, bins](
        xs.context(),
        values,
        _as_float64(weights.to_host()),
        _uniform_edges(lo, hi, bins),
        True,
        density,
    )


def histogram[
    dtype: DType, LayoutType: TensorLayout, m: Int
](
    xs: Tensor[dtype, LayoutType],
    edges: Static[dtype, m],
    density: Bool = False,
) raises -> Histogram[dtype, m - 1] where (
    dtype.is_floating_point() and m >= 2
):
    """`histogram` over the `m - 1` bins between the given ascending
    `edges`, which need not be uniform. `numpy.histogram(a, bins=edges)`.
    The bin is found by bisection; the last edge is inclusive.

    When `edges` happens to have exactly `xs`'s shape this call is
    ambiguous with the weighted uniform form; spell the parameter list out
    (`histogram[bins=...]`) in that one case.
    """
    return _count[dtype, m - 1](
        xs.context(),
        _as_float64(xs.to_host()),
        List[Float64](),
        _as_float64(edges.to_host()),
        False,
        density,
    )


def histogram[
    dtype: DType, LayoutType: TensorLayout, m: Int
](
    xs: Tensor[dtype, LayoutType],
    edges: Static[dtype, m],
    weights: Tensor[dtype, LayoutType],
    density: Bool = False,
) raises -> Histogram[dtype, m - 1] where (
    dtype.is_floating_point() and m >= 2
):
    """The explicit-edges `histogram` with weights."""
    return _count[dtype, m - 1](
        xs.context(),
        _as_float64(xs.to_host()),
        _as_float64(weights.to_host()),
        _as_float64(edges.to_host()),
        False,
        density,
    )


struct Histogram2D[dtype: DType, xbins: Int, ybins: Int](Movable):
    """What `histogram2d` returns: `numpy.histogram2d`'s `(H, xedges,
    yedges)`, `counts[i, j]` the samples with `x` in bin `i` and `y` in
    bin `j`."""

    var counts: Static[Self.dtype, Self.xbins, Self.ybins]
    var xedges: Static[Self.dtype, Self.xbins + 1]
    var yedges: Static[Self.dtype, Self.ybins + 1]

    def __init__(
        out self,
        var counts: Static[Self.dtype, Self.xbins, Self.ybins],
        var xedges: Static[Self.dtype, Self.xbins + 1],
        var yedges: Static[Self.dtype, Self.ybins + 1],
    ):
        self.counts = counts^
        self.xedges = xedges^
        self.yedges = yedges^


def histogram2d[
    dtype: DType, n: Int, xbins: Int = 10, ybins: Int = 10
](
    mut x: Static[dtype, n], mut y: Static[dtype, n], density: Bool = False
) raises -> Histogram2D[dtype, xbins, ybins] where (
    dtype.is_floating_point() and n > 0 and xbins > 0 and ybins > 0
):
    """The joint histogram of the paired samples `(x[i], y[i])` over an
    `xbins x ybins` grid of equal-width cells spanning each sample's
    range. `numpy.histogram2d(x, y, bins=(xbins, ybins), density)`.

    The same edge rules as `histogram` along each axis, a sample counted
    only when it is inside both ranges. `histogramdd` is the same tally at
    any rank; this is its two-axis spelling with the edges as tensors.
    """
    var xs = _as_float64(x.to_host())
    var ys = _as_float64(y.to_host())
    var xspan = _data_range(xs)
    var yspan = _data_range(ys)
    var xe = _uniform_edges(xspan[0], xspan[1], xbins)
    var ye = _uniform_edges(yspan[0], yspan[1], ybins)
    var counts = List[Float64](length=xbins * ybins, fill=0.0)
    for i in range(n):
        var bx = _uniform_bin(xs[i], xe, xbins)
        var by = _uniform_bin(ys[i], ye, ybins)
        if bx >= 0 and by >= 0:
            counts[bx * ybins + by] += 1.0
    if density:
        var total = 0.0
        for k in range(xbins * ybins):
            total += counts[k]
        for bx in range(xbins):
            for by in range(ybins):
                var area = (xe[bx + 1] - xe[bx]) * (ye[by + 1] - ye[by])
                counts[bx * ybins + by] = (
                    counts[bx * ybins + by] / (total * area) if total
                    > 0 else 0.0
                )
    var ctx = x.context()
    var count_values = List[Scalar[dtype]](capacity=xbins * ybins)
    for k in range(xbins * ybins):
        count_values.append(Scalar[dtype](counts[k]))
    var xedge_values = List[Scalar[dtype]](capacity=xbins + 1)
    for k in range(xbins + 1):
        xedge_values.append(Scalar[dtype](xe[k]))
    var yedge_values = List[Scalar[dtype]](capacity=ybins + 1)
    for k in range(ybins + 1):
        yedge_values.append(Scalar[dtype](ye[k]))
    return Histogram2D[dtype, xbins, ybins](
        Static[dtype, xbins, ybins](ctx, count_values^),
        Static[dtype, xbins + 1](ctx, xedge_values^),
        Static[dtype, ybins + 1](ctx, yedge_values^),
    )


struct HistogramDD[dtype: DType, *bins: Int](Movable):
    """What `histogramdd` returns: `numpy.histogramdd`'s `(H, edges)`, the
    counts at the grid's own shape and one edge list per dimension. The
    edges are host lists rather than tensors because the dimensions'
    bin counts differ and a ragged set is what NumPy returns too."""

    var counts: Static[Self.dtype, *Self.bins]
    var edges: List[List[Float64]]

    def __init__(
        out self,
        var counts: Static[Self.dtype, *Self.bins],
        var edges: List[List[Float64]],
    ):
        self.counts = counts^
        self.edges = edges^


def histogramdd[
    dtype: DType, n: Int, d: Int, //, *bins: Int
](mut points: Static[dtype, n, d], density: Bool = False) raises -> HistogramDD[
    dtype, *bins
] where (dtype.is_floating_point() and n > 0 and d > 0):
    """The `d`-dimensional histogram of `n` points, row `i` of `points`
    being one sample, over a grid with `bins[k]` equal-width cells along
    dimension `k`. `numpy.histogramdd(sample, bins=(b0, b1, ...),
    density)`.

    One bin count per dimension, as the explicit compile-time parameters
    (`histogramdd[3, 4](points)`; the dtype and shape are inferred), so the
    count tensor has the grid's shape. `histogram2d` is the `d = 2` case
    with the edges as tensors.
    """
    comptime dims = len(bins)
    comptime assert dims == d, "histogramdd: one bin count per dimension"
    var host = points.to_host()
    var ctx = points.context()

    var edges = List[List[Float64]]()
    var extents = List[Int]()
    comptime for k in range(dims):
        var column = List[Float64](capacity=n)
        for i in range(n):
            column.append(Float64(host[i * d + k]))
        var span = _data_range(column)
        edges.append(_uniform_edges(span[0], span[1], bins[k]))
        extents.append(bins[k])

    var total_cells = 1
    for k in range(dims):
        total_cells *= extents[k]
    var counts = List[Float64](length=total_cells, fill=0.0)
    for i in range(n):
        var flat = 0
        var inside = True
        for k in range(dims):
            var b = _uniform_bin(Float64(host[i * d + k]), edges[k], extents[k])
            if b < 0:
                inside = False
                break
            flat = flat * extents[k] + b
        if inside:
            counts[flat] += 1.0
    if density:
        var total = 0.0
        for c in range(total_cells):
            total += counts[c]
        for c in range(total_cells):
            var volume = 1.0
            var rest = c
            for step in range(dims):
                var k = dims - 1 - step
                var b = rest % extents[k]
                rest //= extents[k]
                volume *= edges[k][b + 1] - edges[k][b]
            counts[c] = counts[c] / (total * volume) if total > 0 else 0.0
    var count_values = List[Scalar[dtype]](capacity=total_cells)
    for c in range(total_cells):
        count_values.append(Scalar[dtype](counts[c]))
    return HistogramDD[dtype, *bins](
        Static[dtype, *bins](ctx, count_values^), edges^
    )


def bincount[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType], minlength: Int = 0) raises -> Dynamic[
    DType.int64, 1
] where dtype.is_integral():
    """How often each non-negative integer occurs in `xs`: `out[v]` is the
    count of `v`, and the result is `max(xs) + 1` long or `minlength`,
    whichever is greater. `numpy.bincount(x, minlength)`. A negative value
    raises, as NumPy's does.
    """
    var host = xs.to_host()
    var largest = -1
    for i in range(len(host)):
        var v = Int(host[i])
        if v < 0:
            raise Error("bincount: values must be non-negative")
        largest = max(largest, v)
    var length = max(largest + 1, minlength)
    var counts = List[Scalar[DType.int64]](length=length, fill=0)
    for i in range(len(host)):
        counts[Int(host[i])] += 1
    return asarray(counts^, xs.context())


def bincount[
    dtype: DType, LayoutType: TensorLayout, wdtype: DType
](
    xs: Tensor[dtype, LayoutType],
    weights: Tensor[wdtype, LayoutType],
    minlength: Int = 0,
) raises -> Dynamic[wdtype, 1] where (
    dtype.is_integral() and wdtype.is_floating_point()
):
    """`bincount` summing each value's weight instead of counting it.
    `numpy.bincount(x, weights=w, minlength)`."""
    var host = xs.to_host()
    var ws = weights.to_host()
    var largest = -1
    for i in range(len(host)):
        var v = Int(host[i])
        if v < 0:
            raise Error("bincount: values must be non-negative")
        largest = max(largest, v)
    var length = max(largest + 1, minlength)
    var totals = List[Scalar[wdtype]](length=length, fill=0)
    for i in range(len(host)):
        totals[Int(host[i])] += ws[i]
    return asarray(totals^, xs.context())


def digitize[
    dtype: DType, LayoutType: TensorLayout, m: Int
](
    xs: Tensor[dtype, LayoutType], bins: Static[dtype, m], right: Bool = False
) raises -> Dynamic[DType.int64, 1] where (dtype.is_floating_point() and m > 0):
    """The index of the bin each element of `xs` falls in, for monotonic
    `bins`: `bins[i-1] <= x < bins[i]` by default, `bins[i-1] < x <=
    bins[i]` with `right`, `0` below the first edge and `m` past the last.
    `numpy.digitize(x, bins, right)`.

    `searchsorted` with the side flipped, which is exactly NumPy's
    definition, and decreasing `bins` handled as NumPy handles them, by
    searching the reversed edges and counting from the far end. Host-side,
    one bisection per element.
    """
    var edges = _as_float64(bins.to_host())
    var increasing = m == 1 or edges[m - 1] >= edges[0]
    if not increasing:
        var reversed = List[Float64](capacity=m)
        for i in range(m):
            reversed.append(edges[m - 1 - i])
        edges = reversed^
    var host = xs.to_host()
    var out = List[Scalar[DType.int64]](capacity=len(host))
    for i in range(len(host)):
        var v = Float64(host[i])
        # `side="right"` when not `right`, so an `x` equal to an edge lands in
        # the bin above it. The same side for decreasing edges: reversing
        # them already turns `bins[i-1] > x >= bins[i]` into the ascending
        # rule, and the count from the far end does the rest.
        var take_right = not right
        var lo = 0
        var hi = m
        while lo < hi:
            var mid = (lo + hi) // 2
            var goes_left = edges[mid] > v if take_right else edges[mid] >= v
            if goes_left:
                hi = mid
            else:
                lo = mid + 1
        out.append(Int64(lo if increasing else m - lo))
    return asarray(out^, xs.context())
