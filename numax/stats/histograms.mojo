"""Binning over `numax.core.tensor.Tensor`: `histogram`, `histogram2d`,
`histogramdd`, `bincount` and `digitize`, with NumPy's edge rules.

**Tier 2.** A histogram is a scatter into bins -- every sample increments
one counter -- and MAX ships no scatter-add or bucketize kernel to build on
(`docs/parity.md` records that `nn.gather_scatter` gathers; its scatter is
a plain store, not an accumulate). So the device path is numax's own: at
`gpu=True` one launch bins every sample by the same arithmetic or
bisection the host uses and adds its `1`, or its weight, with an atomic
add (`std.atomic`). The data range, the NaN check and `bincount`'s extent
are device reductions; the `bins + 1` edges and the density scaling are
host arithmetic over a handful of numbers. On the host the samples come
down once and are counted in a loop. The bin *count* is a compile-time
parameter because it shapes the result.

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

from algorithm.rowwise_types import RowCoord
from layout import Coord, coord_to_index_list
from layout.tile_layout import TensorLayout, row_major
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.atomic import Atomic
from std.utils import IndexList

from ..core._drive import (
    _check_device,
    _flat_unchecked,
    _notice,
    _require_contiguous,
)
from ..core.ops import astype
from ..core.rowwise import reduce_all
from ..core.tensorlike import TensorLike, dim, is_row_major
from ..core.tensor import (
    _canonical,
    _dyn_shape,
    Dynamic,
    Static,
    Tensor,
    asarray,
)
from .statistics import max as _smax, min as _smin


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


def _nan_count_device[T: TensorLike](xs: T) raises -> Int:
    """How many NaNs a contiguous GPU-context `xs` holds: one launch and an
    atomic counter."""
    var ctx = xs.context()
    var nans = Static[DType.int32, 1](ctx)
    var np_ = nans.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var xp = xs.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()

    @always_inline
    def spot[width: Int, alignment: Int = 1](coord: Coord) {var np_, var xp}:
        var v = xp[unsafe_offset=coord_to_index_list(coord)[0]]
        if v != v:
            _ = Atomic[Int32].fetch_add(np_, Int32(1))

    if xs.size() > 0:
        elementwise[simd_width=1, target="gpu"](spot, Coord(xs.size()), ctx)
    return Int(nans.to_host()[0])


def _device_range[T: TensorLike](xs: T) raises -> Tuple[Float64, Float64]:
    """`_data_range` over a GPU-context tensor: device `min`/`max` and a
    device NaN count, three scalars back."""
    if xs.size() == 0:
        raise Error("histogram: no samples")
    _require_contiguous(xs)
    var ctx = xs.context()
    if _nan_count_device(xs) > 0:
        raise Error("histogram: the samples contain NaN")
    var lo = Static[T.dtype, 1](ctx)
    var hi = Static[T.dtype, 1](ctx)

    @always_inline
    def identity[
        w: Int
    ](tile: SIMD[T.dtype, w], idx: RowCoord[1]) {} -> SIMD[T.dtype, w]:
        return tile

    reduce_all[monoid="min", gpu=True](
        _flat_unchecked(xs), lo.tile(), identity, xs.size(), Optional(ctx)
    )
    reduce_all[monoid="max", gpu=True](
        _flat_unchecked(xs), hi.tile(), identity, xs.size(), Optional(ctx)
    )
    var low = Float64(lo.to_host()[0])
    var high = Float64(hi.to_host()[0])
    if low == high:
        return (low - 0.5, high + 0.5)
    return (low, high)


@always_inline
def _bin_of[
    dtype: DType
](
    v: Scalar[dtype],
    edges: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    bins: Int,
    uniform: Bool,
) -> Int:
    """`_uniform_bin` or `_edge_bin` in a kernel body: the bin of `v` over
    `bins + 1` edges, `-1` outside, the last edge inclusive."""
    var low = edges[unsafe_offset=0]
    var high = edges[unsafe_offset=bins]
    if v < low or v > high or v != v:
        return -1
    if v == high:
        return bins - 1
    if uniform:
        var index = Int((v - low) * Scalar[dtype](bins) / (high - low))
        index = min(max(index, 0), bins - 1)
        if v < edges[unsafe_offset=index]:
            index -= 1
        elif index + 1 < bins and v >= edges[unsafe_offset=index + 1]:
            index += 1
        return index
    var lo = 0
    var hi = bins
    while hi - lo > 1:
        var mid = (lo + hi) // 2
        if edges[unsafe_offset=mid] <= v:
            lo = mid
        else:
            hi = mid
    return lo


def _count_device[
    dtype: DType, bins: Int, weighted: Bool, T: TensorLike, W: TensorLike
](
    xs: T, weights: W, edges: List[Float64], uniform: Bool, density: Bool
) raises -> Histogram[dtype, bins] where (
    T.dtype == dtype and W.dtype == dtype
):
    """`_count` on the device: one launch bins every sample and adds `1`,
    or its weight, into the bins with an atomic add; the `bins + 1` edges
    and the density normalization are host arithmetic over `bins` numbers.
    """
    _require_contiguous(xs)
    var ctx = xs.context()
    var edge_values = List[Scalar[dtype]](capacity=bins + 1)
    for b in range(bins + 1):
        edge_values.append(Scalar[dtype](edges[b]))
    var edge_d = Static[dtype, bins + 1](edge_values.copy(), ctx)
    var counts = Static[dtype, bins](ctx)
    var n = xs.size()
    if n > 0:
        var xp = (
            xs.tile()
            .ptr.unsafe_bitcast[Scalar[dtype]]()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )
        var wp = (
            weights.tile()
            .ptr.unsafe_bitcast[Scalar[dtype]]()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )
        var ep = edge_d.tile().ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin]()
        var cp = counts.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

        @always_inline
        def tally[
            width: Int, alignment: Int = 1
        ](coord: Coord) {var xp, var wp, var ep, var cp, var uniform}:
            var i = coord_to_index_list(coord)[0]
            var b = _bin_of(xp[unsafe_offset=i], ep, bins, uniform)
            if b >= 0:
                var add = Scalar[dtype](1)
                comptime if weighted:
                    add = wp[unsafe_offset=i]
                _ = Atomic[Scalar[dtype]].fetch_add(cp + b, add)

        elementwise[simd_width=1, target="gpu"](tally, Coord(n), ctx)
        ctx.synchronize()
    if density:
        var raw = counts.to_host()
        var total = 0.0
        for b in range(bins):
            total += Float64(raw[b])
        var scaled = List[Scalar[dtype]](capacity=bins)
        for b in range(bins):
            var width = edges[b + 1] - edges[b]
            scaled.append(
                Scalar[dtype](
                    Float64(raw[b]) / (total * width) if total > 0 else 0.0
                )
            )
        counts = Static[dtype, bins](scaled^, ctx)
    return Histogram[dtype, bins](counts^, edge_d^)


def _cells_device[
    dtype: DType, dims: Int
](
    base: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    second: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    n: Int,
    row_stride: Int,
    edges: List[List[Float64]],
    ctx: DeviceContext,
) raises -> List[Float64]:
    """The multi-dimensional tally on the device: sample `i`'s coordinate
    `k` is `base[i * row_stride + k]`, except that with `second` set (the
    two-vector `histogram2d`) coordinate 1 is `second[i]`. One launch finds
    each sample's cell and atomically adds one; the counts, one per cell,
    come back for the host's density and assembly."""
    var extents = IndexList[dims]()
    var offsets = IndexList[dims]()
    var flat_edges = List[Scalar[dtype]]()
    var cells = 1
    for k in range(dims):
        var bins = len(edges[k]) - 1
        extents[k] = bins
        offsets[k] = len(flat_edges)
        cells *= bins
        for e in edges[k]:
            flat_edges.append(Scalar[dtype](e))
    var edge_d = asarray(flat_edges^, ctx)
    var counts = Dynamic[dtype, 1](row_major(_dyn_shape[1](cells)), ctx)
    var ep = edge_d.tile().ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin]()
    var cp = counts.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var paired = second != base

    @always_inline
    def tally[
        width: Int, alignment: Int = 1
    ](coord: Coord) {
        var base,
        var second,
        var row_stride,
        var ep,
        var cp,
        var extents,
        var offsets,
        var paired,
    }:
        var i = coord_to_index_list(coord)[0]
        var flat = 0
        comptime for k in range(dims):
            var v = base[unsafe_offset=i * row_stride + k]
            if paired and k == 1:
                v = second[unsafe_offset=i]
            var b = _bin_of(v, ep + offsets[k], extents[k], True)
            if b < 0:
                return
            flat = flat * extents[k] + b
        _ = Atomic[Scalar[dtype]].fetch_add(cp + flat, Scalar[dtype](1))

    if n > 0:
        elementwise[simd_width=1, target="gpu"](tally, Coord(n), ctx)
        ctx.synchronize()
    var raw = counts.to_host()
    var out = List[Float64](capacity=cells)
    for c in range(cells):
        out.append(Float64(raw[c]))
    return out^


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
        Static[dtype, bins](count_values^, ctx),
        Static[dtype, bins + 1](edge_values^, ctx),
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
    T: TensorLike, bins: Int = 10, gpu: Bool = False
](
    xs: T,
    low: Optional[Float64] = None,
    high: Optional[Float64] = None,
    density: Bool = False,
) raises -> Histogram[T.dtype, bins] where (
    T.dtype.is_floating_point() and bins > 0
):
    """The histogram of every element of `xs` over `bins` equal-width bins
    spanning `[low, high]` -- the data's range by default.
    `numpy.histogram(a, bins, range=(low, high), density)`.

    NumPy's rules to the count: the last edge is inclusive, samples outside
    the range are dropped, and the bin index is corrected against the real
    edges so rounding never moves a sample one bin over. `bins` is a
    compile-time parameter because it shapes the result; the explicit-edges
    form takes an `edges` tensor instead. Host-side, one pass.

    Parameters:
        T: The tensor type of `xs`, with a floating-point dtype.
        bins: The number of equal-width bins, fixed at compile time.
        gpu: Tally on the tensor's device; a residency mismatch falls back to
            the host with a notice.

    Args:
        xs: The samples, every element counted.
        low: The lower end of the range; the data's minimum when `None`.
        high: The upper end of the range; the data's maximum when `None`.
        density: Divide each bin by the total and its width, so the result
            integrates to one.

    Returns:
        A `Histogram` holding the `bins` counts (or densities) and the
        `bins + 1` edges.

    Raises:
        If `high <= low`, if `xs` is empty or holds NaN, or if a residency
        mismatch occurs under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    if _check_device[T, gpu](xs):
        comptime if gpu:
            var span = _device_range(xs)
            var lo = low.value() if low else span[0]
            var hi = high.value() if high else span[1]
            if hi <= lo:
                raise Error("histogram: high must exceed low")
            return _count_device[dtype, bins, False](
                xs, xs, _uniform_edges(lo, hi, bins), True, density
            )
    else:
        _notice[gpu]("histogram")
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
    T: TensorLike, bins: Int = 10, gpu: Bool = False
](
    xs: T,
    weights: T,
    low: Optional[Float64] = None,
    high: Optional[Float64] = None,
    density: Bool = False,
) raises -> Histogram[T.dtype, bins] where (
    T.dtype.is_floating_point() and bins > 0
):
    """`histogram` with each sample contributing its weight rather than
    one. `numpy.histogram(a, bins, weights=w)`; `density` normalizes the
    weighted total.

    Parameters:
        T: The tensor type of `xs`, with a floating-point dtype.
        bins: The number of equal-width bins, fixed at compile time.
        gpu: Tally on the tensor's device; a residency mismatch falls back to
            the host with a notice.

    Args:
        xs: The samples, every element counted.
        weights: One weight per sample, the same shape as `xs`.
        low: The lower end of the range; the data's minimum when `None`.
        high: The upper end of the range; the data's maximum when `None`.
        density: Divide each bin by the weighted total and its width, so the
            result integrates to one.

    Returns:
        A `Histogram` holding the `bins` summed weights (or densities) and
        the `bins + 1` edges.

    Raises:
        If `high <= low`, if `xs` is empty or holds NaN, if `weights` is not
        contiguous on the device path, or if a residency mismatch occurs under
        the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    if _check_device[T, gpu](xs) and _check_device[T, gpu](weights):
        comptime if gpu:
            var span = _device_range(xs)
            var lo = low.value() if low else span[0]
            var hi = high.value() if high else span[1]
            if hi <= lo:
                raise Error("histogram: high must exceed low")
            _require_contiguous(weights)
            return _count_device[dtype, bins, True](
                xs, weights, _uniform_edges(lo, hi, bins), True, density
            )
    else:
        _notice[gpu]("histogram")
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
    T: TensorLike, m: Int, gpu: Bool = False
](
    xs: T,
    edges: Static[T.dtype, m],
    density: Bool = False,
) raises -> Histogram[
    T.dtype, m - 1
] where (T.dtype.is_floating_point() and m >= 2):
    """`histogram` over the `m - 1` bins between the given ascending
    `edges`, which need not be uniform. `numpy.histogram(a, bins=edges)`.
    The bin is found by bisection; the last edge is inclusive.

    When `edges` happens to have exactly `xs`'s shape this call is
    ambiguous with the weighted uniform form; spell the parameter list out
    (`histogram[bins=...]`) in that one case.

    Parameters:
        T: The tensor type of `xs`, with a floating-point dtype.
        m: The number of edges, giving `m - 1` bins.
        gpu: Tally on the tensor's device; a residency mismatch falls back to
            the host with a notice.

    Args:
        xs: The samples, every element counted.
        edges: The `m` ascending bin edges, the last one inclusive.
        density: Divide each bin by the total and its width, so the result
            integrates to one.

    Returns:
        A `Histogram` holding the `m - 1` counts (or densities) and a copy of
        `edges`.

    Raises:
        If a residency mismatch occurs under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    if _check_device[T, gpu](xs):
        comptime if gpu:
            return _count_device[dtype, m - 1, False](
                xs, xs, _as_float64(edges.to_host()), False, density
            )
    else:
        _notice[gpu]("histogram")
    return _count[dtype, m - 1](
        xs.context(),
        _as_float64(xs.to_host()),
        List[Float64](),
        _as_float64(edges.to_host()),
        False,
        density,
    )


def histogram[
    T: TensorLike, m: Int, gpu: Bool = False
](
    xs: T,
    edges: Static[T.dtype, m],
    weights: T,
    density: Bool = False,
) raises -> Histogram[T.dtype, m - 1] where (
    T.dtype.is_floating_point() and m >= 2
):
    """The explicit-edges `histogram` with weights.

    Parameters:
        T: The tensor type of `xs`, with a floating-point dtype.
        m: The number of edges, giving `m - 1` bins.
        gpu: Tally on the tensor's device; a residency mismatch falls back to
            the host with a notice.

    Args:
        xs: The samples, every element counted.
        edges: The `m` ascending bin edges, the last one inclusive.
        weights: One weight per sample, the same shape as `xs`.
        density: Divide each bin by the weighted total and its width, so the
            result integrates to one.

    Returns:
        A `Histogram` holding the `m - 1` summed weights (or densities) and a
        copy of `edges`.

    Raises:
        If `weights` is not contiguous on the device path, or if a residency
        mismatch occurs under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    if _check_device[T, gpu](xs) and _check_device[T, gpu](weights):
        comptime if gpu:
            _require_contiguous(weights)
            return _count_device[dtype, m - 1, True](
                xs, weights, _as_float64(edges.to_host()), False, density
            )
    else:
        _notice[gpu]("histogram")
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
    A: TensorLike,
    B: TensorLike,
    xbins: Int = 10,
    ybins: Int = 10,
    gpu: Bool = False,
](x: A, y: B, density: Bool = False) raises -> Histogram2D[
    A.dtype, xbins, ybins
] where (
    (A.dtype.is_floating_point() and dim[A, 0] > 0 and xbins > 0 and ybins > 0)
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 0]
):
    """The joint histogram of the paired samples `(x[i], y[i])` over an
    `xbins x ybins` grid of equal-width cells spanning each sample's
    range. `numpy.histogram2d(x, y, bins=(xbins, ybins), density)`.

    The same edge rules as `histogram` along each axis, a sample counted
    only when it is inside both ranges. `histogramdd` is the same tally at
    any rank; this is its two-axis spelling with the edges as tensors.

    Parameters:
        A: The rank-1, static-shape floating-point tensor type of `x`.
        B: The tensor type of `y`, matching `A`'s dtype and length.
        xbins: The number of equal-width bins along `x`.
        ybins: The number of equal-width bins along `y`.
        gpu: Tally on the tensor's device; a residency mismatch falls back to
            the host with a notice.

    Args:
        x: The first coordinate of each sample.
        y: The second coordinate of each sample, paired with `x` by index.
        density: Divide each cell by the total and its area, so the result
            integrates to one.

    Returns:
        A `Histogram2D` holding the `xbins x ybins` counts (or densities) and
        both edge tensors.

    Raises:
        If `x` or `y` holds NaN, if `y` is not contiguous on the device path, or
        if a residency mismatch occurs under the `"raise"` fallback policy.
    """
    comptime dtype = A.dtype
    comptime n = dim[A, 0]
    var xe = List[Float64]()
    var ye = List[Float64]()
    var counts = List[Float64]()
    var on_device = False
    comptime if gpu:
        on_device = _check_device[A, True](x) and _check_device[B, True](y)
        if on_device:
            var xspan = _device_range(x)
            var yspan = _device_range(y)
            xe = _uniform_edges(xspan[0], xspan[1], xbins)
            ye = _uniform_edges(yspan[0], yspan[1], ybins)
            var both = List[List[Float64]]()
            both.append(xe.copy())
            both.append(ye.copy())
            _require_contiguous(y)
            counts = _cells_device[dtype, 2](
                x.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin](),
                y.tile()
                .ptr.unsafe_bitcast[Scalar[dtype]]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                n,
                1,
                both,
                x.context(),
            )
        else:
            _notice[gpu]("histogram2d")
    if not on_device:
        var xs = _as_float64(x.to_host())
        var ys = _as_float64(y.to_host())
        var xspan = _data_range(xs)
        var yspan = _data_range(ys)
        xe = _uniform_edges(xspan[0], xspan[1], xbins)
        ye = _uniform_edges(yspan[0], yspan[1], ybins)
        counts = List[Float64](length=xbins * ybins, fill=0.0)
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
        Static[dtype, xbins, ybins](count_values^, ctx),
        Static[dtype, xbins + 1](xedge_values^, ctx),
        Static[dtype, ybins + 1](yedge_values^, ctx),
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
    T: TensorLike,
    //,
    *bins: Int,
    gpu: Bool = False,
](points: T, density: Bool = False) raises -> HistogramDD[
    T.dtype, *bins
] where (
    (T.dtype.is_floating_point() and dim[T, 0] > 0 and dim[T, 1] > 0)
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
):
    """The `d`-dimensional histogram of `n` points, row `i` of `points`
    being one sample, over a grid with `bins[k]` equal-width cells along
    dimension `k`. `numpy.histogramdd(sample, bins=(b0, b1, ...),
    density)`.

    One bin count per dimension, as the explicit compile-time parameters
    (`histogramdd[3, 4](points)`; the dtype and shape are inferred), so the
    count tensor has the grid's shape. `histogram2d` is the `d = 2` case
    with the edges as tensors.

    Parameters:
        T: The rank-2, static-shape floating-point tensor type of `points`,
            inferred.
        bins: One bin count per dimension, as many as `points` has columns.
        gpu: Tally on the tensor's device; a residency mismatch falls back to
            the host with a notice.

    Args:
        points: The `n x d` samples, one per row.
        density: Divide each cell by the total and its volume, so the result
            integrates to one.

    Returns:
        A `HistogramDD` holding counts (or densities) of shape `bins` and one
        edge list per dimension.

    Raises:
        If `points` holds NaN, if it is not contiguous on the device path, or if
        a residency mismatch occurs under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime n = dim[T, 0]
    comptime d = dim[T, 1]
    comptime dims = len(bins)
    comptime assert dims == d, "histogramdd: one bin count per dimension"
    var ctx = points.context()
    var edges = List[List[Float64]]()
    var extents = List[Int]()
    comptime for k in range(dims):
        extents.append(bins[k])
    var total_cells = 1
    for k in range(dims):
        total_cells *= extents[k]
    var counts = List[Float64]()
    var on_device = False
    comptime if gpu:
        on_device = _check_device[T, True](points)
        if on_device:
            _require_contiguous(points)
            if _nan_count_device(points) > 0:
                raise Error("histogram: the samples contain NaN")
            var lows = _smin[axis=0, gpu=True](
                _canonical[n, d](points)
            ).to_host()
            var highs = _smax[axis=0, gpu=True](
                _canonical[n, d](points)
            ).to_host()
            comptime for k in range(dims):
                var lo = Float64(lows[k])
                var hi = Float64(highs[k])
                if lo == hi:
                    lo -= 0.5
                    hi += 0.5
                edges.append(_uniform_edges(lo, hi, bins[k]))
            var base = points.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
            counts = _cells_device[dtype, dims](base, base, n, d, edges, ctx)
        else:
            _notice[gpu]("histogramdd")
    if on_device:
        return _histogramdd_finish[dtype, *bins](
            ctx, counts^, edges^, extents, total_cells, density
        )
    var host = points.to_host()
    comptime for k in range(dims):
        var column = List[Float64](capacity=n)
        for i in range(n):
            column.append(Float64(host[i * d + k]))
        var span = _data_range(column)
        edges.append(_uniform_edges(span[0], span[1], bins[k]))
    counts = List[Float64](length=total_cells, fill=0.0)
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
    return _histogramdd_finish[dtype, *bins](
        ctx, counts^, edges^, extents, total_cells, density
    )


def _histogramdd_finish[
    dtype: DType, *bins: Int
](
    ctx: DeviceContext,
    var counts: List[Float64],
    var edges: List[List[Float64]],
    extents: List[Int],
    total_cells: Int,
    density: Bool,
) raises -> HistogramDD[dtype, *bins]:
    """`histogramdd`'s density normalization and assembly, shared by the
    host and device tallies."""
    comptime dims = len(bins)
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
        Static[dtype, *bins](count_values^, ctx), edges^
    )


def _int_range[T: TensorLike](xs: T) raises -> Tuple[Int, Int]:
    """The smallest and largest value of a non-empty integral GPU-context
    tensor, by two device reductions."""
    var ctx = xs.context()
    var lo = Static[T.dtype, 1](ctx)
    var hi = Static[T.dtype, 1](ctx)

    @always_inline
    def identity[
        w: Int
    ](tile: SIMD[T.dtype, w], idx: RowCoord[1]) {} -> SIMD[T.dtype, w]:
        return tile

    reduce_all[monoid="min", gpu=True](
        _flat_unchecked(xs), lo.tile(), identity, xs.size(), Optional(ctx)
    )
    reduce_all[monoid="max", gpu=True](
        _flat_unchecked(xs), hi.tile(), identity, xs.size(), Optional(ctx)
    )
    return (Int(lo.to_host()[0]), Int(hi.to_host()[0]))


def _bincount_device[
    T: TensorLike, wdtype: DType, weighted: Bool, W: TensorLike
](xs: T, weights: W, minlength: Int) raises -> Dynamic[wdtype, 1]:
    """`bincount` on the device: the range by two reductions, then one
    launch adding each sample's `1`, or weight, into its bin atomically."""
    _require_contiguous(xs)
    var ctx = xs.context()
    var n = xs.size()
    var largest = -1
    if n > 0:
        var span = _int_range(xs)
        if span[0] < 0:
            raise Error("bincount: values must be non-negative")
        largest = span[1]
    var length = max(largest + 1, minlength)
    var totals = Dynamic[wdtype, 1](row_major(_dyn_shape[1](length)), ctx)
    if n == 0:
        return totals^
    var xp = xs.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var wp = (
        weights.tile()
        .ptr.unsafe_bitcast[Scalar[wdtype]]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    var tp = totals.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

    @always_inline
    def tally[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var xp, var wp, var tp}:
        var i = coord_to_index_list(coord)[0]
        var add = Scalar[wdtype](1)
        comptime if weighted:
            add = wp[unsafe_offset=i]
        _ = Atomic[Scalar[wdtype]].fetch_add(tp + Int(xp[unsafe_offset=i]), add)

    elementwise[simd_width=1, target="gpu"](tally, Coord(n), ctx)
    ctx.synchronize()
    return totals^


def bincount[
    T: TensorLike, gpu: Bool = False
](xs: T, minlength: Int = 0) raises -> Dynamic[DType.int64, 1] where (
    is_row_major[T] and T.dtype.is_integral()
):
    """How often each non-negative integer occurs in `xs`: `out[v]` is the
    count of `v`, and the result is `max(xs) + 1` long or `minlength`,
    whichever is greater. `numpy.bincount(x, minlength)`. A negative value
    raises, as NumPy's does. At `gpu=True` the counts are `int32` atomics
    on the device, widened to `int64` there.

    Parameters:
        T: The row-major tensor type of `xs`, with an integral dtype.
        gpu: Count with atomics on the tensor's device; a residency mismatch
            falls back to the host with a notice.

    Args:
        xs: The non-negative integers to count, walked flat.
        minlength: The smallest length the result may have.

    Returns:
        A rank-1 `int64` tensor of length `max(max(xs) + 1, minlength)`
        holding the count of each value.

    Raises:
        If any element of `xs` is negative, or if a residency mismatch occurs
        under the `"raise"` fallback policy.
    """
    if _check_device[T, gpu](xs):
        comptime if gpu:
            return astype[DType.int64, gpu=True](
                _bincount_device[T, DType.int32, False](xs, xs, minlength)
            )
    else:
        _notice[gpu]("bincount")
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
    T: TensorLike, wdtype: DType, gpu: Bool = False
](
    xs: T,
    weights: Tensor[wdtype, T.LayoutType],
    minlength: Int = 0,
) raises -> Dynamic[wdtype, 1] where (
    is_row_major[T] and T.dtype.is_integral() and wdtype.is_floating_point()
):
    """`bincount` summing each value's weight instead of counting it.
    `numpy.bincount(x, weights=w, minlength)`.

    Parameters:
        T: The row-major tensor type of `xs`, with an integral dtype.
        wdtype: The floating-point dtype of `weights` and of the result.
        gpu: Sum with atomics on the tensor's device; a residency mismatch
            falls back to the host with a notice.

    Args:
        xs: The non-negative integers to bin, walked flat.
        weights: One weight per element of `xs`, at `xs`'s layout.
        minlength: The smallest length the result may have.

    Returns:
        A rank-1 `wdtype` tensor of length `max(max(xs) + 1, minlength)`
        holding the summed weight of each value.

    Raises:
        If any element of `xs` is negative, if `weights` is not contiguous on
        the device path, or if a residency mismatch occurs under the `"raise"`
        fallback policy.
    """
    if _check_device[T, gpu](xs):
        comptime if gpu:
            _require_contiguous(weights)
            return _bincount_device[T, wdtype, True](xs, weights, minlength)
    else:
        _notice[gpu]("bincount")
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


def _digitize_device[
    T: TensorLike
](xs: T, edges: List[Float64], right: Bool, increasing: Bool) raises -> Dynamic[
    DType.int64, 1
]:
    """`digitize` on the device: the (ascending) edges uploaded once, then
    one binary search per element, exactly the host's side rule."""
    _require_contiguous(xs)
    var ctx = xs.context()
    var m = len(edges)
    var values = List[Scalar[T.dtype]](capacity=m)
    for e in edges:
        values.append(Scalar[T.dtype](e))
    var edge_d = asarray(values^, ctx)
    var n = xs.size()
    var result = Dynamic[DType.int64, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](n))
    )
    if n == 0:
        return result^
    var xp = xs.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var ep = edge_d.tile().ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin]()
    var rp = result.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var take_right = not right

    @always_inline
    def search[
        width: Int, alignment: Int = 1
    ](coord: Coord) {
        var xp, var ep, var rp, var m, var take_right, var increasing
    }:
        var i = coord_to_index_list(coord)[0]
        var v = xp[unsafe_offset=i]
        var lo = 0
        var hi = m
        while lo < hi:
            var mid = (lo + hi) // 2
            var e = ep[unsafe_offset=mid]
            var goes_left = e > v if take_right else e >= v
            if goes_left:
                hi = mid
            else:
                lo = mid + 1
        rp[unsafe_offset=i] = Int64(lo if increasing else m - lo)

    elementwise[simd_width=1, target="gpu"](search, Coord(n), ctx)
    ctx.synchronize()
    return result^


def digitize[
    T: TensorLike, m: Int, gpu: Bool = False
](xs: T, bins: Static[T.dtype, m], right: Bool = False) raises -> Dynamic[
    DType.int64, 1
] where (T.dtype.is_floating_point() and m > 0):
    """The index of the bin each element of `xs` falls in, for monotonic
    `bins`: `bins[i-1] <= x < bins[i]` by default, `bins[i-1] < x <=
    bins[i]` with `right`, `0` below the first edge and `m` past the last.
    `numpy.digitize(x, bins, right)`.

    `searchsorted` with the side flipped, which is exactly NumPy's
    definition, and decreasing `bins` handled as NumPy handles them, by
    searching the reversed edges and counting from the far end. Host-side,
    one bisection per element.

    Parameters:
        T: The tensor type of `xs`, with a floating-point dtype.
        m: The number of bin edges.
        gpu: Search on the tensor's device; a residency mismatch falls back to
            the host with a notice.

    Args:
        xs: The values to place, walked flat.
        bins: The `m` monotonic (ascending or descending) bin edges.
        right: Close each bin on the right, `bins[i-1] < x <= bins[i]`,
            instead of on the left.

    Returns:
        A rank-1 `int64` tensor of `xs.size()` bin indices in `[0, m]`.

    Raises:
        If a residency mismatch occurs under the `"raise"` fallback policy.
    """
    var edges = _as_float64(bins.to_host())
    var increasing = m == 1 or edges[m - 1] >= edges[0]
    if not increasing:
        var reversed = List[Float64](capacity=m)
        for i in range(m):
            reversed.append(edges[m - 1 - i])
        edges = reversed^
    if _check_device[T, gpu](xs):
        comptime if gpu:
            return _digitize_device(xs, edges, right, increasing)
    else:
        _notice[gpu]("digitize")
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
