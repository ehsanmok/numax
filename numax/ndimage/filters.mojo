"""N-dimensional filters over `Tensor`: `correlate`, `convolve`, their 1-D
forms, `uniform_filter`, `gaussian_filter`, `median_filter`,
`minimum_filter` and `maximum_filter`, SciPy's `scipy.ndimage`.

**Tier 2.** Every filter is one launch per pass, one lane per output
element, on the input's device at `gpu=True`, in any rank from 1 to 8.
Outside the input the samples come from SciPy's five boundary modes,
folded into the index rather than padded in memory:

| mode | extension of `a b c d` |
|---|---|
| `"reflect"` (the default) | `d c b a | a b c d | d c b a` |
| `"mirror"` | `d c b | a b c d | c b a` |
| `"nearest"` | `a a a | a b c d | d d d` |
| `"wrap"` | `a b c d | a b c d | a b c d` |
| `"constant"` | `k k k | a b c d | k k k`, `k = cval` |

A weight array of extents `w` is centered at `w // 2 + origin` along each
axis, SciPy's convention; `convolve` flips the weights and negates the
origin, less one on an even extent, as SciPy's does. `uniform_filter` and
`gaussian_filter` are separable -- a `correlate1d` pass per axis, the
Gaussian's weights built as SciPy's `_gaussian_kernel1d` builds them, to
derivative order 3. The rank filters gather each output's box footprint
into the lane (at most 1024 samples) and select from it; the median is
SciPy's `rank = size // 2`, the upper middle of an even footprint.

## The MAX gate

MAX's image operators -- `nn.conv`, `nn.pool` (`max_pool`/`avg_pool`)
and `nn.resize` -- are two-dimensional NHWC kernels with zero padding.
ndimage's filters are `N`-dimensional with the five boundary modes above
and an origin shift. Pre-padding would have to bridge the gap, and
`nn.pad` covers constant, reflect and repeat on the host but only
constant on the device, and its `reflect` is ndimage's `mirror`: nothing
there gives the half-sample `reflect` ndimage defaults to, or `wrap`.
**Extend**, with the image operators recorded in `docs/parity.md` as
what MAX has instead.
"""

from std.collections import Array
from std.math import exp as _exp

from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise

from ..core.tensorlike import TensorLike
from ..core.tensor import (
    Dynamic,
    _dyn_shape_from,
    _dyn_shape,
    _same_order,
    asarray,
)

comptime _REFLECT = 0
comptime _MIRROR = 1
comptime _NEAREST = 2
comptime _WRAP = 3
comptime _CONSTANT = 4
comptime _MAX_FOOTPRINT = 1024


def _target[gpu: Bool]() -> StaticString:
    comptime if gpu:
        return "gpu"
    else:
        return "cpu"


def _mode_code(mode: StaticString) raises -> Int:
    if mode == "reflect" or mode == "grid-mirror":
        return _REFLECT
    if mode == "mirror":
        return _MIRROR
    if mode == "nearest":
        return _NEAREST
    if mode == "wrap" or mode == "grid-wrap":
        return _WRAP
    if mode == "constant" or mode == "grid-constant":
        return _CONSTANT
    raise Error(
        (
            "ndimage: mode must be 'reflect', 'mirror', 'nearest', 'wrap' or"
            " 'constant', got '"
        ),
        mode,
        "'",
    )


@always_inline
def _fold(i: Int, n: Int, code: Int) -> Int:
    """Index `i` of an axis of `n` samples under boundary mode `code`, or
    `-1` for a `"constant"` sample outside."""
    if i >= 0 and i < n:
        return i
    if code == _CONSTANT:
        return -1
    if code == _NEAREST:
        return 0 if i < 0 else n - 1
    if code == _WRAP:
        var r = i % n
        return r + n if r < 0 else r
    if code == _MIRROR:
        if n == 1:
            return 0
        var period = 2 * n - 2
        var r = i % period
        if r < 0:
            r += period
        return period - r if r >= n else r
    var period = 2 * n
    var r = i % period
    if r < 0:
        r += period
    return period - 1 - r if r >= n else r


def _dims[T: TensorLike](a: T) -> SIMD[DType.int64, 8]:
    var out = SIMD[DType.int64, 8](1)
    comptime for axis in range(T.LayoutType.rank):
        out[axis] = Int64(a.dim_at(axis))
    return out


def _strides(dims: SIMD[DType.int64, 8], rank: Int) -> SIMD[DType.int64, 8]:
    var out = SIMD[DType.int64, 8](1)
    for axis in range(rank - 2, -1, -1):
        out[axis] = out[axis + 1] * dims[axis + 1]
    return out


def _flat[T: TensorLike](a: T) raises -> Dynamic[T.dtype, 1]:
    return _same_order(a, row_major(_dyn_shape[1](a.size())))


def _shaped[
    dtype: DType, rank: Int
](var flat: Dynamic[dtype, 1], dims: SIMD[DType.int64, 8]) raises -> Dynamic[
    dtype, rank
]:
    var extents = List[Int](capacity=rank)
    for axis in range(rank):
        extents.append(Int(dims[axis]))
    return Dynamic[dtype, rank](
        flat._buffer,
        row_major(_dyn_shape_from[rank](extents)),
        flat.host_addressable,
    )


def _correlate_flat[
    dtype: DType, rank: Int, gpu: Bool
](
    src: Dynamic[dtype, 1],
    dims: SIMD[DType.int64, 8],
    weights: Dynamic[dtype, 1],
    wdims: SIMD[DType.int64, 8],
    centers: SIMD[DType.int64, 8],
    code: Int,
    cval: Scalar[dtype],
) raises -> Dynamic[dtype, 1] where dtype.is_floating_point():
    """The correlation kernel every linear filter here is: output `i` is
    `sum_j w[j] in[i + j - center]`, each index folded by the mode."""
    var ctx = src.context()
    var count = src.size()
    var wcount = weights.size()
    var out = Dynamic[dtype, 1](row_major(_dyn_shape[1](count)), ctx)
    var strides = _strides(dims, rank)
    var wstrides = _strides(wdims, rank)
    var sp = src.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var wp = weights.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var op = out.tile()

    @always_inline
    def body[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var sp,
        var wp,
        var op,
        var dims,
        var strides,
        var wdims,
        var wstrides,
        var centers,
        var code,
        var cval,
        var wcount,
    }:
        var e = coord_to_index_list(coord)[0]
        var at = SIMD[DType.int64, 8](0)
        var rest = e
        comptime for axis in range(rank):
            at[axis] = Int64(rest // Int(strides[axis]))
            rest = rest % Int(strides[axis])
        var total = Scalar[dtype](0)
        for j in range(wcount):
            var wr = j
            var offset = 0
            var outside = False
            comptime for axis in range(rank):
                var wj = wr // Int(wstrides[axis])
                wr = wr % Int(wstrides[axis])
                var idx = _fold(
                    Int(at[axis]) + wj - Int(centers[axis]),
                    Int(dims[axis]),
                    code,
                )
                if idx < 0:
                    outside = True
                offset += idx * Int(strides[axis])
            var sample = cval if outside else sp[unsafe_offset=offset]
            total += wp[unsafe_offset=j] * sample
        op.ptr[unsafe_offset=e] = total

    elementwise[simd_width=1, target=_target[gpu]()](body, Coord(count), ctx)
    ctx.synchronize()
    return out^


def correlate[
    T: TensorLike, W: TensorLike, gpu: Bool = False
](
    input: T,
    weights: W,
    mode: StaticString = "reflect",
    cval: Float64 = 0.0,
    origin: Int = 0,
) raises -> Dynamic[T.dtype, T.LayoutType.rank] where (
    T.dtype.is_floating_point()
    and W.dtype == T.dtype
    and W.LayoutType.rank == T.LayoutType.rank
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """The multidimensional correlation of `input` with `weights`.
    `scipy.ndimage.correlate(input, weights, mode=mode, cval=cval,
    origin=origin)`.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        W: The tensor type of `weights`, the same rank and `dtype`.
        gpu: Whether to filter on the input's device.

    Args:
        input: The array to filter.
        weights: The correlation weights.
        mode: The boundary mode, per this module's docstring.
        cval: The value outside the input under `"constant"`.
        origin: The shift of the weights' center, the same on every axis.

    Returns:
        The filtered array, `input`'s shape.

    Raises:
        On an unknown mode, an origin past half a weight extent, or a
        device failure.
    """
    comptime rank = T.LayoutType.rank
    var code = _mode_code(mode)
    var dims = _dims(input)
    var wdims = _dims(weights)
    var centers = SIMD[DType.int64, 8](0)
    for axis in range(rank):
        var half = Int(wdims[axis]) // 2
        if origin < -half or origin > (Int(wdims[axis]) - 1) // 2:
            raise Error(
                "correlate: origin ", origin, " is out of range for the weights"
            )
        centers[axis] = Int64(half + origin)
    var src = _flat(input)
    var w = rebind_var[Dynamic[T.dtype, 1]](_flat(weights))
    var flat = _correlate_flat[T.dtype, rank, gpu](
        src, dims, w, wdims, centers, code, Scalar[T.dtype](cval)
    )
    return _shaped[T.dtype, rank](flat^, dims)


def _flip[
    dtype: DType, gpu: Bool
](w: Dynamic[dtype, 1]) raises -> Dynamic[
    dtype, 1
] where dtype.is_floating_point():
    """`w` reversed, which for a flat row-major array is the flip along
    every axis at once."""
    var ctx = w.context()
    var n = w.size()
    var out = Dynamic[dtype, 1](row_major(_dyn_shape[1](n)), ctx)
    var wp = w.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var op = out.tile()

    @always_inline
    def body[ww: Int, alignment: Int = 1](coord: Coord) {var wp, var op, var n}:
        var i = coord_to_index_list(coord)[0]
        op.ptr[unsafe_offset=i] = wp[unsafe_offset=n - 1 - i]

    elementwise[simd_width=1, target=_target[gpu]()](body, Coord(n), ctx)
    ctx.synchronize()
    return out^


def convolve[
    T: TensorLike, W: TensorLike, gpu: Bool = False
](
    input: T,
    weights: W,
    mode: StaticString = "reflect",
    cval: Float64 = 0.0,
    origin: Int = 0,
) raises -> Dynamic[T.dtype, T.LayoutType.rank] where (
    T.dtype.is_floating_point()
    and W.dtype == T.dtype
    and W.LayoutType.rank == T.LayoutType.rank
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """The multidimensional convolution of `input` with `weights`.
    `scipy.ndimage.convolve(input, weights, mode=mode, cval=cval,
    origin=origin)`: `correlate` with the weights flipped along every axis
    and the origin negated, less one on an even extent.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        W: The tensor type of `weights`, the same rank and `dtype`.
        gpu: Whether to filter on the input's device.

    Args:
        input: The array to filter.
        weights: The convolution kernel.
        mode: The boundary mode, per this module's docstring.
        cval: The value outside the input under `"constant"`.
        origin: The shift of the kernel's center, the same on every axis.

    Returns:
        The filtered array, `input`'s shape.

    Raises:
        On an unknown mode, an out-of-range origin, or a device failure.
    """
    comptime rank = T.LayoutType.rank
    var code = _mode_code(mode)
    var dims = _dims(input)
    var wdims = _dims(weights)
    var centers = SIMD[DType.int64, 8](0)
    for axis in range(rank):
        var extent = Int(wdims[axis])
        var o = -origin
        if extent % 2 == 0:
            o -= 1
        if o < -(extent // 2) or o > (extent - 1) // 2:
            raise Error(
                "convolve: origin ", origin, " is out of range for the weights"
            )
        centers[axis] = Int64(extent // 2 + o)
    var src = _flat(input)
    var w = _flip[T.dtype, gpu](rebind_var[Dynamic[T.dtype, 1]](_flat(weights)))
    var flat = _correlate_flat[T.dtype, rank, gpu](
        src, dims, w, wdims, centers, code, Scalar[T.dtype](cval)
    )
    return _shaped[T.dtype, rank](flat^, dims)


def _correlate_axis[
    dtype: DType, rank: Int, gpu: Bool
](
    src: Dynamic[dtype, 1],
    dims: SIMD[DType.int64, 8],
    weights: Dynamic[dtype, 1],
    axis: Int,
    origin: Int,
    code: Int,
    cval: Scalar[dtype],
) raises -> Dynamic[dtype, 1] where dtype.is_floating_point():
    """`correlate1d` on a flat array: the N-d kernel with a weight array of
    extent `len(weights)` along `axis` and one elsewhere."""
    var wdims = SIMD[DType.int64, 8](1)
    var length = weights.size()
    wdims[axis] = Int64(length)
    var centers = SIMD[DType.int64, 8](0)
    var half = length // 2
    if origin < -half or origin > (length - 1) // 2:
        raise Error(
            "correlate1d: origin ", origin, " is out of range for the weights"
        )
    centers[axis] = Int64(half + origin)
    return _correlate_flat[dtype, rank, gpu](
        src, dims, weights, wdims, centers, code, cval
    )


def correlate1d[
    T: TensorLike, W: TensorLike, gpu: Bool = False
](
    input: T,
    weights: W,
    axis: Int = -1,
    mode: StaticString = "reflect",
    cval: Float64 = 0.0,
    origin: Int = 0,
) raises -> Dynamic[T.dtype, T.LayoutType.rank] where (
    T.dtype.is_floating_point()
    and W.dtype == T.dtype
    and W.LayoutType.rank == 1
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """The one-dimensional correlation along `axis`.
    `scipy.ndimage.correlate1d(input, weights, axis, mode=mode, cval=cval,
    origin=origin)`.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        W: The tensor type of `weights`, rank 1, the same `dtype`.
        gpu: Whether to filter on the input's device.

    Args:
        input: The array to filter.
        weights: The 1-D weights.
        axis: The axis to filter along; negative counts from the end.
        mode: The boundary mode, per this module's docstring.
        cval: The value outside the input under `"constant"`.
        origin: The shift of the weights' center.

    Returns:
        The filtered array, `input`'s shape.

    Raises:
        On an unknown mode, an out-of-range axis or origin, or a device
        failure.
    """
    comptime rank = T.LayoutType.rank
    var a = axis + rank if axis < 0 else axis
    if a < 0 or a >= rank:
        raise Error("correlate1d: axis ", axis, " for rank ", rank)
    var dims = _dims(input)
    var flat = _correlate_axis[T.dtype, rank, gpu](
        _flat(input),
        dims,
        rebind_var[Dynamic[T.dtype, 1]](_flat(weights)),
        a,
        origin,
        _mode_code(mode),
        Scalar[T.dtype](cval),
    )
    return _shaped[T.dtype, rank](flat^, dims)


def convolve1d[
    T: TensorLike, W: TensorLike, gpu: Bool = False
](
    input: T,
    weights: W,
    axis: Int = -1,
    mode: StaticString = "reflect",
    cval: Float64 = 0.0,
    origin: Int = 0,
) raises -> Dynamic[T.dtype, T.LayoutType.rank] where (
    T.dtype.is_floating_point()
    and W.dtype == T.dtype
    and W.LayoutType.rank == 1
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """The one-dimensional convolution along `axis`.
    `scipy.ndimage.convolve1d`: `correlate1d` with the weights reversed and
    the origin negated, less one on an even length.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        W: The tensor type of `weights`, rank 1, the same `dtype`.
        gpu: Whether to filter on the input's device.

    Args:
        input: The array to filter.
        weights: The 1-D kernel.
        axis: The axis to filter along; negative counts from the end.
        mode: The boundary mode, per this module's docstring.
        cval: The value outside the input under `"constant"`.
        origin: The shift of the kernel's center.

    Returns:
        The filtered array, `input`'s shape.

    Raises:
        On an unknown mode, an out-of-range axis or origin, or a device
        failure.
    """
    comptime rank = T.LayoutType.rank
    var a = axis + rank if axis < 0 else axis
    if a < 0 or a >= rank:
        raise Error("convolve1d: axis ", axis, " for rank ", rank)
    var dims = _dims(input)
    var length = weights.size()
    var o = -origin
    if length % 2 == 0:
        o -= 1
    var flat = _correlate_axis[T.dtype, rank, gpu](
        _flat(input),
        dims,
        _flip[T.dtype, gpu](rebind_var[Dynamic[T.dtype, 1]](_flat(weights))),
        a,
        o,
        _mode_code(mode),
        Scalar[T.dtype](cval),
    )
    return _shaped[T.dtype, rank](flat^, dims)


def uniform_filter[
    T: TensorLike, gpu: Bool = False
](
    input: T,
    size: Int = 3,
    mode: StaticString = "reflect",
    cval: Float64 = 0.0,
    origin: Int = 0,
) raises -> Dynamic[T.dtype, T.LayoutType.rank] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """The mean over a `size`-wide box, separable: a `correlate1d` of
    `size` equal weights along each axis in turn.
    `scipy.ndimage.uniform_filter(input, size, mode=mode, cval=cval,
    origin=origin)`.

    SciPy's `uniform_filter1d` is a running sum, so the two agree to
    rounding rather than bit for bit.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        gpu: Whether to filter on the input's device.

    Args:
        input: The array to filter.
        size: The box width on every axis.
        mode: The boundary mode, per this module's docstring.
        cval: The value outside the input under `"constant"`.
        origin: The shift of the box's center.

    Returns:
        The filtered array, `input`'s shape.

    Raises:
        On a size below one, an unknown mode, or a device failure.
    """
    comptime dtype = T.dtype
    comptime rank = T.LayoutType.rank
    if size < 1:
        raise Error("uniform_filter: size must be at least 1")
    var code = _mode_code(mode)
    var dims = _dims(input)
    var ctx = input.context()
    var w = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        w.append(Scalar[dtype](1.0 / Float64(size)))
    var weights = asarray(w^, ctx)
    var flat = _flat(input)
    for axis in range(rank):
        flat = _correlate_axis[dtype, rank, gpu](
            flat, dims, weights, axis, origin, code, Scalar[dtype](cval)
        )
    return _shaped[dtype, rank](flat^, dims)


def _gaussian_weights(sigma: Float64, order: Int, radius: Int) -> List[Float64]:
    """SciPy's `_gaussian_kernel1d`: the normalized Gaussian on `[-radius,
    radius]`, times the polynomial that makes its `order`-th derivative."""
    var n = 2 * radius + 1
    var phi = List[Float64](capacity=n)
    var total = 0.0
    var s2 = sigma * sigma
    for i in range(n):
        var x = Float64(i - radius)
        var v = _exp(-0.5 / s2 * x * x)
        phi.append(v)
        total += v
    for i in range(n):
        phi[i] /= total
    if order == 0:
        return phi^
    # `q` holds the polynomial's coefficients; each step is `q' + q p'`
    # with `p' = -x / sigma^2`, SciPy's `Q_deriv` applied `order` times.
    var q = List[Float64](length=order + 1, fill=0.0)
    q[0] = 1.0
    for _ in range(order):
        var next = List[Float64](length=order + 1, fill=0.0)
        for e in range(1, order + 1):
            next[e - 1] += Float64(e) * q[e]
        for e in range(order):
            next[e + 1] += -q[e] / s2
        q = next^
    for i in range(n):
        var x = Float64(i - radius)
        var poly = 0.0
        var power = 1.0
        for e in range(order + 1):
            poly += q[e] * power
            power *= x
        phi[i] *= poly
    return phi^


def gaussian_filter1d[
    T: TensorLike, gpu: Bool = False
](
    input: T,
    sigma: Float64,
    axis: Int = -1,
    order: Int = 0,
    mode: StaticString = "reflect",
    cval: Float64 = 0.0,
    truncate: Float64 = 4.0,
) raises -> Dynamic[T.dtype, T.LayoutType.rank] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """The Gaussian filter, or its `order`-th derivative, along one axis.
    `scipy.ndimage.gaussian_filter1d(input, sigma, axis, order, mode=mode,
    cval=cval, truncate=truncate)`: SciPy's kernel, radius `int(truncate
    sigma + 0.5)`, correlated reversed.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        gpu: Whether to filter on the input's device.

    Args:
        input: The array to filter.
        sigma: The standard deviation, in samples.
        axis: The axis; negative counts from the end.
        order: The derivative order, 0 to 3.
        mode: The boundary mode, per this module's docstring.
        cval: The value outside the input under `"constant"`.
        truncate: The kernel's half-width in standard deviations.

    Returns:
        The filtered array, `input`'s shape.

    Raises:
        On a non-positive `sigma`, an order outside `[0, 3]`, an unknown
        mode or axis, or a device failure.
    """
    comptime dtype = T.dtype
    comptime rank = T.LayoutType.rank
    if not (sigma > 0.0):
        raise Error("gaussian_filter1d: sigma must be positive")
    if order < 0 or order > 3:
        raise Error("gaussian_filter1d: order must be in [0, 3], got ", order)
    var a = axis + rank if axis < 0 else axis
    if a < 0 or a >= rank:
        raise Error("gaussian_filter1d: axis ", axis, " for rank ", rank)
    var radius = Int(truncate * sigma + 0.5)
    var raw = _gaussian_weights(sigma, order, radius)
    var w = List[Scalar[dtype]](capacity=len(raw))
    for i in range(len(raw)):
        w.append(Scalar[dtype](raw[len(raw) - 1 - i]))
    var dims = _dims(input)
    var flat = _correlate_axis[dtype, rank, gpu](
        _flat(input),
        dims,
        asarray(w^, input.context()),
        a,
        0,
        _mode_code(mode),
        Scalar[dtype](cval),
    )
    return _shaped[dtype, rank](flat^, dims)


def gaussian_filter[
    T: TensorLike, gpu: Bool = False
](
    input: T,
    sigma: Float64,
    order: Int = 0,
    mode: StaticString = "reflect",
    cval: Float64 = 0.0,
    truncate: Float64 = 4.0,
) raises -> Dynamic[T.dtype, T.LayoutType.rank] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """The multidimensional Gaussian filter, separable: `gaussian_filter1d`
    along each axis with the same `sigma` and `order`.
    `scipy.ndimage.gaussian_filter(input, sigma, order, mode=mode,
    cval=cval, truncate=truncate)`.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        gpu: Whether to filter on the input's device.

    Args:
        input: The array to filter.
        sigma: The standard deviation on every axis, in samples.
        order: The derivative order on every axis, 0 to 3.
        mode: The boundary mode, per this module's docstring.
        cval: The value outside the input under `"constant"`.
        truncate: The kernel's half-width in standard deviations.

    Returns:
        The filtered array, `input`'s shape.

    Raises:
        As `gaussian_filter1d`.
    """
    comptime dtype = T.dtype
    comptime rank = T.LayoutType.rank
    if not (sigma > 0.0):
        raise Error("gaussian_filter: sigma must be positive")
    if order < 0 or order > 3:
        raise Error("gaussian_filter: order must be in [0, 3], got ", order)
    var code = _mode_code(mode)
    var radius = Int(truncate * sigma + 0.5)
    var raw = _gaussian_weights(sigma, order, radius)
    var w = List[Scalar[dtype]](capacity=len(raw))
    for i in range(len(raw)):
        w.append(Scalar[dtype](raw[len(raw) - 1 - i]))
    var weights = asarray(w^, input.context())
    var dims = _dims(input)
    var flat = _flat(input)
    for axis in range(rank):
        flat = _correlate_axis[dtype, rank, gpu](
            flat, dims, weights, axis, 0, code, Scalar[dtype](cval)
        )
    return _shaped[dtype, rank](flat^, dims)


comptime _MEDIAN = 0
comptime _MINIMUM = 1
comptime _MAXIMUM = 2


def _rank_filter[
    T: TensorLike, kind: Int, gpu: Bool
](input: T, size: Int, mode: StaticString, cval: Float64) raises -> Dynamic[
    T.dtype, T.LayoutType.rank
] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    comptime dtype = T.dtype
    comptime rank = T.LayoutType.rank
    if size < 1:
        raise Error("ndimage: size must be at least 1")
    var footprint = 1
    for _ in range(rank):
        footprint *= size
    if footprint > _MAX_FOOTPRINT:
        raise Error(
            "ndimage: a ",
            size,
            "-wide box in ",
            rank,
            " dimensions is ",
            footprint,
            " samples, past the ",
            _MAX_FOOTPRINT,
            " a lane gathers",
        )
    var code = _mode_code(mode)
    var dims = _dims(input)
    var strides = _strides(dims, rank)
    var src = _flat(input)
    var ctx = src.context()
    var count = src.size()
    var out = Dynamic[dtype, 1](row_major(_dyn_shape[1](count)), ctx)
    var sp = src.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var op = out.tile()
    var fill = Scalar[dtype](cval)
    var half = size // 2

    @always_inline
    def body[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var sp,
        var op,
        var dims,
        var strides,
        var code,
        var fill,
        var size,
        var half,
        var footprint,
    }:
        var e = coord_to_index_list(coord)[0]
        var at = SIMD[DType.int64, 8](0)
        var rest = e
        comptime for axis in range(rank):
            at[axis] = Int64(rest // Int(strides[axis]))
            rest = rest % Int(strides[axis])
        var window = Array[Scalar[dtype], _MAX_FOOTPRINT](fill=Scalar[dtype](0))
        for j in range(footprint):
            var wr = j
            var offset = 0
            var outside = False
            var wstride = footprint
            comptime for axis in range(rank):
                wstride //= size
                var wj = wr // wstride
                wr = wr % wstride
                var idx = _fold(
                    Int(at[axis]) + wj - half, Int(dims[axis]), code
                )
                if idx < 0:
                    outside = True
                offset += idx * Int(strides[axis])
            window[j] = fill if outside else sp[unsafe_offset=offset]
        var result = window[0]
        comptime if kind == _MINIMUM:
            for j in range(1, footprint):
                result = min(result, window[j])
        elif kind == _MAXIMUM:
            for j in range(1, footprint):
                result = max(result, window[j])
        else:
            # SciPy's `rank = size // 2`: selection by insertion sort.
            for a in range(1, footprint):
                var v = window[a]
                var b = a
                while b > 0 and window[b - 1] > v:
                    window[b] = window[b - 1]
                    b -= 1
                window[b] = v
            result = window[footprint // 2]
        op.ptr[unsafe_offset=e] = result

    elementwise[simd_width=1, target=_target[gpu]()](body, Coord(count), ctx)
    ctx.synchronize()
    _ = src^
    return _shaped[dtype, rank](out^, dims)


def median_filter[
    T: TensorLike, gpu: Bool = False
](
    input: T, size: Int = 3, mode: StaticString = "reflect", cval: Float64 = 0.0
) raises -> Dynamic[T.dtype, T.LayoutType.rank] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """The median over a `size`-wide box. `scipy.ndimage.median_filter(input,
    size, mode=mode, cval=cval)`; an even footprint takes the upper of the
    two middle values, SciPy's `rank = size // 2`.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        gpu: Whether to filter on the input's device.

    Args:
        input: The array to filter.
        size: The box width on every axis; `size^rank` at most 1024.
        mode: The boundary mode, per this module's docstring.
        cval: The value outside the input under `"constant"`.

    Returns:
        The filtered array, `input`'s shape.

    Raises:
        On a footprint past 1024 samples, an unknown mode, or a device
        failure.
    """
    return _rank_filter[T, _MEDIAN, gpu](input, size, mode, cval)


def minimum_filter[
    T: TensorLike, gpu: Bool = False
](
    input: T, size: Int = 3, mode: StaticString = "reflect", cval: Float64 = 0.0
) raises -> Dynamic[T.dtype, T.LayoutType.rank] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """The minimum over a `size`-wide box. `scipy.ndimage.minimum_filter`.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        gpu: Whether to filter on the input's device.

    Args:
        input: The array to filter.
        size: The box width on every axis; `size^rank` at most 1024.
        mode: The boundary mode, per this module's docstring.
        cval: The value outside the input under `"constant"`.

    Returns:
        The filtered array, `input`'s shape.

    Raises:
        As `median_filter`.
    """
    return _rank_filter[T, _MINIMUM, gpu](input, size, mode, cval)


def maximum_filter[
    T: TensorLike, gpu: Bool = False
](
    input: T, size: Int = 3, mode: StaticString = "reflect", cval: Float64 = 0.0
) raises -> Dynamic[T.dtype, T.LayoutType.rank] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """The maximum over a `size`-wide box. `scipy.ndimage.maximum_filter`.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        gpu: Whether to filter on the input's device.

    Args:
        input: The array to filter.
        size: The box width on every axis; `size^rank` at most 1024.
        mode: The boundary mode, per this module's docstring.
        cval: The value outside the input under `"constant"`.

    Returns:
        The filtered array, `input`'s shape.

    Raises:
        As `median_filter`.
    """
    return _rank_filter[T, _MAXIMUM, gpu](input, size, mode, cval)
