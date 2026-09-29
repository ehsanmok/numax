"""Spline interpolation of N-dimensional arrays over `Tensor`:
`map_coordinates`, `shift`, `zoom`, `spline_filter` and
`spline_filter1d`, SciPy's `scipy.ndimage`.

**Tier 2.** SciPy's C (`ni_interpolation.c`, `ni_splines.c`) transcribed:

- **Prefilter.** For `order > 1` the input is first converted to B-spline
  coefficients by `spline_filter`: per axis, per line, the causal and
  anticausal recursions for each of the order's poles, with SciPy's
  boundary initialization -- mirror for `"mirror"`, `"constant"` and
  `"wrap"`, reflect for `"reflect"` and `"nearest"`, periodic for
  `"grid-wrap"` -- and its gain. `"nearest"` and `"grid-constant"` pad
  twelve samples first, as SciPy's `_prepad_for_spline_filter` does. One
  lane per line, the line walked in place.
- **Evaluation.** Each output sample's input coordinate -- given, for
  `map_coordinates`, or `k zoom` / `k - shift` for the others -- is mapped
  into the input by SciPy's `map_coordinate` for the mode, and a point
  that falls outside under `"constant"` takes `cval`. Otherwise the
  `order + 1` neighbors along each axis (folded through the mode's
  spline boundary where they leave the input) are weighted by the
  B-spline of that order, SciPy's `get_spline_interpolation_weights`,
  and summed over the `(order + 1)^rank` footprint. One lane per output
  sample.

Orders 0 to 5, ranks 1 to 8. SciPy's interpolation modes differ from the
filters' in two places, and follow SciPy here: `"wrap"` has period `n -
1` (`"grid-wrap"` is the period-`n` one), and `"constant"` does not
interpolate past the edge (`"grid-constant"` does).

## The MAX gate

`nn.resize` (nearest, linear, bicubic) resamples 2-D NHWC images by a
scale factor with its own coordinate conventions and no spline
prefilter; it cannot express B-spline interpolation of order 2 to 5, the
modes above, or arbitrary coordinates. **Extend.**
"""

from std.math import floor, pow

from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise

from ..core.tensorlike import TensorLike
from ..core.tensor import Dynamic, _dyn_shape, _dyn_shape_from, _same_order

comptime _NEAREST = 0
comptime _WRAP = 1
comptime _REFLECT = 2
comptime _MIRROR = 3
comptime _CONSTANT = 4
comptime _GRID_WRAP = 5
comptime _GRID_CONSTANT = 6
comptime _PREPAD = 12


def _target[gpu: Bool]() -> StaticString:
    comptime if gpu:
        return "gpu"
    else:
        return "cpu"


def _mode_code(mode: StaticString) raises -> Int:
    """SciPy's `_extend_mode_to_code`."""
    if mode == "nearest":
        return _NEAREST
    if mode == "wrap":
        return _WRAP
    if mode == "reflect" or mode == "grid-mirror":
        return _REFLECT
    if mode == "mirror":
        return _MIRROR
    if mode == "constant":
        return _CONSTANT
    if mode == "grid-wrap":
        return _GRID_WRAP
    if mode == "grid-constant":
        return _GRID_CONSTANT
    raise Error(
        (
            "ndimage: mode must be 'nearest', 'wrap', 'reflect', 'grid-mirror',"
            " 'mirror', 'constant', 'grid-wrap' or 'grid-constant', got '"
        ),
        mode,
        "'",
    )


@always_inline
def _spline_mode(mode: Int) -> Int:
    """SciPy's `_get_spline_boundary_mode`: `"constant"` and `"wrap"` use
    mirror extension for the spline's neighbors."""
    if mode == _CONSTANT or mode == _WRAP:
        return _MIRROR
    return mode


@always_inline
def _map_coordinate[
    dtype: DType
](v: Scalar[dtype], n: Int, mode: Int) -> Scalar[
    dtype
] where dtype.is_floating_point():
    """SciPy's `map_coordinate`: the coordinate `v` on an axis of `n`
    samples, brought inside by the mode, or `-1` for a constant sample.
    At the tensor's `dtype`, so it compiles where there is no `double`."""
    comptime S = Scalar[dtype]
    var x = v
    var extent = S(n)
    if x < 0:
        if mode == _MIRROR:
            if n <= 1:
                x = 0
            else:
                var sz2 = 2 * n - 2
                x = S(sz2 * Int(-x / S(sz2))) + x
                x = x + S(sz2) if x <= S(1 - n) else -x
        elif mode == _REFLECT:
            if n <= 1:
                x = 0
            else:
                var sz2 = 2 * n
                if x < S(-sz2):
                    x = S(sz2 * Int(-x / S(sz2))) + x
                if x < -extent:
                    x = x + S(sz2)
                else:
                    x = (S(1e-15) if x > S(-1e-15) else -x) - 1
        elif mode == _WRAP:
            if n <= 1:
                x = 0
            else:
                var sz = n - 1
                x += S(sz * (Int(-x / S(sz)) + 1))
        elif mode == _GRID_WRAP:
            if n <= 1:
                x = 0
            else:
                x += S(n * (Int((-1 - x) / extent) + 1))
        elif mode == _NEAREST:
            x = 0
        elif mode == _CONSTANT:
            x = -1
    elif x > S(n - 1):
        if mode == _MIRROR:
            if n <= 1:
                x = 0
            else:
                var sz2 = 2 * n - 2
                x -= S(sz2 * Int(x / S(sz2)))
                if x >= extent:
                    x = S(sz2) - x
        elif mode == _REFLECT:
            if n <= 1:
                x = 0
            else:
                var sz2 = 2 * n
                x -= S(sz2 * Int(x / S(sz2)))
                if x >= extent:
                    x = S(sz2) - x - 1
        elif mode == _WRAP:
            if n <= 1:
                x = 0
            else:
                var sz = n - 1
                x -= S(sz * Int(x / S(sz)))
        elif mode == _GRID_WRAP:
            if n <= 1:
                x = 0
            else:
                x -= S(n * Int(x / extent))
        elif mode == _NEAREST:
            x = S(n - 1)
        elif mode == _CONSTANT:
            x = -1
    return x


@always_inline
def _weights[
    dtype: DType
](v: Scalar[dtype], order: Int) -> SIMD[
    dtype, 8
] where dtype.is_floating_point():
    """SciPy's `get_spline_interpolation_weights` for orders 1 to 5; the
    last weight makes the sum one."""
    comptime S = Scalar[dtype]
    var w = SIMD[dtype, 8](0)
    var x = v - floor(v if order % 2 == 1 else v + S(0.5))
    var y = x
    var z = S(1) - x
    if order == 1:
        w[0] = S(1) - x
    elif order == 2:
        w[1] = S(0.75) - x * x
        y = S(0.5) - x
        w[0] = S(0.5) * y * y
    elif order == 3:
        w[1] = (y * y * (y - S(2)) * S(3) + S(4)) / S(6)
        w[2] = (z * z * (z - S(2)) * S(3) + S(4)) / S(6)
        w[0] = z * z * z / S(6)
    elif order == 4:
        var t = x * x
        w[2] = t * (t * S(0.25) - S(0.625)) + S(115.0 / 192.0)
        y = S(1) + x
        w[1] = y * (y * (y * (S(5) - y) / S(6) - S(1.25)) + S(5.0 / 24.0)) + S(
            55.0 / 96.0
        )
        z = S(1) - x
        w[3] = z * (z * (z * (S(5) - z) / S(6) - S(1.25)) + S(5.0 / 24.0)) + S(
            55.0 / 96.0
        )
        y = S(0.5) - x
        t = y * y
        w[0] = t * t / S(24)
    elif order == 5:
        var t = y * y
        w[2] = t * (t * (S(0.25) - y / S(12)) - S(0.5)) + S(0.55)
        t = z * z
        w[3] = t * (t * (S(0.25) - z / S(12)) - S(0.5)) + S(0.55)
        y += S(1)
        w[1] = y * (
            y * (y * (y * (y / S(24) - S(0.375)) + S(1.25)) - S(1.75))
            + S(0.625)
        ) + S(0.425)
        z += S(1)
        w[4] = z * (
            z * (z * (z * (z / S(24) - S(0.375)) + S(1.25)) - S(1.75))
            + S(0.625)
        ) + S(0.425)
        y = S(1) - x
        t = y * y
        w[0] = y * t * t / S(120)
    var last = S(1)
    for i in range(order):
        last -= w[i]
    w[order] = last
    return w


def _poles(order: Int) -> SIMD[DType.float64, 2]:
    if order == 2:
        return SIMD[DType.float64, 2](
            -0.171572875253809902396622551580603843, 0
        )
    if order == 3:
        return SIMD[DType.float64, 2](
            -0.267949192431122706472553658494127633, 0
        )
    if order == 4:
        return SIMD[DType.float64, 2](
            -0.361341225900220177092212841325675255,
            -0.013725429297339121360331226939128204,
        )
    return SIMD[DType.float64, 2](
        -0.430575347099973791851434783493520110,
        -0.043096288203264653822712376822550182,
    )


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


def _count(dims: SIMD[DType.int64, 8], rank: Int) -> Int:
    var total = 1
    for axis in range(rank):
        total *= Int(dims[axis])
    return total


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


def _filter_axis[
    dtype: DType, rank: Int, gpu: Bool
](
    mut data: Dynamic[dtype, 1],
    dims: SIMD[DType.int64, 8],
    axis: Int,
    order: Int,
    mode: Int,
) raises where dtype.is_floating_point():
    """SciPy's `NI_SplineFilter1D` in place: one lane per line along
    `axis`, the gain, then for each pole the causal and anticausal
    recursions with the mode's initialization. At the tensor's `dtype`;
    the poles and the gain are computed on the host and cast in."""
    comptime S = Scalar[dtype]
    var ctx = data.context()
    var n = Int(dims[axis])
    var strides = _strides(dims, rank)
    var stride = Int(strides[axis])
    var lines = _count(dims, rank) // n
    var dp = data.tile()
    var host_poles = _poles(order)
    var npoles = 1 if order <= 3 else 2
    var host_gain = 1.0
    for p in range(npoles):
        var z = host_poles[p]
        host_gain *= (1.0 - z) * (1.0 - 1.0 / z)
    var poles = SIMD[dtype, 2](S(host_poles[0]), S(host_poles[1]))
    var gain = S(host_gain)
    # The mode's boundary initialization, per SciPy's `apply_filter`.
    var init = 0
    if mode == _GRID_WRAP:
        init = 1
    elif mode == _NEAREST or mode == _REFLECT:
        init = 2

    @always_inline
    def line[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var dp,
        var n,
        var stride,
        var strides,
        var dims,
        var poles,
        var npoles,
        var gain,
        var init,
    }:
        var l = coord_to_index_list(coord)[0]
        # The line's first element: `l` enumerates every other axis.
        var base = 0
        var rest = l
        comptime for a in range(rank - 1, -1, -1):
            if Int(strides[a]) != stride:
                var extent = Int(dims[a])
                base += (rest % extent) * Int(strides[a])
                rest //= extent
        if n == 1:
            return
        var c = dp.ptr
        for i in range(n):
            c[unsafe_offset=base + i * stride] = (
                c[unsafe_offset=base + i * stride] * gain
            )
        for p in range(npoles):
            var z = poles[p]
            var first = base
            var last_at = base + (n - 1) * stride
            if init == 0:
                var z_i = z
                var z_n_1 = pow(z, S(n - 1))
                var c0 = (
                    c[unsafe_offset=first] + z_n_1 * c[unsafe_offset=last_at]
                )
                for i in range(1, n - 1):
                    c0 += z_i * (
                        c[unsafe_offset=base + i * stride]
                        + z_n_1 * c[unsafe_offset=base + (n - 1 - i) * stride]
                    )
                    z_i *= z
                c[unsafe_offset=first] = c0 / (S(1) - z_n_1 * z_n_1)
            elif init == 1:
                var z_i = z
                var c0 = c[unsafe_offset=first]
                for i in range(1, n):
                    c0 += z_i * c[unsafe_offset=base + (n - i) * stride]
                    z_i *= z
                c[unsafe_offset=first] = c0 / (S(1) - z_i)
            else:
                var z_i = z
                var z_n = pow(z, S(n))
                var total = (
                    c[unsafe_offset=first] + z_n * c[unsafe_offset=last_at]
                )
                for i in range(1, n):
                    total += z_i * (
                        c[unsafe_offset=base + i * stride]
                        + z_n * c[unsafe_offset=base + (n - 1 - i) * stride]
                    )
                    z_i *= z
                c[unsafe_offset=first] = c[unsafe_offset=first] + total * z / (
                    S(1) - z_n * z_n
                )
            for i in range(1, n):
                c[unsafe_offset=base + i * stride] = (
                    c[unsafe_offset=base + i * stride]
                    + z * c[unsafe_offset=base + (i - 1) * stride]
                )
            if init == 0:
                c[unsafe_offset=last_at] = (
                    (
                        z * c[unsafe_offset=base + (n - 2) * stride]
                        + c[unsafe_offset=last_at]
                    )
                    * z
                    / (z * z - S(1))
                )
            elif init == 1:
                var z_i = z
                var tail = c[unsafe_offset=last_at]
                for i in range(n - 1):
                    tail += z_i * c[unsafe_offset=base + i * stride]
                    z_i *= z
                c[unsafe_offset=last_at] = tail * z / (z_i - S(1))
            else:
                c[unsafe_offset=last_at] = (
                    c[unsafe_offset=last_at] * z / (z - S(1))
                )
            var i = n - 2
            while i >= 0:
                c[unsafe_offset=base + i * stride] = z * (
                    c[unsafe_offset=base + (i + 1) * stride]
                    - c[unsafe_offset=base + i * stride]
                )
                i -= 1

    elementwise[simd_width=1, target=_target[gpu]()](line, Coord(lines), ctx)
    ctx.synchronize()


def _prepad[
    dtype: DType, rank: Int, gpu: Bool
](
    src: Dynamic[dtype, 1], dims: SIMD[DType.int64, 8], mode: Int, cval: Float64
) raises -> Dynamic[dtype, 1] where dtype.is_floating_point():
    """SciPy's `_prepad_for_spline_filter`: twelve samples on every side,
    edge values for `"nearest"`, `cval` for `"grid-constant"`."""
    var ctx = src.context()
    var padded_dims = dims
    for a in range(rank):
        padded_dims[a] = dims[a] + 2 * _PREPAD
    var count = _count(padded_dims, rank)
    var out = Dynamic[dtype, 1](row_major(_dyn_shape[1](count)), ctx)
    var strides = _strides(dims, rank)
    var pstrides = _strides(padded_dims, rank)
    var sp = src.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var op = out.tile()
    var fill = Scalar[dtype](cval)

    @always_inline
    def body[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var sp, var op, var dims, var strides, var pstrides, var mode, var fill
    }:
        var e = coord_to_index_list(coord)[0]
        var rest = e
        var offset = 0
        var outside = False
        comptime for a in range(rank):
            var i = rest // Int(pstrides[a]) - _PREPAD
            rest = rest % Int(pstrides[a])
            var n = Int(dims[a])
            if i < 0 or i >= n:
                outside = True
                i = 0 if i < 0 else n - 1
            offset += i * Int(strides[a])
        op.ptr[unsafe_offset=e] = fill if (
            outside and mode == _GRID_CONSTANT
        ) else sp[unsafe_offset=offset]

    elementwise[simd_width=1, target=_target[gpu]()](body, Coord(count), ctx)
    ctx.synchronize()
    return out^


def _coefficients[
    T: TensorLike, gpu: Bool
](
    input: T, order: Int, mode: Int, cval: Float64, prefilter: Bool
) raises -> Tuple[
    Dynamic[T.dtype, 1], SIMD[DType.int64, 8], Int
] where T.dtype.is_floating_point():
    """The array the evaluation reads, its extents and the padding it
    carries: the prefiltered, possibly prepadded input for `order > 1`, the
    input itself otherwise."""
    comptime rank = T.LayoutType.rank
    var dims = _dims(input)
    var flat = _same_order(input, row_major(_dyn_shape[1](input.size())))
    if not prefilter or order <= 1:
        return (flat^, dims, 0)
    var npad = 0
    if mode == _NEAREST or mode == _GRID_CONSTANT:
        flat = _prepad[T.dtype, rank, gpu](flat, dims, mode, cval)
        for a in range(rank):
            dims[a] = dims[a] + 2 * _PREPAD
        npad = _PREPAD
    for a in range(rank):
        if Int(dims[a]) > 1:
            _filter_axis[T.dtype, rank, gpu](flat, dims, a, order, mode)
    return (flat^, dims, npad)


def _evaluate[
    dtype: DType, rank: Int, gpu: Bool, affine: Bool
](
    coeffs: Dynamic[dtype, 1],
    dims: SIMD[DType.int64, 8],
    npad: Int,
    coordinates: Dynamic[dtype, 1],
    scale: SIMD[DType.float64, 8],
    offset: SIMD[DType.float64, 8],
    out_dims: SIMD[DType.int64, 8],
    count: Int,
    order: Int,
    mode: Int,
    cval: Float64,
) raises -> Dynamic[dtype, 1] where dtype.is_floating_point():
    """SciPy's geometric transform: output sample `k`'s input coordinate is
    `coordinates[a, k]`, or with `affine` `k_a scale_a + offset_a` from its
    multi-index; then per axis the mode's map, the filter start, the
    weights and the folded neighbors; the `(order + 1)^rank` sum. At the
    tensor's `dtype`."""
    comptime S = Scalar[dtype]
    var ctx = coeffs.context()
    var out = Dynamic[dtype, 1](row_major(_dyn_shape[1](count)), ctx)
    var strides = _strides(dims, rank)
    var ostrides = _strides(out_dims, rank)
    var cp = coeffs.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var xp = coordinates.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var op = out.tile()
    var spline = _spline_mode(mode)
    var footprint = 1
    for _ in range(rank):
        footprint *= order + 1
    var sc = SIMD[dtype, 8](0)
    var of = SIMD[dtype, 8](0)
    for a in range(8):
        sc[a] = S(scale[a])
        of[a] = S(offset[a])
    var fill = S(cval)
    var pad = S(npad)

    @always_inline
    def body[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var cp,
        var xp,
        var op,
        var dims,
        var strides,
        var ostrides,
        var sc,
        var of,
        var pad,
        var order,
        var mode,
        var spline,
        var fill,
        var count,
        var footprint,
    }:
        var k = coord_to_index_list(coord)[0]
        var starts = SIMD[DType.int64, 8](0)
        var constant = False
        var rest = k
        var weights = SIMD[dtype, 64](0)
        comptime for a in range(rank):
            var cc: S
            comptime if affine:
                var ka = rest // Int(ostrides[a])
                rest = rest % Int(ostrides[a])
                cc = S(ka) * sc[a] + of[a]
            else:
                cc = xp[unsafe_offset=a * count + k]
            cc += pad
            var n = Int(dims[a])
            if mode != _GRID_CONSTANT and mode != _NEAREST:
                cc = _map_coordinate[dtype](cc, n, mode)
            if cc > S(-1) or mode == _GRID_CONSTANT or mode == _NEAREST:
                var start: Int
                if order % 2 == 1:
                    start = Int(floor(cc)) - order // 2
                else:
                    start = Int(floor(cc + S(0.5))) - order // 2
                starts[a] = Int64(start)
                var wv = SIMD[dtype, 8](0)
                if order == 0:
                    wv[0] = S(1)
                else:
                    wv = _weights[dtype](cc, order)
                comptime for j in range(8):
                    weights[a * 8 + j] = wv[j]
            else:
                constant = True
        var result = fill
        if not constant:
            result = S(0)
            for f in range(footprint):
                var fr = f
                var idx = 0
                var weight = S(1)
                var is_cval = False
                comptime for a in range(rank - 1, -1, -1):
                    var j = fr % (order + 1)
                    fr //= order + 1
                    var n = Int(dims[a])
                    var i = Int(starts[a]) + j
                    if i < 0 or i >= n:
                        if mode == _GRID_CONSTANT:
                            is_cval = True
                        else:
                            i = Int(_map_coordinate[dtype](S(i), n, spline))
                    idx += i * Int(strides[a])
                    weight *= weights[a * 8 + j]
                var sample = fill if is_cval else cp[unsafe_offset=idx]
                result += weight * sample
        op.ptr[unsafe_offset=k] = result

    elementwise[simd_width=1, target=_target[gpu]()](body, Coord(count), ctx)
    ctx.synchronize()
    return out^


def spline_filter1d[
    T: TensorLike, gpu: Bool = False
](
    input: T, order: Int = 3, axis: Int = -1, mode: StaticString = "mirror"
) raises -> Dynamic[T.dtype, T.LayoutType.rank] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """The B-spline coefficients along one axis: SciPy's
    `scipy.ndimage.spline_filter1d(input, order, axis, mode=mode)`; orders
    0 and 1 return the input.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        gpu: Whether to filter on the input's device.

    Args:
        input: The array.
        order: The spline order, 0 to 5.
        axis: The axis; negative counts from the end.
        mode: The boundary mode the coefficients assume.

    Returns:
        The coefficients, `input`'s shape.

    Raises:
        On an order outside `[0, 5]`, an unknown mode or axis, or a device
        failure.
    """
    comptime rank = T.LayoutType.rank
    if order < 0 or order > 5:
        raise Error("spline_filter1d: order must be in [0, 5], got ", order)
    var a = axis + rank if axis < 0 else axis
    if a < 0 or a >= rank:
        raise Error("spline_filter1d: axis ", axis, " for rank ", rank)
    var code = _mode_code(mode)
    var dims = _dims(input)
    var flat = _same_order(input, row_major(_dyn_shape[1](input.size())))
    if order > 1 and Int(dims[a]) > 1:
        _filter_axis[T.dtype, rank, gpu](flat, dims, a, order, code)
    return _shaped[T.dtype, rank](flat^, dims)


def spline_filter[
    T: TensorLike, gpu: Bool = False
](input: T, order: Int = 3, mode: StaticString = "mirror") raises -> Dynamic[
    T.dtype, T.LayoutType.rank
] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """The multidimensional B-spline coefficients, `spline_filter1d` along
    each axis. `scipy.ndimage.spline_filter(input, order, mode=mode)`.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        gpu: Whether to filter on the input's device.

    Args:
        input: The array.
        order: The spline order, 0 to 5.
        mode: The boundary mode the coefficients assume.

    Returns:
        The coefficients, `input`'s shape.

    Raises:
        As `spline_filter1d`.
    """
    comptime rank = T.LayoutType.rank
    if order < 0 or order > 5:
        raise Error("spline_filter: order must be in [0, 5], got ", order)
    var code = _mode_code(mode)
    var dims = _dims(input)
    var flat = _same_order(input, row_major(_dyn_shape[1](input.size())))
    if order > 1:
        for a in range(rank):
            if Int(dims[a]) > 1:
                _filter_axis[T.dtype, rank, gpu](flat, dims, a, order, code)
    return _shaped[T.dtype, rank](flat^, dims)


def map_coordinates[
    T: TensorLike, C: TensorLike, gpu: Bool = False
](
    input: T,
    coordinates: C,
    order: Int = 3,
    mode: StaticString = "constant",
    cval: Float64 = 0.0,
    prefilter: Bool = True,
) raises -> Dynamic[T.dtype, 1] where (
    T.dtype.is_floating_point()
    and C.dtype == T.dtype
    and C.LayoutType.rank == 2
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """The input interpolated at the given coordinates.
    `scipy.ndimage.map_coordinates(input, coordinates, order=order,
    mode=mode, cval=cval, prefilter=prefilter)`.

    Parameters:
        T: The tensor type of `input`, rank `d` from 1 to 8.
        C: The tensor type of `coordinates`, `d x m`.
        gpu: Whether to filter and evaluate on the input's device.

    Args:
        input: The array to sample.
        coordinates: One column per sample point, row `a` its coordinate
            along axis `a` in index units -- SciPy's `(ndim, m)` array.
        order: The spline order, 0 to 5; SciPy's default `3`.
        mode: The boundary mode, per this module's docstring; SciPy's
            default `"constant"`.
        cval: The value outside the input under the constant modes.
        prefilter: Whether to convert to spline coefficients first;
            `False` means `input` already holds them.

    Returns:
        The `m` interpolated values.

    Raises:
        On an order outside `[0, 5]`, an unknown mode, coordinates whose
        row count is not the input's rank, or a device failure.
    """
    comptime rank = T.LayoutType.rank
    if order < 0 or order > 5:
        raise Error("map_coordinates: order must be in [0, 5], got ", order)
    if coordinates.dim_at(0) != rank:
        raise Error(
            "map_coordinates: coordinates has ",
            coordinates.dim_at(0),
            " rows for a rank-",
            rank,
            " input",
        )
    var code = _mode_code(mode)
    var prepared = _coefficients[T, gpu](input, order, code, cval, prefilter)
    var m = coordinates.dim_at(1)
    var coords = rebind_var[Dynamic[T.dtype, 1]](
        _same_order(coordinates, row_major(_dyn_shape[1](coordinates.size())))
    )
    return _evaluate[T.dtype, rank, gpu, False](
        prepared[0],
        prepared[1],
        prepared[2],
        coords,
        SIMD[DType.float64, 8](0),
        SIMD[DType.float64, 8](0),
        SIMD[DType.int64, 8](1),
        m,
        order,
        code,
        cval,
    )


def shift[
    T: TensorLike, gpu: Bool = False
](
    input: T,
    shift: Float64,
    order: Int = 3,
    mode: StaticString = "constant",
    cval: Float64 = 0.0,
    prefilter: Bool = True,
) raises -> Dynamic[T.dtype, T.LayoutType.rank] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """The input shifted by `shift` samples on every axis, spline
    interpolated: `out[k] = input(k - shift)`.
    `scipy.ndimage.shift(input, shift, order=order, mode=mode, cval=cval,
    prefilter=prefilter)`.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        gpu: Whether to filter and evaluate on the input's device.

    Args:
        input: The array.
        shift: The shift, the same on every axis, in samples.
        order: The spline order, 0 to 5.
        mode: The boundary mode, per this module's docstring.
        cval: The value outside the input under the constant modes.
        prefilter: Whether to convert to spline coefficients first.

    Returns:
        The shifted array, `input`'s shape.

    Raises:
        On an order outside `[0, 5]`, an unknown mode, or a device failure.
    """
    comptime rank = T.LayoutType.rank
    if order < 0 or order > 5:
        raise Error("shift: order must be in [0, 5], got ", order)
    var code = _mode_code(mode)
    var dims = _dims(input)
    var prepared = _coefficients[T, gpu](input, order, code, cval, prefilter)
    var offset = SIMD[DType.float64, 8](0)
    var scale = SIMD[DType.float64, 8](1)
    for a in range(rank):
        offset[a] = -shift
    var unused = Dynamic[T.dtype, 1](
        row_major(_dyn_shape[1](1)), input.context()
    )
    var flat = _evaluate[T.dtype, rank, gpu, True](
        prepared[0],
        prepared[1],
        prepared[2],
        unused,
        scale,
        offset,
        dims,
        _count(dims, rank),
        order,
        code,
        cval,
    )
    return _shaped[T.dtype, rank](flat^, dims)


def zoom[
    T: TensorLike, gpu: Bool = False
](
    input: T,
    zoom: Float64,
    order: Int = 3,
    mode: StaticString = "constant",
    cval: Float64 = 0.0,
    prefilter: Bool = True,
    grid_mode: Bool = False,
) raises -> Dynamic[T.dtype, T.LayoutType.rank] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """The input resampled by `zoom` on every axis, spline interpolated.
    `scipy.ndimage.zoom(input, zoom, order=order, mode=mode, cval=cval,
    prefilter=prefilter, grid_mode=grid_mode)`.

    The output extent is `round(n zoom)` per axis. Without `grid_mode` the
    first and last samples of input and output coincide, output `k` sitting
    at `k (n - 1) / (n_out - 1)`; with it the pixel edges do, SciPy's
    `(k + 0.5) n / n_out - 0.5`.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        gpu: Whether to filter and evaluate on the input's device.

    Args:
        input: The array.
        zoom: The resampling factor, the same on every axis.
        order: The spline order, 0 to 5.
        mode: The boundary mode, per this module's docstring.
        cval: The value outside the input under the constant modes.
        prefilter: Whether to convert to spline coefficients first.
        grid_mode: Align pixel edges rather than pixel centers.

    Returns:
        The resampled array.

    Raises:
        On a non-positive zoom, an order outside `[0, 5]`, an unknown mode,
        or a device failure.
    """
    comptime rank = T.LayoutType.rank
    if not (zoom > 0.0):
        raise Error("zoom: the factor must be positive")
    if order < 0 or order > 5:
        raise Error("zoom: order must be in [0, 5], got ", order)
    var code = _mode_code(mode)
    var dims = _dims(input)
    var out_dims = SIMD[DType.int64, 8](1)
    var scale = SIMD[DType.float64, 8](1)
    var offset = SIMD[DType.float64, 8](0)
    for a in range(rank):
        var n = Int(dims[a])
        var o = Int(Float64(n) * zoom + 0.5)
        # Python's `round`, which the `+ 0.5` matches except at exact
        # halves, where it rounds to even.
        var exact = Float64(n) * zoom
        if exact - floor(exact) == 0.5 and Int(floor(exact)) % 2 == 0:
            o = Int(floor(exact))
        out_dims[a] = Int64(max(o, 1))
        var num = Float64(n) if grid_mode else Float64(n - 1)
        var den = Float64(Int(out_dims[a])) if grid_mode else Float64(
            Int(out_dims[a]) - 1
        )
        var factor = num / den if den != 0 else 1.0
        if grid_mode:
            scale[a] = factor
            offset[a] = 0.5 * factor - 0.5
        else:
            scale[a] = factor
    var prepared = _coefficients[T, gpu](input, order, code, cval, prefilter)
    var unused = Dynamic[T.dtype, 1](
        row_major(_dyn_shape[1](1)), input.context()
    )
    var flat = _evaluate[T.dtype, rank, gpu, True](
        prepared[0],
        prepared[1],
        prepared[2],
        unused,
        scale,
        offset,
        out_dims,
        _count(out_dims, rank),
        order,
        code,
        cval,
    )
    return _shaped[T.dtype, rank](flat^, out_dims)
