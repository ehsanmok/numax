"""Binary morphology and the Euclidean distance transform over `Tensor`:
`binary_erosion`, `binary_dilation` and `distance_transform_edt`,
SciPy's `scipy.ndimage`.

**Tier 2**, on the input's device at `gpu=True`.

- **Erosion and dilation** use SciPy's default structuring element,
  `generate_binary_structure(rank, connectivity)` -- the offsets in
  `{-1, 0, 1}^rank` with at most `connectivity` nonzero -- which is
  symmetric, so the reflection SciPy applies for dilation changes
  nothing. Outside the array the input reads as `border_value`. One
  launch per iteration, one lane per element. The result is the input's
  `dtype` holding `0` and `1`, where SciPy's is `bool`.
- **`distance_transform_edt`** is Felzenszwalb and Huttenlocher's exact
  squared distance transform, separable: starting from `0` at the
  background and infinity elsewhere, each axis's pass replaces every line
  by the lower envelope of the parabolas rooted at its samples, `O(n)`
  per line; one lane per line, the envelope's vertex and boundary arrays
  in device scratch rows. The square root of the last pass is the
  distance from each element to the nearest zero, SciPy's result at unit
  sampling.

## The MAX gate

Nothing: MAX has no morphology and no distance transform; its pooling
(`nn.max_pool`) is a 2-D NHWC window maximum with zero padding, not a
structuring element over an `N`-d binary array. **Extend.**
"""

from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise
from std.math import sqrt as _sqrt

from ..core.tensorlike import TensorLike
from ..core.tensor import (
    Dynamic,
    _dyn_shape,
    _dyn_shape_from,
    _same_order,
    copy,
)


def _target[gpu: Bool]() -> StaticString:
    comptime if gpu:
        return "gpu"
    else:
        return "cpu"


def _shape_of[T: TensorLike](a: T) -> SIMD[DType.int64, 8]:
    var out = SIMD[DType.int64, 8](1)
    comptime for axis in range(T.LayoutType.rank):
        out[axis] = Int64(a.dim_at(axis))
    return out


def _strides(dims: SIMD[DType.int64, 8], rank: Int) -> SIMD[DType.int64, 8]:
    var out = SIMD[DType.int64, 8](1)
    for axis in range(rank - 2, -1, -1):
        out[axis] = out[axis + 1] * dims[axis + 1]
    return out


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


def _morph[
    dtype: DType, rank: Int, gpu: Bool, dilate: Bool
](
    src: Dynamic[dtype, 1],
    dims: SIMD[DType.int64, 8],
    connectivity: Int,
    border: Bool,
) raises -> Dynamic[dtype, 1] where dtype.is_floating_point():
    """One erosion or dilation step: each element's footprint all set
    (erosion) or any set (dilation), `border` outside."""
    var ctx = src.context()
    var count = src.size()
    var out = Dynamic[dtype, 1](row_major(_dyn_shape[1](count)), ctx)
    var strides = _strides(dims, rank)
    var sp = src.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var op = out.tile()
    var neighbors = 1
    for _ in range(rank):
        neighbors *= 3

    @always_inline
    def body[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var sp,
        var op,
        var dims,
        var strides,
        var neighbors,
        var connectivity,
        var border,
    }:
        var i = coord_to_index_list(coord)[0]
        var at = SIMD[DType.int64, 8](0)
        var rest = i
        comptime for a in range(rank):
            at[a] = Int64(rest // Int(strides[a]))
            rest = rest % Int(strides[a])
        var result = not dilate
        for code in range(neighbors):
            var c = code
            var nonzero = 0
            var offset = 0
            var inside = True
            comptime for a in range(rank):
                var d = c % 3 - 1
                c //= 3
                if d != 0:
                    nonzero += 1
                var j = Int(at[a]) + d
                if j < 0 or j >= Int(dims[a]):
                    inside = False
                offset += j * Int(strides[a])
            if nonzero > connectivity:
                continue
            var set = border if not inside else sp[
                unsafe_offset=offset
            ] != Scalar[dtype](0)
            comptime if dilate:
                if set:
                    result = True
            else:
                if not set:
                    result = False
        op.ptr[unsafe_offset=i] = Scalar[dtype](1) if result else Scalar[dtype](
            0
        )

    elementwise[simd_width=1, target=_target[gpu]()](body, Coord(count), ctx)
    ctx.synchronize()
    return out^


def binary_erosion[
    T: TensorLike, gpu: Bool = False
](
    input: T,
    connectivity: Int = 1,
    iterations: Int = 1,
    border_value: Bool = False,
) raises -> Dynamic[T.dtype, T.LayoutType.rank] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """Binary erosion by SciPy's default structuring element.
    `scipy.ndimage.binary_erosion(input, structure=generate_binary_structure(rank,
    connectivity), iterations=iterations, border_value=border_value)`.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        gpu: Whether to run on the input's device.

    Args:
        input: The array; nonzero is set.
        connectivity: The structuring element's reach, `1` to `rank`.
        iterations: How many times to erode, at least 1.
        border_value: What the input reads as outside the array.

    Returns:
        `0`/`1` in `input`'s `dtype` and shape.

    Raises:
        On a connectivity outside `[1, rank]`, fewer than one iteration,
        or a device failure.
    """
    comptime rank = T.LayoutType.rank
    if connectivity < 1 or connectivity > rank:
        raise Error("binary_erosion: connectivity must be in [1, ", rank, "]")
    if iterations < 1:
        raise Error("binary_erosion: iterations must be at least 1")
    var dims = _shape_of(input)
    var flat = _same_order(input, row_major(_dyn_shape[1](input.size())))
    for _ in range(iterations):
        flat = _morph[T.dtype, rank, gpu, False](
            flat, dims, connectivity, border_value
        )
    return _shaped[T.dtype, rank](flat^, dims)


def binary_dilation[
    T: TensorLike, gpu: Bool = False
](
    input: T,
    connectivity: Int = 1,
    iterations: Int = 1,
    border_value: Bool = False,
) raises -> Dynamic[T.dtype, T.LayoutType.rank] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """Binary dilation by SciPy's default structuring element.
    `scipy.ndimage.binary_dilation(input, structure=generate_binary_structure(rank,
    connectivity), iterations=iterations, border_value=border_value)`.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        gpu: Whether to run on the input's device.

    Args:
        input: The array; nonzero is set.
        connectivity: The structuring element's reach, `1` to `rank`.
        iterations: How many times to dilate, at least 1.
        border_value: What the input reads as outside the array.

    Returns:
        `0`/`1` in `input`'s `dtype` and shape.

    Raises:
        As `binary_erosion`.
    """
    comptime rank = T.LayoutType.rank
    if connectivity < 1 or connectivity > rank:
        raise Error("binary_dilation: connectivity must be in [1, ", rank, "]")
    if iterations < 1:
        raise Error("binary_dilation: iterations must be at least 1")
    var dims = _shape_of(input)
    var flat = _same_order(input, row_major(_dyn_shape[1](input.size())))
    for _ in range(iterations):
        flat = _morph[T.dtype, rank, gpu, True](
            flat, dims, connectivity, border_value
        )
    return _shaped[T.dtype, rank](flat^, dims)


def distance_transform_edt[
    T: TensorLike, gpu: Bool = False
](input: T) raises -> Dynamic[T.dtype, T.LayoutType.rank] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank >= 1
    and T.LayoutType.rank <= 8
):
    """The Euclidean distance from every element to the nearest zero
    element. `scipy.ndimage.distance_transform_edt(input)`, unit sampling,
    distances only; per this module's docstring.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8, floating point.
        gpu: Whether to run on the input's device.

    Args:
        input: The array; zero elements are the background the distances
            are measured to.

    Returns:
        The distances, `input`'s shape; infinite where there is no zero
        element at all.

    Raises:
        If a device operation fails.
    """
    comptime dtype = T.dtype
    comptime rank = T.LayoutType.rank
    var dims = _shape_of(input)
    var strides = _strides(dims, rank)
    var count = input.size()
    var src = _same_order(input, row_major(_dyn_shape[1](count)))
    var ctx = src.context()
    var f = Dynamic[dtype, 1](row_major(_dyn_shape[1](count)), ctx)
    var sp = src.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var fp = f.tile()
    var inf = Scalar[dtype].MAX

    @always_inline
    def seed[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var sp, var fp, var inf}:
        var i = coord_to_index_list(coord)[0]
        fp.ptr[unsafe_offset=i] = (
            Scalar[dtype](0) if sp[unsafe_offset=i] == Scalar[dtype](0) else inf
        )

    elementwise[simd_width=1, target=_target[gpu]()](seed, Coord(count), ctx)
    ctx.synchronize()
    for axis in range(rank):
        var n = Int(dims[axis])
        var stride = Int(strides[axis])
        var lines = count // n
        var d = Dynamic[dtype, 1](row_major(_dyn_shape[1](count)), ctx)
        var v = Dynamic[DType.int32, 1](row_major(_dyn_shape[1](count)), ctx)
        var z = Dynamic[dtype, 1](
            row_major(_dyn_shape[1](lines * (n + 1))), ctx
        )
        var inp = f.tile()
        var outp = d.tile()
        var vp = v.tile()
        var zp = z.tile()

        @always_inline
        def envelope[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var inp,
            var outp,
            var vp,
            var zp,
            var n,
            var stride,
            var strides,
            var dims,
            var inf,
        }:
            var l = coord_to_index_list(coord)[0]
            var base = 0
            var rest = l
            comptime for a in range(rank - 1, -1, -1):
                if Int(strides[a]) != stride:
                    var extent = Int(dims[a])
                    base += (rest % extent) * Int(strides[a])
                    rest //= extent
            var vb = l * n
            var zb = l * (n + 1)
            # Felzenszwalb-Huttenlocher, skipping the infinite samples,
            # which root no parabola.
            var k = -1
            for q in range(n):
                var fq = inp.ptr[unsafe_offset=base + q * stride]
                if fq >= inf:
                    continue
                if k < 0:
                    k = 0
                    vp.ptr[unsafe_offset=vb] = Int32(q)
                    zp.ptr[unsafe_offset=zb] = -inf
                    zp.ptr[unsafe_offset=zb + 1] = inf
                    continue
                var s: Scalar[dtype]
                while True:
                    var p = Int(vp.ptr[unsafe_offset=vb + k])
                    var fp_ = inp.ptr[unsafe_offset=base + p * stride]
                    s = (
                        (fq + Scalar[dtype](q * q))
                        - (fp_ + Scalar[dtype](p * p))
                    ) / Scalar[dtype](2 * (q - p))
                    if s <= zp.ptr[unsafe_offset=zb + k] and k > 0:
                        k -= 1
                    else:
                        break
                k += 1
                vp.ptr[unsafe_offset=vb + k] = Int32(q)
                zp.ptr[unsafe_offset=zb + k] = s
                zp.ptr[unsafe_offset=zb + k + 1] = inf
            if k < 0:
                for q in range(n):
                    outp.ptr[unsafe_offset=base + q * stride] = inf
                return
            var j = 0
            for q in range(n):
                while zp.ptr[unsafe_offset=zb + j + 1] < Scalar[dtype](q):
                    j += 1
                var p = Int(vp.ptr[unsafe_offset=vb + j])
                var dq = Scalar[dtype](q - p)
                outp.ptr[unsafe_offset=base + q * stride] = (
                    dq * dq + inp.ptr[unsafe_offset=base + p * stride]
                )

        elementwise[simd_width=1, target=_target[gpu]()](
            envelope, Coord(lines), ctx
        )
        ctx.synchronize()
        _ = v^
        _ = z^
        f = d^
    var rp = f.tile()

    @always_inline
    def root[w: Int, alignment: Int = 1](coord: Coord) {var rp, var inf}:
        var i = coord_to_index_list(coord)[0]
        var v = rp.ptr[unsafe_offset=i]
        rp.ptr[unsafe_offset=i] = v if v >= inf else _sqrt(v)

    elementwise[simd_width=1, target=_target[gpu]()](root, Coord(count), ctx)
    ctx.synchronize()
    _ = src^
    return _shaped[dtype, rank](f^, dims)
