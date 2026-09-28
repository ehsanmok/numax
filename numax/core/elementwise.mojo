"""NumPy-named elementwise mathematics over `Tensor`.

**Tier 2 in shape, both targets in fact**, on the same terms as
`numax.core.rowwise`: these are `Plain`-only -- `TileTensor` holds raw
`dtype` lanes -- but they are not host-only. Every routine here runs through
`numax.core._drive`, which launches one capturing body on the target its
`gpu: Bool` parameter names: `max.algorithm.elementwise` on the device, the
same threaded walk on a large host tensor, a serial SIMD loop on a small
one. The mathematics is tier 1 -- fixed work per element, no branching --
and `numax.core.functional.map`/`map_to` remain the primitives a caller holding
a concrete shape and a hot loop can reach for directly.

`gpu: Bool = False` is the last compile-time parameter of every routine, so
`exp(a)` is the host spelling and `exp[gpu=True](a)` the device one. Asking
for a target the tensor's memory does not live on is not an error: the call
falls back to a host walk over `to_host()` and prints one line on `stderr`
naming the spelling that would not have. `_drive`'s docstring has the rest.

`Plain`-only. `FloatLike` deliberately stays small (`numax/core/numeric.mojo`
lists what it carries and why), and the functions here are `std.math` calls
over raw `SIMD` rather than trait growth: nothing in numax needs `arctanh`
at `Dual`, and adding twenty methods to the trait to reach parity on names
would make every one of the seven conformers implement them.

`min` and `max` are Mojo builtins, so the elementwise forms here are
`minimum` and `maximum` -- NumPy's own names for the two-argument form, so
nothing is lost. `nextafter` is absent from `std.math` under the pinned
toolchain and is therefore not provided.

Every two-argument routine here has a second overload taking two shapes
NumPy would broadcast -- `arctan2` of a column against a row, `maximum` of
a matrix against its per-column maxima. It returns a `Dynamic`, since the
broadcast extents are run-time values; the same-shape overload still
matches first and still returns the input's own layout type. A stretched
axis has stride 0, so the broadcast bodies run one element per thread and
rebuild each operand's offset in integer arithmetic.

`logaddexp`, `logaddexp2`, `sinc`, `heaviside` and `nan_to_num` are the
NumPy names that are compositions rather than one `std.math` call; each is
still one uniform body, both sides of every `select` evaluated and safe.

`tanh` is the one name this module shares with `numax.special.activations`,
which has the `FloatLike` scalar of the same name. The root package exports
the activation, because that is the one a kernel calls; the tensor form here
is `numax.core.tanh`.
"""

from std.utils.numerics import (
    inf as _inf,
    max_finite as _max_finite,
    min_finite as _min_finite,
    neg_inf as _neg_inf,
)
from std.math import (
    acos as _std_acos,
    acosh as _std_acosh,
    asin as _std_asin,
    asinh as _std_asinh,
    atan as _std_atan,
    atan2 as _std_atan2,
    atanh as _std_atanh,
    cbrt as _std_cbrt,
    ceil as _std_ceil,
    copysign as _std_copysign,
    cos as _std_cos,
    expm1 as _std_expm1,
    floor as _std_floor,
    hypot as _std_hypot,
    isnan as _std_isnan,
    log10 as _std_log10,
    remainder as _std_remainder,
    round as _std_round,
    rsqrt as _std_rsqrt,
    sin as _std_sin,
    sinh as _std_sinh,
    sqrt as _std_sqrt,
    tan as _std_tan,
    tanh as _std_tanh,
    trunc as _std_trunc,
)

from layout import Coord, coord_to_index_list
from layout.tile_layout import TensorLayout

# `exp`, `log`, `log1p`, `log2`, `exp2` and `cosh` at float64 are numax's
# own (`numax/core/libm.mojo`); `std.math`'s are from `1e4` to `1e9` ulp
# off there.
from .libm import exp as _std_exp
from .libm import log as _std_log
from .libm import cosh as _std_cosh
from .libm import exp2 as _std_exp2
from .libm import log1p as _std_log1p
from .libm import log2 as _std_log2
from .tensorlike import TensorLike, dim, is_row_major
from .tensor import Dynamic, Static, Tensor
from ._drive import (
    _BroadcastRank,
    _check_device,
    _flat,
    _flat_out,
    _launch,
    _notice,
    _width,
    binary,
    broadcast_binary,
    unary,
)


def _exp_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_exp(x)


def exp[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise `e**x`.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of exponents `x`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `e**x` for every
        element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_exp_op[dtype, _], gpu=gpu, name="exp"](a)


def _exp2_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_exp2(x)


def exp2[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise `2**x`.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of exponents `x`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `2**x` for every
        element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_exp2_op[dtype, _], gpu=gpu, name="exp2"](a)


def _expm1_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_expm1(x)


def expm1[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise `e**x - 1`, accurate for small `x`.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of exponents `x`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `e**x - 1` for
        every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_expm1_op[dtype, _], gpu=gpu, name="expm1"](a)


def _log_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_log(x)


def log[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise natural logarithm.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of arguments `x`; negative `x` gives NaN.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the natural
        logarithm of every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_log_op[dtype, _], gpu=gpu, name="log"](a)


def _log2_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_log2(x)


def log2[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise base-2 logarithm.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of arguments `x`; negative `x` gives NaN.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the base-2
        logarithm of every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_log2_op[dtype, _], gpu=gpu, name="log2"](a)


def _log10_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_log10(x)


def log10[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise base-10 logarithm.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of arguments `x`; negative `x` gives NaN.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the base-10
        logarithm of every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_log10_op[dtype, _], gpu=gpu, name="log10"](a)


def _log1p_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_log1p(x)


def log1p[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise `log(1 + x)`, accurate for small `x`.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of arguments `x`, meaningful for `x > -1`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `log(1 + x)` for
        every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_log1p_op[dtype, _], gpu=gpu, name="log1p"](a)


def _sqrt_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_sqrt(x)


def sqrt[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise square root.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of arguments `x`; negative `x` gives NaN.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the square root of
        every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_sqrt_op[dtype, _], gpu=gpu, name="sqrt"](a)


def _rsqrt_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_rsqrt(x)


def rsqrt[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise `1 / sqrt(x)`.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of arguments `x`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `1 / sqrt(x)` for
        every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_rsqrt_op[dtype, _], gpu=gpu, name="rsqrt"](a)


def _cbrt_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_cbrt(x)


def cbrt[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise cube root.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of arguments `x`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the real cube root
        of every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_cbrt_op[dtype, _], gpu=gpu, name="cbrt"](a)


def _sin_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_sin(x)


def sin[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise sine.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of angles in radians.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the sine of every
        element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_sin_op[dtype, _], gpu=gpu, name="sin"](a)


def _cos_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_cos(x)


def cos[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise cosine.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of angles in radians.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the cosine of every
        element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_cos_op[dtype, _], gpu=gpu, name="cos"](a)


def _tan_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_tan(x)


def tan[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise tangent.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of angles in radians.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the tangent of
        every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_tan_op[dtype, _], gpu=gpu, name="tan"](a)


def _arcsin_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_asin(x)


def arcsin[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise inverse sine. `numpy.arcsin`.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of arguments in `[-1, 1]`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the inverse sine of
        every element, in radians.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_arcsin_op[dtype, _], gpu=gpu, name="arcsin"](a)


def _arccos_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_acos(x)


def arccos[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise inverse cosine. `numpy.arccos`.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of arguments in `[-1, 1]`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the inverse cosine
        of every element, in radians.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_arccos_op[dtype, _], gpu=gpu, name="arccos"](a)


def _arctan_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_atan(x)


def arctan[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise inverse tangent. `numpy.arctan`.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of arguments `x`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the inverse tangent
        of every element, in radians.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_arctan_op[dtype, _], gpu=gpu, name="arctan"](a)


def _sinh_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_sinh(x)


def sinh[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise hyperbolic sine.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of arguments `x`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the hyperbolic sine
        of every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_sinh_op[dtype, _], gpu=gpu, name="sinh"](a)


def _cosh_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_cosh(x)


def cosh[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise hyperbolic cosine.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of arguments `x`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the hyperbolic
        cosine of every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_cosh_op[dtype, _], gpu=gpu, name="cosh"](a)


def _tanh_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_tanh(x)


def tanh[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise hyperbolic tangent.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of arguments `x`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the hyperbolic
        tangent of every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_tanh_op[dtype, _], gpu=gpu, name="tanh"](a)


def _arcsinh_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_asinh(x)


def arcsinh[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise inverse hyperbolic sine. `numpy.arcsinh`.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of arguments `x`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the inverse
        hyperbolic sine of every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_arcsinh_op[dtype, _], gpu=gpu, name="arcsinh"](a)


def _arccosh_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_acosh(x)


def arccosh[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise inverse hyperbolic cosine. `numpy.arccosh`.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of arguments `x >= 1`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the inverse
        hyperbolic cosine of every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_arccosh_op[dtype, _], gpu=gpu, name="arccosh"](a)


def _arctanh_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_atanh(x)


def arctanh[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise inverse hyperbolic tangent. `numpy.arctanh`.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of arguments in `(-1, 1)`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the inverse
        hyperbolic tangent of every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_arctanh_op[dtype, _], gpu=gpu, name="arctanh"](a)


def _floor_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_floor(x)


def floor[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise largest integer `<= x`.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of values to round down.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the floor of every
        element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_floor_op[dtype, _], gpu=gpu, name="floor"](a)


def _ceil_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_ceil(x)


def ceil[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise smallest integer `>= x`.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of values to round up.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the ceiling of
        every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_ceil_op[dtype, _], gpu=gpu, name="ceil"](a)


def _trunc_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_trunc(x)


def trunc[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise `x` rounded toward zero.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of values to round toward zero.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding every element with
        its fractional part dropped.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_trunc_op[dtype, _], gpu=gpu, name="trunc"](a)


def _round_op[dtype: DType, w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return _std_round(x)


def round[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise `x` rounded to nearest, ties to even.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of values to round.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding every element
        rounded to the nearest integer, ties to even.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_round_op[dtype, _], gpu=gpu, name="round"](a)


def rint[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise `x` rounded to nearest, ties to even. `numpy.rint`.

    The same operation as `round`, under the name NumPy gives it when no
    decimal count is involved; both are half-to-even, so `rint` exists to
    be found rather than to do anything `round` does not.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of values to round.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding every element
        rounded to the nearest integer, ties to even.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_round_op[dtype, _], gpu=gpu, name="rint"](a)


def _sign_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    # Not `copysign(1, x)`: that gives -1 at `-0.0`, where NumPy gives 0.
    # Two comparisons keep `sign(0.0) == 0.0`, `sign(-0.0) == -0.0` and
    # `sign(nan) == nan`, which is NumPy's full contract.
    # `.gt`/`.lt` rather than `>`/`<`: at `w == 1` the operators narrow to
    # `Bool`, which has no per-lane `select`.
    var zero = SIMD[dtype, w](0)
    var pos = x.gt(zero).select(SIMD[dtype, w](1), zero)
    # `+ (x - x)` carries a NaN through and is exact everywhere else.
    return x.lt(zero).select(SIMD[dtype, w](-1), pos) + (x - x)


comptime _PI = 3.141592653589793
comptime _LN2 = 0.6931471805599453
comptime _LOG2E = 1.4426950408889634


def sign[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise `-1`, `0` or `1` by the sign of `x`. `numpy.sign`.

    Zero maps to zero and NaN to NaN, which is why this is not
    `copysign(1, x)` -- that answers `-1` for `-0.0` and `1` for NaN.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor whose signs are taken.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `-1`, `0` or `1`
        per element, with NaN kept as NaN.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_sign_op[dtype, _], gpu=gpu, name="sign"](a)


def _square_op[dtype: DType, w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return x * x


def square[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """Elementwise `x * x`. `numpy.square`.

    One multiply rather than `power(a, 2)`'s general exponentiation, and
    exact where the general form is not.

    Parameters:
        T: The `TensorLike` type of `a`; it fixes the dtype and the row-major
            layout.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of values to square.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `x * x` for every
        element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_square_op[dtype, _], gpu=gpu, name="square"](a)


def _reciprocal_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return SIMD[dtype, w](1) / x


def reciprocal[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise `1 / x`. `numpy.reciprocal`.

    A zero gives an infinity rather than raising, as NumPy's does with the
    error state at its default.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of values to invert.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `1 / x` for every
        element, an infinity where `x` is zero.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[
        T,
        op=_reciprocal_op[dtype, _],
        gpu=gpu,
        name="reciprocal",
    ](a)


def _degrees_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return x * SIMD[dtype, w](57.295779513082320876798154814105)


def degrees[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise radians to degrees, `x * 180 / pi`. `numpy.degrees`.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of angles in radians.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding every angle
        converted to degrees.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_degrees_op[dtype, _], gpu=gpu, name="degrees"](a)


def _radians_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return x * SIMD[dtype, w](0.017453292519943295769236907684886)


def radians[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise degrees to radians, `x * pi / 180`. `numpy.radians`.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of angles in degrees.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding every angle
        converted to radians.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_radians_op[dtype, _], gpu=gpu, name="radians"](a)


def _arctan2_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return _std_atan2(a, b)


def arctan2[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise `atan2(a, b)`, quadrant-aware. `numpy.arctan2`.

    Parameters:
        T: The `TensorLike` type of both operands, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of `y` coordinates, the first argument of `atan2`.
        b: Tensor of `x` coordinates, the second argument. At `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `atan2(a, b)` for
        every element pair, in radians in `[-pi, pi]`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_arctan2_op[dtype, _], gpu=gpu, name="arctan2"](a, b)


def _hypot_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return _std_hypot(a, b)


def hypot[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise `sqrt(a*a + b*b)` without intermediate overflow.

    Parameters:
        T: The `TensorLike` type of both operands, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of first legs.
        b: Tensor of second legs. At `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `sqrt(a*a + b*b)`
        for every element pair.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_hypot_op[dtype, _], gpu=gpu, name="hypot"](a, b)


def _copysign_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return _std_copysign(a, b)


def copysign[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise magnitude of `a` with the sign of `b`.

    Parameters:
        T: The `TensorLike` type of both operands, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor supplying the magnitudes.
        b: Tensor supplying the signs. At `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `abs(a)` carrying
        the sign of `b`, per element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_copysign_op[dtype, _], gpu=gpu, name="copysign"](a, b)


def _remainder_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return _std_remainder(a, b)


def remainder[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise IEEE remainder of `a` and `b`.

    Parameters:
        T: The `TensorLike` type of both operands, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of dividends.
        b: Tensor of divisors. At `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the IEEE remainder
        `a - n * b`, with `n` the integer nearest `a / b`, per element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_remainder_op[dtype, _], gpu=gpu, name="remainder"](
        a, b
    )


def _abs_op[dtype: DType, w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    # `x.__abs__()` rather than `abs(x)`: the module-level `abs` below hides
    # the builtin throughout this file.
    return x.__abs__()


def abs[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """Elementwise magnitude. `numpy.abs`.

    Defining `abs` here hides Mojo's builtin `abs` for the rest of this
    file, which is why the private op above spells it `x.__abs__()`. A
    caller who imports this name pays the same price in their own file,
    exactly as `from numpy import abs` does in Python.

    Parameters:
        T: The `TensorLike` type of `a`; it fixes the dtype and the row-major
            layout.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor whose magnitudes are taken.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the absolute value
        of every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_abs_op[dtype, _], gpu=gpu, name="abs"](a)


def _maximum_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    # `max` is IEEE `maxNum`: it *returns the other operand* when one is
    # NaN. `numpy.maximum` propagates instead, so the NaN is put back.
    # `fmax` below is the spelling that wants `max`'s own behavior.
    comptime if dtype.is_floating_point():
        return _std_isnan(a).select(a, _std_isnan(b).select(b, max(a, b)))
    else:
        return max(a, b)


def maximum[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """Elementwise larger of the two, NaN-propagating. `numpy.maximum`.

    A NaN in either operand gives NaN, which is NumPy's rule and *not* the
    hardware's: `max` on a `SIMD` is IEEE `maxNum` and quietly returns the
    other operand. `fmax` is the name for that behavior.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: First tensor compared.
        b: Second tensor compared. At `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the larger of each
        element pair, NaN where either is NaN.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_maximum_op[dtype, _], gpu=gpu, name="maximum"](a, b)


def _minimum_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    comptime if dtype.is_floating_point():
        return _std_isnan(a).select(a, _std_isnan(b).select(b, min(a, b)))
    else:
        return min(a, b)


def minimum[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """Elementwise smaller of the two, NaN-propagating. `numpy.minimum`.

    A NaN in either operand gives NaN, as `maximum` records; `fmin` is the
    name that skips it.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: First tensor compared.
        b: Second tensor compared. At `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the smaller of each
        element pair, NaN where either is NaN.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_minimum_op[dtype, _], gpu=gpu, name="minimum"](a, b)


def _fmax_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return max(a, b)


def fmax[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """Elementwise larger of the two, ignoring NaN. `numpy.fmax`.

    `fmax(nan, x)` is `x`, where `maximum(nan, x)` is NaN. This is the one
    of the pair that maps straight onto the hardware instruction, so it is
    also the cheaper of the two on a float dtype.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: First tensor compared.
        b: Second tensor compared. At `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the larger of each
        element pair, the non-NaN one where one is NaN.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_fmax_op[dtype, _], gpu=gpu, name="fmax"](a, b)


def _fmin_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return min(a, b)


def fmin[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """Elementwise smaller of the two, ignoring NaN. `numpy.fmin`.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: First tensor compared.
        b: Second tensor compared. At `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the smaller of each
        element pair, the non-NaN one where one is NaN.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_fmin_op[dtype, _], gpu=gpu, name="fmin"](a, b)


def clip[
    T: TensorLike, gpu: Bool = False
](a: T, lo: Scalar[T.dtype], hi: Scalar[T.dtype]) raises -> Tensor[
    T.dtype, T.LayoutType
] where is_row_major[T]:
    """Every element confined to `[lo, hi]`. `numpy.clip`.

    Two run-time bounds rather than one, so this carries its own body
    instead of going through `_drive.binary_scalar`; both are captured by
    value and splatted to the launch width.

    Parameters:
        T: The `TensorLike` type of `a`; it fixes the dtype and the row-major
            layout.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor whose elements are clipped.
        lo: Lower bound, applied first.
        hi: Upper bound, applied after `lo`, so it wins when `lo > hi`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `min(max(x, lo),
        hi)` for every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    if not _check_device[gpu=gpu](a):
        _notice[gpu]("clip")
        var n = a.size()
        var values = a.to_host()
        var walked = List[Scalar[dtype]](length=n, fill=0)
        for i in range(n):
            walked[i] = min(max(values[i], lo), hi)
        return Tensor[dtype, LayoutType](a.tile().layout, walked^, a.context())

    var ctx = a.context()
    var out = Tensor[dtype, LayoutType]._uninitialized(ctx, a.tile().layout)
    var xs = _flat(a)
    var ys = _flat_out(out)

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var ys, var lo, var hi}:
        ys.store[width](
            coord,
            min(
                max(xs.load[width](coord), SIMD[dtype, width](lo)),
                SIMD[dtype, width](hi),
            ),
        )

    _launch[gpu=gpu, lanes=_width[dtype, gpu]()](body, a.size(), ctx)
    return out^


def diff[
    T: TensorLike,
    gpu: Bool = False,
](a: T) raises -> Static[T.dtype, dim[T, 0] - 1] where (
    dim[T, 0] >= 1
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
    and is_row_major[T]
):
    """First differences, `out[i] = a[i+1] - a[i]`. `numpy.diff`.

    Rank-1, and one element shorter than its input -- which is why the
    output length is a compile-time function of `n` rather than a runtime
    value. One launch: the shifted read is `Coord(i + 1)` against the same
    flat view, which stays in bounds because the domain is `n - 1` wide.

    Parameters:
        T: The `TensorLike` type of `a`: rank 1, a compile-time length of at
            least 1, row-major.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Rank-1 tensor of `n` samples.

    Returns:
        A new `Static` rank-1 tensor of `T.dtype` and length `n - 1` holding
        `a[i+1] - a[i]`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime n = dim[T, 0]
    if not _check_device[gpu=gpu](a):
        _notice[gpu]("diff")
        var values = a.to_host()
        var walked = List[Scalar[dtype]](length=n - 1, fill=0)
        for i in range(n - 1):
            walked[i] = values[i + 1] - values[i]
        return Static[dtype, n - 1](walked^, a.context())

    var ctx = a.context()
    var out = Static[dtype, n - 1]._uninitialized(ctx)
    var xs = _flat(a)
    var ys = _flat_out(out)

    @always_inline
    def body[width: Int, alignment: Int = 1](coord: Coord) {var xs, var ys}:
        var i = coord_to_index_list(coord)[0]
        ys.store[width](
            coord, xs.load[width](Coord(i + 1)) - xs.load[width](coord)
        )

    _launch[gpu=gpu, lanes=_width[dtype, gpu]()](body, n - 1, ctx)
    return out^


def gradient[
    T: TensorLike,
    gpu: Bool = False,
](a: T, spacing: Scalar[T.dtype] = 1) raises -> Static[
    T.dtype, dim[T, 0]
] where (
    dim[T, 0] >= 2
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
    and is_row_major[T]
):
    """Central differences interior, one-sided at the ends. `numpy.gradient`.

    Second-order accurate in the interior and first-order at the two
    endpoints, matching NumPy's default for a uniform grid.

    One launch at one element per thread. The endpoints are not a special
    case in the body: element `i` reads `a[min(i+1, n-1)] - a[max(i-1, 0)]`
    and divides by the spacing times the number of steps that span covers,
    which is NumPy's definition written so the two ends fall out of it. The
    divisor is selected between two captured scalars rather than built from
    the index, since a run-time `Int` widened to a float is not a Metal
    instruction.

    Parameters:
        T: The `TensorLike` type of `a`: rank 1, a compile-time length of at
            least 2, row-major.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Rank-1 tensor of `n` samples on a uniform grid.
        spacing: Distance between neighboring samples; defaults to 1.

    Returns:
        A new `Static` rank-1 tensor of `T.dtype` and length `n` holding the
        estimated derivative at every sample.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime n = dim[T, 0]
    if not _check_device[gpu=gpu](a):
        _notice[gpu]("gradient")
        var values = a.to_host()
        var walked = List[Scalar[dtype]](length=n, fill=0)
        walked[0] = (values[1] - values[0]) / spacing
        walked[n - 1] = (values[n - 1] - values[n - 2]) / spacing
        for i in range(1, n - 1):
            walked[i] = (values[i + 1] - values[i - 1]) / (spacing + spacing)
        return Static[dtype, n](walked^, a.context())

    var ctx = a.context()
    var out = Static[dtype, n]._uninitialized(ctx)
    var xs = _flat(a)
    var ys = _flat_out(out)
    var two_spacing = spacing + spacing

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var ys, var spacing, var two_spacing}:
        var i = coord_to_index_list(coord)[0]
        var lo = i - 1 if i > 0 else 0
        var hi = i + 1 if i < n - 1 else n - 1
        var divisor = two_spacing if hi - lo == 2 else spacing
        ys.store[width](
            coord,
            (xs.load[width](Coord(hi)) - xs.load[width](Coord(lo)))
            / SIMD[dtype, width](divisor),
        )

    _launch[gpu=gpu, lanes=1](body, n, ctx)
    return out^


def _logaddexp_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    # `max + log1p(exp(min - max))` never overflows. Equal operands take the
    # `a + ln 2` branch, which is what keeps `logaddexp(-inf, -inf) == -inf`
    # and `logaddexp(inf, inf) == inf` rather than the `inf - inf` NaN; a
    # NaN on either side is carried by `a + b`.
    var big = max(a, b)
    var r = big + _std_log1p(_std_exp(min(a, b) - big))
    var same = a.eq(b).select(a + SIMD[dtype, w](_LN2), r)
    return (_std_isnan(a) | _std_isnan(b)).select(a + b, same)


def logaddexp[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise `log(exp(a) + exp(b))` without overflow.
    `numpy.logaddexp`.

    Computed as `max(a, b) + log1p(exp(min(a, b) - max(a, b)))`, so neither exponential
    is formed at full size; equal infinities give that infinity back, and
    a NaN on either side gives NaN. The reduction over a whole tensor is
    `numax.special.logsumexp`.

    Parameters:
        T: The `TensorLike` type of both operands, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of first log-magnitudes.
        b: Tensor of second log-magnitudes. At `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding
        `log(exp(a) + exp(b))` for every element pair.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_logaddexp_op[dtype, _], gpu=gpu, name="logaddexp"](
        a, b
    )


def _logaddexp2_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    # `logaddexp` in base 2: `max + log2(1 + 2**(min - max))`, with the same
    # equal-operand and NaN branches (`a + 1` is `log2(2 * 2**a)`).
    var big = max(a, b)
    var r = big + _std_log1p(_std_exp2(min(a, b) - big)) * SIMD[dtype, w](
        _LOG2E
    )
    var same = a.eq(b).select(a + SIMD[dtype, w](1), r)
    return (_std_isnan(a) | _std_isnan(b)).select(a + b, same)


def logaddexp2[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Elementwise `log2(2**a + 2**b)` without overflow. `numpy.logaddexp2`.

    `logaddexp` in base 2, with the same handling of equal infinities and
    NaN.

    Parameters:
        T: The `TensorLike` type of both operands, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of first base-2 log-magnitudes.
        b: Tensor of second base-2 log-magnitudes. At `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding
        `log2(2**a + 2**b)` for every element pair.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_logaddexp2_op[dtype, _], gpu=gpu, name="logaddexp2"](
        a, b
    )


def _heaviside_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w], h0: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    var zero = SIMD[dtype, w](0)
    var step = x.gt(zero).select(SIMD[dtype, w](1), zero)
    # `+ (x - x)` carries a NaN `x` through, as in `sign`.
    return x.eq(zero).select(h0, step) + (x - x)


def heaviside[
    T: TensorLike, gpu: Bool = False
](x: T, h0: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The Heaviside step: `0` below zero, `h0` at zero, `1` above.
    `numpy.heaviside`.

    `-0.0` is zero, so it takes `h0` too; a NaN `x` gives NaN, as NumPy
    does. The broadcasting overload below is the one a scalar `h0` wants,
    against a one-element tensor.

    Parameters:
        T: The `TensorLike` type of both operands, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        x: Tensor of inputs to the step.
        h0: Tensor of the values taken where `x == 0`. At `x`'s shape.

    Returns:
        A new `Tensor` at `x`'s layout and `T.dtype` holding `0`, `h0` or `1`
        by the sign of each element of `x`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_heaviside_op[dtype, _], gpu=gpu, name="heaviside"](
        x, h0
    )


def _sinc_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    # Both sides of the select are evaluated, so the zero lanes divide by a
    # stand-in `1` rather than by `0`.
    var zero = SIMD[dtype, w](0)
    var px = x * SIMD[dtype, w](_PI)
    var at_zero = x.eq(zero)
    var safe = at_zero.select(SIMD[dtype, w](1), px)
    return at_zero.select(SIMD[dtype, w](1), _std_sin(safe) / safe)


def sinc[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The normalized sinc, `sin(pi x) / (pi x)`, with `sinc(0) == 1`.
    `numpy.sinc`.

    Normalized, as NumPy's is: its zeros are the nonzero integers. The
    unnormalized `sin(x) / x` is `sinc(x / pi)`.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of inputs.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding
        `sin(pi x) / (pi x)` for every element, and `1` where it is zero.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_sinc_op[dtype, _], gpu=gpu, name="sinc"](a)


def nan_to_num[
    T: TensorLike, gpu: Bool = False
](
    a: T,
    nan: Scalar[T.dtype] = 0,
    posinf: Optional[Scalar[T.dtype]] = None,
    neginf: Optional[Scalar[T.dtype]] = None,
) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """NaN and the infinities replaced by finite values. `numpy.nan_to_num`.

    NaN becomes `nan`, `+inf` becomes `posinf` and `-inf` becomes `neginf`;
    an unset infinity takes the largest finite value of its sign, as NumPy
    does. Every other element is copied. A new tensor, never in place --
    NumPy's `copy=False` has no counterpart here.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor to clean.
        nan: Value written for a NaN.
        posinf: Value written for `+inf`; the largest finite value if unset.
        neginf: Value written for `-inf`; the most negative finite value if
            unset.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` with every NaN and
        infinity replaced.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    var nan_value = nan
    var pos_value = posinf.or_else(_max_finite[dtype]())
    var neg_value = neginf.or_else(_min_finite[dtype]())
    if not _check_device[gpu=gpu](a):
        _notice[gpu]("nan_to_num")
        var values = a.to_host()
        for i in range(len(values)):
            var x = values[i]
            if x != x:
                values[i] = nan_value
            elif x == _inf[dtype]():
                values[i] = pos_value
            elif x == _neg_inf[dtype]():
                values[i] = neg_value
        return Tensor[dtype, LayoutType](a.tile().layout, values^, a.context())

    var ctx = a.context()
    var out = Tensor[dtype, LayoutType]._uninitialized(ctx, a.tile().layout)
    var xs = _flat(a)
    var ys = _flat_out(out)

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {
        var xs, var ys, var nan_value, var pos_value, var neg_value
    }:
        var x = xs.load[width](coord)
        var r = x.eq(SIMD[dtype, width](_inf[dtype]())).select(
            SIMD[dtype, width](pos_value), x
        )
        r = x.eq(SIMD[dtype, width](_neg_inf[dtype]())).select(
            SIMD[dtype, width](neg_value), r
        )
        ys.store[width](
            coord, _std_isnan(x).select(SIMD[dtype, width](nan_value), r)
        )

    _launch[gpu=gpu, lanes=_width[dtype, gpu]()](body, a.size(), ctx)
    return out^


# The broadcasting forms, matching `numax.core.ops`: same operation, two
# shapes NumPy would broadcast rather than one shape twice, and a `Dynamic`
# result because the broadcast extents are run-time values.


def arctan2[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == B.dtype
    and is_row_major[A]
    and is_row_major[B]
    and A.dtype.is_floating_point()
):
    """Elementwise `atan2(a, b)` at two broadcastable shapes.

    Parameters:
        A: The `TensorLike` type of `a`, row-major and floating-point.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of `y` coordinates. Read through stride 0 on any axis it
            stretches.
        b: Tensor of `x` coordinates. At a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding `atan2(a, b)` in radians.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A,
        B,
        op=_arctan2_op[dtype, _],
        gpu=gpu,
        name="arctan2",
    ](a, b)


def hypot[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == B.dtype
    and is_row_major[A]
    and is_row_major[B]
    and A.dtype.is_floating_point()
):
    """Elementwise `sqrt(a*a + b*b)` at two broadcastable shapes.

    Parameters:
        A: The `TensorLike` type of `a`, row-major and floating-point.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of first legs. Read through stride 0 on any axis it stretches.
        b: Tensor of second legs. At a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding `sqrt(a*a + b*b)`.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A,
        B,
        op=_hypot_op[dtype, _],
        gpu=gpu,
        name="hypot",
    ](a, b)


def copysign[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == B.dtype
    and is_row_major[A]
    and is_row_major[B]
    and A.dtype.is_floating_point()
):
    """Elementwise `copysign` at two broadcastable shapes.

    Parameters:
        A: The `TensorLike` type of `a`, row-major and floating-point.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor supplying the magnitudes. Read through stride 0 on any axis it
            stretches.
        b: Tensor supplying the signs. At a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding `abs(a)` carrying the sign of `b`.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A,
        B,
        op=_copysign_op[dtype, _],
        gpu=gpu,
        name="copysign",
    ](a, b)


def remainder[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == B.dtype
    and is_row_major[A]
    and is_row_major[B]
    and A.dtype.is_floating_point()
):
    """Elementwise IEEE remainder at two broadcastable shapes.

    Parameters:
        A: The `TensorLike` type of `a`, row-major and floating-point.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of dividends. Read through stride 0 on any axis it stretches.
        b: Tensor of divisors. At a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding the IEEE remainder of `a` and `b`.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A,
        B,
        op=_remainder_op[dtype, _],
        gpu=gpu,
        name="remainder",
    ](a, b)


def maximum[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """Elementwise larger of the two, at two broadcastable shapes.

    `numpy.maximum`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: First tensor compared. Read through stride 0 on any axis it
            stretches.
        b: Second tensor compared. At a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding the larger of each pair, NaN where either
        is NaN.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A,
        B,
        op=_maximum_op[dtype, _],
        gpu=gpu,
        name="maximum",
    ](a, b)


def minimum[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """Elementwise smaller of the two, at two broadcastable shapes.

    `numpy.minimum`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: First tensor compared. Read through stride 0 on any axis it
            stretches.
        b: Second tensor compared. At a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding the smaller of each pair, NaN where either
        is NaN.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A,
        B,
        op=_minimum_op[dtype, _],
        gpu=gpu,
        name="minimum",
    ](a, b)


def fmax[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """Elementwise larger of the two ignoring NaN, at two broadcastable
    shapes. `numpy.fmax`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: First tensor compared. Read through stride 0 on any axis it
            stretches.
        b: Second tensor compared. At a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding the larger of each pair, the non-NaN one
        where one is NaN.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A,
        B,
        op=_fmax_op[dtype, _],
        gpu=gpu,
        name="fmax",
    ](a, b)


def fmin[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """Elementwise smaller of the two ignoring NaN, at two broadcastable
    shapes. `numpy.fmin`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: First tensor compared. Read through stride 0 on any axis it
            stretches.
        b: Second tensor compared. At a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding the smaller of each pair, the non-NaN one
        where one is NaN.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A,
        B,
        op=_fmin_op[dtype, _],
        gpu=gpu,
        name="fmin",
    ](a, b)


def logaddexp[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == B.dtype
    and is_row_major[A]
    and is_row_major[B]
    and A.dtype.is_floating_point()
):
    """`logaddexp` at two broadcastable shapes.

    Parameters:
        A: The `TensorLike` type of `a`, row-major and floating-point.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of first log-magnitudes. Read through stride 0 on any axis it stretches.
        b: Tensor of second log-magnitudes. At a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding `log(exp(a) + exp(b))`.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A,
        B,
        op=_logaddexp_op[dtype, _],
        gpu=gpu,
        name="logaddexp",
    ](a, b)


def logaddexp2[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == B.dtype
    and is_row_major[A]
    and is_row_major[B]
    and A.dtype.is_floating_point()
):
    """`logaddexp2` at two broadcastable shapes.

    Parameters:
        A: The `TensorLike` type of `a`, row-major and floating-point.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of first base-2 log-magnitudes. Read through stride 0 on any axis it stretches.
        b: Tensor of second base-2 log-magnitudes. At a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding `log2(2**a + 2**b)`.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A,
        B,
        op=_logaddexp2_op[dtype, _],
        gpu=gpu,
        name="logaddexp2",
    ](a, b)


def heaviside[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == B.dtype
    and is_row_major[A]
    and is_row_major[B]
    and A.dtype.is_floating_point()
):
    """`heaviside` at two broadcastable shapes.

    Parameters:
        A: The `TensorLike` type of `a`, row-major and floating-point.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor of inputs to the step. Read through stride 0 on any axis it stretches.
        b: Tensor of the values taken where `a == 0`. At a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding the step of `a` with `b` at zero.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A,
        B,
        op=_heaviside_op[dtype, _],
        gpu=gpu,
        name="heaviside",
    ](a, b)
