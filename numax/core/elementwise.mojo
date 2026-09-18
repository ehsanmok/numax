"""NumPy-named elementwise mathematics over `Tensor`.

**Tier 2 in shape, both targets in fact**, on the same terms as
`numax.core.rowwise`: these are `Plain`-only -- `TileTensor` holds raw
`dtype` lanes -- but they are not host-only. Every routine here runs through
`numax.core._drive`, which launches one capturing body on the target its
`gpu: Bool` parameter names: `max.algorithm.elementwise` on the device, the
same threaded walk on a large host tensor, a serial SIMD loop on a small
one. The mathematics is tier 1 -- fixed work per element, no branching --
and `numax.core.tensor.map`/`map_to` remain the primitives a caller holding
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

`tanh` is the one name this module shares with `numax.special.activations`,
which has the `FloatLike` scalar of the same name. The root package exports
the activation, because that is the one a kernel calls; the tensor form here
is `numax.core.tanh`.
"""

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
    cosh as _std_cosh,
    exp2 as _std_exp2,
    expm1 as _std_expm1,
    floor as _std_floor,
    hypot as _std_hypot,
    isnan as _std_isnan,
    log10 as _std_log10,
    log1p as _std_log1p,
    log2 as _std_log2,
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

# `exp` and `log` at float64 are numax's own one-ulp versions; `std.math`'s
# are `1e5` and `9e6` ulp off there (`numax/core/libm.mojo`).
from .libm import exp as _std_exp
from .libm import log as _std_log
from .array import Dynamic, Static, Tensor
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
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise `e**x`."""
    return unary[dtype, LayoutType, op=_exp_op[dtype, _], gpu=gpu, name="exp"](
        a
    )


def _exp2_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_exp2(x)


def exp2[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise `2**x`."""
    return unary[
        dtype, LayoutType, op=_exp2_op[dtype, _], gpu=gpu, name="exp2"
    ](a)


def _expm1_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_expm1(x)


def expm1[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise `e**x - 1`, accurate for small `x`."""
    return unary[
        dtype, LayoutType, op=_expm1_op[dtype, _], gpu=gpu, name="expm1"
    ](a)


def _log_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_log(x)


def log[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise natural logarithm."""
    return unary[dtype, LayoutType, op=_log_op[dtype, _], gpu=gpu, name="log"](
        a
    )


def _log2_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_log2(x)


def log2[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise base-2 logarithm."""
    return unary[
        dtype, LayoutType, op=_log2_op[dtype, _], gpu=gpu, name="log2"
    ](a)


def _log10_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_log10(x)


def log10[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise base-10 logarithm."""
    return unary[
        dtype, LayoutType, op=_log10_op[dtype, _], gpu=gpu, name="log10"
    ](a)


def _log1p_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_log1p(x)


def log1p[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise `log(1 + x)`, accurate for small `x`."""
    return unary[
        dtype, LayoutType, op=_log1p_op[dtype, _], gpu=gpu, name="log1p"
    ](a)


def _sqrt_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_sqrt(x)


def sqrt[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise square root."""
    return unary[
        dtype, LayoutType, op=_sqrt_op[dtype, _], gpu=gpu, name="sqrt"
    ](a)


def _rsqrt_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_rsqrt(x)


def rsqrt[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise `1 / sqrt(x)`."""
    return unary[
        dtype, LayoutType, op=_rsqrt_op[dtype, _], gpu=gpu, name="rsqrt"
    ](a)


def _cbrt_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_cbrt(x)


def cbrt[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise cube root."""
    return unary[
        dtype, LayoutType, op=_cbrt_op[dtype, _], gpu=gpu, name="cbrt"
    ](a)


def _sin_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_sin(x)


def sin[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise sine."""
    return unary[dtype, LayoutType, op=_sin_op[dtype, _], gpu=gpu, name="sin"](
        a
    )


def _cos_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_cos(x)


def cos[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise cosine."""
    return unary[dtype, LayoutType, op=_cos_op[dtype, _], gpu=gpu, name="cos"](
        a
    )


def _tan_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_tan(x)


def tan[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise tangent."""
    return unary[dtype, LayoutType, op=_tan_op[dtype, _], gpu=gpu, name="tan"](
        a
    )


def _arcsin_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_asin(x)


def arcsin[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise inverse sine. `numpy.arcsin`."""
    return unary[
        dtype, LayoutType, op=_arcsin_op[dtype, _], gpu=gpu, name="arcsin"
    ](a)


def _arccos_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_acos(x)


def arccos[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise inverse cosine. `numpy.arccos`."""
    return unary[
        dtype, LayoutType, op=_arccos_op[dtype, _], gpu=gpu, name="arccos"
    ](a)


def _arctan_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_atan(x)


def arctan[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise inverse tangent. `numpy.arctan`."""
    return unary[
        dtype, LayoutType, op=_arctan_op[dtype, _], gpu=gpu, name="arctan"
    ](a)


def _sinh_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_sinh(x)


def sinh[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise hyperbolic sine."""
    return unary[
        dtype, LayoutType, op=_sinh_op[dtype, _], gpu=gpu, name="sinh"
    ](a)


def _cosh_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_cosh(x)


def cosh[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise hyperbolic cosine."""
    return unary[
        dtype, LayoutType, op=_cosh_op[dtype, _], gpu=gpu, name="cosh"
    ](a)


def _tanh_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_tanh(x)


def tanh[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise hyperbolic tangent."""
    return unary[
        dtype, LayoutType, op=_tanh_op[dtype, _], gpu=gpu, name="tanh"
    ](a)


def _arcsinh_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_asinh(x)


def arcsinh[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise inverse hyperbolic sine. `numpy.arcsinh`."""
    return unary[
        dtype, LayoutType, op=_arcsinh_op[dtype, _], gpu=gpu, name="arcsinh"
    ](a)


def _arccosh_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_acosh(x)


def arccosh[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise inverse hyperbolic cosine. `numpy.arccosh`."""
    return unary[
        dtype, LayoutType, op=_arccosh_op[dtype, _], gpu=gpu, name="arccosh"
    ](a)


def _arctanh_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_atanh(x)


def arctanh[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise inverse hyperbolic tangent. `numpy.arctanh`."""
    return unary[
        dtype, LayoutType, op=_arctanh_op[dtype, _], gpu=gpu, name="arctanh"
    ](a)


def _floor_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_floor(x)


def floor[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise largest integer `<= x`."""
    return unary[
        dtype, LayoutType, op=_floor_op[dtype, _], gpu=gpu, name="floor"
    ](a)


def _ceil_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_ceil(x)


def ceil[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise smallest integer `>= x`."""
    return unary[
        dtype, LayoutType, op=_ceil_op[dtype, _], gpu=gpu, name="ceil"
    ](a)


def _trunc_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return _std_trunc(x)


def trunc[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise `x` rounded toward zero."""
    return unary[
        dtype, LayoutType, op=_trunc_op[dtype, _], gpu=gpu, name="trunc"
    ](a)


def _round_op[dtype: DType, w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return _std_round(x)


def round[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise `x` rounded to nearest, ties to even."""
    return unary[
        dtype, LayoutType, op=_round_op[dtype, _], gpu=gpu, name="round"
    ](a)


def rint[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise `x` rounded to nearest, ties to even. `numpy.rint`.

    The same operation as `round`, under the name NumPy gives it when no
    decimal count is involved; both are half-to-even, so `rint` exists to
    be found rather than to do anything `round` does not.
    """
    return unary[
        dtype, LayoutType, op=_round_op[dtype, _], gpu=gpu, name="rint"
    ](a)


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


def sign[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise `-1`, `0` or `1` by the sign of `x`. `numpy.sign`.

    Zero maps to zero and NaN to NaN, which is why this is not
    `copysign(1, x)` -- that answers `-1` for `-0.0` and `1` for NaN.
    """
    return unary[
        dtype, LayoutType, op=_sign_op[dtype, _], gpu=gpu, name="sign"
    ](a)


def _square_op[dtype: DType, w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return x * x


def square[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[dtype, LayoutType]:
    """Elementwise `x * x`. `numpy.square`.

    One multiply rather than `power(a, 2)`'s general exponentiation, and
    exact where the general form is not.
    """
    return unary[
        dtype, LayoutType, op=_square_op[dtype, _], gpu=gpu, name="square"
    ](a)


def _reciprocal_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return SIMD[dtype, w](1) / x


def reciprocal[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise `1 / x`. `numpy.reciprocal`.

    A zero gives an infinity rather than raising, as NumPy's does with the
    error state at its default.
    """
    return unary[
        dtype,
        LayoutType,
        op=_reciprocal_op[dtype, _],
        gpu=gpu,
        name="reciprocal",
    ](a)


def _degrees_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return x * SIMD[dtype, w](57.295779513082320876798154814105)


def degrees[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise radians to degrees, `x * 180 / pi`. `numpy.degrees`."""
    return unary[
        dtype, LayoutType, op=_degrees_op[dtype, _], gpu=gpu, name="degrees"
    ](a)


def _radians_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_floating_point():
    return x * SIMD[dtype, w](0.017453292519943295769236907684886)


def radians[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise degrees to radians, `x * pi / 180`. `numpy.radians`."""
    return unary[
        dtype, LayoutType, op=_radians_op[dtype, _], gpu=gpu, name="radians"
    ](a)


def _arctan2_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return _std_atan2(a, b)


def arctan2[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise `atan2(a, b)`, quadrant-aware. `numpy.arctan2`."""
    return binary[
        dtype, LayoutType, op=_arctan2_op[dtype, _], gpu=gpu, name="arctan2"
    ](a, b)


def _hypot_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return _std_hypot(a, b)


def hypot[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise `sqrt(a*a + b*b)` without intermediate overflow."""
    return binary[
        dtype, LayoutType, op=_hypot_op[dtype, _], gpu=gpu, name="hypot"
    ](a, b)


def _copysign_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return _std_copysign(a, b)


def copysign[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise magnitude of `a` with the sign of `b`."""
    return binary[
        dtype, LayoutType, op=_copysign_op[dtype, _], gpu=gpu, name="copysign"
    ](a, b)


def _remainder_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return _std_remainder(a, b)


def remainder[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Elementwise IEEE remainder of `a` and `b`."""
    return binary[
        dtype, LayoutType, op=_remainder_op[dtype, _], gpu=gpu, name="remainder"
    ](a, b)


def _abs_op[dtype: DType, w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    # `x.__abs__()` rather than `abs(x)`: the module-level `abs` below hides
    # the builtin throughout this file.
    return x.__abs__()


def abs[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[dtype, LayoutType]:
    """Elementwise magnitude. `numpy.abs`.

    Defining `abs` here hides Mojo's builtin `abs` for the rest of this
    file, which is why the private op above spells it `x.__abs__()`. A
    caller who imports this name pays the same price in their own file,
    exactly as `from numpy import abs` does in Python.
    """
    return unary[dtype, LayoutType, op=_abs_op[dtype, _], gpu=gpu, name="abs"](
        a
    )


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
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    """Elementwise larger of the two, NaN-propagating. `numpy.maximum`.

    A NaN in either operand gives NaN, which is NumPy's rule and *not* the
    hardware's: `max` on a `SIMD` is IEEE `maxNum` and quietly returns the
    other operand. `fmax` is the name for that behavior.
    """
    return binary[
        dtype, LayoutType, op=_maximum_op[dtype, _], gpu=gpu, name="maximum"
    ](a, b)


def _minimum_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    comptime if dtype.is_floating_point():
        return _std_isnan(a).select(a, _std_isnan(b).select(b, min(a, b)))
    else:
        return min(a, b)


def minimum[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    """Elementwise smaller of the two, NaN-propagating. `numpy.minimum`.

    A NaN in either operand gives NaN, as `maximum` records; `fmin` is the
    name that skips it.
    """
    return binary[
        dtype, LayoutType, op=_minimum_op[dtype, _], gpu=gpu, name="minimum"
    ](a, b)


def _fmax_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return max(a, b)


def fmax[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    """Elementwise larger of the two, ignoring NaN. `numpy.fmax`.

    `fmax(nan, x)` is `x`, where `maximum(nan, x)` is NaN. This is the one
    of the pair that maps straight onto the hardware instruction, so it is
    also the cheaper of the two on a float dtype.
    """
    return binary[
        dtype, LayoutType, op=_fmax_op[dtype, _], gpu=gpu, name="fmax"
    ](a, b)


def _fmin_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return min(a, b)


def fmin[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    """Elementwise smaller of the two, ignoring NaN. `numpy.fmin`."""
    return binary[
        dtype, LayoutType, op=_fmin_op[dtype, _], gpu=gpu, name="fmin"
    ](a, b)


def clip[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](
    a: Tensor[dtype, LayoutType], lo: Scalar[dtype], hi: Scalar[dtype]
) raises -> Tensor[dtype, LayoutType]:
    """Every element confined to `[lo, hi]`. `numpy.clip`.

    Two run-time bounds rather than one, so this carries its own body
    instead of going through `_drive.binary_scalar`; both are captured by
    value and splatted to the launch width.
    """
    if not _check_device[gpu=gpu](a):
        _notice[gpu]("clip")
        var n = a.size()
        var values = a.to_host()
        var walked = List[Scalar[dtype]](length=n, fill=0)
        for i in range(n):
            walked[i] = min(max(values[i], lo), hi)
        return Tensor[dtype, LayoutType](a.context(), a.layout, walked^)

    var ctx = a.context()
    var out = Tensor[dtype, LayoutType]._uninitialized(ctx, a.layout)
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
    dtype: DType, n: Int, gpu: Bool = False
](a: Static[dtype, n]) raises -> Static[dtype, n - 1] where n >= 1:
    """First differences, `out[i] = a[i+1] - a[i]`. `numpy.diff`.

    Rank-1, and one element shorter than its input -- which is why the
    output length is a compile-time function of `n` rather than a runtime
    value. One launch: the shifted read is `Coord(i + 1)` against the same
    flat view, which stays in bounds because the domain is `n - 1` wide.
    """
    if not _check_device[gpu=gpu](a):
        _notice[gpu]("diff")
        var values = a.to_host()
        var walked = List[Scalar[dtype]](length=n - 1, fill=0)
        for i in range(n - 1):
            walked[i] = values[i + 1] - values[i]
        return Static[dtype, n - 1](a.context(), walked^)

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
    dtype: DType, n: Int, gpu: Bool = False
](a: Static[dtype, n], spacing: Scalar[dtype] = 1) raises -> Static[
    dtype, n
] where (n >= 2):
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
    """
    if not _check_device[gpu=gpu](a):
        _notice[gpu]("gradient")
        var values = a.to_host()
        var walked = List[Scalar[dtype]](length=n, fill=0)
        walked[0] = (values[1] - values[0]) / spacing
        walked[n - 1] = (values[n - 1] - values[n - 2]) / spacing
        for i in range(1, n - 1):
            walked[i] = (values[i + 1] - values[i - 1]) / (spacing + spacing)
        return Static[dtype, n](a.context(), walked^)

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


# The broadcasting forms, matching `numax.core.ops`: same operation, two
# shapes NumPy would broadcast rather than one shape twice, and a `Dynamic`
# result because the broadcast extents are run-time values.


def arctan2[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, _BroadcastRank[ALayout, BLayout]
] where dtype.is_floating_point():
    """Elementwise `atan2(a, b)` at two broadcastable shapes."""
    return broadcast_binary[
        dtype,
        ALayout,
        BLayout,
        op=_arctan2_op[dtype, _],
        gpu=gpu,
        name="arctan2",
    ](a, b)


def hypot[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, _BroadcastRank[ALayout, BLayout]
] where dtype.is_floating_point():
    """Elementwise `sqrt(a*a + b*b)` at two broadcastable shapes."""
    return broadcast_binary[
        dtype,
        ALayout,
        BLayout,
        op=_hypot_op[dtype, _],
        gpu=gpu,
        name="hypot",
    ](a, b)


def copysign[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, _BroadcastRank[ALayout, BLayout]
] where dtype.is_floating_point():
    """Elementwise `copysign` at two broadcastable shapes."""
    return broadcast_binary[
        dtype,
        ALayout,
        BLayout,
        op=_copysign_op[dtype, _],
        gpu=gpu,
        name="copysign",
    ](a, b)


def remainder[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, _BroadcastRank[ALayout, BLayout]
] where dtype.is_floating_point():
    """Elementwise IEEE remainder at two broadcastable shapes."""
    return broadcast_binary[
        dtype,
        ALayout,
        BLayout,
        op=_remainder_op[dtype, _],
        gpu=gpu,
        name="remainder",
    ](a, b)


def maximum[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, _BroadcastRank[ALayout, BLayout]
]:
    """Elementwise larger of the two, at two broadcastable shapes.

    `numpy.maximum`."""
    return broadcast_binary[
        dtype,
        ALayout,
        BLayout,
        op=_maximum_op[dtype, _],
        gpu=gpu,
        name="maximum",
    ](a, b)


def minimum[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, _BroadcastRank[ALayout, BLayout]
]:
    """Elementwise smaller of the two, at two broadcastable shapes.

    `numpy.minimum`."""
    return broadcast_binary[
        dtype,
        ALayout,
        BLayout,
        op=_minimum_op[dtype, _],
        gpu=gpu,
        name="minimum",
    ](a, b)


def fmax[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, _BroadcastRank[ALayout, BLayout]
]:
    """Elementwise larger of the two ignoring NaN, at two broadcastable
    shapes. `numpy.fmax`."""
    return broadcast_binary[
        dtype,
        ALayout,
        BLayout,
        op=_fmax_op[dtype, _],
        gpu=gpu,
        name="fmax",
    ](a, b)


def fmin[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, _BroadcastRank[ALayout, BLayout]
]:
    """Elementwise smaller of the two ignoring NaN, at two broadcastable
    shapes. `numpy.fmin`."""
    return broadcast_binary[
        dtype,
        ALayout,
        BLayout,
        op=_fmin_op[dtype, _],
        gpu=gpu,
        name="fmin",
    ](a, b)
