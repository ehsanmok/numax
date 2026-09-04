"""NumPy-named elementwise mathematics over `Tensor`.

**This module is tier 2**, on the same terms as `numax.core.logic`: each routine
walks a host copy of the elements and returns a new tensor. The mathematics
is tier 1 -- fixed work per element, no branching -- and
`numax.core.tensor.map`/`map_to` are the GPU-launchable primitives that express
it, so a caller holding a concrete shape and a hot loop should reach for
those directly. These names exist so that a NumPy caller finds them.

`Plain`-only. `FloatLike` deliberately stays small (`numax/core/numeric.mojo`
lists what it carries and why), and the functions here are `std.math` calls
over raw `SIMD` rather than trait growth: nothing in numax needs `arctanh`
at `Dual`, and adding twenty methods to the trait to reach parity on names
would make every one of the seven conformers implement them.

`min` and `max` are Mojo builtins, so the elementwise forms here are
`minimum` and `maximum` -- NumPy's own names for the two-argument form, so
nothing is lost. `nextafter` is absent from `std.math` under the pinned
toolchain and is therefore not provided.

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
    exp as _std_exp,
    exp2 as _std_exp2,
    expm1 as _std_expm1,
    floor as _std_floor,
    hypot as _std_hypot,
    log as _std_log,
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

from .array import Tensor, _product


def _unary[
    dtype: DType,
    *dims: Int,
    op: def(Scalar[dtype]) thin -> Scalar[dtype],
](a: Tensor[dtype, *dims]) raises -> Tensor[dtype, *dims]:
    comptime n = _product[*dims]()
    var values = a.to_host()
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = op(values[i])
    return Tensor[dtype, *dims](a.context(), out^)


def _binary[
    dtype: DType,
    *dims: Int,
    op: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[dtype],
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
]:
    comptime n = _product[*dims]()
    var a_values = a.to_host()
    var b_values = b.to_host()
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = op(a_values[i], b_values[i])
    return Tensor[dtype, *dims](a.context(), out^)


def _exp_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_exp(x)


def exp[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise `e**x`."""
    return _unary[dtype, *dims, op=_exp_op[dtype]](a)


def _exp2_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_exp2(x)


def exp2[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise `2**x`."""
    return _unary[dtype, *dims, op=_exp2_op[dtype]](a)


def _expm1_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_expm1(x)


def expm1[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise `e**x - 1`, accurate for small `x`."""
    return _unary[dtype, *dims, op=_expm1_op[dtype]](a)


def _log_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_log(x)


def log[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise natural logarithm."""
    return _unary[dtype, *dims, op=_log_op[dtype]](a)


def _log2_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_log2(x)


def log2[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise base-2 logarithm."""
    return _unary[dtype, *dims, op=_log2_op[dtype]](a)


def _log10_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_log10(x)


def log10[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise base-10 logarithm."""
    return _unary[dtype, *dims, op=_log10_op[dtype]](a)


def _log1p_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_log1p(x)


def log1p[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise `log(1 + x)`, accurate for small `x`."""
    return _unary[dtype, *dims, op=_log1p_op[dtype]](a)


def _sqrt_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_sqrt(x)


def sqrt[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise square root."""
    return _unary[dtype, *dims, op=_sqrt_op[dtype]](a)


def _rsqrt_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_rsqrt(x)


def rsqrt[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise `1 / sqrt(x)`."""
    return _unary[dtype, *dims, op=_rsqrt_op[dtype]](a)


def _cbrt_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_cbrt(x)


def cbrt[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise cube root."""
    return _unary[dtype, *dims, op=_cbrt_op[dtype]](a)


def _sin_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_sin(x)


def sin[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise sine."""
    return _unary[dtype, *dims, op=_sin_op[dtype]](a)


def _cos_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_cos(x)


def cos[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise cosine."""
    return _unary[dtype, *dims, op=_cos_op[dtype]](a)


def _tan_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_tan(x)


def tan[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise tangent."""
    return _unary[dtype, *dims, op=_tan_op[dtype]](a)


def _arcsin_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_asin(x)


def arcsin[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise inverse sine. `numpy.arcsin`."""
    return _unary[dtype, *dims, op=_arcsin_op[dtype]](a)


def _arccos_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_acos(x)


def arccos[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise inverse cosine. `numpy.arccos`."""
    return _unary[dtype, *dims, op=_arccos_op[dtype]](a)


def _arctan_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_atan(x)


def arctan[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise inverse tangent. `numpy.arctan`."""
    return _unary[dtype, *dims, op=_arctan_op[dtype]](a)


def _sinh_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_sinh(x)


def sinh[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise hyperbolic sine."""
    return _unary[dtype, *dims, op=_sinh_op[dtype]](a)


def _cosh_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_cosh(x)


def cosh[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise hyperbolic cosine."""
    return _unary[dtype, *dims, op=_cosh_op[dtype]](a)


def _tanh_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_tanh(x)


def tanh[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise hyperbolic tangent."""
    return _unary[dtype, *dims, op=_tanh_op[dtype]](a)


def _arcsinh_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_asinh(x)


def arcsinh[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise inverse hyperbolic sine. `numpy.arcsinh`."""
    return _unary[dtype, *dims, op=_arcsinh_op[dtype]](a)


def _arccosh_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_acosh(x)


def arccosh[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise inverse hyperbolic cosine. `numpy.arccosh`."""
    return _unary[dtype, *dims, op=_arccosh_op[dtype]](a)


def _arctanh_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_atanh(x)


def arctanh[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise inverse hyperbolic tangent. `numpy.arctanh`."""
    return _unary[dtype, *dims, op=_arctanh_op[dtype]](a)


def _floor_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_floor(x)


def floor[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise largest integer `<= x`."""
    return _unary[dtype, *dims, op=_floor_op[dtype]](a)


def _ceil_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_ceil(x)


def ceil[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise smallest integer `>= x`."""
    return _unary[dtype, *dims, op=_ceil_op[dtype]](a)


def _trunc_op[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _std_trunc(x)


def trunc[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise `x` rounded toward zero."""
    return _unary[dtype, *dims, op=_trunc_op[dtype]](a)


def _round_op[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    return _std_round(x)


def round[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise `x` rounded to nearest, ties to even."""
    return _unary[dtype, *dims, op=_round_op[dtype]](a)


def _arctan2_op[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    return _std_atan2(a, b)


def arctan2[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise `atan2(a, b)`, quadrant-aware. `numpy.arctan2`."""
    return _binary[dtype, *dims, op=_arctan2_op[dtype]](a, b)


def _hypot_op[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    return _std_hypot(a, b)


def hypot[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise `sqrt(a*a + b*b)` without intermediate overflow."""
    return _binary[dtype, *dims, op=_hypot_op[dtype]](a, b)


def _copysign_op[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    return _std_copysign(a, b)


def copysign[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise magnitude of `a` with the sign of `b`."""
    return _binary[dtype, *dims, op=_copysign_op[dtype]](a, b)


def _remainder_op[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    return _std_remainder(a, b)


def remainder[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """Elementwise IEEE remainder of `a` and `b`."""
    return _binary[dtype, *dims, op=_remainder_op[dtype]](a, b)


def _abs_op[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    # `x.__abs__()` rather than `abs(x)`: the module-level `abs` below hides
    # the builtin throughout this file.
    return x.__abs__()


def abs[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[dtype, *dims]:
    """Elementwise magnitude. `numpy.abs`.

    Defining `abs` here hides Mojo's builtin `abs` for the rest of this
    file, which is why the private op above spells it `x.__abs__()`. A
    caller who imports this name pays the same price in their own file,
    exactly as `from numpy import abs` does in Python.
    """
    return _unary[dtype, *dims, op=_abs_op[dtype]](a)


def _maximum_op[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return max(a, b)


def maximum[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
]:
    """Elementwise larger of the two. `numpy.maximum`."""
    return _binary[dtype, *dims, op=_maximum_op[dtype]](a, b)


def _minimum_op[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return min(a, b)


def minimum[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
]:
    """Elementwise smaller of the two. `numpy.minimum`."""
    return _binary[dtype, *dims, op=_minimum_op[dtype]](a, b)


def clip[
    dtype: DType, *dims: Int
](
    a: Tensor[dtype, *dims], lo: Scalar[dtype], hi: Scalar[dtype]
) raises -> Tensor[dtype, *dims]:
    """Every element confined to `[lo, hi]`. `numpy.clip`.

    Bounds are runtime values, so this walks directly rather than composing
    `maximum`/`minimum` against two full tensors of the bounds.
    """
    comptime n = _product[*dims]()
    var values = a.to_host()
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = min(max(values[i], lo), hi)
    return Tensor[dtype, *dims](a.context(), out^)


def diff[
    dtype: DType, n: Int
](a: Tensor[dtype, n]) raises -> Tensor[dtype, n - 1] where n >= 1:
    """First differences, `out[i] = a[i+1] - a[i]`. `numpy.diff`.

    Rank-1, and one element shorter than its input -- which is why the
    output length is a compile-time function of `n` rather than a runtime
    value.
    """
    var values = a.to_host()
    var out = List[Scalar[dtype]](length=n - 1, fill=0)
    for i in range(n - 1):
        out[i] = values[i + 1] - values[i]
    return Tensor[dtype, n - 1](a.context(), out^)


def gradient[
    dtype: DType, n: Int
](a: Tensor[dtype, n], spacing: Scalar[dtype] = 1) raises -> Tensor[
    dtype, n
] where (n >= 2):
    """Central differences interior, one-sided at the ends. `numpy.gradient`.

    Second-order accurate in the interior and first-order at the two
    endpoints, matching NumPy's default for a uniform grid.
    """
    var values = a.to_host()
    var out = List[Scalar[dtype]](length=n, fill=0)
    out[0] = (values[1] - values[0]) / spacing
    out[n - 1] = (values[n - 1] - values[n - 2]) / spacing
    for i in range(1, n - 1):
        out[i] = (values[i + 1] - values[i - 1]) / (spacing + spacing)
    return Tensor[dtype, n](a.context(), out^)
