"""Elementwise arithmetic over `Tensor`, as functions and as operators.

**This module is tier 2**, on the same terms as `numax.logic` and
`numax.elementwise`: a host walk returning a new tensor. `numax.tensor.map`
with `add_step`/`mul_step` is the tier-1, GPU-launchable form.

Every routine has a tensor-tensor and a tensor-scalar overload, which is
what `a * 2.0` needs. Shapes must match exactly -- there is no broadcasting
in numax yet, so `(2, 3)` and `(3,)` do not combine.

The operators on `Tensor` itself (`+`, `-`, `*`, `/`, unary `-`) forward
here, so `a + b` and `add(a, b)` are the same call. `numpy.power` is
`power` rather than `**`: `__pow__` on a tensor would have to choose
between an elementwise power and a matrix power, and NumPy's own answer to
that (`**` is elementwise, `numpy.linalg.matrix_power` is separate) is
worth stating explicitly rather than implying.
"""

from .array import Tensor, _product


def _zip[
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


def _zip_scalar[
    dtype: DType,
    *dims: Int,
    op: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[dtype],
](a: Tensor[dtype, *dims], b: Scalar[dtype]) raises -> Tensor[dtype, *dims]:
    comptime n = _product[*dims]()
    var a_values = a.to_host()
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = op(a_values[i], b)
    return Tensor[dtype, *dims](a.context(), out^)


def _add_op[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return a + b


def _sub_op[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return a - b


def _mul_op[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return a * b


def _div_op[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return a / b


def _floordiv_op[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return a // b


def _mod_op[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return a % b


def _pow_op[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    return a**b


def add[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
]:
    """`a + b`, elementwise. `numpy.add`."""
    return _zip[dtype, *dims, op=_add_op[dtype]](a, b)


def add[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Scalar[dtype]) raises -> Tensor[dtype, *dims]:
    """`a + b` with a scalar `b`."""
    return _zip_scalar[dtype, *dims, op=_add_op[dtype]](a, b)


def subtract[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
]:
    """`a - b`, elementwise. `numpy.subtract`."""
    return _zip[dtype, *dims, op=_sub_op[dtype]](a, b)


def subtract[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Scalar[dtype]) raises -> Tensor[dtype, *dims]:
    """`a - b` with a scalar `b`."""
    return _zip_scalar[dtype, *dims, op=_sub_op[dtype]](a, b)


def multiply[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
]:
    """`a * b`, elementwise. `numpy.multiply`."""
    return _zip[dtype, *dims, op=_mul_op[dtype]](a, b)


def multiply[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Scalar[dtype]) raises -> Tensor[dtype, *dims]:
    """`a * b` with a scalar `b`."""
    return _zip_scalar[dtype, *dims, op=_mul_op[dtype]](a, b)


def divide[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
]:
    """`a / b`, elementwise. `numpy.divide`."""
    return _zip[dtype, *dims, op=_div_op[dtype]](a, b)


def divide[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Scalar[dtype]) raises -> Tensor[dtype, *dims]:
    """`a / b` with a scalar `b`."""
    return _zip_scalar[dtype, *dims, op=_div_op[dtype]](a, b)


def floor_divide[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
]:
    """`a // b`, elementwise. `numpy.floor_divide`."""
    return _zip[dtype, *dims, op=_floordiv_op[dtype]](a, b)


def mod[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
]:
    """`a % b`, elementwise. `numpy.mod`."""
    return _zip[dtype, *dims, op=_mod_op[dtype]](a, b)


def power[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """`a ** b`, elementwise. `numpy.power`."""
    return _zip[dtype, *dims, op=_pow_op[dtype]](a, b)


def power[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Scalar[dtype]) raises -> Tensor[
    dtype, *dims
] where dtype.is_floating_point():
    """`a ** b` with a scalar exponent."""
    return _zip_scalar[dtype, *dims, op=_pow_op[dtype]](a, b)


def negative[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[dtype, *dims]:
    """`-a`, elementwise. `numpy.negative`."""
    comptime n = _product[*dims]()
    var values = a.to_host()
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = -values[i]
    return Tensor[dtype, *dims](a.context(), out^)


def astype[
    target: DType, dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[target, *dims]:
    """`a` converted to `target`, elementwise. `numpy.astype`.

    Explicit, because numax has no dtype promotion: a binary operation
    requires both sides to already share a dtype, and this is how a caller
    makes that true. NuMojo made the same call.
    """
    comptime n = _product[*dims]()
    var values = a.to_host()
    var out = List[Scalar[target]](length=n, fill=0)
    for i in range(n):
        out[i] = values[i].cast[target]()
    return Tensor[target, *dims](a.context(), out^)


def invert[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, *dims
] where dtype.is_integral():
    """Bitwise NOT, elementwise. `numpy.invert`.

    Integral dtypes only. The boolean form is
    `numax.logic.logical_not`, which is a different operation on a
    different type rather than the same one spelled twice.
    """
    comptime n = _product[*dims]()
    var values = a.to_host()
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = ~values[i]
    return Tensor[dtype, *dims](a.context(), out^)
