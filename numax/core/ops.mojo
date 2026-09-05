"""Elementwise arithmetic over `Tensor`, as functions and as operators.

**This module is tier 2**, on the same terms as `numax.core.logic` and
`numax.core.elementwise`: a host walk returning a new tensor. `numax.core.tensor.map`
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

from layout.tile_layout import TensorLayout

from .array import Tensor


def _zip[
    dtype: DType,
    LayoutType: TensorLayout,
    op: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[dtype],
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    var n = a.size()
    var a_values = a.to_host()
    var b_values = b.to_host()
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = op(a_values[i], b_values[i])
    return Tensor[dtype, LayoutType](a.context(), a.layout, out^)


def _zip_scalar[
    dtype: DType,
    LayoutType: TensorLayout,
    op: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[dtype],
](a: Tensor[dtype, LayoutType], b: Scalar[dtype]) raises -> Tensor[
    dtype, LayoutType
]:
    var n = a.size()
    var a_values = a.to_host()
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = op(a_values[i], b)
    return Tensor[dtype, LayoutType](a.context(), a.layout, out^)


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
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a + b`, elementwise. `numpy.add`."""
    return _zip[dtype, LayoutType, op=_add_op[dtype]](a, b)


def add[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Scalar[dtype]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a + b` with a scalar `b`."""
    return _zip_scalar[dtype, LayoutType, op=_add_op[dtype]](a, b)


def subtract[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a - b`, elementwise. `numpy.subtract`."""
    return _zip[dtype, LayoutType, op=_sub_op[dtype]](a, b)


def subtract[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Scalar[dtype]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a - b` with a scalar `b`."""
    return _zip_scalar[dtype, LayoutType, op=_sub_op[dtype]](a, b)


def multiply[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a * b`, elementwise. `numpy.multiply`."""
    return _zip[dtype, LayoutType, op=_mul_op[dtype]](a, b)


def multiply[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Scalar[dtype]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a * b` with a scalar `b`."""
    return _zip_scalar[dtype, LayoutType, op=_mul_op[dtype]](a, b)


def divide[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a / b`, elementwise. `numpy.divide`."""
    return _zip[dtype, LayoutType, op=_div_op[dtype]](a, b)


def divide[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Scalar[dtype]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a / b` with a scalar `b`."""
    return _zip_scalar[dtype, LayoutType, op=_div_op[dtype]](a, b)


def floor_divide[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a // b`, elementwise. `numpy.floor_divide`."""
    return _zip[dtype, LayoutType, op=_floordiv_op[dtype]](a, b)


def mod[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a % b`, elementwise. `numpy.mod`."""
    return _zip[dtype, LayoutType, op=_mod_op[dtype]](a, b)


def power[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """`a ** b`, elementwise. `numpy.power`."""
    return _zip[dtype, LayoutType, op=_pow_op[dtype]](a, b)


def power[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Scalar[dtype]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """`a ** b` with a scalar exponent."""
    return _zip_scalar[dtype, LayoutType, op=_pow_op[dtype]](a, b)


def negative[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Tensor[dtype, LayoutType]:
    """`-a`, elementwise. `numpy.negative`."""
    var n = a.size()
    var values = a.to_host()
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = -values[i]
    return Tensor[dtype, LayoutType](a.context(), a.layout, out^)


def astype[
    target: DType, dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Tensor[target, LayoutType]:
    """`a` converted to `target`, elementwise. `numpy.astype`.

    Explicit, because numax has no dtype promotion: a binary operation
    requires both sides to already share a dtype, and this is how a caller
    makes that true. Implicit promotion in a language that infers
    parameters turns a dtype mismatch into a surprise rather than an
    error.
    """
    var n = a.size()
    var values = a.to_host()
    var out = List[Scalar[target]](length=n, fill=0)
    for i in range(n):
        out[i] = values[i].cast[target]()
    return Tensor[target, LayoutType](a.context(), out^)


def invert[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_integral():
    """Bitwise NOT, elementwise. `numpy.invert`.

    Integral dtypes only. The boolean form is
    `numax.core.logic.logical_not`, which is a different operation on a
    different type rather than the same one spelled twice.
    """
    var n = a.size()
    var values = a.to_host()
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = ~values[i]
    return Tensor[dtype, LayoutType](a.context(), a.layout, out^)
