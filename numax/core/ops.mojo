"""Elementwise arithmetic over `Tensor`, as functions and as operators.

**This module is tier 2**, on the same terms as `numax.core.logic` and
`numax.core.elementwise`: a host walk returning a new tensor. `numax.core.tensor.map`
with `add_step`/`mul_step` is the tier-1, GPU-launchable form.

Every routine has three overloads: tensor-tensor at one shape,
tensor-scalar (which is what `a * 2.0` needs), and tensor-tensor at two
shapes NumPy would broadcast. The first returns the input's own layout
type and so keeps a compile-time shape compile-time; the broadcasting one
returns a `Dynamic`, since the extents it computes are run-time values.
A caller who wants the shape back in the type broadcasts explicitly with
`broadcast_to` and stays on the first.

Broadcasting here does not materialize either operand:
`numax.core.array._stretch_strides` gives a stretched axis stride 0, so
the walk reads the same element for every position along it. The rule
itself is `broadcast_shapes`, written down once.

The operators on `Tensor` itself (`+`, `-`, `*`, `/`, unary `-`) forward
here, so `a + b` and `add(a, b)` are the same call. `numpy.power` is
`power` rather than `**`: `__pow__` on a tensor would have to choose
between an elementwise power and a matrix power, and NumPy's own answer to
that (`**` is elementwise, `numpy.linalg.matrix_power` is separate) is
worth stating explicitly rather than implying.
"""

from layout.tile_layout import TensorLayout, row_major

from .array import (
    Dynamic,
    Tensor,
    _dyn_shape_from,
    _extents_of,
    _stretch_strides,
    _strides_of,
    broadcast_shapes,
)


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


def _zip_broadcast[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    op: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[dtype],
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`op` over two tensors of different shapes, under NumPy's rules.

    Neither operand is materialized at the broadcast shape: `broadcast_to`
    would copy each of them, and for an `(n, n)` against an `(n,)` that is a
    whole extra square buffer. `_stretch_strides` gives a stretched axis
    stride 0 instead, so the walk reads the same element for every position
    along it and the cost is index arithmetic.

    The result is a `Dynamic`, since the broadcast extents are run-time
    values even when both inputs' are not -- which is also why the result
    rank is the larger of the two input ranks, computed in the signature.
    """
    comptime rank = ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
    var a_extents = _extents_of(a)
    var b_extents = _extents_of(b)
    var extents = broadcast_shapes(a_extents, b_extents)
    var a_strides = _stretch_strides(a_extents, _strides_of(a), rank)
    var b_strides = _stretch_strides(b_extents, _strides_of(b), rank)

    var count = 1
    for d in range(rank):
        count *= extents[d]

    var a_values = a.to_host()
    var b_values = b.to_host()
    var out = List[Scalar[dtype]](length=count, fill=0)
    for flat in range(count):
        var rem = flat
        var ai = 0
        var bi = 0
        for k in range(rank):
            var d = rank - 1 - k
            var c = rem % extents[d]
            rem //= extents[d]
            ai += c * a_strides[d]
            bi += c * b_strides[d]
        out[flat] = op(a_values[ai], b_values[bi])

    var result = Dynamic[dtype, rank](
        a.context(), row_major(_dyn_shape_from[rank](extents))
    )
    result.copy_from_host(out)
    return result^


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


# The broadcasting forms. Each is the same operation as the overload of its
# name above, at two shapes NumPy would broadcast rather than one shape
# twice -- a `(3, 1)` against a `(4,)`, a matrix against its row of column
# means. The result is a `Dynamic` because the broadcast extents are
# run-time values, so a caller who needs the shape back in the type
# broadcasts explicitly with `broadcast_to` and stays on the overload above.
# The same-shape overloads still match first and still return the input's
# own layout type, so nothing that compiled before changes shape.


def add[
    dtype: DType, ALayout: TensorLayout, BLayout: TensorLayout
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`a + b` at two broadcastable shapes. `numpy.add`."""
    return _zip_broadcast[dtype, ALayout, BLayout, op=_add_op[dtype]](a, b)


def subtract[
    dtype: DType, ALayout: TensorLayout, BLayout: TensorLayout
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`a - b` at two broadcastable shapes. `numpy.subtract`."""
    return _zip_broadcast[dtype, ALayout, BLayout, op=_sub_op[dtype]](a, b)


def multiply[
    dtype: DType, ALayout: TensorLayout, BLayout: TensorLayout
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`a * b` at two broadcastable shapes. `numpy.multiply`."""
    return _zip_broadcast[dtype, ALayout, BLayout, op=_mul_op[dtype]](a, b)


def divide[
    dtype: DType, ALayout: TensorLayout, BLayout: TensorLayout
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`a / b` at two broadcastable shapes. `numpy.divide`."""
    return _zip_broadcast[dtype, ALayout, BLayout, op=_div_op[dtype]](a, b)


def floor_divide[
    dtype: DType, ALayout: TensorLayout, BLayout: TensorLayout
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`a // b` at two broadcastable shapes. `numpy.floor_divide`."""
    return _zip_broadcast[dtype, ALayout, BLayout, op=_floordiv_op[dtype]](a, b)


def mod[
    dtype: DType, ALayout: TensorLayout, BLayout: TensorLayout
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`a % b` at two broadcastable shapes. `numpy.mod`."""
    return _zip_broadcast[dtype, ALayout, BLayout, op=_mod_op[dtype]](a, b)


def power[
    dtype: DType, ALayout: TensorLayout, BLayout: TensorLayout
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
] where dtype.is_floating_point():
    """`a ** b` at two broadcastable shapes. `numpy.power`."""
    return _zip_broadcast[dtype, ALayout, BLayout, op=_pow_op[dtype]](a, b)


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
