"""Elementwise arithmetic over `Tensor`, as functions and as operators.

**Tier 2 in shape, both targets in fact**, on the same terms as
`numax.core.elementwise` and `numax.core.rowwise`: these are `Plain`-only --
`TileTensor` holds raw `dtype` lanes -- but they are not host-only. Every
routine here runs through `numax.core._drive`, which launches one capturing
body on the target its `gpu: Bool` parameter names: `max.algorithm.elementwise`
on the device, the same threaded walk on a large host tensor, a serial SIMD
loop on a small one. `numax.core.tensor.map` with `add_step`/`mul_step` is
the tier-1 form, generic over every `FloatLike` conformer.

`gpu: Bool = False` is the last compile-time parameter of every routine, so
`add(a, b)` is the host spelling and `add[gpu=True](a, b)` the device one.
Asking for a target the tensor's memory does not live on is not an error:
the call falls back to a host walk over `to_host()` and prints one line on
`stderr` naming the spelling that would not have. `_drive`'s docstring has
the rest.

Every routine has three overloads: tensor-tensor at one shape,
tensor-scalar (which is what `a * 2.0` needs), and tensor-tensor at two
shapes NumPy would broadcast. The first returns the input's own layout
type and so keeps a compile-time shape compile-time; the broadcasting one
returns a `Dynamic`, since the extents it computes are run-time values.
A caller who wants the shape back in the type broadcasts explicitly with
`broadcast_to` and stays on the first.

Broadcasting here does not materialize either operand:
`numax.core.array._stretch_strides` gives a stretched axis stride 0, so
the body reads the same element for every position along it. The rule
itself is `broadcast_shapes`, written down once.

The operators on `Tensor` itself (`+`, `-`, `*`, `/`, unary `-`) forward
here at the default `gpu=False`, so `a + b` and `add(a, b)` are the same
call; a device tensor wanting the device path spells the function with
`[gpu=True]`. `numpy.power` is `power` rather than `**`: `__pow__` on a
tensor would have to choose between an elementwise power and a matrix
power, and NumPy's own answer to that (`**` is elementwise,
`numpy.linalg.matrix_power` is separate) is worth stating explicitly rather
than implying.
"""

from layout.tile_layout import TensorLayout

from .array import Dynamic, Tensor
from ._drive import (
    _BroadcastRank,
    binary,
    binary_scalar,
    broadcast_binary,
    unary,
    unary_to,
)


def _add_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return a + b


def _sub_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return a - b


def _mul_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return a * b


def _div_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return a / b


def _floordiv_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return a // b


def _mod_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return a % b


def _pow_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return a**b


def add[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a + b`, elementwise. `numpy.add`."""
    return binary[dtype, LayoutType, op=_add_op[dtype, _], gpu=gpu, name="add"](
        a, b
    )


def add[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Scalar[dtype]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a + b` with a scalar `b`."""
    return binary_scalar[
        dtype, LayoutType, op=_add_op[dtype, _], gpu=gpu, name="add"
    ](a, b)


def subtract[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a - b`, elementwise. `numpy.subtract`."""
    return binary[
        dtype, LayoutType, op=_sub_op[dtype, _], gpu=gpu, name="subtract"
    ](a, b)


def subtract[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Scalar[dtype]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a - b` with a scalar `b`."""
    return binary_scalar[
        dtype, LayoutType, op=_sub_op[dtype, _], gpu=gpu, name="subtract"
    ](a, b)


def multiply[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a * b`, elementwise. `numpy.multiply`."""
    return binary[
        dtype, LayoutType, op=_mul_op[dtype, _], gpu=gpu, name="multiply"
    ](a, b)


def multiply[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Scalar[dtype]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a * b` with a scalar `b`."""
    return binary_scalar[
        dtype, LayoutType, op=_mul_op[dtype, _], gpu=gpu, name="multiply"
    ](a, b)


def divide[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a / b`, elementwise. `numpy.divide`."""
    return binary[
        dtype, LayoutType, op=_div_op[dtype, _], gpu=gpu, name="divide"
    ](a, b)


def divide[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Scalar[dtype]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a / b` with a scalar `b`."""
    return binary_scalar[
        dtype, LayoutType, op=_div_op[dtype, _], gpu=gpu, name="divide"
    ](a, b)


def floor_divide[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a // b`, elementwise. `numpy.floor_divide`."""
    return binary[
        dtype,
        LayoutType,
        op=_floordiv_op[dtype, _],
        gpu=gpu,
        name="floor_divide",
    ](a, b)


def mod[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
]:
    """`a % b`, elementwise. `numpy.mod`."""
    return binary[dtype, LayoutType, op=_mod_op[dtype, _], gpu=gpu, name="mod"](
        a, b
    )


def power[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """`a ** b`, elementwise. `numpy.power`."""
    return binary[
        dtype, LayoutType, op=_pow_op[dtype, _], gpu=gpu, name="power"
    ](a, b)


def power[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Scalar[dtype]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """`a ** b` with a scalar exponent."""
    return binary_scalar[
        dtype, LayoutType, op=_pow_op[dtype, _], gpu=gpu, name="power"
    ](a, b)


# The broadcasting forms. Each is the same operation as the overload of its
# name above, at two shapes NumPy would broadcast rather than one shape
# twice -- a `(3, 1)` against a `(4,)`, a matrix against its row of column
# means. The result is a `Dynamic` because the broadcast extents are
# run-time values, so a caller who needs the shape back in the type
# broadcasts explicitly with `broadcast_to` and stays on the overload above.
# The same-shape overloads still match first and still return the input's
# own layout type, so nothing that compiled before changes shape.


def add[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, _BroadcastRank[ALayout, BLayout]
]:
    """`a + b` at two broadcastable shapes. `numpy.add`."""
    return broadcast_binary[
        dtype, ALayout, BLayout, op=_add_op[dtype, _], gpu=gpu, name="add"
    ](a, b)


def subtract[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, _BroadcastRank[ALayout, BLayout]
]:
    """`a - b` at two broadcastable shapes. `numpy.subtract`."""
    return broadcast_binary[
        dtype, ALayout, BLayout, op=_sub_op[dtype, _], gpu=gpu, name="subtract"
    ](a, b)


def multiply[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, _BroadcastRank[ALayout, BLayout]
]:
    """`a * b` at two broadcastable shapes. `numpy.multiply`."""
    return broadcast_binary[
        dtype, ALayout, BLayout, op=_mul_op[dtype, _], gpu=gpu, name="multiply"
    ](a, b)


def divide[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, _BroadcastRank[ALayout, BLayout]
]:
    """`a / b` at two broadcastable shapes. `numpy.divide`."""
    return broadcast_binary[
        dtype, ALayout, BLayout, op=_div_op[dtype, _], gpu=gpu, name="divide"
    ](a, b)


def floor_divide[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, _BroadcastRank[ALayout, BLayout]
]:
    """`a // b` at two broadcastable shapes. `numpy.floor_divide`."""
    return broadcast_binary[
        dtype,
        ALayout,
        BLayout,
        op=_floordiv_op[dtype, _],
        gpu=gpu,
        name="floor_divide",
    ](a, b)


def mod[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, _BroadcastRank[ALayout, BLayout]
]:
    """`a % b` at two broadcastable shapes. `numpy.mod`."""
    return broadcast_binary[
        dtype, ALayout, BLayout, op=_mod_op[dtype, _], gpu=gpu, name="mod"
    ](a, b)


def power[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, _BroadcastRank[ALayout, BLayout]
] where dtype.is_floating_point():
    """`a ** b` at two broadcastable shapes. `numpy.power`."""
    return broadcast_binary[
        dtype, ALayout, BLayout, op=_pow_op[dtype, _], gpu=gpu, name="power"
    ](a, b)


def _negative_op[dtype: DType, w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return -x


def negative[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[dtype, LayoutType]:
    """`-a`, elementwise. `numpy.negative`."""
    return unary[
        dtype, LayoutType, op=_negative_op[dtype, _], gpu=gpu, name="negative"
    ](a)


def _cast_op[
    target: DType, dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[target, w]:
    return x.cast[target]()


def astype[
    target: DType, dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[target, LayoutType]:
    """`a` converted to `target`, elementwise. `numpy.astype`.

    Explicit, because numax has no dtype promotion: a binary operation
    requires both sides to already share a dtype, and this is how a caller
    makes that true. Implicit promotion in a language that infers
    parameters turns a dtype mismatch into a surprise rather than an
    error.

    The one routine here whose result dtype differs from its input's, so
    it goes through `_drive.unary_to` rather than `unary`.
    """
    return unary_to[
        dtype,
        target,
        LayoutType,
        op=_cast_op[target, dtype, _],
        gpu=gpu,
        name="astype",
    ](a)


def _invert_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_integral():
    return ~x


def invert[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_integral():
    """Bitwise NOT, elementwise. `numpy.invert`.

    Integral dtypes only. The boolean form is
    `numax.core.logic.logical_not`, which is a different operation on a
    different type rather than the same one spelled twice.
    """
    return unary[
        dtype, LayoutType, op=_invert_op[dtype, _], gpu=gpu, name="invert"
    ](a)
