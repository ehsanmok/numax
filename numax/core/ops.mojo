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

from .tensorlike import TensorLike, is_row_major
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
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """`a + b`, elementwise. `numpy.add`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_add_op[dtype, _], gpu=gpu, name="add"](a, b)


def add[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[
    T.dtype, T.LayoutType
] where is_row_major[T]:
    """`a + b` with a scalar `b`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_scalar[T, op=_add_op[dtype, _], gpu=gpu, name="add"](a, b)


def subtract[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """`a - b`, elementwise. `numpy.subtract`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_sub_op[dtype, _], gpu=gpu, name="subtract"](a, b)


def subtract[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[
    T.dtype, T.LayoutType
] where is_row_major[T]:
    """`a - b` with a scalar `b`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_scalar[T, op=_sub_op[dtype, _], gpu=gpu, name="subtract"](
        a, b
    )


def multiply[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """`a * b`, elementwise. `numpy.multiply`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_mul_op[dtype, _], gpu=gpu, name="multiply"](a, b)


def multiply[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[
    T.dtype, T.LayoutType
] where is_row_major[T]:
    """`a * b` with a scalar `b`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_scalar[T, op=_mul_op[dtype, _], gpu=gpu, name="multiply"](
        a, b
    )


def divide[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """`a / b`, elementwise. `numpy.divide`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_div_op[dtype, _], gpu=gpu, name="divide"](a, b)


def divide[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[
    T.dtype, T.LayoutType
] where is_row_major[T]:
    """`a / b` with a scalar `b`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_scalar[T, op=_div_op[dtype, _], gpu=gpu, name="divide"](a, b)


def floor_divide[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """`a // b`, elementwise. `numpy.floor_divide`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[
        T,
        op=_floordiv_op[dtype, _],
        gpu=gpu,
        name="floor_divide",
    ](a, b)


def mod[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """`a % b`, elementwise. `numpy.mod`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_mod_op[dtype, _], gpu=gpu, name="mod"](a, b)


def power[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """`a ** b`, elementwise. `numpy.power`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_pow_op[dtype, _], gpu=gpu, name="power"](a, b)


def power[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """`a ** b` with a scalar exponent."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_scalar[T, op=_pow_op[dtype, _], gpu=gpu, name="power"](a, b)


# The broadcasting forms, at two shapes NumPy would broadcast. The result is
# a `Dynamic`, since the extents are run-time values; the same-shape
# overloads still match first and keep the input's own layout type.


def add[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """`a + b` at two broadcastable shapes. `numpy.add`."""
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[A, B, op=_add_op[dtype, _], gpu=gpu, name="add"](
        a, b
    )


def subtract[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """`a - b` at two broadcastable shapes. `numpy.subtract`."""
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A, B, op=_sub_op[dtype, _], gpu=gpu, name="subtract"
    ](a, b)


def multiply[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """`a * b` at two broadcastable shapes. `numpy.multiply`."""
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A, B, op=_mul_op[dtype, _], gpu=gpu, name="multiply"
    ](a, b)


def divide[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """`a / b` at two broadcastable shapes. `numpy.divide`."""
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[A, B, op=_div_op[dtype, _], gpu=gpu, name="divide"](
        a, b
    )


def floor_divide[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """`a // b` at two broadcastable shapes. `numpy.floor_divide`."""
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A,
        B,
        op=_floordiv_op[dtype, _],
        gpu=gpu,
        name="floor_divide",
    ](a, b)


def mod[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """`a % b` at two broadcastable shapes. `numpy.mod`."""
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[A, B, op=_mod_op[dtype, _], gpu=gpu, name="mod"](
        a, b
    )


def power[
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
    """`a ** b` at two broadcastable shapes. `numpy.power`."""
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[A, B, op=_pow_op[dtype, _], gpu=gpu, name="power"](
        a, b
    )


def _negative_op[dtype: DType, w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return -x


def negative[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """`-a`, elementwise. `numpy.negative`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_negative_op[dtype, _], gpu=gpu, name="negative"](a)


def _cast_op[
    target: DType, dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[target, w]:
    return x.cast[target]()


def astype[
    target: DType, T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[target, T.LayoutType] where is_row_major[T]:
    """`a` converted to `target`, elementwise. `numpy.astype`.

    Explicit, because numax has no dtype promotion: a binary operation
    requires both sides to already share a dtype, and this is how a caller
    makes that true. Implicit promotion in a language that infers
    parameters turns a dtype mismatch into a surprise rather than an
    error.

    The one routine here whose result dtype differs from its input's, so
    it goes through `_drive.unary_to` rather than `unary`.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary_to[
        T,
        target,
        op=_cast_op[target, dtype, _],
        gpu=gpu,
        name="astype",
    ](a)


def _invert_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w] where dtype.is_integral():
    return ~x


def invert[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_integral()
):
    """Bitwise NOT, elementwise. `numpy.invert`.

    Integral dtypes only. The boolean form is
    `numax.core.logic.logical_not`, which is a different operation on a
    different type rather than the same one spelled twice.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_invert_op[dtype, _], gpu=gpu, name="invert"](a)
