"""Elementwise arithmetic over `Tensor`, as functions and as operators.

**Tier 2 in shape, both targets in fact**, on the same terms as
`numax.core.elementwise` and `numax.core.rowwise`: these are `Plain`-only --
`TileTensor` holds raw `dtype` lanes -- but they are not host-only. Every
routine here runs through `numax.core._drive`, which launches one capturing
body on the target its `gpu: Bool` parameter names: `max.algorithm.elementwise`
on the device, the same threaded walk on a large host tensor, a serial SIMD
loop on a small one. `numax.core.functional.map` with `add_step`/`mul_step` is
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
`numax.core.tensor._stretch_strides` gives a stretched axis stride 0, so
the body reads the same element for every position along it. The rule
itself is `broadcast_shapes`, written down once.

The integer surface -- `bitwise_and`, `bitwise_or`, `bitwise_xor`,
`left_shift`, `right_shift`, `gcd`, `lcm`, beside `invert` -- has the same
three overloads and constrains its dtype to integers (and, for the three
bitwise operations, `bool`). `gcd` runs Euclid's algorithm for a fixed
number of steps, enough for any pair at the dtype's width, so it is one
uniform body on either target.

The operators on `Tensor` itself (`+`, `-`, `*`, `/`, unary `-`) forward
here at the default `gpu=False`, so `a + b` and `add(a, b)` are the same
call; a device tensor wanting the device path spells the function with
`[gpu=True]`. `numpy.power` is `power` rather than `**`: `__pow__` on a
tensor would have to choose between an elementwise power and a matrix
power, and NumPy's own answer to that (`**` is elementwise,
`numpy.linalg.matrix_power` is separate) is worth stating explicitly rather
than implying.
"""

from std.sys.info import size_of

from layout.tile_layout import TensorLayout

from .tensorlike import TensorLike, is_row_major
from .tensor import Dynamic, Tensor
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


def _rsub_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return b - a


def _rdiv_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return b / a


def _rpow_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return b**a


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
    """`a + b`, elementwise. `numpy.add`.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Left-hand operand, a tensor of type `T`.
        b: Right-hand operand, a tensor of type `T` at `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the elementwise sum
        `a + b`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_add_op[dtype, _], gpu=gpu, name="add"](a, b)


def add[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[
    T.dtype, T.LayoutType
] where is_row_major[T]:
    """`a + b` with a scalar `b`.

    Parameters:
        T: The `TensorLike` type of `a`; it fixes the dtype and the row-major
            layout.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Left-hand operand, a tensor of type `T`.
        b: Scalar right-hand operand, applied to every element of `a`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `a + b` for every
        element of `a`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_scalar[T, op=_add_op[dtype, _], gpu=gpu, name="add"](a, b)


def subtract[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """`a - b`, elementwise. `numpy.subtract`.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Minuend, a tensor of type `T`.
        b: Subtrahend, a tensor of type `T` at `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the elementwise
        difference `a - b`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_sub_op[dtype, _], gpu=gpu, name="subtract"](a, b)


def subtract[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[
    T.dtype, T.LayoutType
] where is_row_major[T]:
    """`a - b` with a scalar `b`.

    Parameters:
        T: The `TensorLike` type of `a`; it fixes the dtype and the row-major
            layout.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Minuend, a tensor of type `T`.
        b: Scalar subtrahend, applied to every element of `a`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `a - b` for every
        element of `a`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_scalar[T, op=_sub_op[dtype, _], gpu=gpu, name="subtract"](
        a, b
    )


def multiply[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """`a * b`, elementwise. `numpy.multiply`.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Left-hand operand, a tensor of type `T`.
        b: Right-hand operand, a tensor of type `T` at `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the elementwise
        product `a * b`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_mul_op[dtype, _], gpu=gpu, name="multiply"](a, b)


def multiply[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[
    T.dtype, T.LayoutType
] where is_row_major[T]:
    """`a * b` with a scalar `b`.

    Parameters:
        T: The `TensorLike` type of `a`; it fixes the dtype and the row-major
            layout.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Left-hand operand, a tensor of type `T`.
        b: Scalar right-hand operand, applied to every element of `a`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `a * b` for every
        element of `a`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_scalar[T, op=_mul_op[dtype, _], gpu=gpu, name="multiply"](
        a, b
    )


def divide[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """`a / b`, elementwise. `numpy.divide`.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Dividend, a tensor of type `T`.
        b: Divisor, a tensor of type `T` at `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the elementwise
        quotient `a / b`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_div_op[dtype, _], gpu=gpu, name="divide"](a, b)


def divide[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[
    T.dtype, T.LayoutType
] where is_row_major[T]:
    """`a / b` with a scalar `b`.

    Parameters:
        T: The `TensorLike` type of `a`; it fixes the dtype and the row-major
            layout.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Dividend, a tensor of type `T`.
        b: Scalar divisor, applied to every element of `a`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `a / b` for every
        element of `a`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_scalar[T, op=_div_op[dtype, _], gpu=gpu, name="divide"](a, b)


def floor_divide[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """`a // b`, elementwise. `numpy.floor_divide`.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Dividend, a tensor of type `T`.
        b: Divisor, a tensor of type `T` at `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the elementwise
        floor quotient `a // b`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
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
    """`a % b`, elementwise. `numpy.mod`.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Dividend, a tensor of type `T`.
        b: Divisor, a tensor of type `T` at `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the elementwise
        remainder `a % b`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_mod_op[dtype, _], gpu=gpu, name="mod"](a, b)


def power[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """`a ** b`, elementwise. `numpy.power`.

    Parameters:
        T: The `TensorLike` type of both operands, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Base, a tensor of type `T`.
        b: Exponent, a tensor of type `T` at `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the elementwise
        power `a ** b`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_pow_op[dtype, _], gpu=gpu, name="power"](a, b)


def power[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """`a ** b` with a scalar exponent.

    Parameters:
        T: The `TensorLike` type of `a`, a row-major layout over a
            floating-point dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Base, a tensor of type `T`.
        b: Scalar exponent, applied to every element of `a`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `a ** b` for every
        element of `a`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
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
    """`a + b` at two broadcastable shapes. `numpy.add`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Left-hand operand; it is read through stride 0 on any axis it
            stretches.
        b: Right-hand operand, at a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding the sum `a + b`.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
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
    """`a - b` at two broadcastable shapes. `numpy.subtract`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Minuend; it is read through stride 0 on any axis it stretches.
        b: Subtrahend, at a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding the difference `a - b`.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
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
    """`a * b` at two broadcastable shapes. `numpy.multiply`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Left-hand operand; it is read through stride 0 on any axis it
            stretches.
        b: Right-hand operand, at a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding the product `a * b`.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
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
    """`a / b` at two broadcastable shapes. `numpy.divide`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Dividend; it is read through stride 0 on any axis it stretches.
        b: Divisor, at a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding the quotient `a / b`.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
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
    """`a // b` at two broadcastable shapes. `numpy.floor_divide`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Dividend; it is read through stride 0 on any axis it stretches.
        b: Divisor, at a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding the floor quotient `a // b`.

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
    """`a % b` at two broadcastable shapes. `numpy.mod`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Dividend; it is read through stride 0 on any axis it stretches.
        b: Divisor, at a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding the remainder `a % b`.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
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
    """`a ** b` at two broadcastable shapes. `numpy.power`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major and floating-point.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Base; it is read through stride 0 on any axis it stretches.
        b: Exponent, at a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding the power `a ** b`.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[A, B, op=_pow_op[dtype, _], gpu=gpu, name="power"](
        a, b
    )


def _negative_op[dtype: DType, w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return -x


def _reflected[
    T: TensorLike, kind: StaticString, gpu: Bool = False
](a: T, s: Scalar[T.dtype]) raises -> Tensor[
    T.dtype, T.LayoutType
] where is_row_major[T]:
    """`s - a` or `s / a` (`kind` `"sub"` or `"div"`) with the scalar on the
    left: what `Tensor`'s reflected operators forward to. One launch with
    the operands swapped inside the op, so the rounding is the left-hand
    spelling's rather than a negation or reciprocal of the other."""
    comptime dtype = T.dtype
    comptime if kind == "sub":
        return binary_scalar[
            T, op=_rsub_op[dtype, _], gpu=gpu, name="subtract"
        ](a, s)
    else:
        return binary_scalar[T, op=_rdiv_op[dtype, _], gpu=gpu, name="divide"](
            a, s
        )


def _rpower[
    T: TensorLike, gpu: Bool = False
](a: T, s: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """`s ** a` with the scalar base on the left, for `__rpow__`."""
    comptime dtype = T.dtype
    return binary_scalar[T, op=_rpow_op[dtype, _], gpu=gpu, name="power"](a, s)


def negative[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """`-a`, elementwise. `numpy.negative`.

    Parameters:
        T: The `TensorLike` type of `a`; it fixes the dtype and the row-major
            layout.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor to negate.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `-a` for every
        element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
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

    Parameters:
        target: The dtype every element is cast to.
        T: The `TensorLike` type of `a`, row-major, at any dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor whose elements are converted.

    Returns:
        A new `Tensor` of `target` at `a`'s layout holding each element of `a`
        cast to `target`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
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

    Parameters:
        T: The `TensorLike` type of `a`, row-major over an integral dtype.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Tensor whose bits are complemented.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the bitwise NOT
        `~a` of every element.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary[T, op=_invert_op[dtype, _], gpu=gpu, name="invert"](a)


# The integer surface: bitwise operations, shifts and `gcd`/`lcm`. Each has
# the three overloads the arithmetic above has.


comptime _GCD_STEPS[dtype: DType] = size_of[Scalar[dtype]]() * 12 + 2
"""Euclid steps enough for any pair of `dtype` values: the worst case is
consecutive Fibonacci numbers, and `F(k)` passes `2**b` near `k = 1.44 b`,
so `1.5 b` steps (12 per byte) and two to spare cover every pair."""


def _gcd_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_integral():
    var x = abs(a)
    var y = abs(b)
    var zero = SIMD[dtype, w](0)
    var one = SIMD[dtype, w](1)
    for _ in range(_GCD_STEPS[dtype]):
        # A finished lane has `y == 0`; it divides by a stand-in `1` and
        # keeps its `x`, so every lane does the same work.
        var live = y.ne(zero)
        var r = x % live.select(y, one)
        x = live.select(y, x)
        y = live.select(r, zero)
    return x


def _lcm_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_integral():
    var zero = SIMD[dtype, w](0)
    var g = _gcd_op[dtype, w](a, b)
    var safe = g.eq(zero).select(SIMD[dtype, w](1), g)
    return g.eq(zero).select(zero, abs(a) // safe * abs(b))


def _bitwise_and_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w] where (
    dtype.is_integral() or dtype == DType.bool
):
    return a & b


def bitwise_and[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[
    T
] and (T.dtype.is_integral() or T.dtype == DType.bool):
    """`a & b`, elementwise. `numpy.bitwise_and`.

    Integral or `bool` dtypes; on `bool` it is `numax.core.logic.logical_and`
    by another name, as in NumPy.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Bits of the left operand, a tensor of type `T`.
        b: Bits of the right operand, a tensor of type `T` at `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the bitwise AND `a & b` for every
        element pair.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_bitwise_and_op[dtype, _], gpu=gpu, name="bitwise_and"](
        a, b
    )


def bitwise_and[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[
    T.dtype, T.LayoutType
] where is_row_major[T] and (T.dtype.is_integral() or T.dtype == DType.bool):
    """`bitwise_and` with a scalar `b`.

    Parameters:
        T: The `TensorLike` type of `a`; it fixes the dtype and the row-major
            layout.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Bits of the left operand, a tensor of type `T`.
        b: Scalar right-hand operand, applied to every element of `a`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the bitwise AND `a & b` for every
        element of `a`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_scalar[
        T, op=_bitwise_and_op[dtype, _], gpu=gpu, name="bitwise_and"
    ](a, b)


def bitwise_and[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == B.dtype
    and is_row_major[A]
    and is_row_major[B]
    and (A.dtype.is_integral() or A.dtype == DType.bool)
):
    """`bitwise_and` at two broadcastable shapes. `numpy.bitwise_and`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Bits of the left operand; it is read through stride 0 on any axis it stretches.
        b: Bits of the right operand, at a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding the bitwise AND `a & b`.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A, B, op=_bitwise_and_op[dtype, _], gpu=gpu, name="bitwise_and"
    ](a, b)


def _bitwise_or_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w] where (
    dtype.is_integral() or dtype == DType.bool
):
    return a | b


def bitwise_or[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[
    T
] and (T.dtype.is_integral() or T.dtype == DType.bool):
    """`a | b`, elementwise. `numpy.bitwise_or`.

    Integral or `bool` dtypes, as `bitwise_and`.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Bits of the left operand, a tensor of type `T`.
        b: Bits of the right operand, a tensor of type `T` at `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the bitwise OR `a | b` for every
        element pair.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_bitwise_or_op[dtype, _], gpu=gpu, name="bitwise_or"](
        a, b
    )


def bitwise_or[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[
    T.dtype, T.LayoutType
] where is_row_major[T] and (T.dtype.is_integral() or T.dtype == DType.bool):
    """`bitwise_or` with a scalar `b`.

    Parameters:
        T: The `TensorLike` type of `a`; it fixes the dtype and the row-major
            layout.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Bits of the left operand, a tensor of type `T`.
        b: Scalar right-hand operand, applied to every element of `a`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the bitwise OR `a | b` for every
        element of `a`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_scalar[
        T, op=_bitwise_or_op[dtype, _], gpu=gpu, name="bitwise_or"
    ](a, b)


def bitwise_or[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == B.dtype
    and is_row_major[A]
    and is_row_major[B]
    and (A.dtype.is_integral() or A.dtype == DType.bool)
):
    """`bitwise_or` at two broadcastable shapes. `numpy.bitwise_or`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Bits of the left operand; it is read through stride 0 on any axis it stretches.
        b: Bits of the right operand, at a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding the bitwise OR `a | b`.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A, B, op=_bitwise_or_op[dtype, _], gpu=gpu, name="bitwise_or"
    ](a, b)


def _bitwise_xor_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w] where (
    dtype.is_integral() or dtype == DType.bool
):
    return a ^ b


def bitwise_xor[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[
    T
] and (T.dtype.is_integral() or T.dtype == DType.bool):
    """`a ^ b`, elementwise. `numpy.bitwise_xor`.

    Integral or `bool` dtypes, as `bitwise_and`.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Bits of the left operand, a tensor of type `T`.
        b: Bits of the right operand, a tensor of type `T` at `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the bitwise XOR `a ^ b` for every
        element pair.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_bitwise_xor_op[dtype, _], gpu=gpu, name="bitwise_xor"](
        a, b
    )


def bitwise_xor[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[
    T.dtype, T.LayoutType
] where is_row_major[T] and (T.dtype.is_integral() or T.dtype == DType.bool):
    """`bitwise_xor` with a scalar `b`.

    Parameters:
        T: The `TensorLike` type of `a`; it fixes the dtype and the row-major
            layout.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Bits of the left operand, a tensor of type `T`.
        b: Scalar right-hand operand, applied to every element of `a`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding the bitwise XOR `a ^ b` for every
        element of `a`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_scalar[
        T, op=_bitwise_xor_op[dtype, _], gpu=gpu, name="bitwise_xor"
    ](a, b)


def bitwise_xor[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == B.dtype
    and is_row_major[A]
    and is_row_major[B]
    and (A.dtype.is_integral() or A.dtype == DType.bool)
):
    """`bitwise_xor` at two broadcastable shapes. `numpy.bitwise_xor`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Bits of the left operand; it is read through stride 0 on any axis it stretches.
        b: Bits of the right operand, at a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding the bitwise XOR `a ^ b`.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A, B, op=_bitwise_xor_op[dtype, _], gpu=gpu, name="bitwise_xor"
    ](a, b)


def _left_shift_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_integral():
    return a << b


def left_shift[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_integral()
):
    """`a << b`, elementwise. `numpy.left_shift`.

    Integral dtypes only. A count at or past the bit width is undefined in
    the hardware shift, as it is in C and NumPy; it is not clamped.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Values to shift, a tensor of type `T`.
        b: Shift counts, a tensor of type `T` at `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `a` shifted left by `b` bits for every
        element pair.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_left_shift_op[dtype, _], gpu=gpu, name="left_shift"](
        a, b
    )


def left_shift[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_integral()
):
    """`left_shift` with a scalar `b`.

    Parameters:
        T: The `TensorLike` type of `a`; it fixes the dtype and the row-major
            layout.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Values to shift, a tensor of type `T`.
        b: Scalar right-hand operand, applied to every element of `a`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `a` shifted left by `b` bits for every
        element of `a`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_scalar[
        T, op=_left_shift_op[dtype, _], gpu=gpu, name="left_shift"
    ](a, b)


def left_shift[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == B.dtype
    and is_row_major[A]
    and is_row_major[B]
    and A.dtype.is_integral()
):
    """`left_shift` at two broadcastable shapes. `numpy.left_shift`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Values to shift; it is read through stride 0 on any axis it stretches.
        b: Shift counts, at a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding `a` shifted left by `b` bits.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A, B, op=_left_shift_op[dtype, _], gpu=gpu, name="left_shift"
    ](a, b)


def _right_shift_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_integral():
    return a >> b


def right_shift[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_integral()
):
    """`a >> b`, elementwise. `numpy.right_shift`.

    Integral dtypes only. Arithmetic on signed dtypes (the sign bit is
    copied in) and logical on unsigned ones, which is what NumPy does; a
    count at or past the bit width is undefined, as `left_shift` says.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Values to shift, a tensor of type `T`.
        b: Shift counts, a tensor of type `T` at `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `a` shifted right by `b` bits for every
        element pair.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_right_shift_op[dtype, _], gpu=gpu, name="right_shift"](
        a, b
    )


def right_shift[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_integral()
):
    """`right_shift` with a scalar `b`.

    Parameters:
        T: The `TensorLike` type of `a`; it fixes the dtype and the row-major
            layout.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Values to shift, a tensor of type `T`.
        b: Scalar right-hand operand, applied to every element of `a`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `a` shifted right by `b` bits for every
        element of `a`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_scalar[
        T, op=_right_shift_op[dtype, _], gpu=gpu, name="right_shift"
    ](a, b)


def right_shift[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == B.dtype
    and is_row_major[A]
    and is_row_major[B]
    and A.dtype.is_integral()
):
    """`right_shift` at two broadcastable shapes. `numpy.right_shift`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: Values to shift; it is read through stride 0 on any axis it stretches.
        b: Shift counts, at a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding `a` shifted right by `b` bits.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[
        A, B, op=_right_shift_op[dtype, _], gpu=gpu, name="right_shift"
    ](a, b)


def gcd[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_integral()
):
    """The greatest common divisor, elementwise. `numpy.gcd`.

    Integral dtypes only, and never negative: `gcd(0, 0) == 0` and
    `gcd(a, 0) == |a|`, as NumPy has it. Euclid's algorithm at a fixed
    step count -- enough for the worst case of the dtype, the Fibonacci
    pair at its width -- so every lane does the same work and the body
    runs unchanged on the device.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: First integers, a tensor of type `T`.
        b: Second integers, a tensor of type `T` at `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `gcd(|a|, |b|)` for every
        element pair.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_gcd_op[dtype, _], gpu=gpu, name="gcd"](a, b)


def gcd[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_integral()
):
    """`gcd` with a scalar `b`.

    Parameters:
        T: The `TensorLike` type of `a`; it fixes the dtype and the row-major
            layout.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: First integers, a tensor of type `T`.
        b: Scalar right-hand operand, applied to every element of `a`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `gcd(|a|, |b|)` for every
        element of `a`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_scalar[T, op=_gcd_op[dtype, _], gpu=gpu, name="gcd"](a, b)


def gcd[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == B.dtype
    and is_row_major[A]
    and is_row_major[B]
    and A.dtype.is_integral()
):
    """`gcd` at two broadcastable shapes. `numpy.gcd`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: First integers; it is read through stride 0 on any axis it stretches.
        b: Second integers, at a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding `gcd(|a|, |b|)`.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[A, B, op=_gcd_op[dtype, _], gpu=gpu, name="gcd"](
        a, b
    )


def lcm[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_integral()
):
    """The least common multiple, elementwise. `numpy.lcm`.

    Integral dtypes only, never negative, and `0` when either operand is
    `0`. `|a| / gcd(a, b) * |b|`, dividing first so the intermediate is
    no larger than the answer; an answer past the dtype's range wraps, as
    it does in NumPy.

    Parameters:
        T: The `TensorLike` type of both operands; it fixes the dtype and the
            row-major layout they share.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: First integers, a tensor of type `T`.
        b: Second integers, a tensor of type `T` at `a`'s shape.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `lcm(|a|, |b|)` for every
        element pair.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary[T, op=_lcm_op[dtype, _], gpu=gpu, name="lcm"](a, b)


def lcm[
    T: TensorLike, gpu: Bool = False
](a: T, b: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_integral()
):
    """`lcm` with a scalar `b`.

    Parameters:
        T: The `TensorLike` type of `a`; it fixes the dtype and the row-major
            layout.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: First integers, a tensor of type `T`.
        b: Scalar right-hand operand, applied to every element of `a`.

    Returns:
        A new `Tensor` at `a`'s layout and `T.dtype` holding `lcm(|a|, |b|)` for every
        element of `a`.

    Raises:
        If allocating the result or launching the walk fails, or on a residency
        mismatch under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_scalar[T, op=_lcm_op[dtype, _], gpu=gpu, name="lcm"](a, b)


def lcm[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == B.dtype
    and is_row_major[A]
    and is_row_major[B]
    and A.dtype.is_integral()
):
    """`lcm` at two broadcastable shapes. `numpy.lcm`.

    Parameters:
        A: The `TensorLike` type of `a`, row-major.
        B: The `TensorLike` type of `b`, row-major, with `A`'s dtype.
        gpu: `True` runs on the tensors' device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        a: First integers; it is read through stride 0 on any axis it stretches.
        b: Second integers, at a shape that broadcasts against `a`'s.

    Returns:
        A new run-time-shaped `Dynamic` tensor of `A.dtype` at the broadcast
        shape of `a` and `b`, holding `lcm(|a|, |b|)`.

    Raises:
        If the two shapes do not broadcast, if allocating the result or
        launching the walk fails, or on a residency mismatch under the `"raise"`
        fallback policy.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary[A, B, op=_lcm_op[dtype, _], gpu=gpu, name="lcm"](
        a, b
    )
