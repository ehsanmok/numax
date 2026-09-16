"""Tests for `numax.core.ops` and the operators on `Tensor`.

Each function is checked against hand-computed values, the tensor-scalar
overloads alongside the tensor-tensor ones, and every operator is checked
to agree with the function it forwards to -- the point of the operators is
that `a + b` and `add(a, b)` are one call.

The routing tests at the bottom make the claims the `numax.core._drive`
move added: a run-time-shaped `Dynamic` operand gives the same answer as
the `Static` of the same extents, and `astype` truncates toward zero on
the way to an integer and reads every nonzero element as true on the way
to `DType.bool`. Asking for a target the tensor is not on is proven by
`examples/advanced/unified_tensor_gpu.mojo`, not here: `gpu=True`
compiles a device kernel, which a GPU-less CI runner cannot do.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax.core.array import (
    Dynamic,
    Static,
    Tensor,
    full,
    ones,
    zeros,
    zeros_dyn,
)
from numax.core.ops import (
    add,
    astype,
    divide,
    floor_divide,
    invert,
    mod,
    multiply,
    negative,
    power,
    subtract,
)

comptime dtype = DType.float64


def _t[n: Int](values: List[Float64]) raises -> Static[dtype, n]:
    var ctx = DeviceContext(api="cpu")
    var elements = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        elements.append(Scalar[dtype](values[i]))
    return Static[dtype, n](ctx, elements^)


def test_add_and_subtract() raises:
    var a = _t[3]([1.0, 2.0, 3.0])
    var b = _t[3]([10.0, 20.0, 30.0])
    var sum_ = add(a, b).to_host()
    assert_equal(sum_[0], 11.0)
    assert_equal(sum_[2], 33.0)
    var diff = subtract(b, a).to_host()
    assert_equal(diff[0], 9.0)
    assert_equal(diff[2], 27.0)


def test_multiply_and_divide() raises:
    var a = _t[3]([1.0, 2.0, 4.0])
    var b = _t[3]([2.0, 4.0, 8.0])
    var product = multiply(a, b).to_host()
    assert_equal(product[0], 2.0)
    assert_equal(product[2], 32.0)
    var quotient = divide(b, a).to_host()
    assert_equal(quotient[0], 2.0)
    assert_equal(quotient[2], 2.0)


def test_scalar_overloads() raises:
    var a = _t[3]([1.0, 2.0, 3.0])
    assert_equal(add(a, 10.0).to_host()[0], 11.0)
    assert_equal(subtract(a, 1.0).to_host()[0], 0.0)
    assert_equal(multiply(a, 3.0).to_host()[1], 6.0)
    assert_equal(divide(a, 2.0).to_host()[1], 1.0)


def test_floor_divide_and_mod() raises:
    var a = _t[3]([7.0, 8.0, 9.0])
    var b = _t[3]([2.0, 3.0, 4.0])
    var fd = floor_divide(a, b).to_host()
    assert_equal(fd[0], 3.0)
    assert_equal(fd[1], 2.0)
    var m = mod(a, b).to_host()
    assert_equal(m[0], 1.0)
    assert_equal(m[1], 2.0)
    assert_equal(m[2], 1.0)


def test_power_both_forms() raises:
    var a = _t[3]([1.0, 2.0, 3.0])
    var b = _t[3]([3.0, 2.0, 2.0])
    var p = power(a, b).to_host()
    assert_almost_equal(p[0], 1.0)
    assert_almost_equal(p[1], 4.0)
    assert_almost_equal(p[2], 9.0)
    var squared = power(a, 2.0).to_host()
    assert_almost_equal(squared[2], 9.0)


def test_negative() raises:
    var a = _t[3]([1.0, -2.0, 0.0])
    var n = negative(a).to_host()
    assert_equal(n[0], -1.0)
    assert_equal(n[1], 2.0)
    assert_equal(n[2], 0.0)


def test_operators_agree_with_their_functions() raises:
    var a = _t[4]([1.0, 2.0, 3.0, 4.0])
    var b = _t[4]([5.0, 6.0, 7.0, 8.0])
    var by_op = (a + b).to_host()
    var by_fn = add(a, b).to_host()
    for i in range(4):
        assert_equal(by_op[i], by_fn[i])
    var sub_op = (b - a).to_host()
    var sub_fn = subtract(b, a).to_host()
    for i in range(4):
        assert_equal(sub_op[i], sub_fn[i])
    var mul_op = (a * b).to_host()
    var mul_fn = multiply(a, b).to_host()
    for i in range(4):
        assert_equal(mul_op[i], mul_fn[i])
    var div_op = (b / a).to_host()
    var div_fn = divide(b, a).to_host()
    for i in range(4):
        assert_equal(div_op[i], div_fn[i])


def test_scalar_operators() raises:
    var a = _t[3]([1.0, 2.0, 3.0])
    assert_equal((a + 1.0).to_host()[0], 2.0)
    assert_equal((a - 1.0).to_host()[0], 0.0)
    assert_equal((a * 2.0).to_host()[2], 6.0)
    assert_equal((a / 2.0).to_host()[1], 1.0)


def test_unary_minus_operator() raises:
    var a = _t[2]([1.5, -2.5])
    var n = (-a).to_host()
    assert_equal(n[0], -1.5)
    assert_equal(n[1], 2.5)


def test_operators_preserve_rank() raises:
    var ctx = DeviceContext(api="cpu")
    var a = ones[dtype, 2, 3](ctx)
    var b = full[dtype, 2, 3](2.0, ctx=ctx)
    var result = a + b
    assert_equal(result.num_elements, 6)
    assert_equal(result.rank, 2)
    var values = result.to_host()
    for i in range(6):
        assert_equal(values[i], 3.0)


def test_astype_narrows_and_widens() raises:
    var a = _t[3]([1.5, 2.5, -3.5])
    var as_i32 = astype[DType.int32](a).to_host()
    assert_equal(Int(as_i32[0]), 1)
    assert_equal(Int(as_i32[1]), 2)
    assert_equal(Int(as_i32[2]), -3)
    var back = astype[DType.float32](a).to_host()
    assert_almost_equal(Float64(back[0]), 1.5)


def test_astype_preserves_shape() raises:
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, 2, 2](ctx)
    var converted = astype[DType.float32](a)
    assert_equal(converted.num_elements, 4)
    assert_equal(converted.rank, 2)


def _m[
    rows: Int, cols: Int
](values: List[Float64]) raises -> Static[dtype, rows, cols]:
    var ctx = DeviceContext(api="cpu")
    var elements = List[Scalar[dtype]](capacity=rows * cols)
    for i in range(rows * cols):
        elements.append(Scalar[dtype](values[i]))
    return Static[dtype, rows, cols](ctx, elements^)


def test_broadcasting_add_stretches_a_row_across_a_matrix() raises:
    # numpy: np.arange(6).reshape(2, 3) + np.array([10, 20, 30])
    var a = _m[2, 3]([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    var row = _t[3]([10.0, 20.0, 30.0])

    var got = add(a, row)
    assert_equal(got.dim_at(0), 2)
    assert_equal(got.dim_at(1), 3)
    var out = got.to_host()
    var expected = [10.0, 21.0, 32.0, 13.0, 24.0, 35.0]
    for i in range(6):
        assert_almost_equal(out[i], Scalar[dtype](expected[i]))


def test_broadcasting_stretches_both_operands() raises:
    # numpy: np.array([[1.], [2.], [3.]]) * np.array([10., 20., 30., 40.])
    # -- a (3, 1) against a (4,) gives a (3, 4), neither operand's shape.
    var col = _m[3, 1]([1.0, 2.0, 3.0])
    var row = _t[4]([10.0, 20.0, 30.0, 40.0])

    var got = multiply(col, row)
    assert_equal(got.dim_at(0), 3)
    assert_equal(got.dim_at(1), 4)
    var out = got.to_host()
    for r in range(3):
        for c in range(4):
            assert_almost_equal(
                out[r * 4 + c], Scalar[dtype]((r + 1) * (c + 1) * 10)
            )


def test_broadcasting_agrees_with_the_same_shape_overload() raises:
    # The claim worth pinning: broadcasting a (1, 3) up to a (2, 3) has to
    # give exactly what the same-shape overload gives for the materialized
    # operand, or the two spellings of `a - b` mean different things.
    var a = _m[2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var thin = _m[1, 3]([10.0, 20.0, 30.0])
    var wide = _m[2, 3]([10.0, 20.0, 30.0, 10.0, 20.0, 30.0])

    var broadcast = subtract(a, thin).to_host()
    var direct = subtract(a, wide).to_host()
    for i in range(6):
        assert_equal(broadcast[i], direct[i])


def test_broadcasting_rejects_an_incompatible_pair() raises:
    var a = _t[3]([1.0, 2.0, 3.0])
    var b = _t[4]([1.0, 2.0, 3.0, 4.0])
    var raised = False
    try:
        _ = add(a, b)
    except:
        raised = True
    assert_true(raised)


def test_same_shape_overload_still_keeps_its_layout_type() raises:
    # The broadcasting overload must not shadow the same-shape one: a
    # `Static` in has to stay a `Static` out, or every caller holding a
    # compile-time shape loses it.
    var a = _t[3]([1.0, 2.0, 3.0])
    var b = _t[3]([10.0, 20.0, 30.0])
    var got: Static[dtype, 3] = add(a, b)
    assert_equal(got.to_host()[2], 33.0)


def test_invert_flips_every_bit() raises:
    # numpy: ~np.array([0, 1, -1, 5], dtype=np.int32)
    var ctx = DeviceContext(api="cpu")
    var a = Static[DType.int32, 4](ctx, [0, 1, -1, 5])
    var got = invert(a).to_host()
    assert_equal(Int(got[0]), -1)
    assert_equal(Int(got[1]), -2)
    assert_equal(Int(got[2]), 0)
    assert_equal(Int(got[3]), -6)


def test_astype_to_bool_is_nonzero() raises:
    # numpy: np.array([0., -0., 1.5, -2.], np.float32).astype(bool)
    # -- the cast is "is this element nonzero", not a rounding, and the
    # destination comes from `_uninitialized`, so an all-false answer being
    # all false is the garbage check as much as the value check.
    var ctx = DeviceContext(api="cpu")
    var a = Static[DType.float32, 4](ctx, [0.0, -0.0, 1.5, -2.0])
    var got = astype[DType.bool](a).to_host()
    assert_equal(got[0], False)
    assert_equal(got[1], False)
    assert_equal(got[2], True)
    assert_equal(got[3], True)

    var zeroed = zeros[DType.float32, 5](ctx)
    var none = astype[DType.bool](zeroed).to_host()
    for i in range(5):
        assert_equal(none[i], False)


def test_a_dynamic_operand_gives_the_static_answer() raises:
    """A run-time shape takes the same driver, so it must give the same
    values as the `Static` of the same extents -- there is one signature per
    name and the flattening is a layout, not a `coalesce()`."""
    var ctx = DeviceContext(api="cpu")
    var left = List[Scalar[dtype]](capacity=6)
    var right = List[Scalar[dtype]](capacity=6)
    for i in range(6):
        left.append(Scalar[dtype](i) * 0.5 - 1.0)
        right.append(Scalar[dtype](i) * 0.25 + 2.0)

    var a_fixed = Static[dtype, 2, 3](ctx, left.copy())
    var b_fixed = Static[dtype, 2, 3](ctx, right.copy())
    var a_runtime = zeros_dyn[dtype, 2](2, 3, ctx=ctx)
    var b_runtime = zeros_dyn[dtype, 2](2, 3, ctx=ctx)
    a_runtime.copy_from_host(left)
    b_runtime.copy_from_host(right)

    var from_static = multiply(a_fixed, b_fixed).to_host()
    var from_dynamic = multiply(a_runtime, b_runtime).to_host()
    assert_equal(len(from_dynamic), 6)
    for i in range(6):
        assert_equal(from_dynamic[i], from_static[i])

    var neg_static = negative(a_fixed).to_host()
    var neg_dynamic = negative(a_runtime).to_host()
    for i in range(6):
        assert_equal(neg_dynamic[i], neg_static[i])

    var scaled_static = multiply(a_fixed, 3.0).to_host()
    var scaled_dynamic = multiply(a_runtime, 3.0).to_host()
    for i in range(6):
        assert_equal(scaled_dynamic[i], scaled_static[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
