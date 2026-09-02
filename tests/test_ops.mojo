"""Tests for `numax.ops` and the operators on `Tensor`.

Each function is checked against hand-computed values, the tensor-scalar
overloads alongside the tensor-tensor ones, and every operator is checked
to agree with the function it forwards to -- the point of the operators is
that `a + b` and `add(a, b)` are one call.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.array import Tensor, full, ones, zeros
from numax.ops import (
    add,
    astype,
    divide,
    floor_divide,
    mod,
    multiply,
    negative,
    power,
    subtract,
)

comptime dtype = DType.float64


def _t[n: Int](values: List[Float64]) raises -> Tensor[dtype, n]:
    var ctx = DeviceContext(api="cpu")
    var elements = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        elements.append(Scalar[dtype](values[i]))
    return Tensor[dtype, n](ctx, elements^)


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
    var b = full[dtype, 2, 3](ctx, 2.0)
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
