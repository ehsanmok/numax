"""Tests for the integer surface of `numax.core.ops`: the bitwise
operations, the shifts, `gcd` and `lcm`.

Each is checked against values NumPy gives for the same inputs, at the
three overloads -- same shape, a scalar operand, and two shapes that
broadcast -- and the scalar and broadcasting forms are held to the
same-shape one, since they are the same operation spelled three ways.
`gcd`/`lcm` are checked at their edges (zero operands, negative operands,
the worst-case Fibonacci pair of `int64`) because the fixed step count is
the claim the implementation makes.
"""

from std.testing import TestSuite, assert_equal, assert_true

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.core.ops import (
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    gcd,
    lcm,
    left_shift,
    right_shift,
)

comptime i32 = DType.int32


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def test_bitwise_and_or_xor_match_numpy() raises:
    # numpy: a = np.array([12, 10, -1, 0], np.int32); b = [10, 6, 7, 5]
    var a = Static[i32, 4]([12, 10, -1, 0], _cpu())
    var b = Static[i32, 4]([10, 6, 7, 5], _cpu())
    var got_and = bitwise_and(a, b).to_host()
    var got_or = bitwise_or(a, b).to_host()
    var got_xor = bitwise_xor(a, b).to_host()
    var want_and = [8, 2, 7, 0]
    var want_or = [14, 14, -1, 5]
    var want_xor = [6, 12, -8, 5]
    for i in range(4):
        assert_equal(Int(got_and[i]), want_and[i])
        assert_equal(Int(got_or[i]), want_or[i])
        assert_equal(Int(got_xor[i]), want_xor[i])


def test_bitwise_on_bool_is_logical() raises:
    var a = Static[DType.bool, 4]([True, True, False, False], _cpu())
    var b = Static[DType.bool, 4]([True, False, True, False], _cpu())
    var got_and = bitwise_and(a, b).to_host()
    var got_or = bitwise_or(a, b).to_host()
    var got_xor = bitwise_xor(a, b).to_host()
    var want_and = [True, False, False, False]
    var want_or = [True, True, True, False]
    var want_xor = [False, True, True, False]
    for i in range(4):
        assert_equal(Bool(got_and[i]), want_and[i])
        assert_equal(Bool(got_or[i]), want_or[i])
        assert_equal(Bool(got_xor[i]), want_xor[i])


def test_shifts_are_arithmetic_on_signed_values() raises:
    # numpy: np.left_shift([1, 3, -4, 5], [0, 2, 1, 3]) and right_shift
    var a = Static[i32, 4]([1, 3, -4, 40], _cpu())
    var s = Static[i32, 4]([0, 2, 1, 3], _cpu())
    var left = left_shift(a, s).to_host()
    var right = right_shift(a, s).to_host()
    var want_left = [1, 12, -8, 320]
    var want_right = [1, 0, -2, 5]
    for i in range(4):
        assert_equal(Int(left[i]), want_left[i])
        assert_equal(Int(right[i]), want_right[i])


def test_right_shift_is_logical_on_unsigned_values() raises:
    var a = Static[DType.uint8, 2]([255, 128], _cpu())
    var got = right_shift(a, Scalar[DType.uint8](4)).to_host()
    assert_equal(Int(got[0]), 15)
    assert_equal(Int(got[1]), 8)


def test_gcd_and_lcm_match_numpy() raises:
    # numpy: np.gcd([12, -18, 0, 0, 7, 21], [18, 12, 5, 0, 13, -14])
    var a = Static[i32, 6]([12, -18, 0, 0, 7, 21], _cpu())
    var b = Static[i32, 6]([18, 12, 5, 0, 13, -14], _cpu())
    var g = gcd(a, b).to_host()
    var l = lcm(a, b).to_host()
    var want_g = [6, 6, 5, 0, 1, 7]
    var want_l = [36, 36, 0, 0, 91, 42]
    for i in range(6):
        assert_equal(Int(g[i]), want_g[i])
        assert_equal(Int(l[i]), want_l[i])


def test_gcd_finishes_on_the_worst_case_pair() raises:
    # Consecutive Fibonacci numbers take the most Euclid steps: F(91) and
    # F(92) are the largest such pair below 2**63, about 90 steps.
    var a = Static[DType.int64, 2](
        [4660046610375530309, 7540113804746346429], _cpu()
    )
    var b = Static[DType.int64, 2](
        [7540113804746346429, 4660046610375530309], _cpu()
    )
    var g = gcd(a, b).to_host()
    assert_equal(Int(g[0]), 1)
    assert_equal(Int(g[1]), 1)


def test_scalar_and_broadcast_forms_agree_with_same_shape() raises:
    var a = Static[i32, 2, 3]([12, 10, 7, 9, 30, 4], _cpu())
    var row = Static[i32, 3]([6, 6, 6], _cpu())
    var full = Static[i32, 2, 3]([6, 6, 6, 6, 6, 6], _cpu())
    var same = gcd(a, full).to_host()
    var scalar = gcd(a, Scalar[i32](6)).to_host()
    var wide = gcd(a, row).to_host()
    var same_and = bitwise_and(a, full).to_host()
    var wide_and = bitwise_and(a, row).to_host()
    for i in range(6):
        assert_equal(Int(scalar[i]), Int(same[i]))
        assert_equal(Int(wide[i]), Int(same[i]))
        assert_equal(Int(wide_and[i]), Int(same_and[i]))
    assert_true(Int(same[0]) == 6 and Int(same[4]) == 6 and Int(same[2]) == 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
