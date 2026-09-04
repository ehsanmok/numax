"""Tests for `numax.special` against closed-form values and derivatives."""

from std.testing import TestSuite, assert_almost_equal

from numax import Dual, Plain, gaussian, sigmoid, swish, tanh

comptime dtype = DType.float64
comptime width = 1
comptime D = Dual[Plain[dtype, width]]


def pv(x: Float64) -> Plain[dtype, width]:
    return Plain[dtype, width].constant(x)


def test_gaussian_value() raises:
    # gaussian(0) = exp(0) = 1.
    var x = Plain[dtype, width].constant(0)
    assert_almost_equal(gaussian(x).v, SIMD[dtype, width](1))


def test_gaussian_derivative() raises:
    # d/dx[exp(-x^2)] = -2x*exp(-x^2); at x=1 that's -2/e.
    var x = D(pv(1), pv(1))
    var g = gaussian(x)
    assert_almost_equal(g.deriv.v, SIMD[dtype, width](-2.0 / 2.718281828459045))


def test_sigmoid_at_zero() raises:
    var x = Plain[dtype, width].constant(0)
    assert_almost_equal(sigmoid(x).v, SIMD[dtype, width](0.5))


def test_sigmoid_derivative_is_p_times_one_minus_p() raises:
    var x = D(pv(0.5), pv(1))
    var s = sigmoid(x)
    assert_almost_equal(
        s.deriv.v, s.value.v * (SIMD[dtype, width](1) - s.value.v)
    )


def test_swish_is_x_times_sigmoid() raises:
    var x = Plain[dtype, width].constant(1.3)
    var s = sigmoid(x)
    var sw = swish(x)
    assert_almost_equal(sw.v, x.v * s.v)


def test_tanh_matches_sigmoid_identity() raises:
    # tanh(x) = 2*sigmoid(2x) - 1.
    var x = SIMD[dtype, width](0.8)
    var t = tanh(Plain[dtype, width](x)).v
    var s2x = sigmoid(Plain[dtype, width](x + x)).v
    assert_almost_equal(t, SIMD[dtype, width](2) * s2x - SIMD[dtype, width](1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
