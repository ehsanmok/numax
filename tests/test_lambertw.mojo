"""Tests for `ember.lambertw` against known values and its defining identity."""

from std.math import exp
from std.testing import TestSuite, assert_almost_equal

from ember import Dual, Plain, lambertw

comptime dtype = DType.float64
comptime width = 1


def test_lambertw_at_zero_is_zero() raises:
    var x = Plain[dtype, width](SIMD[dtype, width](0))
    assert_almost_equal(lambertw(x).v, SIMD[dtype, width](0), atol=1e-12)


def test_lambertw_at_one_is_omega_constant() raises:
    var x = Plain[dtype, width](SIMD[dtype, width](1))
    assert_almost_equal(
        lambertw(x).v, SIMD[dtype, width](0.5671432904097838), atol=1e-9
    )


def test_lambertw_at_e_is_one() raises:
    var x = Plain[dtype, width](SIMD[dtype, width](2.718281828459045))
    assert_almost_equal(lambertw(x).v, SIMD[dtype, width](1), atol=1e-9)


def test_lambertw_satisfies_w_exp_w_equals_x() raises:
    # The defining identity, checked directly rather than against a
    # separately-sourced reference value.
    for x_raw in [0.0, 0.5, 1.0, 5.0, 20.0]:
        var x = Plain[dtype, width](SIMD[dtype, width](x_raw))
        var w = lambertw(x)
        var check = w.v * exp(w.v)
        assert_almost_equal(check, SIMD[dtype, width](x_raw), atol=1e-8)


def test_lambertw_derivative_matches_closed_form() raises:
    # dW/dx = W(x) / (x * (1 + W(x))), for x != 0.
    var x = Dual[Plain[dtype, width]](
        Plain[dtype, width](SIMD[dtype, width](1)),
        Plain[dtype, width](SIMD[dtype, width](1)),
    )
    var w = lambertw(x)
    assert_almost_equal(
        w.deriv.v, SIMD[dtype, width](0.36189625663488917), atol=1e-8
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
