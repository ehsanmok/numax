"""Tests for `numax.special.lambertw`/`lambertw_m1` against known values and the
defining identity `w*exp(w) = x`, across both real branches.
"""

from std.math import exp
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Dual, Plain, lambertw, lambertw_m1

comptime dtype = DType.float64
comptime width = 1
comptime D = Dual[Plain[dtype, width]]

comptime _NEG_ONE_OVER_E = -0.36787944117144233


def pv(x: Float64) -> Plain[dtype, width]:
    return Plain[dtype, width].constant(x)


def test_lambertw_at_zero_is_zero() raises:
    var x = Plain[dtype, width].constant(0)
    assert_almost_equal(lambertw(x).v, SIMD[dtype, width](0), atol=1e-12)


def test_lambertw_at_one_is_omega_constant() raises:
    var x = Plain[dtype, width].constant(1)
    assert_almost_equal(
        lambertw(x).v, SIMD[dtype, width](0.5671432904097838), atol=1e-9
    )


def test_lambertw_at_e_is_one() raises:
    var x = Plain[dtype, width].constant(2.718281828459045)
    assert_almost_equal(lambertw(x).v, SIMD[dtype, width](1), atol=1e-9)


def test_lambertw_satisfies_w_exp_w_equals_x() raises:
    # The defining identity, checked directly rather than against a
    # separately-sourced reference value.
    for x_raw in [0.0, 0.5, 1.0, 5.0, 20.0]:
        var x = Plain[dtype, width].constant(x_raw)
        var w = lambertw(x)
        var check = w.v * exp(w.v)
        assert_almost_equal(check, SIMD[dtype, width](x_raw), atol=1e-8)


def test_lambertw_derivative_matches_closed_form() raises:
    # dW/dx = W(x) / (x * (1 + W(x))), for x != 0.
    var x = Dual[Plain[dtype, width]](
        Plain[dtype, width].constant(1),
        Plain[dtype, width].constant(1),
    )
    var w = lambertw(x)
    assert_almost_equal(
        w.deriv.v, SIMD[dtype, width](0.36189625663488917), atol=1e-8
    )


def test_lambertw_extends_to_negative_x_down_to_branch_point() raises:
    # x in [-1/e, 0) is the newly-extended part of `lambertw`'s (W0's)
    # domain -- checked against the defining identity, since these land in
    # [-1, 0), the same range where W0 is >= -1 by definition.
    for x_raw in [-0.01, -0.1, -0.2, -0.3, -0.36, _NEG_ONE_OVER_E + 1e-9]:
        var x = pv(x_raw)
        var w = lambertw(x)
        assert_true(w.v[0] >= -1.0)
        var check = w.v * exp(w.v)
        assert_almost_equal(check, SIMD[dtype, width](x_raw), atol=1e-8)


def test_lambertw_m1_satisfies_w_exp_w_equals_x() raises:
    for x_raw in [-0.01, -0.1, -0.2, -0.3, -0.36, _NEG_ONE_OVER_E + 1e-9]:
        var x = pv(x_raw)
        var w = lambertw_m1(x)
        assert_true(w.v[0] <= -1.0)
        var check = w.v * exp(w.v)
        assert_almost_equal(check, SIMD[dtype, width](x_raw), atol=1e-8)


def test_lambertw_m1_matches_known_values() raises:
    # Cross-checked against a from-scratch bisection reference in Python,
    # since there's no `std.math` Lambert W to check against directly. The
    # defining identity `w*exp(w) = x` is also checked separately below,
    # which needs no reference values at all.
    assert_almost_equal(
        lambertw_m1(pv(-0.1)).v, SIMD[dtype, width](-3.5771520640), atol=1e-8
    )
    assert_almost_equal(
        lambertw_m1(pv(-0.01)).v, SIMD[dtype, width](-6.4727751244), atol=1e-8
    )


def test_lambertw_m1_stays_accurate_at_extreme_small_x() raises:
    # Far from the branch point, where the branch-point series alone (no
    # blend) would badly under-seed Halley's method -- see this module's
    # docstring on why `lambertw_m1` blends two seeds instead of one.
    for x_raw in [-1e-6, -1e-15, -1e-30]:
        var x = pv(x_raw)
        var w = lambertw_m1(x)
        var check = w.v * exp(w.v)
        assert_almost_equal(check, SIMD[dtype, width](x_raw), atol=1e-8)


def test_lambertw_and_lambertw_m1_agree_at_the_branch_point() raises:
    # Both branches meet at x = -1/e, w = -1.
    var x = pv(_NEG_ONE_OVER_E + 1e-9)
    assert_almost_equal(lambertw(x).v, SIMD[dtype, width](-1), atol=1e-4)
    assert_almost_equal(lambertw_m1(x).v, SIMD[dtype, width](-1), atol=1e-4)


def test_lambertw_m1_derivative_matches_closed_form() raises:
    # Same closed form as W0's -- dW/dx = W(x) / (x * (1 + W(x))) doesn't
    # depend on which branch `W` came from.
    var x = D(pv(-0.1), pv(1))
    var w = lambertw_m1(x)
    var expected = w.value.v / (x.value.v * (1.0 + w.value.v))
    assert_almost_equal(w.deriv.v, expected, atol=1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
