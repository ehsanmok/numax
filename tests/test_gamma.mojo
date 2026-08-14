"""Tests for `ember.gamma` against closed-form values and derivatives."""

from std.testing import TestSuite, assert_almost_equal, assert_true

from ember import Dual, Plain, gamma, gammainc, gammaincc, lgamma

comptime dtype = DType.float64
comptime width = 1
comptime D = Dual[Plain[dtype, width]]


def pv(x: Float64) -> Plain[dtype, width]:
    return Plain[dtype, width](SIMD[dtype, width](x))


def test_gamma_at_positive_integers_matches_factorial() raises:
    # Gamma(n) = (n-1)!.
    assert_almost_equal(
        gamma(Plain[dtype, width](SIMD[dtype, width](1))).v,
        SIMD[dtype, width](1),
    )
    assert_almost_equal(
        gamma(Plain[dtype, width](SIMD[dtype, width](5))).v,
        SIMD[dtype, width](24),
        atol=1e-9,
    )


def test_gamma_at_one_half_is_sqrt_pi() raises:
    var x = Plain[dtype, width](SIMD[dtype, width](0.5))
    assert_almost_equal(
        gamma(x).v, SIMD[dtype, width](1.7724538509055159), atol=1e-9
    )


def test_lgamma_matches_ln_of_gamma() raises:
    var x = Plain[dtype, width](SIMD[dtype, width](5))
    assert_almost_equal(
        lgamma(x).v, SIMD[dtype, width](3.1780538303479458), atol=1e-9
    )


def test_lgamma_derivative_matches_digamma() raises:
    # d/dx[lgamma(x)] = digamma(x); digamma(5) = -euler_gamma + (1 + 1/2 +
    # 1/3 + 1/4).
    var x = D(pv(5), pv(1))
    var l = lgamma(x)
    assert_almost_equal(
        l.deriv.v, SIMD[dtype, width](1.5061176684318003), atol=1e-8
    )


def test_gamma_derivative_matches_gamma_times_digamma() raises:
    # d/dx[Gamma(x)] = Gamma(x) * digamma(x); at x=2, Gamma(2)=1 and
    # digamma(2) = 1 - euler_gamma.
    var x = D(pv(2), pv(1))
    var g = gamma(x)
    assert_almost_equal(
        g.deriv.v, SIMD[dtype, width](0.42278433509846713), atol=1e-8
    )


def test_gammainc_matches_closed_form_for_integer_a() raises:
    # P(2, x) = 1 - (1+x)*exp(-x) for integer a=2 -- a closed form to check
    # the series against independently of `gammainc`'s own implementation.
    var a = Plain[dtype, width](SIMD[dtype, width](2))
    var x = Plain[dtype, width](SIMD[dtype, width](3))
    assert_almost_equal(
        gammainc(a, x).v, SIMD[dtype, width](0.8008517265285442), atol=1e-9
    )


def test_gammainc_at_a_equals_one_is_exponential_cdf() raises:
    # P(1, x) = 1 - exp(-x).
    var a = Plain[dtype, width](SIMD[dtype, width](1))
    var x = Plain[dtype, width](SIMD[dtype, width](1))
    assert_almost_equal(
        gammainc(a, x).v, SIMD[dtype, width](0.6321205588285577), atol=1e-9
    )


def test_gammaincc_is_one_minus_gammainc() raises:
    var a = Plain[dtype, width](SIMD[dtype, width](1))
    var x = Plain[dtype, width](SIMD[dtype, width](1))
    assert_almost_equal(
        gammaincc(a, x).v,
        SIMD[dtype, width](1) - gammainc(a, x).v,
        atol=1e-12,
    )


def test_gammainc_approaches_one_for_large_x() raises:
    var a = Plain[dtype, width](SIMD[dtype, width](2))
    var x = Plain[dtype, width](SIMD[dtype, width](50))
    assert_true(Float64(gammainc(a, x).v) > 0.9999)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
