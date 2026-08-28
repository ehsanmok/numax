"""Tests for `numax.gamma` against closed-form values and derivatives."""

from std.math import cos as cos_f64
from std.math import exp as exp_f64
from std.math import gamma as gamma_f64
from std.math import lgamma as lgamma_f64
from std.math import sin as sin_f64
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Dual, Plain, digamma, gamma, gammainc, gammaincc, lgamma

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


def test_gamma_reflects_to_negative_non_integers() raises:
    # Checked against `std.math.gamma` (CPU-only, used here only as a test
    # reference -- see `numax/gamma.mojo`'s module docstring for why
    # `numax.gamma` doesn't just call it directly).
    for x64 in [-0.5, -2.5, -1.5, -4.3, -10.5, -0.1]:
        var x = pv(x64)
        assert_almost_equal(
            gamma(x).v, SIMD[dtype, width](gamma_f64(x64)), atol=1e-8
        )


def test_lgamma_reflects_to_negative_non_integers() raises:
    for x64 in [-0.5, -2.5, -1.5, -4.3, -10.5, -0.1]:
        var x = pv(x64)
        assert_almost_equal(
            lgamma(x).v, SIMD[dtype, width](lgamma_f64(x64)), atol=1e-8
        )


def test_gamma_agrees_at_the_reflection_boundary() raises:
    # x = 0.5 is the boundary the branchless blend picks a side at --
    # check both sides of it land on the same, correct value.
    assert_almost_equal(
        gamma(pv(0.5)).v,
        SIMD[dtype, width](1.7724538509055159),
        atol=1e-9,
    )
    assert_almost_equal(
        gamma(pv(0.4999999)).v,
        SIMD[dtype, width](gamma_f64(0.4999999)),
        atol=1e-6,
    )


def test_gamma_derivative_matches_gamma_times_digamma_for_negative_x() raises:
    # d/dx[Gamma(x)] = Gamma(x) * digamma(x); digamma(-0.5) =
    # 2 - 2*ln(2) - euler_gamma.
    var x = D(pv(-0.5), pv(1))
    var g = gamma(x)
    var digamma_neg_half = 0.03648997397857652
    assert_almost_equal(
        g.deriv.v,
        SIMD[dtype, width](gamma_f64(-0.5) * digamma_neg_half),
        atol=1e-6,
    )


def test_gammainc_matches_recurrence_for_negative_a() raises:
    # P(a,x) - P(a+1,x) = x^a * exp(-x) / Gamma(a+1) -- a standard identity
    # that holds for any non-pole `a`, checked here at a negative one to
    # confirm `gammainc`'s series (and its new `_gamma_sign` correction)
    # stays consistent past the domain it was originally scoped to.
    var a = pv(-0.5)
    var a_plus_1 = pv(0.5)
    var x = pv(2.0)
    var lhs = gammainc(a, x).v - gammainc(a_plus_1, x).v
    var rhs = (
        Float64(2.0) ** Float64(-0.5) * Float64(exp_f64(-2.0)) / gamma_f64(0.5)
    )
    assert_almost_equal(lhs, SIMD[dtype, width](rhs), atol=1e-8)


def test_digamma_matches_closed_form_values() raises:
    # psi(1) = -euler_gamma, psi(2) = 1 - euler_gamma,
    # psi(1/2) = -euler_gamma - 2*ln(2), psi(5) = -euler_gamma + 1 + 1/2 +
    # 1/3 + 1/4 -- all exact expressions, not reference-library output.
    comptime euler_gamma = 0.5772156649015329
    assert_almost_equal(
        digamma(pv(1)).v, SIMD[dtype, width](-euler_gamma), atol=1e-8
    )
    assert_almost_equal(
        digamma(pv(2)).v, SIMD[dtype, width](1.0 - euler_gamma), atol=1e-8
    )
    assert_almost_equal(
        digamma(pv(0.5)).v,
        SIMD[dtype, width](-euler_gamma - 2.0 * 0.6931471805599453),
        atol=1e-8,
    )
    assert_almost_equal(
        digamma(pv(5)).v,
        SIMD[dtype, width](1.5061176684318003),
        atol=1e-8,
    )


def test_digamma_satisfies_the_recurrence() raises:
    # psi(x+1) = psi(x) + 1/x, for positive and negative x alike.
    for x64 in [0.3, 1.7, 4.2, 11.0, -0.4, -2.7]:
        var lhs = Float64(digamma(pv(x64 + 1.0)).v)
        var rhs = Float64(digamma(pv(x64)).v) + 1.0 / x64
        assert_almost_equal(lhs, rhs, atol=1e-7)


def test_digamma_reflects_to_negative_arguments() raises:
    # psi(1-x) - psi(x) = pi * cot(pi*x), the reflection formula. `digamma`
    # has no reflection logic of its own -- it inherits `lgamma`'s, because
    # `Dual` differentiates through whichever side that blend selects.
    comptime pi = 3.141592653589793
    for x64 in [0.2, 0.35, 0.8]:
        var lhs = Float64(digamma(pv(1.0 - x64)).v) - Float64(
            digamma(pv(x64)).v
        )
        var rhs = pi * (cos_f64(pi * x64) / sin_f64(pi * x64))
        assert_almost_equal(lhs, rhs, atol=1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
