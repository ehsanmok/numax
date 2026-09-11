"""Tests for `numax.special.erf` against closed-form values and derivatives,
and for the inverses against SciPy's digits and the round trip."""

from std.math import exp as _exp
from std.testing import TestSuite, assert_almost_equal

from numax import Dual, Plain, erf, erfc, erfcinv, erfinv

comptime dtype = DType.float64
comptime width = 1
comptime D = Dual[Plain[dtype, width]]


def pv(x: Float64) -> Plain[dtype, width]:
    return Plain[dtype, width].constant(x)


def test_erf_at_zero_is_zero() raises:
    var x = Plain[dtype, width].constant(0)
    assert_almost_equal(erf(x).v, SIMD[dtype, width](0))


def test_erf_matches_known_values() raises:
    # Reference values from Python's math.erf.
    var x1 = Plain[dtype, width].constant(0.5)
    assert_almost_equal(erf(x1).v, SIMD[dtype, width](0.5204998778130465))

    var x2 = Plain[dtype, width].constant(1.0)
    assert_almost_equal(erf(x2).v, SIMD[dtype, width](0.8427007929497149))

    var x3 = Plain[dtype, width].constant(2.0)
    assert_almost_equal(erf(x3).v, SIMD[dtype, width](0.9953222650189527))


def test_erf_is_odd() raises:
    var x = Plain[dtype, width].constant(1.3)
    var neg_x = Plain[dtype, width].constant(-1.3)
    assert_almost_equal(erf(neg_x).v, -erf(x).v)


def test_erf_derivative_matches_closed_form() raises:
    # d/dx[erf(x)] = (2/sqrt(pi)) * exp(-x^2); at x=1 that's ~0.4151074974.
    var x = D(pv(1), pv(1))
    var e = erf(x)
    assert_almost_equal(e.deriv.v, SIMD[dtype, width](0.4151074974205947))


def test_erfc_is_one_minus_erf() raises:
    var x = Plain[dtype, width].constant(0.75)
    assert_almost_equal(erfc(x).v, SIMD[dtype, width](1) - erf(x).v)


def test_erfinv_matches_scipy_at_known_points() raises:
    # scipy.special.erfinv, to the digits float64 keeps. The last is past
    # the `w = 5` region split, so it exercises the tail polynomial.
    assert_almost_equal(
        erfinv(pv(0.5)).v, SIMD[dtype, width](0.4769362762044699), atol=1e-8
    )
    assert_almost_equal(
        erfinv(pv(0.9)).v, SIMD[dtype, width](1.1630871536766743), atol=1e-8
    )
    assert_almost_equal(
        erfinv(pv(0.999)).v, SIMD[dtype, width](2.3267537655135246), atol=1e-8
    )


def test_erfinv_inverts_erf_both_ways() raises:
    # The definition rather than a table: erf(erfinv(y)) == y on (-1, 1)
    # and erfinv(erf(x)) == x, to the floor `std.math.erf` sets.
    var ys = [-0.95, -0.5, -0.1, 0.0, 0.3, 0.7, 0.99]
    for i in range(7):
        var y = pv(ys[i])
        assert_almost_equal(erf(erfinv(y)).v, y.v, atol=1e-8)
    var xs = [-2.0, -0.7, 0.2, 1.5]
    for i in range(4):
        var x = pv(xs[i])
        assert_almost_equal(erfinv(erf(x)).v, x.v, atol=1e-7)


def test_erfinv_is_odd_and_zero_at_zero() raises:
    assert_almost_equal(erfinv(pv(0.0)).v, SIMD[dtype, width](0), atol=1e-15)
    assert_almost_equal(erfinv(pv(-0.6)).v, -erfinv(pv(0.6)).v)


def test_erfinv_is_finite_at_the_clamped_endpoint() raises:
    # SciPy returns inf at +-1; a tier-1 kernel clamps the gap `1 - |y|` to
    # 1e-30 instead and says so. mpmath: erfinv(1 - 1e-30) = 8.1494702154...
    # Loose, on purpose: at a gap of 1e-30 the answer is conditioned on
    # `erfc` at 1e-30, and the claim here is finite and odd, not the digits.
    var edge = erfinv(pv(1.0)).v
    assert_almost_equal(edge, SIMD[dtype, width](8.149470215450434), atol=1e-2)
    assert_almost_equal(erfinv(pv(-1.0)).v, -edge)


def test_erfinv_stays_accurate_where_erf_saturates_to_one() raises:
    # The largest float64 below 1 is 1 - 2^-53, where erf(x) == 1.0 exactly
    # for the true x and a Newton step against erf alone would walk away
    # from the answer. mpmath: erfinv(1 - 2^-53) = 5.8635847487551679...
    var y = pv(1.0 - 1.1102230246251565e-16)
    assert_almost_equal(
        erfinv(y).v, SIMD[dtype, width](5.863584748755168), atol=1e-9
    )


def test_erfinv_derivative_matches_closed_form() raises:
    # d/dy erfinv(y) = sqrt(pi)/2 * exp(erfinv(y)^2); at y = 0.5 that is
    # 0.8862269 * exp(0.4769363^2) = 1.1125...
    var y = D(pv(0.5), pv(1))
    var x = erfinv(y)
    var expected = 0.8862269254527580 * _exp(0.4769362762044699**2)
    assert_almost_equal(x.deriv.v, SIMD[dtype, width](expected), atol=1e-7)


def test_erfcinv_is_erfinv_of_one_minus_y() raises:
    # erfcinv(0.5) == erfinv(0.5), and erfcinv(1) == 0.
    assert_almost_equal(erfcinv(pv(0.5)).v, erfinv(pv(0.5)).v)
    assert_almost_equal(erfcinv(pv(1.0)).v, SIMD[dtype, width](0), atol=1e-15)
    assert_almost_equal(
        erfc(erfcinv(pv(0.3))).v, SIMD[dtype, width](0.3), atol=1e-8
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
