"""Tests for `numax.special.erf` against closed-form values and derivatives."""

from std.testing import TestSuite, assert_almost_equal

from numax import Dual, Plain, erf, erfc

comptime dtype = DType.float64
comptime width = 1
comptime D = Dual[Plain[dtype, width]]


def pv(x: Float64) -> Plain[dtype, width]:
    return Plain[dtype, width](SIMD[dtype, width](x))


def test_erf_at_zero_is_zero() raises:
    var x = Plain[dtype, width](SIMD[dtype, width](0))
    assert_almost_equal(erf(x).v, SIMD[dtype, width](0))


def test_erf_matches_known_values() raises:
    # Reference values from Python's math.erf.
    var x1 = Plain[dtype, width](SIMD[dtype, width](0.5))
    assert_almost_equal(erf(x1).v, SIMD[dtype, width](0.5204998778130465))

    var x2 = Plain[dtype, width](SIMD[dtype, width](1.0))
    assert_almost_equal(erf(x2).v, SIMD[dtype, width](0.8427007929497149))

    var x3 = Plain[dtype, width](SIMD[dtype, width](2.0))
    assert_almost_equal(erf(x3).v, SIMD[dtype, width](0.9953222650189527))


def test_erf_is_odd() raises:
    var x = Plain[dtype, width](SIMD[dtype, width](1.3))
    var neg_x = Plain[dtype, width](SIMD[dtype, width](-1.3))
    assert_almost_equal(erf(neg_x).v, -erf(x).v)


def test_erf_derivative_matches_closed_form() raises:
    # d/dx[erf(x)] = (2/sqrt(pi)) * exp(-x^2); at x=1 that's ~0.4151074974.
    var x = D(pv(1), pv(1))
    var e = erf(x)
    assert_almost_equal(e.deriv.v, SIMD[dtype, width](0.4151074974205947))


def test_erfc_is_one_minus_erf() raises:
    var x = Plain[dtype, width](SIMD[dtype, width](0.75))
    assert_almost_equal(erfc(x).v, SIMD[dtype, width](1) - erf(x).v)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
