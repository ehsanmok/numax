"""Tests for `ember.bessel` against known `J0` values."""

from std.testing import TestSuite, assert_almost_equal

from ember import Plain, bessel_j0

comptime dtype = DType.float64
comptime width = 1


def test_bessel_j0_at_zero_is_one() raises:
    var x = Plain[dtype, width](SIMD[dtype, width](0))
    assert_almost_equal(bessel_j0(x).v, SIMD[dtype, width](1))


def test_bessel_j0_matches_known_values() raises:
    # Reference values from standard Bessel function tables.
    var x1 = Plain[dtype, width](SIMD[dtype, width](1))
    assert_almost_equal(
        bessel_j0(x1).v, SIMD[dtype, width](0.7651976865579666), atol=1e-7
    )

    var x2 = Plain[dtype, width](SIMD[dtype, width](2))
    assert_almost_equal(
        bessel_j0(x2).v, SIMD[dtype, width](0.22389077914123567), atol=1e-7
    )

    # J0 crosses zero near x=2.405, so J0(3) is negative.
    var x3 = Plain[dtype, width](SIMD[dtype, width](3))
    assert_almost_equal(
        bessel_j0(x3).v, SIMD[dtype, width](-0.26005195490193343), atol=1e-7
    )


def test_bessel_j0_is_even() raises:
    var x = Plain[dtype, width](SIMD[dtype, width](2.5))
    var neg_x = Plain[dtype, width](SIMD[dtype, width](-2.5))
    assert_almost_equal(bessel_j0(x).v, bessel_j0(neg_x).v)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
