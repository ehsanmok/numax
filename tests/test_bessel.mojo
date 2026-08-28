"""Tests for `numax.bessel` against known values and `std.math` references."""

from std.math import j0 as j0_f64
from std.math import j1 as j1_f64
from std.math import y0 as y0_f64
from std.math import y1 as y1_f64
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Dual, Plain, bessel_j0, bessel_j1, bessel_y0, bessel_y1

comptime dtype = DType.float64
comptime width = 1
comptime D = Dual[Plain[dtype, width]]


def v(x: Float64) -> Plain[dtype, width]:
    return Plain[dtype, width](SIMD[dtype, width](x))


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


def test_bessel_j0_far_branch_matches_std_math() raises:
    # x > 3 now reaches the A&S 9.4.3 asymptotic branch instead of the
    # near-branch polynomial extrapolated out of its own domain.
    for x64 in [3.5, 5.0, 8.0, 10.0, 20.0, 50.0, 100.0]:
        assert_almost_equal(
            bessel_j0(v(x64)).v, SIMD[dtype, width](j0_f64(x64)), atol=1e-7
        )


def test_bessel_j0_blend_is_continuous_at_the_boundary() raises:
    # x = 3 is where the branchless blend switches sides -- check both
    # sides land close to each other and to the reference there.
    assert_almost_equal(
        bessel_j0(v(2.999999)).v,
        SIMD[dtype, width](j0_f64(2.999999)),
        atol=1e-6,
    )
    assert_almost_equal(
        bessel_j0(v(3.000001)).v,
        SIMD[dtype, width](j0_f64(3.000001)),
        atol=1e-6,
    )


def test_bessel_j1_is_odd() raises:
    var x = v(1.7)
    var neg_x = v(-1.7)
    assert_almost_equal(bessel_j1(x).v, -bessel_j1(neg_x).v)


def test_bessel_j1_at_zero_is_zero() raises:
    assert_almost_equal(bessel_j1(v(0)).v, SIMD[dtype, width](0))


def test_bessel_j1_matches_std_math_near_and_far() raises:
    for x64 in [0.5, 1.0, 2.0, 2.9, 3.0, 4.0, 8.0, 20.0, -1.5, -10.0]:
        assert_almost_equal(
            bessel_j1(v(x64)).v, SIMD[dtype, width](j1_f64(x64)), atol=1e-7
        )


def test_bessel_y0_matches_std_math_near_and_far() raises:
    for x64 in [0.1, 0.5, 1.0, 2.0, 2.9, 3.0, 4.0, 8.0, 20.0, 50.0]:
        assert_almost_equal(
            bessel_y0(v(x64)).v, SIMD[dtype, width](y0_f64(x64)), atol=1e-6
        )


def test_bessel_y1_matches_std_math_near_and_far() raises:
    for x64 in [0.1, 0.5, 1.0, 2.0, 2.9, 3.0, 4.0, 8.0, 20.0, 50.0]:
        assert_almost_equal(
            bessel_y1(v(x64)).v, SIMD[dtype, width](y1_f64(x64)), atol=1e-6
        )


def test_bessel_j0_derivative_is_negative_j1() raises:
    # d/dx[J0(x)] = -J1(x) -- a standard identity, checked here via `Dual`
    # across both branches.
    for x64 in [1.0, 2.0, 5.0, 10.0]:
        var x = D(v(x64), v(1))
        var j0d = bessel_j0(x)
        assert_almost_equal(j0d.deriv.v, -bessel_j1(v(x64)).v, atol=1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
