"""Tests for `numax.optimize`.

The kernels solved here are the ordinary trait-generic `def[U: FloatLike]`
shape every other module in `numax` is written in -- none of them supply a
derivative, which is the point.
"""

from std.math import cos as cos_f64
from std.math import sqrt as sqrt_f64
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Dual, FloatLike, Plain, bisection, halley, newton

comptime dtype = DType.float64
comptime width = 1
comptime P = Plain[dtype, width]
comptime D = Dual[P]


def pv(x: Float64) -> P:
    return P.constant(x)


def cubic_minus_two[U: FloatLike](x: U) -> U:
    """`x^3 - 2`, whose positive root is the cube root of 2."""
    return x * x * x - U.constant(2.0)


def cos_minus_x[U: FloatLike](x: U) -> U:
    """`cos(x) - x`, the Dottie-number equation."""
    return x.cos() - x


def exp_minus_three[U: FloatLike](x: U) -> U:
    """`exp(x) - 3`, whose root is `ln(3)`."""
    return x.exp() - U.constant(3.0)


def sin_kernel[U: FloatLike](x: U) -> U:
    return x.sin()


def test_newton_finds_the_cube_root_of_two() raises:
    var root = newton[f=cubic_minus_two](pv(1.0))
    assert_almost_equal(
        root.v, SIMD[dtype, width](2.0 ** (1.0 / 3.0)), atol=1e-14
    )


def test_newton_finds_the_dottie_number() raises:
    var root = newton[f=cos_minus_x](pv(1.0))
    assert_almost_equal(
        root.v, SIMD[dtype, width](0.7390851332151607), atol=1e-14
    )
    # The defining property, independent of the reference value.
    assert_almost_equal(cos_f64(Float64(root.v)), Float64(root.v), atol=1e-14)


def test_newton_finds_a_logarithm() raises:
    var root = newton[f=exp_minus_three](pv(1.0))
    assert_almost_equal(
        root.v, SIMD[dtype, width](1.0986122886681098), atol=1e-14
    )


def test_newton_converges_from_a_poor_guess() raises:
    var root = newton[f=cubic_minus_two](pv(50.0))
    assert_almost_equal(
        root.v, SIMD[dtype, width](2.0 ** (1.0 / 3.0)), atol=1e-12
    )


def test_newton_solves_each_simd_lane_independently() raises:
    # Four lanes, four different starting guesses, one call -- the reason
    # the iteration count is fixed rather than tested for convergence.
    comptime w4 = 4
    comptime P4 = Plain[dtype, w4]
    var x0 = P4(SIMD[dtype, w4](0.5, 1.0, 5.0, 20.0))
    var roots = newton[f=cubic_minus_two](x0)
    var expected = 2.0 ** (1.0 / 3.0)
    for lane in range(w4):
        assert_almost_equal(Float64(roots.v[lane]), expected, atol=1e-12)


def test_newton_at_a_critical_point_stays_finite() raises:
    # sin'(x) = 0 exactly at x = pi/2, so the first step divides by a
    # floored denominator instead of zero. The result need not be a root;
    # it must not be NaN.
    var root = newton[f=sin_kernel, num_iters=3](pv(1.5707963267948966))
    assert_true(Float64(root.v) == Float64(root.v))


def test_halley_matches_newton() raises:
    var by_halley = halley[f=cubic_minus_two](pv(1.0))
    var by_newton = newton[f=cubic_minus_two](pv(1.0))
    assert_almost_equal(by_halley.v, by_newton.v, atol=1e-14)


def test_halley_converges_faster_than_newton() raises:
    # Cubic vs. quadratic convergence: at a deliberately small iteration
    # count from the same guess, Halley should be closer.
    var expected = 2.0 ** (1.0 / 3.0)
    var h = Float64(halley[f=cubic_minus_two, num_iters=3](pv(5.0)).v)
    var n = Float64(newton[f=cubic_minus_two, num_iters=3](pv(5.0)).v)
    assert_true(abs(h - expected) < abs(n - expected))


def test_halley_on_a_transcendental() raises:
    var root = halley[f=cos_minus_x](pv(1.0))
    assert_almost_equal(
        root.v, SIMD[dtype, width](0.7390851332151607), atol=1e-14
    )


def test_bisection_finds_the_cube_root_of_two() raises:
    var root = bisection[f=cubic_minus_two](pv(0.0), pv(4.0))
    assert_almost_equal(
        root.v, SIMD[dtype, width](2.0 ** (1.0 / 3.0)), atol=1e-12
    )


def test_bisection_works_with_the_bracket_either_way_around() raises:
    # The blend keys on a sign change, not on lo < hi, so a reversed
    # bracket converges to the same root.
    var forward = bisection[f=cos_minus_x](pv(0.0), pv(2.0))
    var reversed_bracket = bisection[f=cos_minus_x](pv(2.0), pv(0.0))
    assert_almost_equal(forward.v, reversed_bracket.v, atol=1e-12)


def test_bisection_needs_no_derivative_and_cannot_diverge() raises:
    # A bracket Newton would struggle with from its midpoint, since the
    # function is flat there.
    var root = bisection[f=exp_minus_three](pv(-10.0), pv(10.0))
    assert_almost_equal(
        root.v, SIMD[dtype, width](1.0986122886681098), atol=1e-9
    )


def test_bisection_bisects_lanes_independently() raises:
    comptime w4 = 4
    comptime P4 = Plain[dtype, w4]
    var lo = P4(SIMD[dtype, w4](0.0, 0.5, 1.0, -1.0))
    var hi = P4(SIMD[dtype, w4](4.0, 3.0, 2.0, 8.0))
    var roots = bisection[f=cubic_minus_two](lo, hi)
    var expected = 2.0 ** (1.0 / 3.0)
    for lane in range(w4):
        assert_almost_equal(Float64(roots.v[lane]), expected, atol=1e-12)


def test_solving_at_dual_differentiates_the_root() raises:
    # The solvers are `FloatLike`-generic in T, so solving at `Dual`
    # propagates a derivative through the iteration. Here the root of
    # `x^3 - 2` is a constant, so its derivative with respect to the
    # starting guess must be zero -- a real check that the fixed-point
    # iteration's derivative also converges, not just its value.
    var root = newton[f=cubic_minus_two](D(pv(1.0), pv(1.0)))
    assert_almost_equal(
        root.value.v, SIMD[dtype, width](2.0 ** (1.0 / 3.0)), atol=1e-14
    )
    assert_almost_equal(root.deriv.v, SIMD[dtype, width](0.0), atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
