"""Tests for `numax.interpolate`."""

from std.collections import Array
from std.math import cos as cos_f64
from std.math import exp as exp_f64
from std.math import sin as sin_f64
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Dual, FloatLike, Plain
from numax.interpolate import (
    Chebyshev,
    CubicSpline,
    chebyshev_eval,
    chebyshev_fit,
    cubic_spline_eval,
    cubic_spline_moments,
    horner,
)

comptime dtype = DType.float64
comptime width = 1
comptime P = Plain[dtype, width]
comptime D = Dual[P]
comptime PI = 3.141592653589793


def pv(x: Float64) -> P:
    return P.constant(x)


def s(x: P) -> Float64:
    return Float64(x.v)


def sine[U: FloatLike](x: U) -> U:
    return x.sin()


def bell[U: FloatLike](x: U) -> U:
    return (-(x * x)).exp()


def runge[U: FloatLike](x: U) -> U:
    """`1/(1 + 25x^2)` -- the classic example where equally spaced
    polynomial interpolation diverges and Chebyshev doesn't."""
    return U.one() / (U.one() + U.constant(25.0) * x * x)


# --------------------------------------------------------------- horner


def test_horner_matches_the_expanded_polynomial() raises:
    # 1 + 2x + 3x^2
    var c = Array[P, 3](fill=pv(0.0))
    c[0] = pv(1.0)
    c[1] = pv(2.0)
    c[2] = pv(3.0)
    for x in [-2.0, 0.0, 0.5, 3.0]:
        assert_almost_equal(
            s(horner[P, 3](c, pv(x))),
            1.0 + 2.0 * x + 3.0 * x * x,
            atol=1e-13,
        )


def test_horner_of_a_constant_is_constant() raises:
    var c = Array[P, 1](fill=pv(7.0))
    assert_almost_equal(s(horner[P, 1](c, pv(100.0))), 7.0)


def test_horner_is_differentiable() raises:
    # d/dx[1 + 2x + 3x^2] = 2 + 6x.
    var c = Array[D, 3](fill=D.constant(0.0))
    c[0] = D.constant(1.0)
    c[1] = D.constant(2.0)
    c[2] = D.constant(3.0)
    var x = 1.5
    var result = horner[D, 3](c, D(pv(x), pv(1.0)))
    assert_almost_equal(s(result.deriv), 2.0 + 6.0 * x, atol=1e-13)


# --------------------------------------------------------------- splines


def sine_spline[n: Int]() -> Tuple[Array[P, n], Array[P, n], Float64]:
    """`sin` sampled at `n` knots across `[0, pi]`, with its moments."""
    var h = PI / Float64(n - 1)
    var y = Array[P, n](fill=pv(0.0))
    for i in range(n):
        y[i] = pv(sin_f64(Float64(i) * h))
    var moments = cubic_spline_moments[P, n](y, pv(h))
    return (y^, moments^, h)


def test_spline_passes_through_every_knot() raises:
    comptime n = 9
    var built = sine_spline[n]()
    for i in range(n):
        var x = Float64(i) * built[2]
        assert_almost_equal(
            s(
                cubic_spline_eval[P, n](
                    built[0], built[1], pv(0.0), pv(built[2]), pv(x)
                )
            ),
            s(built[0][i]),
            atol=1e-12,
        )


def test_natural_spline_has_zero_end_moments() raises:
    comptime n = 9
    var built = sine_spline[n]()
    assert_almost_equal(s(built[1][0]), 0.0)
    assert_almost_equal(s(built[1][n - 1]), 0.0)


def test_spline_approximates_between_knots() raises:
    comptime n = 17
    var built = sine_spline[n]()
    var worst = 0.0
    for i in range(200):
        var x = PI * Float64(i) / 199.0
        var got = s(
            cubic_spline_eval[P, n](
                built[0], built[1], pv(0.0), pv(built[2]), pv(x)
            )
        )
        worst = max(worst, abs(got - sin_f64(x)))
    assert_true(worst < 1e-5)


def test_spline_error_falls_like_the_fourth_power_of_spacing() raises:
    # Doubling the knot count should cut the error by roughly 16.
    var coarse = _worst_spline_error[5]()
    var fine = _worst_spline_error[9]()
    var ratio = coarse / fine
    assert_true(ratio > 8.0 and ratio < 30.0)


def _worst_spline_error[n: Int]() -> Float64:
    var built = sine_spline[n]()
    var worst = 0.0
    for i in range(101):
        var x = PI * Float64(i) / 100.0
        var got = s(
            cubic_spline_eval[P, n](
                built[0], built[1], pv(0.0), pv(built[2]), pv(x)
            )
        )
        worst = max(worst, abs(got - sin_f64(x)))
    return worst


def test_spline_of_a_cubic_is_exact_in_the_interior() raises:
    # A natural spline forces zero second derivative at the ends, which a
    # general cubic doesn't have -- so it can't reproduce one exactly. A
    # *linear* function has zero second derivative everywhere and is
    # reproduced exactly, which is the sharper statement.
    comptime n = 6
    var h = 0.5
    var y = Array[P, n](fill=pv(0.0))
    for i in range(n):
        y[i] = pv(3.0 * Float64(i) * h + 1.0)
    var moments = cubic_spline_moments[P, n](y, pv(h))

    for i in range(50):
        var x = 2.5 * Float64(i) / 49.0
        assert_almost_equal(
            s(cubic_spline_eval[P, n](y, moments, pv(0.0), pv(h), pv(x))),
            3.0 * x + 1.0,
            atol=1e-12,
        )


def test_spline_clamps_outside_the_grid() raises:
    comptime n = 9
    var built = sine_spline[n]()
    var left = s(
        cubic_spline_eval[P, n](
            built[0], built[1], pv(0.0), pv(built[2]), pv(-5.0)
        )
    )
    var right = s(
        cubic_spline_eval[P, n](
            built[0], built[1], pv(0.0), pv(built[2]), pv(10.0)
        )
    )
    assert_almost_equal(left, s(built[0][0]), atol=1e-12)
    assert_almost_equal(right, s(built[0][n - 1]), atol=1e-12)


def test_spline_is_differentiable() raises:
    # d/dx of the spline should track cos(x), since the spline tracks sin.
    comptime n = 17
    var built = sine_spline[n]()
    for x in [0.4, 1.2, 2.7]:
        var seeded = cubic_spline_eval[D, n](
            _to_dual[n](built[0]),
            _to_dual[n](built[1]),
            D.constant(0.0),
            D.constant(built[2]),
            D(pv(x), pv(1.0)),
        )
        assert_almost_equal(s(seeded.deriv), cos_f64(x), atol=1e-4)


def _to_dual[n: Int](values: Array[P, n]) -> Array[D, n]:
    var out = Array[D, n](fill=D.constant(0.0))
    for i in range(n):
        out[i] = D(values[i].copy(), pv(0.0))
    return out^


# ------------------------------------------------------------- Chebyshev


def test_chebyshev_reproduces_a_smooth_function() raises:
    comptime n = 16
    var coefficients = chebyshev_fit[P, f=sine, n_terms=n](pv(-2.0), pv(2.0))
    for i in range(50):
        var x = -2.0 + 4.0 * Float64(i) / 49.0
        assert_almost_equal(
            s(chebyshev_eval[P, n](coefficients, pv(-2.0), pv(2.0), pv(x))),
            sin_f64(x),
            atol=1e-12,
        )


def test_chebyshev_reproduces_a_gaussian() raises:
    # Wider interval than the `sin` fit above, so more terms are needed
    # for the same accuracy: 28 terms lands near 4e-8 here, where 20 would
    # only reach 5e-5.
    comptime n = 28
    var coefficients = chebyshev_fit[P, f=bell, n_terms=n](pv(-3.0), pv(3.0))
    for i in range(40):
        var x = -3.0 + 6.0 * Float64(i) / 39.0
        assert_almost_equal(
            s(chebyshev_eval[P, n](coefficients, pv(-3.0), pv(3.0), pv(x))),
            exp_f64(-x * x),
            atol=1e-6,
        )


def test_chebyshev_is_exact_at_its_own_nodes() raises:
    # The discrete fit interpolates exactly at the Chebyshev nodes, whether
    # or not it's accurate between them -- true even for the Runge
    # function, which converges only slowly here.
    comptime n = 8
    var coefficients = chebyshev_fit[P, f=runge, n_terms=n](pv(-1.0), pv(1.0))
    for j in range(n):
        var node = cos_f64(PI * (Float64(j) + 0.5) / Float64(n))
        assert_almost_equal(
            s(chebyshev_eval[P, n](coefficients, pv(-1.0), pv(1.0), pv(node))),
            1.0 / (1.0 + 25.0 * node * node),
            atol=1e-12,
        )


def test_chebyshev_converges_geometrically_on_the_runge_function() raises:
    # Equally spaced polynomial interpolation *diverges* on this function.
    # Chebyshev converges at a rate set by the distance from [-1,1] to the
    # poles at +/- i/5, which predicts 1.2198^-N. Doubling 16 terms to 32
    # should then improve the error by 1.2198^16, about 24x -- and it does,
    # which is a sharper check than any single error threshold.
    var coarse = _worst_runge_error[16]()
    var fine = _worst_runge_error[32]()
    var ratio = coarse / fine
    assert_true(ratio > 15.0 and ratio < 40.0)
    assert_true(fine < 5e-3)


def _worst_runge_error[n: Int]() -> Float64:
    var coefficients = chebyshev_fit[P, f=runge, n_terms=n](pv(-1.0), pv(1.0))
    var worst = 0.0
    for i in range(101):
        var x = -1.0 + 0.02 * Float64(i)
        var got = s(
            chebyshev_eval[P, n](coefficients, pv(-1.0), pv(1.0), pv(x))
        )
        worst = max(worst, abs(got - 1.0 / (1.0 + 25.0 * x * x)))
    return worst


def test_chebyshev_evaluation_is_differentiable() raises:
    comptime n = 16
    var coefficients = chebyshev_fit[P, f=sine, n_terms=n](pv(-2.0), pv(2.0))
    var dual_coefficients = _to_dual[n](coefficients)
    for x in [-1.3, 0.2, 1.7]:
        var seeded = chebyshev_eval[D, n](
            dual_coefficients,
            D.constant(-2.0),
            D.constant(2.0),
            D(pv(x), pv(1.0)),
        )
        assert_almost_equal(s(seeded.deriv), cos_f64(x), atol=1e-10)


def test_simd_lanes_evaluate_independently() raises:
    comptime w = 4
    comptime PW = Plain[dtype, w]
    comptime n = 3
    var c = Array[PW, n](fill=PW.constant(0.0))
    c[0] = PW.constant(1.0)
    c[1] = PW.constant(2.0)
    c[2] = PW.constant(3.0)
    var x = PW(SIMD[dtype, w](-1.0, 0.0, 1.0, 2.0))
    var result = horner[PW, n](c, x)
    var expected = SIMD[dtype, w](2.0, 1.0, 6.0, 17.0)
    for lane in range(w):
        assert_almost_equal(
            Float64(result.v[lane]), Float64(expected[lane]), atol=1e-13
        )


def test_cubic_spline_object_matches_the_two_call_form() raises:
    var y = Array[P, 5](fill=P.constant(0.0))
    for i in range(5):
        y[i] = P.constant(Float64(i) * Float64(i))
    var x0 = P.constant(0.0)
    var h = P.one()

    var spline = CubicSpline[P, 5](y, x0, h)
    var moments = cubic_spline_moments[P, 5](y, h)
    for k in range(9):
        var x = P.constant(Float64(k) * 0.5)
        assert_almost_equal(
            spline(x).v,
            cubic_spline_eval[P, 5](y, moments, x0, h, x).v,
        )


def test_chebyshev_object_matches_the_two_call_form() raises:
    var a = P.constant(0.0)
    var b = P.one()
    var series = Chebyshev[P, 16].fit[sine](a, b)
    var coefficients = chebyshev_fit[P, f=sine, n_terms=16](a, b)
    for k in range(5):
        var x = P.constant(Float64(k) * 0.25)
        assert_almost_equal(
            series(x).v, chebyshev_eval[P, 16](coefficients, a, b, x).v
        )


def test_chebyshev_object_approximates_the_function_it_fit() raises:
    var series = Chebyshev[P, 16].fit[sine](P.constant(0.0), P.one())
    assert_almost_equal(series(P.constant(0.5)).v, sin_f64(0.5), atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
