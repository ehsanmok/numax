"""Tests for `numax.optimize`, the tier-2 (converge-to-tolerance) module.

Two things are being checked, and they are different in kind. That each
routine lands on the right answer is ordinary correctness. That the gradient
BFGS uses is *exact* -- not a finite difference -- is the reason the module
exists, so `test_ad_gradient_beats_a_central_difference` measures the
difference rather than taking it on faith.
"""

from std.collections import Array
from std.math import cos as cos_f64, exp as exp_f64, pi, sin as sin_f64
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Dual, FloatLike, Gradient, Plain
from numax.optimize import bfgs, brentq, newton_tol

comptime P = Plain[DType.float64, 1]


def cos_minus_x[U: FloatLike](x: U) -> U:
    """`cos(x) - x`, whose root is the Dottie number."""
    return x.cos() + (-x)


def cubic[U: FloatLike](x: U) -> U:
    """`x**3 - 2x - 5`, Wallis's cubic. Single real root near 2.0946."""
    return x * x * x + (-(U.constant(2.0) * x)) + (-U.constant(5.0))


def sin_shifted[U: FloatLike](x: U) -> U:
    """`sin(x)`, bracketed so the root is exactly `pi`."""
    return x.sin()


def rosenbrock[U: FloatLike](v: Array[U, 2]) -> U:
    """The standard test: `(1-x)**2 + 100*(y - x*x)**2`, minimum 0 at
    (1, 1), reached along a curved narrow valley that defeats naive
    steepest descent."""
    var a = U.one() + (-v[0])
    var b = v[1] + (-(v[0] * v[0]))
    return a * a + U.constant(100.0) * b * b


def quadratic_bowl[U: FloatLike](v: Array[U, 3]) -> U:
    """`(x-1)**2 + 2*(y+2)**2 + 3*(z-3)**2`: minimum 0 at (1, -2, 3), and
    exactly quadratic, so BFGS should reach it in very few steps."""
    var dx = v[0] + (-U.one())
    var dy = v[1] + U.constant(2.0)
    var dz = v[2] + (-U.constant(3.0))
    return dx * dx + U.constant(2.0) * dy * dy + U.constant(3.0) * dz * dz


def steep_exponential[U: FloatLike](v: Array[U, 1]) -> U:
    """`exp(3x) - 3x`. Its derivative `3*exp(3x) - 3` is zero at `x = 0`,
    which is therefore the minimum. Chosen because the derivative grows
    fast enough away from it that a badly scaled step would show up."""
    return (U.constant(3.0) * v[0]).exp() + (-(U.constant(3.0) * v[0]))


# ------------------------------------------------------------------
# newton_tol
# ------------------------------------------------------------------


def test_newton_tol_finds_the_dottie_number() raises:
    var result = newton_tol[cos_minus_x](0.5)
    assert_true(result.converged)
    assert_almost_equal(result.x, 0.7390851332151607, atol=1e-14)


def test_newton_tol_converges_quadratically() raises:
    # A quadratically converging method should need very few steps from a
    # decent start. If this ever needs many, the derivative is wrong.
    var result = newton_tol[cos_minus_x](0.5)
    assert_true(result.iterations <= 8)


def test_newton_tol_solves_wallis_cubic() raises:
    var result = newton_tol[cubic](2.0)
    assert_true(result.converged)
    assert_almost_equal(result.x, 2.0945514815423265, atol=1e-12)


def test_newton_tol_reports_failure_on_a_zero_derivative() raises:
    # `sin` at exactly pi/2 has zero derivative, so the first step is
    # undefined. The point is that it says so rather than returning a
    # plausible-looking number.
    var result = newton_tol[sin_shifted](pi / 2)
    assert_true(not result.converged)


def test_newton_tol_agrees_with_the_tier_one_sibling() raises:
    # numax.optimize.newton runs a fixed 20 iterations of the same
    # mathematics. On a well-behaved root the two must land in the same
    # place -- that is what makes them siblings rather than alternatives.
    from numax.optimize import newton

    var fixed = newton[P, cos_minus_x](P.constant(0.5))
    var adaptive = newton_tol[cos_minus_x](0.5)
    assert_almost_equal(Float64(fixed.v), adaptive.x, atol=1e-12)


# ------------------------------------------------------------------
# brentq
# ------------------------------------------------------------------


def test_brentq_finds_the_dottie_number() raises:
    var result = brentq[cos_minus_x](0.0, 2.0)
    assert_true(result.converged)
    assert_almost_equal(result.x, 0.7390851332151607, atol=1e-12)


def test_brentq_beats_pure_bisection_in_iteration_count() raises:
    # Bisecting [0, 2] down to 1e-12 takes ~41 halvings. Brent's
    # interpolation should get there in well under half that; if this
    # regresses to ~41 the step-acceptance test has started rejecting
    # every interpolated point, which is a silent performance bug that
    # correctness assertions alone would not catch.
    var result = brentq[cos_minus_x](0.0, 2.0)
    assert_true(result.iterations < 20)


def test_brentq_finds_pi() raises:
    var result = brentq[sin_shifted](3.0, 3.5)
    assert_true(result.converged)
    assert_almost_equal(result.x, pi, atol=1e-12)


def test_brentq_returns_an_exact_endpoint_root_immediately() raises:
    var result = brentq[sin_shifted](0.0, 3.0)
    assert_true(result.converged)
    assert_almost_equal(result.x, 0.0)
    assert_true(result.iterations == 0)


def test_brentq_reports_failure_on_a_bracket_without_a_sign_change() raises:
    # Every guarantee Brent's method has follows from the bracket
    # straddling a root, so an unbracketed call has to be refused rather
    # than answered.
    var result = brentq[cos_minus_x](2.0, 3.0)
    assert_true(not result.converged)


def test_brentq_and_newton_tol_agree() raises:
    var bracketed = brentq[cubic](2.0, 3.0)
    var newtonian = newton_tol[cubic](2.5)
    assert_almost_equal(bracketed.x, newtonian.x, atol=1e-11)


# ------------------------------------------------------------------
# bfgs
# ------------------------------------------------------------------


def test_bfgs_minimizes_a_quadratic_bowl() raises:
    var start = Array[Float64, 3](fill=0)
    start[0] = -5.0
    start[1] = 5.0
    start[2] = -5.0
    var result = bfgs[3, quadratic_bowl](start)
    assert_true(result.converged)
    assert_almost_equal(result.x[0], 1.0, atol=1e-6)
    assert_almost_equal(result.x[1], -2.0, atol=1e-6)
    assert_almost_equal(result.x[2], 3.0, atol=1e-6)


def test_bfgs_minimizes_rosenbrock_from_the_standard_start() raises:
    # (-1.2, 1) is the classic starting point. Reaching (1, 1) means the
    # method followed the curved valley rather than bouncing across it.
    var start = Array[Float64, 2](fill=0)
    start[0] = -1.2
    start[1] = 1.0
    var result = bfgs[2, rosenbrock](start)
    assert_true(result.converged)
    assert_almost_equal(result.x[0], 1.0, atol=1e-6)
    assert_almost_equal(result.x[1], 1.0, atol=1e-6)


def test_bfgs_drives_the_objective_essentially_to_zero() raises:
    # Rosenbrock's minimum value is exactly 0. With an exact gradient the
    # residual lands near float64's own floor, not near the ~1e-16 a
    # finite-difference gradient would stall at.
    var start = Array[Float64, 2](fill=0)
    start[0] = -1.2
    start[1] = 1.0
    var result = bfgs[2, rosenbrock](start)
    assert_true(result.f_x < 1e-18)


def test_bfgs_reports_the_gradient_norm_it_converged_on() raises:
    var start = Array[Float64, 2](fill=0)
    start[0] = 0.5
    start[1] = 0.5
    var result = bfgs[2, rosenbrock](start)
    assert_true(result.converged)
    assert_true(result.grad_norm < 1e-8)


def test_bfgs_starting_at_the_minimum_converges_immediately() raises:
    var start = Array[Float64, 3](fill=0)
    start[0] = 1.0
    start[1] = -2.0
    start[2] = 3.0
    var result = bfgs[3, quadratic_bowl](start)
    assert_true(result.converged)
    assert_true(result.iterations == 0)


def test_bfgs_minimizes_a_one_variable_objective() raises:
    var start = Array[Float64, 1](fill=0)
    start[0] = 1.0
    var result = bfgs[1, steep_exponential](start)
    assert_true(result.converged)
    assert_almost_equal(result.x[0], 0.0, atol=1e-7)


# ------------------------------------------------------------------
# The reason the module exists
# ------------------------------------------------------------------


def test_ad_gradient_beats_a_central_difference() raises:
    """The claim in `numax.optimize`'s docstring, measured.

    A central difference of step `h` carries a truncation error of order
    `h**2` and a cancellation error of order `eps/h`; the best achievable
    accuracy is therefore around `eps**(2/3)`, and no choice of `h` does
    better. Forward-mode AD has neither term. This checks that the exact
    gradient is closer to the analytic derivative than the best central
    difference is, by a wide enough margin that it is not measuring noise.
    """
    comptime G = Gradient[P, 2]

    # d/dx of Rosenbrock at (0.5, 0.5), analytically:
    #   -2*(1-x) - 400*x*(y - x*x) = -2*0.5 - 400*0.5*(0.5 - 0.25) = -51.0
    comptime x0 = 0.5
    comptime y0 = 0.5
    comptime analytic = -51.0

    var seeded = Array[G, 2](fill=G.constant(0.0))
    seeded[0] = G.variable(x0, 0)
    seeded[1] = G.variable(y0, 1)
    var exact = Float64(rosenbrock[G](seeded^).grad[0].v)

    def plain_at(x: Float64, y: Float64) -> Float64:
        var v = Array[P, 2](fill=P.constant(0.0))
        v[0] = P.constant(x)
        v[1] = P.constant(y)
        return Float64(rosenbrock[P](v^).v)

    # Sweep h and keep the *best* central difference, so the comparison is
    # against finite differencing at its most favourable rather than a
    # strawman step size.
    var best_fd_error = 1e30
    var h = 1e-2
    for _ in range(12):
        var fd = (plain_at(x0 + h, y0) - plain_at(x0 - h, y0)) / (2 * h)
        var err = abs(fd - analytic)
        if err < best_fd_error:
            best_fd_error = err
        h = h / 10

    var ad_error = abs(exact - analytic)
    # AD should be exact to the last bit here; the best FD cannot be.
    assert_true(ad_error <= 1e-13)
    assert_true(ad_error < best_fd_error)


def test_bfgs_objective_is_the_same_function_evaluated_three_ways() raises:
    """`rosenbrock` is one `FloatLike` kernel. This calls it at `Plain`
    (value), `Dual` (one derivative) and `Gradient` (both partials) and
    checks the three agree -- the property that lets the optimizer pick
    whichever conformer it needs without a second implementation."""
    comptime D = Dual[P]
    comptime G = Gradient[P, 2]

    var plain_v = Array[P, 2](fill=P.constant(0.0))
    plain_v[0] = P.constant(0.5)
    plain_v[1] = P.constant(0.5)
    var value_plain = Float64(rosenbrock[P](plain_v^).v)

    var dual_v = Array[D, 2](fill=D.constant(0.0))
    dual_v[0] = D(P.constant(0.5), P.one())
    dual_v[1] = D.constant(0.5)
    var evaluated_dual = rosenbrock[D](dual_v^)

    var grad_v = Array[G, 2](fill=G.constant(0.0))
    grad_v[0] = G.variable(0.5, 0)
    grad_v[1] = G.variable(0.5, 1)
    var evaluated_grad = rosenbrock[G](grad_v^)

    assert_almost_equal(value_plain, Float64(evaluated_dual.value.v))
    assert_almost_equal(value_plain, Float64(evaluated_grad.value.v))
    # Dual seeded on x tracks the same partial Gradient reports at index 0.
    assert_almost_equal(
        Float64(evaluated_dual.deriv.v), Float64(evaluated_grad.grad[0].v)
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
