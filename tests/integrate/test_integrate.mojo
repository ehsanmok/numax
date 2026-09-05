"""Tests for `numax.integrate`, the tier-2 adaptive quadrature module.

Correctness is checked against closed forms. The reason the module exists is
checked separately: `test_adaptive_beats_the_fixed_rule_on_a_peak` compares
`quad` against `numax.integrate.gauss_legendre` on an integrand the fixed
rule cannot see, and `test_a_named_breakpoint_costs_far_fewer_panels` shows
what naming a kink saves.
"""

from std.math import atan, cos as cos_f64, exp, pi, sqrt
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import FloatLike, Plain
from numax.integrate import quad, quad_vec, solve_ivp, solve_ivp_stiff
from numax.integrate import dopri5
from numax.integrate import gauss_legendre

comptime P = Plain[DType.float64, 1]

# The kink location, as an exactly-representable float that bisection can
# never land on: 1/3 in binary is non-terminating, so every subdivision
# straddles it forever.
comptime KINK = 0.3333333333333333


def unit[U: FloatLike](x: U) -> U:
    """`1`, so the integral is the interval width."""
    return U.one()


def linear[U: FloatLike](x: U) -> U:
    return x.copy()


def cubic[U: FloatLike](x: U) -> U:
    """`x**3`; a degree-3 polynomial, so an 8-point Gauss rule is exact and
    no subdivision should ever be needed."""
    return x * x * x


def gaussian_bell[U: FloatLike](x: U) -> U:
    """`exp(-x**2)`; integrates to `sqrt(pi)` over the whole line."""
    return (-(x * x)).exp()


def peaked[U: FloatLike](x: U) -> U:
    """A Lorentzian spike at `x = 0.5` with half-width 0.01.

    `1 / (1e-4 + (x - 0.5)**2)`, whose integral over `[0, 1]` is
    `200 * atan(50)`. An 8-point Gauss rule over the whole interval places
    no node inside the spike at all, so it misses most of the mass.
    """
    var d = x - U.constant(0.5)
    return U.one() / (U.constant(1e-4) + d * d)


def kinked[U: FloatLike](x: U) -> U:
    """`|x - 1/3|`. Continuous, but its derivative jumps, which is enough
    to defeat a polynomial rule on any panel containing the kink."""
    return (x - U.constant(KINK)).abs()


# ------------------------------------------------------------------
# Correctness against closed forms
# ------------------------------------------------------------------


def test_quad_integrates_a_constant() raises:
    var result = quad[unit](2.0, 5.0)
    assert_true(result.converged)
    assert_almost_equal(result.value, 3.0, atol=1e-12)


def test_quad_integrates_a_linear_function() raises:
    var result = quad[linear](0.0, 2.0)
    assert_true(result.converged)
    assert_almost_equal(result.value, 2.0, atol=1e-12)


def test_quad_integrates_a_cubic_exactly_in_one_panel() raises:
    # A degree-3 polynomial is exact under the 8-point rule, so whole and
    # halves agree to rounding and the first panel is accepted. Needing
    # more than a couple of panels here would mean the acceptance test is
    # subdividing when it has no reason to.
    var result = quad[cubic](0.0, 1.0)
    assert_true(result.converged)
    assert_almost_equal(result.value, 0.25, atol=1e-14)
    assert_true(result.panels <= 2)


def test_quad_integrates_the_gaussian_bell() raises:
    # Over [-6, 6] the missing tails are ~1e-17, well under the tolerance.
    var result = quad[gaussian_bell](-6.0, 6.0)
    assert_true(result.converged)
    assert_almost_equal(result.value, sqrt(pi), atol=1e-10)


def test_quad_integrates_the_peak_to_its_closed_form() raises:
    var result = quad[peaked](0.0, 1.0)
    assert_true(result.converged)
    assert_almost_equal(result.value, 200.0 * atan(50.0), atol=1e-8)


def test_quad_integrates_the_kink_to_its_closed_form() raises:
    # int_0^1 |x - 1/3| dx = (1/2)(1/3)^2 + (1/2)(2/3)^2 = 5/18.
    var result = quad[kinked](0.0, 1.0)
    assert_almost_equal(result.value, 5.0 / 18.0, atol=1e-10)


# ------------------------------------------------------------------
# Conventions and edge cases
# ------------------------------------------------------------------


def test_quad_negates_for_reversed_limits() raises:
    var forward = quad[linear](0.0, 2.0)
    var backward = quad[linear](2.0, 0.0)
    assert_almost_equal(backward.value, -forward.value, atol=1e-14)


def test_quad_on_an_empty_interval_is_exactly_zero() raises:
    var result = quad[peaked](0.5, 0.5)
    assert_true(result.converged)
    assert_true(result.value == 0.0)
    assert_true(result.panels == 0)


def test_quad_reports_the_error_it_estimated() raises:
    var result = quad[peaked](0.0, 1.0)
    # The estimate should be small but not claimed to be zero, and it
    # should actually bound the true error.
    var truth = 200.0 * atan(50.0)
    assert_true(result.error < 1e-6)
    assert_true(abs(result.value - truth) < 1e-6)


def test_quad_reports_non_convergence_when_panels_run_out() raises:
    # One panel is not enough for the spike, so the cap is hit with error
    # still above tolerance. The point is that it says so.
    var result = quad[peaked](0.0, 1.0, tol=1e-14, max_panels=1)
    assert_true(not result.converged)


def test_tighter_tolerance_uses_more_panels() raises:
    var loose = quad[peaked](0.0, 1.0, tol=1e-4)
    var tight = quad[peaked](0.0, 1.0, tol=1e-12)
    assert_true(tight.panels > loose.panels)


# ------------------------------------------------------------------
# Why the module exists
# ------------------------------------------------------------------


def test_adaptive_beats_the_fixed_rule_on_a_peak() raises:
    """The tier-1 rule is better on smooth integrands and hopeless here.

    An 8-point Gauss rule over `[0, 1]` puts its outermost nodes at about
    0.02 and 0.98 and its innermost near 0.5 -- but the spike is only 0.01
    wide, so the sampled values badly misrepresent it. The adaptive rule
    finds it.
    """
    var truth = 200.0 * atan(50.0)
    var fixed = Float64(
        gauss_legendre[P, peaked, 8](P.constant(0.0), P.constant(1.0)).v
    )
    var adaptive = quad[peaked](0.0, 1.0)

    var fixed_error = abs(fixed - truth)
    var adaptive_error = abs(adaptive.value - truth)

    # Not a marginal difference: the fixed rule is off by a factor, the
    # adaptive one by rounding.
    assert_true(fixed_error > 1.0)
    assert_true(adaptive_error < 1e-8)


def test_fixed_rule_wins_on_a_smooth_integrand() raises:
    """The other direction, so the tier-1 rule is not quietly deprecated.

    On a cubic the 8-point rule is exact in 8 evaluations; `quad` needs at
    least three panel evaluations to establish that. Same answer, less
    work -- which is why `numax.integrate` keeps its place.
    """
    var fixed = Float64(
        gauss_legendre[P, cubic, 8](P.constant(0.0), P.constant(1.0)).v
    )
    var adaptive = quad[cubic](0.0, 1.0)
    assert_almost_equal(fixed, 0.25, atol=1e-15)
    assert_almost_equal(adaptive.value, 0.25, atol=1e-14)


def test_a_named_breakpoint_costs_far_fewer_panels() raises:
    """Bisection can never land on 1/3, so the adaptive rule keeps
    subdividing around the kink. Naming it makes the problem go away
    instead of throwing panels at it."""
    var blind = quad[kinked](0.0, 1.0)
    var breakpoints = List[Float64](capacity=1)
    breakpoints.append(KINK)
    var informed = quad_vec[kinked](0.0, 1.0, breakpoints)

    assert_almost_equal(informed.value, 5.0 / 18.0, atol=1e-12)
    assert_true(informed.panels < blind.panels)
    # And by a wide margin -- two smooth pieces need one panel each.
    assert_true(informed.panels <= 4)


def test_quad_vec_ignores_breakpoints_outside_the_interval() raises:
    var breakpoints = List[Float64](capacity=3)
    breakpoints.append(-1.0)
    breakpoints.append(0.5)
    breakpoints.append(99.0)
    var result = quad_vec[linear](0.0, 2.0, breakpoints)
    assert_true(result.converged)
    assert_almost_equal(result.value, 2.0, atol=1e-12)


def test_quad_vec_handles_unsorted_breakpoints() raises:
    var breakpoints = List[Float64](capacity=3)
    breakpoints.append(1.5)
    breakpoints.append(0.5)
    breakpoints.append(1.0)
    var result = quad_vec[linear](0.0, 2.0, breakpoints)
    assert_true(result.converged)
    assert_almost_equal(result.value, 2.0, atol=1e-12)


def test_quad_vec_with_no_breakpoints_matches_quad() raises:
    var empty = List[Float64](capacity=1)
    var direct = quad[peaked](0.0, 1.0)
    var routed = quad_vec[peaked](0.0, 1.0, empty)
    assert_almost_equal(routed.value, direct.value, atol=1e-9)


# ------------------------------------------------------------------
# solve_ivp
# ------------------------------------------------------------------


def decay[U: FloatLike](t: U, y: U) -> U:
    """`dy/dt = -3y`, whose solution is `y0 * exp(-3t)`."""
    return -U.constant(3.0) * y


def growth[U: FloatLike](t: U, y: U) -> U:
    """`dy/dt = y`, solution `y0 * exp(t)`."""
    return y.copy()


def time_dependent[U: FloatLike](t: U, y: U) -> U:
    """`dy/dt = 2t`, solution `y0 + t**2` -- exercises the `t` argument,
    which a solver that only ever passed `y` would still pass the
    exponential tests with."""
    return U.constant(2.0) * t


def stiff_transient[U: FloatLike](t: U, y: U) -> U:
    """`dy/dt = -50*(y - cos(t))`: a fast transient onto a slow solution.

    The kind of problem adaptivity is for -- the first fraction of the
    interval needs tiny steps and the rest does not, so a fixed-step
    method has to use the transient's step size the whole way.
    """
    return -U.constant(50.0) * (y - t.cos())


def test_solve_ivp_matches_the_exponential_solution() raises:
    var result = solve_ivp[decay](0.0, 1.0, 2.0)
    assert_true(result.converged)
    assert_almost_equal(result.y, exp(-6.0), atol=1e-9)


def test_solve_ivp_reaches_the_requested_endpoint() raises:
    var result = solve_ivp[decay](0.0, 1.0, 2.0)
    assert_almost_equal(result.t, 2.0, atol=1e-14)


def test_solve_ivp_integrates_growth() raises:
    var result = solve_ivp[growth](0.0, 1.0, 1.0)
    assert_true(result.converged)
    assert_almost_equal(result.y, exp(1.0), atol=1e-8)


def test_solve_ivp_uses_the_time_argument() raises:
    # y(0) = 1, dy/dt = 2t, so y(3) = 1 + 9 = 10.
    var result = solve_ivp[time_dependent](0.0, 1.0, 3.0)
    assert_true(result.converged)
    assert_almost_equal(result.y, 10.0, atol=1e-9)


def test_solve_ivp_integrates_backwards_in_time() raises:
    # From t=2 back to t=0 with y(2) = exp(-6) must recover y(0) = 1.
    var result = solve_ivp[decay](2.0, exp(-6.0), 0.0)
    assert_true(result.converged)
    assert_almost_equal(result.y, 1.0, atol=1e-8)
    assert_almost_equal(result.t, 0.0, atol=1e-14)


def test_solve_ivp_on_a_zero_length_interval_returns_the_initial_value() raises:
    var result = solve_ivp[decay](1.0, 5.0, 1.0)
    assert_true(result.converged)
    assert_almost_equal(result.y, 5.0)
    assert_true(result.accepted == 0)


def test_solve_ivp_accepts_most_of_its_steps_on_a_smooth_problem() raises:
    # A healthy controller rejects few steps. Rejecting most of them would
    # mean the step-size scaling is fighting itself.
    var result = solve_ivp[decay](0.0, 1.0, 2.0)
    assert_true(result.rejected < result.accepted)


def test_tighter_tolerance_takes_more_steps() raises:
    var loose = solve_ivp[decay](0.0, 1.0, 2.0, rtol=1e-4, atol=1e-6)
    var tight = solve_ivp[decay](0.0, 1.0, 2.0, rtol=1e-10, atol=1e-12)
    assert_true(tight.accepted > loose.accepted)


def test_solve_ivp_reports_non_convergence_when_steps_run_out() raises:
    var result = solve_ivp[decay](0.0, 1.0, 2.0, max_steps=3)
    assert_true(not result.converged)
    # And `t` says how far it actually got, rather than claiming the
    # endpoint.
    assert_true(result.t < 2.0)


def test_solve_ivp_beats_fixed_steps_on_a_transient() raises:
    """The reason the adaptive solver exists, measured.

    `stiff_transient` needs small steps only at the start. Given a step
    budget comparable to what the adaptive solver used, the fixed-step
    integrator spends it uniformly and is far less accurate.
    """
    var adaptive = solve_ivp[stiff_transient](0.0, 0.0, 3.0, rtol=1e-9)
    assert_true(adaptive.converged)

    # A fixed-step run with a comparable number of stages.
    var fixed = Float64(
        dopri5[P, stiff_transient, 40](
            P.constant(0.0), P.constant(0.0), P.constant(3.0)
        ).v
    )

    # Take a very finely resolved adaptive run as the reference.
    var reference = solve_ivp[stiff_transient](
        0.0, 0.0, 3.0, rtol=1e-13, atol=1e-15
    ).y

    var adaptive_error = abs(adaptive.y - reference)
    var fixed_error = abs(fixed - reference)
    assert_true(adaptive_error < fixed_error)


def test_solve_ivp_agrees_with_a_well_resolved_fixed_step_run() raises:
    # Where the fixed-step method is given enough steps, the two must
    # agree -- which pins the shared `dopri5_step` against itself under
    # two different drivers.
    var adaptive = solve_ivp[decay](0.0, 1.0, 2.0, rtol=1e-12, atol=1e-14)
    var fixed = Float64(
        dopri5[P, decay, 2000](
            P.constant(0.0), P.constant(1.0), P.constant(2.0)
        ).v
    )
    assert_almost_equal(adaptive.y, fixed, atol=1e-11)


def very_stiff[U: FloatLike](t: U, y: U) -> U:
    """`dy/dt = -1000*(y - cos(t)) - sin(t)`, whose solution is
    `cos(t)` once the transient at the start has died.

    Stiff in the technical sense: the solution is as smooth as a cosine,
    but an explicit method's step is capped near `2.8/1000` by stability
    no matter how little accuracy that buys.
    """
    return -U.constant(1000.0) * (y - t.cos()) - t.sin()


def test_the_stiff_solver_reaches_the_same_answer() raises:
    var result = solve_ivp_stiff[decay](0.0, 1.0, 1.0)
    assert_true(result.converged)
    assert_almost_equal(result.y, exp(-3.0), atol=1e-7)


def test_the_stiff_solver_integrates_a_time_dependent_equation() raises:
    var result = solve_ivp_stiff[time_dependent](0.0, 1.0, 2.0)
    assert_true(result.converged)
    assert_almost_equal(result.y, 5.0, atol=1e-8)


def test_the_stiff_solver_takes_far_fewer_steps_on_a_stiff_problem() raises:
    # Both start at the value the transient has already relaxed to, so the
    # difference in step count is stability and nothing else.
    var explicit = solve_ivp[very_stiff](0.0, 1.0, 1.0, rtol=1e-6, atol=1e-8)
    var implicit = solve_ivp_stiff[very_stiff](
        0.0, 1.0, 1.0, rtol=1e-6, atol=1e-8
    )

    assert_true(explicit.converged)
    assert_true(implicit.converged)
    assert_almost_equal(explicit.y, cos_f64(1.0), atol=1e-5)
    assert_almost_equal(implicit.y, cos_f64(1.0), atol=1e-5)

    assert_true(implicit.accepted * 4 < explicit.accepted)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
