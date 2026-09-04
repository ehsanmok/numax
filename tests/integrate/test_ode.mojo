"""Tests for `numax.integrate`.

Every problem here has a closed-form solution, so the expected values are
exact rather than reference-library output, and the convergence-order
checks compare the integrator against itself at two step counts.
"""

from std.collections import Array
from std.math import cos as cos_f64
from std.math import exp as exp_f64
from std.math import sin as sin_f64
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Dual, FloatLike, Plain
from numax.integrate import dopri5, dopri5_with_error, rk4, rk4_system

comptime dtype = DType.float64
comptime width = 1
comptime P = Plain[dtype, width]
comptime D = Dual[P]


def pv(x: Float64) -> P:
    return P.constant(x)


def decay[U: FloatLike](t: U, y: U) -> U:
    """`dy/dt = -2y`, so `y(t) = y0*exp(-2t)`."""
    return -(U.constant(2.0) * y)


def logistic_blowup[U: FloatLike](t: U, y: U) -> U:
    """`dy/dt = y^2`, so `y(t) = 1/(1-t)` from `y(0) = 1`."""
    return y * y


def linear_in_t[U: FloatLike](t: U, y: U) -> U:
    """`dy/dt = t`, so `y(t) = y0 + t^2/2`; exact for both integrators."""
    return t.copy()


def oscillator[U: FloatLike](t: U, y: Array[U, 2]) -> Array[U, 2]:
    """`y'' = -y` as a first-order system, so `y(t) = cos(t)` from
    `y(0) = 1`, `y'(0) = 0`."""
    var out = Array[U, 2](fill=U.constant(0.0))
    out[0] = y[1].copy()
    out[1] = -y[0]
    return out^


def parametrized_decay[U: FloatLike](t: U, y: Array[U, 2]) -> Array[U, 2]:
    """`dy/dt = -k*y` with `k` carried as a second state component obeying
    `dk/dt = 0` -- the augmentation trick that turns a parameter
    sensitivity into an initial-condition sensitivity."""
    var out = Array[U, 2](fill=U.constant(0.0))
    out[0] = -(y[1] * y[0])
    out[1] = U.constant(0.0)
    return out^


def test_rk4_matches_an_exponential() raises:
    var y = rk4[f=decay, num_steps=200](pv(0.0), pv(1.0), pv(1.0))
    assert_almost_equal(y.v, SIMD[dtype, width](exp_f64(-2.0)), atol=1e-11)


def test_rk4_is_fourth_order() raises:
    # Halving the step should cut the error by about 2^4 = 16.
    var exact = exp_f64(-2.0)
    var coarse = abs(
        Float64(rk4[f=decay, num_steps=10](pv(0.0), pv(1.0), pv(1.0)).v) - exact
    )
    var fine = abs(
        Float64(rk4[f=decay, num_steps=20](pv(0.0), pv(1.0), pv(1.0)).v) - exact
    )
    var ratio = coarse / fine
    assert_true(ratio > 12.0 and ratio < 20.0)


def test_dopri5_is_fifth_order() raises:
    # Halving the step should cut the error by about 2^5 = 32. The step
    # counts here are larger than the fourth-order check above because the
    # ratio approaches 32 from above -- it's still 44 at 5-vs-10 steps, and
    # only settles into the asymptotic regime around 20.
    var exact = exp_f64(-2.0)
    var coarse = abs(
        Float64(dopri5[f=decay, num_steps=20](pv(0.0), pv(1.0), pv(1.0)).v)
        - exact
    )
    var fine = abs(
        Float64(dopri5[f=decay, num_steps=40](pv(0.0), pv(1.0), pv(1.0)).v)
        - exact
    )
    var ratio = coarse / fine
    assert_true(ratio > 28.0 and ratio < 40.0)


def test_dopri5_beats_rk4_at_the_same_step_count() raises:
    var exact = exp_f64(-2.0)
    var r = abs(
        Float64(rk4[f=decay, num_steps=8](pv(0.0), pv(1.0), pv(1.0)).v) - exact
    )
    var d = abs(
        Float64(dopri5[f=decay, num_steps=8](pv(0.0), pv(1.0), pv(1.0)).v)
        - exact
    )
    assert_true(d < r)


def test_both_integrators_are_exact_on_a_polynomial() raises:
    # dy/dt = t integrates to t^2/2, which every method of order >= 2
    # reproduces exactly regardless of step count.
    var r = rk4[f=linear_in_t, num_steps=3](pv(0.0), pv(1.0), pv(2.0))
    var d = dopri5[f=linear_in_t, num_steps=3](pv(0.0), pv(1.0), pv(2.0))
    assert_almost_equal(r.v, SIMD[dtype, width](3.0), atol=1e-13)
    assert_almost_equal(d.v, SIMD[dtype, width](3.0), atol=1e-13)


def test_nonlinear_right_hand_side() raises:
    # dy/dt = y^2 from y(0)=1 gives 1/(1-t); at t=0.5 that's 2.
    var y = rk4[f=logistic_blowup, num_steps=500](pv(0.0), pv(1.0), pv(0.5))
    assert_almost_equal(y.v, SIMD[dtype, width](2.0), atol=1e-9)


def test_backwards_integration_inverts_forwards() raises:
    # Integrating to t1 and back to t0 should return the starting value.
    var forward = rk4[f=decay, num_steps=200](pv(0.0), pv(1.0), pv(1.0))
    var back = rk4[f=decay, num_steps=200](pv(1.0), forward, pv(0.0))
    assert_almost_equal(back.v, SIMD[dtype, width](1.0), atol=1e-10)


def test_error_estimate_is_positive_and_shrinks_with_step_count() raises:
    var coarse = dopri5_with_error[f=decay, num_steps=10](
        pv(0.0), pv(1.0), pv(1.0)
    )
    var fine = dopri5_with_error[f=decay, num_steps=40](
        pv(0.0), pv(1.0), pv(1.0)
    )
    assert_true(Float64(coarse[1].v) > 0.0)
    assert_true(Float64(fine[1].v) < Float64(coarse[1].v))


def test_error_estimate_tracks_the_true_error() raises:
    # Not a bound -- just that it lands in the right order of magnitude,
    # which is what makes it usable for choosing a step count.
    var result = dopri5_with_error[f=decay, num_steps=10](
        pv(0.0), pv(1.0), pv(1.0)
    )
    var true_error = abs(Float64(result[0].v) - exp_f64(-2.0))
    var estimate = Float64(result[1].v)
    assert_true(estimate > 0.1 * true_error and estimate < 100.0 * true_error)


def test_system_solves_the_harmonic_oscillator() raises:
    var y0 = Array[P, 2](fill=pv(0.0))
    y0[0] = pv(1.0)
    y0[1] = pv(0.0)
    var t = 2.5
    var y = rk4_system[n=2, f=oscillator, num_steps=1000](pv(0.0), y0, pv(t))
    assert_almost_equal(y[0].v, SIMD[dtype, width](cos_f64(t)), atol=1e-10)
    assert_almost_equal(y[1].v, SIMD[dtype, width](-sin_f64(t)), atol=1e-10)


def test_system_nearly_conserves_energy() raises:
    # RK4 isn't symplectic, so energy drifts -- just not much over a few
    # periods at this step size.
    var y0 = Array[P, 2](fill=pv(0.0))
    y0[0] = pv(1.0)
    y0[1] = pv(0.0)
    var y = rk4_system[n=2, f=oscillator, num_steps=4000](pv(0.0), y0, pv(20.0))
    var energy = Float64(y[0].v) ** 2 + Float64(y[1].v) ** 2
    assert_almost_equal(energy, 1.0, atol=1e-8)


def test_initial_condition_sensitivity_via_dual() raises:
    # d/dy0 [y(t)] for dy/dt = -2y is exp(-2t), since the equation is
    # linear -- and it comes out of the integrator itself, with no
    # variational equation written anywhere.
    var t = 0.7
    var y = rk4[f=decay, num_steps=200](
        D.constant(0.0), D(pv(1.0), pv(1.0)), D.constant(t)
    )
    assert_almost_equal(
        y.deriv.v, SIMD[dtype, width](exp_f64(-2.0 * t)), atol=1e-10
    )


def test_parameter_sensitivity_via_augmented_state() raises:
    # y(t) = exp(-k*t), so dy/dk = -t*exp(-k*t). The parameter rides along
    # as a state component with zero derivative; seeding *its* Dual is what
    # turns the integrator into a forward-sensitivity solver.
    var k = 1.5
    var t = 0.8
    var y0 = Array[D, 2](fill=D.constant(0.0))
    y0[0] = D.constant(1.0)
    y0[1] = D(pv(k), pv(1.0))

    var y = rk4_system[n=2, f=parametrized_decay, num_steps=400](
        D.constant(0.0), y0, D.constant(t)
    )

    assert_almost_equal(
        y[0].value.v, SIMD[dtype, width](exp_f64(-k * t)), atol=1e-10
    )
    assert_almost_equal(
        y[0].deriv.v,
        SIMD[dtype, width](-t * exp_f64(-k * t)),
        atol=1e-9,
    )
    # The parameter itself is unchanged, and its own sensitivity is 1.
    assert_almost_equal(y[1].value.v, SIMD[dtype, width](k), atol=1e-14)
    assert_almost_equal(y[1].deriv.v, SIMD[dtype, width](1.0), atol=1e-14)


def test_simd_lanes_integrate_independently() raises:
    comptime w = 4
    comptime PW = Plain[dtype, w]
    var y0 = PW(SIMD[dtype, w](1.0, 2.0, 3.0, 4.0))
    var y = rk4[f=decay, num_steps=200](PW.constant(0.0), y0, PW.constant(1.0))
    var expected = exp_f64(-2.0)
    for lane in range(w):
        assert_almost_equal(
            Float64(y.v[lane]),
            Float64(lane + 1) * expected,
            atol=1e-10,
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
