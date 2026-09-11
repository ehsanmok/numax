"""Tests for `numax.integrate`'s initial-value solvers over a `Tensor` state.

The `Tensor` forms share the Dormand-Prince tableau and the stage structure
with `numax.integrate.array`, so the strongest check is agreement between
the two on the same problem, component by component -- a divergence there
is a wrong stage weight on one side. The rest are the properties every
integrator has to satisfy: the order of accuracy, exactness on a
polynomial, and the adaptive controller taking the same steps as the scalar
one at `n == 1`.
"""

from std.collections import Array
from std.math import cos as cos_f64
from std.math import exp as exp_f64
from std.math import sin as sin_f64
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax import FloatLike, Plain
from numax.core.array import Static
from numax.integrate import dopri5, rk4_system, solve_ivp
from numax.integrate.array import rk4_system as array_rk4_system

comptime dtype = DType.float64
comptime P = Plain[dtype, 1]


def oscillator(
    t: Scalar[dtype], y: Static[dtype, 2], ctx: DeviceContext
) raises -> Static[dtype, 2]:
    """`y'' = -y` as a first-order system: `y(t) = cos(t)` from (1, 0)."""
    var h = y.to_host()
    return Static[dtype, 2](ctx, [h[1], -h[0]])


def decay(
    t: Scalar[dtype], y: Static[dtype, 1], ctx: DeviceContext
) raises -> Static[dtype, 1]:
    """`dy/dt = -2y`, so `y(t) = y0 exp(-2t)`."""
    var h = y.to_host()
    return Static[dtype, 1](ctx, [Scalar[dtype](-2.0) * h[0]])


def linear_in_t(
    t: Scalar[dtype], y: Static[dtype, 1], ctx: DeviceContext
) raises -> Static[dtype, 1]:
    """`dy/dt = t`, so `y(t) = y0 + t^2/2`; exact for both integrators."""
    return Static[dtype, 1](ctx, [t])


def array_oscillator[U: FloatLike](t: U, y: Array[U, 2]) -> Array[U, 2]:
    var out = Array[U, 2](fill=U.constant(0.0))
    out[0] = y[1].copy()
    out[1] = -y[0]
    return out^


def scalar_decay[U: FloatLike](t: U, y: U) -> U:
    return -(U.constant(2.0) * y)


def test_rk4_system_solves_the_harmonic_oscillator() raises:
    var ctx = DeviceContext(api="cpu")
    var y0 = Static[dtype, 2](ctx, [1.0, 0.0])
    var y = rk4_system[f=oscillator, num_steps=200](0.0, y0, 2.0).to_host()
    assert_almost_equal(y[0], Scalar[dtype](cos_f64(2.0)), atol=1e-7)
    assert_almost_equal(y[1], Scalar[dtype](-sin_f64(2.0)), atol=1e-7)


def test_rk4_system_agrees_with_the_array_tier_component_by_component() raises:
    # Same tableau, same stages, same problem: the two tiers must agree to
    # rounding, which is the check that neither has a wrong stage weight.
    var ctx = DeviceContext(api="cpu")
    var y0 = Static[dtype, 2](ctx, [1.0, 0.0])
    var here = rk4_system[f=oscillator, num_steps=50](0.0, y0, 1.5).to_host()

    var start = Array[P, 2](fill=P.constant(0.0))
    start[0] = P.constant(1.0)
    var there = array_rk4_system[P, 2, array_oscillator, 50](
        P.constant(0.0), start, P.constant(1.5)
    )
    for i in range(2):
        assert_almost_equal(
            Float64(here[i]), Float64(there[i].v[0]), atol=1e-14
        )


def test_rk4_system_is_fourth_order() raises:
    # Halving the step should cut the error by about 2^4 = 16. On the
    # oscillator the ratio approaches 16 from above -- 20.8 at 20-vs-40
    # steps, 18.9 at 40-vs-80 -- so the window is set for the finer pair.
    var ctx = DeviceContext(api="cpu")
    var y0 = Static[dtype, 2](ctx, [1.0, 0.0])
    var coarse = rk4_system[f=oscillator, num_steps=40](0.0, y0, 3.0).to_host()
    var fine = rk4_system[f=oscillator, num_steps=80](0.0, y0, 3.0).to_host()
    var err_coarse = abs(Float64(coarse[0]) - cos_f64(3.0))
    var err_fine = abs(Float64(fine[0]) - cos_f64(3.0))
    var ratio = err_coarse / err_fine
    assert_true(ratio > 12.0 and ratio < 24.0)


def test_dopri5_is_fifth_order_and_beats_rk4() raises:
    var ctx = DeviceContext(api="cpu")
    var y0 = Static[dtype, 2](ctx, [1.0, 0.0])
    var coarse = dopri5[f=oscillator, num_steps=20](0.0, y0, 3.0).to_host()
    var fine = dopri5[f=oscillator, num_steps=40](0.0, y0, 3.0).to_host()
    var err_coarse = abs(Float64(coarse[0]) - cos_f64(3.0))
    var err_fine = abs(Float64(fine[0]) - cos_f64(3.0))
    var ratio = err_coarse / err_fine
    assert_true(ratio > 24.0 and ratio < 40.0)

    var rk = rk4_system[f=oscillator, num_steps=20](0.0, y0, 3.0).to_host()
    assert_true(err_coarse < abs(Float64(rk[0]) - cos_f64(3.0)))


def test_both_integrators_are_exact_on_a_polynomial() raises:
    var ctx = DeviceContext(api="cpu")
    var y0 = Static[dtype, 1](ctx, [1.0])
    var a = rk4_system[f=linear_in_t, num_steps=7](0.0, y0, 3.0).to_host()
    var b = dopri5[f=linear_in_t, num_steps=7](0.0, y0, 3.0).to_host()
    assert_almost_equal(a[0], Scalar[dtype](5.5), atol=1e-13)
    assert_almost_equal(b[0], Scalar[dtype](5.5), atol=1e-13)


def test_backwards_integration_inverts_forwards() raises:
    var ctx = DeviceContext(api="cpu")
    var y0 = Static[dtype, 2](ctx, [1.0, 0.0])
    var forward = rk4_system[f=oscillator, num_steps=100](0.0, y0, 2.0)
    var back = rk4_system[f=oscillator, num_steps=100](
        2.0, forward, 0.0
    ).to_host()
    assert_almost_equal(back[0], Scalar[dtype](1.0), atol=1e-9)
    assert_almost_equal(back[1], Scalar[dtype](0.0), atol=1e-9)


def test_solve_ivp_reaches_the_exponential_within_tolerance() raises:
    var ctx = DeviceContext(api="cpu")
    var y0 = Static[dtype, 1](ctx, [3.0])
    var result = solve_ivp[f=decay](0.0, y0, 2.0, rtol=1e-8, atol=1e-10)
    assert_true(result.converged)
    assert_almost_equal(result.t, 2.0)
    assert_almost_equal(
        result.y.to_host()[0], Scalar[dtype](3.0 * exp_f64(-4.0)), atol=1e-7
    )
    assert_true(result.accepted > 0)


def test_solve_ivp_takes_the_same_steps_as_the_scalar_controller() raises:
    # At n == 1 the infinity-norm ratio is the scalar ratio, so the two
    # overloads of `solve_ivp` must accept and reject identically.
    from numax.integrate import solve_ivp as scalar_solve_ivp

    var ctx = DeviceContext(api="cpu")
    var y0 = Static[dtype, 1](ctx, [3.0])
    var tensor_result = solve_ivp[f=decay](0.0, y0, 2.0)
    var scalar_result = scalar_solve_ivp[scalar_decay](0.0, 3.0, 2.0)
    assert_equal(tensor_result.accepted, scalar_result.accepted)
    assert_equal(tensor_result.rejected, scalar_result.rejected)
    assert_almost_equal(
        tensor_result.y.to_host()[0], Scalar[dtype](scalar_result.y), atol=1e-14
    )


def test_solve_ivp_reports_non_convergence_when_steps_run_out() raises:
    var ctx = DeviceContext(api="cpu")
    var y0 = Static[dtype, 1](ctx, [3.0])
    var result = solve_ivp[f=decay](0.0, y0, 2.0, max_steps=3)
    assert_true(not result.converged)
    assert_true(result.t < 2.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
