"""Fixed-step Runge-Kutta integrators for initial-value problems.

Solve `dy/dt = f(t, y)` from `t0` to `t1` given `y(t0) = y0`, for a scalar
`y` (`rk4`, `dopri5`) or an `n`-component system (`rk4_system`).

Two things fall out of writing these against `FloatLike` rather than a
concrete float type.

**Sensitivities come from the integrator you already have.** Integrating at
`Dual` propagates a derivative through every stage, so seeding `y0` with
derivative `1` returns `dy(t1)/dy0` alongside the solution -- the
initial-condition sensitivity, with no variational equation to derive and
no adjoint pass. Sensitivity to a *parameter* needs no new API either:
augment the state with the parameter and give it `dp/dt = 0`, then seed
that component's derivative instead. `tests/test_ode.mojo` checks both
against closed forms.

**One thread per initial condition is the natural GPU shape.** The whole
integration is a fixed number of stages over a fixed number of steps with
no data-dependent control flow, so it compiles into a `map[gpu=True]`
kernel body directly: each thread integrates its own trajectory to
completion. `examples/advanced/ode.mojo` runs an ensemble that way, verified
on Metal and on CUDA.

## Scope: fixed steps, no adaptive control

Both integrators take a step count, not a tolerance, because no kernel in
`numax` runs a data-dependent number of iterations -- that's what keeps every
one of them launchable inside a GPU thread. This is a real limitation and
worth being precise about, because Dormand-Prince is *usually* an adaptive
method -- its embedded 4th-order pair exists to estimate the local error so
a controller can resize the step. `dopri5` here still computes that
estimate (`dopri5_with_error` returns it), but nothing acts on it: an
adaptive controller would have each SIMD lane and each GPU thread accepting
or rejecting steps on its own schedule, which `FloatLike` has no way to
express and which would destroy the one-thread-per-trajectory shape above.

What you get instead is a 5th-order solution for the same step count a
4th-order RK4 would need, plus an honest error number you can inspect to
decide whether to re-run with more steps. Stiff problems still want an
implicit method, which isn't here.
"""

from std.collections import Array

from .numeric import FloatLike


def rk4[
    T: FloatLike,
    f: def[U: FloatLike](U, U) thin -> U,
    num_steps: Int = 100,
](t0: T, y0: T, t1: T) -> T:
    """Integrate the scalar `dy/dt = f(t, y)` from `t0` to `t1` with
    `num_steps` steps of the classical fourth-order Runge-Kutta method.

    Fourth-order accurate: halving the step size cuts the error by about
    16x, which `tests/test_ode.mojo` verifies directly rather than
    asserting.

    `t1 < t0` integrates backwards, since the step size is just
    `(t1-t0)/num_steps` and nothing here assumes its sign.
    """
    var h = (t1 + (-t0)) / T.constant(Float64(num_steps))
    var half = h / T.constant(2.0)
    var y = y0.copy()
    # The step index is carried as a running `T` rather than converted from
    # the loop's `Int` -- see `numax.orthopoly`'s module docstring for the
    # Metal instruction that rules out the obvious `T.constant(Float64(i))`.
    var step_index = T.constant(0.0)

    for _ in range(num_steps):
        var t = t0 + step_index * h

        var k1 = f[T](t.copy(), y.copy())
        var k2 = f[T](t + half, y + half * k1)
        var k3 = f[T](t + half, y + half * k2)
        var k4 = f[T](t + h, y + h * k3)

        y = y + h * (
            k1 + T.constant(2.0) * k2 + T.constant(2.0) * k3 + k4
        ) / T.constant(6.0)
        step_index = step_index + T.one()

    return y^


def rk4_system[
    T: FloatLike,
    n: Int,
    f: def[U: FloatLike](U, Array[U, n]) thin -> Array[U, n],
    num_steps: Int = 100,
](t0: T, y0: Array[T, n], t1: T) -> Array[T, n]:
    """Integrate the `n`-component system `dy/dt = f(t, y)` with the same
    method as `rk4`.

    `n` is a compile-time size, so the state lives in registers and the
    whole integration stays GPU-launchable. This is also where parameter
    sensitivity goes: append the parameter to the state with `dp/dt = 0`,
    seed its `Dual` derivative, and read the sensitivity off the components
    you care about.
    """
    var h = (t1 + (-t0)) / T.constant(Float64(num_steps))
    var half = h / T.constant(2.0)
    var y = _copy_state[T, n](y0)
    var step_index = T.constant(0.0)

    for _ in range(num_steps):
        var t = t0 + step_index * h

        var k1 = f[T](t.copy(), _copy_state[T, n](y))
        var k2 = f[T](t + half, _axpy[T, n](half, k1, y))
        var k3 = f[T](t + half, _axpy[T, n](half, k2, y))
        var k4 = f[T](t + h, _axpy[T, n](h.copy(), k3, y))

        var sixth = h / T.constant(6.0)
        for i in range(n):
            var slope = (
                k1[i]
                + T.constant(2.0) * k2[i]
                + T.constant(2.0) * k3[i]
                + k4[i]
            )
            y[i] = y[i] + sixth * slope
        step_index = step_index + T.one()

    return y^


def dopri5[
    T: FloatLike,
    f: def[U: FloatLike](U, U) thin -> U,
    num_steps: Int = 100,
](t0: T, y0: T, t1: T) -> T:
    """Integrate the scalar `dy/dt = f(t, y)` with fixed-step
    Dormand-Prince 5(4).

    Fifth-order accurate for seven stages per step against `rk4`'s four,
    which is the trade: more work per step, but the error falls off as
    `h^5`, so it wins decisively once the step count is anywhere near
    adequate.

    Use `dopri5_with_error` if you want the embedded 4th-order pair's
    disagreement as a local error indicator; see this module's docstring for
    why nothing here acts on it automatically.
    """
    var result = dopri5_with_error[T, f, num_steps](t0, y0, t1)
    return result[0].copy()


def dopri5_with_error[
    T: FloatLike,
    f: def[U: FloatLike](U, U) thin -> U,
    num_steps: Int = 100,
](t0: T, y0: T, t1: T) -> Tuple[T, T]:
    """`dopri5`, also returning the summed magnitude of the 5th- and
    4th-order solutions' per-step disagreement.

    That second value is the standard local truncation error estimate an
    adaptive controller would drive the step size with. Summed over fixed
    steps it's a rough global error proxy: useful for deciding whether
    `num_steps` was enough, not a rigorous bound.
    """
    var h = (t1 + (-t0)) / T.constant(Float64(num_steps))
    var y = y0.copy()
    var error_sum = T.constant(0.0)
    var step_index = T.constant(0.0)

    for _ in range(num_steps):
        var t = t0 + step_index * h

        var k1 = f[T](t.copy(), y.copy())
        var k2 = f[T](t + _c(h, _C2), y + h * (_c(k1, _A21)))
        var k3 = f[T](t + _c(h, _C3), y + h * (_c(k1, _A31) + _c(k2, _A32)))
        var k4 = f[T](
            t + _c(h, _C4),
            y + h * (_c(k1, _A41) + _c(k2, _A42) + _c(k3, _A43)),
        )
        var k5 = f[T](
            t + _c(h, _C5),
            y + h * (_c(k1, _A51) + _c(k2, _A52) + _c(k3, _A53) + _c(k4, _A54)),
        )
        var k6 = f[T](
            t + h,
            y
            + h
            * (
                _c(k1, _A61)
                + _c(k2, _A62)
                + _c(k3, _A63)
                + _c(k4, _A64)
                + _c(k5, _A65)
            ),
        )

        # The 5th-order solution. Its stage weights are also row 7 of the
        # Butcher tableau, which is what makes `k7` below the next step's
        # `k1` (the "first same as last" property) -- not exploited here,
        # since `k7` is only needed for the error estimate.
        var increment = (
            _c(k1, _B1) + _c(k3, _B3) + _c(k4, _B4) + _c(k5, _B5) + _c(k6, _B6)
        )
        var y5 = y + h * increment

        var k7 = f[T](t + h, y5.copy())
        var increment_hat = (
            _c(k1, _BH1)
            + _c(k3, _BH3)
            + _c(k4, _BH4)
            + _c(k5, _BH5)
            + _c(k6, _BH6)
            + _c(k7, _BH7)
        )
        var y4 = y + h * increment_hat

        error_sum = error_sum + (y5 + (-y4)).abs()
        y = y5.copy()
        step_index = step_index + T.one()

    return (y^, error_sum^)


def _c[T: FloatLike](x: T, coefficient: Float64) -> T:
    """`coefficient * x`, with the coefficient a compile-time literal.

    Every caller passes one of the `_A`/`_B`/`_C` constants below, so
    `T.constant` folds into a `dtype`-native literal and no float64
    arithmetic survives into the generated code.
    """
    return T.constant(coefficient) * x


def _copy_state[T: FloatLike, n: Int](y: Array[T, n]) -> Array[T, n]:
    var out = Array[T, n](fill=T.constant(0.0))
    for i in range(n):
        out[i] = y[i].copy()
    return out^


def _axpy[
    T: FloatLike, n: Int
](a: T, x: Array[T, n], y: Array[T, n]) -> Array[T, n]:
    """`y + a*x`, componentwise -- the stage argument every RK stage builds."""
    var out = Array[T, n](fill=T.constant(0.0))
    for i in range(n):
        out[i] = y[i] + a * x[i]
    return out^


# Dormand-Prince 5(4) Butcher tableau (Dormand & Prince 1980, "A family of
# embedded Runge-Kutta formulae"). `_C*` are the stage times as fractions of
# the step, `_A*` the stage coefficients, `_B*` the 5th-order weights, and
# `_BH*` the embedded 4th-order weights. `_B2`/`_BH2` are zero and omitted
# rather than written out, which is why `k2` appears in no weighted sum.
comptime _C2 = 1.0 / 5.0
comptime _C3 = 3.0 / 10.0
comptime _C4 = 4.0 / 5.0
comptime _C5 = 8.0 / 9.0

comptime _A21 = 1.0 / 5.0
comptime _A31 = 3.0 / 40.0
comptime _A32 = 9.0 / 40.0
comptime _A41 = 44.0 / 45.0
comptime _A42 = -56.0 / 15.0
comptime _A43 = 32.0 / 9.0
comptime _A51 = 19372.0 / 6561.0
comptime _A52 = -25360.0 / 2187.0
comptime _A53 = 64448.0 / 6561.0
comptime _A54 = -212.0 / 729.0
comptime _A61 = 9017.0 / 3168.0
comptime _A62 = -355.0 / 33.0
comptime _A63 = 46732.0 / 5247.0
comptime _A64 = 49.0 / 176.0
comptime _A65 = -5103.0 / 18656.0

comptime _B1 = 35.0 / 384.0
comptime _B3 = 500.0 / 1113.0
comptime _B4 = 125.0 / 192.0
comptime _B5 = -2187.0 / 6784.0
comptime _B6 = 11.0 / 84.0

comptime _BH1 = 5179.0 / 57600.0
comptime _BH3 = 7571.0 / 16695.0
comptime _BH4 = 393.0 / 640.0
comptime _BH5 = -92097.0 / 339200.0
comptime _BH6 = 187.0 / 2100.0
comptime _BH7 = 1.0 / 40.0
