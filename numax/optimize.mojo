"""Root finding and minimization that iterate until they converge.

**This module is tier 2.** Every function here loops until a tolerance is
met or an iteration cap is hit, and branches on data to decide. That is a
deliberate exception to the fixed-iteration invariant the rest of `numax`
holds absolutely, and it is why nothing here is `FloatLike`-generic in its
*driver*: these run on the host, on `Plain` values, and none of them is
launchable inside a GPU thread. See `docs/architecture.md`'s "Two tiers".

The objective function is a different matter, and this is the whole point of
the module. `f` is still an ordinary `FloatLike` kernel -- written once,
generic over its own conformer -- so the *optimizer* evaluates it at `Dual`
or `Gradient` and gets exact derivatives out, to machine precision, with no
adjoint rule and no finite-difference step size to tune:

```mojo
def rosenbrock[U: FloatLike](v: Array[U, 2]) -> U:
    var a = U.one() + (-v[0])
    var b = v[1] + (-(v[0] * v[0]))
    return a * a + U.constant(100.0) * b * b

# BFGS gets the exact gradient by calling that same function at
# `Gradient[Plain[float64, 1], 2]` -- one call, both partials.
var result = bfgs[2, rosenbrock](start)
```

A finite-difference gradient has to trade truncation error against
cancellation error, and the best achievable accuracy is roughly
`sqrt(eps)` -- about 1e-8 at float64. Forward-mode AD has neither error
term: the derivative is computed by the chain rule alongside the value, so
it is as accurate as the value itself. `tests/test_optimize.mojo` measures
the difference rather than asserting it.

## Tier 1 siblings

`numax.solve` has `newton`, `halley` and `bisection` at *fixed* iteration
counts -- same mathematics, no convergence test, launchable inside a GPU
thread. They are not superseded by anything here:

| tier 1 (`numax.solve`) | tier 2 (here) | difference |
|---|---|---|
| `newton` | `newton_tol` | fixed 20 steps vs. loop until `\\|dx\\| < tol` |
| `bisection` | `brentq` | fixed halvings vs. inverse-quadratic interpolation to tolerance |

Reach for the tier-1 version inside a kernel or when the iteration count
has to be predictable; reach for these when accuracy matters more than
uniformity and you are on the host anyway.

## What converged, and what didn't

Every function returns its status rather than raising or silently
returning a wrong answer. `OptimizeResult` carries `converged` and
`iterations` alongside the value, because "Newton wandered off and hit the
cap" and "Newton converged in 4 steps" are not distinguishable from the
returned number alone, and a caller that ignores the difference will
eventually be wrong about something quiet.
"""

from std.collections import Array

from .dual import Dual
from .gradient import Gradient
from .numeric import FloatLike
from .plain import Plain

# The conformer every driver here evaluates `f` at. Fixed to float64 on
# purpose, and not a parameter, for two reasons. Convergence work belongs at
# the widest available precision -- a tolerance of 1e-12 is meaningless at
# float32 -- and Mojo will not accept a struct instantiated with a
# *function-level* `DType` parameter as a `FloatLike` type argument
# (`f[Plain[dtype, 1]]` inside a `def foo[dtype: DType]` fails with
# "parameter 'U' has 'FloatLike' type, but value has type
# 'AnyStruct[Plain[dtype, Int(1)]]'"), so a per-call dtype would not compile
# at all. Anything needing another dtype is doing kernel work, which is
# tier 1 and lives in `numax.solve`.
comptime _P = Plain[DType.float64, 1]


@fieldwise_init
struct OptimizeResult(Copyable, Movable):
    """The outcome of a tier-2 iteration: the answer, plus whether it is
    one.

    `converged` false means the iteration cap was reached with the
    tolerance unmet -- `x` is then the last iterate, which may be
    perfectly good or may be nonsense, and only the caller's own problem
    knows which.
    """

    var x: Float64
    var f_x: Float64
    var iterations: Int
    var converged: Bool


@fieldwise_init
struct MinimizeResult[n_vars: Int](Copyable, Movable):
    """`OptimizeResult` for a multi-variable minimization: `x` is the
    argument vector, `grad_norm` the infinity-norm of the gradient at it
    (the quantity the convergence test actually looks at)."""

    var x: Array[Float64, Self.n_vars]
    var f_x: Float64
    var grad_norm: Float64
    var iterations: Int
    var converged: Bool


def newton_tol[
    f: def[U: FloatLike](U) thin -> U,
](x0: Float64, tol: Float64 = 1e-12, max_iter: Int = 64,) -> OptimizeResult:
    """Newton's method on `f`, iterating until the step is smaller than
    `tol`.

    The derivative comes from evaluating `f` at `Dual` -- there is no
    second function to supply and no finite difference taken, exactly as in
    `numax.solve.newton`. What differs is only the stopping rule: this one
    tests the step and reports whether the test was met.

    Quadratic convergence near a simple root, and no convergence guarantee
    away from one: a zero (or near-zero) derivative sends the step to
    infinity. `brentq` is the one to reach for when the root is bracketed
    but the function is not well behaved, since it cannot leave the
    bracket.
    """

    var x = x0
    for i in range(max_iter):
        var evaluated = f[Dual[_P]](Dual[_P](_P(x), _P.one()))
        var value = evaluated.value.v
        var derivative = evaluated.deriv.v
        if derivative == 0:
            return OptimizeResult(x, value, i + 1, False)
        var step = value / derivative
        x = x - step
        if abs(step) < tol:
            return OptimizeResult(x, f[_P](_P(x)).v, i + 1, True)
    return OptimizeResult(x, f[_P](_P(x)).v, max_iter, False)


def brentq[
    f: def[U: FloatLike](U) thin -> U,
](
    a: Float64,
    b: Float64,
    tol: Float64 = 1e-12,
    max_iter: Int = 128,
) -> OptimizeResult:
    """Brent's method: find a root of `f` in `[a, b]`, where `f(a)` and
    `f(b)` must have opposite signs.

    The workhorse root finder, and the one to prefer when a bracket is
    available. It keeps the root bracketed at every step -- so it cannot
    diverge the way `newton_tol` can -- while using inverse quadratic
    interpolation to converge superlinearly where the function is smooth,
    falling back to bisection whenever interpolation would step outside the
    bracket or fail to shrink it fast enough.

    `f` is evaluated at `Plain` only: Brent uses no derivative. Returns
    immediately with `converged=False` if the bracket does not straddle a
    sign change, since every guarantee the method has follows from that
    precondition.
    """

    var lo = a
    var hi = b
    var f_lo = f[_P](_P(lo)).v
    var f_hi = f[_P](_P(hi)).v

    if f_lo == 0:
        return OptimizeResult(lo, f_lo, 0, True)
    if f_hi == 0:
        return OptimizeResult(hi, f_hi, 0, True)
    if (f_lo > 0) == (f_hi > 0):
        return OptimizeResult(lo, f_lo, 0, False)

    # Keep `hi` as the better of the two endpoints, which is what makes the
    # interpolation formulas below well-conditioned.
    if abs(f_lo) < abs(f_hi):
        var swap_x = lo
        lo = hi
        hi = swap_x
        var swap_f = f_lo
        f_lo = f_hi
        f_hi = swap_f

    var prev = lo
    var f_prev = f_lo
    var older = lo
    var used_bisection = True

    for i in range(max_iter):
        var candidate: Float64
        if f_hi != f_prev and f_lo != f_prev:
            # Inverse quadratic interpolation through the three points.
            var d1 = (f_hi - f_lo) * (f_hi - f_prev)
            var d2 = (f_lo - f_hi) * (f_lo - f_prev)
            var d3 = (f_prev - f_lo) * (f_prev - f_hi)
            candidate = (
                hi * f_lo * f_prev / d1
                + lo * f_hi * f_prev / d2
                + prev * f_hi * f_lo / d3
            )
        else:
            # Secant through the two current endpoints.
            candidate = hi - f_hi * (hi - lo) / (f_hi - f_lo)

        # Accept the interpolated point only if it lands in the outer
        # quarter-to-endpoint window `[(3*lo + hi)/4, hi]` *and* is at
        # least halving the step relative to the previous move. Reject on
        # either count and bisect instead. This pair of tests is what makes
        # Brent's method no worse than bisection in the limit rather than
        # merely faster than it when the function cooperates -- an
        # interpolation that keeps landing just outside the bracket, or that
        # stalls, would otherwise let the interval stop shrinking.
        var midpoint = (lo + hi) / 2
        var quarter = (3 * lo + hi) / 4
        var window_lo = min(quarter, hi)
        var window_hi = max(quarter, hi)
        var in_window = candidate > window_lo and candidate < window_hi
        var shrinking: Bool
        if used_bisection:
            shrinking = abs(candidate - hi) < abs(hi - prev) / 2
        else:
            shrinking = abs(candidate - hi) < abs(prev - older) / 2

        if in_window and shrinking:
            used_bisection = False
        else:
            candidate = midpoint
            used_bisection = True

        var f_candidate = f[_P](_P(candidate)).v
        older = prev
        prev = hi
        f_prev = f_hi

        if (f_lo > 0) == (f_candidate > 0):
            lo = candidate
            f_lo = f_candidate
        else:
            hi = candidate
            f_hi = f_candidate

        if abs(f_lo) < abs(f_hi):
            var swap_x = lo
            lo = hi
            hi = swap_x
            var swap_f = f_lo
            f_lo = f_hi
            f_hi = swap_f

        if f_hi == 0 or abs(hi - lo) < tol:
            return OptimizeResult(hi, f_hi, i + 1, True)

    return OptimizeResult(hi, f_hi, max_iter, False)


def bfgs[
    n_vars: Int,
    f: def[U: FloatLike](Array[U, n_vars]) thin -> U,
](
    x0: Array[Float64, n_vars],
    tol: Float64 = 1e-8,
    max_iter: Int = 200,
) -> MinimizeResult[n_vars]:
    """Minimize `f` by BFGS with a backtracking line search.

    **The gradient is exact.** `f` is evaluated once per iteration at
    `Gradient[_P, n_vars]`, which returns `f(x)` and all
    `n_vars` partial derivatives from that single call, by the chain rule.
    No finite differences, so no step size to choose and none of the
    accuracy loss that choice costs: a central difference is limited to
    about `sqrt(eps)` relative accuracy no matter how carefully the step is
    picked, while forward-mode AD is as accurate as the function value.

    That is what makes this different from a SciPy port. `f` is an ordinary
    `FloatLike` kernel -- the same one a caller might evaluate at `Plain`
    for speed, at `Compensated` for precision, or here at `Gradient` for
    derivatives -- and the optimizer picks the conformer it needs.

    The inverse-Hessian approximation `H` starts at the identity and is
    updated by the standard BFGS formula. The line search backtracks by
    halving until it sees a decrease (an Armijo condition with `c1 = 1e-4`),
    which is the cheap and robust choice; it is not a Wolfe search, so
    convergence on badly scaled problems is slower than a production
    implementation would manage.

    Convergence is `max|grad| < tol`. `MinimizeResult.grad_norm` reports
    that quantity so a caller can see how close a non-converged run got.
    """

    var x = x0.copy()

    # Inverse Hessian approximation, row-major n_vars x n_vars, starting at
    # the identity: the first step is therefore plain steepest descent.
    var h = Array[Float64, n_vars * n_vars](fill=0)
    for i in range(n_vars):
        h[i * n_vars + i] = 1

    var f_x: Float64 = 0
    var grad = Array[Float64, n_vars](fill=0)
    var grad_norm: Float64 = 0

    for iteration in range(max_iter):
        # One call: value and every partial derivative, exactly.
        var seeded = Array[Gradient[_P, n_vars], n_vars](
            fill=Gradient[_P, n_vars].constant(0.0)
        )
        for i in range(n_vars):
            seeded[i] = Gradient[_P, n_vars].variable(_P(x[i]), i)
        var evaluated = f[Gradient[_P, n_vars]](seeded^)
        f_x = evaluated.value.v
        for i in range(n_vars):
            grad[i] = evaluated.grad[i].v

        grad_norm = 0
        for i in range(n_vars):
            grad_norm = max(grad_norm, abs(grad[i]))
        if grad_norm < tol:
            return MinimizeResult[n_vars](x^, f_x, grad_norm, iteration, True)

        # Search direction p = -H @ grad.
        var p = Array[Float64, n_vars](fill=0)
        for i in range(n_vars):
            var total: Float64 = 0
            for j in range(n_vars):
                total += h[i * n_vars + j] * grad[j]
            p[i] = -total

        # Backtracking line search: halve until Armijo is satisfied.
        var directional: Float64 = 0
        for i in range(n_vars):
            directional += grad[i] * p[i]
        var step: Float64 = 1
        var accepted = False
        var candidate = Array[Float64, n_vars](fill=0)
        for _ in range(60):
            for i in range(n_vars):
                candidate[i] = x[i] + step * p[i]
            var as_plain = Array[_P, n_vars](fill=_P.constant(0.0))
            for i in range(n_vars):
                as_plain[i] = _P(candidate[i])
            if f[_P](as_plain^).v <= f_x + Float64(1e-4) * step * directional:
                accepted = True
                break
            step = step / 2
        if not accepted:
            # The direction is not a descent direction any more, which
            # means `H` has gone bad. Report rather than spin.
            return MinimizeResult[n_vars](
                x^, f_x, grad_norm, iteration + 1, False
            )

        # s = x_new - x, y = grad_new - grad. The new gradient needs
        # another AD evaluation; that is the honest cost of BFGS, and it is
        # one call rather than `n_vars` finite-difference pairs.
        var s = Array[Float64, n_vars](fill=0)
        for i in range(n_vars):
            s[i] = candidate[i] - x[i]

        var seeded_new = Array[Gradient[_P, n_vars], n_vars](
            fill=Gradient[_P, n_vars].constant(0.0)
        )
        for i in range(n_vars):
            seeded_new[i] = Gradient[_P, n_vars].variable(_P(candidate[i]), i)
        var evaluated_new = f[Gradient[_P, n_vars]](seeded_new^)
        var y = Array[Float64, n_vars](fill=0)
        for i in range(n_vars):
            y[i] = evaluated_new.grad[i].v - grad[i]

        var sy: Float64 = 0
        for i in range(n_vars):
            sy += s[i] * y[i]

        for i in range(n_vars):
            x[i] = candidate[i]

        # Skip the update when s.y is not positive: the BFGS formula
        # divides by it, and a non-positive value means the curvature
        # condition failed, so applying it would make `H` indefinite.
        # Keeping the previous `H` costs one slower step; a NaN-filled `H`
        # costs the whole run.
        if sy <= 0:
            continue

        # H <- (I - s y^T / sy) H (I - y s^T / sy) + s s^T / sy
        var hy = Array[Float64, n_vars](fill=0)
        for i in range(n_vars):
            var total: Float64 = 0
            for j in range(n_vars):
                total += h[i * n_vars + j] * y[j]
            hy[i] = total
        var yhy: Float64 = 0
        for i in range(n_vars):
            yhy += y[i] * hy[i]

        for i in range(n_vars):
            for j in range(n_vars):
                var updated = h[i * n_vars + j]
                updated -= (s[i] * hy[j] + hy[i] * s[j]) / sy
                updated += s[i] * s[j] * (1 + yhy / sy) / sy
                h[i * n_vars + j] = updated

    return MinimizeResult[n_vars](x^, f_x, grad_norm, max_iter, False)
