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
    var a = U.one() - v[0]
    var b = v[1] - (v[0] * v[0])
    return a * a + U.constant(100.0) * b * b

# BFGS gets the exact gradient by calling that same function at
# `Gradient[Plain[float64], 2]` -- one call, both partials.
var result = bfgs[2, rosenbrock](start)
```

A finite-difference gradient has to trade truncation error against
cancellation error, and the best achievable accuracy is roughly
`sqrt(eps)` -- about 1e-8 at float64. Forward-mode AD has neither error
term: the derivative is computed by the chain rule alongside the value, so
it is as accurate as the value itself. `tests/optimize/test_optimize.mojo` measures
the difference rather than asserting it.

## Tier 1 siblings

`numax.optimize` has `newton`, `halley` and `bisection` at *fixed* iteration
counts -- same mathematics, no convergence test, launchable inside a GPU
thread. They are not superseded by anything here:

| tier 1 (`numax.optimize`) | tier 2 (here) | difference |
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

from ...core.dual import Dual
from ...core.gradient import Gradient
from ...core.numeric import FloatLike
from ...core.plain import Plain
from ...linalg.array.cholesky import cholesky, cholesky_solve

# The conformer every driver here evaluates `f` at. Fixed to float64 on
# purpose, and not a parameter, for two reasons. Convergence work belongs at
# the widest available precision -- a tolerance of 1e-12 is meaningless at
# float32 -- and Mojo will not accept a struct instantiated with a
# *function-level* `DType` parameter as a `FloatLike` type argument
# (`f[Plain[dtype]]` inside a `def foo[dtype: DType]` fails with
# "parameter 'U' has 'FloatLike' type, but value has type
# 'AnyStruct[Plain[dtype, Int(1)]]'"), so a per-call dtype would not compile
# at all. Anything needing another dtype is doing kernel work, which is
# tier 1 and lives in `numax.optimize`.
comptime _P = Plain[DType.float64]


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
    `numax.optimize.newton`. What differs is only the stopping rule: this one
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


# The golden ratio's two useful constants. `_GOLDEN_SECTION` is `2 - phi`,
# the fraction of a bracket a golden-section step moves; `_GOLDEN_GROW` is
# `phi` itself, the factor the bracketing search expands by.
comptime _GOLDEN_SECTION = 0.3819660112501051
comptime _GOLDEN_GROW = 1.618033988749895

# The best a minimizer can locate `x` to. Near a minimum `f` is quadratic, so
# a change in `x` of `d` moves `f` by `O(d^2)`: once `d` falls below
# `sqrt(eps)` the change is below the noise in `f` itself and no further
# progress is real. This is `sqrt(2.22e-16)`, and it is why the scalar
# minimizers default to a tolerance eight orders looser than the root
# finders in this module -- `brentq` locates a *crossing*, which has no such
# floor.
comptime _MINIMIZER_TOL = 1.48e-8


@fieldwise_init
struct _Bracket(Copyable, Movable):
    """Three points with `f(b)` below both `f(a)` and `f(c)`, which is what
    guarantees a minimum lies between `a` and `c`. `found` is false when the
    search ran out of expansions, which means `f` decreased monotonically
    the whole way and has no minimum in that direction."""

    var a: Float64
    var b: Float64
    var c: Float64
    var f_b: Float64
    var evaluations: Int
    var found: Bool


def _bracket_minimum[
    f: def[U: FloatLike](U) thin -> U,
](start: Float64, second: Float64) -> _Bracket:
    """Expand `[start, second]` downhill until it brackets a minimum.

    Orient the interval so `f` decreases from `a` to `b`, then keep stepping
    a golden ratio further in that direction until `f` turns back up. SciPy
    accelerates this with a parabolic extrapolation and a growth limit; this
    does not, and the cost is a few extra evaluations on a long shallow
    descent rather than a different answer -- the bracket either exists in
    that direction or it does not.

    An unbounded descent is reported (`found=False`) rather than raised,
    matching how everything else in this module reports rather than throws.
    """
    var a = start
    var b = second
    var f_a = f[_P](_P(a)).v
    var f_b = f[_P](_P(b)).v
    var evaluations = 2

    # Walk downhill: `b` must be the lower of the two.
    if f_a < f_b:
        var swap_x = a
        a = b
        b = swap_x
        var swap_f = f_a
        f_a = f_b
        f_b = swap_f

    var c = b + _GOLDEN_GROW * (b - a)
    var f_c = f[_P](_P(c)).v
    evaluations += 1

    for _ in range(80):
        if f_c >= f_b:
            return _Bracket(a, b, c, f_b, evaluations, True)
        a = b
        f_a = f_b
        b = c
        f_b = f_c
        c = b + _GOLDEN_GROW * (b - a)
        f_c = f[_P](_P(c)).v
        evaluations += 1

    return _Bracket(a, b, c, f_b, evaluations, False)


def _brent_on_interval[
    f: def[U: FloatLike](U) thin -> U,
](
    lower: Float64,
    upper: Float64,
    start: Float64,
    tol: Float64,
    max_iter: Int,
) -> OptimizeResult:
    """Brent's minimization on `[lower, upper]`, starting from `start`.

    The shared engine under `brent` and `fminbound`; the two differ only in
    how they arrive at the interval. Parabolic interpolation through the
    three best points so far, falling back to a golden-section step whenever
    the parabola would land outside the interval, would not at least halve
    the step before last, or is simply not yet available. That fallback is
    what bounds the worst case: golden section alone converges linearly, and
    the interpolation only ever accelerates it.

    Derivative-free on purpose. `f` is evaluated at `Plain` only -- a
    minimizer with an exact gradient is `bfgs` or `cg`, and for one variable
    the bracket is the more robust structure anyway, since it cannot leave
    the interval the way a Newton step can.
    """
    var a = lower
    var b = upper

    # `x` is the best point, `w` the second best, `v` the previous `w`.
    var x = start
    var w = start
    var v = start
    var f_x = f[_P](_P(x)).v
    var f_w = f_x
    var f_v = f_x

    var step: Float64 = 0
    var previous_step: Float64 = 0

    for iteration in range(max_iter):
        var middle = (a + b) / 2
        var tol1 = tol * abs(x) + 1e-11
        var tol2 = 2 * tol1
        if abs(x - middle) <= tol2 - (b - a) / 2:
            return OptimizeResult(x, f_x, iteration, True)

        var take_golden = True
        if abs(previous_step) > tol1:
            # Fit a parabola through (x, f_x), (w, f_w), (v, f_v).
            var t1 = (x - w) * (f_x - f_v)
            var t2 = (x - v) * (f_x - f_w)
            var numerator = (x - v) * t2 - (x - w) * t1
            var denominator = 2 * (t2 - t1)
            if denominator > 0:
                numerator = -numerator
            denominator = abs(denominator)
            var before_last = previous_step
            previous_step = step

            # Accept the parabolic step only if it lands inside the
            # interval and is less than half the step before last --
            # otherwise the parabola is not describing this function and
            # the fallback is the honest move.
            if (
                numerator > denominator * (a - x)
                and numerator < denominator * (b - x)
                and abs(numerator) < abs(denominator * before_last / 2)
            ):
                step = numerator / denominator
                var landing = x + step
                if landing - a < tol2 or b - landing < tol2:
                    step = tol1 if middle - x >= 0 else -tol1
                take_golden = False

        if take_golden:
            # Step into the *larger* of the two sub-intervals, which is the
            # one on the far side of the midpoint from `x`. Stepping the
            # other way shrinks the side already known to be small and the
            # search stalls against the near bound.
            previous_step = (a - x) if x >= middle else (b - x)
            step = _GOLDEN_SECTION * previous_step

        # Never step by less than `tol1`: a shorter one cannot be resolved.
        var trial: Float64
        if abs(step) >= tol1:
            trial = x + step
        else:
            trial = x + (tol1 if step >= 0 else -tol1)
        var f_trial = f[_P](_P(trial)).v

        if f_trial <= f_x:
            if trial >= x:
                a = x
            else:
                b = x
            v = w
            f_v = f_w
            w = x
            f_w = f_x
            x = trial
            f_x = f_trial
        else:
            if trial < x:
                a = trial
            else:
                b = trial
            if f_trial <= f_w or w == x:
                v = w
                f_v = f_w
                w = trial
                f_w = f_trial
            elif f_trial <= f_v or v == x or v == w:
                v = trial
                f_v = f_trial

    return OptimizeResult(x, f_x, max_iter, False)


def brent[
    f: def[U: FloatLike](U) thin -> U,
](
    xa: Float64 = 0,
    xb: Float64 = 1,
    tol: Float64 = _MINIMIZER_TOL,
    max_iter: Int = 500,
) -> OptimizeResult:
    """Minimize a one-variable `f` by Brent's method. `scipy.optimize.brent`,
    and `scipy.optimize.minimize_scalar(method="Brent")`.

    `xa` and `xb` are a *downhill direction*, not a bracket: the search
    expands from them until it finds three points that do bracket a minimum,
    then interpolates within those. That is why the defaults `(0, 1)` are
    usable on a problem whose minimum is at 300 -- they only have to say
    which way to walk. Pass a real bracket when one is known and the
    expansion phase costs nothing.

    `converged=False` means one of two different things, and the caller can
    tell them apart from `iterations`: `0` means no bracket was found at all
    (`f` decreased for 80 golden expansions, so it is unbounded below in
    that direction), and anything else means the cap was reached with the
    interval still wider than `tol`.

    The tolerance is on `x` and defaults to `sqrt(eps)` for the reason
    `_MINIMIZER_TOL` documents: asking for less is asking for digits that a
    quadratic minimum does not have.
    """
    var bracket = _bracket_minimum[f](xa, xb)
    if not bracket.found:
        return OptimizeResult(bracket.b, bracket.f_b, 0, False)
    var lower = min(bracket.a, bracket.c)
    var upper = max(bracket.a, bracket.c)
    return _brent_on_interval[f](lower, upper, bracket.b, tol, max_iter)


def golden[
    f: def[U: FloatLike](U) thin -> U,
](
    xa: Float64 = 0,
    xb: Float64 = 1,
    tol: Float64 = _MINIMIZER_TOL,
    max_iter: Int = 500,
) -> OptimizeResult:
    """Minimize a one-variable `f` by golden-section search.
    `scipy.optimize.golden`, and `minimize_scalar(method="Golden")`.

    The same bracketing phase as `brent`, then the plainest possible
    refinement: keep the two interior points at the golden ratio, discard
    the worse end, and repeat. Each iteration costs exactly one evaluation
    and shrinks the interval by a fixed factor of 0.618, so the iteration
    count is knowable in advance from the starting width and `tol`.

    `brent` is the one to reach for by default -- it takes this same step
    whenever interpolation is not available and beats it whenever it is.
    `golden` is here because that predictable linear shrink is occasionally
    what is wanted, and because it has no failure mode of its own to reason
    about.
    """
    var bracket = _bracket_minimum[f](xa, xb)
    if not bracket.found:
        return OptimizeResult(bracket.b, bracket.f_b, 0, False)

    var lower = min(bracket.a, bracket.c)
    var upper = max(bracket.a, bracket.c)
    return _golden_on_interval[f](lower, upper, tol, max_iter)


def _golden_on_interval[
    f: def[U: FloatLike](U) thin -> U,
](
    lower: Float64, upper: Float64, tol: Float64, max_iter: Int
) -> OptimizeResult:
    """Golden-section refinement of a known interval. Shared by `golden` and
    nothing else so far; kept separate so `golden`'s own body is the
    bracketing decision and this is the loop."""
    comptime shrink = 0.6180339887498949

    var a = lower
    var b = upper
    var x1 = b - shrink * (b - a)
    var x2 = a + shrink * (b - a)
    var f1 = f[_P](_P(x1)).v
    var f2 = f[_P](_P(x2)).v

    for iteration in range(max_iter):
        if abs(b - a) < tol:
            var best = x1 if f1 < f2 else x2
            var f_best = f1 if f1 < f2 else f2
            return OptimizeResult(best, f_best, iteration, True)
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - shrink * (b - a)
            f1 = f[_P](_P(x1)).v
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + shrink * (b - a)
            f2 = f[_P](_P(x2)).v

    var best = x1 if f1 < f2 else x2
    var f_best = f1 if f1 < f2 else f2
    return OptimizeResult(best, f_best, max_iter, False)


def fminbound[
    f: def[U: FloatLike](U) thin -> U,
](
    lower: Float64,
    upper: Float64,
    tol: Float64 = 1e-5,
    max_iter: Int = 500,
) -> OptimizeResult:
    """Minimize a one-variable `f` *inside* `[lower, upper]`.
    `scipy.optimize.fminbound`, and `minimize_scalar(method="Bounded")`.

    The same engine as `brent` with the bracketing phase removed, because
    the bounds already are the interval and the answer is not allowed to
    leave it. That is the whole difference, and it is the right choice
    whenever `f` is undefined or meaningless outside a range -- a bracket
    search would happily walk out of it.

    The minimum may sit *on* a bound, in which case the returned `x`
    approaches it from inside and never crosses.

    `tol` defaults to `1e-5` rather than `brent`'s `sqrt(eps)`, which is
    SciPy's choice for this method and is kept for parity. Pass
    `tol=1.48e-8` for the tighter one.
    """
    var start = lower + _GOLDEN_SECTION * (upper - lower)
    return _brent_on_interval[f](lower, upper, start, tol, max_iter)


def minimize_scalar[
    f: def[U: FloatLike](U) thin -> U,
    method: StaticString = "brent",
](
    bracket: Optional[Tuple[Float64, Float64]] = None,
    bounds: Optional[Tuple[Float64, Float64]] = None,
    tol: Optional[Float64] = None,
    max_iter: Optional[Int] = None,
) raises -> OptimizeResult:
    """Minimize a one-variable `f`. `scipy.optimize.minimize_scalar`.

    | `method` | Runs | Wants |
    | --- | --- | --- |
    | `"brent"` (default) | `brent` | `bracket`, a downhill direction; defaults to `(0, 1)` |
    | `"golden"` | `golden` | the same |
    | `"bounded"` | `fminbound` | `bounds`, and they are mandatory |

    **`bracket` and `bounds` are not the same argument.** A `bracket` says
    which way to walk and the search may leave it; `bounds` are a
    constraint the answer may not leave. Passing `bracket` to `"bounded"`,
    or omitting `bounds` from it, raises -- silently reinterpreting one as
    the other is how a constrained problem quietly returns an
    out-of-range answer.

    **`tol` defaults per method.** `"brent"` and `"golden"` use
    `sqrt(eps)`, the floor a quadratic minimum imposes on locating `x`;
    `"bounded"` uses SciPy's looser `1e-5` for that method. See `minimize`
    for why the defaults are per method rather than shared, and for why an
    unrecognized `method` raises rather than failing to compile.
    """
    comptime if method == "brent" or method == "golden":
        if bounds:
            raise Error(
                "minimize_scalar: method '",
                method,
                (
                    "' takes 'bracket', not 'bounds' -- a bracket is a"
                    " direction to search in and may be left, bounds are a"
                    " constraint that may not. Use method='bounded' to"
                    " constrain the answer."
                ),
            )
        var start: Float64 = 0
        var second: Float64 = 1
        if bracket:
            start = bracket.value()[0]
            second = bracket.value()[1]
        var resolved_tol = tol.value() if tol else _MINIMIZER_TOL
        var resolved_iter = max_iter.value() if max_iter else 500
        comptime if method == "brent":
            return brent[f](start, second, resolved_tol, resolved_iter)
        else:
            return golden[f](start, second, resolved_tol, resolved_iter)
    elif method == "bounded":
        if not bounds:
            raise Error(
                "minimize_scalar: method 'bounded' requires 'bounds'"
                " -- there is nothing to bound the search to otherwise."
            )
        var pair = bounds.value()
        if not (pair[0] < pair[1]):
            raise Error(
                "minimize_scalar: 'bounds' must be increasing, got a lower"
                " bound at or above the upper one"
            )
        return fminbound[f](
            pair[0],
            pair[1],
            tol.value() if tol else 1e-5,
            max_iter.value() if max_iter else 500,
        )
    else:
        raise Error(
            "minimize_scalar: unknown method '",
            method,
            "'; expected 'brent', 'golden' or 'bounded'",
        )


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
        f_x = _value_and_grad[n_vars, f](x, grad)

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
            if (
                _evaluate_at[n_vars, f](candidate)
                <= f_x + Float64(1e-4) * step * directional
            ):
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

        var grad_new = Array[Float64, n_vars](fill=0)
        _ = _value_and_grad[n_vars, f](candidate, grad_new)
        var y = Array[Float64, n_vars](fill=0)
        for i in range(n_vars):
            y[i] = grad_new[i] - grad[i]

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


def cg[
    n_vars: Int,
    f: def[U: FloatLike](Array[U, n_vars]) thin -> U,
](
    x0: Array[Float64, n_vars],
    tol: Float64 = 1e-8,
    max_iter: Int = 200,
) -> MinimizeResult[n_vars]:
    """Minimize `f` by nonlinear conjugate gradients (Polak-Ribiere).
    `scipy.optimize.minimize(method="CG")`.

    The same exact gradient `bfgs` uses, from one `Gradient[_P, n_vars]`
    evaluation per iteration. What differs is the memory: `bfgs` carries an
    `n_vars x n_vars` inverse Hessian and this carries one direction vector,
    so the per-iteration cost is `O(n_vars)` against `O(n_vars^2)`. It is
    the method to reach for when `n_vars` is large enough that the matrix is
    the expense, and the one to avoid otherwise -- without curvature
    information it takes more iterations to get to the same place.

    **The line search is not `bfgs`'s.** `bfgs` backtracks on the Armijo
    condition alone, which is enough there because `H` supplies the scale of
    a good step. CG has no such model, and a backtracking search can only
    ever *shorten* its trial step, so a CG that starts at 1 and halves
    stalls in a curved valley with the gradient still large -- measurably:
    on Rosenbrock from `(-1.2, 1)` it is still at `f = 4.1` after 5,000
    iterations. `_wolfe_step` supplies the strong Wolfe conditions instead,
    which is what nonlinear CG actually requires, and the same problem then
    converges. That helper's docstring carries the details.

    Two standard safeguards, both of which matter more here than the
    formula does:

    - The `beta` is Polak-Ribiere **clamped at zero** (`PR+`). A negative
      `beta` means the previous direction is no longer helping, and using
      it unclamped is what makes plain Polak-Ribiere fail to converge on
      some problems; clamping restarts at steepest descent instead.
    - The direction is reset to steepest descent every `n_vars` iterations
      regardless. Conjugacy is a property of a quadratic, and it decays on
      anything else, so a periodic restart bounds how stale the direction
      can get.

    A direction that is not downhill at all -- which rounding can produce
    near the minimum -- is also reset rather than searched along, since the
    line search would reject every step and report a false failure.

    Convergence is `max|grad| < tol`, the same test and the same default as
    `bfgs`, so the two are directly comparable. `nelder_mead`'s is not; see
    its docstring.
    """

    var x = x0.copy()
    var grad = Array[Float64, n_vars](fill=0)
    var f_x = _value_and_grad[n_vars, f](x, grad)
    var grad_norm: Float64 = 0
    for i in range(n_vars):
        grad_norm = max(grad_norm, abs(grad[i]))

    var direction = Array[Float64, n_vars](fill=0)
    for i in range(n_vars):
        direction[i] = -grad[i]

    # The previous accepted step and the slope it was accepted at, which
    # together set the next iteration's first trial step. Zero means "no
    # previous step yet"; see the guess below.
    var previous_step: Float64 = 0
    var previous_directional: Float64 = 0

    for iteration in range(max_iter):
        if grad_norm < tol:
            return MinimizeResult[n_vars](x^, f_x, grad_norm, iteration, True)

        var directional: Float64 = 0
        for i in range(n_vars):
            directional += grad[i] * direction[i]

        # Rounding near the minimum can leave the carried direction
        # pointing uphill. Steepest descent always points downhill unless
        # the gradient is zero, which the tolerance test above already
        # caught, so this restart cannot loop.
        if directional >= 0:
            directional = 0
            for i in range(n_vars):
                direction[i] = -grad[i]
                directional -= grad[i] * grad[i]

        # The first trial step. `bfgs` can start at 1 every iteration because
        # its `H` carries the problem's scale; CG carries none, so the guess
        # is Nocedal & Wright (3.60) -- the previous step rescaled by the
        # ratio of the slopes -- with the first iteration falling back to a
        # step that moves `x` by about one unit.
        var guess: Float64
        if previous_step > 0 and directional < 0:
            guess = previous_step * previous_directional / directional
        else:
            var longest: Float64 = 0
            for i in range(n_vars):
                longest = max(longest, abs(direction[i]))
            guess = 1 / longest if longest > 1 else 1

        var step = _wolfe_step[n_vars, f](x, direction, f_x, directional, guess)
        if step <= 0:
            return MinimizeResult[n_vars](
                x^, f_x, grad_norm, iteration + 1, False
            )
        var candidate = Array[Float64, n_vars](fill=0)
        for i in range(n_vars):
            candidate[i] = x[i] + step * direction[i]
        previous_step = step
        previous_directional = directional

        var grad_new = Array[Float64, n_vars](fill=0)
        var f_new = _value_and_grad[n_vars, f](candidate, grad_new)

        # Polak-Ribiere: beta = grad_new . (grad_new - grad) / (grad . grad),
        # clamped at zero, and forced to zero on the periodic restart.
        var numerator: Float64 = 0
        var denominator: Float64 = 0
        for i in range(n_vars):
            numerator += grad_new[i] * (grad_new[i] - grad[i])
            denominator += grad[i] * grad[i]
        var beta: Float64 = 0
        if denominator > 0 and (iteration + 1) % n_vars != 0:
            beta = max(Float64(0), numerator / denominator)

        for i in range(n_vars):
            direction[i] = -grad_new[i] + beta * direction[i]
            x[i] = candidate[i]
            grad[i] = grad_new[i]
        f_x = f_new
        grad_norm = 0
        for i in range(n_vars):
            grad_norm = max(grad_norm, abs(grad[i]))

    return MinimizeResult[n_vars](x^, f_x, grad_norm, max_iter, False)


def nelder_mead[
    n_vars: Int,
    f: def[U: FloatLike](Array[U, n_vars]) thin -> U,
](
    x0: Array[Float64, n_vars],
    tol: Float64 = 1e-10,
    max_iter: Int = 1000,
) -> MinimizeResult[n_vars]:
    """Minimize `f` by the Nelder-Mead simplex method.
    `scipy.optimize.minimize(method="Nelder-Mead")`.

    `bfgs` is the better choice whenever the objective is smooth, since it
    has the exact gradient available and this deliberately ignores it. What
    this is for is the objective that has a gradient *and should not be
    trusted*: numax's own branchless kernels are built from `max_of`,
    `min_of` and `copysign` blends, which have kinks, and a derivative at a
    kink points somewhere a line search will not go. A simplex only ever
    compares function values, so a kink is not a special case for it.

    The standard coefficients: reflect by 1, expand by 2, contract by 0.5,
    shrink by 0.5. The initial simplex perturbs each coordinate by 5% of
    itself, or by a small absolute step where the coordinate is zero, which
    is what SciPy does and matters more than it looks -- a simplex badly
    scaled against the problem spends its first several iterations only
    fixing its own shape.

    Convergence is the spread of function values across the simplex,
    `max(f) - min(f) < tol`. That is the honest test for a method with no
    derivative to look at, and it is why `tol` here is not comparable to
    `bfgs`'s gradient tolerance.

    `MinimizeResult.grad_norm` is still filled in, by one extra evaluation
    at `Gradient` after the loop. Nelder-Mead does not use it, but a
    caller deserves to know how flat the point it stopped at actually is --
    a large gradient norm at a converged simplex means the simplex
    collapsed rather than found anything.
    """
    comptime n_points = n_vars + 1

    var points = Array[Float64, n_points * n_vars](fill=0)
    var values = Array[Float64, n_points](fill=0)

    for j in range(n_vars):
        points[j] = x0[j]
    for i in range(1, n_points):
        for j in range(n_vars):
            points[i * n_vars + j] = x0[j]
        var coordinate = x0[i - 1]
        points[i * n_vars + (i - 1)] = (
            coordinate * 1.05 if coordinate != 0 else 0.00025
        )

    for i in range(n_points):
        values[i] = _evaluate[n_vars, f](points, i)

    var iteration = 0
    var converged = False
    for step in range(max_iter):
        iteration = step

        # Insertion sort: `n_points` is small and the simplex is nearly
        # sorted already after the first pass.
        for i in range(1, n_points):
            var j = i
            while j > 0 and values[j - 1] > values[j]:
                var swap_value = values[j - 1]
                values[j - 1] = values[j]
                values[j] = swap_value
                for k in range(n_vars):
                    var swap_point = points[(j - 1) * n_vars + k]
                    points[(j - 1) * n_vars + k] = points[j * n_vars + k]
                    points[j * n_vars + k] = swap_point
                j -= 1

        if values[n_points - 1] - values[0] < tol:
            converged = True
            break

        # Centroid of everything but the worst point.
        var centroid = Array[Float64, n_vars](fill=0)
        for i in range(n_points - 1):
            for j in range(n_vars):
                centroid[j] += points[i * n_vars + j] / Float64(n_vars)

        var reflected = Array[Float64, n_vars](fill=0)
        for j in range(n_vars):
            reflected[j] = (
                centroid[j] + centroid[j] - points[(n_points - 1) * n_vars + j]
            )
        var reflected_value = _evaluate_at[n_vars, f](reflected)

        if reflected_value < values[0]:
            var expanded = Array[Float64, n_vars](fill=0)
            for j in range(n_vars):
                expanded[j] = centroid[j] + 2 * (reflected[j] - centroid[j])
            var expanded_value = _evaluate_at[n_vars, f](expanded)
            if expanded_value < reflected_value:
                _replace_worst[n_vars, n_points](points, expanded)
                values[n_points - 1] = expanded_value
            else:
                _replace_worst[n_vars, n_points](points, reflected)
                values[n_points - 1] = reflected_value
            continue

        if reflected_value < values[n_points - 2]:
            _replace_worst[n_vars, n_points](points, reflected)
            values[n_points - 1] = reflected_value
            continue

        # Contract toward whichever of the reflection and the worst point
        # is better, which is what keeps a contraction from stepping the
        # wrong way across a valley.
        var toward_reflection = reflected_value < values[n_points - 1]
        var contracted = Array[Float64, n_vars](fill=0)
        for j in range(n_vars):
            var target = reflected[j] if toward_reflection else points[
                (n_points - 1) * n_vars + j
            ]
            contracted[j] = centroid[j] + 0.5 * (target - centroid[j])
        var contracted_value = _evaluate_at[n_vars, f](contracted)

        if contracted_value < min(reflected_value, values[n_points - 1]):
            _replace_worst[n_vars, n_points](points, contracted)
            values[n_points - 1] = contracted_value
            continue

        # Nothing helped: shrink the whole simplex toward the best point.
        for i in range(1, n_points):
            for j in range(n_vars):
                points[i * n_vars + j] = points[j] + 0.5 * (
                    points[i * n_vars + j] - points[j]
                )
            values[i] = _evaluate[n_vars, f](points, i)

    var best = Array[Float64, n_vars](fill=0)
    for j in range(n_vars):
        best[j] = points[j]

    var seeded = Array[Gradient[_P, n_vars], n_vars](
        fill=Gradient[_P, n_vars].constant(0.0)
    )
    for i in range(n_vars):
        seeded[i] = Gradient[_P, n_vars].variable(_P(best[i]), i)
    var evaluated = f[Gradient[_P, n_vars]](seeded^)
    var grad_norm: Float64 = 0
    for i in range(n_vars):
        grad_norm = max(grad_norm, abs(evaluated.grad[i].v))

    return MinimizeResult[n_vars](
        best^, values[0], grad_norm, iteration, converged
    )


def minimize[
    n_vars: Int,
    f: def[U: FloatLike](Array[U, n_vars]) thin -> U,
    method: StaticString = "bfgs",
](
    x0: Array[Float64, n_vars],
    tol: Optional[Float64] = None,
    max_iter: Optional[Int] = None,
) raises -> MinimizeResult[n_vars]:
    """Minimize `f` from `x0`. `scipy.optimize.minimize`.

    The SciPy-shaped entry point over the minimizers in this module, with
    SciPy's own method spelling:

    | `method` | Runs | Uses the gradient |
    | --- | --- | --- |
    | `"bfgs"` (default) | `bfgs` | yes, exactly, plus an `n x n` inverse Hessian |
    | `"cg"` | `cg` | yes, exactly, no matrix |
    | `"nelder-mead"` | `nelder_mead` | no, on purpose |

    Each is also callable directly under its own name, and that is the
    better spelling when the method is not a choice the caller is making --
    `bfgs[2, rosenbrock](start)` says which algorithm runs without a reader
    resolving a string.

    **`tol` and `max_iter` default per method, not globally.** Passing
    nothing gives each method the default its own docstring documents, and
    that is deliberate: the three do not measure the same quantity. `bfgs`
    and `cg` test `max|grad| < 1e-8`; `nelder_mead` tests the spread of
    function values across the simplex against `1e-10`, because a method
    with no derivative has nothing else to look at. A single shared default
    would silently change one method's stopping rule, so there is no
    `tol: Float64 = 1e-8` here.

    **An unrecognized `method` raises rather than failing to compile**, and
    that is a Mojo limitation rather than a choice. `method` is a
    compile-time parameter and `comptime if method == "bfgs"` dispatches on
    it fine, but a `where` clause listing the alternatives cannot be
    discharged: both the `StringLiteral` -> `StaticString` conversion and
    `StringSpan.__eq__` are non-builtin calls, which constraint evaluation
    refuses. Nor can the unreachable branch reject the value at compile time
    -- an untaken `comptime if` branch is still constraint-checked, so a
    deliberately unsatisfiable call there rejects the *valid* methods too.
    What is left is this: a typo is an `Error` naming the bad method on the
    first call, never a silent fallthrough to a different algorithm.
    """
    comptime if method == "bfgs":
        return bfgs[n_vars, f](
            x0,
            tol.value() if tol else 1e-8,
            max_iter.value() if max_iter else 200,
        )
    elif method == "cg":
        return cg[n_vars, f](
            x0,
            tol.value() if tol else 1e-8,
            max_iter.value() if max_iter else 200,
        )
    elif method == "nelder-mead":
        return nelder_mead[n_vars, f](
            x0,
            tol.value() if tol else 1e-10,
            max_iter.value() if max_iter else 1000,
        )
    else:
        raise Error(
            "minimize: unknown method '",
            method,
            "'; expected 'bfgs', 'cg' or 'nelder-mead'",
        )


def _slope_along[
    n_vars: Int,
    f: def[U: FloatLike](Array[U, n_vars]) thin -> U,
](
    x: Array[Float64, n_vars],
    direction: Array[Float64, n_vars],
    alpha: Float64,
    mut value: Float64,
) -> Float64:
    """`phi'(alpha)` for `phi(alpha) = f(x + alpha * direction)`, with
    `phi(alpha)` left in `value`.

    One `Gradient` evaluation gives both, so a line search that needs the
    slope costs no more than one that needs only the value -- which is what
    makes a Wolfe search affordable here and is the reason `_wolfe_step`
    exists rather than a cheaper backtracking loop.
    """
    var trial = Array[Float64, n_vars](fill=0)
    for i in range(n_vars):
        trial[i] = x[i] + alpha * direction[i]
    var grad = Array[Float64, n_vars](fill=0)
    value = _value_and_grad[n_vars, f](trial, grad)
    var slope: Float64 = 0
    for i in range(n_vars):
        slope += grad[i] * direction[i]
    return slope


def _wolfe_step[
    n_vars: Int,
    f: def[U: FloatLike](Array[U, n_vars]) thin -> U,
](
    x: Array[Float64, n_vars],
    direction: Array[Float64, n_vars],
    f_x: Float64,
    directional: Float64,
    first_guess: Float64,
) -> Float64:
    """A step length satisfying the **strong Wolfe** conditions, or `0` if
    none was found. Nocedal & Wright algorithms 3.5 and 3.6.

    Two conditions, and `cg` needs both where `bfgs` needs only the first:

    - **Armijo** (`c1 = 1e-4`): the step must reduce `f` by at least a
      fraction of what the slope predicted. This alone is what `bfgs`
      backtracks for, and it is enough there because `H` supplies the scale
      of a good step.
    - **Curvature** (`c2 = 0.1`): the slope at the new point must be
      flatter than at the old one, `|phi'(a)| <= c2 * |phi'(0)|`. Nonlinear
      CG needs this. Without it the Polak-Ribiere direction is not
      guaranteed to be a descent direction at all, and in practice the
      steps collapse: a backtracking search only ever *shortens* a trial
      step, so once CG needs a longer one -- which it does in a curved
      valley, having no curvature model to lengthen it for free -- it stalls
      with the gradient still large. `c2 = 0.1` rather than BFGS's usual
      `0.9` is the standard CG choice, and it is the tighter one.

    The shape is the textbook two-phase search: **bracket** by increasing
    the trial step until an interval known to contain an acceptable point
    appears, then **zoom** into that interval by bisection. Bisection rather
    than interpolation on purpose -- it cannot produce a point outside the
    bracket, and the extra iterations are cheap next to an evaluation of
    `f`.
    """
    comptime c1 = 1e-4
    comptime c2 = 0.1

    var lo: Float64 = 0
    var lo_value = f_x
    var hi: Float64 = 0
    var bracketed = False

    var previous = Float64(0)
    var previous_value = f_x
    var alpha = first_guess

    # Phase 1: bracket.
    for attempt in range(40):
        var value = f_x
        var slope = _slope_along[n_vars, f](x, direction, alpha, value)

        if value > f_x + c1 * alpha * directional or (
            attempt > 0 and value >= previous_value
        ):
            # `alpha` is too long: an acceptable point lies between the
            # previous trial and this one.
            lo = previous
            lo_value = previous_value
            hi = alpha
            bracketed = True
            break
        if abs(slope) <= -c2 * directional:
            return alpha
        if slope >= 0:
            # The slope has turned uphill, so the minimum along the ray was
            # passed; the bracket is this interval reversed.
            lo = alpha
            lo_value = value
            hi = previous
            bracketed = True
            break
        previous = alpha
        previous_value = value
        alpha = alpha * 2

    if not bracketed:
        return 0

    # Phase 2: zoom.
    for _ in range(60):
        var mid = (lo + hi) / 2
        if mid <= 0:
            return 0
        var value = f_x
        var slope = _slope_along[n_vars, f](x, direction, mid, value)

        if value > f_x + c1 * mid * directional or value >= lo_value:
            hi = mid
        else:
            if abs(slope) <= -c2 * directional:
                return mid
            if slope * (hi - lo) >= 0:
                hi = lo
            lo = mid
            lo_value = value

    # The bracket collapsed without meeting the curvature condition. `lo` is
    # still an Armijo-acceptable point whenever it is not the origin, so
    # return it rather than reporting failure on a step that does decrease
    # the objective.
    return lo


def _value_and_grad[
    n_vars: Int,
    f: def[U: FloatLike](Array[U, n_vars]) thin -> U,
](x: Array[Float64, n_vars], mut grad: Array[Float64, n_vars]) -> Float64:
    """`f(x)` and every partial derivative at `x`, from one evaluation.

    The whole point of the conformer tier in one function: seeding each
    coordinate as a `Gradient[_P, n_vars]` variable and calling `f` once
    returns the value and all `n_vars` partials by the chain rule, exactly.
    A forward difference would cost `n_vars + 1` calls and cap accuracy near
    `sqrt(eps)`.

    `grad` is filled in place rather than returned beside the value because
    a caller that already owns the destination -- which every driver here
    does, across iterations -- should not allocate a second one per step.
    """
    var seeded = Array[Gradient[_P, n_vars], n_vars](
        fill=Gradient[_P, n_vars].constant(0.0)
    )
    for i in range(n_vars):
        seeded[i] = Gradient[_P, n_vars].variable(_P(x[i]), i)
    var evaluated = f[Gradient[_P, n_vars]](seeded^)
    for i in range(n_vars):
        grad[i] = evaluated.grad[i].v
    return evaluated.value.v


def _evaluate_at[
    n_vars: Int,
    f: def[U: FloatLike](Array[U, n_vars]) thin -> U,
](x: Array[Float64, n_vars]) -> Float64:
    var as_plain = Array[_P, n_vars](fill=_P.constant(0.0))
    for i in range(n_vars):
        as_plain[i] = _P(x[i])
    return f[_P](as_plain^).v


def _evaluate[
    n_vars: Int,
    f: def[U: FloatLike](Array[U, n_vars]) thin -> U,
](points: Array[Float64, (n_vars + 1) * n_vars], row: Int) -> Float64:
    var as_plain = Array[_P, n_vars](fill=_P.constant(0.0))
    for i in range(n_vars):
        as_plain[i] = _P(points[row * n_vars + i])
    return f[_P](as_plain^).v


def _replace_worst[
    n_vars: Int, n_points: Int
](mut points: Array[Float64, n_points * n_vars], x: Array[Float64, n_vars]):
    for j in range(n_vars):
        points[(n_points - 1) * n_vars + j] = x[j]


def _lm_step[
    n_params: Int, n_resid: Int
](
    jacobian: Array[Float64, n_resid * n_params],
    residual: Array[Float64, n_resid],
    damping: Float64,
) -> Array[Float64, n_params]:
    """The Levenberg-Marquardt step: solve `(J.T J + damping * diag(J.T J))
    d = -J.T r`.

    The damping is applied as a multiple of each diagonal entry rather than
    of the identity, which makes the step invariant to rescaling a
    parameter -- fitting a rate in seconds and in milliseconds then takes
    the same path. A small absolute term is added as well, since a
    parameter the residuals do not depend on at all has a zero diagonal
    that no multiple of itself can lift.
    """
    var normal = Array[_P, n_params * n_params](fill=_P.constant(0.0))
    var gradient = Array[_P, n_params](fill=_P.constant(0.0))

    for i in range(n_params):
        for j in range(n_params):
            var total: Float64 = 0
            for k in range(n_resid):
                total += jacobian[k * n_params + i] * jacobian[k * n_params + j]
            normal[i * n_params + j] = _P(total)
        var slope: Float64 = 0
        for k in range(n_resid):
            slope += jacobian[k * n_params + i] * residual[k]
        gradient[i] = _P(-slope)

    for i in range(n_params):
        var diagonal = normal[i * n_params + i].v
        normal[i * n_params + i] = _P(diagonal * (1 + damping) + 1e-12)

    # Positive definite by construction once damped, so Cholesky needs no
    # pivoting and the unpivoted factorization is exact for it.
    var factor = cholesky[_P, n_params](normal)
    var step = cholesky_solve[_P, n_params](factor, gradient)

    var out = Array[Float64, n_params](fill=0)
    for i in range(n_params):
        out[i] = step[i].v
    return out^


def least_squares[
    n_params: Int,
    n_resid: Int,
    residuals: def[U: FloatLike](Array[U, n_params]) thin -> Array[U, n_resid],
](
    x0: Array[Float64, n_params],
    tol: Float64 = 1e-10,
    max_iter: Int = 100,
) -> MinimizeResult[n_params]:
    """Minimize `sum(residuals(x)**2) / 2` by Levenberg-Marquardt.
    `scipy.optimize.least_squares`.

    **The Jacobian is exact, and it costs one evaluation.** `residuals` is
    called at `Gradient[_P, n_params]`, so every residual comes back
    carrying all `n_params` of its partial derivatives -- the entire
    `n_resid x n_params` Jacobian from a single call, by the chain rule.
    SciPy's default is a forward difference costing `n_params + 1` calls
    and capping accuracy near `sqrt(eps)`; there is no step size here to
    choose and nothing to trade off.

    Levenberg-Marquardt interpolates between Gauss-Newton and gradient
    descent by damping the normal equations. A step that reduces the cost
    is accepted and the damping is relaxed, moving toward Gauss-Newton's
    quadratic convergence; a step that does not is rejected and the damping
    raised, shortening the step toward the gradient direction. That is what
    makes it converge from a poor starting point where plain Gauss-Newton
    diverges.

    Convergence is `max|J.T r| < tol`, the same first-order condition
    `bfgs` uses, reported in `MinimizeResult.grad_norm`. `f_x` is the cost
    `sum(r**2) / 2`, matching what SciPy reports rather than the raw sum.

    Bounds, robust loss functions and a sparse Jacobian are all out of
    scope. Bounds in particular would need an active-set or trust-region
    treatment rather than a clamp, since clamping an LM step silently
    breaks the step's own model of the cost.
    """
    var x = x0.copy()
    var cost: Float64 = 0
    var grad_norm: Float64 = 0
    var damping: Float64 = 1e-3

    for iteration in range(max_iter):
        var seeded = Array[Gradient[_P, n_params], n_params](
            fill=Gradient[_P, n_params].constant(0.0)
        )
        for i in range(n_params):
            seeded[i] = Gradient[_P, n_params].variable(_P(x[i]), i)
        var evaluated = residuals[Gradient[_P, n_params]](seeded^)

        var residual = Array[Float64, n_resid](fill=0)
        var jacobian = Array[Float64, n_resid * n_params](fill=0)
        cost = 0
        for k in range(n_resid):
            residual[k] = evaluated[k].value.v
            cost += residual[k] * residual[k]
            for j in range(n_params):
                jacobian[k * n_params + j] = evaluated[k].grad[j].v
        cost = cost / 2

        grad_norm = 0
        for j in range(n_params):
            var slope: Float64 = 0
            for k in range(n_resid):
                slope += jacobian[k * n_params + j] * residual[k]
            grad_norm = max(grad_norm, abs(slope))
        if grad_norm < tol:
            return MinimizeResult[n_params](
                x^, cost, grad_norm, iteration, True
            )

        var accepted = False
        var candidate = Array[Float64, n_params](fill=0)
        for _ in range(30):
            var step = _lm_step[n_params, n_resid](jacobian, residual, damping)
            for i in range(n_params):
                candidate[i] = x[i] + step[i]

            var as_plain = Array[_P, n_params](fill=_P.constant(0.0))
            for i in range(n_params):
                as_plain[i] = _P(candidate[i])
            var trial = residuals[_P](as_plain^)
            var trial_cost: Float64 = 0
            for k in range(n_resid):
                trial_cost += trial[k].v * trial[k].v
            trial_cost = trial_cost / 2

            if trial_cost < cost:
                accepted = True
                damping = max(damping / 3, 1e-12)
                break
            damping = damping * 3
        if not accepted:
            return MinimizeResult[n_params](
                x^, cost, grad_norm, iteration + 1, False
            )

        for i in range(n_params):
            x[i] = candidate[i]

    return MinimizeResult[n_params](x^, cost, grad_norm, max_iter, False)


def curve_fit[
    n_params: Int,
    n_points: Int,
    model: def[U: FloatLike](U, Array[U, n_params]) thin -> U,
](
    xdata: Array[Float64, n_points],
    ydata: Array[Float64, n_points],
    p0: Array[Float64, n_params],
    tol: Float64 = 1e-10,
    max_iter: Int = 100,
) -> MinimizeResult[n_params]:
    """Fit `model(x, params)` to `(xdata, ydata)` by least squares.
    `scipy.optimize.curve_fit`, first return value.

    The residuals are `model(x_i, p) - y_i`, so this is `least_squares` for
    the case that motivates it, with the data supplied at run time. It is a
    separate function rather than a wrapper because `least_squares` takes
    its residuals as a compile-time parameter, which a non-capturing
    function cannot reach runtime data through; here the data is an
    argument to the fit and the model never has to see it.

    `model` is an ordinary `FloatLike` kernel -- the same one that
    evaluates the fitted curve afterward at `Plain` -- and this evaluates
    it at `Gradient` to get the exact Jacobian in one call per point. The
    `x` argument is passed as a constant, so only the parameters carry
    derivatives.

    The parameter covariance SciPy returns second is not computed. It is
    `(J.T J)^-1` scaled by the residual variance, which needs an
    assumption about the noise that belongs to the caller's problem rather
    than to the fit.
    """
    var p = p0.copy()
    var cost: Float64 = 0
    var grad_norm: Float64 = 0
    var damping: Float64 = 1e-3

    for iteration in range(max_iter):
        var seeded = Array[Gradient[_P, n_params], n_params](
            fill=Gradient[_P, n_params].constant(0.0)
        )
        for i in range(n_params):
            seeded[i] = Gradient[_P, n_params].variable(_P(p[i]), i)

        var residual = Array[Float64, n_points](fill=0)
        var jacobian = Array[Float64, n_points * n_params](fill=0)
        cost = 0
        for k in range(n_points):
            var predicted = model[Gradient[_P, n_params]](
                Gradient[_P, n_params].constant(xdata[k]), seeded.copy()
            )
            residual[k] = predicted.value.v - ydata[k]
            cost += residual[k] * residual[k]
            for j in range(n_params):
                jacobian[k * n_params + j] = predicted.grad[j].v
        cost = cost / 2

        grad_norm = 0
        for j in range(n_params):
            var slope: Float64 = 0
            for k in range(n_points):
                slope += jacobian[k * n_params + j] * residual[k]
            grad_norm = max(grad_norm, abs(slope))
        if grad_norm < tol:
            return MinimizeResult[n_params](
                p^, cost, grad_norm, iteration, True
            )

        var accepted = False
        var candidate = Array[Float64, n_params](fill=0)
        for _ in range(30):
            var step = _lm_step[n_params, n_points](jacobian, residual, damping)
            for i in range(n_params):
                candidate[i] = p[i] + step[i]

            var as_plain = Array[_P, n_params](fill=_P.constant(0.0))
            for i in range(n_params):
                as_plain[i] = _P(candidate[i])
            var trial_cost: Float64 = 0
            for k in range(n_points):
                var difference = (
                    model[_P](_P(xdata[k]), as_plain.copy()).v - ydata[k]
                )
                trial_cost += difference * difference
            trial_cost = trial_cost / 2

            if trial_cost < cost:
                accepted = True
                damping = max(damping / 3, 1e-12)
                break
            damping = damping * 3
        if not accepted:
            return MinimizeResult[n_params](
                p^, cost, grad_norm, iteration + 1, False
            )

        for i in range(n_params):
            p[i] = candidate[i]

    return MinimizeResult[n_params](p^, cost, grad_norm, max_iter, False)
