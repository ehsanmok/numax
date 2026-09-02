"""Adaptive integration: quadrature that subdivides, and an ODE solver that
picks its own step size.

**This module is tier 2.** `quad` decides where to subdivide by looking at
the integrand's values, so the amount of work depends on the data and the
control flow branches on it. That is what `numax.quadrature`'s module
docstring rules out for `FloatLike`-generic code, and correctly: two SIMD
lanes integrating different functions would want different grids, and there
is no per-lane way to give them one. So the adaptive rule lives here
instead, `Plain`-only and host-side. See `docs/architecture.md`'s "Two
tiers".

## Tier 1 sibling

`numax.quadrature.gauss_legendre` is the fixed-node rule, and it is not
superseded by anything here -- it is exact for polynomials of degree
`2n-1`, costs exactly `n` evaluations, and runs inside a GPU thread. For a
smooth integrand it is both faster and more accurate than `quad`.

Reach for `quad` when the integrand is *not* smooth on the whole interval:
a kink, a near-singularity, a sharp peak, or a scale that varies by orders
of magnitude across the range. That is precisely the case
`gauss_legendre`'s own docstring says to handle by splitting the interval
by hand -- `quad` is that, done automatically and to a tolerance.

## How the panels are chosen

Each panel is integrated twice: once whole with `gauss_legendre[n]`, and
once as the sum of its two halves. The difference between those two
estimates is the error estimate for that panel -- the standard
subdivision test, and it composes out of `numax.quadrature` rather than
needing its own quadrature rule. A panel is accepted when its error is
below its share of the global tolerance; otherwise both halves go back on
the work list.

The work list is an explicit stack rather than recursion, so the memory
cost is bounded by `max_panels` and visible in the signature instead of
living on the call stack.

Because the panel rule is `gauss_legendre`, which is `FloatLike`-generic,
the integrand is still an ordinary `FloatLike` kernel. `quad` evaluates it
at `Plain` -- the adaptive machinery has no meaning at `Dual` -- but the
same `f` can be integrated at `Dual` by the tier-1 rule to differentiate
under the integral sign, with no second implementation. See
`examples/intermediate/quadrature.mojo`.
"""

from std.collections import Array

from .numeric import FloatLike
from .plain import Plain
from .ode import dopri5_step
from .quadrature import gauss_legendre

# Fixed to float64 for the same two reasons as `numax.optimize`: an error
# tolerance of 1e-10 is meaningless at float32, and Mojo will not accept a
# struct instantiated with a function-level `DType` parameter as a
# `FloatLike` type argument, so a per-call dtype would not compile.
comptime _P = Plain[DType.float64, 1]


@fieldwise_init
struct QuadResult(Copyable, Movable):
    """The integral, plus how confident to be about it.

    `error` is the sum of the accepted panels' own error estimates -- a
    realistic bound in practice, not a rigorous one, since a panel whose
    two estimates happen to agree can still both be wrong (an integrand
    with a feature narrower than the panel is the classic way to fool it).

    `converged` false means `max_panels` ran out with panels still above
    tolerance: `value` is then the best available sum, `error` says how bad
    it might be, and neither is nonsense -- just not to the tolerance
    asked for.
    """

    var value: Float64
    var error: Float64
    var panels: Int
    var converged: Bool


def quad[
    f: def[U: FloatLike](U) thin -> U,
    n: Int = 8,
](
    a: Float64,
    b: Float64,
    tol: Float64 = 1e-10,
    max_panels: Int = 2048,
) -> QuadResult:
    """Integrate `f` over `[a, b]` adaptively, to an absolute tolerance.

    `n` is the Gauss-Legendre order used on each panel; the default 8 is
    accurate enough that subdivision only happens where the integrand
    genuinely needs it.

    Reversed limits are handled by integrating forward and negating, the
    same convention as the fixed-node rules. An empty interval returns
    exactly zero without evaluating `f`.

    ```mojo
    def peaked[U: FloatLike](x: U) -> U:
        # A spike at x = 0.5 that a fixed 8-point rule cannot see.
        var d = x + (-U.constant(0.5))
        return U.one() / (U.constant(1e-4) + d * d)

    var result = quad[peaked](0.0, 1.0)
    ```
    """
    if a == b:
        return QuadResult(0.0, 0.0, 0, True)
    if a > b:
        var flipped = quad[f, n](b, a, tol, max_panels)
        return QuadResult(
            -flipped.value, flipped.error, flipped.panels, flipped.converged
        )

    # The work list, as parallel arrays of panel endpoints. An explicit
    # stack rather than recursion: the bound is visible and the memory is
    # not the call stack's problem.
    var lo = List[Float64](capacity=64)
    var hi = List[Float64](capacity=64)
    lo.append(a)
    hi.append(b)

    var total = 0.0
    var total_error = 0.0
    var panels = 0
    var converged = True

    while len(lo) > 0:
        var panel_lo = lo.pop()
        var panel_hi = hi.pop()
        var mid = (panel_lo + panel_hi) / 2

        var whole = Float64(
            gauss_legendre[_P, f, n](
                _P.constant(panel_lo), _P.constant(panel_hi)
            ).v
        )
        var left = Float64(
            gauss_legendre[_P, f, n](_P.constant(panel_lo), _P.constant(mid)).v
        )
        var right = Float64(
            gauss_legendre[_P, f, n](_P.constant(mid), _P.constant(panel_hi)).v
        )
        var halves = left + right
        var panel_error = abs(halves - whole)

        # A panel's share of the tolerance is proportional to its width, so
        # the accepted panels' errors sum to about `tol` rather than to
        # `tol` times the panel count.
        var share = tol * (panel_hi - panel_lo) / (b - a)

        if panel_error <= share or panels >= max_panels:
            if panel_error > share:
                converged = False
            # The two-half sum is the better estimate of the two, so it is
            # what gets accumulated -- there is no reason to keep the
            # coarser number once both have been computed.
            total += halves
            total_error += panel_error
            panels += 1
        else:
            lo.append(panel_lo)
            hi.append(mid)
            lo.append(mid)
            hi.append(panel_hi)

    return QuadResult(total, total_error, panels, converged)


def quad_vec[
    f: def[U: FloatLike](U) thin -> U,
    n: Int = 8,
](
    a: Float64,
    b: Float64,
    breakpoints: List[Float64],
    tol: Float64 = 1e-10,
) -> QuadResult:
    """`quad` over `[a, b]`, forced to place a panel boundary at each of
    `breakpoints`.

    For an integrand with a known kink or jump, this is strictly better
    than letting the adaptive rule discover it: subdivision can only
    bisect, so a feature at, say, `x = 1/3` is approached but never landed
    on exactly, and every panel straddling it stays inaccurate no matter
    how small it gets. Naming the point up front removes the problem
    instead of throwing panels at it.

    Breakpoints outside `[a, b]` are ignored; duplicates and unsorted
    input are handled. Sums the sub-integrals' errors and reports
    `converged` false if any piece failed.
    """
    var cuts = List[Float64](capacity=len(breakpoints) + 2)
    cuts.append(a)
    for i in range(len(breakpoints)):
        var x = breakpoints[i]
        if x > a and x < b:
            cuts.append(x)
    cuts.append(b)

    # Insertion sort: the breakpoint list is short by construction (one per
    # known feature), so this is not the place for anything cleverer.
    for i in range(1, len(cuts)):
        var key = cuts[i]
        var j = i - 1
        while j >= 0 and cuts[j] > key:
            cuts[j + 1] = cuts[j]
            j -= 1
        cuts[j + 1] = key

    var total = 0.0
    var total_error = 0.0
    var panels = 0
    var converged = True
    for i in range(len(cuts) - 1):
        if cuts[i] == cuts[i + 1]:
            continue
        var piece = quad[f, n](
            cuts[i], cuts[i + 1], tol / Float64(len(cuts) - 1)
        )
        total += piece.value
        total_error += piece.error
        panels += piece.panels
        if not piece.converged:
            converged = False

    return QuadResult(total, total_error, panels, converged)


@fieldwise_init
struct IVPResult(Copyable, Movable):
    """The outcome of an adaptive ODE integration.

    `accepted` and `rejected` are the step counts, and they are the two
    numbers that tell you whether the controller was working: a healthy run
    rejects a small fraction of its steps, and a run that rejects most of
    them is fighting the problem (a stiff system, or a tolerance tighter
    than the arithmetic can deliver).

    `converged` false means `max_steps` ran out before reaching `t1`, so
    `t` is where it actually got to -- which is why `t` is returned at all
    rather than assumed equal to the requested endpoint.
    """

    var t: Float64
    var y: Float64
    var accepted: Int
    var rejected: Int
    var converged: Bool


def solve_ivp[
    f: def[U: FloatLike](U, U) thin -> U,
](
    t0: Float64,
    y0: Float64,
    t1: Float64,
    rtol: Float64 = 1e-8,
    atol: Float64 = 1e-10,
    max_steps: Int = 10000,
) -> IVPResult:
    """Integrate `dy/dt = f(t, y)` from `t0` to `t1` with adaptive step
    control. The tier-2 counterpart of `numax.ode.dopri5`.

    Dormand-Prince 5(4) with the classic proportional controller: each step
    is taken with `numax.ode.dopri5_step`, which returns the 5th-order
    solution and its disagreement with the embedded 4th-order one; that
    error is compared against `atol + rtol*|y|`, and the step is accepted
    or rejected accordingly. The next step size is scaled by
    `(1/error_ratio)**(1/5)`, clamped to a factor of 5 in either direction
    so one anomalous step cannot make the controller wild.

    **Why this is tier 2 and `numax.ode.dopri5` is not.** The step *body*
    is identical -- the same seven stages from the same tableau, shared
    rather than duplicated. What differs is that this decides, per step and
    based on the values, whether to keep the result and how far to go next.
    That is a data-dependent iteration count, so a SIMD `T` whose lanes
    disagreed about acceptance could not be served, and the whole thing
    runs on the host at `Plain` instead.

    The payoff is the usual one for adaptivity: on a solution with a sharp
    transient followed by a smooth tail, a fixed-step integrator has to use
    the transient's step size everywhere. `tests/test_integrate.mojo`
    measures that against `dopri5` on such a problem.

    `f` is still an ordinary `FloatLike` kernel, so the same equation can be
    integrated by the tier-1 `rk4` or `dopri5` inside a GPU kernel -- see
    `examples/advanced/ode.mojo`, which runs an ensemble that way.
    """
    if t0 == t1:
        return IVPResult(t0, y0, 0, 0, True)

    var direction = 1.0 if t1 > t0 else -1.0
    var span = abs(t1 - t0)

    var t = t0
    var y = y0
    # Start at a hundredth of the interval: small enough not to overshoot a
    # transient at the very beginning, large enough not to waste steps
    # crawling out of the start on a smooth problem. The controller
    # corrects either way within a step or two.
    var h = direction * span / 100.0

    var accepted = 0
    var rejected = 0

    for _ in range(max_steps):
        if abs(t - t1) <= 0.0:
            return IVPResult(t, y, accepted, rejected, True)

        # Never step past the endpoint.
        if abs(h) > abs(t1 - t):
            h = t1 - t

        var stepped = dopri5_step[_P, f](
            _P.constant(t), _P.constant(y), _P.constant(h)
        )
        var y_next = Float64(stepped[0].v)
        var error = Float64(stepped[1].v)

        var tolerance = atol + rtol * max(abs(y), abs(y_next))
        # A step whose error estimate underflows to zero is as good as it
        # gets; treat the ratio as tiny rather than dividing by zero.
        var ratio = error / tolerance if tolerance > 0.0 else 0.0

        if ratio <= 1.0:
            t += h
            y = y_next
            accepted += 1
            if abs(t - t1) <= 0.0:
                return IVPResult(t, y, accepted, rejected, True)
        else:
            rejected += 1

        # The order-5 scaling law, with a safety factor and clamps. 0.9
        # keeps the next step slightly inside what the estimate allows,
        # which is what stops the controller oscillating between accept and
        # reject.
        var scale: Float64
        if ratio <= 0.0:
            scale = 5.0
        else:
            scale = 0.9 * (1.0 / ratio) ** 0.2
            scale = min(5.0, max(0.2, scale))
        h = h * scale

    return IVPResult(t, y, accepted, rejected, False)
