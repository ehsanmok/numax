"""Adaptive integration: subdivide where the integrand misbehaves.

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
