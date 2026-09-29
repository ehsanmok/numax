"""Adaptive integration: quadrature that subdivides, and an ODE solver that
picks its own step size.

**This module is tier 2.** `quad` decides where to subdivide by looking at
the integrand's values, so the amount of work depends on the data and the
control flow branches on it. That is what `numax.integrate`'s module
docstring rules out for `FloatLike`-generic code, and correctly: two SIMD
lanes integrating different functions would want different grids, and there
is no per-lane way to give them one. So the adaptive rule lives here
instead, `Plain`-only and host-side. See `docs/architecture.md`'s "Two
tiers".

## Tier 1 sibling

`numax.integrate.gauss_legendre` is the fixed-node rule, and it is not
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
subdivision test, and it composes out of `numax.integrate` rather than
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

from ..core.dual import Dual
from ..core.numeric import FloatLike
from ..core.plain import Plain
from ._array.ode import dopri5_step
from .ode import dopri5_step as _tensor_dopri5_step
from .ode import _axpy_into, _dopri5_stages, _Stages
from .ode import _target as _ode_target
from ..core.tensorlike import TensorLike, dim
from ..core.tensor import _same_order, Dynamic, Static, _dyn_shape, copy
from layout.tile_layout import row_major
from max.gpu.host import DeviceContext
from algorithm.rowwise_types import RowCoord
from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from ..core.rowwise import reduce_all
from ._array.quadrature import (
    _gauss_legendre_nodes,
    _gauss_legendre_weights,
    gauss_legendre,
)

# Fixed to float64 because an error tolerance of 1e-10 is meaningless at
# float32 and these integrators are host drivers. Mojo 1.1 does accept a
# struct instantiated with a function-level `DType` as a `FloatLike`
# argument -- `numax.optimize`'s `Array` tier's drivers take `dtype` that way -- so
# a per-call dtype is a change of this file, not a language limit.
comptime _P = Plain[DType.float64, 1]


@fieldwise_init
struct QuadResult(Copyable):
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
        var d = x - U.constant(0.5)
        return U.one() / (U.constant(1e-4) + d * d)

    var result = quad[peaked](0.0, 1.0)
    ```

    Parameters:
        f: The `FloatLike`-generic integrand, evaluated here at `Plain`
            float64.
        n: The Gauss-Legendre order applied to each panel.

    Args:
        a: The lower limit of integration.
        b: The upper limit of integration; `b < a` negates the result.
        tol: The absolute error tolerance, shared across panels in
            proportion to their width.
        max_panels: The panel budget; once reached, remaining panels are
            accepted as they are and `converged` is false.

    Returns:
        A `QuadResult` with the integral, the summed panel error estimate,
        the accepted panel count and whether the tolerance was met.
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

    Parameters:
        f: The `FloatLike`-generic integrand, evaluated here at `Plain`
            float64.
        n: The Gauss-Legendre order applied to each panel.

    Args:
        a: The lower limit of integration.
        b: The upper limit of integration.
        breakpoints: Points forced to be panel boundaries; those outside
            `(a, b)` are dropped and the rest are sorted.
        tol: The absolute error tolerance, split evenly across the pieces
            between consecutive cuts.

    Returns:
        A `QuadResult` summing the pieces' values, errors and panel counts,
        with `converged` false if any piece did not converge.
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
struct ScalarIVPResult(Copyable):
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
) -> ScalarIVPResult:
    """Integrate `dy/dt = f(t, y)` from `t0` to `t1` with adaptive step
    control. The tier-2 counterpart of `numax.integrate.dopri5`.

    Dormand-Prince 5(4) with the classic proportional controller: each step
    is taken with `numax.integrate.dopri5_step`, which returns the 5th-order
    solution and its disagreement with the embedded 4th-order one; that
    error is compared against `atol + rtol*|y|`, and the step is accepted
    or rejected accordingly. The next step size is scaled by
    `(1/error_ratio)**(1/5)`, clamped to a factor of 5 in either direction
    so one anomalous step cannot make the controller wild.

    **Why this is tier 2 and `numax.integrate.dopri5` is not.** The step *body*
    is identical -- the same seven stages from the same tableau, shared
    rather than duplicated. What differs is that this decides, per step and
    based on the values, whether to keep the result and how far to go next.
    That is a data-dependent iteration count, so a SIMD `T` whose lanes
    disagreed about acceptance could not be served, and the whole thing
    runs on the host at `Plain` instead.

    The payoff is the usual one for adaptivity: on a solution with a sharp
    transient followed by a smooth tail, a fixed-step integrator has to use
    the transient's step size everywhere. `tests/integrate/test_integrate.mojo`
    measures that against `dopri5` on such a problem.

    `f` is still an ordinary `FloatLike` kernel, so the same equation can be
    integrated by the tier-1 `rk4` or `dopri5` inside a GPU kernel -- see
    `examples/advanced/ode.mojo`, which runs an ensemble that way.

    Parameters:
        f: The right-hand side `f(t, y)`, a `FloatLike`-generic kernel
            evaluated here at `Plain` float64.

    Args:
        t0: The initial time.
        y0: The state at `t0`.
        t1: The final time; it may be less than `t0` to integrate backward.
        rtol: The relative error tolerance per step.
        atol: The absolute error tolerance per step.
        max_steps: The cap on attempted steps, accepted plus rejected.

    Returns:
        A `ScalarIVPResult` with the time reached, the state there, the
        accepted and rejected step counts and whether `t1` was reached.
    """
    if t0 == t1:
        return ScalarIVPResult(t0, y0, 0, 0, True)

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
            return ScalarIVPResult(t, y, accepted, rejected, True)

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
                return ScalarIVPResult(t, y, accepted, rejected, True)
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

    return ScalarIVPResult(t, y, accepted, rejected, False)


def _max_abs_ratio[
    dtype: DType, n: Int, gpu: Bool = False
](
    mut y: Static[dtype, n],
    mut y_next: Static[dtype, n],
    mut y_hat: Static[dtype, n],
    rtol: Float64,
    atol: Float64,
) raises -> Float64 where dtype.is_floating_point():
    """`max_i |y_next_i - y_hat_i| / (atol + rtol * max(|y_i|, |y_next_i|))`
    -- the infinity-norm error ratio the controller accepts a step on.

    At `n == 1` this is exactly the scalar `solve_ivp`'s ratio, which is
    what lets the test pin the two against each other step for step.

    At `gpu=True` on device-resident states it is one launch writing each
    component's ratio and a device `max`, so one scalar crosses back per
    step rather than three state vectors.
    """
    comptime if gpu:
        if not y.on_host():
            var ctx = y.context()
            var ratios = Static[dtype, n]._uninitialized(ctx)
            var ap = y.tile()
            var bp = y_next.tile()
            var cp = y_hat.tile()
            var rp = ratios.tile()
            var lo = Scalar[dtype](atol)
            var rel = Scalar[dtype](rtol)

            @always_inline
            def ratio[
                width: Int, alignment: Int = 1
            ](coord: Coord) {var ap, var bp, var cp, var rp, var lo, var rel}:
                var a = ap[coord][0]
                var b = bp[coord][0]
                var scale = lo + rel * max(abs(a), abs(b))
                var err = abs(b - cp[coord][0])
                rp.store[1](
                    coord, err / scale if scale > 0 else Scalar[dtype](0)
                )

            elementwise[simd_width=1, target="gpu"](ratio, Coord(n), ctx)
            var worst = Static[dtype, 1](ctx)

            @always_inline
            def identity[
                w: Int
            ](tile: SIMD[dtype, w], idx: RowCoord[1]) {} -> SIMD[dtype, w]:
                return tile

            reduce_all[monoid="max", gpu=True](
                ratios.tile(), worst.tile(), identity, n, Optional(ctx)
            )
            return Float64(worst.to_host()[0])
    var a = y.to_host()
    var b = y_next.to_host()
    var c = y_hat.to_host()
    var worst = 0.0
    for i in range(n):
        var scale = atol + rtol * max(abs(Float64(a[i])), abs(Float64(b[i])))
        var err = abs(Float64(b[i] - c[i]))
        var ratio = err / scale if scale > 0.0 else 0.0
        if ratio > worst:
            worst = ratio
    return worst


struct IVPResult[dtype: DType, n: Int](Movable where dtype.is_floating_point()):
    """The outcome of an adaptive integration over a `Tensor` state: the
    `Tensor` form of `ScalarIVPResult`, with the same fields and the same meaning
    for `accepted`, `rejected` and `converged`."""

    var t: Float64
    var y: Static[Self.dtype, Self.n]
    var accepted: Int
    var rejected: Int
    var converged: Bool

    def __init__(
        out self,
        t: Float64,
        var y: Static[Self.dtype, Self.n],
        accepted: Int,
        rejected: Int,
        converged: Bool,
    ):
        self.t = t
        self.y = y^
        self.accepted = accepted
        self.rejected = rejected
        self.converged = converged


def solve_ivp[
    T: TensorLike,
    f: def(
        Scalar[T.dtype], Static[T.dtype, dim[T, 0]], DeviceContext
    ) raises thin -> Static[T.dtype, dim[T, 0]],
    gpu: Bool = False,
](
    t0: Float64,
    y0: T,
    t1: Float64,
    rtol: Float64 = 1e-8,
    atol: Float64 = 1e-10,
    max_steps: Int = 10000,
) raises -> IVPResult[T.dtype, dim[T, 0]] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
):
    """Integrate the system from `t0` to `t1` with adaptive step control.
    The `Tensor` form of `solve_ivp`, and the same controller: Dormand-Prince
    5(4) steps, accepted when the error ratio is at most one, the next step
    scaled by `0.9 * ratio^(-1/5)` clamped to a factor of five each way.

    The error ratio is the infinity norm over components of
    `|y5 - y4| / (atol + rtol * max(|y|, |y5|))`, which at `n == 1` is the
    scalar controller's ratio exactly -- so on a one-component problem the
    two take the same steps and return the same counts, and a test says so.
    The step decision is host control flow over one scalar per step, which
    is what makes this tier 2; at `gpu=True` the ratio itself is computed on
    the device and the state never leaves it.

    Parameters:
        T: The rank-1, static-extent floating-point tensor type of `y0`.
        f: The right-hand side `f(t, y, ctx)`, returning `dy/dt` as a
            `Static` tensor of `y`'s length.
        gpu: Whether the stages and the error ratio run on the state's
            device; a host-resident state falls back to the host path.

    Args:
        t0: The initial time.
        y0: The state at `t0`, of length `dim[T, 0]`.
        t1: The final time; it may be less than `t0` to integrate backward.
        rtol: The relative error tolerance per component and step.
        atol: The absolute error tolerance per component and step.
        max_steps: The cap on attempted steps, accepted plus rejected.

    Returns:
        An `IVPResult` with the time reached, the state there as a
        `Static` tensor, the step counts and whether `t1` was reached.

    Raises:
        If `f` raises, or if a tensor copy, launch or transfer fails.
    """
    comptime dtype = T.dtype
    comptime n = dim[T, 0]
    if t0 == t1:
        return IVPResult[dtype, n](
            t0, _same_order(y0, Static[dtype, n]._static_layout()), 0, 0, True
        )

    var direction = 1.0 if t1 > t0 else -1.0
    var span = abs(t1 - t0)
    var t = t0
    var y = _same_order(y0, Static[dtype, n]._static_layout())
    var h = direction * span / 100.0
    var accepted = 0
    var rejected = 0

    for _ in range(max_steps):
        if abs(t - t1) <= 0.0:
            return IVPResult[dtype, n](t, y^, accepted, rejected, True)
        if abs(h) > abs(t1 - t):
            h = t1 - t

        var stepped = _tensor_dopri5_step[f=f, gpu=gpu](t, y, h)
        var ratio = _max_abs_ratio[gpu=gpu](
            y, stepped.y, stepped.y_hat, rtol, atol
        )

        if ratio <= 1.0:
            t += h
            y = copy(stepped.y)
            accepted += 1
            if abs(t - t1) <= 0.0:
                return IVPResult[dtype, n](t, y^, accepted, rejected, True)
        else:
            rejected += 1

        var scale: Float64
        if ratio <= 0.0:
            scale = 5.0
        else:
            scale = 0.9 * (1.0 / ratio) ** 0.2
            scale = min(5.0, max(0.2, scale))
        h = h * scale

    return IVPResult[dtype, n](t, y^, accepted, rejected, False)


def _trapezoid_step[
    f: def[U: FloatLike](U, U) thin -> U,
](t: Float64, y: Float64, h: Float64) -> Float64:
    """One implicit trapezoid step, `y1 = y + h/2*(f(t,y) + f(t+h,y1))`,
    solved for `y1` by Newton's method.

    The derivative `df/dy` that Newton needs comes from evaluating `f` at
    `Dual` with the state seeded -- the same trick the root finders use,
    which is why an implicit method costs no extra function from the
    caller. Explicit Euler supplies the initial guess; near-quadratic
    convergence takes it from there in a handful of iterations.
    """
    var f0 = f[_P](_P.constant(t), _P.constant(y)).v
    var guess = y + h * Float64(f0)

    for _ in range(20):
        var evaluated = f[Dual[_P]](
            Dual[_P].constant(t + h), Dual[_P](_P(guess), _P.one())
        )
        var residual = guess - y - 0.5 * h * (Float64(f0) + evaluated.value.v)
        var slope = 1.0 - 0.5 * h * evaluated.deriv.v
        if abs(slope) < 1e-300:
            break
        var step = residual / slope
        guess -= step
        if abs(step) <= 1e-14 * (1.0 + abs(guess)):
            break

    return guess


def solve_ivp_stiff[
    f: def[U: FloatLike](U, U) thin -> U,
](
    t0: Float64,
    y0: Float64,
    t1: Float64,
    rtol: Float64 = 1e-8,
    atol: Float64 = 1e-10,
    max_steps: Int = 10000,
) -> ScalarIVPResult:
    """Integrate a *stiff* `dy/dt = f(t, y)` from `t0` to `t1`. The
    A-stable counterpart of `solve_ivp`.

    A stiff problem is one where the step size an explicit method can take
    is set by stability rather than by accuracy: a decay a thousand times
    faster than anything the solution actually does still forces
    Dormand-Prince to step around a thousand times more finely, long after
    that component has vanished. The implicit trapezoid has no such limit
    -- it is A-stable, so the step size is chosen for accuracy alone --
    which is the entire reason to pay for the nonlinear solve each step
    costs. `tests/integrate/test_integrate.mojo` measures the step counts
    against `solve_ivp` on such a problem rather than asserting the
    difference.

    Second order, with the error estimated by step doubling: one step of
    `h` against two of `h/2`, whose difference over 3 is the Richardson
    estimate. That is three implicit solves per attempted step, which is
    the honest price of not carrying an embedded pair; a production BDF or
    Radau code would do better, and this is the shape that fits in the
    module rather than the fastest one available.

    The Newton iteration inside each step takes `df/dy` from `Dual`, so
    there is no `jac` argument here either -- the caller writes one
    ordinary `FloatLike` kernel and it serves the explicit solver, the
    implicit one, and a GPU launch.

    Only scalar equations, matching `solve_ivp`. A stiff *system* needs the
    full Jacobian and a linear solve per Newton iteration; `Gradient` and
    `numax.linalg.solve` are both here, so it is a continuation rather than
    a redesign.
    """
    if t0 == t1:
        return ScalarIVPResult(t0, y0, 0, 0, True)

    var direction = 1.0 if t1 > t0 else -1.0
    var span = abs(t1 - t0)

    var t = t0
    var y = y0
    var h = direction * span / 100.0

    var accepted = 0
    var rejected = 0

    for _ in range(max_steps):
        if abs(t - t1) <= 0.0:
            return ScalarIVPResult(t, y, accepted, rejected, True)

        if abs(h) > abs(t1 - t):
            h = t1 - t

        var coarse = _trapezoid_step[f](t, y, h)
        var midpoint = _trapezoid_step[f](t, y, h / 2.0)
        var fine = _trapezoid_step[f](t + h / 2.0, midpoint, h / 2.0)

        # Richardson at order 2: the two-half-step result is four times as
        # accurate, so their difference over 3 estimates what remains.
        var error = abs(fine - coarse) / 3.0
        var tolerance = atol + rtol * max(abs(y), abs(fine))
        var ratio = error / tolerance if tolerance > 0.0 else 0.0

        if ratio <= 1.0:
            t += h
            y = fine
            accepted += 1
            if abs(t - t1) <= 0.0:
                return ScalarIVPResult(t, y, accepted, rejected, True)
        else:
            rejected += 1

        var scale: Float64
        if ratio <= 0.0:
            scale = 5.0
        else:
            scale = 0.9 * (1.0 / ratio) ** (1.0 / 3.0)
            scale = min(5.0, max(0.2, scale))
        h = h * scale

    return ScalarIVPResult(t, y, accepted, rejected, False)


def fixed_quad[
    f: def[U: FloatLike](U) thin -> U, n: Int = 8
](a: Float64, b: Float64) -> Float64:
    """Integrate `f` over `[a, b]` with a fixed `n`-point Gauss-Legendre
    rule, no subdivision. `scipy.integrate.fixed_quad(f, a, b, n=n)`,
    first return value.

    `quad` above is the adaptive routine and is what to reach for when the
    integrand's behavior is unknown; this is the one to reach for when it
    is known to be smooth, because then `n = 8` is already exact to
    rounding for a degree-15 polynomial and the adaptivity is pure
    overhead. Exactly `n` evaluations, no error estimate, no convergence
    flag -- which is why it returns a bare `Float64` where `quad` returns
    a `QuadResult`.

    `numax.integrate.gauss_legendre` is the same rule at any
    `FloatLike`, so it differentiates and runs in a kernel; this is its
    `Float64` front door under SciPy's name.

    Parameters:
        f: The `FloatLike`-generic integrand, evaluated here at `Plain`
            float64.
        n: The number of Gauss-Legendre nodes.

    Args:
        a: The lower limit of integration.
        b: The upper limit of integration.

    Returns:
        The `n`-point Gauss-Legendre estimate of the integral.
    """
    return gauss_legendre[_P, f, n](_P(a), _P(b)).v[0]


def dblquad[
    f: def[U: FloatLike](U, U) thin -> U, n: Int = 8
](ax: Float64, bx: Float64, ay: Float64, by: Float64) -> Float64:
    """Integrate `f(x, y)` over the rectangle `[ax, bx] x [ay, by]` with a
    tensor-product Gauss-Legendre rule. Close to
    `scipy.integrate.dblquad(f, ax, bx, ay, by)`.

    `n * n` evaluations on the product of two `n`-point rules, exact to
    rounding for any polynomial of degree `2n - 1` in each variable
    separately.

    **Two deliberate differences from SciPy**, both worth knowing before
    reaching for this. The inner bounds are **constants, not functions of
    `x`**, so the region is a rectangle rather than SciPy's `gfun`/`hfun`
    region. And the rule is **fixed-order, not adaptive**.

    Both come from one constraint rather than from taste: `quad`'s
    integrand is a compile-time non-capturing function parameter, so the
    inner integral -- which is a function of the outer variable and must
    therefore capture it -- cannot be handed to `quad` at all. A nested
    adaptive `dblquad` needs a capturing integrand, and `f(x, y)` taking
    both variables at once is what sidesteps it. The argument order is
    `f(x, y)`, not SciPy's reversed `f(y, x)`; swapping it silently would
    be worse than saying so.

    Parameters:
        f: The integrand `f(x, y)`, a `FloatLike`-generic kernel evaluated
            here at `Plain` float64.
        n: The number of Gauss-Legendre nodes along each axis.

    Args:
        ax: The lower limit in `x`.
        bx: The upper limit in `x`.
        ay: The lower limit in `y`.
        by: The upper limit in `y`.

    Returns:
        The `n * n`-point tensor-product estimate of the integral over the
        rectangle.
    """
    comptime nodes = _gauss_legendre_nodes[n]()
    comptime weights = _gauss_legendre_weights[n]()

    var mid_x = (ax + bx) / 2.0
    var span_x = (bx - ax) / 2.0
    var mid_y = (ay + by) / 2.0
    var span_y = (by - ay) / 2.0

    var total = 0.0
    comptime for i in range(n):
        comptime node_i = nodes[i]
        comptime weight_i = weights[i]
        var xi = mid_x + span_x * node_i
        var inner = 0.0
        comptime for j in range(n):
            comptime node_j = nodes[j]
            comptime weight_j = weights[j]
            var yj = mid_y + span_y * node_j
            inner += weight_j * f(_P(xi), _P(yj)).v[0]
        total += weight_i * inner
    return total * span_x * span_y


# ------------------------------------------------ dense output and events

comptime _DENSE_P: Array[Float64, 28] = [
    1.0,
    -8048581381.0 / 2820520608.0,
    8663915743.0 / 2820520608.0,
    -12715105075.0 / 11282082432.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    131558114200.0 / 32700410799.0,
    -68118460800.0 / 10900136933.0,
    87487479700.0 / 32700410799.0,
    0.0,
    -1754552775.0 / 470086768.0,
    14199869525.0 / 1410260304.0,
    -10690763975.0 / 1880347072.0,
    0.0,
    127303824393.0 / 49829197408.0,
    -318862633887.0 / 49829197408.0,
    701980252875.0 / 199316789632.0,
    0.0,
    -282668133.0 / 205662961.0,
    2019193451.0 / 616988883.0,
    -1453857185.0 / 822651844.0,
    0.0,
    40617522.0 / 29380423.0,
    -110615467.0 / 29380423.0,
    69997945.0 / 29380423.0,
]
"""SciPy `RK45.P`: Shampine's fourth-order continuous extension of the
Dormand-Prince pair, `7 x 4` row-major. Stage `i`'s weight at the
fraction `x` of a step is `sum_j P[i][j] x^(j+1)`."""


def _dense_weights(x: Float64) -> SIMD[DType.float64, 8]:
    """The seven stage weights of the continuous extension at `x`."""
    var out = SIMD[DType.float64, 8](0)
    comptime for i in range(7):
        var acc = 0.0
        var power = x
        comptime for j in range(4):
            comptime c = _DENSE_P[i * 4 + j]
            acc += c * power
            power *= x
        out[i] = acc
    return out


def _interpolate[
    dtype: DType, n: Int, gpu: Bool
](
    mut y_old: Static[dtype, n],
    mut stages: _Stages[dtype, n],
    t_old: Float64,
    h: Float64,
    t: Float64,
    ctx: DeviceContext,
) raises -> Static[dtype, n] where dtype.is_floating_point():
    """The step's continuous extension at `t`: `y_old + h sum_i b_i(x) k_i`,
    seven axpys on the state's device."""
    var b = _dense_weights((t - t_old) / h)
    var y = copy(y_old)
    _axpy_into[gpu=gpu](y, stages.k1, Scalar[dtype](h * b[0]), ctx)
    _axpy_into[gpu=gpu](y, stages.k3, Scalar[dtype](h * b[2]), ctx)
    _axpy_into[gpu=gpu](y, stages.k4, Scalar[dtype](h * b[3]), ctx)
    _axpy_into[gpu=gpu](y, stages.k5, Scalar[dtype](h * b[4]), ctx)
    _axpy_into[gpu=gpu](y, stages.k6, Scalar[dtype](h * b[5]), ctx)
    _axpy_into[gpu=gpu](y, stages.k7, Scalar[dtype](h * b[6]), ctx)
    return y^


struct _RowBuffer[dtype: DType, n: Int, gpu: Bool](Movable):
    """A growable `rows x n` matrix on the state's device: states appended
    one row at a time, capacity doubled by a device copy when full."""

    var data: Dynamic[Self.dtype, 2]
    var rows: Int
    var capacity: Int

    def __init__(out self, ctx: DeviceContext) raises:
        self.capacity = 16
        self.rows = 0
        self.data = Dynamic[Self.dtype, 2](
            row_major(_dyn_shape[2](self.capacity, Self.n)), ctx
        )

    def push(mut self, mut v: Static[Self.dtype, Self.n]) raises:
        """Append `v` as the next row."""
        var ctx = v.context()
        if self.rows == self.capacity:
            var grown = Dynamic[Self.dtype, 2](
                row_major(_dyn_shape[2](2 * self.capacity, Self.n)), ctx
            )
            var src = self.data.tile()
            var dst = grown.tile()
            var used = self.rows * Self.n

            @always_inline
            def move[
                w: Int, alignment: Int = 1
            ](coord: Coord) {var src, var dst}:
                var e = coord_to_index_list(coord)[0]
                dst.ptr[unsafe_offset=e] = src.ptr[unsafe_offset=e]

            elementwise[simd_width=1, target=_ode_target[Self.gpu]()](
                move, Coord(used), ctx
            )
            ctx.synchronize()
            self.data = grown^
            self.capacity *= 2
        var dst = self.data.tile()
        var vs = v.tile()
        var base = self.rows * Self.n

        @always_inline
        def write[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var dst, var vs, var base}:
            var e = coord_to_index_list(coord)[0]
            dst.ptr[unsafe_offset=base + e] = vs.ptr[unsafe_offset=e]

        elementwise[simd_width=1, target=_ode_target[Self.gpu]()](
            write, Coord(Self.n), ctx
        )
        ctx.synchronize()
        self.rows += 1

    def columns(mut self, ctx: DeviceContext) raises -> Dynamic[Self.dtype, 2]:
        """The rows as columns: an `n x rows` matrix, SciPy's `sol.y`
        layout."""
        var count = self.rows
        var out = Dynamic[Self.dtype, 2](
            row_major(_dyn_shape[2](Self.n, max(count, 1))), ctx
        )
        if count == 0:
            return Dynamic[Self.dtype, 2](
                row_major(_dyn_shape[2](Self.n, 0)), ctx
            )
        var src = self.data.tile()
        var dst = out.tile()

        @always_inline
        def flip[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var src, var dst, var count}:
            var e = coord_to_index_list(coord)[0]
            var i = e // count
            var j = e % count
            dst.ptr[unsafe_offset=e] = src.ptr[unsafe_offset=j * Self.n + i]

        elementwise[simd_width=1, target=_ode_target[Self.gpu]()](
            flip, Coord(Self.n * count), ctx
        )
        ctx.synchronize()
        return out^


struct DenseOutput[dtype: DType, n: Int, gpu: Bool = False](Movable):
    """The continuous solution `solve_ivp(..., dense_output=True)` returns,
    SciPy's `OdeSolution`: `sol(t)` evaluates the fourth-order continuous
    extension of the step that contains `t`.

    Each accepted step keeps its start, its size, its initial state and
    its seven stages -- eight rows of a device matrix -- so evaluating is
    a host binary search over the step starts and one weighted sum on the
    state's device. Empty unless `dense_output` was asked for.
    """

    var _starts: List[Float64]
    var _steps: List[Float64]
    var _rows: _RowBuffer[Self.dtype, Self.n, Self.gpu]

    def __init__(out self, ctx: DeviceContext) raises:
        self._starts = List[Float64]()
        self._steps = List[Float64]()
        self._rows = _RowBuffer[Self.dtype, Self.n, Self.gpu](ctx)

    def _record(
        mut self,
        t_old: Float64,
        h: Float64,
        mut y_old: Static[Self.dtype, Self.n],
        mut stages: _Stages[Self.dtype, Self.n],
    ) raises:
        self._starts.append(t_old)
        self._steps.append(h)
        self._rows.push(y_old)
        self._rows.push(stages.k1)
        self._rows.push(stages.k2)
        self._rows.push(stages.k3)
        self._rows.push(stages.k4)
        self._rows.push(stages.k5)
        self._rows.push(stages.k6)
        self._rows.push(stages.k7)

    def __call__(mut self, t: Float64) raises -> Static[Self.dtype, Self.n]:
        """The solution at `t`, which must lie in the integrated span.

        Args:
            t: The time to evaluate at.

        Returns:
            The interpolated state, on the state's device.

        Raises:
            If no step was recorded, or `t` is outside the span.
        """
        var count = len(self._starts)
        if count == 0:
            raise Error("DenseOutput: empty; pass dense_output=True")
        var forward = self._steps[0] > 0
        var first = self._starts[0]
        var last = self._starts[count - 1] + self._steps[count - 1]
        var lo_t = first if forward else last
        var hi_t = last if forward else first
        if t < lo_t or t > hi_t:
            raise Error(
                "DenseOutput: t = ", t, " is outside [", lo_t, ", ", hi_t, "]"
            )
        # The last step whose start is not past `t` in the direction of
        # integration.
        var lo = 0
        var hi = count - 1
        while lo < hi:
            var mid = (lo + hi + 1) // 2
            var past = (self._starts[mid] > t) if forward else (
                self._starts[mid] < t
            )
            if past:
                hi = mid - 1
            else:
                lo = mid
        var s = lo
        var h = self._steps[s]
        var b = _dense_weights((t - self._starts[s]) / h)
        var ctx = self._rows.data.context()
        var out = Static[Self.dtype, Self.n](ctx)
        var src = self._rows.data.tile()
        var dst = out.tile()
        var base = 8 * s * Self.n
        var w1 = Scalar[Self.dtype](h * b[0])
        var w3 = Scalar[Self.dtype](h * b[2])
        var w4 = Scalar[Self.dtype](h * b[3])
        var w5 = Scalar[Self.dtype](h * b[4])
        var w6 = Scalar[Self.dtype](h * b[5])
        var w7 = Scalar[Self.dtype](h * b[6])

        @always_inline
        def combine[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var src,
            var dst,
            var base,
            var w1,
            var w3,
            var w4,
            var w5,
            var w6,
            var w7,
        }:
            var i = coord_to_index_list(coord)[0]
            var r = base + i
            var v = (
                src.ptr[unsafe_offset=r]
                + w1 * src.ptr[unsafe_offset=r + Self.n]
                + w3 * src.ptr[unsafe_offset=r + 3 * Self.n]
                + w4 * src.ptr[unsafe_offset=r + 4 * Self.n]
                + w5 * src.ptr[unsafe_offset=r + 5 * Self.n]
                + w6 * src.ptr[unsafe_offset=r + 6 * Self.n]
                + w7 * src.ptr[unsafe_offset=r + 7 * Self.n]
            )
            dst.ptr[unsafe_offset=i] = v

        elementwise[simd_width=1, target=_ode_target[Self.gpu]()](
            combine, Coord(Self.n), ctx
        )
        ctx.synchronize()
        return out^


struct IVPSolution[dtype: DType, n: Int, gpu: Bool = False](Movable):
    """What the recording `solve_ivp` returns, SciPy's `OdeResult`.

    `y` is `n x m`, SciPy's layout, one column per time in `t`; the event
    fields hold the located crossings, `y_events` a column per crossing.
    `status` is SciPy's: `0` reached the end, `1` stopped at a terminal
    event, `-1` ran out of steps.
    """

    var t: List[Float64]
    """The output times: the `t_eval` points the integration reached."""
    var y: Dynamic[Self.dtype, 2]
    """The states at `t`, `n x len(t)`, on the state's device."""
    var t_events: List[Float64]
    """The times the event function crossed zero, in order."""
    var y_events: Dynamic[Self.dtype, 2]
    """The states at `t_events`, `n x len(t_events)`."""
    var status: Int
    """`0`: reached `t1`; `1`: a terminal event; `-1`: out of steps."""
    var accepted: Int
    """Accepted steps."""
    var rejected: Int
    """Rejected steps."""
    var sol: DenseOutput[Self.dtype, Self.n, Self.gpu]
    """The continuous solution, when `dense_output` was asked for."""

    def __init__(
        out self,
        var t: List[Float64],
        var y: Dynamic[Self.dtype, 2],
        var t_events: List[Float64],
        var y_events: Dynamic[Self.dtype, 2],
        status: Int,
        accepted: Int,
        rejected: Int,
        var sol: DenseOutput[Self.dtype, Self.n, Self.gpu],
    ):
        """Build from the parts.

        Args:
            t: The output times.
            y: The states at them.
            t_events: The event times.
            y_events: The states at them.
            status: SciPy's status code.
            accepted: Accepted steps.
            rejected: Rejected steps.
            sol: The continuous solution.
        """
        self.t = t^
        self.y = y^
        self.t_events = t_events^
        self.y_events = y_events^
        self.status = status
        self.accepted = accepted
        self.rejected = rejected
        self.sol = sol^


def _no_event[
    dtype: DType, n: Int
](t: Scalar[dtype], y: Static[dtype, n], ctx: DeviceContext) raises -> Scalar[
    dtype
]:
    return Scalar[dtype](1)


def _crossed(g: Float64, g_new: Float64, direction: Int) -> Bool:
    """SciPy's `find_active_events` for one event."""
    var up = g <= 0.0 and g_new >= 0.0
    var down = g >= 0.0 and g_new <= 0.0
    if direction > 0:
        return up
    if direction < 0:
        return down
    return up or down


def _solve_ivp_recording[
    dtype: DType,
    n: Int,
    f: def(
        Scalar[dtype], Static[dtype, n], DeviceContext
    ) raises thin -> Static[dtype, n],
    has_event: Bool,
    event: def(
        Scalar[dtype], Static[dtype, n], DeviceContext
    ) raises thin -> Scalar[dtype],
    terminal: Bool,
    direction: Int,
    dense_output: Bool,
    gpu: Bool,
](
    t0: Float64,
    var y: Static[dtype, n],
    t1: Float64,
    t_eval: List[Float64],
    rtol: Float64,
    atol: Float64,
    max_steps: Int,
) raises -> IVPSolution[dtype, n, gpu] where dtype.is_floating_point():
    var ctx = y.context()
    var forward = t1 >= t0
    var span = abs(t1 - t0)
    var t = t0
    var h = (1.0 if forward else -1.0) * span / 100.0
    var accepted = 0
    var rejected = 0
    var status = -1
    var ts = List[Float64]()
    var ys = _RowBuffer[dtype, n, gpu](ctx)
    var tev = List[Float64]()
    var yev = _RowBuffer[dtype, n, gpu](ctx)
    var sol = DenseOutput[dtype, n, gpu](ctx)
    var next_eval = 0
    var g_old = 0.0
    comptime if has_event:
        g_old = Float64(event(Scalar[dtype](t), y, ctx))

    # A `t_eval` point at `t0` is reported before the first step.
    while next_eval < len(t_eval) and t_eval[next_eval] == t0:
        ts.append(t0)
        ys.push(y)
        next_eval += 1

    if t0 == t1:
        status = 0
    else:
        for _ in range(max_steps):
            if abs(h) > abs(t1 - t):
                h = t1 - t
            var stages = _dopri5_stages[f=f, gpu=gpu](t, y, h)
            var ratio = _max_abs_ratio[gpu=gpu](
                y, stages.y5, stages.y4, rtol, atol
            )
            if ratio <= 1.0:
                accepted += 1
                var t_old = t
                var t_new = t + h
                var stop = t_new
                var terminate = False
                comptime if has_event:
                    var g_new = Float64(
                        event(Scalar[dtype](t_new), stages.y5, ctx)
                    )
                    if _crossed(g_old, g_new, direction):
                        # Brent on the step's continuous extension, SciPy's
                        # `solve_event_equation` with `xtol = rtol = 4 eps`.
                        var a = t_old
                        var b = t_new
                        var fa = g_old
                        var fb = g_new
                        var c = a
                        var fc = fa
                        var d = b - a
                        var e = d
                        comptime eps = 2.220446049250313e-16
                        for _ in range(200):
                            if (fb > 0.0 and fc > 0.0) or (
                                fb < 0.0 and fc < 0.0
                            ):
                                c = a
                                fc = fa
                                d = b - a
                                e = d
                            if abs(fc) < abs(fb):
                                a = b
                                b = c
                                c = a
                                fa = fb
                                fb = fc
                                fc = fa
                            var tol = 2.0 * 4.0 * eps * abs(b) + 0.5 * 4.0 * eps
                            var m = 0.5 * (c - b)
                            if abs(m) <= tol or fb == 0.0:
                                break
                            if abs(e) >= tol and abs(fa) > abs(fb):
                                var s = fb / fa
                                var p: Float64
                                var q: Float64
                                if a == c:
                                    p = 2.0 * m * s
                                    q = 1.0 - s
                                else:
                                    var qa = fa / fc
                                    var r = fb / fc
                                    p = s * (
                                        2.0 * m * qa * (qa - r)
                                        - (b - a) * (r - 1.0)
                                    )
                                    q = (qa - 1.0) * (r - 1.0) * (s - 1.0)
                                if p > 0.0:
                                    q = -q
                                else:
                                    p = -p
                                if 2.0 * p < min(
                                    3.0 * m * q - abs(tol * q), abs(e * q)
                                ):
                                    e = d
                                    d = p / q
                                else:
                                    d = m
                                    e = d
                            else:
                                d = m
                                e = d
                            a = b
                            fa = fb
                            if abs(d) > tol:
                                b += d
                            else:
                                b += tol if m > 0.0 else -tol
                            var yb = _interpolate[gpu=gpu](
                                y, stages, t_old, h, b, ctx
                            )
                            fb = Float64(event(Scalar[dtype](b), yb, ctx))
                        var root = b
                        tev.append(root)
                        var y_root = _interpolate[gpu=gpu](
                            y, stages, t_old, h, root, ctx
                        )
                        yev.push(y_root)
                        comptime if terminal:
                            terminate = True
                            stop = root
                    g_old = g_new
                # The `t_eval` points this step covers, up to where it
                # stops.
                while next_eval < len(t_eval):
                    var te = t_eval[next_eval]
                    var covered = (te <= stop) if forward else (te >= stop)
                    if not covered:
                        break
                    ts.append(te)
                    var ye = _interpolate[gpu=gpu](y, stages, t_old, h, te, ctx)
                    ys.push(ye)
                    next_eval += 1
                comptime if dense_output:
                    sol._record(t_old, h, y, stages)
                if terminate:
                    t = stop
                    y = _interpolate[gpu=gpu](y, stages, t_old, h, stop, ctx)
                    status = 1
                    break
                t = t_new
                y = copy(stages.y5)
                if abs(t - t1) <= 0.0:
                    status = 0
                    break
            else:
                rejected += 1
            var scale: Float64
            if ratio <= 0.0:
                scale = 5.0
            else:
                scale = 0.9 * (1.0 / ratio) ** 0.2
                scale = min(5.0, max(0.2, scale))
            h = h * scale
    var y_out = ys.columns(ctx)
    var yev_out = yev.columns(ctx)
    return IVPSolution[dtype, n, gpu](
        ts^, y_out^, tev^, yev_out^, status, accepted, rejected, sol^
    )


def _eval_times[E: TensorLike](t_eval: E) raises -> List[Float64]:
    var values = t_eval.to_host()
    var out = List[Float64](capacity=len(values))
    for i in range(len(values)):
        out.append(Float64(values[i]))
    return out^


def solve_ivp[
    T: TensorLike,
    E: TensorLike,
    f: def(
        Scalar[T.dtype], Static[T.dtype, dim[T, 0]], DeviceContext
    ) raises thin -> Static[T.dtype, dim[T, 0]],
    dense_output: Bool = False,
    gpu: Bool = False,
](
    t0: Float64,
    y0: T,
    t1: Float64,
    t_eval: E,
    rtol: Float64 = 1e-8,
    atol: Float64 = 1e-10,
    max_steps: Int = 10000,
) raises -> IVPSolution[T.dtype, dim[T, 0], gpu] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
    and E.LayoutType.rank == 1
):
    """Integrate the system and report the solution at the times `t_eval`.
    `scipy.integrate.solve_ivp(fun, (t0, t1), y0, t_eval=t_eval,
    dense_output=dense_output)`.

    The same controller and steps as the final-state `solve_ivp`; each
    accepted step's `t_eval` points are evaluated on its continuous
    extension, SciPy RK45's fourth-order interpolant (`_DENSE_P`), seven
    axpys on the state's device -- so the requested times cost no extra
    steps and do not change the ones taken. `t_eval` must be sorted in the
    direction of integration and inside `[t0, t1]`, as SciPy requires.
    With `dense_output`, `sol` evaluates the same interpolant at any time
    in the span afterwards.

    Parameters:
        T: The rank-1, static-extent floating-point tensor type of `y0`.
        E: The rank-1 tensor type of `t_eval`.
        f: The right-hand side `f(t, y, ctx)`.
        dense_output: Whether to keep every step for `sol`.
        gpu: Whether the stages, the error ratio and the interpolation run
            on the state's device.

    Args:
        t0: The initial time.
        y0: The state at `t0`.
        t1: The final time.
        t_eval: The output times.
        rtol: The relative error tolerance per component and step.
        atol: The absolute error tolerance per component and step.
        max_steps: The cap on attempted steps.

    Returns:
        An `IVPSolution` with `t`, `y` (`n x len(t)`), the step counts,
        `status`, and `sol` when asked.

    Raises:
        If `f` raises, or a tensor copy, launch or transfer fails.
    """
    comptime dtype = T.dtype
    comptime n = dim[T, 0]
    return _solve_ivp_recording[
        dtype,
        n,
        f,
        False,
        _no_event[dtype, n],
        False,
        0,
        dense_output,
        gpu,
    ](
        t0,
        _same_order(y0, Static[dtype, n]._static_layout()),
        t1,
        _eval_times(t_eval),
        rtol,
        atol,
        max_steps,
    )


def solve_ivp[
    T: TensorLike,
    E: TensorLike,
    f: def(
        Scalar[T.dtype], Static[T.dtype, dim[T, 0]], DeviceContext
    ) raises thin -> Static[T.dtype, dim[T, 0]],
    event: def(
        Scalar[T.dtype], Static[T.dtype, dim[T, 0]], DeviceContext
    ) raises thin -> Scalar[T.dtype],
    terminal: Bool = False,
    direction: Int = 0,
    dense_output: Bool = False,
    gpu: Bool = False,
](
    t0: Float64,
    y0: T,
    t1: Float64,
    t_eval: E,
    rtol: Float64 = 1e-8,
    atol: Float64 = 1e-10,
    max_steps: Int = 10000,
) raises -> IVPSolution[T.dtype, dim[T, 0], gpu] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
    and E.LayoutType.rank == 1
):
    """Integrate the system with an event function, reporting its zero
    crossings and optionally stopping at the first. `scipy.integrate.solve_ivp`
    with `events=event`, `event.terminal`, `event.direction`.

    After each accepted step the event is evaluated at its end; a sign
    change is a crossing under SciPy's rule (`direction > 0` counts only
    rising, `< 0` only falling, `0` either), and it is located by Brent's
    method on the step's continuous extension to `4 eps`, as SciPy's
    `brentq` locates it. Each crossing's time and state are recorded; with
    `terminal` the integration stops there, `status = 1`, and the
    `t_eval` points past it are not reported. One event function; SciPy's
    list of several is the upgrade, as is `max_events`.

    Parameters:
        T: The rank-1, static-extent floating-point tensor type of `y0`.
        E: The rank-1 tensor type of `t_eval`.
        f: The right-hand side `f(t, y, ctx)`.
        event: The event function `g(t, y, ctx)`, whose zeros are sought.
        terminal: Whether the first crossing stops the integration.
        direction: `1` counts only rising crossings, `-1` only falling,
            `0` both.
        dense_output: Whether to keep every step for `sol`.
        gpu: Whether the stages and interpolation run on the device.

    Args:
        t0: The initial time.
        y0: The state at `t0`.
        t1: The final time.
        t_eval: The output times.
        rtol: The relative error tolerance per component and step.
        atol: The absolute error tolerance per component and step.
        max_steps: The cap on attempted steps.

    Returns:
        An `IVPSolution` with the output, the event crossings in
        `t_events`/`y_events`, and `status`.

    Raises:
        If `f` or `event` raises, or a tensor copy, launch or transfer fails.
    """
    comptime dtype = T.dtype
    comptime n = dim[T, 0]
    return _solve_ivp_recording[
        dtype, n, f, True, event, terminal, direction, dense_output, gpu
    ](
        t0,
        _same_order(y0, Static[dtype, n]._static_layout()),
        t1,
        _eval_times(t_eval),
        rtol,
        atol,
        max_steps,
    )
