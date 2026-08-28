"""Root finders where the caller supplies only `f`.

`newton` needs `f'`, and `halley` needs `f'` and `f''`. Every other library
makes that the caller's problem -- hand in a second (and third) function,
or accept a finite-difference approximation and the step-size tuning that
comes with it. Here the caller writes one function, generic over
`FloatLike` the way every kernel in `numax` already is, and the solver
evaluates it at `Dual` (or `Dual[Dual[...]]`) internally to get the
derivatives exactly:

```mojo
def cubic[U: FloatLike](x: U) -> U:
    return x * x * x + (-U.constant(2.0))

var root = newton[f=cubic](Plain[DType.float64, 1].constant(1.0))
```

That `f: def[U: FloatLike](U) thin -> U` parameter is the whole design.
It's the same trait-generic shape `numax`'s own kernels are written in, so
any of them can be handed to a solver as-is, and the derivative that comes
back is analytic rather than a difference quotient -- exact to the last
bit, with no step size to pick and no cancellation at small steps.

`numax.lambertw` already did this by hand: its Halley iteration hardcodes
`f`, `f'`, and `f''` for `w*exp(w) - x` because there was nowhere to put
the general version. This is that general version. It's left alone rather
than rewritten on top of this -- its seed logic is what's interesting
there, and it doesn't need the generality.

Every solver here runs a **fixed** iteration count with no convergence
test, the same rule every kernel in `numax` follows: a data-dependent
stopping rule would mean SIMD lanes disagreeing about when to stop, which
`FloatLike` has no way to express. Iterations past convergence are wasted
work, not lost accuracy -- Newton at a converged root computes a correction
of zero.
"""

from .dual import Dual
from .numeric import FloatLike, blend, ge_indicator, guard_nonzero

comptime _DERIVATIVE_FLOOR = 1e-300


def newton[
    T: FloatLike,
    f: def[U: FloatLike](U) thin -> U,
    num_iters: Int = 20,
](x0: T) -> T:
    """Solve `f(x) = 0` by Newton's method from the initial guess `x0`.

    `x <- x - f(x)/f'(x)`, with `f'` obtained by evaluating `f` at a `Dual`
    seeded with derivative `1` -- one pass produces both, since forward-mode
    carries the derivative alongside the value rather than in a second
    evaluation.

    Quadratically convergent near a simple root, so the default 20
    iterations is generous for a decent starting guess and useless for a bad
    one: like every fixed-iteration algorithm here, this converges or it
    doesn't, and there's no error flag to check. The usual Newton caveats
    apply unchanged (a starting guess in the wrong basin finds a different
    root; a multiple root degrades to linear convergence).

    The denominator is floored away from zero by magnitude, so a lane
    landing exactly on a critical point produces a huge-but-finite step
    rather than a NaN that would then contaminate every subsequent
    iteration for that lane.
    """
    var x = x0.copy()

    for _ in range(num_iters):
        var fx = f[Dual[T]](Dual[T](x.copy(), T.one()))
        var denom = guard_nonzero(fx.deriv, T.constant(_DERIVATIVE_FLOOR))
        x = x + (-(fx.value / denom))

    return x^


def halley[
    T: FloatLike,
    f: def[U: FloatLike](U) thin -> U,
    num_iters: Int = 10,
](x0: T) -> T:
    """Solve `f(x) = 0` by Halley's method from the initial guess `x0`.

    `x <- x - 2*f*f' / (2*f'^2 - f*f'')`, cubically convergent -- roughly
    tripling the number of correct digits per step where Newton doubles it,
    at the cost of needing `f''`.

    Getting `f''` costs nothing at the call site: `f` is evaluated at
    `Dual[Dual[T]]`, and the outer `Dual` differentiating the inner one is
    exactly the second derivative (the same nesting `tests/test_dual.mojo`
    checks, and the same one `Gradient[Dual[...]]` uses for Hessians). The
    caller still writes one function.

    Fewer default iterations than `newton` for the same reason Halley is
    worth using at all -- if it's going to converge, it does so in fewer
    steps.
    """
    comptime DT = Dual[T]
    comptime DDT = Dual[DT]

    var x = x0.copy()

    for _ in range(num_iters):
        # value.value = f, deriv.value = f', deriv.deriv = f''.
        var seed = DDT(DT(x.copy(), T.one()), DT(T.one(), T.constant(0.0)))
        var r = f[DDT](seed)
        var fx = r.value.value.copy()
        var d1 = r.deriv.value.copy()
        var d2 = r.deriv.deriv.copy()

        var numerator = T.constant(2.0) * fx * d1
        var denominator = T.constant(2.0) * (d1 * d1) + (-(fx * d2))
        x = x + (
            -(
                numerator
                / guard_nonzero(denominator, T.constant(_DERIVATIVE_FLOOR))
            )
        )

    return x^


def bisection[
    T: FloatLike,
    f: def[U: FloatLike](U) thin -> U,
    num_iters: Int = 60,
](lo0: T, hi0: T) -> T:
    """Solve `f(x) = 0` on a bracket where `f(lo0)` and `f(hi0)` differ in
    sign, by branchless bisection.

    Unlike `newton`/`halley` this needs no derivative and cannot diverge --
    given a genuine sign-change bracket it halves the interval every
    iteration, so 60 iterations narrows a bracket by a factor of `2^60`,
    past `float64` resolution for any reasonable starting width. That makes
    it the fallback when a starting guess is unavailable or Newton's basin
    is unclear.

    Bisection's textbook form is a branch (`if f(lo)*f(mid) < 0: hi = mid
    else: lo = mid`), which is exactly what a SIMD lane can't have -- two
    lanes bisecting different functions, or the same function from
    different brackets, will want opposite updates on the same iteration.
    Both updates are therefore computed and blended on a `0`/`1` indicator,
    which is not a workaround here so much as the natural form: bisection
    does the same amount of work either way, so there is nothing lost by
    computing both endpoints and discarding one.

    A bracket that doesn't actually straddle a root converges to an
    endpoint rather than reporting an error -- fixed-work algorithms have no
    channel to report one through.
    """
    var lo = lo0.copy()
    var hi = hi0.copy()
    var f_lo = f[T](lo.copy())

    for _ in range(num_iters):
        var mid = (lo + hi) / T.constant(2.0)
        var f_mid = f[T](mid.copy())

        # 1 where f(lo) and f(mid) share a sign, meaning the root is in
        # [mid, hi] and `lo` should move up to `mid`.
        var same_sign = ge_indicator(f_lo * f_mid, T.constant(0.0))

        lo = blend(same_sign, mid.copy(), lo.copy())
        hi = blend(same_sign, hi.copy(), mid.copy())
        f_lo = blend(same_sign, f_mid.copy(), f_lo.copy())

    return (lo + hi) / T.constant(2.0)
