"""The Beta family: `beta(a,b)`, the regularized incomplete beta
`betainc(x,a,b)`, and its inverse `betaincinv(y,a,b)`.

**This module is tier 1.** The continued fraction runs a fixed 100
iterations with no convergence test, `betaincinv` runs twelve Halley
steps against it, and the fraction's two branches are blended with
each argument clamped to the side where that branch is selected, since the
discarded one can be infinite and `0 * inf` is NaN.

`beta` is a one-liner over `lgamma` -- `B(a,b) = Gamma(a)Gamma(b)/Gamma(a+b)`
computed in log space so the individual Gammas can't overflow before their
ratio comes back down.

`betainc` is the substantial one, and it's the first kernel in `numax` where
the textbook algorithm has to be actively reshaped to fit the
fixed-iteration invariant rather than merely truncated. The standard
approach (Numerical Recipes' `betacf`) is a modified-Lentz continued
fraction with *three* data-dependent branches per iteration:

1. `if |del - 1| < EPS: break` -- the convergence test. Dropped: the
   fraction always runs its full iteration count here, the same trade
   `gammainc`'s fixed 100-term series makes.
2. `if |d| < FPMIN: d = FPMIN` -- Lentz's guard against dividing by a
   denominator that has landed on zero. This one can't just be dropped
   (a zero denominator is a real numerical hazard, not an accuracy
   nicety), so it becomes `_guard_away_from_zero` below: branchless, and
   sign-preserving, which the original `if` also was.
3. `if x < (a+1)/(a+b+2)` -- picks between the fraction in `x` and the
   fraction in `1-x` via the symmetry `I_x(a,b) = 1 - I_{1-x}(b,a)`,
   because each converges quickly only on its own side. This becomes the
   usual `0`/`1` blend, with both fractions always evaluated -- which
   means both must be finite everywhere, including on the side where the
   blend is about to discard them. That's what the argument clamping in
   `betainc` is for; see its docstring.
"""

from .gamma import lgamma
from ..core.numeric import (
    FloatLike,
    blend,
    ge_indicator,
    guard_nonzero,
    max_of,
    min_of,
)


def beta[T: FloatLike](a: T, b: T) -> T:
    """The Beta function, `B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)`.

    Scoped to `a > 0`, `b > 0`, where all three Gammas are positive and no
    sign correction is needed. (`lgamma` itself reflects to negative
    arguments, so an `a < 0` extension would work the same way `gammainc`'s
    does -- via `_gamma_sign` -- but negative-parameter Beta has no
    standard use here to justify carrying it.)

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the inputs.

    Args:
        a: The first shape argument, `a > 0`.
        b: The second shape argument, `b > 0`.

    Returns:
        `B(a, b)`, computed as `exp` of a sum of `lgamma` terms.
    """
    return (lgamma(a) + lgamma(b) - lgamma(a + b)).exp()


def betaln[T: FloatLike](a: T, b: T) -> T:
    """The natural log of the Beta function,
    `ln B(a,b) = lgamma(a) + lgamma(b) - lgamma(a+b)`.
    `scipy.special.betaln`.

    Not `beta(a, b).ln()`, and that is the entire point of the name: `beta`
    exponentiates a sum of log-gammas, so for large `a` or `b` it
    underflows to zero and the log of that is `-inf` where the true value
    is merely a large negative number. `betaln(1e4, 1e4)` is about
    `-13864`, which `beta` cannot represent at all.

    Same domain as `beta` -- `a > 0`, `b > 0` -- and **tier 1**: one
    subtraction over three `lgamma` calls, no branching, so it
    differentiates at `Dual` and runs in a kernel.

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the inputs.

    Args:
        a: The first shape argument, `a > 0`.
        b: The second shape argument, `b > 0`.

    Returns:
        `ln B(a, b)`, finite where `beta` itself underflows.
    """
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def _guard_away_from_zero[T: FloatLike](d: T) -> T:
    """Lentz's `if |d| < tiny: d = tiny`, branchless and sign-preserving --
    `guard_nonzero` next to the trait, at this module's chosen floor."""
    comptime tiny = 1e-30
    return guard_nonzero(d, T.constant(tiny))


def _betacf[T: FloatLike](a: T, b: T, x: T) -> T:
    """The incomplete beta continued fraction, by modified Lentz.

    A **fixed** 100 iterations (each covering the fraction's even and odd
    coefficient, so 200 terms), with no convergence test -- see this
    module's docstring. Converges quickly for `x < (a+1)/(a+b+2)` and
    slowly outside it, which is what `betainc`'s symmetry blend exists to
    avoid; iterations past convergence multiply by `d*c` that has settled
    to `1`, so they cost time rather than accuracy.
    """
    comptime num_iters = 100

    var qab = a + b
    var qap = a + T.one()
    var qam = a - T.one()

    var c = T.one()
    var d = T.one() / _guard_away_from_zero(T.one() - (qab * x / qap))
    var h = d.copy()

    # Counters carried as `T` values rather than converted from the `Int`
    # index per iteration -- see `numax.special.orthopoly`'s module docstring for
    # why the obvious `T.constant(Float64(m))` would make this CPU-only.
    var mf = T.one()
    var m2 = T.constant(2.0)

    for _ in range(1, num_iters + 1):
        # Even step: the d_{2m} coefficient.
        var aa = mf * (b - mf) * x / ((qam + m2) * (a + m2))
        d = T.one() / _guard_away_from_zero(T.one() + aa * d)
        c = _guard_away_from_zero(T.one() + aa / c)
        h = h * d * c

        # Odd step: the d_{2m+1} coefficient.
        aa = -((a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2)))
        d = T.one() / _guard_away_from_zero(T.one() + aa * d)
        c = _guard_away_from_zero(T.one() + aa / c)
        h = h * d * c

        mf = mf + T.one()
        m2 = m2 + T.constant(2.0)

    return h^


def betainc[T: FloatLike](x: T, a: T, b: T) -> T:
    """The regularized incomplete beta function, `I_x(a,b)`.

    `I_x(a,b) = B(x;a,b)/B(a,b)` -- the Beta distribution's CDF, and by
    extension the CDF behind Student-t, F, and the binomial. Scoped to
    `0 <= x <= 1`, `a > 0`, `b > 0`.

    Both the direct fraction and its mirror image are evaluated, then
    blended on `x` versus `(a+1)/(a+b+2)`. Two clamps make that safe:

    - `x` is clamped into `[0, 1]` before reaching `ln`, so a caller whose
      `x` drifted a rounding step outside gets an endpoint rather than a
      NaN. The clamp is exact, not `[eps, 1 - eps]`: at `x = 0` the
      logarithm is `-inf` and the prefactor `exp(-inf)` is exactly `0` on
      the side the blend selects, so the lower tail is `I_x ~ x^a / (a B)`
      all the way down rather than frozen at its value at `eps`.
    - Each fraction's own argument is clamped to the side of the threshold
      where that fraction is the one actually selected (`min_of(x, t)` for
      the direct one, `min_of(1-x, 1-t)` for the mirror). This is a no-op
      wherever the blend selects that branch, and elsewhere it keeps a
      fraction that would otherwise be evaluated at a badly conditioned
      argument -- `x = 1` in particular, where the direct fraction can
      diverge -- finite instead. Without it the discarded branch could be
      an infinity, and `0 * inf` is NaN, which would poison the blend that
      was supposed to throw it away.

    At the endpoints the prefactor `exp(a*ln(x) + b*ln(1-x) + ...)`
    underflows to exactly zero, so `I_0 = 0` and `I_1 = 1` fall out without
    a special case.
    """
    var xs = min_of(max_of(x, T.constant(0.0)), T.one())
    var one_minus = T.one() - xs

    var log_prefactor = (
        lgamma(a + b) - lgamma(a) - lgamma(b) + a * xs.ln() + b * one_minus.ln()
    )
    var prefactor = log_prefactor.exp()

    var threshold = (a + T.one()) / (a + b + T.constant(2.0))
    var direct = prefactor * _betacf(a, b, min_of(xs, threshold)) / a
    var mirrored = T.one() + (
        -(prefactor * _betacf(b, a, min_of(one_minus, T.one() - threshold)) / b)
    )

    return blend(ge_indicator(xs, threshold), mirrored, direct)


def betaincc[T: FloatLike](x: T, a: T, b: T) -> T:
    """The complement of `betainc`, `1 - I_x(a,b)`.

    Equal to `I_{1-x}(b,a)` by the same symmetry `betainc` uses
    internally; written as the subtraction here to match `gammaincc`'s
    shape next door.
    """
    return T.one() - betainc(x, a, b)


def betaincinv[T: FloatLike](y: T, a: T, b: T) -> T:
    """The inverse of `betainc` in `x`: the `x` in `[0, 1]` with
    `betainc(x, a, b) == y`, for `a, b > 0`. `scipy.special.betaincinv(a,
    b, y)`, with the target first to mirror this module's `betainc(x, a,
    b)` rather than SciPy's order.

    Tier 1. Numerical Recipes' starting guess (*Numerical Recipes*, 3rd
    ed., 6.4) -- for `a, b >= 1` a normal quantile mapped through the
    Beta's Cornish-Fisher form, bounded below by the lower tail's leading
    term `(y a B(a, b))^(1/a)` and above by the mirror of the upper
    tail's, neither of which overshoots there; otherwise that leading term
    of whichever tail `y` falls in, split where NR splits. Then twelve Halley steps
    against `betainc`, with `dI/dx = x^(a-1) (1-x)^(b-1) / B(a, b)` and NR's
    guards: a step that would leave `(0, 1)` goes halfway to the edge
    instead. Every guess is evaluated at shapes clamped into its own
    region, so the choice is a blend and the function runs in a kernel.

    As accurate as `betainc` at the result, which a small shape amplifies:
    near 0 the root goes as `y^(1/a)`, so `betainc`'s relative error
    reaches `x` multiplied by `1/a`. `pixi run accuracy` reads `3e-15`
    relative or better at `(2, 3)` and `(5, 1.5)` for `y` in `[1e-12,
    0.99]`, `8e-15` at `(0.5, 0.5)`, and `1.3e-13` at `(0.3, 50)`. `betaincinv(0, a, b)` is `0` and
    `betaincinv(1, a, b)` is `1`.

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the inputs.

    Args:
        y: The target probability, in `[0, 1]`.
        a: The first shape, `a > 0`.
        b: The second shape, `b > 0`.

    Returns:
        The `x` with `betainc(x, a, b) == y`.
    """
    var zero = T.constant(0.0)
    var one = T.one()
    var half = T.constant(0.5)
    var two = T.constant(2.0)
    var at_zero = ge_indicator(zero, y.abs())
    var at_one = ge_indicator(y, one)
    var p = y + at_zero * T.constant(0.25) - at_one * half
    var log_b = lgamma(a) + lgamma(b) - lgamma(a + b)

    # `a, b >= 1`: a normal deviate through the Beta's Cornish-Fisher form.
    var both = ge_indicator(a, one) * ge_indicator(b, one)
    var ab = max_of(a, one)
    var bb = max_of(b, one)
    var pp = min_of(p, one - p)
    var t = (-two * pp.ln()).sqrt()
    var z = (T.constant(2.30753) + t * T.constant(0.27061)) / (
        one + t * (T.constant(0.99229) + t * T.constant(0.04481))
    ) - t
    z = blend(ge_indicator(p, half), z, -z)
    var al = (z * z - T.constant(3.0)) / T.constant(6.0)
    var ia = one / (two * ab - one)
    var ib = one / (two * bb - one)
    var h = two / (ia + ib)
    var w = z * (al + h).sqrt() / h - (ib - ia) * (
        al + T.constant(5.0 / 6.0) - two / (T.constant(3.0) * h)
    )
    var guess_normal = ab / (ab + bb * (two * w).exp())

    # The two tails' leading terms: `I_x ~ x^a / (a B)` near 0 and
    # `1 - I_x ~ (1 - x)^b / (b B)` near 1. For `a, b >= 1` they bound the
    # root from below and above; they also keep the Cornish-Fisher guess
    # off exactly 0 or 1.
    var lead_low = ((p.ln() + a.ln() + log_b) / a).exp()
    var lead_high = one - (((one - p).ln() + b.ln() + log_b) / b).exp()
    var bounded = min_of(max_of(guess_normal, lead_low), lead_high)

    # Otherwise: whichever leading term belongs to the tail `p` falls in,
    # split where NR splits, at the lower tail's share `t_a / (t_a + t_b)`
    # of `x^a / a` and `(1 - x)^b / b` at the mode-like point `a / (a + b)`.
    # NR inverts the unnormalized power laws there; the normalized ones
    # are the same shape and stay right as `y -> 0`.
    var s = a + b
    var ta = ((a * (a / s).ln()).exp()) / a
    var tb = ((b * (b / s).ln()).exp()) / b
    var total = max_of(ta + tb, T.constant(1e-30))
    var lower = one - ge_indicator(p, ta / total)
    var guess_power = blend(lower, lead_low, lead_high)

    var x = blend(both, bounded, guess_power)

    var a1 = a - one
    var b1 = b - one
    for _ in range(12):
        # `1 - x` floored: a root that rounds to 1 would otherwise put
        # `ln(0)` in the density. The floor never binds below 1, where
        # `1 - x` is at least an ulp of 1.
        var gap = max_of(one - x, T.constant(1e-30))
        var err = betainc(x, a, b) - p
        var density = (a1 * x.ln() + b1 * gap.ln() - log_b).exp()
        var u = err / density
        var step = u / (one - half * min_of(one, u * (a1 / x - b1 / gap)))
        var next = x - step
        var below = ge_indicator(zero, next)
        var above = ge_indicator(next, one)
        x = blend(below, half * x, blend(above, half * (x + one), next))
    return blend(at_zero, zero, blend(at_one, one, x))
