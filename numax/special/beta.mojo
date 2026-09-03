"""The Beta family: `beta(a,b)` and the regularized incomplete beta
`betainc(x,a,b)`.

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
    """
    return (lgamma(a) + lgamma(b) + (-lgamma(a + b))).exp()


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
    var qam = a + (-T.one())

    var c = T.one()
    var d = T.one() / _guard_away_from_zero(T.one() + (-(qab * x / qap)))
    var h = d.copy()

    # Counters carried as `T` values rather than converted from the `Int`
    # index per iteration -- see `numax.special.orthopoly`'s module docstring for
    # why the obvious `T.constant(Float64(m))` would make this CPU-only.
    var mf = T.one()
    var m2 = T.constant(2.0)

    for _ in range(1, num_iters + 1):
        # Even step: the d_{2m} coefficient.
        var aa = mf * (b + (-mf)) * x / ((qam + m2) * (a + m2))
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

    - `x` is clamped into `[eps, 1-eps]` before reaching `ln`, so a caller
      whose `x` drifted a rounding step outside `[0,1]` gets `ln` of a
      positive number rather than a NaN.
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
    comptime edge = 1e-30

    var xs = min_of(max_of(x, T.constant(edge)), T.one() + (-T.constant(edge)))
    var one_minus = T.one() + (-xs)

    var log_prefactor = (
        lgamma(a + b)
        + (-lgamma(a))
        + (-lgamma(b))
        + a * xs.ln()
        + b * one_minus.ln()
    )
    var prefactor = log_prefactor.exp()

    var threshold = (a + T.one()) / (a + b + T.constant(2.0))
    var direct = prefactor * _betacf(a, b, min_of(xs, threshold)) / a
    var mirrored = T.one() + (
        -(
            prefactor
            * _betacf(b, a, min_of(one_minus, T.one() + (-threshold)))
            / b
        )
    )

    return blend(ge_indicator(xs, threshold), mirrored, direct)


def betaincc[T: FloatLike](x: T, a: T, b: T) -> T:
    """The complement of `betainc`, `1 - I_x(a,b)`.

    Equal to `I_{1-x}(b,a)` by the same symmetry `betainc` uses
    internally; written as the subtraction here to match `gammaincc`'s
    shape next door.
    """
    return T.one() + (-betainc(x, a, b))
