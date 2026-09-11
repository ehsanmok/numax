"""Probability densities, cumulative distributions, and quantiles.

**This module is tier 1.** The CDFs are changes of variable over
`gammainc`/`betainc`/`erfc`, and each quantile runs its own fixed-iteration
Newton loop against the analytic density rather than iterating to a
tolerance.

Nine namespaces, spelled the way `scipy.stats` spells them -- `norm`,
`expon`, `gamma`, `chi2`, `beta`, `t`, `f`, `poisson`, `binom` -- each
carrying the eight methods a `scipy.stats` distribution carries: `.pdf`
(or `.pmf`) and `.logpdf` (`.logpmf`), `.cdf` and `.logcdf`, `.sf` and
`.logsf`, `.ppf` and `.isf`:

```mojo
from numax.stats import norm, chi2
var p = norm.cdf(x, mu, sigma)
var q = chi2.ppf(P(0.95), P(3.0))
```

`gamma` and `beta` are the *distributions*; `numax.special.gamma` and
`numax.special.beta` are the functions they are named after. That is why
these namespaces are reached through `numax.stats` and are not re-exported
at the root.

Almost none of this is new numerics. A distribution's CDF is nearly always
a special function already in `numax` under a change of variables -- the
gamma family is `gammainc`, the beta family (which includes Student-t, F,
and the binomial) is `betainc`, the normal family is `erfc` -- so the work
here is composition, and the payoff is that every one of them inherits
`Dual`-differentiability, `Compensated` precision, and GPU-launchability
from the function underneath.

`d/dx` of a CDF is the PDF, and evaluating any `.cdf` here at `Dual`
recovers exactly that, which is how `tests/stats/test_distributions.mojo` checks
each pair against the other rather than against a table.

## Conventions

- Shape parameters are `T` values, not `Int`s, so they can vary per SIMD
  lane. That includes the discrete distributions' counts: `poisson.cdf`'s
  `k` and `binom.cdf`'s `n`/`k` are continuous extensions of the usual
  integer-argument definitions, which is what the underlying `gammaincc`
  and `betainc` compute anyway.
- Densities are `0` outside their support rather than undefined, applied by
  a branchless indicator. The log-space interior is always evaluated on a
  clamped argument first, so the discarded side never produces the NaN that
  would survive multiplication by a `0` indicator.
- Everything is scoped to valid parameters (positive shapes, `sigma > 0`,
  `0 < p < 1`); nothing validates, in keeping with the rest of `numax`.
- Log densities are the log-space interior directly, never `ln(pdf)`, so
  they stay finite where the density underflows -- which is what a log
  density is for. Outside the support they return `_LOG_ZERO`, a large
  finite negative, rather than SciPy's `-inf`: finite so a `Dual`
  derivative through it stays finite, and chosen so that `exp` of it is
  exactly the `0` the density returns there.
- Survival functions read the upper tail off the complement special
  function (`erfc`, `betaincc`, `gammainc` for the discrete upper tails)
  where one exists, so they keep their digits where `1 - cdf` would cancel
  them. `gamma.sf` is the exception in effect if not in spelling: numax's
  `gammaincc` is literally `1 - gammainc`, so the name is there and the
  tail accuracy waits on an upper-tail continued fraction.

## Quantiles

Each quantile is a seed plus a fixed number of Newton steps against its own
CDF, using the analytic PDF as the derivative. They do *not* go through
`numax.optimize.newton`, and the reason is a real limitation worth naming:
`solve`'s `f` is `def[U: FloatLike](U) thin -> U`, a non-capturing function
required to be `thin` so a solver can be launched on GPU. A distribution's
parameters have nowhere to live in that signature -- `beta.ppf(p, a,
b)` would need `a` and `b` inside `f`, which a `thin` function cannot
close over. Writing the loop locally also happens to be cheaper here, since
the exact derivative is already in hand and doesn't need a `Dual` pass.

Seeds matter more than iteration counts for a fixed-work solver, so each
one uses the standard published approximation for its family rather than a
constant: Abramowitz & Stegun 26.2.23 for the normal, Wilson-Hilferty for
the gamma, the distribution mean for the beta. `f.ppf` is `beta.ppf` under
the change of variables its CDF already uses, and `expon.ppf` is closed
form. Every `isf` is its `ppf` at `1 - p`, or the symmetric spelling of it.

## Discrete quantiles

`poisson.ppf` and `binom.ppf` return what SciPy returns -- the smallest
integer `k` with `cdf(k) >= p` -- and do it branchlessly: `k` starts at
`0` and, for a compile-time `max_k` steps, adds `1` wherever `cdf(k)` is
still below `p`. Once the CDF has caught up the indicator is `0` and `k`
stops. Fixed work, no per-lane branch, and no `floor`, which `FloatLike`
does not have and which is why a Newton step on the continuous extension
is not the route.

`ponytail:` the cost is `max_k` CDF evaluations per lane -- `max_k`
incomplete-gamma series or incomplete-beta continued fractions -- and a
quantile above `max_k` is silently reported as `max_k`. The default of
`64` covers a Poisson mean up to about `40` at `p = 0.999`; pass a larger
`max_k` for a larger rate. The upgrade is a normal-approximation seed plus
a short scan around it, once `FloatLike` can round a seed to an integer.
"""

from ..special.beta import betainc, betaincc
from ..special.gamma import gammainc, gammaincc, lgamma
from ..core.numeric import FloatLike, blend, ge_indicator, max_of, min_of

comptime _SQRT_2 = 1.4142135623730951
comptime _SQRT_2PI = 2.5066282746310002
comptime _LN_2PI = 1.8378770664093453
comptime _LN_PI = 1.1447298858494002

# Small enough to be indistinguishable from the boundary at any supported
# `dtype`, large enough that `ln` of it is finite in float32.
comptime _TINY = 1e-30

# What a log density returns outside its support: finite, representable at
# float32, and with `exp` of it exactly `0` -- see the module docstring.
comptime _LOG_ZERO = -1e30


def _safe_ln[T: FloatLike](x: T) -> T:
    """`ln(x)` with the argument floored away from zero.

    Every density below evaluates its log-space interior even on lanes
    outside the support, because a branchless blend evaluates both sides.
    This keeps that evaluation finite so the discarded side contributes a
    large negative number rather than the `inf - inf` NaN that would
    survive multiplication by a `0` indicator.

    Note the two-stage clamp. Flooring straight at `_TINY` is not enough
    for an argument that can be arbitrarily negative -- `max_of` is an
    arithmetic identity, and `max_of(-1.0, 1e-30)` cancels to exactly `0.0`
    (see its own docstring). Clamping at `0` first is exact, and the second
    clamp then lifts that `0` to `_TINY` as intended.
    """
    return max_of(max_of(x, T.constant(0.0)), T.constant(_TINY)).ln()


def _log_beta[T: FloatLike](a: T, b: T) -> T:
    return lgamma(a.copy()) + lgamma(b.copy()) - lgamma(a + b)


# ---------------------------------------------------------------- normal


struct norm:
    """The normal (Gaussian) distribution. `scipy.stats.norm`."""

    @staticmethod
    def pdf[T: FloatLike](x: T, mu: T, sigma: T) -> T:
        """The normal density with mean `mu` and standard deviation `sigma`."""
        var z = (x - mu) / sigma
        return (-(z * z) / T.constant(2.0)).exp() / (
            sigma * T.constant(_SQRT_2PI)
        )

    @staticmethod
    def cdf[T: FloatLike](x: T, mu: T, sigma: T) -> T:
        """The normal CDF, as `0.5*erfc(-z/sqrt(2))`.

        Written with `erfc` rather than the equivalent `0.5*(1 + erf(z/sqrt2))`
        on purpose: for `z` around `-6` the `erf` form is `0.5*(1 - 0.999...)`,
        which cancels away most of its significant digits, while `erfc` returns
        the small tail probability directly. `Plain.erfc` delegates to
        `std.math`, so that accuracy is real rather than nominal.
        """
        var z = (x - mu) / sigma
        return T.constant(0.5) * (-(z / T.constant(_SQRT_2))).erfc()

    @staticmethod
    def ppf[T: FloatLike](p: T, mu: T, sigma: T) -> T:
        """The inverse normal CDF, for `0 < p < 1`."""
        return mu + sigma * _standard_normal_quantile(p)

    @staticmethod
    def sf[T: FloatLike](x: T, mu: T, sigma: T) -> T:
        """`P(X > x)`, as `0.5*erfc(z/sqrt(2))` -- the upper tail read off
        `erfc` directly, so it keeps its digits where `1 - cdf` would
        cancel them. `scipy.stats.norm.sf`."""
        var z = (x - mu) / sigma
        return T.constant(0.5) * (z / T.constant(_SQRT_2)).erfc()

    @staticmethod
    def isf[T: FloatLike](p: T, mu: T, sigma: T) -> T:
        """The inverse survival function, `ppf(1 - p)`; by symmetry
        `mu - sigma * z(p)`, which loses nothing to `1 - p`."""
        return mu - sigma * _standard_normal_quantile(p)

    @staticmethod
    def logpdf[T: FloatLike](x: T, mu: T, sigma: T) -> T:
        var z = (x - mu) / sigma
        return (
            -(z * z) / T.constant(2.0)
            - _safe_ln(sigma)
            - T.constant(0.5 * _LN_2PI)
        )

    @staticmethod
    def logcdf[T: FloatLike](x: T, mu: T, sigma: T) -> T:
        return _safe_ln(norm.cdf(x, mu, sigma))

    @staticmethod
    def logsf[T: FloatLike](x: T, mu: T, sigma: T) -> T:
        return _safe_ln(norm.sf(x, mu, sigma))


def _standard_normal_quantile[T: FloatLike, num_iters: Int = 3](p: T) -> T:
    """The standard normal quantile: an Abramowitz & Stegun 26.2.23 seed
    (max error ~4.5e-4) refined by Newton against `norm.cdf`.

    The seed is defined for the lower half only, so the upper half is
    handled by the symmetry `z(p) = -z(1-p)`, applied branchlessly: `q` is
    the smaller of `p` and `1-p` either way, and the sign comes from an
    indicator on `p >= 0.5`.
    """
    var q = min_of(p, T.one() - p)
    var t = (-(T.constant(2.0) * _safe_ln(q))).sqrt()

    var numerator = (
        T.constant(2.515517)
        + T.constant(0.802853) * t
        + (T.constant(0.010328) * t * t)
    )
    var denominator = (
        T.one()
        + T.constant(1.432788) * t
        + T.constant(0.189269) * t * t
        + T.constant(0.001308) * t * t * t
    )
    # A&S 26.2.23 gives the *upper*-tail value, hence the negation for the
    # lower tail `q` names.
    var lower = -(t - (numerator / denominator))
    var z = blend(ge_indicator(p, T.constant(0.5)), -lower, lower.copy())

    var zero = T.constant(0.0)
    var one = T.one()
    for _ in range(num_iters):
        var residual = norm.cdf(z.copy(), zero.copy(), one.copy())
        z = z + (
            -(
                (residual - p)
                / max_of(
                    norm.pdf(z.copy(), zero.copy(), one.copy()),
                    T.constant(_TINY),
                )
            )
        )

    return z^


# ----------------------------------------------------------- exponential


struct expon:
    """The exponential distribution, parameterized by `rate` (`1/scale`). `scipy.stats.expon`.
    """

    @staticmethod
    def pdf[T: FloatLike](x: T, rate: T) -> T:
        """`rate*exp(-rate*x)` on `x >= 0`, and `0` below it."""
        var inside = rate * (-(rate * max_of(x, T.constant(0.0)))).exp()
        return inside * ge_indicator(x, T.constant(0.0))

    @staticmethod
    def cdf[T: FloatLike](x: T, rate: T) -> T:
        var inside = T.one() - (-(rate * max_of(x, T.constant(0.0)))).exp()
        return inside * ge_indicator(x, T.constant(0.0))

    @staticmethod
    def sf[T: FloatLike](x: T, rate: T) -> T:
        """`exp(-rate*x)` on `x >= 0`, and `1` below it."""
        var inside = (-(rate * max_of(x, T.constant(0.0)))).exp()
        return blend(ge_indicator(x, T.constant(0.0)), inside, T.one())

    @staticmethod
    def ppf[T: FloatLike](p: T, rate: T) -> T:
        """The inverse CDF, closed form: `-ln(1 - p) / rate`."""
        return -(_safe_ln(T.one() - p)) / rate

    @staticmethod
    def isf[T: FloatLike](p: T, rate: T) -> T:
        """`-ln(p) / rate`, which is `ppf(1 - p)` without the `1 - p`."""
        return -(_safe_ln(p)) / rate

    @staticmethod
    def logpdf[T: FloatLike](x: T, rate: T) -> T:
        var interior = _safe_ln(rate) - rate * max_of(x, T.constant(0.0))
        return blend(
            ge_indicator(x, T.constant(0.0)), interior, T.constant(_LOG_ZERO)
        )

    @staticmethod
    def logcdf[T: FloatLike](x: T, rate: T) -> T:
        return _safe_ln(expon.cdf(x, rate))

    @staticmethod
    def logsf[T: FloatLike](x: T, rate: T) -> T:
        return _safe_ln(expon.sf(x, rate))

    # ----------------------------------------------------------------- gamma


struct gamma:
    """The gamma distribution. `scipy.stats.gamma`. Distinct from `numax.special.gamma`, the function -- this is the distribution named after it.
    """

    @staticmethod
    def pdf[T: FloatLike](x: T, shape: T, scale: T) -> T:
        """The gamma density in the shape/scale parameterization.

        Computed in log space, which is what keeps it usable for large `shape`:
        the direct form needs `x^(shape-1)` and `Gamma(shape)` separately, and
        both overflow long before their ratio does.
        """
        return gamma.logpdf(x, shape, scale).exp()

    @staticmethod
    def logpdf[T: FloatLike](x: T, shape: T, scale: T) -> T:
        """The gamma log density; `_LOG_ZERO` below the support."""
        var z = max_of(x, T.constant(0.0)) / scale
        var interior = (
            (shape - T.one()) * _safe_ln(z)
            - z
            - lgamma(shape.copy())
            - _safe_ln(scale)
        )
        return blend(
            ge_indicator(x, T.constant(0.0)), interior, T.constant(_LOG_ZERO)
        )

    @staticmethod
    def cdf[T: FloatLike](x: T, shape: T, scale: T) -> T:
        """The gamma CDF -- `gammainc` under the substitution `x/scale`."""
        var z = max_of(x, T.constant(0.0)) / scale
        return gammainc(shape.copy(), z^) * ge_indicator(x, T.constant(0.0))

    @staticmethod
    def sf[T: FloatLike](x: T, shape: T, scale: T) -> T:
        """`P(X > x)` -- `gammaincc` under the same substitution, `1` below
        the support. See the module docstring on what `gammaincc` is."""
        var z = max_of(x, T.constant(0.0)) / scale
        return blend(
            ge_indicator(x, T.constant(0.0)),
            gammaincc(shape.copy(), z^),
            T.one(),
        )

    @staticmethod
    def logcdf[T: FloatLike](x: T, shape: T, scale: T) -> T:
        return _safe_ln(gamma.cdf(x, shape, scale))

    @staticmethod
    def logsf[T: FloatLike](x: T, shape: T, scale: T) -> T:
        return _safe_ln(gamma.sf(x, shape, scale))

    @staticmethod
    def ppf[T: FloatLike, num_iters: Int = 12](p: T, shape: T, scale: T) -> T:
        """The inverse gamma CDF, from a Wilson-Hilferty seed.

        Wilson-Hilferty approximates the gamma as a cube-rooted normal, which
        is accurate for `shape` above about 1 and increasingly rough below it.
        The Newton refinement covers the difference; the seed is floored away
        from zero because the cube can go negative for small `shape` and large
        lower-tail `p`, and `ln` of a negative seed would poison the lane.
        """
        var z = _standard_normal_quantile(p.copy())
        var a9 = T.constant(9.0) * shape
        var base = T.one() - (T.one() / a9) + z / a9.sqrt()
        var seed = max_of(shape * base * base * base, T.constant(1e-6))

        var x = seed * scale
        for _ in range(num_iters):
            var residual = gamma.cdf(x.copy(), shape.copy(), scale.copy())
            var density = max_of(
                gamma.pdf(x.copy(), shape.copy(), scale.copy()),
                T.constant(_TINY),
            )
            x = max_of(x - ((residual - p) / density), T.constant(_TINY))

        return x^

    @staticmethod
    def isf[T: FloatLike, num_iters: Int = 12](p: T, shape: T, scale: T) -> T:
        """`ppf(1 - p)`."""
        return gamma.ppf[T, num_iters](T.one() - p, shape, scale)

    # ------------------------------------------------------------ chi-square


struct chi2:
    """The chi-squared distribution. `scipy.stats.chi2`."""

    @staticmethod
    def pdf[T: FloatLike](x: T, df: T) -> T:
        """Chi-square with `df` degrees of freedom -- gamma with
        `shape = df/2`, `scale = 2`."""
        return gamma.pdf(x, df / T.constant(2.0), T.constant(2.0))

    @staticmethod
    def cdf[T: FloatLike](x: T, df: T) -> T:
        return gamma.cdf(x, df / T.constant(2.0), T.constant(2.0))

    @staticmethod
    def ppf[T: FloatLike](p: T, df: T) -> T:
        return gamma.ppf(p, df / T.constant(2.0), T.constant(2.0))

    @staticmethod
    def sf[T: FloatLike](x: T, df: T) -> T:
        return gamma.sf(x, df / T.constant(2.0), T.constant(2.0))

    @staticmethod
    def isf[T: FloatLike](p: T, df: T) -> T:
        return gamma.isf(p, df / T.constant(2.0), T.constant(2.0))

    @staticmethod
    def logpdf[T: FloatLike](x: T, df: T) -> T:
        return gamma.logpdf(x, df / T.constant(2.0), T.constant(2.0))

    @staticmethod
    def logcdf[T: FloatLike](x: T, df: T) -> T:
        return gamma.logcdf(x, df / T.constant(2.0), T.constant(2.0))

    @staticmethod
    def logsf[T: FloatLike](x: T, df: T) -> T:
        return gamma.logsf(x, df / T.constant(2.0), T.constant(2.0))

    # ------------------------------------------------------------------ beta


struct beta:
    """The beta distribution. `scipy.stats.beta`. Distinct from `numax.special.beta`, the function.
    """

    @staticmethod
    def pdf[T: FloatLike](x: T, a: T, b: T) -> T:
        """The beta density on `0 < x < 1`, and `0` outside it."""
        return beta.logpdf(x, a, b).exp()

    @staticmethod
    def logpdf[T: FloatLike](x: T, a: T, b: T) -> T:
        """The beta log density; `_LOG_ZERO` outside `[0, 1]`."""
        var xc = min_of(max_of(x, T.constant(0.0)), T.one())
        var interior = (
            (a - T.one()) * _safe_ln(xc)
            + (b - T.one()) * _safe_ln(T.one() - xc)
            - _log_beta(a, b)
        )
        var support = ge_indicator(x, T.constant(0.0)) * ge_indicator(
            T.one(), x.copy()
        )
        return blend(support, interior, T.constant(_LOG_ZERO))

    @staticmethod
    def cdf[T: FloatLike](x: T, a: T, b: T) -> T:
        """The beta CDF -- `betainc` directly, clamped to its support."""
        var xc = min_of(max_of(x, T.constant(0.0)), T.one())
        return betainc(xc^, a, b)

    @staticmethod
    def sf[T: FloatLike](x: T, a: T, b: T) -> T:
        """`P(X > x)` -- `betaincc`, which is `I_{1-x}(b, a)` and so reads
        the upper tail directly rather than as `1 - cdf`."""
        var xc = min_of(max_of(x, T.constant(0.0)), T.one())
        return betaincc(xc^, a, b)

    @staticmethod
    def logcdf[T: FloatLike](x: T, a: T, b: T) -> T:
        return _safe_ln(beta.cdf(x, a, b))

    @staticmethod
    def logsf[T: FloatLike](x: T, a: T, b: T) -> T:
        return _safe_ln(beta.sf(x, a, b))

    @staticmethod
    def ppf[T: FloatLike, num_iters: Int = 20](p: T, a: T, b: T) -> T:
        """The inverse beta CDF, Newton from the distribution's mean.

        Each step is clamped back inside `(0, 1)`: Newton on a CDF that
        saturates near both endpoints readily proposes a point outside the
        support, and a clamped step is a bounded loss where an escaped one is
        unrecoverable.
        """
        var lo = T.constant(1e-8)
        var hi = T.one() - lo
        var x = min_of(max_of(a / (a + b), lo.copy()), hi.copy())

        for _ in range(num_iters):
            var residual = beta.cdf(x.copy(), a.copy(), b.copy())
            var density = max_of(
                beta.pdf(x.copy(), a.copy(), b.copy()), T.constant(_TINY)
            )
            var proposal = x - ((residual - p) / density)
            x = min_of(max_of(proposal^, lo.copy()), hi.copy())

        return x^

    @staticmethod
    def isf[T: FloatLike, num_iters: Int = 20](p: T, a: T, b: T) -> T:
        """`ppf(1 - p)`."""
        return beta.ppf[T, num_iters](T.one() - p, a, b)

    # ------------------------------------------------------------- Student-t


struct t:
    """Student's t distribution. `scipy.stats.t`."""

    @staticmethod
    def pdf[T: FloatLike](x: T, df: T) -> T:
        return t.logpdf(x, df).exp()

    @staticmethod
    def logpdf[T: FloatLike](x: T, df: T) -> T:
        var half = (df + T.one()) / T.constant(2.0)
        return (
            lgamma(half.copy())
            - lgamma(df / T.constant(2.0))
            - (T.constant(0.5) * (_safe_ln(df) + T.constant(_LN_PI)))
            - (half * (T.one() + x * x / df).ln())
        )

    @staticmethod
    def sf[T: FloatLike](x: T, df: T) -> T:
        """`P(T > x)`, which by symmetry is `cdf(-x)` -- exact, and it
        keeps the upper tail's digits."""
        return t.cdf(-x, df)

    @staticmethod
    def logcdf[T: FloatLike](x: T, df: T) -> T:
        return _safe_ln(t.cdf(x, df))

    @staticmethod
    def logsf[T: FloatLike](x: T, df: T) -> T:
        return _safe_ln(t.sf(x, df))

    @staticmethod
    def cdf[T: FloatLike](x: T, df: T) -> T:
        """`P(T <= x)`, via the beta identity
        `P(T <= x) = 1 - 0.5*I_z(df/2, 1/2)` for `x >= 0`, with
        `z = df/(df + x^2)`.

        The two halves are combined branchlessly by sign rather than by an
        `if`, using `0.5 + 0.5*sign(x)*(1 - I_z)`: `z` depends on `x` only
        through `x^2`, so both halves share one `betainc` call and the sign is
        all that distinguishes them. At `x = 0` this gives `z = 1`,
        `I_1 = 1`, and exactly `0.5`.
        """
        var z = df / (df + x * x)
        var tail = betainc(z^, df / T.constant(2.0), T.constant(0.5))
        var sign = T.one().copysign(x)
        return T.constant(0.5) + T.constant(0.5) * sign * (T.one() - tail)

    @staticmethod
    def ppf[T: FloatLike, num_iters: Int = 12](p: T, df: T) -> T:
        """The inverse Student-t CDF, Newton from the normal quantile.

        The normal is the `df -> infinity` limit of the t, so it's a good seed
        for large `df` and a merely-adequate one for small `df`, where the t's
        heavier tails put the true quantile further out.
        """
        var x = _standard_normal_quantile(p.copy())

        for _ in range(num_iters):
            var residual = t.cdf(x.copy(), df.copy())
            var density = max_of(t.pdf(x.copy(), df.copy()), T.constant(_TINY))
            x = x - ((residual - p) / density)

        return x^

    @staticmethod
    def isf[T: FloatLike, num_iters: Int = 12](p: T, df: T) -> T:
        """`ppf(1 - p)`, which by symmetry is `-ppf(p)`."""
        return -t.ppf[T, num_iters](p, df)

    # --------------------------------------------------------------------- F


struct f:
    """The F distribution. `scipy.stats.f`."""

    @staticmethod
    def pdf[T: FloatLike](x: T, df1: T, df2: T) -> T:
        """The F density with `df1` and `df2` degrees of freedom."""
        return f.logpdf(x, df1, df2).exp()

    @staticmethod
    def logpdf[T: FloatLike](x: T, df1: T, df2: T) -> T:
        """The F log density; `_LOG_ZERO` below the support."""
        var xc = max_of(x, T.constant(0.0))
        var d1x = df1 * xc
        var interior = (
            T.constant(0.5)
            * (
                df1 * _safe_ln(d1x)
                + df2 * _safe_ln(df2)
                - ((df1 + df2) * _safe_ln(d1x + df2))
            )
            - _safe_ln(xc)
            - _log_beta(df1 / T.constant(2.0), df2 / T.constant(2.0))
        )
        return blend(
            ge_indicator(x, T.constant(0.0)), interior, T.constant(_LOG_ZERO)
        )

    @staticmethod
    def cdf[T: FloatLike](x: T, df1: T, df2: T) -> T:
        """The F CDF -- `betainc` at `z = df1*x/(df1*x + df2)`."""
        var d1x = df1 * max_of(x, T.constant(0.0))
        var z = d1x / (d1x + df2)
        return betainc(
            z^, df1 / T.constant(2.0), df2 / T.constant(2.0)
        ) * ge_indicator(x, T.constant(0.0))

    @staticmethod
    def sf[T: FloatLike](x: T, df1: T, df2: T) -> T:
        """`P(F > x)` -- `betaincc` at the same `z`, `1` below the
        support."""
        var d1x = df1 * max_of(x, T.constant(0.0))
        var z = d1x / (d1x + df2)
        return blend(
            ge_indicator(x, T.constant(0.0)),
            betaincc(z^, df1 / T.constant(2.0), df2 / T.constant(2.0)),
            T.one(),
        )

    @staticmethod
    def ppf[T: FloatLike, num_iters: Int = 20](p: T, df1: T, df2: T) -> T:
        """The inverse F CDF, through `beta.ppf`.

        `Z = df1 X / (df1 X + df2)` is `Beta(df1/2, df2/2)` when `X` is
        `F(df1, df2)` -- the same change of variables `cdf` uses, inverted:
        `x = df2 z / (df1 (1 - z))`. `beta.ppf` keeps `z` inside
        `(1e-8, 1 - 1e-8)`, so the division is always finite.
        """
        var z = beta.ppf[T, num_iters](
            p, df1 / T.constant(2.0), df2 / T.constant(2.0)
        )
        return df2 * z / (df1 * (T.one() - z))

    @staticmethod
    def isf[T: FloatLike, num_iters: Int = 20](p: T, df1: T, df2: T) -> T:
        """`ppf(1 - p)`."""
        return f.ppf[T, num_iters](T.one() - p, df1, df2)

    @staticmethod
    def logcdf[T: FloatLike](x: T, df1: T, df2: T) -> T:
        return _safe_ln(f.cdf(x, df1, df2))

    @staticmethod
    def logsf[T: FloatLike](x: T, df1: T, df2: T) -> T:
        return _safe_ln(f.sf(x, df1, df2))

    # --------------------------------------------------------------- discrete


struct poisson:
    """The Poisson distribution. `scipy.stats.poisson`."""

    @staticmethod
    def pmf[T: FloatLike](k: T, rate: T) -> T:
        """`P(X = k)` for a Poisson with mean `rate`.

        `exp(k*ln(rate) - rate - lgamma(k+1))`: `lgamma(k+1)` is `ln(k!)`
        extended to non-integer `k`, so this is the usual PMF wherever `k` is a
        whole number and its standard continuous extension elsewhere.
        """
        return poisson.logpmf(k, rate).exp()

    @staticmethod
    def logpmf[T: FloatLike](k: T, rate: T) -> T:
        """The Poisson log PMF; `_LOG_ZERO` for `k < 0`.

        The interior is evaluated at `k` clamped to the support: `lgamma`
        has a pole at `0`, so an unclamped `k = -1` would put `+inf` on the
        discarded side, and `inf * 0` is the NaN no indicator can remove.
        `pmf` used to survive this because `exp(-inf)` is `0` before the
        multiply; a log density has no such rescue.
        """
        var kc = max_of(k, T.constant(0.0))
        var interior = kc * _safe_ln(rate) - rate - lgamma(kc + T.one())
        return blend(
            ge_indicator(k, T.constant(0.0)), interior, T.constant(_LOG_ZERO)
        )

    @staticmethod
    def cdf[T: FloatLike](k: T, rate: T) -> T:
        """`P(X <= k)`, which is exactly `gammaincc(k+1, rate)`.

        Not an approximation of the sum -- the regularized upper incomplete
        gamma equals that partial sum identically for integer `k`, which is
        what makes a discrete CDF fall out of a continuous special function.
        """
        return gammaincc(
            max_of(k, T.constant(0.0)) + T.one(), rate.copy()
        ) * ge_indicator(k, T.constant(0.0))

    @staticmethod
    def sf[T: FloatLike](k: T, rate: T) -> T:
        """`P(X > k)`, which is exactly `gammainc(k+1, rate)` -- the lower
        incomplete gamma, read directly rather than as `1 - cdf`. `1` for
        `k < 0`."""
        return blend(
            ge_indicator(k, T.constant(0.0)),
            gammainc(max_of(k, T.constant(0.0)) + T.one(), rate.copy()),
            T.one(),
        )

    @staticmethod
    def ppf[T: FloatLike, max_k: Int = 64](p: T, rate: T) -> T:
        """The smallest integer `k` with `cdf(k) >= p`, as SciPy defines
        it -- by the fixed-count branchless scan the module docstring
        describes, capped at `max_k`."""
        var k = T.constant(0.0)
        for _ in range(max_k):
            var caught_up = ge_indicator(poisson.cdf(k.copy(), rate.copy()), p)
            k = k + (T.one() - caught_up)
        return k^

    @staticmethod
    def isf[T: FloatLike, max_k: Int = 64](p: T, rate: T) -> T:
        """`ppf(1 - p)`."""
        return poisson.ppf[T, max_k](T.one() - p, rate)

    @staticmethod
    def logcdf[T: FloatLike](k: T, rate: T) -> T:
        return _safe_ln(poisson.cdf(k, rate))

    @staticmethod
    def logsf[T: FloatLike](k: T, rate: T) -> T:
        return _safe_ln(poisson.sf(k, rate))


struct binom:
    """The binomial distribution. `scipy.stats.binom`."""

    @staticmethod
    def pmf[T: FloatLike](k: T, n: T, p: T) -> T:
        """`P(X = k)` for `n` trials with success probability `p`."""
        return binom.logpmf(k, n, p).exp()

    @staticmethod
    def logpmf[T: FloatLike](k: T, n: T, p: T) -> T:
        """The binomial log PMF; `_LOG_ZERO` outside `0 <= k <= n`.

        The interior is evaluated at `k` clamped into `[0, n]`, for the
        reason `poisson.logpmf` gives: `lgamma(n - k + 1)` has its pole
        exactly one step past `n`.
        """
        var kc = min_of(max_of(k, T.constant(0.0)), n.copy())
        var interior = (
            lgamma(n + T.one())
            - lgamma(kc + T.one())
            - lgamma(n - kc + T.one())
            + kc * _safe_ln(p)
            + (n - kc) * _safe_ln(T.one() - p)
        )
        var support = ge_indicator(k, T.constant(0.0)) * ge_indicator(
            n, k.copy()
        )
        return blend(support, interior, T.constant(_LOG_ZERO))

    @staticmethod
    def cdf[T: FloatLike](k: T, n: T, p: T) -> T:
        """`P(X <= k)`, which is `I_{1-p}(n-k, k+1)` -- the same
        special-function-in-disguise relationship `poisson.cdf` has with the
        incomplete gamma."""
        var kc = min_of(max_of(k, T.constant(0.0)), n.copy())
        # At `k = n` the first beta parameter is exactly `0`, where `betainc`
        # is undefined and returns NaN -- and a NaN survives being multiplied
        # by a `0` indicator, so the blend below can't clean it up afterwards.
        # Flooring the parameter keeps that lane finite; the blend then
        # discards it in favour of the exact answer, `P(X <= n) = 1`.
        var trials_left = max_of(n - kc, T.constant(1e-8))
        var upper = betainc(T.one() - p, trials_left^, kc + T.one())
        return blend(ge_indicator(k, n.copy()), T.one(), upper^)

    @staticmethod
    def sf[T: FloatLike](k: T, n: T, p: T) -> T:
        """`P(X > k)`, which is `I_p(k+1, n-k)` -- the upper tail read
        straight off `betainc`. `0` at and past `k = n`, `1` below `k = 0`.
        """
        var kc = min_of(max_of(k, T.constant(0.0)), n.copy())
        # As in `cdf`: the second parameter is exactly `0` at `k = n`, and
        # `betainc` returns NaN there, which no indicator can discard.
        var trials_left = max_of(n - kc, T.constant(1e-8))
        var upper = betainc(p.copy(), kc + T.one(), trials_left^)
        var above = blend(ge_indicator(k, n.copy()), T.constant(0.0), upper^)
        return blend(ge_indicator(k, T.constant(0.0)), above^, T.one())

    @staticmethod
    def ppf[T: FloatLike, max_k: Int = 64](p: T, n: T, prob: T) -> T:
        """The smallest integer `k` with `cdf(k) >= p`, as SciPy defines
        it -- the module docstring's fixed-count scan, capped at `max_k`.
        `cdf(n) == 1` exactly, so the scan stops at `n` on its own."""
        var k = T.constant(0.0)
        for _ in range(max_k):
            var caught_up = ge_indicator(
                binom.cdf(k.copy(), n.copy(), prob.copy()), p
            )
            k = k + (T.one() - caught_up)
        return k^

    @staticmethod
    def isf[T: FloatLike, max_k: Int = 64](p: T, n: T, prob: T) -> T:
        """`ppf(1 - p)`."""
        return binom.ppf[T, max_k](T.one() - p, n, prob)

    @staticmethod
    def logcdf[T: FloatLike](k: T, n: T, p: T) -> T:
        return _safe_ln(binom.cdf(k, n, p))

    @staticmethod
    def logsf[T: FloatLike](k: T, n: T, p: T) -> T:
        return _safe_ln(binom.sf(k, n, p))
