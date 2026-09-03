"""Probability densities, cumulative distributions, and quantiles.

Almost none of this is new numerics. A distribution's CDF is nearly always
a special function already in `numax` under a change of variables -- the
gamma family is `gammainc`, the beta family (which includes Student-t, F,
and the binomial) is `betainc`, the normal family is `erfc` -- so the work
here is composition, and the payoff is that every one of them inherits
`Dual`-differentiability, `Compensated` precision, and GPU-launchability
from the function underneath.

`d/dx` of a CDF is the PDF, and evaluating any `*_cdf` here at `Dual`
recovers exactly that, which is how `tests/stats/test_distributions.mojo` checks
each pair against the other rather than against a table.

## Conventions

- Shape parameters are `T` values, not `Int`s, so they can vary per SIMD
  lane. That includes the discrete distributions' counts: `poisson_cdf`'s
  `k` and `binomial_cdf`'s `n`/`k` are continuous extensions of the usual
  integer-argument definitions, which is what the underlying `gammaincc`
  and `betainc` compute anyway.
- Densities are `0` outside their support rather than undefined, applied by
  a branchless indicator. The log-space interior is always evaluated on a
  clamped argument first, so the discarded side never produces the NaN that
  would survive multiplication by a `0` indicator.
- Everything is scoped to valid parameters (positive shapes, `sigma > 0`,
  `0 < p < 1`); nothing validates, in keeping with the rest of `numax`.

## Quantiles

Each quantile is a seed plus a fixed number of Newton steps against its own
CDF, using the analytic PDF as the derivative. They do *not* go through
`numax.optimize.newton`, and the reason is a real limitation worth naming:
`solve`'s `f` is `def[U: FloatLike](U) thin -> U`, a non-capturing function
required to be `thin` so a solver can be launched on GPU. A distribution's
parameters have nowhere to live in that signature -- `beta_quantile(p, a,
b)` would need `a` and `b` inside `f`, which a `thin` function cannot
close over. Writing the loop locally also happens to be cheaper here, since
the exact derivative is already in hand and doesn't need a `Dual` pass.

Seeds matter more than iteration counts for a fixed-work solver, so each
one uses the standard published approximation for its family rather than a
constant: Abramowitz & Stegun 26.2.23 for the normal, Wilson-Hilferty for
the gamma, the distribution mean for the beta.
"""

from ..special.beta import betainc
from ..special.gamma import gammainc, gammaincc, lgamma
from ..core.numeric import FloatLike, blend, ge_indicator, max_of, min_of

comptime _SQRT_2 = 1.4142135623730951
comptime _SQRT_2PI = 2.5066282746310002
comptime _LN_2PI = 1.8378770664093453
comptime _LN_PI = 1.1447298858494002

# Small enough to be indistinguishable from the boundary at any supported
# `dtype`, large enough that `ln` of it is finite in float32.
comptime _TINY = 1e-30


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
    return lgamma(a.copy()) + lgamma(b.copy()) + (-lgamma(a + b))


# ---------------------------------------------------------------- normal


def normal_pdf[T: FloatLike](x: T, mu: T, sigma: T) -> T:
    """The normal density with mean `mu` and standard deviation `sigma`."""
    var z = (x + (-mu)) / sigma
    return (-(z * z) / T.constant(2.0)).exp() / (sigma * T.constant(_SQRT_2PI))


def normal_cdf[T: FloatLike](x: T, mu: T, sigma: T) -> T:
    """The normal CDF, as `0.5*erfc(-z/sqrt(2))`.

    Written with `erfc` rather than the equivalent `0.5*(1 + erf(z/sqrt2))`
    on purpose: for `z` around `-6` the `erf` form is `0.5*(1 - 0.999...)`,
    which cancels away most of its significant digits, while `erfc` returns
    the small tail probability directly. `Plain.erfc` delegates to
    `std.math`, so that accuracy is real rather than nominal.
    """
    var z = (x + (-mu)) / sigma
    return T.constant(0.5) * (-(z / T.constant(_SQRT_2))).erfc()


def normal_quantile[T: FloatLike](p: T, mu: T, sigma: T) -> T:
    """The inverse normal CDF, for `0 < p < 1`."""
    return mu + sigma * _standard_normal_quantile(p)


def _standard_normal_quantile[T: FloatLike, num_iters: Int = 3](p: T) -> T:
    """The standard normal quantile: an Abramowitz & Stegun 26.2.23 seed
    (max error ~4.5e-4) refined by Newton against `normal_cdf`.

    The seed is defined for the lower half only, so the upper half is
    handled by the symmetry `z(p) = -z(1-p)`, applied branchlessly: `q` is
    the smaller of `p` and `1-p` either way, and the sign comes from an
    indicator on `p >= 0.5`.
    """
    var q = min_of(p, T.one() + (-p))
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
    var lower = -(t + (-(numerator / denominator)))
    var z = blend(ge_indicator(p, T.constant(0.5)), -lower, lower.copy())

    var zero = T.constant(0.0)
    var one = T.one()
    for _ in range(num_iters):
        var residual = normal_cdf(z.copy(), zero.copy(), one.copy())
        z = z + (
            -(
                (residual + (-p))
                / max_of(
                    normal_pdf(z.copy(), zero.copy(), one.copy()),
                    T.constant(_TINY),
                )
            )
        )

    return z^


# ----------------------------------------------------------- exponential


def exponential_pdf[T: FloatLike](x: T, rate: T) -> T:
    """`rate*exp(-rate*x)` on `x >= 0`, and `0` below it."""
    var inside = rate * (-(rate * max_of(x, T.constant(0.0)))).exp()
    return inside * ge_indicator(x, T.constant(0.0))


def exponential_cdf[T: FloatLike](x: T, rate: T) -> T:
    var inside = T.one() + (-(-(rate * max_of(x, T.constant(0.0)))).exp())
    return inside * ge_indicator(x, T.constant(0.0))


# ----------------------------------------------------------------- gamma


def gamma_pdf[T: FloatLike](x: T, shape: T, scale: T) -> T:
    """The gamma density in the shape/scale parameterization.

    Computed in log space, which is what keeps it usable for large `shape`:
    the direct form needs `x^(shape-1)` and `Gamma(shape)` separately, and
    both overflow long before their ratio does.
    """
    var z = max_of(x, T.constant(0.0)) / scale
    var log_density = (
        (shape + (-T.one())) * _safe_ln(z)
        + (-z)
        + (-lgamma(shape.copy()))
        + (-_safe_ln(scale))
    )
    return log_density.exp() * ge_indicator(x, T.constant(0.0))


def gamma_cdf[T: FloatLike](x: T, shape: T, scale: T) -> T:
    """The gamma CDF -- `gammainc` under the substitution `x/scale`."""
    var z = max_of(x, T.constant(0.0)) / scale
    return gammainc(shape.copy(), z^) * ge_indicator(x, T.constant(0.0))


def gamma_quantile[
    T: FloatLike, num_iters: Int = 12
](p: T, shape: T, scale: T) -> T:
    """The inverse gamma CDF, from a Wilson-Hilferty seed.

    Wilson-Hilferty approximates the gamma as a cube-rooted normal, which
    is accurate for `shape` above about 1 and increasingly rough below it.
    The Newton refinement covers the difference; the seed is floored away
    from zero because the cube can go negative for small `shape` and large
    lower-tail `p`, and `ln` of a negative seed would poison the lane.
    """
    var z = _standard_normal_quantile(p.copy())
    var a9 = T.constant(9.0) * shape
    var base = T.one() + (-(T.one() / a9)) + z / a9.sqrt()
    var seed = max_of(shape * base * base * base, T.constant(1e-6))

    var x = seed * scale
    for _ in range(num_iters):
        var residual = gamma_cdf(x.copy(), shape.copy(), scale.copy())
        var density = max_of(
            gamma_pdf(x.copy(), shape.copy(), scale.copy()),
            T.constant(_TINY),
        )
        x = max_of(x + (-((residual + (-p)) / density)), T.constant(_TINY))

    return x^


# ------------------------------------------------------------ chi-square


def chi2_pdf[T: FloatLike](x: T, df: T) -> T:
    """Chi-square with `df` degrees of freedom -- gamma with
    `shape = df/2`, `scale = 2`."""
    return gamma_pdf(x, df / T.constant(2.0), T.constant(2.0))


def chi2_cdf[T: FloatLike](x: T, df: T) -> T:
    return gamma_cdf(x, df / T.constant(2.0), T.constant(2.0))


def chi2_quantile[T: FloatLike](p: T, df: T) -> T:
    return gamma_quantile(p, df / T.constant(2.0), T.constant(2.0))


# ------------------------------------------------------------------ beta


def beta_pdf[T: FloatLike](x: T, a: T, b: T) -> T:
    """The beta density on `0 < x < 1`, and `0` outside it."""
    var xc = min_of(max_of(x, T.constant(0.0)), T.one())
    var log_density = (
        (a + (-T.one())) * _safe_ln(xc)
        + (b + (-T.one())) * _safe_ln(T.one() + (-xc))
        + (-_log_beta(a, b))
    )
    var support = ge_indicator(x, T.constant(0.0)) * ge_indicator(
        T.one(), x.copy()
    )
    return log_density.exp() * support


def beta_cdf[T: FloatLike](x: T, a: T, b: T) -> T:
    """The beta CDF -- `betainc` directly, clamped to its support."""
    var xc = min_of(max_of(x, T.constant(0.0)), T.one())
    return betainc(xc^, a, b)


def beta_quantile[T: FloatLike, num_iters: Int = 20](p: T, a: T, b: T) -> T:
    """The inverse beta CDF, Newton from the distribution's mean.

    Each step is clamped back inside `(0, 1)`: Newton on a CDF that
    saturates near both endpoints readily proposes a point outside the
    support, and a clamped step is a bounded loss where an escaped one is
    unrecoverable.
    """
    var lo = T.constant(1e-8)
    var hi = T.one() + (-lo)
    var x = min_of(max_of(a / (a + b), lo.copy()), hi.copy())

    for _ in range(num_iters):
        var residual = beta_cdf(x.copy(), a.copy(), b.copy())
        var density = max_of(
            beta_pdf(x.copy(), a.copy(), b.copy()), T.constant(_TINY)
        )
        var proposal = x + (-((residual + (-p)) / density))
        x = min_of(max_of(proposal^, lo.copy()), hi.copy())

    return x^


# ------------------------------------------------------------- Student-t


def student_t_pdf[T: FloatLike](x: T, df: T) -> T:
    var half = (df + T.one()) / T.constant(2.0)
    var log_density = (
        lgamma(half.copy())
        + (-lgamma(df / T.constant(2.0)))
        + (-T.constant(0.5) * (_safe_ln(df) + T.constant(_LN_PI)))
        + (-half * (T.one() + x * x / df).ln())
    )
    return log_density.exp()


def student_t_cdf[T: FloatLike](x: T, df: T) -> T:
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
    return T.constant(0.5) + T.constant(0.5) * sign * (T.one() + (-tail))


def student_t_quantile[T: FloatLike, num_iters: Int = 12](p: T, df: T) -> T:
    """The inverse Student-t CDF, Newton from the normal quantile.

    The normal is the `df -> infinity` limit of the t, so it's a good seed
    for large `df` and a merely-adequate one for small `df`, where the t's
    heavier tails put the true quantile further out.
    """
    var x = _standard_normal_quantile(p.copy())

    for _ in range(num_iters):
        var residual = student_t_cdf(x.copy(), df.copy())
        var density = max_of(
            student_t_pdf(x.copy(), df.copy()), T.constant(_TINY)
        )
        x = x + (-((residual + (-p)) / density))

    return x^


# --------------------------------------------------------------------- F


def f_pdf[T: FloatLike](x: T, df1: T, df2: T) -> T:
    """The F density with `df1` and `df2` degrees of freedom."""
    var xc = max_of(x, T.constant(0.0))
    var d1x = df1 * xc
    var log_density = (
        T.constant(0.5)
        * (
            df1 * _safe_ln(d1x)
            + df2 * _safe_ln(df2)
            + (-((df1 + df2) * _safe_ln(d1x + df2)))
        )
        + (-_safe_ln(xc))
        + (-_log_beta(df1 / T.constant(2.0), df2 / T.constant(2.0)))
    )
    return log_density.exp() * ge_indicator(x, T.constant(0.0))


def f_cdf[T: FloatLike](x: T, df1: T, df2: T) -> T:
    """The F CDF -- `betainc` at `z = df1*x/(df1*x + df2)`."""
    var d1x = df1 * max_of(x, T.constant(0.0))
    var z = d1x / (d1x + df2)
    return betainc(
        z^, df1 / T.constant(2.0), df2 / T.constant(2.0)
    ) * ge_indicator(x, T.constant(0.0))


# --------------------------------------------------------------- discrete


def poisson_pmf[T: FloatLike](k: T, rate: T) -> T:
    """`P(X = k)` for a Poisson with mean `rate`.

    `exp(k*ln(rate) - rate - lgamma(k+1))`: `lgamma(k+1)` is `ln(k!)`
    extended to non-integer `k`, so this is the usual PMF wherever `k` is a
    whole number and its standard continuous extension elsewhere.
    """
    var log_pmf = k * _safe_ln(rate) + (-rate) + (-lgamma(k + T.one()))
    return log_pmf.exp() * ge_indicator(k, T.constant(0.0))


def poisson_cdf[T: FloatLike](k: T, rate: T) -> T:
    """`P(X <= k)`, which is exactly `gammaincc(k+1, rate)`.

    Not an approximation of the sum -- the regularized upper incomplete
    gamma equals that partial sum identically for integer `k`, which is
    what makes a discrete CDF fall out of a continuous special function.
    """
    return gammaincc(
        max_of(k, T.constant(0.0)) + T.one(), rate.copy()
    ) * ge_indicator(k, T.constant(0.0))


def binomial_pmf[T: FloatLike](k: T, n: T, p: T) -> T:
    """`P(X = k)` for `n` trials with success probability `p`."""
    var log_pmf = (
        lgamma(n + T.one())
        + (-lgamma(k + T.one()))
        + (-lgamma(n + (-k) + T.one()))
        + k * _safe_ln(p)
        + (n + (-k)) * _safe_ln(T.one() + (-p))
    )
    var support = ge_indicator(k, T.constant(0.0)) * ge_indicator(n, k.copy())
    return log_pmf.exp() * support


def binomial_cdf[T: FloatLike](k: T, n: T, p: T) -> T:
    """`P(X <= k)`, which is `I_{1-p}(n-k, k+1)` -- the same
    special-function-in-disguise relationship `poisson_cdf` has with the
    incomplete gamma."""
    var kc = min_of(max_of(k, T.constant(0.0)), n.copy())
    # At `k = n` the first beta parameter is exactly `0`, where `betainc`
    # is undefined and returns NaN -- and a NaN survives being multiplied
    # by a `0` indicator, so the blend below can't clean it up afterwards.
    # Flooring the parameter keeps that lane finite; the blend then
    # discards it in favour of the exact answer, `P(X <= n) = 1`.
    var trials_left = max_of(n + (-kc), T.constant(1e-8))
    var upper = betainc(T.one() + (-p), trials_left^, kc + T.one())
    return blend(ge_indicator(k, n.copy()), T.one(), upper^)
