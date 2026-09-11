"""Hypothesis tests over `numax.core.array.Tensor`: the three `t` tests,
`chisquare`, `ks_1samp`, `f_oneway` and `mannwhitneyu`, each returning
SciPy's statistic and p-value.

**Tier 2, host-side**, in `Float64`: a test statistic is a few sums (or,
for the rank tests, a sort) and its p-value is one tail of a distribution
`numax.stats` already has -- `t.sf`, `chi2.sf`, `f.sf`, `norm.sf` -- so
each test is that arithmetic and that call, the shape
`numax.stats.correlation` gives its reason for.

## SciPy's conventions, and the one place they are not met

Every test takes SciPy's `alternative` (`"two-sided"`, `"less"`,
`"greater"`) and forms its p-value the way SciPy does: the two-sided `t`
and `F` tails doubled, `mannwhitneyu` asymptotic with its tie correction
and continuity correction on `max(U1, U2)`, `chisquare` on `k - 1 - ddof`
degrees of freedom. `ks_1samp`'s one-sided p-values are the exact
Birnbaum-Tingey sum SciPy's are; its **two-sided p-value is the asymptotic
Kolmogorov distribution** (`method="asymp"` in SciPy), where SciPy's
default computes the exact two-sided distribution by Marsaglia-Tsang-Wang.
That is a divergence recorded in `docs/parity.md`: the exact two-sided
law is a substantial algorithm of its own, and at the sample sizes a
device tensor holds the two agree; at `n = 16` they differ in the second
digit. `mannwhitneyu` likewise is always asymptotic, where SciPy
enumerates exactly for tiny untied samples.

## The MAX gate

Nothing: MAX has no statistical tests. **Extend.**
"""

from std.builtin.sort import sort as _sort
from std.math import exp as _exp, sqrt as _sqrt

from layout.tile_layout import TensorLayout

from ..core.array import Static, Tensor
from ..core.plain import Plain
from .distributions import chi2, f, norm, t

comptime _P = Plain[DType.float64]


@fieldwise_init
struct TestResult(Copyable, Movable):
    """What every test here returns: SciPy's `statistic` and `pvalue`, and
    the degrees of freedom `df` where the test has one (`0` where it does
    not)."""

    var statistic: Float64
    var pvalue: Float64
    var df: Float64


def _values[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> List[Float64]:
    var host = xs.to_host()
    var out = List[Float64](capacity=len(host))
    for i in range(len(host)):
        out.append(Float64(host[i]))
    return out^


def _mean(values: List[Float64]) -> Float64:
    var total = 0.0
    for i in range(len(values)):
        total += values[i]
    return total / Float64(len(values))


def _variance(values: List[Float64], ddof: Int) -> Float64:
    var centre = _mean(values)
    var total = 0.0
    for i in range(len(values)):
        total += (values[i] - centre) * (values[i] - centre)
    return total / Float64(len(values) - ddof)


def _check_alternative(alternative: StaticString) raises:
    if not (
        alternative == "two-sided"
        or alternative == "less"
        or alternative == "greater"
    ):
        raise Error(
            "alternative must be 'two-sided', 'less' or 'greater', not '",
            alternative,
            "'",
        )


def _t_pvalue(
    statistic: Float64, df: Float64, alternative: StaticString
) -> Float64:
    """SciPy's `_get_pvalue` for a `t` statistic."""
    if alternative == "less":
        return Float64(t.cdf[_P](_P(statistic), _P(df)).v)
    if alternative == "greater":
        return Float64(t.sf[_P](_P(statistic), _P(df)).v)
    return 2.0 * Float64(t.sf[_P](_P(abs(statistic)), _P(df)).v)


def ttest_1samp[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType],
    popmean: Float64,
    alternative: StaticString = "two-sided",
) raises -> TestResult where dtype.is_floating_point():
    """The one-sample `t` test that the mean of `xs` is `popmean`.
    `scipy.stats.ttest_1samp(a, popmean, alternative)`: `t = (mean -
    popmean) / (std_1 / sqrt(n))` on `n - 1` degrees of freedom."""
    _check_alternative(alternative)
    var values = _values(xs)
    var n = len(values)
    if n < 2:
        raise Error("ttest_1samp: at least two observations are needed")
    var df = Float64(n - 1)
    var statistic = (_mean(values) - popmean) / _sqrt(
        _variance(values, 1) / Float64(n)
    )
    return TestResult(statistic, _t_pvalue(statistic, df, alternative), df)


def ttest_ind[
    dtype: DType, XLayout: TensorLayout, YLayout: TensorLayout
](
    xs: Tensor[dtype, XLayout],
    ys: Tensor[dtype, YLayout],
    equal_var: Bool = True,
    alternative: StaticString = "two-sided",
) raises -> TestResult where dtype.is_floating_point():
    """The two-sample `t` test that two independent samples share a mean.
    `scipy.stats.ttest_ind(a, b, equal_var, alternative)`: Student's
    pooled-variance test by default, Welch's unequal-variance test with
    its Welch-Satterthwaite degrees of freedom when `equal_var=False`."""
    _check_alternative(alternative)
    var a = _values(xs)
    var b = _values(ys)
    var n1 = Float64(len(a))
    var n2 = Float64(len(b))
    if len(a) < 2 or len(b) < 2:
        raise Error("ttest_ind: each sample needs at least two observations")
    var v1 = _variance(a, 1)
    var v2 = _variance(b, 1)
    var df: Float64
    var denominator: Float64
    if equal_var:
        df = n1 + n2 - 2.0
        var pooled = ((n1 - 1.0) * v1 + (n2 - 1.0) * v2) / df
        denominator = _sqrt(pooled * (1.0 / n1 + 1.0 / n2))
    else:
        var vn1 = v1 / n1
        var vn2 = v2 / n2
        df = (
            (vn1 + vn2)
            * (vn1 + vn2)
            / (vn1 * vn1 / (n1 - 1.0) + vn2 * vn2 / (n2 - 1.0))
        )
        denominator = _sqrt(vn1 + vn2)
    var statistic = (_mean(a) - _mean(b)) / denominator
    return TestResult(statistic, _t_pvalue(statistic, df, alternative), df)


def ttest_rel[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType],
    ys: Tensor[dtype, LayoutType],
    alternative: StaticString = "two-sided",
) raises -> TestResult where dtype.is_floating_point():
    """The paired `t` test: `ttest_1samp` of the differences against zero.
    `scipy.stats.ttest_rel(a, b, alternative)`."""
    _check_alternative(alternative)
    var a = _values(xs)
    var b = _values(ys)
    var n = len(a)
    if n < 2:
        raise Error("ttest_rel: at least two pairs are needed")
    var differences = List[Float64](capacity=n)
    for i in range(n):
        differences.append(a[i] - b[i])
    var df = Float64(n - 1)
    var statistic = _mean(differences) / _sqrt(
        _variance(differences, 1) / Float64(n)
    )
    return TestResult(statistic, _t_pvalue(statistic, df, alternative), df)


def chisquare[
    dtype: DType, LayoutType: TensorLayout
](
    observed: Tensor[dtype, LayoutType], ddof: Int = 0
) raises -> TestResult where dtype.is_floating_point():
    """Pearson's chi-squared test that the observed counts are uniform:
    `sum((o - e)^2 / e)` against `chi2` on `k - 1 - ddof` degrees of
    freedom, `e` the mean count. `scipy.stats.chisquare(f_obs, ddof)`."""
    var o = _values(observed)
    var k = len(o)
    var expected = _mean(o)
    var statistic = 0.0
    for i in range(k):
        statistic += (o[i] - expected) * (o[i] - expected) / expected
    var df = Float64(k - 1 - ddof)
    return TestResult(
        statistic, Float64(chi2.sf[_P](_P(statistic), _P(df)).v), df
    )


def chisquare[
    dtype: DType, LayoutType: TensorLayout
](
    observed: Tensor[dtype, LayoutType],
    expected: Tensor[dtype, LayoutType],
    ddof: Int = 0,
) raises -> TestResult where dtype.is_floating_point():
    """Pearson's chi-squared test against the given expected counts.
    `scipy.stats.chisquare(f_obs, f_exp, ddof)`. SciPy checks that the two
    sum to the same total to a relative `1e-8`; so does this, raising."""
    var o = _values(observed)
    var e = _values(expected)
    var k = len(o)
    var so = 0.0
    var se = 0.0
    for i in range(k):
        so += o[i]
        se += e[i]
    if abs(so - se) > 1e-8 * max(abs(so), abs(se)):
        raise Error(
            "chisquare: observed and expected counts must sum to the same total"
        )
    var statistic = 0.0
    for i in range(k):
        statistic += (o[i] - e[i]) * (o[i] - e[i]) / e[i]
    var df = Float64(k - 1 - ddof)
    return TestResult(
        statistic, Float64(chi2.sf[_P](_P(statistic), _P(df)).v), df
    )


def _binomial(n: Int, k: Int) -> Float64:
    var result = 1.0
    for i in range(1, k + 1):
        result *= Float64(n - k + i) / Float64(i)
    return result


def _ks_one_sided_exact(n: Int, d: Float64) -> Float64:
    """`P(D+ >= d)` for `n` samples: the Birnbaum-Tingey sum SciPy's
    `ksone.sf` evaluates, `d sum_j C(n, j) (1 - d - j/n)^(n-j) (d +
    j/n)^(j-1)` over `j <= n(1 - d)`."""
    if d <= 0:
        return 1.0
    if d >= 1:
        return 0.0
    var total = 0.0
    var limit = Int(Float64(n) * (1.0 - d))
    for j in range(limit + 1):
        var a = 1.0 - d - Float64(j) / Float64(n)
        var b = d + Float64(j) / Float64(n)
        var term = _binomial(n, j)
        for _ in range(n - j):
            term *= a
        if j >= 1:
            for _ in range(j - 1):
                term *= b
        else:
            term /= b
        total += term
    return min(1.0, max(0.0, d * total))


def _kolmogorov_sf(x: Float64) -> Float64:
    """The limiting two-sided Kolmogorov tail, `2 sum (-1)^(k-1) exp(-2 k^2
    x^2)` -- SciPy's `kstwobign.sf`."""
    if x <= 0:
        return 1.0
    var total = 0.0
    var sign = 1.0
    for k in range(1, 200):
        var term = _exp(-2.0 * Float64(k * k) * x * x)
        total += sign * term
        sign = -sign
        if term < 1e-17:
            break
    return min(1.0, max(0.0, 2.0 * total))


def ks_1samp[
    dtype: DType,
    LayoutType: TensorLayout,
    cdf: def(Float64) thin -> Float64,
](
    xs: Tensor[dtype, LayoutType], alternative: StaticString = "two-sided"
) raises -> TestResult where dtype.is_floating_point():
    """The one-sample Kolmogorov-Smirnov test of `xs` against the
    continuous distribution with the given `cdf`.
    `scipy.stats.ks_1samp(x, cdf, alternative)`.

    The statistic is the largest gap between the empirical and the
    hypothesized distribution -- `D+`, `D-` or their maximum by
    `alternative`. The one-sided p-values are the exact Birnbaum-Tingey
    sum; the two-sided one is the asymptotic Kolmogorov distribution of
    `sqrt(n) D`, SciPy's `method="asymp"` -- the module docstring records
    why, and `docs/parity.md` the divergence. `cdf` is a compile-time
    function parameter, so a `numax.stats` distribution's `cdf` is passed
    through a one-line wrapper that fixes its parameters.
    """
    _check_alternative(alternative)
    var values = _values(xs)
    var n = len(values)
    if n < 1:
        raise Error("ks_1samp: no samples")
    _sort(values)
    var d_plus = 0.0
    var d_minus = 0.0
    for i in range(n):
        var c = cdf(values[i])
        d_plus = max(d_plus, Float64(i + 1) / Float64(n) - c)
        d_minus = max(d_minus, c - Float64(i) / Float64(n))
    if alternative == "greater":
        return TestResult(d_plus, _ks_one_sided_exact(n, d_plus), Float64(n))
    if alternative == "less":
        return TestResult(d_minus, _ks_one_sided_exact(n, d_minus), Float64(n))
    var d = max(d_plus, d_minus)
    return TestResult(d, _kolmogorov_sf(_sqrt(Float64(n)) * d), Float64(n))


def f_oneway[
    dtype: DType, LayoutType: TensorLayout
](
    *groups: Tensor[dtype, LayoutType]
) raises -> TestResult where dtype.is_floating_point():
    """The one-way ANOVA `F` test that every group shares a mean:
    between-group over within-group mean square, against `f` on `k - 1`
    and `N - k` degrees of freedom. `scipy.stats.f_oneway(*samples)`, the
    groups passed as separate arguments of one shape. `df` in the result
    is the numerator's; the denominator's is `N - k`."""
    var k = len(groups)
    if k < 2:
        raise Error("f_oneway: at least two groups are needed")
    var means = List[Float64](capacity=k)
    var sizes = List[Int](capacity=k)
    var grand = 0.0
    var total_n = 0
    var within = 0.0
    for g in range(k):
        var values = _values(groups[g])
        var m = _mean(values)
        means.append(m)
        sizes.append(len(values))
        for i in range(len(values)):
            grand += values[i]
            within += (values[i] - m) * (values[i] - m)
        total_n += len(values)
    grand /= Float64(total_n)
    var between = 0.0
    for g in range(k):
        between += Float64(sizes[g]) * (means[g] - grand) * (means[g] - grand)
    var dfb = Float64(k - 1)
    var dfw = Float64(total_n - k)
    var statistic = (between / dfb) / (within / dfw)
    return TestResult(
        statistic, Float64(f.sf[_P](_P(statistic), _P(dfb), _P(dfw)).v), dfb
    )


def mannwhitneyu[
    dtype: DType, XLayout: TensorLayout, YLayout: TensorLayout
](
    xs: Tensor[dtype, XLayout],
    ys: Tensor[dtype, YLayout],
    alternative: StaticString = "two-sided",
    use_continuity: Bool = True,
) raises -> TestResult where dtype.is_floating_point():
    """The Mann-Whitney `U` test that two independent samples come from
    one distribution. `scipy.stats.mannwhitneyu(x, y, use_continuity,
    alternative, method="asymptotic")`.

    The statistic is `U1`, the count of `(x, y)` pairs with `x > y` (ties
    half), as SciPy reports it; the p-value is the normal approximation
    with SciPy's tie correction and, by default, its continuity correction,
    taken on `U1`, `U2` or their maximum by `alternative`. Always
    asymptotic, where SciPy enumerates exactly for a small untied sample.
    """
    _check_alternative(alternative)
    var a = _values(xs)
    var b = _values(ys)
    var n1 = len(a)
    var n2 = len(b)
    var n = n1 + n2
    # Average ranks of the pooled sample, and the tie groups for the
    # variance correction.
    var pooled = List[Float64](capacity=n)
    for i in range(n1):
        pooled.append(a[i])
    for i in range(n2):
        pooled.append(b[i])
    var order = List[Int](capacity=n)
    for i in range(n):
        order.append(i)

    @parameter
    def by_value(i: Int, j: Int) -> Bool:
        return pooled[i] < pooled[j] or (pooled[i] == pooled[j] and i < j)

    _sort[by_value](order)
    var ranks = List[Float64](length=n, fill=0.0)
    var tie_term = 0.0
    var i = 0
    while i < n:
        var j = i
        while j + 1 < n and pooled[order[j + 1]] == pooled[order[i]]:
            j += 1
        var size = Float64(j - i + 1)
        tie_term += size * size * size - size
        for k in range(i, j + 1):
            ranks[order[k]] = Float64(i + j + 2) / 2.0
        i = j + 1
    var r1 = 0.0
    for k in range(n1):
        r1 += ranks[k]
    var u1 = r1 - Float64(n1) * Float64(n1 + 1) / 2.0
    var u2 = Float64(n1) * Float64(n2) - u1

    var u: Float64
    var factor: Float64
    if alternative == "greater":
        u = u1
        factor = 1.0
    elif alternative == "less":
        u = u2
        factor = 1.0
    else:
        u = max(u1, u2)
        factor = 2.0
    var mu = Float64(n1) * Float64(n2) / 2.0
    var nf = Float64(n)
    var sigma = _sqrt(
        Float64(n1)
        * Float64(n2)
        / 12.0
        * ((nf + 1.0) - tie_term / (nf * (nf - 1.0)))
    )
    var numerator = u - mu - (0.5 if use_continuity else 0.0)
    var z = numerator / sigma
    var p = factor * Float64(norm.sf[_P](_P(z), _P(0.0), _P(1.0)).v)
    return TestResult(u1, min(1.0, max(0.0, p)), 0.0)
