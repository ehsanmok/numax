"""Tests for `numax.stats.descriptive` and `numax.stats.hypothesis`, each
against `scipy.stats`' own values on 16-sample series: the shape
statistics with and without bias correction, the means and entropy,
`describe`, and every hypothesis test's statistic, p-value and degrees of
freedom under its alternatives.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax.core.array import Static
from numax.core.plain import Plain
from numax.stats import (
    chisquare,
    describe,
    entropy,
    f_oneway,
    gmean,
    hmean,
    kurtosis,
    ks_1samp,
    mannwhitneyu,
    norm,
    sem,
    skew,
    trim_mean,
    ttest_1samp,
    ttest_ind,
    ttest_rel,
)

comptime dtype = DType.float64
comptime P = Plain[DType.float64]


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _from[n: Int](values: List[Float64]) raises -> Static[dtype, n]:
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype](values[i]))
    return Static[dtype, n](_cpu(), out^)


def _x() -> List[Float64]:
    return [
        1.0,
        0.0,
        -1.0,
        2.0,
        0.5,
        3.0,
        -2.0,
        1.0,
        0.75,
        -0.5,
        1.5,
        2.5,
        -1.0,
        0.0,
        1.0,
        -3.0,
    ]


def _y() -> List[Float64]:
    return [
        0.5,
        0.2,
        -0.8,
        2.5,
        0.0,
        2.0,
        -1.5,
        1.2,
        0.5,
        -1.0,
        1.0,
        3.0,
        -0.5,
        0.5,
        0.8,
        -2.0,
    ]


def _z() -> List[Float64]:
    return [
        2.0,
        1.5,
        0.5,
        3.0,
        1.0,
        4.0,
        -1.0,
        2.0,
        1.75,
        0.5,
        2.5,
        3.5,
        0.0,
        1.0,
        2.0,
        -2.0,
    ]


def _positive() -> List[Float64]:
    """`|x| + 0.5`, for the means that need positive input."""
    var out = List[Float64]()
    for v in _x():
        out.append(abs(v) + 0.5)
    return out^


def test_skew_and_kurtosis_match_scipy() raises:
    var x = _from[16](_x())
    assert_almost_equal(skew(x), -0.36549053241418256, atol=1e-13)
    assert_almost_equal(skew(x, bias=False), -0.4044396414961746, atol=1e-13)
    assert_almost_equal(kurtosis(x), -0.35654852283529737, atol=1e-13)
    assert_almost_equal(
        kurtosis(x, fisher=False), 2.6434514771647026, atol=1e-13
    )
    assert_almost_equal(
        kurtosis(x, bias=False), -0.005054249027477198, atol=1e-13
    )


def test_sem_gmean_hmean_trim_mean_and_entropy_match_scipy() raises:
    var x = _from[16](_x())
    assert_almost_equal(sem(x), 0.40050017556175926, atol=1e-13)
    assert_almost_equal(sem(x, ddof=0), 0.3877826275259601, atol=1e-13)
    var pos = _from[16](_positive())
    assert_almost_equal(gmean(pos), 1.5482536216828005, atol=1e-13)
    assert_almost_equal(hmean(pos), 1.2967966036279428, atol=1e-13)
    var weights = List[Float64]()
    for i in range(1, 17):
        weights.append(Float64(i))
    assert_almost_equal(
        gmean(pos, _from[16](weights)), 1.6196966942087356, atol=1e-13
    )
    assert_almost_equal(trim_mean(x, 0.1), 0.4107142857142857, atol=1e-13)
    assert_almost_equal(trim_mean(x, 0.25), 0.46875, atol=1e-13)
    assert_almost_equal(trim_mean(x, 0.0), 0.359375, atol=1e-13)
    var pk = _from[4]([0.1, 0.2, 0.3, 0.4])
    var qk = _from[4]([0.25, 0.25, 0.25, 0.25])
    assert_almost_equal(entropy(pk), 1.2798542258336676, atol=1e-13)
    assert_almost_equal(entropy(pk, qk), 0.10644013528622318, atol=1e-13)
    assert_almost_equal(entropy(pk, base=2.0), 1.8464393446710157, atol=1e-13)
    var unnormalized = _from[4]([1.0, 2.0, 3.0, 4.0])
    assert_almost_equal(entropy(unnormalized), 1.2798542258336676, atol=1e-13)


def test_describe_matches_scipy() raises:
    var x = _from[16](_x())
    var d = describe(x)
    assert_equal(d.nobs, 16)
    assert_almost_equal(d.min, -3.0)
    assert_almost_equal(d.max, 3.0)
    assert_almost_equal(d.mean, 0.359375, atol=1e-15)
    assert_almost_equal(d.variance, 2.56640625, atol=1e-13)
    assert_almost_equal(d.skewness, -0.36549053241418256, atol=1e-13)
    assert_almost_equal(d.kurtosis, -0.35654852283529737, atol=1e-13)
    var corrected = describe(x, ddof=0, bias=False)
    assert_almost_equal(corrected.variance, 2.406005859375, atol=1e-13)
    assert_almost_equal(corrected.skewness, -0.4044396414961746, atol=1e-13)
    assert_almost_equal(corrected.kurtosis, -0.005054249027477198, atol=1e-13)


def test_t_tests_match_scipy() raises:
    """One-sample (two-sided and `"greater"`), independent (Student and
    Welch), and paired, with statistics, p-values and degrees of
    freedom."""
    var x = _from[16](_x())
    var y = _from[16](_y())
    var one = ttest_1samp(x, 0.0)
    assert_almost_equal(one.statistic, 0.8973154618370011, atol=1e-13)
    assert_almost_equal(one.pvalue, 0.3837269187663, atol=1e-12)
    assert_almost_equal(one.df, 15.0)
    var greater = ttest_1samp(x, 0.5, "greater")
    assert_almost_equal(greater.statistic, -0.3511234415883917, atol=1e-13)
    assert_almost_equal(greater.pvalue, 0.6348106186275799, atol=1e-12)
    var student = ttest_ind(x, y)
    assert_almost_equal(student.statistic, -0.0767762650654446, atol=1e-13)
    assert_almost_equal(student.pvalue, 0.9393112896009698, atol=1e-12)
    assert_almost_equal(student.df, 30.0)
    var welch = ttest_ind(x, y, equal_var=False)
    assert_almost_equal(welch.statistic, -0.07677626506544459, atol=1e-13)
    assert_almost_equal(welch.pvalue, 0.9393220730980776, atol=1e-12)
    assert_almost_equal(welch.df, 29.375682123139853, atol=1e-11)
    var paired = ttest_rel(x, y)
    assert_almost_equal(paired.statistic, -0.2984761862150484, atol=1e-13)
    assert_almost_equal(paired.pvalue, 0.7694346800628272, atol=1e-12)
    assert_almost_equal(paired.df, 15.0)
    var raised = False
    try:
        _ = ttest_1samp(x, 0.0, "sideways")
    except:
        raised = True
    assert_true(raised)


def test_chisquare_matches_scipy() raises:
    var observed = _from[6]([16.0, 18.0, 16.0, 14.0, 12.0, 12.0])
    var uniform = chisquare(observed)
    assert_almost_equal(uniform.statistic, 2.0, atol=1e-13)
    assert_almost_equal(uniform.pvalue, 0.8491450360846096, atol=1e-12)
    assert_almost_equal(uniform.df, 5.0)
    var expected = _from[6]([16.0, 16.0, 16.0, 16.0, 16.0, 8.0])
    var against = chisquare(observed, expected)
    assert_almost_equal(against.statistic, 3.5, atol=1e-13)
    assert_almost_equal(against.pvalue, 0.6233876277495822, atol=1e-12)
    var fewer = chisquare(observed, expected, ddof=1)
    assert_almost_equal(fewer.pvalue, 0.477878344488724, atol=1e-12)
    var mismatched = _from[6]([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    var raised = False
    try:
        _ = chisquare(observed, mismatched)
    except:
        raised = True
    assert_true(raised)


def _standard_normal_cdf(x: Float64) -> Float64:
    return Float64(norm.cdf[P](P(x), P(0.0), P(1.0)).v)


def _shifted_normal_cdf(x: Float64) -> Float64:
    return Float64(norm.cdf[P](P(x), P(0.5), P(2.0)).v)


def test_ks_1samp_matches_scipy() raises:
    """The statistic under every alternative, the exact one-sided p-values,
    and the two-sided asymptotic p-value SciPy gives with
    `method="asymp"` -- the stated divergence from its exact default."""
    var x = _from[16](_x())
    var two = ks_1samp[cdf=_standard_normal_cdf](x)
    assert_almost_equal(two.statistic, 0.2788447460685429, atol=1e-13)
    assert_almost_equal(two.pvalue, 0.1660333420592824, atol=1e-12)
    var less = ks_1samp[cdf=_standard_normal_cdf](x, "less")
    assert_almost_equal(less.statistic, 0.2788447460685429, atol=1e-13)
    assert_almost_equal(less.pvalue, 0.06814455516517184, atol=1e-12)
    var greater = ks_1samp[cdf=_standard_normal_cdf](x, "greater")
    assert_almost_equal(greater.statistic, 0.1022498680518208, atol=1e-13)
    assert_almost_equal(greater.pvalue, 0.6718170332331084, atol=1e-12)
    var shifted = ks_1samp[cdf=_shifted_normal_cdf](x)
    assert_almost_equal(shifted.statistic, 0.1512936743170763, atol=1e-13)


def test_f_oneway_and_mannwhitneyu_match_scipy() raises:
    var x = _from[16](_x())
    var y = _from[16](_y())
    var z = _from[16](_z())
    var anova = f_oneway(x, y, z)
    assert_almost_equal(anova.statistic, 2.3484975270086514, atol=1e-12)
    assert_almost_equal(anova.pvalue, 0.10711582580463717, atol=1e-12)
    assert_almost_equal(anova.df, 2.0)
    var u = mannwhitneyu(x, y)
    assert_almost_equal(u.statistic, 130.5, atol=1e-13)
    assert_almost_equal(u.pvalue, 0.939731900374268, atol=1e-12)
    var less = mannwhitneyu(x, y, "less")
    assert_almost_equal(less.pvalue, 0.5451473426736253, atol=1e-12)
    var plain = mannwhitneyu(x, y, use_continuity=False)
    assert_almost_equal(plain.pvalue, 0.9247051982468886, atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
