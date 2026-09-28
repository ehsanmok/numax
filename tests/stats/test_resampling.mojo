"""Tests for `bootstrap` and `permutation_test` against SciPy: the exact
p-values of all three permutation types on samples small enough to
enumerate, a Monte Carlo p-value within its sampling error, and the
bootstrap's three intervals within theirs; plus the identities that
hold draw for draw -- `"basic"` is `"percentile"` reflected about the
observed statistic, and the percentile interval is the distribution's
own quantiles."""

from std.math import sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax.core.ops import multiply, subtract
from numax.core.tensor import Dynamic, Static
from numax.stats import (
    Generator,
    bootstrap,
    mean,
    percentile,
    permutation_test,
    sum,
)

comptime f64 = DType.float64


def _row_mean[gpu: Bool](x: Dynamic[f64, 2]) raises -> Dynamic[f64, 1]:
    return mean[axis=1, gpu=gpu](x)


def _diff_means[
    gpu: Bool
](a: Dynamic[f64, 2], b: Dynamic[f64, 2]) raises -> Dynamic[f64, 1]:
    return subtract[gpu=gpu](mean[axis=1, gpu=gpu](a), mean[axis=1, gpu=gpu](b))


def _dot[
    gpu: Bool
](a: Dynamic[f64, 2], b: Dynamic[f64, 2]) raises -> Dynamic[f64, 1]:
    return sum[axis=1, gpu=gpu](multiply[gpu=gpu](a, b))


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _big_x() raises -> Static[f64, 30]:
    return Static[f64, 30](
        [
            0.126,
            -0.132,
            0.64,
            0.105,
            -0.536,
            0.362,
            1.304,
            0.947,
            -0.704,
            -1.265,
            -0.623,
            0.041,
            -2.325,
            -0.219,
            -1.246,
            -0.732,
            -0.544,
            -0.316,
            0.412,
            1.043,
            -0.129,
            1.366,
            -0.665,
            0.352,
            0.903,
            0.094,
            -0.743,
            -0.922,
            -0.458,
            0.22,
        ],
        _cpu(),
    )


def _big_y() raises -> Static[f64, 30]:
    return Static[f64, 30](
        [
            -1.01,
            -0.209,
            -0.159,
            0.541,
            0.215,
            0.355,
            -0.654,
            -0.13,
            0.784,
            1.493,
            -1.259,
            1.514,
            1.346,
            0.781,
            0.264,
            -0.314,
            1.458,
            1.96,
            1.802,
            1.315,
            0.357,
            -1.208,
            -0.004,
            0.656,
            -1.288,
            0.395,
            0.43,
            0.696,
            -1.184,
            -0.662,
        ],
        _cpu(),
    )


def test_independent_exact_matches_scipy() raises:
    var x = Static[f64, 5]([2.1, 3.5, 2.9, 4.0, 5.2], _cpu())
    var y = Static[f64, 6]([1.2, 2.8, 1.9, 3.3, 2.2, 0.7], _cpu())
    var rng = Generator(seed=0)
    var two = permutation_test[statistic=_diff_means](x, y, rng)
    assert_almost_equal(two.statistic, 1.5233333333333339, atol=1e-13)
    assert_almost_equal(two.pvalue, 0.047619047619047616, atol=1e-15)
    assert_equal(two.null_distribution.size(), 462)
    var less = permutation_test[statistic=_diff_means](
        x, y, rng, alternative="less"
    )
    assert_almost_equal(less.pvalue, 0.9826839826839827, atol=1e-15)
    var greater = permutation_test[statistic=_diff_means](
        x, y, rng, alternative="greater"
    )
    assert_almost_equal(greater.pvalue, 0.023809523809523808, atol=1e-15)


def test_samples_exact_matches_scipy() raises:
    var a = Static[f64, 8]([1.1, 2.3, 0.4, 3.1, 2.2, 1.8, 0.9, 2.7], _cpu())
    var b = Static[f64, 8]([0.8, 2.5, 0.1, 2.2, 2.4, 1.1, 0.3, 2.0], _cpu())
    var rng = Generator(seed=0)
    var two = permutation_test[statistic=_diff_means](
        a, b, rng, permutation_type="samples"
    )
    assert_almost_equal(two.statistic, 0.3875000000000002, atol=1e-13)
    assert_almost_equal(two.pvalue, 0.046875, atol=1e-15)
    assert_equal(two.null_distribution.size(), 256)
    var greater = permutation_test[statistic=_diff_means](
        a, b, rng, permutation_type="samples", alternative="greater"
    )
    assert_almost_equal(greater.pvalue, 0.0234375, atol=1e-15)


def test_pairings_exact_matches_scipy() raises:
    # SciPy counts `(n!)^2` pairings and samples at this size; with
    # `n_resamples=inf` it enumerates, and that is the value to match.
    var u = Static[f64, 6]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], _cpu())
    var v = Static[f64, 6]([2.0, 1.0, 4.0, 3.0, 6.0, 5.0], _cpu())
    var rng = Generator(seed=0)
    var two = permutation_test[statistic=_dot](
        u, v, rng, permutation_type="pairings"
    )
    assert_almost_equal(two.statistic, 88.0, atol=1e-12)
    assert_equal(two.null_distribution.size(), 720)
    assert_almost_equal(two.pvalue, 0.058333333333333334, atol=1e-15)
    var greater = permutation_test[statistic=_dot](
        u, v, rng, permutation_type="pairings", alternative="greater"
    )
    assert_almost_equal(greater.pvalue, 0.029166666666666667, atol=1e-15)


def test_monte_carlo_pvalue_within_sampling_error() raises:
    # SciPy with 10^6 resamples: 0.0886. At 9999 draws the standard
    # error is sqrt(p (1 - p) / 9999) ~ 0.0028 per tail, doubled.
    var rng = Generator(seed=11)
    var res = permutation_test[statistic=_diff_means](_big_x(), _big_y(), rng)
    assert_equal(res.null_distribution.size(), 9999)
    assert_almost_equal(res.statistic, -0.39749999999999996, atol=1e-13)
    assert_true(abs(res.pvalue - 0.08863991136008864) < 0.02)
    # The draws are the generator's: the same seed, the same p-value.
    var again = Generator(seed=11)
    var repeat = permutation_test[statistic=_diff_means](
        _big_x(), _big_y(), again
    )
    assert_equal(repeat.pvalue, res.pvalue)


def test_monte_carlo_other_types_are_permutations() raises:
    # Each drawn `"samples"` rearrangement keeps every pair, so the sum
    # of both groups' means is the observed one on every row.
    var rng = Generator(seed=5)
    var x = _big_x()
    var y = _big_y()
    var paired = permutation_test[statistic=_diff_means](
        x, y, rng, permutation_type="samples", n_resamples=500
    )
    assert_equal(paired.null_distribution.size(), 500)
    assert_true(paired.pvalue > 0.0 and paired.pvalue <= 1.0)
    var dot = permutation_test[statistic=_dot](
        x, y, rng, permutation_type="pairings", n_resamples=500
    )
    # A reordering of `y` keeps `sum(x y)` within Cauchy-Schwarz.
    var bound = 0.0
    var xx = 0.0
    var yy = 0.0
    var xh = x.to_host()
    var yh = y.to_host()
    for i in range(30):
        xx += xh[i] * xh[i]
        yy += yh[i] * yh[i]
    bound = sqrt(xx * yy)
    var null = dot.null_distribution.to_host()
    for i in range(500):
        assert_true(abs(null[i]) <= bound + 1e-12)


def test_bootstrap_intervals_match_scipy() raises:
    # SciPy with 2 * 10^5 resamples. The interval ends move by about
    # 1.5 sigma / sqrt(n_resamples) in quantile terms; 0.02 is a few
    # of those at 9999.
    var rng = Generator(seed=3)
    var pct = bootstrap[statistic=_row_mean](_big_x(), rng, method="percentile")
    assert_true(abs(pct.confidence_interval.low - -0.4146341666666665) < 0.02)
    assert_true(abs(pct.confidence_interval.high - 0.1629341666666665) < 0.02)
    assert_true(abs(pct.standard_error - 0.14747138096120413) < 0.005)
    var bca = bootstrap[statistic=_row_mean](_big_x(), rng)
    assert_true(abs(bca.confidence_interval.low - -0.4217333333333334) < 0.02)
    assert_true(abs(bca.confidence_interval.high - 0.15619180630992535) < 0.02)
    var basic = bootstrap[statistic=_row_mean](_big_x(), rng, method="basic")
    assert_true(abs(basic.confidence_interval.low - -0.4058674999999998) < 0.02)
    assert_true(
        abs(basic.confidence_interval.high - 0.17170083333333316) < 0.02
    )


def test_basic_reflects_percentile_draw_for_draw() raises:
    var observed = mean(_big_x())
    var r1 = Generator(seed=9)
    var pct = bootstrap[statistic=_row_mean](
        _big_x(), r1, n_resamples=2000, method="percentile"
    )
    var r2 = Generator(seed=9)
    var basic = bootstrap[statistic=_row_mean](
        _big_x(), r2, n_resamples=2000, method="basic"
    )
    assert_almost_equal(
        basic.confidence_interval.low,
        2.0 * observed - pct.confidence_interval.high,
        atol=1e-14,
    )
    assert_almost_equal(
        basic.confidence_interval.high,
        2.0 * observed - pct.confidence_interval.low,
        atol=1e-14,
    )
    # The percentile interval is `numpy.percentile` of the distribution.
    assert_almost_equal(
        pct.confidence_interval.low,
        Float64(percentile(pct.bootstrap_distribution, 2.5)),
        atol=1e-14,
    )
    assert_almost_equal(
        pct.confidence_interval.high,
        Float64(percentile(pct.bootstrap_distribution, 97.5)),
        atol=1e-14,
    )


def test_one_sided_bootstrap() raises:
    var rng = Generator(seed=4)
    var less = bootstrap[statistic=_row_mean](
        _big_x(), rng, n_resamples=500, alternative="less"
    )
    assert_true(less.confidence_interval.low < -1e300)
    var greater = bootstrap[statistic=_row_mean](
        _big_x(), rng, n_resamples=500, alternative="greater"
    )
    assert_true(greater.confidence_interval.high > 1e300)


def test_argument_errors() raises:
    var rng = Generator(seed=0)
    with assert_raises(contains="method"):
        _ = bootstrap[statistic=_row_mean](_big_x(), rng, method="bca")
    with assert_raises(contains="permutation_type"):
        _ = permutation_test[statistic=_diff_means](
            _big_x(), _big_y(), rng, permutation_type="paired"
        )
    var short = Static[f64, 3]([1.0, 2.0, 3.0], _cpu())
    with assert_raises(contains="one length"):
        _ = permutation_test[statistic=_diff_means](
            _big_x(), short, rng, permutation_type="samples"
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
