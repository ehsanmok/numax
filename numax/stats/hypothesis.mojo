"""Hypothesis tests over `numax.core.tensor.Tensor`: the three `t` tests,
`chisquare`, `ks_1samp`, `f_oneway` and `mannwhitneyu`, the `k`-group
`kruskal`, `levene` and `bartlett`, and the normality tests `skewtest`,
`kurtosistest`, `normaltest` and `jarque_bera`, and the exact and
contingency tests `fisher_exact`, `binomtest` and `chi2_contingency`,
each returning SciPy's statistic and
p-value.

**Tier 2**: a test statistic is a few sums (or, for the rank tests, a
sort) and its p-value is one tail of a distribution `numax.stats`
already has -- `t.sf`, `chi2.sf`, `f.sf`, `norm.sf` -- so each test is
that arithmetic and that call. On the host all of it is `Float64`. The
`t` tests, both `chisquare`s, `f_oneway`, `mannwhitneyu`, `wilcoxon` and
`ks_2samp` take `gpu: Bool = False`: at `gpu=True`, with the samples on
a device, their sums are device reductions (`_moments_device`: count,
mean and the sum of squared deviations), the rank tests rank on the
device (`_rank_sum_device`) and `ks_2samp` counts its gaps there as
exact integers (`_ks_gaps_device`), and only scalars come back; the tail is host scalar
work either way, which the device-complete rule allows. The device sums
are at the samples' dtype and reassociated, so a `float32` statistic
agrees with the host's to `float32` precision. `ks_1samp` is the one
host-only test: its `cdf` is a `Float64` function parameter, and a
`Float64` kernel does not compile on Metal.

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
digit. `mannwhitneyu`, `ks_2samp` and `wilcoxon` are likewise always
asymptotic, where SciPy enumerates exactly for tiny untied samples;
`ks_2samp`'s one-sided tails carry Hodges' finite-sample correction,
which is SciPy's `method="asymp"` formula and not optional -- without it
the tail reads 0.607 where SciPy reads 0.472 for two samples of eight.

## The MAX gate

Nothing: MAX has no statistical tests. **Extend.**
"""

from std.builtin.sort import sort as _sort
from std.math import (
    ceil as _ceil,
    exp as _exp,
    floor as _floor,
    lgamma as _lgamma,
    log as _log,
    sqrt as _sqrt,
)

from max.gpu.host import DeviceContext

from layout.tile_layout import TensorLayout

from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise

from ..core._drive import (
    _check_device,
    _flat_out,
    _flat_unchecked,
    _notice,
)
from ..core.ops import (
    multiply as _tmultiply,
    subtract as _tsubtract,
)
from algorithm.rowwise_types import RowCoord
from ..core.rowwise import reduce_all
from ..core.sorting import _pack_device, _sort_device, argsort, take
from ..core.tensorlike import TensorLike, dim, is_row_major
from ..core.tensor import Dynamic, Static, Tensor, _dyn_shape
from ..core.plain import Plain
from .distributions import binom, chi2, f, norm, t
from .statistics import median as _tmedian, sum as _tsum
from ..core.elementwise import abs as _tabs
from .descriptive import kurtosis as _kurtosis, skew as _skew

comptime _P = Plain[DType.float64]


@fieldwise_init
struct TestResult(Copyable):
    """What every test here returns: SciPy's `statistic` and `pvalue`, and
    the degrees of freedom `df` where the test has one (`0` where it does
    not)."""

    var statistic: Float64
    var pvalue: Float64
    var df: Float64


def _values[T: TensorLike](xs: T) raises -> List[Float64]:
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
    var center = _mean(values)
    var total = 0.0
    for i in range(len(values)):
        total += (values[i] - center) * (values[i] - center)
    return total / Float64(len(values) - ddof)


def _moments_device[
    T: TensorLike
](xs: T) raises -> Tuple[Int, Float64, Float64] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """`(n, mean, sum of squared deviations)` of a device sample, as
    device sums: one for the mean, a centering launch, a square and one
    more sum. Three scalars come back."""
    var n = xs.size()
    var mean = Float64(_tsum[gpu=True](xs)) / Float64(n)
    var d = _tsubtract[gpu=True](xs, Scalar[T.dtype](mean))
    var m2 = Float64(_tsum[gpu=True](_tmultiply[gpu=True](d, d)))
    return (n, mean, m2)


def _chi2_terms_device[
    T: TensorLike
](observed: T, expected: T) raises -> Dynamic[T.dtype, 1] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """`(o - e)^2 / e` per element, flat, in one launch on the device."""
    var ctx = observed.context()
    var n = observed.size()
    var out = Dynamic[T.dtype, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](n))
    )
    var ov = _flat_unchecked(observed)
    var ev = _flat_unchecked(expected)
    var tv = _flat_out(out)

    @always_inline
    def term[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var ov, var ev, var tv}:
        var e = ev[coord][0]
        var d = ov[coord][0] - e
        tv.store[1](coord, rebind[Scalar[T.dtype]](d * d / e))

    elementwise[simd_width=1, target="gpu"](term, Coord(n), ctx)
    return out^


def _dtype_epsilon[dtype: DType]() -> Float64:
    """The spacing of `dtype` at one, `numpy.finfo(dtype).eps`."""
    comptime if dtype == DType.float64:
        return 2.220446049250313e-16
    elif dtype == DType.float32:
        return 1.1920928955078125e-07
    elif dtype == DType.float16:
        return 9.765625e-04
    else:
        return 7.8125e-03


def _check_alternative(name: StaticString, alternative: StaticString) raises:
    if not (
        alternative == "two-sided"
        or alternative == "less"
        or alternative == "greater"
    ):
        raise Error(
            name,
            ": alternative must be 'two-sided', 'less' or 'greater', not '",
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
    T: TensorLike, gpu: Bool = False
](
    xs: T,
    popmean: Float64,
    alternative: StaticString = "two-sided",
) raises -> TestResult where (is_row_major[T] and T.dtype.is_floating_point()):
    """The one-sample `t` test that the mean of `xs` is `popmean`.
    `scipy.stats.ttest_1samp(a, popmean, alternative)`: `t = (mean -
    popmean) / (std_1 / sqrt(n))` on `n - 1` degrees of freedom. Device
    sums at `gpu=True`; see the module docstring.

    Parameters:
        T: The tensor type of the sample, row-major and floating-point.
        gpu: Whether the sums run on the sample's device; a residency
            mismatch falls back to the host with a notice.

    Args:
        xs: The sample, read flat in any shape.
        popmean: The hypothesized population mean.
        alternative: `"two-sided"`, `"less"` or `"greater"`, the tail
            the p-value is taken on.

    Returns:
        A `TestResult` with the `t` statistic, its p-value and `df = n - 1`.

    Raises:
        If `alternative` is not one of the three names, if `xs` has fewer
        than two elements, or on a fallback under the `"raise"` policy.
    """
    _check_alternative("ttest_1samp", alternative)
    if _check_device[T, gpu](xs):
        comptime if gpu:
            var m = _moments_device(xs)
            if m[0] < 2:
                raise Error("ttest_1samp: at least two observations are needed")
            var dfd = Float64(m[0] - 1)
            var stat = (m[1] - popmean) / _sqrt(m[2] / dfd / Float64(m[0]))
            return TestResult(stat, _t_pvalue(stat, dfd, alternative), dfd)
    else:
        _notice[gpu]("ttest_1samp")
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
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](
    xs: A,
    ys: B,
    equal_var: Bool = True,
    alternative: StaticString = "two-sided",
) raises -> TestResult where (
    A.dtype.is_floating_point()
    and B.dtype.is_floating_point()
    and B.dtype == A.dtype
    and is_row_major[A]
    and is_row_major[B]
):
    """The two-sample `t` test that two independent samples share a mean.
    `scipy.stats.ttest_ind(a, b, equal_var, alternative)`: Student's
    pooled-variance test by default, Welch's unequal-variance test with
    its Welch-Satterthwaite degrees of freedom when `equal_var=False`.
    Device sums at `gpu=True`; see the module docstring.

    Parameters:
        A: The tensor type of the first sample, row-major and floating-point.
        B: The tensor type of the second sample, at `A`'s dtype.
        gpu: Whether the sums run on the samples' device; if either sample
            is not where `gpu` expects, both fall back to the host with a
            notice.

    Args:
        xs: The first sample, read flat in any shape.
        ys: The second sample, read flat; its length may differ from `xs`.
        equal_var: `True` for Student's pooled-variance test, `False` for
            Welch's test.
        alternative: `"two-sided"`, `"less"` or `"greater"`, the tail
            the p-value is taken on.

    Returns:
        A `TestResult` with the `t` statistic, its p-value and the degrees
        of freedom (`n1 + n2 - 2`, or Welch-Satterthwaite's).

    Raises:
        If `alternative` is not one of the three names, if either sample has
        fewer than two elements, or on a fallback under the `"raise"`
        policy.
    """
    _check_alternative("ttest_ind", alternative)
    var n1: Float64
    var n2: Float64
    var v1: Float64
    var v2: Float64
    var m1: Float64
    var m2: Float64
    if _check_device[A, gpu](xs) and _check_device[B, gpu](ys):
        comptime if gpu:
            var ma = _moments_device(xs)
            var mb = _moments_device(ys)
            if ma[0] < 2 or mb[0] < 2:
                raise Error(
                    "ttest_ind: each sample needs at least two observations"
                )
            n1 = Float64(ma[0])
            n2 = Float64(mb[0])
            m1 = ma[1]
            m2 = mb[1]
            v1 = ma[2] / (n1 - 1.0)
            v2 = mb[2] / (n2 - 1.0)
            return _ttest_ind_finish(
                n1, n2, m1, m2, v1, v2, equal_var, alternative
            )
    else:
        _notice[gpu]("ttest_ind")
    var a = _values(xs)
    var b = _values(ys)
    n1 = Float64(len(a))
    n2 = Float64(len(b))
    if len(a) < 2 or len(b) < 2:
        raise Error("ttest_ind: each sample needs at least two observations")
    v1 = _variance(a, 1)
    v2 = _variance(b, 1)
    m1 = _mean(a)
    m2 = _mean(b)
    return _ttest_ind_finish(n1, n2, m1, m2, v1, v2, equal_var, alternative)


def _ttest_ind_finish(
    n1: Float64,
    n2: Float64,
    m1: Float64,
    m2: Float64,
    v1: Float64,
    v2: Float64,
    equal_var: Bool,
    alternative: StaticString,
) -> TestResult:
    """Student's or Welch's statistic, degrees of freedom and p-value from
    the two samples' sizes, means and `ddof = 1` variances."""
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
    var statistic = (m1 - m2) / denominator
    return TestResult(statistic, _t_pvalue(statistic, df, alternative), df)


def ttest_rel[
    T: TensorLike, gpu: Bool = False
](
    xs: T,
    ys: T,
    alternative: StaticString = "two-sided",
) raises -> TestResult where (is_row_major[T] and T.dtype.is_floating_point()):
    """The paired `t` test: `ttest_1samp` of the differences against zero.
    `scipy.stats.ttest_rel(a, b, alternative)`. At `gpu=True` the
    differences are one device `subtract` and their moments device
    sums.

    Parameters:
        T: The tensor type of both samples, row-major and floating-point.
        gpu: Whether the differences and sums run on the samples' device; a
            residency mismatch falls back to the host with a notice.

    Args:
        xs: The first sample of each pair, read flat.
        ys: The second sample of each pair, at the same length as `xs`.
        alternative: `"two-sided"`, `"less"` or `"greater"`, the tail
            the p-value is taken on.

    Returns:
        A `TestResult` with the `t` statistic of `xs - ys`, its p-value and
        `df = n - 1`.

    Raises:
        If `alternative` is not one of the three names, if there are fewer
        than two pairs, or on a fallback under the `"raise"` policy.
    """
    _check_alternative("ttest_rel", alternative)
    if _check_device[T, gpu](xs) and _check_device[T, gpu](ys):
        comptime if gpu:
            var m = _moments_device(_tsubtract[gpu=True](xs, ys))
            if m[0] < 2:
                raise Error("ttest_rel: at least two pairs are needed")
            var dfd = Float64(m[0] - 1)
            var stat = m[1] / _sqrt(m[2] / dfd / Float64(m[0]))
            return TestResult(stat, _t_pvalue(stat, dfd, alternative), dfd)
    else:
        _notice[gpu]("ttest_rel")
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
    T: TensorLike, gpu: Bool = False
](observed: T, ddof: Int = 0) raises -> TestResult where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Pearson's chi-squared test that the observed counts are uniform:
    `sum((o - e)^2 / e)` against `chi2` on `k - 1 - ddof` degrees of
    freedom, `e` the mean count. `scipy.stats.chisquare(f_obs, ddof)`.
    At `gpu=True` the statistic is the device sum of squared deviations
    over the mean count.

    Parameters:
        T: The tensor type of the counts, row-major and floating-point.
        gpu: Whether the sums run on the counts' device; a residency
            mismatch falls back to the host with a notice.

    Args:
        observed: The `k` observed counts, read flat.
        ddof: The adjustment subtracted from the `k - 1` degrees of freedom.

    Returns:
        A `TestResult` with the chi-squared statistic, its upper-tail
        p-value and `df = k - 1 - ddof`.

    Raises:
        If the device reduction fails, or on a fallback under the `"raise"`
        policy.
    """
    if _check_device[T, gpu](observed):
        comptime if gpu:
            var m = _moments_device(observed)
            var stat = m[2] / m[1]
            var dfd = Float64(m[0] - 1 - ddof)
            return TestResult(
                stat, Float64(chi2.sf[_P](_P(stat), _P(dfd)).v), dfd
            )
    else:
        _notice[gpu]("chisquare")
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
    T: TensorLike, gpu: Bool = False
](
    observed: T,
    expected: T,
    ddof: Int = 0,
) raises -> TestResult where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Pearson's chi-squared test against the given expected counts.
    `scipy.stats.chisquare(f_obs, f_exp, ddof)`. SciPy checks that the two
    sum to the same total to a relative `1e-8`; so does this, raising.
    At `gpu=True` both totals and the statistic are device sums at the
    samples' dtype, so the totals check allows the reassociated sum's own
    rounding, `n` units in the last place of that dtype, when that is
    looser than `1e-8`.

    Parameters:
        T: The tensor type of both count tensors, row-major and
            floating-point.
        gpu: Whether the totals and statistic run on the counts' device; a
            residency mismatch falls back to the host with a notice.

    Args:
        observed: The `k` observed counts, read flat.
        expected: The `k` expected counts, at the same length as `observed`.
        ddof: The adjustment subtracted from the `k - 1` degrees of freedom.

    Returns:
        A `TestResult` with the chi-squared statistic, its upper-tail
        p-value and `df = k - 1 - ddof`.

    Raises:
        If `observed` and `expected` do not sum to the same total within
        the tolerance above, or on a fallback under the `"raise"` policy.
    """
    if _check_device[T, gpu](observed) and _check_device[T, gpu](expected):
        comptime if gpu:
            var so = Float64(_tsum[gpu=True](observed))
            var se = Float64(_tsum[gpu=True](expected))
            var rtol = max(
                1e-8, Float64(observed.size()) * _dtype_epsilon[T.dtype]()
            )
            if abs(so - se) > rtol * max(abs(so), abs(se)):
                raise Error(
                    "chisquare: observed and expected counts must sum to the"
                    " same total"
                )
            var stat = Float64(
                _tsum[gpu=True](_chi2_terms_device(observed, expected))
            )
            var dfd = Float64(observed.size() - 1 - ddof)
            return TestResult(
                stat, Float64(chi2.sf[_P](_P(stat), _P(dfd)).v), dfd
            )
    else:
        _notice[gpu]("chisquare")
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
    T: TensorLike,
    cdf: def(Float64) thin -> Float64,
](xs: T, alternative: StaticString = "two-sided") raises -> TestResult where (
    is_row_major[T] and T.dtype.is_floating_point()
):
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

    Parameters:
        T: The tensor type of the sample, row-major and floating-point.
        cdf: The hypothesized continuous CDF, evaluated at `Float64` on the
            host.

    Args:
        xs: The sample, read flat and sorted on the host.
        alternative: `"two-sided"`, `"less"` or `"greater"`, selecting the
            statistic `max(D+, D-)`, `D-` or `D+`.

    Returns:
        A `TestResult` with the KS statistic, its p-value and `df` set to the
        sample size `n`.

    Raises:
        If `alternative` is not one of the three names, or if `xs` is empty.
    """
    _check_alternative("ks_1samp", alternative)
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
    T: TensorLike, gpu: Bool = False
](*groups: T) raises -> TestResult where (
    T.dtype.is_floating_point() and is_row_major[T]
):
    """The one-way ANOVA `F` test that every group shares a mean:
    between-group over within-group mean square, against `f` on `k - 1`
    and `N - k` degrees of freedom. `scipy.stats.f_oneway(*samples)`, the
    groups passed as separate arguments of one shape. `df` in the result
    is the numerator's; the denominator's is `N - k`. At `gpu=True`, with
    every group on a device, each group's size, mean and within sum of
    squares are device sums (`_moments_device`).

    Parameters:
        T: The tensor type of every group, row-major and floating-point.
        gpu: Whether the per-group sums run on the groups' device; if any
            group is not on a device, all fall back to the host with a
            notice.

    Args:
        groups: The `k` samples, all of tensor type `T`, each read flat.

    Returns:
        A `TestResult` with the `F` statistic, its upper-tail p-value and
        the numerator degrees of freedom `k - 1`.

    Raises:
        If fewer than two groups are given, or on a fallback under the
        `"raise"` policy.
    """
    var k = len(groups)
    if k < 2:
        raise Error("f_oneway: at least two groups are needed")
    comptime if gpu:
        var on_device = True
        for g in range(k):
            on_device = on_device and _check_device[T, gpu](groups[g])
        if on_device:
            var means = List[Float64](capacity=k)
            var sizes = List[Int](capacity=k)
            var grand = 0.0
            var total_n = 0
            var within = 0.0
            for g in range(k):
                var m = _moments_device(groups[g])
                sizes.append(m[0])
                means.append(m[1])
                grand += m[1] * Float64(m[0])
                within += m[2]
                total_n += m[0]
            return _f_oneway_finish(means, sizes, grand, total_n, within)
        _notice[gpu]("f_oneway")
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
    return _f_oneway_finish(means, sizes, grand, total_n, within)


def _f_oneway_finish(
    means: List[Float64],
    sizes: List[Int],
    total: Float64,
    total_n: Int,
    within: Float64,
) raises -> TestResult:
    """The `F` statistic and tail from each group's size and mean, the
    grand total and the pooled within-group sum of squares."""
    var k = len(means)
    var grand = total / Float64(total_n)
    var between = 0.0
    for g in range(k):
        between += Float64(sizes[g]) * (means[g] - grand) * (means[g] - grand)
    var dfb = Float64(k - 1)
    var dfw = Float64(total_n - k)
    var statistic = (between / dfb) / (within / dfw)
    return TestResult(
        statistic, Float64(f.sf[_P](_P(statistic), _P(dfb), _P(dfw)).v), dfb
    )


def _rank_sum_device[
    dtype: DType
](values: Dynamic[dtype, 1], flags: Dynamic[dtype, 1]) raises -> Tuple[
    Float64, Float64
] where dtype.is_floating_point():
    """On `values`' device: the sum of the average ranks of the elements
    whose `flags` entry is nonzero, and the tie term `sum (t^3 - t)` over
    the groups of equal values.

    The device `argsort` and a gather give the ascending values; one
    launch over the sorted positions binary-searches each value's run,
    `[first, last)`, whose average rank is `(first + 1 + last) / 2` and
    whose size `t` contributes `t^2 - 1` per element, so `t^3 - t` per
    group; two device sums finish. Two scalars come back.
    """
    var ctx = values.context()
    var n = values.size()
    var order = argsort[gpu=True](values)
    var sorted = take[axis=0, gpu=True](values, order)
    var ranked = Dynamic[dtype, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](n))
    )
    var ties = Dynamic[dtype, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](n))
    )
    var sv = _flat_unchecked(sorted)
    var ov = _flat_unchecked(order)
    var fv = _flat_unchecked(flags)
    var rv = _flat_out(ranked)
    var tv = _flat_out(ties)

    @always_inline
    def run[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var sv, var ov, var fv, var rv, var tv, var n}:
        var i = coord_to_index_list(coord)[0]
        var value = sv[coord][0]
        var lo = 0
        var hi = i
        while lo < hi:
            var mid = (lo + hi) // 2
            if sv[Coord(mid)][0] < value:
                lo = mid + 1
            else:
                hi = mid
        var first = lo
        lo = i
        hi = n
        while lo < hi:
            var mid = (lo + hi) // 2
            if sv[Coord(mid)][0] <= value:
                lo = mid + 1
            else:
                hi = mid
        var size = Scalar[dtype](lo - first)
        var flagged = fv[Coord(Int(ov[coord][0]))][0] != 0
        rv.store[1](
            coord,
            Scalar[dtype](first + 1 + lo) / 2 if flagged else Scalar[dtype](0),
        )
        tv.store[1](coord, size * size - 1)

    elementwise[simd_width=1, target="gpu"](run, Coord(n), ctx)
    return (
        Float64(_tsum[gpu=True](ranked)),
        Float64(_tsum[gpu=True](ties)),
    )


def _pooled_device[
    A: TensorLike, B: TensorLike
](xs: A, ys: B) raises -> Tuple[
    Dynamic[A.dtype, 1], Dynamic[A.dtype, 1]
] where (B.dtype == A.dtype):
    """`xs` then `ys` in one flat device vector, and a flag vector that is
    one over the `xs` part: one launch."""
    comptime dtype = A.dtype
    var ctx = xs.context()
    var n1 = xs.size()
    var n = n1 + ys.size()
    var pooled = Dynamic[dtype, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](n))
    )
    var flags = Dynamic[dtype, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](n))
    )
    var av = _flat_unchecked(xs)
    var bv = _flat_unchecked(ys)
    var pv = _flat_out(pooled)
    var fv = _flat_out(flags)

    @always_inline
    def join[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var av, var bv, var pv, var fv, var n1}:
        var i = coord_to_index_list(coord)[0]
        if i < n1:
            pv.store[1](coord, av[coord][0])
            fv.store[1](coord, Scalar[dtype](1))
        else:
            pv.store[1](coord, rebind[Scalar[dtype]](bv[Coord(i - n1)][0]))
            fv.store[1](coord, Scalar[dtype](0))

    elementwise[simd_width=1, target="gpu"](join, Coord(n), ctx)
    return (pooled^, flags^)


def _mannwhitneyu_finish(
    r1: Float64,
    tie_term: Float64,
    n1: Int,
    n2: Int,
    alternative: StaticString,
    use_continuity: Bool,
) -> TestResult:
    """`U1`, and the tie- and continuity-corrected normal p-value, from
    the first sample's rank sum and the pooled tie term."""
    var n = n1 + n2
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


def mannwhitneyu[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](
    xs: A,
    ys: B,
    alternative: StaticString = "two-sided",
    use_continuity: Bool = True,
) raises -> TestResult where (
    A.dtype.is_floating_point()
    and B.dtype == A.dtype
    and is_row_major[A]
    and is_row_major[B]
):
    """The Mann-Whitney `U` test that two independent samples come from
    one distribution. `scipy.stats.mannwhitneyu(x, y, use_continuity,
    alternative, method="asymptotic")`.

    The statistic is `U1`, the count of `(x, y)` pairs with `x > y` (ties
    half), as SciPy reports it; the p-value is the normal approximation
    with SciPy's tie correction and, by default, its continuity correction,
    taken on `U1`, `U2` or their maximum by `alternative`. Always
    asymptotic, where SciPy enumerates exactly for a small untied sample.

    At `gpu=True`, with both samples on a device, the pooled sample is
    ranked there (`_rank_sum_device`) and only the first sample's rank sum
    and the tie term come back.

    Parameters:
        A: The tensor type of the first sample, row-major and floating-point.
        B: The tensor type of the second sample, at `A`'s dtype.
        gpu: Whether the ranking runs on the samples' device; if either
            sample is not where `gpu` expects, both fall back to the host
            with a notice.

    Args:
        xs: The first sample, read flat.
        ys: The second sample, read flat; its length may differ from `xs`.
        alternative: `"two-sided"`, `"less"` or `"greater"`, selecting
            `max(U1, U2)`, `U2` or `U1` for the normal tail.
        use_continuity: Whether to subtract SciPy's `0.5` continuity
            correction before standardizing.

    Returns:
        A `TestResult` with `U1`, its asymptotic p-value and `df = 0`.

    Raises:
        If `alternative` is not one of the three names, or on a fallback
        under the `"raise"` policy.
    """
    _check_alternative("mannwhitneyu", alternative)
    if _check_device[A, gpu](xs) and _check_device[B, gpu](ys):
        comptime if gpu:
            var joined = _pooled_device(xs, ys)
            var sums = _rank_sum_device(joined[0], joined[1])
            return _mannwhitneyu_finish(
                sums[0],
                sums[1],
                xs.size(),
                ys.size(),
                alternative,
                use_continuity,
            )
    else:
        _notice[gpu]("mannwhitneyu")
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

    def by_value(i: Int, j: Int) {imm} -> Bool:
        return pooled[i] < pooled[j] or (pooled[i] == pooled[j] and i < j)

    _sort(order, by_value)
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
    return _mannwhitneyu_finish(
        r1, tie_term, n1, n2, alternative, use_continuity
    )


def _ks_gaps_device[
    A: TensorLike, B: TensorLike
](xs: A, ys: B) raises -> Tuple[Int, Int] where B.dtype == A.dtype:
    """The two-sample KS gaps as exact integers: over every sample value
    `v`, the largest and the most negative `n2 F1(v) - n1 F2(v)`.

    Both samples sort on the device; one launch over all `n1 + n2`
    values reads each empirical distribution at the value by a binary
    search for its upper bound -- the value after every equal element
    in both samples, which is the host walk's tie rule -- and two
    device `max` reductions finish. Two integers come back.
    """
    var ctx = xs.context()
    var n1 = xs.size()
    var n2 = ys.size()
    var sa = _sort_device(xs)
    var sb = _sort_device(ys)
    var gaps = Dynamic[DType.int64, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](n1 + n2))
    )
    var flipped = Dynamic[DType.int64, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](n1 + n2))
    )
    var av = _flat_unchecked(sa)
    var bv = _flat_unchecked(sb)
    var gv = _flat_out(gaps)
    var hv = _flat_out(flipped)

    @always_inline
    def gap[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var av, var bv, var gv, var hv, var n1, var n2}:
        var i = coord_to_index_list(coord)[0]
        var value = av[coord][0] if i < n1 else rebind[Scalar[A.dtype]](
            bv[Coord(i - n1)][0]
        )
        var lo = 0
        var hi = n1
        while lo < hi:
            var mid = (lo + hi) // 2
            if av[Coord(mid)][0] <= value:
                lo = mid + 1
            else:
                hi = mid
        var in_a = lo
        lo = 0
        hi = n2
        while lo < hi:
            var mid = (lo + hi) // 2
            if rebind[Scalar[A.dtype]](bv[Coord(mid)][0]) <= value:
                lo = mid + 1
            else:
                hi = mid
        var d = Int64(in_a) * Int64(n2) - Int64(lo) * Int64(n1)
        gv.store[1](coord, d)
        hv.store[1](coord, -d)

    elementwise[simd_width=1, target="gpu"](gap, Coord(n1 + n2), ctx)

    @always_inline
    def identity[
        w: Int
    ](tile: SIMD[DType.int64, w], idx: RowCoord[1]) {} -> SIMD[DType.int64, w]:
        return tile

    var top = Static[DType.int64, 2](ctx)
    var tv = top.tile()
    reduce_all[monoid="max", gpu=True](
        _flat_unchecked(gaps),
        tv.slice[0:1]().as_unsafe_any_origin(),
        identity,
        n1 + n2,
        Optional(ctx),
    )
    reduce_all[monoid="max", gpu=True](
        _flat_unchecked(flipped),
        tv.slice[1:2]().as_unsafe_any_origin(),
        identity,
        n1 + n2,
        Optional(ctx),
    )
    var raw = top.to_host()
    return (Int(raw[0]), Int(raw[1]))


def _ks_2samp_finish(
    d_plus: Float64,
    d_minus: Float64,
    n1: Int,
    n2: Int,
    alternative: StaticString,
) -> TestResult:
    """The statistic and asymptotic tail from the two one-sided gaps."""
    var effective = Float64(n1 * n2) / Float64(n1 + n2)
    if alternative == "greater":
        return TestResult(d_plus, _ks_hodges(n1, n2, d_plus), effective)
    if alternative == "less":
        return TestResult(d_minus, _ks_hodges(n1, n2, d_minus), effective)
    var d = max(d_plus, d_minus)
    return TestResult(d, _kolmogorov_sf(_sqrt(effective) * d), effective)


def ks_2samp[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](
    xs: A, ys: B, alternative: StaticString = "two-sided"
) raises -> TestResult where (
    A.dtype.is_floating_point()
    and B.dtype == A.dtype
    and is_row_major[A]
    and is_row_major[B]
):
    """The two-sample Kolmogorov-Smirnov test that `xs` and `ys` come from
    one continuous distribution.
    `scipy.stats.ks_2samp(x, y, alternative, method="asymp")`.

    The statistic is the largest gap between the two empirical
    distributions -- `D+`, `D-` or their maximum by `alternative`, with
    `"greater"` and `"less"` naming the sign of the gap SciPy names. Where
    `ks_1samp` compares a sample against a `cdf`, this walks both sorted
    samples together and compares the two step functions.

    The p-value is asymptotic: the effective sample size
    `n1 n2 / (n1 + n2)` is substituted into the same Kolmogorov tail
    `ks_1samp` uses two-sided, and into the Birnbaum-Tingey exponential
    one-sided. SciPy computes an exact p-value for small untied samples
    and this does not -- the same declared divergence `ks_1samp` and
    `mannwhitneyu` carry, and `docs/parity.md` records.

    `df` carries the effective sample size rather than a degrees of
    freedom, since this test has none.

    At `gpu=True`, with both samples on a device, both sort there and the
    gaps are exact integer numerators (`_ks_gaps_device`); two integers
    come back.

    Parameters:
        A: The tensor type of the first sample, row-major and floating-point.
        B: The tensor type of the second sample, at `A`'s dtype.
        gpu: Whether the sorts and gaps run on the samples' device; if
            either sample is not where `gpu` expects, both fall back to the
            host with a notice.

    Args:
        xs: The first sample, read flat.
        ys: The second sample, read flat; its length may differ from `xs`.
        alternative: `"two-sided"`, `"less"` or `"greater"`, selecting
            `max(D+, D-)`, `D-` or `D+`.

    Returns:
        A `TestResult` with the KS statistic, its asymptotic p-value and
        `df` set to the effective sample size `n1 n2 / (n1 + n2)`.

    Raises:
        If `alternative` is not one of the three names, if either sample
        is empty, or on a fallback under the `"raise"` policy.
    """
    _check_alternative("ks_2samp", alternative)
    if _check_device[A, gpu](xs) and _check_device[B, gpu](ys):
        comptime if gpu:
            var m1 = xs.size()
            var m2 = ys.size()
            if m1 < 1 or m2 < 1:
                raise Error("ks_2samp: both samples must be non-empty")
            var g = _ks_gaps_device(xs, ys)
            var scale = Float64(m1) * Float64(m2)
            return _ks_2samp_finish(
                max(0.0, Float64(g[0]) / scale),
                max(0.0, Float64(g[1]) / scale),
                m1,
                m2,
                alternative,
            )
    else:
        _notice[gpu]("ks_2samp")
    var a = _values(xs)
    var b = _values(ys)
    var n1 = len(a)
    var n2 = len(b)
    if n1 < 1 or n2 < 1:
        raise Error("ks_2samp: both samples must be non-empty")
    _sort(a)
    _sort(b)

    # Walk the union of the two sorted samples, tracking each empirical
    # distribution at the current value. Ties have to advance *both*
    # before the gap is read, or a shared value reports a gap that the
    # step functions do not actually have.
    var i = 0
    var j = 0
    var d_plus = 0.0
    var d_minus = 0.0
    while i < n1 and j < n2:
        var at = min(a[i], b[j])
        while i < n1 and a[i] <= at:
            i += 1
        while j < n2 and b[j] <= at:
            j += 1
        var fa = Float64(i) / Float64(n1)
        var fb = Float64(j) / Float64(n2)
        d_plus = max(d_plus, fa - fb)
        d_minus = max(d_minus, fb - fa)
    return _ks_2samp_finish(d_plus, d_minus, n1, n2, alternative)


def _ks_hodges(n1: Int, n2: Int, d: Float64) -> Float64:
    """The one-sided two-sample KS tail with Hodges' correction, which is
    the formula `scipy.stats.ks_2samp` uses at `method="asymp"`.

    `exp(-2 z^2)` is the Birnbaum-Tingey limit; the second term is Hodges'
    equation 5.3, a finite-sample correction in `(m + 2n) / sqrt(m n
    (m + n))`. Without it the tail is visibly wrong at the sample sizes a
    two-sample test is actually run at -- 0.607 against SciPy's 0.472 for
    two samples of eight -- so it is not an optional refinement.
    """
    if d <= 0.0:
        return 1.0
    var m = Float64(n1)
    var nn = Float64(n2)
    var effective = m * nn / (m + nn)
    var z = _sqrt(effective) * d
    var correction = 2.0 * z * (m + 2.0 * nn) / _sqrt(m * nn * (m + nn)) / 3.0
    return min(1.0, max(0.0, _exp(-2.0 * z * z - correction)))


def _signed_magnitudes_device[
    A: TensorLike, B: TensorLike
](xs: A, ys: B) raises -> Tuple[
    Dynamic[A.dtype, 1], Dynamic[A.dtype, 1]
] where (B.dtype == A.dtype):
    """The nonzero paired differences' magnitudes and a positive-sign
    flag for each, packed on the device: one launch for `d = x - y`,
    `|d|` and the sign, and two compactions by `d`."""
    comptime dtype = A.dtype
    var ctx = xs.context()
    var n = xs.size()
    var diffs = Dynamic[dtype, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](n))
    )
    var mags = Dynamic[dtype, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](n))
    )
    var signs = Dynamic[dtype, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](n))
    )
    var av = _flat_unchecked(xs)
    var bv = _flat_unchecked(ys)
    var dv = _flat_out(diffs)
    var mv = _flat_out(mags)
    var sv = _flat_out(signs)

    @always_inline
    def split[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var av, var bv, var dv, var mv, var sv}:
        var d = av[coord][0] - rebind[Scalar[dtype]](bv[coord][0])
        dv.store[1](coord, d)
        mv.store[1](coord, abs(d))
        sv.store[1](coord, Scalar[dtype](1) if d > 0 else Scalar[dtype](0))

    elementwise[simd_width=1, target="gpu"](split, Coord(n), ctx)
    return (
        _pack_device[indices=False](diffs, mags, n),
        _pack_device[indices=False](diffs, signs, n),
    )


def _wilcoxon_finish(
    w_plus: Float64,
    tie_term: Float64,
    n: Int,
    alternative: StaticString,
    use_continuity: Bool,
) raises -> TestResult:
    """`W` and the tie- and continuity-corrected normal p-value from the
    positive rank sum over the `n` nonzero differences."""
    var total = Float64(n) * Float64(n + 1) / 2.0
    var w_minus = total - w_plus

    var mean = total / 2.0
    var variance = (
        Float64(n) * Float64(n + 1) * Float64(2 * n + 1) / 24.0
        - tie_term / 48.0
    )
    if variance <= 0.0:
        raise Error("wilcoxon: every difference is tied, so W has no spread")
    var spread = _sqrt(variance)

    var statistic = w_plus
    if alternative == "two-sided":
        statistic = min(w_plus, w_minus)

    var correction = 0.5 if use_continuity else 0.0
    if alternative == "greater":
        var z = (w_plus - mean - correction) / spread
        return TestResult(
            statistic, norm.sf(_P(z), _P(0.0), _P(1.0)).v[0], Float64(n)
        )
    if alternative == "less":
        var z = (w_plus - mean + correction) / spread
        return TestResult(
            statistic, norm.cdf(_P(z), _P(0.0), _P(1.0)).v[0], Float64(n)
        )
    var z = ((w_plus - mean).__abs__() - correction) / spread
    var tail = norm.sf(_P(z), _P(0.0), _P(1.0)).v[0]
    return TestResult(statistic, min(1.0, 2.0 * tail), Float64(n))


def wilcoxon[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](
    xs: A,
    ys: B,
    alternative: StaticString = "two-sided",
    use_continuity: Bool = True,
) raises -> TestResult where (
    A.dtype.is_floating_point()
    and B.dtype == A.dtype
    and is_row_major[A]
    and is_row_major[B]
):
    """The Wilcoxon signed-rank test that the paired differences
    `xs - ys` are centered on zero.
    `scipy.stats.wilcoxon(x, y, alternative, correction, method="approx")`.

    `ttest_rel`'s nonparametric counterpart: it ranks the *magnitudes* of
    the paired differences and sums the ranks of the positive ones, so a
    pair of outliers cannot move it the way they move a mean. The
    statistic is SciPy's `W`, the smaller of the positive and negative
    rank sums for `"two-sided"` and the positive sum for the one-sided
    alternatives.

    Zero differences are **dropped** before ranking, SciPy's `"wilcox"`
    zero_method and its default, and the sample size falls with them. Ties
    among the magnitudes take average ranks and enter the variance
    correction.

    Always the normal approximation, with SciPy's continuity correction by
    default, where SciPy enumerates the exact distribution for a small
    untied sample. Same declared divergence as `mannwhitneyu`.

    `df` carries the number of pairs that survived the zero-dropping,
    which is what the approximation was taken at.

    At `gpu=True`, with both samples on a device, the differences, the
    zero-dropping (a compaction) and the ranking all run there, and only
    the positive rank sum, the tie term and the count come back.

    Parameters:
        A: The tensor type of the first sample, row-major and floating-point.
        B: The tensor type of the second sample, at `A`'s dtype.
        gpu: Whether the differences and ranking run on the samples'
            device; if either sample is not where `gpu` expects, both fall
            back to the host with a notice.

    Args:
        xs: The first sample of each pair, read flat.
        ys: The second sample of each pair, at the same length as `xs`.
        alternative: `"two-sided"`, `"less"` or `"greater"`, the tail
            the normal p-value is taken on.
        use_continuity: Whether to apply SciPy's `0.5` continuity
            correction to `W`.

    Returns:
        A `TestResult` with `W`, its asymptotic p-value and `df` set to the
        number of nonzero differences.

    Raises:
        If `alternative` is not one of the three names, if the lengths
        differ, if every difference is zero or every magnitude is tied, or
        on a fallback under the `"raise"` policy.
    """
    _check_alternative("wilcoxon", alternative)
    if _check_device[A, gpu](xs) and _check_device[B, gpu](ys):
        comptime if gpu:
            if xs.size() != ys.size():
                raise Error(
                    "wilcoxon: ",
                    xs.size(),
                    " and ",
                    ys.size(),
                    " are not paired lengths",
                )
            var packed = _signed_magnitudes_device(xs, ys)
            var kept = packed[0].size()
            if kept < 1:
                raise Error("wilcoxon: every pair is a zero difference")
            var sums = _rank_sum_device(packed[0], packed[1])
            return _wilcoxon_finish(
                sums[0], sums[1], kept, alternative, use_continuity
            )
    else:
        _notice[gpu]("wilcoxon")
    var a = _values(xs)
    var b = _values(ys)
    if len(a) != len(b):
        raise Error(
            "wilcoxon: ", len(a), " and ", len(b), " are not paired lengths"
        )

    var magnitudes = List[Float64]()
    var positive = List[Bool]()
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d == 0.0:
            continue
        magnitudes.append(d.__abs__())
        positive.append(d > 0.0)

    var n = len(magnitudes)
    if n < 1:
        raise Error("wilcoxon: every pair is a zero difference")

    # Average ranks of the magnitudes, and the tie groups for the variance.
    var order = List[Int](capacity=n)
    for i in range(n):
        order.append(i)
    for i in range(1, n):
        var key = order[i]
        var k = i - 1
        while k >= 0 and magnitudes[order[k]] > magnitudes[key]:
            order[k + 1] = order[k]
            k -= 1
        order[k + 1] = key

    var ranks = List[Float64](length=n, fill=0.0)
    var tie_term = 0.0
    var at = 0
    while at < n:
        var stop = at + 1
        while stop < n and magnitudes[order[stop]] == magnitudes[order[at]]:
            stop += 1
        var group = stop - at
        var average = (Float64(at + 1) + Float64(stop)) / 2.0
        for k in range(at, stop):
            ranks[order[k]] = average
        tie_term += Float64(group * group * group - group)
        at = stop

    var w_plus = 0.0
    for i in range(n):
        if positive[i]:
            w_plus += ranks[i]
    return _wilcoxon_finish(w_plus, tie_term, n, alternative, use_continuity)


def _chi2_tail(statistic: Float64, df: Float64) -> Float64:
    return Float64(chi2.sf[_P](_P(statistic), _P(df)).v)


def _mark_device[
    dtype: DType
](ctx: DeviceContext, n: Int, start: Int, stop: Int) raises -> Dynamic[
    dtype, 1
]:
    """A length-`n` device vector, one on `[start, stop)` and zero elsewhere:
    the membership flags `_rank_sum_device` sums ranks under."""
    var flags = Dynamic[dtype, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](n))
    )
    var fv = _flat_out(flags)

    @always_inline
    def mark[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var fv, var start, var stop}:
        var i = coord_to_index_list(coord)[0]
        fv.store[1](
            coord,
            Scalar[dtype](1) if i >= start and i < stop else Scalar[dtype](0),
        )

    elementwise[simd_width=1, target="gpu"](mark, Coord(n), ctx)
    return flags^


def kruskal[
    T: TensorLike, gpu: Bool = False
](*groups: T) raises -> TestResult where (
    T.dtype.is_floating_point() and is_row_major[T]
):
    """The Kruskal-Wallis `H` test that the groups share a median: the
    rank-sum analogue of the one-way ANOVA. `scipy.stats.kruskal(*samples)`.

    `H = 12 / (N (N+1)) sum R_i^2 / n_i - 3 (N+1)` over the ranks of the
    pooled sample, divided by SciPy's tie correction `1 - sum (t^3 - t) /
    (N^3 - N)`, against `chi2` on `k - 1` degrees of freedom. At
    `gpu=True` the pooled sample is ranked on the device, one rank sum per
    group (`_rank_sum_device`), and only those sums come back.

    Parameters:
        T: The tensor type of every group, row-major and floating-point.
        gpu: Whether the ranking runs on the groups' device.

    Args:
        groups: The `k` samples, each read flat.

    Returns:
        A `TestResult` with `H`, its upper-tail p-value and `df = k - 1`.

    Raises:
        If fewer than two groups are given, if every value is equal, or on a
        fallback under the `"raise"` policy.
    """
    var k = len(groups)
    if k < 2:
        raise Error("kruskal: at least two groups are needed")
    var sizes = List[Int](capacity=k)
    var total = 0
    for g in range(k):
        sizes.append(groups[g].size())
        total += groups[g].size()
    var rank_sums = List[Float64](capacity=k)
    var tie_term = 0.0
    var on_device = True
    comptime if gpu:
        for g in range(k):
            on_device = on_device and _check_device[T, gpu](groups[g])
        if on_device:
            var ctx = groups[0].context()
            var pooled = Dynamic[T.dtype, 1]._uninitialized(
                ctx, row_major(_dyn_shape[1](total))
            )
            var pv = _flat_out(pooled)
            var at = 0
            for g in range(k):
                var src = _flat_unchecked(groups[g])
                var base = at

                @always_inline
                def copy[
                    width: Int, alignment: Int = 1
                ](coord: Coord) {var src, var pv, var base}:
                    var i = coord_to_index_list(coord)[0]
                    pv.store[1](Coord(i + base), src[coord][0])

                elementwise[simd_width=1, target="gpu"](
                    copy, Coord(sizes[g]), ctx
                )
                at += sizes[g]
            ctx.synchronize()
            var offset = 0
            for g in range(k):
                var flags = _mark_device[T.dtype](
                    pooled.context(), total, offset, offset + sizes[g]
                )
                var sums = _rank_sum_device(pooled, flags)
                rank_sums.append(sums[0])
                tie_term = sums[1]
                offset += sizes[g]
            return _kruskal_finish(rank_sums, sizes, total, tie_term)
        _notice[gpu]("kruskal")
    var pooled = List[Float64](capacity=total)
    var owner = List[Int](capacity=total)
    for g in range(k):
        var values = _values(groups[g])
        for i in range(len(values)):
            pooled.append(values[i])
            owner.append(g)
    var order = List[Int](capacity=total)
    for i in range(total):
        order.append(i)

    def by_value(i: Int, j: Int) {imm} -> Bool:
        return pooled[i] < pooled[j] or (pooled[i] == pooled[j] and i < j)

    _sort(order, by_value)
    for _ in range(k):
        rank_sums.append(0.0)
    var i = 0
    while i < total:
        var j = i
        while j + 1 < total and pooled[order[j + 1]] == pooled[order[i]]:
            j += 1
        var size = Float64(j - i + 1)
        tie_term += size * size * size - size
        var rank = Float64(i + j + 2) / 2.0
        for m in range(i, j + 1):
            rank_sums[owner[order[m]]] += rank
        i = j + 1
    return _kruskal_finish(rank_sums, sizes, total, tie_term)


def _kruskal_finish(
    rank_sums: List[Float64], sizes: List[Int], total: Int, tie_term: Float64
) raises -> TestResult:
    var n = Float64(total)
    var h = 0.0
    for g in range(len(sizes)):
        h += rank_sums[g] * rank_sums[g] / Float64(sizes[g])
    h = 12.0 / (n * (n + 1.0)) * h - 3.0 * (n + 1.0)
    var correction = 1.0 - tie_term / (n * n * n - n)
    if correction == 0.0:
        raise Error("kruskal: all numbers are identical")
    h /= correction
    var df = Float64(len(sizes) - 1)
    return TestResult(h, _chi2_tail(h, df), df)


def levene[
    T: TensorLike, center: StaticString = "median", gpu: Bool = False
](*groups: T) raises -> TestResult where (
    T.dtype.is_floating_point() and is_row_major[T]
):
    """Levene's test that the groups share a variance, in the
    Brown-Forsythe form by default. `scipy.stats.levene(*samples, center)`.

    The one-way ANOVA `F` of the absolute deviations `|y - c_i|` from each
    group's `center` -- `"median"` (the default, Brown-Forsythe, robust to
    non-normality) or `"mean"` (Levene's original) -- against `f` on
    `k - 1` and `N - k` degrees of freedom. At `gpu=True` the centers, the
    deviations and the ANOVA sums are all device work.

    Parameters:
        T: The tensor type of every group, row-major and floating-point.
        center: `"median"` or `"mean"`.
        gpu: Whether to compute on the groups' device.

    Args:
        groups: The `k` samples, each read flat.

    Returns:
        A `TestResult` with the `W` statistic, its upper-tail p-value and
        the numerator degrees of freedom `k - 1`.

    Raises:
        If fewer than two groups are given, or on a fallback under the
        `"raise"` policy.
    """
    comptime assert (
        center == "median" or center == "mean"
    ), 'levene: center is "median" or "mean"'
    var k = len(groups)
    if k < 2:
        raise Error("levene: at least two groups are needed")
    var means = List[Float64](capacity=k)
    var sizes = List[Int](capacity=k)
    var grand = 0.0
    var total_n = 0
    var within = 0.0
    var on_device = True
    comptime if gpu:
        for g in range(k):
            on_device = on_device and _check_device[T, gpu](groups[g])
        if on_device:
            for g in range(k):
                var c: Scalar[T.dtype]
                comptime if center == "median":
                    c = _tmedian[gpu=True](groups[g])
                else:
                    c = Scalar[T.dtype](
                        _tsum[gpu=True](groups[g])
                        / Scalar[T.dtype](groups[g].size())
                    )
                var z = _tabs[gpu=True](_tsubtract[gpu=True](groups[g], c))
                var m = _moments_device(z)
                sizes.append(m[0])
                means.append(m[1])
                grand += m[1] * Float64(m[0])
                within += m[2]
                total_n += m[0]
            return _f_oneway_finish(means, sizes, grand, total_n, within)
        _notice[gpu]("levene")
    for g in range(k):
        var values = _values(groups[g])
        var c: Float64
        comptime if center == "median":
            c = _host_median(values)
        else:
            c = _mean(values)
        var z = List[Float64](capacity=len(values))
        for i in range(len(values)):
            z.append(abs(values[i] - c))
        var m = _mean(z)
        means.append(m)
        sizes.append(len(z))
        for i in range(len(z)):
            grand += z[i]
            within += (z[i] - m) * (z[i] - m)
        total_n += len(z)
    return _f_oneway_finish(means, sizes, grand, total_n, within)


def _host_median(values: List[Float64]) -> Float64:
    var sorted = values.copy()
    _sort(sorted)
    var n = len(sorted)
    if n % 2 == 1:
        return sorted[n // 2]
    return 0.5 * (sorted[n // 2 - 1] + sorted[n // 2])


def bartlett[
    T: TensorLike, gpu: Bool = False
](*groups: T) raises -> TestResult where (
    T.dtype.is_floating_point() and is_row_major[T]
):
    """Bartlett's test that normal groups share a variance.
    `scipy.stats.bartlett(*samples)`.

    `((N - k) ln s_p^2 - sum (n_i - 1) ln s_i^2) / C` with `s_p^2` the
    pooled variance and `C = 1 + (sum 1/(n_i - 1) - 1/(N - k)) / (3 (k-1))`,
    against `chi2` on `k - 1` degrees of freedom. Sensitive to
    non-normality, which is what `levene` is for. At `gpu=True` each
    group's variance is a device sum (`_moments_device`).

    Parameters:
        T: The tensor type of every group, row-major and floating-point.
        gpu: Whether the per-group sums run on the groups' device.

    Args:
        groups: The `k` samples, each read flat, each at least two long.

    Returns:
        A `TestResult` with the statistic, its upper-tail p-value and
        `df = k - 1`.

    Raises:
        If fewer than two groups are given, or on a fallback under the
        `"raise"` policy.
    """
    var k = len(groups)
    if k < 2:
        raise Error("bartlett: at least two groups are needed")
    var sizes = List[Int](capacity=k)
    var variances = List[Float64](capacity=k)
    var on_device = True
    comptime if gpu:
        for g in range(k):
            on_device = on_device and _check_device[T, gpu](groups[g])
    if gpu and on_device:
        comptime if gpu:
            for g in range(k):
                var m = _moments_device(groups[g])
                sizes.append(m[0])
                variances.append(m[2] / Float64(m[0] - 1))
    else:
        comptime if gpu:
            _notice[gpu]("bartlett")
        for g in range(k):
            var values = _values(groups[g])
            sizes.append(len(values))
            variances.append(_variance(values, 1))
    var total = 0
    for g in range(k):
        total += sizes[g]
    var pooled = 0.0
    var log_sum = 0.0
    var inverse_sum = 0.0
    for g in range(k):
        var dof = Float64(sizes[g] - 1)
        pooled += dof * variances[g]
        log_sum += dof * _log(variances[g])
        inverse_sum += 1.0 / dof
    var nk = Float64(total - k)
    pooled /= nk
    var numer = nk * _log(pooled) - log_sum
    var denom = 1.0 + (inverse_sum - 1.0 / nk) / (3.0 * Float64(k - 1))
    var statistic = numer / denom
    var df = Float64(k - 1)
    return TestResult(statistic, _chi2_tail(statistic, df), df)


def _two_sided_normal(z: Float64, alternative: StaticString) -> Float64:
    """The standard normal tail SciPy's normality tests read."""
    if alternative == "less":
        return Float64(norm.cdf[_P](_P(z), _P(0.0), _P(1.0)).v)
    if alternative == "greater":
        return Float64(norm.sf[_P](_P(z), _P(0.0), _P(1.0)).v)
    return 2.0 * Float64(norm.sf[_P](_P(abs(z)), _P(0.0), _P(1.0)).v)


def _skew_z(b2: Float64, n: Float64) -> Float64:
    """D'Agostino's transform of the sample skewness to a standard normal,
    SciPy's `skewtest` formula."""
    var y = b2 * _sqrt(((n + 1.0) * (n + 3.0)) / (6.0 * (n - 2.0)))
    var beta2 = (
        3.0
        * (n * n + 27.0 * n - 70.0)
        * (n + 1.0)
        * (n + 3.0)
        / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0))
    )
    var w2 = -1.0 + _sqrt(2.0 * (beta2 - 1.0))
    var delta = 1.0 / _sqrt(0.5 * _log(w2))
    var alpha = _sqrt(2.0 / (w2 - 1.0))
    if y == 0.0:
        y = 1.0
    var r = y / alpha
    return delta * _log(r + _sqrt(r * r + 1.0))


def _kurtosis_z(b2: Float64, n: Float64) -> Float64:
    """Anscombe and Glynn's transform of the sample (Pearson) kurtosis to a
    standard normal, SciPy's `kurtosistest` formula."""
    var e = 3.0 * (n - 1.0) / (n + 1.0)
    var varb2 = (
        24.0
        * n
        * (n - 2.0)
        * (n - 3.0)
        / ((n + 1.0) * (n + 1.0) * (n + 3.0) * (n + 5.0))
    )
    var x = (b2 - e) / _sqrt(varb2)
    var sqrtbeta1 = (
        6.0
        * (n * n - 5.0 * n + 2.0)
        / ((n + 7.0) * (n + 9.0))
        * _sqrt((6.0 * (n + 3.0) * (n + 5.0)) / (n * (n - 2.0) * (n - 3.0)))
    )
    var a = 6.0 + 8.0 / sqrtbeta1 * (
        2.0 / sqrtbeta1 + _sqrt(1.0 + 4.0 / (sqrtbeta1 * sqrtbeta1))
    )
    var term1 = 1.0 - 2.0 / (9.0 * a)
    var denom = 1.0 + x * _sqrt(2.0 / (a - 4.0))
    var magnitude = ((1.0 - 2.0 / a) / abs(denom)) ** (1.0 / 3.0)
    var term2 = magnitude if denom > 0 else -magnitude
    return (term1 - term2) / _sqrt(2.0 / (9.0 * a))


def skewtest[
    T: TensorLike, gpu: Bool = False
](a: T, alternative: StaticString = "two-sided") raises -> TestResult where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Whether the skewness differs from a normal's. `scipy.stats.skewtest`.

    D'Agostino's transform of the biased sample skewness to a standard
    normal `z`; needs `n >= 8`. The skewness is `numax.stats.skew`, device
    moments at `gpu=True`.

    Parameters:
        T: The tensor type of `a`, row-major and floating-point.
        gpu: Whether the moments run on `a`'s device.

    Args:
        a: The sample, read flat.
        alternative: `"two-sided"`, `"less"` or `"greater"`.

    Returns:
        A `TestResult` with `z`, its normal p-value and `df = 0`.

    Raises:
        If `a` has fewer than 8 elements or `alternative` is not a known
        name.
    """
    _check_alternative("skewtest", alternative)
    var n = Float64(a.size())
    if n < 8:
        raise Error("skewtest: the sample needs at least 8 values")
    var z = _skew_z(_skew[gpu=gpu](a), n)
    return TestResult(z, _two_sided_normal(z, alternative), 0.0)


def kurtosistest[
    T: TensorLike, gpu: Bool = False
](a: T, alternative: StaticString = "two-sided") raises -> TestResult where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Whether the kurtosis differs from a normal's.
    `scipy.stats.kurtosistest`.

    Anscombe and Glynn's transform of the biased Pearson kurtosis to a
    standard normal `z`; needs `n >= 5`, and SciPy warns below 20.

    Parameters:
        T: The tensor type of `a`, row-major and floating-point.
        gpu: Whether the moments run on `a`'s device.

    Args:
        a: The sample, read flat.
        alternative: `"two-sided"`, `"less"` or `"greater"`.

    Returns:
        A `TestResult` with `z`, its normal p-value and `df = 0`.

    Raises:
        If `a` has fewer than 5 elements or `alternative` is not a known
        name.
    """
    _check_alternative("kurtosistest", alternative)
    var n = Float64(a.size())
    if n < 5:
        raise Error("kurtosistest: the sample needs at least 5 values")
    var z = _kurtosis_z(_kurtosis[gpu=gpu](a, False), n)
    return TestResult(z, _two_sided_normal(z, alternative), 0.0)


def normaltest[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> TestResult where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """D'Agostino and Pearson's omnibus test of normality.
    `scipy.stats.normaltest`.

    `K^2 = z_skew^2 + z_kurtosis^2`, the two transforms `skewtest` and
    `kurtosistest` make, against `chi2` on 2 degrees of freedom.

    Parameters:
        T: The tensor type of `a`, row-major and floating-point.
        gpu: Whether the moments run on `a`'s device.

    Args:
        a: The sample, read flat, at least 8 long.

    Returns:
        A `TestResult` with `K^2`, its upper-tail p-value and `df = 2`.

    Raises:
        If `a` has fewer than 8 elements.
    """
    var n = Float64(a.size())
    if n < 8:
        raise Error("normaltest: the sample needs at least 8 values")
    var s = _skew_z(_skew[gpu=gpu](a), n)
    var k = _kurtosis_z(_kurtosis[gpu=gpu](a, False), n)
    var statistic = s * s + k * k
    return TestResult(statistic, _chi2_tail(statistic, 2.0), 2.0)


def jarque_bera[
    T: TensorLike, gpu: Bool = False
](x: T) raises -> TestResult where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The Jarque-Bera test of normality. `scipy.stats.jarque_bera`.

    `JB = n / 6 (S^2 + (K - 3)^2 / 4)` from the biased skewness `S` and
    Pearson kurtosis `K`, against `chi2` on 2 degrees of freedom -- the
    large-sample test, less accurate than `normaltest` for small `n`.

    Parameters:
        T: The tensor type of `x`, row-major and floating-point.
        gpu: Whether the moments run on `x`'s device.

    Args:
        x: The sample, read flat.

    Returns:
        A `TestResult` with `JB`, its upper-tail p-value and `df = 2`.

    Raises:
        If `x` is empty.
    """
    var n = Float64(x.size())
    if n == 0:
        raise Error("jarque_bera: the sample is empty")
    var s = _skew[gpu=gpu](x)
    var k = _kurtosis[gpu=gpu](x, False)
    var statistic = n / 6.0 * (s * s + (k - 3.0) * (k - 3.0) / 4.0)
    return TestResult(statistic, _chi2_tail(statistic, 2.0), 2.0)


def _lchoose(n: Float64, k: Float64) -> Float64:
    return _lgamma(n + 1.0) - _lgamma(k + 1.0) - _lgamma(n - k + 1.0)


@fieldwise_init
struct _Hypergeom(Copyable):
    """A hypergeometric law with `M` balls, `n` good, `N` drawn, evaluated
    on the host in `Float64` by direct sums of its PMF."""

    var M: Float64
    var n: Float64
    var N: Float64

    def lo(self) -> Int:
        return max(0, Int(self.N - (self.M - self.n)))

    def hi(self) -> Int:
        return Int(min(self.n, self.N))

    def pmf(self, x: Int) -> Float64:
        if x < self.lo() or x > self.hi():
            return 0.0
        var xf = Float64(x)
        return _exp(
            _lchoose(self.n, xf)
            + _lchoose(self.M - self.n, self.N - xf)
            - _lchoose(self.M, self.N)
        )

    def cdf(self, x: Int) -> Float64:
        var total = 0.0
        for j in range(self.lo(), min(x, self.hi()) + 1):
            total += self.pmf(j)
        return min(total, 1.0)

    def sf(self, x: Int) -> Float64:
        var total = 0.0
        for j in range(max(x + 1, self.lo()), self.hi() + 1):
            total += self.pmf(j)
        return min(total, 1.0)


def _search_up[
    F: def(Int) raises capturing -> Float64
](d: Float64, lo_in: Int, hi_in: Int) raises -> Int:
    """SciPy's `_binary_search_for_binom_tst` for an ascending `F` on
    `[lo, hi]`: the `i` with `F(i) <= d < F(i + 1)`."""
    var lo = lo_in
    var hi = hi_in
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        var midval = F(mid)
        if midval < d:
            lo = mid + 1
        elif midval > d:
            hi = mid - 1
        else:
            lo = mid
            hi = mid
    return lo if F(lo) <= d else lo - 1


def fisher_exact[
    T: TensorLike
](
    table: T, alternative: StaticString = "two-sided"
) raises -> TestResult where (
    T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
    and dim[T, 0] == 2
    and dim[T, 1] == 2
):
    """Fisher's exact test of independence on a 2x2 contingency table.
    `scipy.stats.fisher_exact(table, alternative)`.

    The sample odds ratio `(a d) / (b c)` and the exact hypergeometric
    p-value, SciPy's construction step for step: the one-sided tails are
    the CDF and survival function at the observed count, and the
    two-sided p-value adds the far tail from the other side of the mode,
    found by SciPy's binary search at a relative tolerance of `1e-14`. A
    table with a zero margin gives NaN and 1, as SciPy's does. Host-side:
    four counts in, sums over at most `min(row, column)` terms.

    Parameters:
        T: The tensor type of `table`, `(2, 2)`, counts in any dtype.

    Args:
        table: The contingency table.
        alternative: `"two-sided"`, `"less"` or `"greater"`.

    Returns:
        A `TestResult` with the odds ratio, the p-value and `df = 0`.

    Raises:
        If a count is negative or `alternative` is not a known name.
    """
    _check_alternative("fisher_exact", alternative)
    var c = table.to_host()
    var a = Int(c[0])
    var b = Int(c[1])
    var cc = Int(c[2])
    var d = Int(c[3])
    if a < 0 or b < 0 or cc < 0 or d < 0:
        raise Error("fisher_exact: every count must be nonnegative")
    if a + b == 0 or cc + d == 0 or a + cc == 0 or b + d == 0:
        return TestResult(_nan_f64(), 1.0, 0.0)
    var odds = (
        Float64(a * d) / Float64(cc * b) if cc > 0 and b > 0 else _inf_f64()
    )
    var n1 = a + b
    var n2 = cc + d
    var n = a + cc
    var law = _Hypergeom(Float64(n1 + n2), Float64(n1), Float64(n))
    var pvalue: Float64
    if alternative == "less":
        pvalue = law.cdf(a)
    elif alternative == "greater":
        var other = _Hypergeom(Float64(n1 + n2), Float64(n1), Float64(b + d))
        pvalue = other.cdf(b)
    else:
        var mode = Int(Float64((n + 1) * (n1 + 1)) / Float64(n1 + n2 + 2))
        var pexact = law.pmf(a)
        var pmode = law.pmf(mode)
        var gamma = 1.0 + 1e-14
        if abs(pexact - pmode) / max(pexact, pmode) <= 1e-14:
            return TestResult(odds, 1.0, 0.0)
        if a < mode:
            var plower = law.cdf(a)
            if law.pmf(n) > pexact * gamma:
                return TestResult(odds, min(plower, 1.0), 0.0)

            @__parameter
            def neg(x: Int) raises -> Float64:
                return -law.pmf(x)

            var guess = _search_up[neg](-pexact * gamma, mode, n)
            pvalue = plower + law.sf(guess)
        else:
            var pupper = law.sf(a - 1)
            if law.pmf(0) > pexact * gamma:
                return TestResult(odds, min(pupper, 1.0), 0.0)

            @__parameter
            def pos(x: Int) raises -> Float64:
                return law.pmf(x)

            var guess = _search_up[pos](pexact * gamma, 0, mode)
            pvalue = pupper + law.cdf(guess)
    return TestResult(odds, min(pvalue, 1.0), 0.0)


def _binom_pmf(x: Int, n: Int, p: Float64) -> Float64:
    if x < 0 or x > n:
        return 0.0
    var xf = Float64(x)
    var nf = Float64(n)
    if p == 0.0:
        return 1.0 if x == 0 else 0.0
    if p == 1.0:
        return 1.0 if x == n else 0.0
    return _exp(_lchoose(nf, xf) + xf * _log(p) + (nf - xf) * _log(1.0 - p))


def _binom_cdf(x: Int, n: Int, p: Float64) -> Float64:
    if x < 0:
        return 0.0
    if x >= n:
        return 1.0
    return Float64(binom.cdf[_P](_P(Float64(x)), _P(Float64(n)), _P(p)).v)


def _binom_sf(x: Int, n: Int, p: Float64) -> Float64:
    if x < 0:
        return 1.0
    if x >= n:
        return 0.0
    return Float64(binom.sf[_P](_P(Float64(x)), _P(Float64(n)), _P(p)).v)


def binomtest(
    k: Int, n: Int, p: Float64 = 0.5, alternative: StaticString = "two-sided"
) raises -> TestResult:
    """The exact binomial test that `k` successes in `n` trials came from
    success probability `p`. `scipy.stats.binomtest(k, n, p, alternative)`.

    SciPy's p-value: the lower or upper tail for a one-sided test, and for
    the two-sided test every outcome at most as likely as `k` (to a
    relative `1e-7`), the far side found by SciPy's binary search from the
    mean. The statistic is the sample proportion `k / n`.

    Args:
        k: The number of successes, in `[0, n]`.
        n: The number of trials, at least 1.
        p: The hypothesized success probability, in `[0, 1]`.
        alternative: `"two-sided"`, `"less"` or `"greater"`.

    Returns:
        A `TestResult` with `k / n`, the p-value and `df = 0`.

    Raises:
        If `k`, `n` or `p` is out of range, or `alternative` is not a known
        name.
    """
    _check_alternative("binomtest", alternative)
    if n < 1 or k < 0 or k > n or p < 0.0 or p > 1.0:
        raise Error("binomtest: need n >= 1, 0 <= k <= n and 0 <= p <= 1")
    var statistic = Float64(k) / Float64(n)
    var pvalue: Float64
    if alternative == "less":
        pvalue = _binom_cdf(k, n, p)
    elif alternative == "greater":
        pvalue = _binom_sf(k - 1, n, p)
    else:
        var d = _binom_pmf(k, n, p)
        var rerr = 1.0 + 1e-7
        var pn = p * Float64(n)
        if Float64(k) < pn:

            @__parameter
            def neg(x: Int) raises -> Float64:
                return -_binom_pmf(x, n, p)

            var ix = _search_up[neg](-d * rerr, Int(_ceil(pn)), n)
            var y = n - ix + (1 if d * rerr == _binom_pmf(ix, n, p) else 0)
            pvalue = _binom_cdf(k, n, p) + _binom_sf(n - y, n, p)
        else:

            @__parameter
            def pos(x: Int) raises -> Float64:
                return _binom_pmf(x, n, p)

            var ix = _search_up[pos](d * rerr, 0, Int(_floor(pn)))
            var y = ix + 1
            pvalue = _binom_cdf(y - 1, n, p) + _binom_sf(k - 1, n, p)
        pvalue = min(1.0, pvalue)
    return TestResult(statistic, pvalue, 0.0)


struct Chi2ContingencyResult[dtype: DType, rows: Int, cols: Int](Movable):
    """What `chi2_contingency` returns, SciPy's four fields."""

    var statistic: Float64
    """Pearson's chi-squared statistic, Yates-corrected where it applies."""
    var pvalue: Float64
    """Its upper-tail p-value."""
    var dof: Int
    """The degrees of freedom, `(rows - 1) (cols - 1)`."""
    var expected_freq: Static[Self.dtype, Self.rows, Self.cols]
    """The expected counts under independence."""

    def __init__(
        out self,
        statistic: Float64,
        pvalue: Float64,
        dof: Int,
        var expected_freq: Static[Self.dtype, Self.rows, Self.cols],
    ):
        """Build from the four fields.

        Args:
            statistic: The statistic.
            pvalue: The p-value.
            dof: The degrees of freedom.
            expected_freq: The expected counts.
        """
        self.statistic = statistic
        self.pvalue = pvalue
        self.dof = dof
        self.expected_freq = expected_freq^


def chi2_contingency[
    T: TensorLike, gpu: Bool = False
](observed: T, correction: Bool = True) raises -> Chi2ContingencyResult[
    T.dtype, dim[T, 0], dim[T, 1]
] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
    and is_row_major[T]
):
    """Pearson's chi-squared test of independence on an `r x c`
    contingency table. `scipy.stats.chi2_contingency(observed, correction)`.

    The expected counts are the outer product of the margins over the
    total; with one degree of freedom and `correction`, each observed
    count moves toward its expectation by `min(0.5, |e - o|)` first,
    Yates's correction as SciPy applies it. At `gpu=True` the margins are
    device axis sums and the expected table and the terms one launch; the
    statistic comes back as one sum.

    Parameters:
        T: The tensor type of `observed`, rank 2, floating-point counts.
        gpu: Whether to compute on `observed`'s device.

    Args:
        observed: The table of observed counts.
        correction: Apply Yates's correction when there is one degree of
            freedom.

    Returns:
        A `Chi2ContingencyResult` with the statistic, p-value, degrees of
        freedom and expected counts.

    Raises:
        If an expected count is zero, or a device operation fails.
    """
    comptime dtype = T.dtype
    comptime r = dim[T, 0]
    comptime c = dim[T, 1]
    var dof = (r - 1) * (c - 1)
    var ctx = observed.context()
    var yates = correction and dof == 1
    if _check_device[T, gpu](observed):
        comptime if gpu:
            # The margins, one lane per row and one per column.
            var row = Static[dtype, r]._uninitialized(ctx)
            var col = Static[dtype, c]._uninitialized(ctx)
            var src = observed.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
            var rs = row.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
            var cs = col.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

            @always_inline
            def margins[
                width: Int, alignment: Int = 1
            ](coord: Coord) {var src, var rs, var cs}:
                var f = coord_to_index_list(coord)[0]
                var acc = Scalar[dtype](0)
                if f < r:
                    for j in range(c):
                        acc += rebind[Scalar[dtype]](
                            src[unsafe_offset=f * c + j]
                        )
                    rs[unsafe_offset=f] = acc
                else:
                    var j = f - r
                    for i in range(r):
                        acc += rebind[Scalar[dtype]](
                            src[unsafe_offset=i * c + j]
                        )
                    cs[unsafe_offset=j] = acc

            elementwise[simd_width=1, target="gpu"](margins, Coord(r + c), ctx)
            ctx.synchronize()
            var total = _tsum[gpu=True](row)
            var expected = Static[dtype, r, c]._uninitialized(ctx)
            var terms = Static[dtype, r, c]._uninitialized(ctx)
            var op = observed.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
            var rp = row.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
            var cp = col.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
            var ep = expected.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
            var tp = terms.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

            @always_inline
            def cell[
                width: Int, alignment: Int = 1
            ](coord: Coord) {
                var op, var rp, var cp, var ep, var tp, var total, var yates
            }:
                var f = coord_to_index_list(coord)[0]
                var e = (
                    rebind[Scalar[dtype]](rp[unsafe_offset=f // c])
                    * rebind[Scalar[dtype]](cp[unsafe_offset=f % c])
                    / rebind[Scalar[dtype]](total)
                )
                var o = rebind[Scalar[dtype]](op[unsafe_offset=f])
                if yates:
                    var diff = e - o
                    var mag = min(Scalar[dtype](0.5), abs(diff))
                    o = o + (mag if diff > 0 else -mag)
                ep[unsafe_offset=f] = e
                tp[unsafe_offset=f] = (o - e) * (o - e) / e

            elementwise[simd_width=1, target="gpu"](cell, Coord(r * c), ctx)
            ctx.synchronize()
            var statistic = Float64(_tsum[gpu=True](terms))
            _ = row^
            _ = col^
            return Chi2ContingencyResult(
                statistic, _chi2_tail(statistic, Float64(dof)), dof, expected^
            )
    else:
        _notice[gpu]("chi2_contingency")
    var values = _values(observed)
    var rows = List[Float64](length=r, fill=0.0)
    var cols = List[Float64](length=c, fill=0.0)
    var total = 0.0
    for i in range(r):
        for j in range(c):
            rows[i] += values[i * c + j]
            cols[j] += values[i * c + j]
            total += values[i * c + j]
    var expected = List[Scalar[dtype]](capacity=r * c)
    var statistic = 0.0
    for i in range(r):
        for j in range(c):
            var e = rows[i] * cols[j] / total
            if e == 0.0:
                raise Error("chi2_contingency: an expected count is zero")
            var o = values[i * c + j]
            if yates:
                var diff = e - o
                var mag = min(0.5, abs(diff))
                o = o + (mag if diff > 0 else -mag)
            statistic += (o - e) * (o - e) / e
            expected.append(Scalar[dtype](e))
    return Chi2ContingencyResult(
        statistic,
        _chi2_tail(statistic, Float64(dof)),
        dof,
        Static[dtype, r, c](expected^, ctx),
    )


def _nan_f64() -> Float64:
    return _inf_f64() - _inf_f64()


def _inf_f64() -> Float64:
    return Float64.MAX * 2.0
