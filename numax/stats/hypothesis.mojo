"""Hypothesis tests over `numax.core.tensor.Tensor`: the three `t` tests,
`chisquare`, `ks_1samp`, `f_oneway` and `mannwhitneyu`, each returning
SciPy's statistic and p-value.

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
from std.math import exp as _exp, sqrt as _sqrt

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
from .distributions import chi2, f, norm, t
from .statistics import sum as _tsum

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
