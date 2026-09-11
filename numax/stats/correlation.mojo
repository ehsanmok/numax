"""Correlation over `numax.core.array.Tensor`: `cov`, `corrcoef`,
`pearsonr`, `spearmanr`, `kendalltau`, `linregress`, `rankdata` and
`zscore`, with NumPy's and SciPy's conventions and SciPy's p-values.

**Tier 2, host-side**, in `Float64`. Every routine here is a handful of
sums over one or two vectors -- or, for the rank correlations, a sort and
a pair count -- followed by a tail probability of Student's `t` or the
normal, and the p-value is the part that decides the placement: it comes
from `numax.stats.t.sf` and `numax.stats.norm.sf`, scalar `FloatLike`
kernels evaluated once. The data comes down once and the answer is a few
scalars or a small matrix, so a device pass would move more than it
computed. The one exception in spirit, `zscore`, returns a tensor the
shape of its input and is host-side only because the whole-tensor
`mean`/`stddev` it standardizes by are.

## Conventions

`cov` and `corrcoef` are NumPy's: rows are variables, columns are
observations, `ddof = 1` unless `bias` says otherwise. `pearsonr`'s
p-value is the two-sided `t` test with `n - 2` degrees of freedom;
`spearmanr` is `pearsonr` on average ranks, with the same test;
`kendalltau` is tau-b with tie corrections and SciPy's asymptotic normal
p-value -- always the asymptotic one, where SciPy switches to an exact
enumeration for small untied samples, so a tiny tie-free sample's p-value
differs from SciPy's by the approximation's error and nothing else.
`linregress` is SciPy's, standard errors included. `rankdata`'s five
`method`s are SciPy's, `"average"` by default.

## The MAX gate

Nothing: MAX has no covariance, correlation or ranking. **Extend.**
"""

from std.builtin.sort import sort as _sort
from std.math import sqrt as _sqrt

from layout.tile_layout import TensorLayout

from ..core.array import Static, Tensor
from ..core.plain import Plain
from .distributions import norm, t

comptime _P = Plain[DType.float64]


def _as_float64[dtype: DType](values: List[Scalar[dtype]]) -> List[Float64]:
    var out = List[Float64](capacity=len(values))
    for i in range(len(values)):
        out.append(Float64(values[i]))
    return out^


def _mean(values: List[Float64]) -> Float64:
    var total = 0.0
    for i in range(len(values)):
        total += values[i]
    return total / Float64(len(values))


def _covariance(
    xs: List[Float64], ys: List[Float64], ddof: Int
) raises -> Float64:
    """`sum((x - mx)(y - my)) / (n - ddof)`."""
    var n = len(xs)
    if n - ddof <= 0:
        raise Error("cov: not enough observations for ddof ", ddof)
    var mx = _mean(xs)
    var my = _mean(ys)
    var total = 0.0
    for i in range(n):
        total += (xs[i] - mx) * (ys[i] - my)
    return total / Float64(n - ddof)


def _t_two_sided(statistic: Float64, df: Float64) -> Float64:
    """`2 * P(T > |statistic|)` on `df` degrees of freedom."""
    var tail = t.sf[_P](_P(abs(statistic)), _P(df))
    return 2.0 * Float64(tail.v)


def _normal_two_sided(z: Float64) -> Float64:
    var tail = norm.sf[_P](_P(abs(z)), _P(0.0), _P(1.0))
    return 2.0 * Float64(tail.v)


def cov[
    dtype: DType, n: Int
](
    mut x: Static[dtype, n],
    mut y: Static[dtype, n],
    bias: Bool = False,
    ddof: Optional[Int] = None,
) raises -> Static[dtype, 2, 2] where (dtype.is_floating_point() and n > 1):
    """The covariance matrix of two variables, `[[var x, cov], [cov, var
    y]]`. `numpy.cov(x, y, bias, ddof)`: `ddof = 1` by default, `0` with
    `bias`, or as given."""
    var dof = ddof.value() if ddof else (0 if bias else 1)
    var xs = _as_float64(x.to_host())
    var ys = _as_float64(y.to_host())
    var values = List[Scalar[dtype]](capacity=4)
    values.append(Scalar[dtype](_covariance(xs, xs, dof)))
    var cross = Scalar[dtype](_covariance(xs, ys, dof))
    values.append(cross)
    values.append(cross)
    values.append(Scalar[dtype](_covariance(ys, ys, dof)))
    return Static[dtype, 2, 2](x.context(), values^)


def cov[
    dtype: DType, rows: Int, n: Int
](
    mut m: Static[dtype, rows, n],
    bias: Bool = False,
    ddof: Optional[Int] = None,
) raises -> Static[dtype, rows, rows] where (
    dtype.is_floating_point() and rows > 0 and n > 1
):
    """The covariance matrix of `rows` variables observed `n` times each,
    one variable per row. `numpy.cov(m)` with its default `rowvar=True`."""
    var dof = ddof.value() if ddof else (0 if bias else 1)
    var host = _as_float64(m.to_host())
    var series = List[List[Float64]]()
    for r in range(rows):
        var row = List[Float64](capacity=n)
        for c in range(n):
            row.append(host[r * n + c])
        series.append(row^)
    var values = List[Scalar[dtype]](length=rows * rows, fill=0)
    for a in range(rows):
        for b in range(a, rows):
            var c = Scalar[dtype](_covariance(series[a], series[b], dof))
            values[a * rows + b] = c
            values[b * rows + a] = c
    return Static[dtype, rows, rows](m.context(), values^)


def corrcoef[
    dtype: DType, n: Int
](mut x: Static[dtype, n], mut y: Static[dtype, n]) raises -> Static[
    dtype, 2, 2
] where (dtype.is_floating_point() and n > 1):
    """The Pearson correlation matrix of two variables, ones on the
    diagonal. `numpy.corrcoef(x, y)`."""
    var xs = _as_float64(x.to_host())
    var ys = _as_float64(y.to_host())
    var r = _covariance(xs, ys, 0) / _sqrt(
        _covariance(xs, xs, 0) * _covariance(ys, ys, 0)
    )
    var values = List[Scalar[dtype]](capacity=4)
    values.append(Scalar[dtype](1))
    values.append(Scalar[dtype](r))
    values.append(Scalar[dtype](r))
    values.append(Scalar[dtype](1))
    return Static[dtype, 2, 2](x.context(), values^)


def corrcoef[
    dtype: DType, rows: Int, n: Int
](mut m: Static[dtype, rows, n]) raises -> Static[dtype, rows, rows] where (
    dtype.is_floating_point() and rows > 0 and n > 1
):
    """The Pearson correlation matrix of `rows` variables, one per row.
    `numpy.corrcoef(m)`."""
    var host = _as_float64(m.to_host())
    var series = List[List[Float64]]()
    for r in range(rows):
        var row = List[Float64](capacity=n)
        for c in range(n):
            row.append(host[r * n + c])
        series.append(row^)
    var scale = List[Float64](capacity=rows)
    for r in range(rows):
        scale.append(_sqrt(_covariance(series[r], series[r], 0)))
    var values = List[Scalar[dtype]](length=rows * rows, fill=0)
    for a in range(rows):
        for b in range(a, rows):
            var r = 1.0 if a == b else _covariance(series[a], series[b], 0) / (
                scale[a] * scale[b]
            )
            values[a * rows + b] = Scalar[dtype](r)
            values[b * rows + a] = Scalar[dtype](r)
    return Static[dtype, rows, rows](m.context(), values^)


@fieldwise_init
struct CorrelationResult(Copyable, Movable):
    """What `pearsonr`, `spearmanr` and `kendalltau` return: the
    correlation `statistic` and the two-sided `pvalue` of the test that it
    is zero, SciPy's result shape."""

    var statistic: Float64
    var pvalue: Float64


def _pearson(xs: List[Float64], ys: List[Float64]) raises -> CorrelationResult:
    var n = len(xs)
    if n < 3:
        raise Error("pearsonr: at least three observations are needed")
    var r = _covariance(xs, ys, 0) / _sqrt(
        _covariance(xs, xs, 0) * _covariance(ys, ys, 0)
    )
    r = min(1.0, max(-1.0, r))
    var df = Float64(n - 2)
    if abs(r) == 1.0:
        return CorrelationResult(r, 0.0)
    var statistic = r * _sqrt(df / (1.0 - r * r))
    return CorrelationResult(r, _t_two_sided(statistic, df))


def pearsonr[
    dtype: DType, n: Int
](
    mut x: Static[dtype, n], mut y: Static[dtype, n]
) raises -> CorrelationResult where (dtype.is_floating_point() and n > 2):
    """Pearson's `r` and the two-sided p-value of `r = 0`.
    `scipy.stats.pearsonr(x, y)`.

    The p-value is the `t` test on `n - 2` degrees of freedom, `t = r
    sqrt((n - 2) / (1 - r^2))`, which is the same number SciPy's beta form
    gives.
    """
    return _pearson(_as_float64(x.to_host()), _as_float64(y.to_host()))


def _ranks(values: List[Float64], method: StaticString) raises -> List[Float64]:
    """`scipy.stats.rankdata` on the host: one-based ranks under the five
    tie methods."""
    var n = len(values)
    var order = List[Int](capacity=n)
    for i in range(n):
        order.append(i)

    @parameter
    def by_value(a: Int, b: Int) -> Bool:
        return values[a] < values[b] or (values[a] == values[b] and a < b)

    _sort[by_value](order)
    var ranks = List[Float64](length=n, fill=0.0)
    var dense = 0
    var i = 0
    while i < n:
        var j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        dense += 1
        for k in range(i, j + 1):
            var rank: Float64
            if method == "average":
                rank = Float64(i + j + 2) / 2.0
            elif method == "min":
                rank = Float64(i + 1)
            elif method == "max":
                rank = Float64(j + 1)
            elif method == "dense":
                rank = Float64(dense)
            elif method == "ordinal":
                rank = Float64(k + 1)
            else:
                raise Error(
                    "rankdata: unknown method '",
                    method,
                    "'; expected average, min, max, dense or ordinal",
                )
            ranks[order[k]] = rank
        i = j + 1
    return ranks^


def rankdata[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType], method: StaticString = "average"
) raises -> Tensor[dtype, LayoutType] where dtype.is_floating_point():
    """The one-based rank of every element, ties resolved by `method`:
    `"average"` (the default), `"min"`, `"max"`, `"dense"` or
    `"ordinal"`. `scipy.stats.rankdata(a, method)`, the same shape back."""
    var ranks = _ranks(_as_float64(xs.to_host()), method)
    var values = List[Scalar[dtype]](capacity=len(ranks))
    for i in range(len(ranks)):
        values.append(Scalar[dtype](ranks[i]))
    return Tensor[dtype, LayoutType](xs.context(), xs.layout, values^)


def spearmanr[
    dtype: DType, n: Int
](
    mut x: Static[dtype, n], mut y: Static[dtype, n]
) raises -> CorrelationResult where (dtype.is_floating_point() and n > 2):
    """Spearman's rank correlation and its two-sided p-value:
    `pearsonr` on the average ranks, with the same `t` test on `n - 2`
    degrees of freedom SciPy uses. `scipy.stats.spearmanr(x, y)`."""
    return _pearson(
        _ranks(_as_float64(x.to_host()), "average"),
        _ranks(_as_float64(y.to_host()), "average"),
    )


def kendalltau[
    dtype: DType, n: Int
](
    mut x: Static[dtype, n], mut y: Static[dtype, n]
) raises -> CorrelationResult where (dtype.is_floating_point() and n > 1):
    """Kendall's tau-b and its two-sided p-value.
    `scipy.stats.kendalltau(x, y, method="asymptotic")`.

    Tau-b counts concordant less discordant pairs over the geometric mean
    of the pairs untied in each variable; the p-value is the normal
    approximation with SciPy's tie-corrected variance. SciPy switches to an
    exact enumeration for a small sample with no ties, which this does not
    do, so there and only there its p-value differs by the approximation's
    error. `O(n^2)` in the pair count, host-side.
    """
    var xs = _as_float64(x.to_host())
    var ys = _as_float64(y.to_host())
    var concordant = 0
    var discordant = 0
    var xtie = 0
    var ytie = 0
    var both = 0
    for i in range(n):
        for j in range(i + 1, n):
            var dx = xs[i] - xs[j]
            var dy = ys[i] - ys[j]
            if dx == 0 and dy == 0:
                both += 1
            elif dx == 0:
                xtie += 1
            elif dy == 0:
                ytie += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    var total = n * (n - 1) // 2
    var x_tied = xtie + both
    var y_tied = ytie + both
    if x_tied == total or y_tied == total:
        raise Error("kendalltau: one variable is constant")
    var tau = Float64(concordant - discordant) / (
        _sqrt(Float64(total - x_tied)) * _sqrt(Float64(total - y_tied))
    )
    tau = min(1.0, max(-1.0, tau))

    # SciPy's variance of `con - dis` with tie corrections: the group sizes
    # of equal values in each variable enter through three sums.
    var stats_x = _tie_sums(xs)
    var stats_y = _tie_sums(ys)
    var m = Float64(n) * Float64(n - 1)
    var variance = (
        (m * Float64(2 * n + 5) - stats_x[1] - stats_y[1]) / 18.0
        + 2.0 * stats_x[0] * stats_y[0] / m
        + stats_x[2] * stats_y[2] / (9.0 * m * Float64(n - 2))
    )
    var z = Float64(concordant - discordant) / _sqrt(variance)
    return CorrelationResult(tau, _normal_two_sided(z))


def _tie_sums(values: List[Float64]) -> Tuple[Float64, Float64, Float64]:
    """Over the groups of equal values with sizes `c`: `sum c(c-1)/2`,
    `sum c(c-1)(2c+5)` and `sum c(c-1)(c-2)` -- SciPy's `count_rank_tie`."""
    var sorted = values.copy()
    _sort(sorted)
    var pairs = 0.0
    var weighted = 0.0
    var cubic = 0.0
    var i = 0
    while i < len(sorted):
        var j = i
        while j + 1 < len(sorted) and sorted[j + 1] == sorted[i]:
            j += 1
        var c = Float64(j - i + 1)
        if c > 1:
            pairs += c * (c - 1.0) / 2.0
            weighted += c * (c - 1.0) * (2.0 * c + 5.0)
            cubic += c * (c - 1.0) * (c - 2.0)
        i = j + 1
    return (pairs, weighted, cubic)


@fieldwise_init
struct LinregressResult(Copyable, Movable):
    """What `linregress` returns: `scipy.stats.linregress`'s result fields,
    the fitted line, its correlation, the two-sided p-value of a zero
    slope, and the standard errors of slope and intercept."""

    var slope: Float64
    var intercept: Float64
    var rvalue: Float64
    var pvalue: Float64
    var stderr: Float64
    var intercept_stderr: Float64


def linregress[
    dtype: DType, n: Int
](
    mut x: Static[dtype, n], mut y: Static[dtype, n]
) raises -> LinregressResult where (dtype.is_floating_point() and n > 2):
    """The least-squares line `y = slope x + intercept` through the points,
    with `r`, the two-sided p-value of `slope = 0` and both standard
    errors. `scipy.stats.linregress(x, y)`, formula for formula.
    """
    var xs = _as_float64(x.to_host())
    var ys = _as_float64(y.to_host())
    var ssxm = _covariance(xs, xs, 0)
    var ssym = _covariance(ys, ys, 0)
    var ssxym = _covariance(xs, ys, 0)
    if ssxm == 0:
        raise Error("linregress: x is constant")
    var slope = ssxym / ssxm
    var intercept = _mean(ys) - slope * _mean(xs)
    var r = 0.0
    if ssym > 0:
        r = min(1.0, max(-1.0, ssxym / _sqrt(ssxm * ssym)))
    var df = Float64(n - 2)
    comptime tiny = 1e-20
    var statistic = r * _sqrt(df / ((1.0 - r + tiny) * (1.0 + r + tiny)))
    var pvalue = _t_two_sided(statistic, df)
    var slope_stderr = _sqrt((1.0 - r * r) * ssym / ssxm / df)
    var xm = _mean(xs)
    var intercept_stderr = slope_stderr * _sqrt(ssxm + xm * xm)
    return LinregressResult(
        slope, intercept, r, pvalue, slope_stderr, intercept_stderr
    )


def zscore[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType], ddof: Int = 0) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Every element standardized by the tensor's mean and standard
    deviation, `(x - mean) / std` with `ddof` degrees of freedom in the
    standard deviation. `scipy.stats.zscore(a, ddof)`, the same shape
    back. A constant tensor raises rather than dividing by zero."""
    var values = _as_float64(xs.to_host())
    var n = len(values)
    if n - ddof <= 0:
        raise Error("zscore: not enough elements for ddof ", ddof)
    var centre = _mean(values)
    var scale = _sqrt(_covariance(values, values, ddof))
    if scale == 0:
        raise Error("zscore: the tensor is constant")
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype]((values[i] - centre) / scale))
    return Tensor[dtype, LayoutType](xs.context(), xs.layout, out^)
