"""Shape and summary statistics over `numax.core.array.Tensor`: `skew`,
`kurtosis`, `sem`, `gmean`, `hmean`, `entropy`, `trim_mean` and
`describe`, with `scipy.stats`' conventions and bias corrections.

**Tier 2, host-side**, in `Float64`: each is a handful of sums over one
vector and a scalar out, the shape `numax.stats.correlation` gives its
reason for. `skew` and `kurtosis` carry SciPy's `bias` switch and its
`G1`/`G2` corrections; `kurtosis` is Fisher's excess by default;
`entropy` normalizes its inputs and takes a `base`; `trim_mean` cuts the
same count from both ends SciPy does; `describe` gathers the six numbers
SciPy's does, with its `ddof=1` variance and biased shape statistics.

## The MAX gate

Nothing: MAX has no shape statistics. **Extend.**
"""

from std.builtin.sort import sort as _sort
from std.math import exp as _exp, log as _log, sqrt as _sqrt
from std.utils.numerics import inf as _inf

from layout.tile_layout import TensorLayout

from ..core.array import Static, Tensor


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


def _central_moment(values: List[Float64], order: Int) -> Float64:
    var centre = _mean(values)
    var total = 0.0
    for i in range(len(values)):
        var d = values[i] - centre
        var p = 1.0
        for _ in range(order):
            p *= d
        total += p
    return total / Float64(len(values))


def skew[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType], bias: Bool = True
) raises -> Float64 where dtype.is_floating_point():
    """The sample skewness `m3 / m2^(3/2)`, or with `bias=False` the
    adjusted Fisher-Pearson `G1 = sqrt(n(n-1)) / (n-2) * g1`.
    `scipy.stats.skew(a, bias)`. Zero for a symmetric sample; `0` too when
    `m2` is zero, as SciPy returns."""
    var values = _values(xs)
    var n = Float64(len(values))
    var m2 = _central_moment(values, 2)
    var m3 = _central_moment(values, 3)
    if m2 == 0:
        return 0.0
    var g1 = m3 / (m2 * _sqrt(m2))
    if bias or len(values) < 3:
        return g1
    return _sqrt(n * (n - 1.0)) / (n - 2.0) * g1


def kurtosis[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType], fisher: Bool = True, bias: Bool = True
) raises -> Float64 where dtype.is_floating_point():
    """The sample kurtosis `m4 / m2^2`, less `3` when `fisher` (the
    default, so a normal sample reads `0`), and with `bias=False` the
    unbiased `G2` correction. `scipy.stats.kurtosis(a, fisher, bias)`."""
    var values = _values(xs)
    var n = Float64(len(values))
    var m2 = _central_moment(values, 2)
    var m4 = _central_moment(values, 4)
    if m2 == 0:
        return 0.0 if fisher else 3.0
    var g2 = m4 / (m2 * m2)
    if not bias and len(values) > 3:
        g2 = (
            1.0
            / (n - 2.0)
            / (n - 3.0)
            * ((n * n - 1.0) * g2 - 3.0 * (n - 1.0) * (n - 1.0))
            + 3.0
        )
    return g2 - 3.0 if fisher else g2


def sem[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType], ddof: Int = 1
) raises -> Float64 where dtype.is_floating_point():
    """The standard error of the mean, `std(ddof) / sqrt(n)`, `ddof = 1` by
    default as SciPy's is. `scipy.stats.sem(a, ddof)`."""
    var values = _values(xs)
    var n = len(values)
    if n - ddof <= 0:
        raise Error("sem: not enough elements for ddof ", ddof)
    var centre = _mean(values)
    var total = 0.0
    for i in range(n):
        total += (values[i] - centre) * (values[i] - centre)
    return _sqrt(total / Float64(n - ddof)) / _sqrt(Float64(n))


def gmean[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType]
) raises -> Float64 where dtype.is_floating_point():
    """The geometric mean, `exp(mean(log x))`. `scipy.stats.gmean(a)`. A
    non-positive element raises rather than returning NaN."""
    var values = _values(xs)
    var total = 0.0
    for i in range(len(values)):
        if values[i] <= 0:
            raise Error("gmean: every element must be positive")
        total += _log(values[i])
    return _exp(total / Float64(len(values)))


def gmean[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType], weights: Tensor[dtype, LayoutType]
) raises -> Float64 where dtype.is_floating_point():
    """The weighted geometric mean, `exp(sum(w log x) / sum(w))`.
    `scipy.stats.gmean(a, weights=w)`."""
    var values = _values(xs)
    var ws = _values(weights)
    var total = 0.0
    var weight = 0.0
    for i in range(len(values)):
        if values[i] <= 0:
            raise Error("gmean: every element must be positive")
        total += ws[i] * _log(values[i])
        weight += ws[i]
    return _exp(total / weight)


def hmean[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType]
) raises -> Float64 where dtype.is_floating_point():
    """The harmonic mean, `n / sum(1 / x)`. `scipy.stats.hmean(a)`. A
    non-positive element raises."""
    var values = _values(xs)
    var total = 0.0
    for i in range(len(values)):
        if values[i] <= 0:
            raise Error("hmean: every element must be positive")
        total += 1.0 / values[i]
    return Float64(len(values)) / total


def entropy[
    dtype: DType, LayoutType: TensorLayout
](
    pk: Tensor[dtype, LayoutType], base: Optional[Float64] = None
) raises -> Float64 where dtype.is_floating_point():
    """The Shannon entropy `-sum(p log p)` of the distribution `pk`,
    normalized to sum to one first, in nats or in the given `base`.
    `scipy.stats.entropy(pk, base=base)`. Zero-probability terms
    contribute zero."""
    var p = _values(pk)
    var total = 0.0
    for i in range(len(p)):
        total += p[i]
    var s = 0.0
    for i in range(len(p)):
        var q = p[i] / total
        if q > 0:
            s -= q * _log(q)
    if base:
        s /= _log(base.value())
    return s


def entropy[
    dtype: DType, LayoutType: TensorLayout
](
    pk: Tensor[dtype, LayoutType],
    qk: Tensor[dtype, LayoutType],
    base: Optional[Float64] = None,
) raises -> Float64 where dtype.is_floating_point():
    """The relative entropy `sum(p log(p / q))`, the Kullback-Leibler
    divergence of `pk` from `qk`, both normalized first.
    `scipy.stats.entropy(pk, qk, base)`. Infinite where `qk` is zero and
    `pk` is not, as SciPy's is."""
    var p = _values(pk)
    var q = _values(qk)
    var pt = 0.0
    var qt = 0.0
    for i in range(len(p)):
        pt += p[i]
        qt += q[i]
    var s = 0.0
    for i in range(len(p)):
        var a = p[i] / pt
        var b = q[i] / qt
        if a > 0:
            if b == 0:
                return _inf[DType.float64]()
            s += a * _log(a / b)
    if base:
        s /= _log(base.value())
    return s


def trim_mean[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType], proportiontocut: Float64
) raises -> Float64 where dtype.is_floating_point():
    """The mean after dropping `int(proportiontocut * n)` of the smallest
    and as many of the largest values. `scipy.stats.trim_mean(a,
    proportiontocut)`. Raises when the cuts would leave nothing, as SciPy
    does."""
    var values = _values(xs)
    var n = len(values)
    var cut = Int(proportiontocut * Float64(n))
    if cut > n - cut:
        raise Error("trim_mean: proportion too big")
    _sort(values)
    var total = 0.0
    for i in range(cut, n - cut):
        total += values[i]
    return total / Float64(n - 2 * cut)


@fieldwise_init
struct Description(Copyable, Movable):
    """What `describe` returns: `scipy.stats.describe`'s six fields --
    the count, the extremes, the mean, the `ddof=1` variance, and the
    skewness and Fisher kurtosis."""

    var nobs: Int
    var min: Float64
    var max: Float64
    var mean: Float64
    var variance: Float64
    var skewness: Float64
    var kurtosis: Float64


def describe[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType], ddof: Int = 1, bias: Bool = True
) raises -> Description where dtype.is_floating_point():
    """The summary SciPy's `describe(a, ddof, bias)` returns: count,
    minimum and maximum, mean, variance with `ddof` degrees of freedom,
    skewness and Fisher kurtosis with the given `bias`."""
    var values = _values(xs)
    var n = len(values)
    if n - ddof <= 0:
        raise Error("describe: not enough elements for ddof ", ddof)
    var lo = values[0]
    var hi = values[0]
    var centre = _mean(values)
    var total = 0.0
    for i in range(n):
        lo = min(lo, values[i])
        hi = max(hi, values[i])
        total += (values[i] - centre) * (values[i] - centre)
    return Description(
        n,
        lo,
        hi,
        centre,
        total / Float64(n - ddof),
        skew(xs, bias),
        kurtosis(xs, True, bias),
    )
