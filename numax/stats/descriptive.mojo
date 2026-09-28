"""Shape and summary statistics over `numax.core.tensor.Tensor`: `skew`,
`kurtosis`, `sem`, `gmean`, `hmean`, `entropy`, `trim_mean` and
`describe`, with `scipy.stats`' conventions and bias corrections.

**Tier 2.** Each is a handful of sums over one vector and a scalar out.
On the host they run in `Float64`; at `gpu=True` the sums run on the
device at the tensor's dtype through elementwise launches and MAX's
`ReduceSum`, and only scalars cross back -- every routine here except the
weighted `gmean` and `trim_mean`, which needs a sort. `skew` and `kurtosis` carry SciPy's `bias` switch and its
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

from ..core._drive import _check_device, _notice
from ..core.elementwise import clip as _clip, log as _vlog, reciprocal
from ..core.logic import any as _any, equal, greater, logical_and
from ..core.ops import divide, multiply, subtract
from ..core.tensorlike import TensorLike, dim, is_row_major
from ..core.tensor import Static, Tensor, zeros_like
from .statistics import max as _tmax, min as _tmin, sum as _tsum


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


def _moments[
    T: TensorLike, gpu: Bool
](xs: T) raises -> List[Float64] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """`[n, mean, m2, m3, m4]`: the count, the mean and the second to fourth
    central moments, from which `skew`, `kurtosis`, `sem` and `describe`
    are each a formula.

    On the host a `Float64` walk. At `gpu=True` the same sums on the
    device, at the tensor's dtype: the mean from `sum`, then `x - mean` and
    its powers by elementwise launches, each folded by MAX's `ReduceSum`,
    so four scalars cross back rather than the tensor. `float32` on Metal,
    so the device answer is good to a few `float32` ulp of the moments.
    """
    var count = xs.size()
    var n = Float64(count)
    if _check_device[T, gpu](xs):
        comptime if gpu:
            var centre = Float64(_tsum[gpu=True](xs)) / n
            var d = subtract[gpu=True](xs, Scalar[T.dtype](centre))
            var d2 = multiply[gpu=True](d, d)
            var d3 = multiply[gpu=True](d2, d)
            var d4 = multiply[gpu=True](d2, d2)
            return [
                n,
                centre,
                Float64(_tsum[gpu=True](d2)) / n,
                Float64(_tsum[gpu=True](d3)) / n,
                Float64(_tsum[gpu=True](d4)) / n,
            ]
    else:
        _notice[gpu]("descriptive statistics")
    var values = _values(xs)
    return [
        n,
        _mean(values),
        _central_moment(values, 2),
        _central_moment(values, 3),
        _central_moment(values, 4),
    ]


def _skew_of(m: List[Float64], bias: Bool) -> Float64:
    var n = m[0]
    if m[2] == 0:
        return 0.0
    var g1 = m[3] / (m[2] * _sqrt(m[2]))
    if bias or n < 3:
        return g1
    return _sqrt(n * (n - 1.0)) / (n - 2.0) * g1


def _kurtosis_of(m: List[Float64], fisher: Bool, bias: Bool) -> Float64:
    var n = m[0]
    if m[2] == 0:
        return 0.0 if fisher else 3.0
    var g2 = m[4] / (m[2] * m[2])
    if not bias and n > 3:
        g2 = (
            1.0
            / (n - 2.0)
            / (n - 3.0)
            * ((n * n - 1.0) * g2 - 3.0 * (n - 1.0) * (n - 1.0))
            + 3.0
        )
    return g2 - 3.0 if fisher else g2


def skew[
    T: TensorLike, gpu: Bool = False
](xs: T, bias: Bool = True) raises -> Float64 where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The sample skewness `m3 / m2^(3/2)`, or with `bias=False` the
    adjusted Fisher-Pearson `G1 = sqrt(n(n-1)) / (n-2) * g1`.
    `scipy.stats.skew(a, bias)`. Zero for a symmetric sample; `0` too when
    `m2` is zero, as SciPy returns."""
    return _skew_of(_moments[gpu=gpu](xs), bias)


def kurtosis[
    T: TensorLike, gpu: Bool = False
](xs: T, fisher: Bool = True, bias: Bool = True) raises -> Float64 where (
    T.dtype.is_floating_point() and is_row_major[T]
):
    """The sample kurtosis `m4 / m2^2`, less `3` when `fisher` (the
    default, so a normal sample reads `0`), and with `bias=False` the
    unbiased `G2` correction. `scipy.stats.kurtosis(a, fisher, bias)`."""
    return _kurtosis_of(_moments[gpu=gpu](xs), fisher, bias)


def sem[
    T: TensorLike, gpu: Bool = False
](xs: T, ddof: Int = 1) raises -> Float64 where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The standard error of the mean, `std(ddof) / sqrt(n)`, `ddof = 1` by
    default as SciPy's is. `scipy.stats.sem(a, ddof)`."""
    var n = xs.size()
    if n - ddof <= 0:
        raise Error("sem: not enough elements for ddof ", ddof)
    var m = _moments[gpu=gpu](xs)
    return _sqrt(m[2] * m[0] / Float64(n - ddof)) / _sqrt(m[0])


def gmean[
    T: TensorLike, gpu: Bool = False
](xs: T) raises -> Float64 where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The geometric mean, `exp(mean(log x))`. `scipy.stats.gmean(a)`. A
    non-positive element raises rather than returning NaN."""
    if _check_device[T, gpu](xs):
        comptime if gpu:
            if Float64(_tmin[gpu=True](xs)) <= 0:
                raise Error("gmean: every element must be positive")
            return _exp(
                Float64(_tsum[gpu=True](_vlog[gpu=True](xs)))
                / Float64(xs.size())
            )
    else:
        _notice[gpu]("gmean")
    var values = _values(xs)
    var total = 0.0
    for i in range(len(values)):
        if values[i] <= 0:
            raise Error("gmean: every element must be positive")
        total += _log(values[i])
    return _exp(total / Float64(len(values)))


def gmean[
    T: TensorLike
](xs: T, weights: T) raises -> Float64 where (
    is_row_major[T] and T.dtype.is_floating_point()
):
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
    T: TensorLike, gpu: Bool = False
](xs: T) raises -> Float64 where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The harmonic mean, `n / sum(1 / x)`. `scipy.stats.hmean(a)`. A
    non-positive element raises."""
    if _check_device[T, gpu](xs):
        comptime if gpu:
            if Float64(_tmin[gpu=True](xs)) <= 0:
                raise Error("hmean: every element must be positive")
            return Float64(xs.size()) / Float64(
                _tsum[gpu=True](reciprocal[gpu=True](xs))
            )
    else:
        _notice[gpu]("hmean")
    var values = _values(xs)
    var total = 0.0
    for i in range(len(values)):
        if values[i] <= 0:
            raise Error("hmean: every element must be positive")
        total += 1.0 / values[i]
    return Float64(len(values)) / total


def entropy[
    T: TensorLike, gpu: Bool = False
](pk: T, base: Optional[Float64] = None) raises -> Float64 where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The Shannon entropy `-sum(p log p)` of the distribution `pk`,
    normalized to sum to one first, in nats or in the given `base`.
    `scipy.stats.entropy(pk, base=base)`. Zero-probability terms
    contribute zero."""
    if _check_device[T, gpu](pk):
        comptime if gpu:
            # `q log q` with `q` clipped up to `1e-30` before the log: a
            # zero `q` then contributes `0 * finite = 0`, the limit, where
            # `log(0)` would make it NaN; a `q` below the clip is off by
            # under `1e-28`.
            var q = divide[gpu=True](pk, _tsum[gpu=True](pk))
            var logq = _vlog[gpu=True](
                _clip[gpu=True](q, Scalar[T.dtype](1e-30), 1)
            )
            var s = -Float64(_tsum[gpu=True](multiply[gpu=True](q, logq)))
            if base:
                s /= _log(base.value())
            return s
    else:
        _notice[gpu]("entropy")
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
    T: TensorLike, gpu: Bool = False
](
    pk: T,
    qk: T,
    base: Optional[Float64] = None,
) raises -> Float64 where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The relative entropy `sum(p log(p / q))`, the Kullback-Leibler
    divergence of `pk` from `qk`, both normalized first.
    `scipy.stats.entropy(pk, qk, base)`. Infinite where `qk` is zero and
    `pk` is not, as SciPy's is."""
    if _check_device[T, gpu](pk) and _check_device[T, gpu](qk):
        comptime if gpu:
            var a = divide[gpu=True](pk, _tsum[gpu=True](pk))
            var b = divide[gpu=True](qk, _tsum[gpu=True](qk))
            var z = zeros_like(a)
            if _any[gpu=True](
                logical_and[gpu=True](
                    greater[gpu=True](a, z), equal[gpu=True](b, z)
                )
            ):
                return _inf[DType.float64]()
            var tiny = Scalar[T.dtype](1e-30)
            var ratio = subtract[gpu=True](
                _vlog[gpu=True](_clip[gpu=True](a, tiny, 1)),
                _vlog[gpu=True](_clip[gpu=True](b, tiny, 1)),
            )
            var s = Float64(_tsum[gpu=True](multiply[gpu=True](a, ratio)))
            if base:
                s /= _log(base.value())
            return s
    else:
        _notice[gpu]("entropy")
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
    T: TensorLike
](
    xs: T, proportiontocut: Float64
) raises -> Float64 where T.dtype.is_floating_point():
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
struct Description(Copyable):
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
    T: TensorLike, gpu: Bool = False
](xs: T, ddof: Int = 1, bias: Bool = True) raises -> Description where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The summary SciPy's `describe(a, ddof, bias)` returns: count,
    minimum and maximum, mean, variance with `ddof` degrees of freedom,
    skewness and Fisher kurtosis with the given `bias`."""
    var n = xs.size()
    if n - ddof <= 0:
        raise Error("describe: not enough elements for ddof ", ddof)
    if _check_device[T, gpu](xs):
        comptime if gpu:
            var m = _moments[gpu=True](xs)
            return Description(
                n,
                Float64(_tmin[gpu=True](xs)),
                Float64(_tmax[gpu=True](xs)),
                m[1],
                m[2] * m[0] / Float64(n - ddof),
                _skew_of(m, bias),
                _kurtosis_of(m, True, bias),
            )
    else:
        _notice[gpu]("describe")
    var values = _values(xs)
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
