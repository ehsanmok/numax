"""Resampling inference over `numax.core.tensor.Tensor`: `bootstrap` and
`permutation_test`, SciPy's.

**Tier 2.** Both take the statistic as a compile-time function over a
batch -- a matrix with one resample per row, returning one value per row
-- which is SciPy's `vectorized=True, axis=-1` contract and what lets the
whole null or bootstrap distribution be one call rather than `9999`:

```mojo
def row_mean[
    gpu: Bool
](x: Dynamic[DType.float64, 2]) raises -> Dynamic[DType.float64, 1]:
    return mean[axis=1, gpu=gpu](x)

var rng = Generator(seed=0)
var res = bootstrap[statistic=row_mean](data, rng)
```

The resample indices are drawn by one kernel, one lane per resample,
from `Generator`'s Philox stream -- a uniform index per element for the
bootstrap, a Fisher-Yates shuffle of the lane's row for a permutation --
and the data are gathered through them with `take`, so at `gpu=True` the
draw, the gather and the statistic all run on the input's device. Only
the `n_resamples` statistic values come back to the host, where the
interval or the p-value is read off them.

`permutation_test` enumerates every distinct rearrangement instead of
sampling whenever there are no more of them than `n_resamples`, as SciPy
does, so on small samples the two return the same p-value. The one
difference is `"pairings"`: reordering `y` against a fixed `x` has `n!`
distinct outcomes, which is the count used here, while SciPy counts the
`(n!)^2` reorderings of both and so samples at sizes this enumerates.

The batch is materialized whole: a bootstrap of `n` values holds
`n_resamples * n` of them, and `method="BCa"`'s jackknife another
`n * (n - 1)`. SciPy's `batch=` argument, which bounds that by running
the statistic over slices of the resamples, is the upgrade when memory
rather than time is the limit.

## The MAX gate

Nothing: MAX has neither. **Extend.**
"""

from std.math import inf as _inf, isnan as _isnan, nan as _nan, sqrt as _sqrt
from std.builtin.sort import sort as _std_sort
from std.random.philox import Random

from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core._drive import _check_device, _notice
from ..core.plain import Plain
from ..core.sorting import take
from ..core.tensorlike import TensorLike
from ..core.tensor import (
    Dynamic,
    _dyn_shape,
    _same_order,
    asarray,
    concatenate_dyn,
)
from .distributions import norm
from .random import Generator
from .statistics import _target

comptime _P = Plain[DType.float64]

comptime _INDEPENDENT = 0
comptime _SAMPLES = 1
comptime _PAIRINGS = 2


def _ndtr(x: Float64) -> Float64:
    return norm.cdf(_P(x), _P(0.0), _P(1.0)).v[0]


def _ndtri(p: Float64) -> Float64:
    if p <= 0.0:
        return -_inf[DType.float64]()
    if p >= 1.0:
        return _inf[DType.float64]()
    return norm.ppf(_P(p), _P(0.0), _P(1.0)).v[0]


def _as_vector[T: TensorLike](x: T) raises -> Dynamic[T.dtype, 1]:
    """`x`'s elements as a rank-1 run-time-shaped tensor on its device."""
    return _same_order(x, row_major(_dyn_shape[1](x.size())))


def _as_rows[
    dtype: DType
](var flat: Dynamic[dtype, 1], rows: Int, cols: Int) -> Dynamic[dtype, 2]:
    """`flat`'s buffer retyped to `rows x cols`, row-major, no copy."""
    return Dynamic[dtype, 2](
        flat._buffer,
        row_major(_dyn_shape[2](rows, cols)),
        flat.host_addressable,
    )


def _host_values[T: TensorLike](x: T) raises -> List[Float64]:
    var values = x.to_host()
    var out = List[Float64](capacity=len(values))
    for i in range(len(values)):
        out.append(Float64(values[i]))
    return out^


@always_inline
def _word(seed: UInt64, word: Int) -> UInt64:
    """Word `word` of the Philox stream `seed`."""
    var r = Random(seed=seed, offset=UInt64(word // 4))
    return UInt64(r.step()[word % 4])


@always_inline
def _below(seed: UInt64, word: Int, bound: Int) -> Int:
    """A uniform integer in `[0, bound)` from one word, Lemire's
    multiply-shift; the bias is `bound / 2^32`."""
    return Int((UInt64(bound) * _word(seed, word)) >> 32)


def _draw_indices[
    kind: Int, gpu: Bool
](
    seed: UInt64,
    rows: Int,
    n1: Int,
    n2: Int,
    mut ix: Dynamic[DType.int64, 1],
    mut iy: Dynamic[DType.int64, 1],
    ctx: DeviceContext,
) raises:
    """Fill `ix` (`rows x n1`) and `iy` (`rows x n2`), flat, with `rows`
    random rearrangements of `concat(x, y)`: row `r` of `ix` indexes the
    new `x`, row `r` of `iy` the new `y`.

    One lane per row, reading words `[r (n1 + n2), (r + 1)(n1 + n2))` of
    the stream, so the rows are independent and the fill is the same body
    on either side of the launch:

    - `_INDEPENDENT`: a Fisher-Yates shuffle of `0 .. n1+n2-1` across the
      two rows, the first `n1` positions being the new `x`.
    - `_SAMPLES`: a coin per pair `i` sends `x_i` or `y_i` to the new `x`
      and the other to the new `y` (`n1 == n2`).
    - `_PAIRINGS`: `x` stays; `y`'s row is a Fisher-Yates shuffle.
    """
    var n = n1 + n2
    var xs = ix.tile()
    var ys = iy.tile()

    @always_inline
    def body[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var ys, var seed, var n1, var n2, var n}:
        var row = coord_to_index_list(coord)[0]
        var xo = row * n1
        var yo = row * n2
        var base = row * n
        comptime if kind == _SAMPLES:
            for i in range(n1):
                var c = Int(_word(seed, base + i) >> 31)
                xs.ptr[unsafe_offset=xo + i] = Int64(i + c * n1)
                ys.ptr[unsafe_offset=yo + i] = Int64(i + (1 - c) * n1)
        elif kind == _PAIRINGS:
            for i in range(n1):
                xs.ptr[unsafe_offset=xo + i] = Int64(i)
            for i in range(n2):
                ys.ptr[unsafe_offset=yo + i] = Int64(n1 + i)
            var j = n2 - 1
            while j > 0:
                var k = _below(seed, base + j, j + 1)
                var held = ys.ptr[unsafe_offset=yo + j]
                ys.ptr[unsafe_offset=yo + j] = ys.ptr[unsafe_offset=yo + k]
                ys.ptr[unsafe_offset=yo + k] = held
                j -= 1
        else:
            for i in range(n1):
                xs.ptr[unsafe_offset=xo + i] = Int64(i)
            for i in range(n2):
                ys.ptr[unsafe_offset=yo + i] = Int64(n1 + i)
            # Position `p` of the row is `x`'s slot `p` below `n1`, `y`'s
            # slot `p - n1` above.
            var j = n - 1
            while j > 0:
                var k = _below(seed, base + j, j + 1)
                var jp = xo + j if j < n1 else yo + j - n1
                var kp = xo + k if k < n1 else yo + k - n1
                var held: Int64
                var other: Int64
                if j < n1:
                    held = xs.ptr[unsafe_offset=jp]
                else:
                    held = ys.ptr[unsafe_offset=jp]
                if k < n1:
                    other = xs.ptr[unsafe_offset=kp]
                    xs.ptr[unsafe_offset=kp] = held
                else:
                    other = ys.ptr[unsafe_offset=kp]
                    ys.ptr[unsafe_offset=kp] = held
                if j < n1:
                    xs.ptr[unsafe_offset=jp] = other
                else:
                    ys.ptr[unsafe_offset=jp] = other
                j -= 1

    elementwise[simd_width=1, target=_target[gpu]()](body, Coord(rows), ctx)
    ctx.synchronize()


def _resample_indices[
    gpu: Bool
](seed: UInt64, rows: Int, n: Int, ctx: DeviceContext) raises -> Dynamic[
    DType.int64, 1
]:
    """`rows x n` indices drawn uniformly from `[0, n)`, flat: element `i`
    is word `i` of the stream, multiply-shifted into range."""
    var idx = Dynamic[DType.int64, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](rows * n))
    )
    var out = idx.tile()

    @always_inline
    def body[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var out, var seed, var n}:
        var i = coord_to_index_list(coord)[0]
        out.ptr[unsafe_offset=i] = Int64(_below(seed, i, n))

    elementwise[simd_width=1, target=_target[gpu]()](body, Coord(rows * n), ctx)
    ctx.synchronize()
    return idx^


def _percentile(sorted: List[Float64], q: Float64) -> Float64:
    """`numpy.percentile(a, 100 q)` on sorted `a`, the default `"linear"`
    method; NaN for a NaN `q`, as BCa's degenerate case produces."""
    if _isnan(q):
        return _nan[DType.float64]()
    var n = len(sorted)
    var h = Float64(n - 1) * q
    var lo = Int(h)
    if lo >= n - 1:
        return sorted[n - 1]
    if lo < 0:
        return sorted[0]
    return sorted[lo] + (h - Float64(lo)) * (sorted[lo + 1] - sorted[lo])


# ---------------------------------------------------------------- bootstrap


struct ConfidenceInterval(Copyable, Movable):
    """A confidence interval, SciPy's `ConfidenceInterval(low, high)`; a
    one-sided interval has an infinite end."""

    var low: Float64
    """The lower end."""
    var high: Float64
    """The upper end."""

    def __init__(out self, low: Float64, high: Float64):
        """Build from the two ends.

        Args:
            low: The lower end.
            high: The upper end.
        """
        self.low = low
        self.high = high


struct BootstrapResult[dtype: DType](Movable):
    """What `bootstrap` returns, SciPy's `BootstrapResult`."""

    var confidence_interval: ConfidenceInterval
    """The interval for the statistic at the requested level."""
    var bootstrap_distribution: Dynamic[Self.dtype, 1]
    """The statistic on every resample, on the data's device."""
    var standard_error: Float64
    """The sample standard deviation (`ddof=1`) of the bootstrap
    distribution."""

    def __init__(
        out self,
        confidence_interval: ConfidenceInterval,
        var bootstrap_distribution: Dynamic[Self.dtype, 1],
        standard_error: Float64,
    ):
        """Build from the parts.

        Args:
            confidence_interval: The interval.
            bootstrap_distribution: The statistic on every resample.
            standard_error: The distribution's standard deviation.
        """
        self.confidence_interval = confidence_interval.copy()
        self.bootstrap_distribution = bootstrap_distribution^
        self.standard_error = standard_error


def _bootstrap[
    T: TensorLike,
    statistic: def[gpu: Bool](Dynamic[T.dtype, 2]) raises thin -> Dynamic[
        T.dtype, 1
    ],
    gpu: Bool,
](
    data: T,
    mut rng: Generator,
    n_resamples: Int,
    confidence_level: Float64,
    method: StaticString,
    alternative: StaticString,
) raises -> BootstrapResult[T.dtype]:
    comptime dtype = T.dtype
    var ctx = data.context()
    var n = data.size()
    var x = _as_vector(data)
    var idx = _resample_indices[gpu](rng._advance(), n_resamples, n, ctx)
    var resampled = take[axis=0, gpu=gpu](x, idx)
    var theta_b = statistic[gpu](_as_rows(resampled^, n_resamples, n))
    if theta_b.size() != n_resamples:
        raise Error(
            "bootstrap: the statistic returned ",
            theta_b.size(),
            " values for ",
            n_resamples,
            " resamples; it must reduce each row",
        )
    var values = _host_values(theta_b)
    var sorted = values.copy()
    _std_sort(sorted)

    var mean_b = 0.0
    for v in values:
        mean_b += v
    mean_b /= Float64(n_resamples)
    var ss = 0.0
    for v in values:
        ss += (v - mean_b) * (v - mean_b)
    var standard_error = _sqrt(ss / Float64(n_resamples - 1))

    var alpha = (1.0 - confidence_level) / 2.0
    if alternative != "two-sided":
        alpha = 1.0 - confidence_level
    var q_lo = alpha
    var q_hi = 1.0 - alpha

    var theta_hat = 0.0
    if method != "percentile":
        var observed = statistic[gpu](_as_rows(_as_vector(data), 1, n))
        theta_hat = _host_values(observed)[0]

    if method == "BCa":
        # z0: where the observed statistic sits in the bootstrap
        # distribution, SciPy's `_percentile_of_score` (ties count half).
        var below = 0
        var at_or_below = 0
        for v in values:
            if v < theta_hat:
                below += 1
            if v <= theta_hat:
                at_or_below += 1
        var z0 = _ndtri(Float64(below + at_or_below) / Float64(2 * n_resamples))
        # The acceleration, from the jackknife: row `i` drops element `i`.
        var jack = List[Int64](capacity=n * (n - 1))
        for i in range(n):
            for j in range(n):
                if j != i:
                    jack.append(Int64(j))
        var jidx = asarray(jack^, ctx)
        var dropped = take[axis=0, gpu=gpu](x, jidx)
        var theta_i = _host_values(statistic[gpu](_as_rows(dropped^, n, n - 1)))
        var dot = 0.0
        for v in theta_i:
            dot += v
        dot /= Float64(n)
        var num = 0.0
        var den = 0.0
        for v in theta_i:
            var d = dot - v
            num += d * d * d
            den += d * d
        var a_hat = num / (6.0 * den**1.5)
        var z_alpha = _ndtri(alpha)
        var n1 = z0 + z_alpha
        var n2 = z0 - z_alpha
        q_lo = _ndtr(z0 + n1 / (1.0 - a_hat * n1))
        q_hi = _ndtr(z0 + n2 / (1.0 - a_hat * n2))

    var low = _percentile(sorted, q_lo)
    var high = _percentile(sorted, q_hi)
    if method == "basic":
        var reflected_low = 2.0 * theta_hat - high
        high = 2.0 * theta_hat - low
        low = reflected_low
    if alternative == "less":
        low = -_inf[DType.float64]()
    elif alternative == "greater":
        high = _inf[DType.float64]()
    return BootstrapResult[dtype](
        ConfidenceInterval(low, high), theta_b^, standard_error
    )


def bootstrap[
    T: TensorLike,
    statistic: def[gpu: Bool](Dynamic[T.dtype, 2]) raises thin -> Dynamic[
        T.dtype, 1
    ],
    gpu: Bool = False,
](
    data: T,
    mut rng: Generator,
    n_resamples: Int = 9999,
    confidence_level: Float64 = 0.95,
    method: StaticString = "BCa",
    alternative: StaticString = "two-sided",
) raises -> BootstrapResult[T.dtype] where T.dtype.is_floating_point():
    """A bootstrap confidence interval for `statistic` of one sample.
    `scipy.stats.bootstrap((data,), statistic, vectorized=True)`.

    Draws `n_resamples` samples of `data`'s size with replacement, applies
    `statistic` to all of them in one call, and reads the interval off the
    resulting distribution: its quantiles (`"percentile"`), those
    reflected about the observed statistic (`"basic"`), or quantiles
    shifted by the bias correction `z0` and the jackknife acceleration
    `a` (`"BCa"`, the default, Efron's bias-corrected and accelerated
    interval). SciPy's formulas throughout, NumPy's `"linear"` quantile
    included; the resamples come from `rng`'s stream rather than NumPy's,
    so the interval agrees with SciPy's to Monte Carlo error rather than
    digit for digit.

    At `gpu=True` the index draw, the gather and `statistic[gpu=True]`
    run on `data`'s device; only the `n_resamples` statistic values are
    downloaded.

    Parameters:
        T: The tensor type of `data`, read flat; floating point.
        statistic: The statistic of each row of a batch, one value per
            row, SciPy's `vectorized=True` statistic with `axis=-1`. It is
            called with the same `gpu` as `bootstrap`.
        gpu: Whether to resample and evaluate on `data`'s device.

    Args:
        data: The sample.
        rng: The generator to draw from; advanced by one draw.
        n_resamples: How many resamples, at least 2.
        confidence_level: The interval's coverage, in `(0, 1)`.
        method: `"percentile"`, `"basic"` or `"BCa"`.
        alternative: `"two-sided"`, `"less"` (the interval is
            `(-inf, high]`) or `"greater"` (`[low, inf)`).

    Returns:
        A `BootstrapResult` with the interval, the bootstrap distribution
        and its standard deviation.

    Raises:
        On an unknown `method` or `alternative`, fewer than two resamples
        or elements, a `confidence_level` outside `(0, 1)`, a statistic
        that does not return one value per row, or a device failure.
    """
    if method != "percentile" and method != "basic" and method != "BCa":
        raise Error(
            "bootstrap: method must be 'percentile', 'basic' or 'BCa', got '",
            method,
            "'",
        )
    if (
        alternative != "two-sided"
        and alternative != "less"
        and alternative != "greater"
    ):
        raise Error(
            (
                "bootstrap: alternative must be 'two-sided', 'less' or"
                " 'greater', got '"
            ),
            alternative,
            "'",
        )
    if n_resamples < 2:
        raise Error("bootstrap: n_resamples must be at least 2")
    if data.size() < 2:
        raise Error("bootstrap: the sample needs at least two elements")
    if not (confidence_level > 0.0 and confidence_level < 1.0):
        raise Error("bootstrap: confidence_level must lie in (0, 1)")
    if _check_device[T, gpu](data):
        return _bootstrap[T, statistic, gpu](
            data, rng, n_resamples, confidence_level, method, alternative
        )
    _notice[gpu]("bootstrap")
    var host = asarray(data.to_host())
    return _bootstrap[Dynamic[T.dtype, 1], statistic, False](
        host, rng, n_resamples, confidence_level, method, alternative
    )


# ---------------------------------------------------------- permutation_test


struct PermutationTestResult[dtype: DType](Movable):
    """What `permutation_test` returns, SciPy's `PermutationTestResult`."""

    var statistic: Float64
    """The statistic on the observed data."""
    var pvalue: Float64
    """The p-value under the chosen alternative."""
    var null_distribution: Dynamic[Self.dtype, 1]
    """The statistic on every rearrangement, on the data's device."""

    def __init__(
        out self,
        statistic: Float64,
        pvalue: Float64,
        var null_distribution: Dynamic[Self.dtype, 1],
    ):
        """Build from the parts.

        Args:
            statistic: The observed statistic.
            pvalue: The p-value.
            null_distribution: The statistic on every rearrangement.
        """
        self.statistic = statistic
        self.pvalue = pvalue
        self.null_distribution = null_distribution^


def _count_rearrangements(kind: Int, n1: Int, n2: Int, cap: Int) -> Int:
    """How many distinct rearrangements `kind` has, or `cap + 1` once it is
    known to exceed `cap`: `C(n1 + n2, n1)`, `2^n1` or `n2!`."""
    var total = 1
    if kind == _INDEPENDENT:
        var small = n1 if n1 < n2 else n2
        var n = n1 + n2
        for i in range(small):
            total = total * (n - i) // (i + 1)
            if total > cap:
                return cap + 1
    elif kind == _SAMPLES:
        for _ in range(n1):
            total *= 2
            if total > cap:
                return cap + 1
    else:
        for i in range(2, n2 + 1):
            total *= i
            if total > cap:
                return cap + 1
    return total


def _exact_indices(
    kind: Int, n1: Int, n2: Int
) -> Tuple[List[Int64], List[Int64], Int]:
    """Every distinct rearrangement as two flat index matrices over
    `concat(x, y)`, and their count, enumerated on the host."""
    var ix = List[Int64]()
    var iy = List[Int64]()
    var rows = 0
    if kind == _INDEPENDENT:
        var n = n1 + n2
        var pick = List[Int](capacity=n1)
        for i in range(n1):
            pick.append(i)
        while True:
            var chosen = List[Bool](length=n, fill=False)
            for i in range(n1):
                ix.append(Int64(pick[i]))
                chosen[pick[i]] = True
            for i in range(n):
                if not chosen[i]:
                    iy.append(Int64(i))
            rows += 1
            # The next `n1`-combination of `0 .. n-1` in lexicographic order.
            var i = n1 - 1
            while i >= 0 and pick[i] == n - n1 + i:
                i -= 1
            if i < 0:
                break
            pick[i] += 1
            for j in range(i + 1, n1):
                pick[j] = pick[j - 1] + 1
    elif kind == _SAMPLES:
        var total = 1 << n1
        for mask in range(total):
            for i in range(n1):
                var c = (mask >> i) & 1
                ix.append(Int64(i + c * n1))
                iy.append(Int64(i + (1 - c) * n1))
            rows += 1
    else:
        var perm = List[Int](capacity=n2)
        for i in range(n2):
            perm.append(i)
        while True:
            for i in range(n1):
                ix.append(Int64(i))
            for i in range(n2):
                iy.append(Int64(n1 + perm[i]))
            rows += 1
            # The next permutation in lexicographic order.
            var i = n2 - 2
            while i >= 0 and perm[i] >= perm[i + 1]:
                i -= 1
            if i < 0:
                break
            var j = n2 - 1
            while perm[j] <= perm[i]:
                j -= 1
            var held = perm[i]
            perm[i] = perm[j]
            perm[j] = held
            var lo = i + 1
            var hi = n2 - 1
            while lo < hi:
                held = perm[lo]
                perm[lo] = perm[hi]
                perm[hi] = held
                lo += 1
                hi -= 1
    return (ix^, iy^, rows)


def _permutation_test[
    dtype: DType,
    statistic: def[gpu: Bool](
        Dynamic[dtype, 2], Dynamic[dtype, 2]
    ) raises thin -> Dynamic[dtype, 1],
    gpu: Bool,
](
    x: Dynamic[dtype, 1],
    y: Dynamic[dtype, 1],
    mut rng: Generator,
    kind: Int,
    n_resamples: Int,
    alternative: StaticString,
) raises -> PermutationTestResult[dtype]:
    var ctx = x.context()
    var n1 = x.size()
    var n2 = y.size()
    var pooled = concatenate_dyn[gpu=gpu](x, y)

    var observed = statistic[gpu](
        _as_rows(_as_vector(x), 1, n1), _as_rows(_as_vector(y), 1, n2)
    )
    var stat = _host_values(observed)[0]

    var total = _count_rearrangements(kind, n1, n2, n_resamples)
    var exact = total <= n_resamples
    var rows = total if exact else n_resamples
    var ix: Dynamic[DType.int64, 1]
    var iy: Dynamic[DType.int64, 1]
    if exact:
        var enumerated = _exact_indices(kind, n1, n2)
        ix = asarray(enumerated[0].copy(), ctx)
        iy = asarray(enumerated[1].copy(), ctx)
    else:
        var seed = rng._advance()
        ix = Dynamic[DType.int64, 1]._uninitialized(
            ctx, row_major(_dyn_shape[1](rows * n1))
        )
        iy = Dynamic[DType.int64, 1]._uninitialized(
            ctx, row_major(_dyn_shape[1](rows * n2))
        )
        if kind == _SAMPLES:
            _draw_indices[_SAMPLES, gpu](seed, rows, n1, n2, ix, iy, ctx)
        elif kind == _PAIRINGS:
            _draw_indices[_PAIRINGS, gpu](seed, rows, n1, n2, ix, iy, ctx)
        else:
            _draw_indices[_INDEPENDENT, gpu](seed, rows, n1, n2, ix, iy, ctx)
    var xr = take[axis=0, gpu=gpu](pooled, ix)
    var yr = take[axis=0, gpu=gpu](pooled, iy)
    var null = statistic[gpu](_as_rows(xr^, rows, n1), _as_rows(yr^, rows, n2))
    if null.size() != rows:
        raise Error(
            "permutation_test: the statistic returned ",
            null.size(),
            " values for ",
            rows,
            " rearrangements; it must reduce each row",
        )
    var values = _host_values(null)

    # SciPy's comparison: a relative slack of `100 eps` so that
    # rearrangements equal to the observed arrangement in exact arithmetic
    # count as at least as extreme, and the `+1` of an estimated p-value.
    comptime eps = 2.220446049250313e-14 if dtype == DType.float64 else 1.1920928955078125e-05
    var slack = abs(eps * stat)
    var adjustment = 0.0 if exact else 1.0
    var at_most = 0
    var at_least = 0
    for v in values:
        if v <= stat + slack:
            at_most += 1
        if v >= stat - slack:
            at_least += 1
    var p_less = (Float64(at_most) + adjustment) / (Float64(rows) + adjustment)
    var p_greater = (Float64(at_least) + adjustment) / (
        Float64(rows) + adjustment
    )
    var pvalue: Float64
    if alternative == "less":
        pvalue = p_less
    elif alternative == "greater":
        pvalue = p_greater
    else:
        pvalue = 2.0 * (p_less if p_less < p_greater else p_greater)
    if pvalue > 1.0:
        pvalue = 1.0
    return PermutationTestResult[dtype](stat, pvalue, null^)


def permutation_test[
    X: TensorLike,
    Y: TensorLike,
    statistic: def[gpu: Bool](
        Dynamic[X.dtype, 2], Dynamic[X.dtype, 2]
    ) raises thin -> Dynamic[X.dtype, 1],
    gpu: Bool = False,
](
    x: X,
    y: Y,
    mut rng: Generator,
    permutation_type: StaticString = "independent",
    n_resamples: Int = 9999,
    alternative: StaticString = "two-sided",
) raises -> PermutationTestResult[X.dtype] where (
    X.dtype == Y.dtype and X.dtype.is_floating_point()
):
    """A permutation test of `statistic` on two samples.
    `scipy.stats.permutation_test((x, y), statistic, vectorized=True)`.

    The null distribution is `statistic` on rearrangements of the data
    that the null hypothesis says are exchangeable:

    - `"independent"`: the pooled values dealt back into groups of `x`'s
      and `y`'s sizes, for a difference between the samples.
    - `"samples"`: each pair `(x_i, y_i)` swapped or not, for paired
      samples (`x` and `y` the same length).
    - `"pairings"`: `y` reordered against a fixed `x`, for an association
      between them (the same length).

    When there are at most `n_resamples` distinct rearrangements they are
    all enumerated and the p-value is exact, as SciPy's is; otherwise
    `n_resamples` are drawn from `rng` and the p-value is SciPy's
    `(count + 1) / (n_resamples + 1)`. A two-sided p-value is twice the
    smaller one-sided one, capped at 1.

    At `gpu=True` the rearrangement draw, the gathers and
    `statistic[gpu=True]` run on the data's device; an exact test
    enumerates its indices on the host and uploads them.

    Parameters:
        X: The tensor type of `x`, read flat; floating point.
        Y: The tensor type of `y`, read flat; the same `dtype`.
        statistic: The statistic of each row pair of two batches, one
            value per row, SciPy's `vectorized=True` statistic with
            `axis=-1`. Called with the same `gpu` as `permutation_test`.
        gpu: Whether to rearrange and evaluate on the data's device.

    Args:
        x: The first sample.
        y: The second sample.
        rng: The generator to draw from; advanced by one draw unless the
            test is exact.
        permutation_type: `"independent"`, `"samples"` or `"pairings"`.
        n_resamples: How many rearrangements to draw when not exact.
        alternative: `"two-sided"`, `"less"` (the observed statistic is
            small) or `"greater"`.

    Returns:
        A `PermutationTestResult` with the observed statistic, the
        p-value and the null distribution.

    Raises:
        On an unknown `permutation_type` or `alternative`, samples of
        different lengths under `"samples"`/`"pairings"`, an empty sample,
        `n_resamples < 1`, a statistic that does not return one value per
        row, or a device failure.
    """
    var kind: Int
    if permutation_type == "independent":
        kind = _INDEPENDENT
    elif permutation_type == "samples":
        kind = _SAMPLES
    elif permutation_type == "pairings":
        kind = _PAIRINGS
    else:
        raise Error(
            (
                "permutation_test: permutation_type must be 'independent',"
                " 'samples' or 'pairings', got '"
            ),
            permutation_type,
            "'",
        )
    if (
        alternative != "two-sided"
        and alternative != "less"
        and alternative != "greater"
    ):
        raise Error(
            (
                "permutation_test: alternative must be 'two-sided', 'less' or"
                " 'greater', got '"
            ),
            alternative,
            "'",
        )
    if x.size() == 0 or y.size() == 0:
        raise Error("permutation_test: a sample is empty")
    if kind != _INDEPENDENT and x.size() != y.size():
        raise Error(
            "permutation_test: '",
            permutation_type,
            "' needs samples of one length, got ",
            x.size(),
            " and ",
            y.size(),
        )
    if n_resamples < 1:
        raise Error("permutation_test: n_resamples must be at least 1")
    if _check_device[X, gpu](x) and _check_device[Y, gpu](y):
        var xv = _as_vector(x)
        var yv = rebind_var[Dynamic[X.dtype, 1]](_as_vector(y))
        return _permutation_test[X.dtype, statistic, gpu](
            xv, yv, rng, kind, n_resamples, alternative
        )
    _notice[gpu]("permutation_test")
    var hx = asarray(x.to_host())
    var hy = rebind_var[Dynamic[X.dtype, 1]](asarray(y.to_host()))
    return _permutation_test[X.dtype, statistic, False](
        hx, hy, rng, kind, n_resamples, alternative
    )
