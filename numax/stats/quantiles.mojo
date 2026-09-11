"""Order statistics over `numax.core.array.Tensor`: `quantile`,
`percentile`, their NaN-ignoring forms, `nanmedian` and `iqr`, with every
`method` NumPy names.

**Tier 2, host-side**, the way `median` is: a quantile is a position in the
sorted data, so the tensor comes to the host once, is sorted once, and
every requested quantile is read off it. There is no device path because
there is no MAX sort to build one on -- `docs/parity.md` records that
`nn.argsort` is the only ordering kernel and it is host-side too -- and
because a sort is the one step here that is not a reduction.

## The methods

NumPy's `method` names, all thirteen, in two families. The continuous ones
are Hyndman and Fan's `(alpha, beta)` family: a virtual index `q (n - alpha
- beta + 1) + alpha - 1` into the sorted sample, linearly interpolated
between its neighbours -- `"linear"` (NumPy's default, `alpha = beta =
1`), `"interpolated_inverted_cdf"` `(0, 1)`, `"hazen"` `(1/2, 1/2)`,
`"weibull"` `(0, 0)`, `"median_unbiased"` `(1/3, 1/3)`,
`"normal_unbiased"` `(3/8, 3/8)`. The discontinuous ones pick a sample:
`"inverted_cdf"`, `"averaged_inverted_cdf"`, `"closest_observation"`, and
the four that round the `"linear"` index -- `"lower"`, `"higher"`,
`"midpoint"`, `"nearest"` (round half to even, as NumPy rounds). Every one
is pinned to `numpy.quantile` in the tests.
"""

from std.builtin.sort import sort as _sort
from std.math import floor as _floor, ceil as _ceil
from std.utils.numerics import nan as _nan

from layout.tile_layout import TensorLayout

from ..core.array import Static, Tensor


def _virtual_index(
    q: Float64, n: Int, alpha: Float64, beta: Float64
) -> Float64:
    """Hyndman-Fan's position of quantile `q` in `n` sorted samples,
    zero-based and clipped to the sample."""
    var index = q * (Float64(n) - alpha - beta + 1.0) + alpha - 1.0
    if index < 0:
        return 0.0
    if index > Float64(n - 1):
        return Float64(n - 1)
    return index


def _interpolated(values: List[Float64], index: Float64) -> Float64:
    var lo = Int(_floor(index))
    var hi = Int(_ceil(index))
    var fraction = index - Float64(lo)
    return values[lo] + fraction * (values[hi] - values[lo])


def _round_half_even(x: Float64) -> Int:
    var lo = _floor(x)
    var fraction = x - lo
    if fraction > 0.5:
        return Int(lo) + 1
    if fraction < 0.5:
        return Int(lo)
    return Int(lo) if Int(lo) % 2 == 0 else Int(lo) + 1


def _quantile_of(
    values: List[Float64], q: Float64, method: StaticString
) raises -> Float64:
    """One quantile of an ascending `values`, by `method`."""
    var n = len(values)
    if n == 0:
        raise Error("quantile: no values")
    if q < 0 or q > 1:
        raise Error("quantile: q must lie in [0, 1]")
    if method == "linear":
        return _interpolated(values, _virtual_index(q, n, 1.0, 1.0))
    if method == "interpolated_inverted_cdf":
        return _interpolated(values, _virtual_index(q, n, 0.0, 1.0))
    if method == "hazen":
        return _interpolated(values, _virtual_index(q, n, 0.5, 0.5))
    if method == "weibull":
        return _interpolated(values, _virtual_index(q, n, 0.0, 0.0))
    if method == "median_unbiased":
        return _interpolated(values, _virtual_index(q, n, 1.0 / 3.0, 1.0 / 3.0))
    if method == "normal_unbiased":
        return _interpolated(values, _virtual_index(q, n, 0.375, 0.375))
    var linear = _virtual_index(q, n, 1.0, 1.0)
    if method == "lower":
        return values[Int(_floor(linear))]
    if method == "higher":
        return values[Int(_ceil(linear))]
    if method == "midpoint":
        return 0.5 * (values[Int(_floor(linear))] + values[Int(_ceil(linear))])
    if method == "nearest":
        return values[_round_half_even(linear)]
    # The three sample-picking rules on `q * n`.
    var position = q * Float64(n)
    var whole = Int(_floor(position))
    var fraction = position - Float64(whole)
    if method == "inverted_cdf":
        var index = whole if fraction > 0 else whole - 1
        return values[min(max(index, 0), n - 1)]
    if method == "averaged_inverted_cdf":
        if fraction > 0:
            return values[min(max(whole, 0), n - 1)]
        var above = min(max(whole, 0), n - 1)
        var below = min(max(whole - 1, 0), n - 1)
        return 0.5 * (values[below] + values[above])
    if method == "closest_observation":
        # Hyndman-Fan 3: `inverted_cdf` at `q n - 1/2`, taking the even order
        # statistic on a tie, which is NumPy's `_closest_observation`.
        var shifted = position - 0.5
        var j = Int(_floor(shifted))
        var g = shifted - Float64(j)
        var gamma = 0 if (g == 0 and j % 2 == 0) else 1
        return values[min(max(j - 1 + gamma, 0), n - 1)]
    raise Error("quantile: unknown method '", method, "'")


def _sorted_host[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType], drop_nan: Bool) raises -> List[Float64]:
    """The tensor's values on the host, ascending, NaNs removed when
    `drop_nan`. Without `drop_nan` a NaN anywhere empties the list, which
    every caller reads as "the answer is NaN" -- NumPy's propagation rule,
    spelled once."""
    var host = xs.to_host()
    var values = List[Float64](capacity=len(host))
    for i in range(len(host)):
        var v = Float64(host[i])
        if v != v:
            if drop_nan:
                continue
            return List[Float64]()
        values.append(v)
    _sort(values)
    return values^


def _read[
    dtype: DType
](values: List[Float64], q: Float64, method: StaticString) raises -> Scalar[
    dtype
]:
    """One quantile, or NaN when `_sorted_host` reported a NaN."""
    if len(values) == 0:
        return _nan[dtype]()
    return Scalar[dtype](_quantile_of(values, q, method))


def quantile[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType], q: Float64, method: StaticString = "linear"
) raises -> Scalar[dtype] where dtype.is_floating_point():
    """The `q`-th quantile of every element of `xs`, `q` in `[0, 1]`.
    `numpy.quantile(a, q, method=method)`.

    `"linear"` by default, which interpolates between the two nearest order
    statistics and is what `median` is at `q = 0.5`. The module docstring
    lists the other twelve methods; an unknown one raises. A NaN in `xs`
    propagates, as NumPy's does -- `nanquantile` is the one that ignores it.
    """
    return _read[dtype](_sorted_host(xs, False), q, method)


def quantile[
    dtype: DType, LayoutType: TensorLayout, m: Int
](
    xs: Tensor[dtype, LayoutType],
    q: Static[dtype, m],
    method: StaticString = "linear",
) raises -> Static[dtype, m] where (dtype.is_floating_point() and m > 0):
    """Several quantiles at once: one sort, `m` reads.
    `numpy.quantile(a, [q0, q1, ...])`."""
    var sorted = _sorted_host(xs, False)
    var qs = q.to_host()
    var out = List[Scalar[dtype]](capacity=m)
    for i in range(m):
        out.append(_read[dtype](sorted, Float64(qs[i]), method))
    return Static[dtype, m](q.context(), out^)


def percentile[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType], q: Float64, method: StaticString = "linear"
) raises -> Scalar[dtype] where dtype.is_floating_point():
    """`quantile` with `q` in percent. `numpy.percentile(a, q)`."""
    return quantile(xs, q / 100.0, method)


def percentile[
    dtype: DType, LayoutType: TensorLayout, m: Int
](
    xs: Tensor[dtype, LayoutType],
    q: Static[dtype, m],
    method: StaticString = "linear",
) raises -> Static[dtype, m] where (dtype.is_floating_point() and m > 0):
    """Several percentiles at once."""
    var sorted = _sorted_host(xs, False)
    var qs = q.to_host()
    var out = List[Scalar[dtype]](capacity=m)
    for i in range(m):
        out.append(_read[dtype](sorted, Float64(qs[i]) / 100.0, method))
    return Static[dtype, m](q.context(), out^)


def nanquantile[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType], q: Float64, method: StaticString = "linear"
) raises -> Scalar[dtype] where dtype.is_floating_point():
    """`quantile` over the non-NaN elements. `numpy.nanquantile`. Raises
    when every element is NaN, where NumPy warns and returns NaN."""
    return Scalar[dtype](_quantile_of(_sorted_host(xs, True), q, method))


def nanquantile[
    dtype: DType, LayoutType: TensorLayout, m: Int
](
    xs: Tensor[dtype, LayoutType],
    q: Static[dtype, m],
    method: StaticString = "linear",
) raises -> Static[dtype, m] where (dtype.is_floating_point() and m > 0):
    """Several NaN-ignoring quantiles at once."""
    var sorted = _sorted_host(xs, True)
    var qs = q.to_host()
    var out = List[Scalar[dtype]](capacity=m)
    for i in range(m):
        out.append(Scalar[dtype](_quantile_of(sorted, Float64(qs[i]), method)))
    return Static[dtype, m](q.context(), out^)


def nanpercentile[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType], q: Float64, method: StaticString = "linear"
) raises -> Scalar[dtype] where dtype.is_floating_point():
    """`percentile` over the non-NaN elements. `numpy.nanpercentile`."""
    return nanquantile(xs, q / 100.0, method)


def nanmedian[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The median of the non-NaN elements. `numpy.nanmedian` -- the
    `"linear"` quantile at `1/2`, which is `median`'s even-count average."""
    return nanquantile(xs, 0.5)


def iqr[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType],
    interpolation: StaticString = "linear",
    nan_policy: StaticString = "propagate",
) raises -> Scalar[dtype] where dtype.is_floating_point():
    """The interquartile range, the 75th percentile less the 25th.
    `scipy.stats.iqr(x, interpolation=..., nan_policy=...)`.

    `interpolation` is any `quantile` method; `nan_policy` is
    `"propagate"` (a NaN makes the answer NaN, SciPy's default) or
    `"omit"`. The robust spread: unmoved by the outliers that dominate the
    standard deviation.
    """
    if not (nan_policy == "propagate" or nan_policy == "omit"):
        raise Error("iqr: nan_policy must be 'propagate' or 'omit'")
    var sorted = _sorted_host(xs, nan_policy == "omit")
    if len(sorted) == 0:
        return _nan[dtype]()
    return Scalar[dtype](
        _quantile_of(sorted, 0.75, interpolation)
        - _quantile_of(sorted, 0.25, interpolation)
    )
