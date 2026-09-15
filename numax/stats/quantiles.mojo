"""Order statistics over `numax.core.array.Tensor`: `quantile`,
`percentile`, their NaN-ignoring forms, `nanmedian` and `iqr`, with every
`method` NumPy names.

**Tier 2, host-side**, the way `median` is -- but by **selection, not a
sort**. A quantile reads one or two order statistics, so the tensor comes
to the host once and one `O(n)` three-way quickselect puts exactly those
in place; nothing else is ordered. The selection is `_select_pair` here
rather than `std.builtin.sort.partition`, whose two-way split goes
quadratic on a near-constant sample -- the reason is in its docstring. The values arrive at
`dtype` rather than widened to `Float64`, and only the final blend between
the two neighbours is evaluated at `Float64`, which is where NumPy
evaluates it too.

MAX-first: there is no partition, selection or quantile kernel in `linalg`,
`nn`, `algorithm` or `layout`. `nn.top_k` is the device selection MAX does
ship, and it is what `numax.core.sorting.top_k` delegates to -- but its
cost grows with `k`, and the median's `k = n / 2 + 1` is its worst case: it
would order half the tensor to answer what one `O(n)` selection answers.
So the selection here runs on a host copy, and a caller who wants the `k`
smallest for a small `k` on a device should call `top_k[largest=False]`
directly. (`nn.argsort` is *not* host-only, as an earlier revision of this
docstring said: it has a real GPU path, `numax/core/sorting.mojo:28-29`.
What it does not give is a value sort or a selection.)

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

All thirteen are the same shape once written as positions: `v[lo] + w
(v[hi] - v[lo])` over the ascending sample, with `hi` equal to `lo` or
`lo + 1`. `_positions` is that table, and it is what lets every method be a
selection rather than a sort.
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


def _neighbours(index: Float64) -> Tuple[Int, Int, Float64]:
    """The two order statistics a continuous method interpolates between,
    and the weight on the upper one."""
    var lo = Int(_floor(index))
    var hi = Int(_ceil(index))
    return (lo, hi, index - Float64(lo))


def _single(index: Int) -> Tuple[Int, Int, Float64]:
    """One order statistic, picked rather than interpolated."""
    return (index, index, 0.0)


def _round_half_even(x: Float64) -> Int:
    var lo = _floor(x)
    var fraction = x - lo
    if fraction > 0.5:
        return Int(lo) + 1
    if fraction < 0.5:
        return Int(lo)
    return Int(lo) if Int(lo) % 2 == 0 else Int(lo) + 1


def _positions(
    q: Float64, n: Int, method: StaticString
) raises -> Tuple[Int, Int, Float64]:
    """The one or two order statistics `method` reads at `q`, as
    `(lo, hi, weight)`.

    The quantile is `v[lo] + weight (v[hi] - v[lo])` over the ascending
    sample `v`, where `0 <= lo <= hi <= n - 1` and `hi` is `lo` or
    `lo + 1`. Writing all thirteen methods this way is what makes a
    quantile a selection: two indices out of `n`, never an ordering of all
    of them.
    """
    if n == 0:
        raise Error("quantile: no values")
    if q < 0 or q > 1:
        raise Error("quantile: q must lie in [0, 1]")
    if method == "linear":
        return _neighbours(_virtual_index(q, n, 1.0, 1.0))
    if method == "interpolated_inverted_cdf":
        return _neighbours(_virtual_index(q, n, 0.0, 1.0))
    if method == "hazen":
        return _neighbours(_virtual_index(q, n, 0.5, 0.5))
    if method == "weibull":
        return _neighbours(_virtual_index(q, n, 0.0, 0.0))
    if method == "median_unbiased":
        return _neighbours(_virtual_index(q, n, 1.0 / 3.0, 1.0 / 3.0))
    if method == "normal_unbiased":
        return _neighbours(_virtual_index(q, n, 0.375, 0.375))
    var linear = _virtual_index(q, n, 1.0, 1.0)
    if method == "lower":
        return _single(Int(_floor(linear)))
    if method == "higher":
        return _single(Int(_ceil(linear)))
    if method == "midpoint":
        return (Int(_floor(linear)), Int(_ceil(linear)), 0.5)
    if method == "nearest":
        return _single(_round_half_even(linear))
    # The three sample-picking rules on `q * n`.
    var position = q * Float64(n)
    var whole = Int(_floor(position))
    var fraction = position - Float64(whole)
    if method == "inverted_cdf":
        var index = whole if fraction > 0 else whole - 1
        return _single(min(max(index, 0), n - 1))
    if method == "averaged_inverted_cdf":
        if fraction > 0:
            return _single(min(max(whole, 0), n - 1))
        var above = min(max(whole, 0), n - 1)
        var below = min(max(whole - 1, 0), n - 1)
        return (below, above, 0.5)
    if method == "closest_observation":
        # Hyndman-Fan 3: `inverted_cdf` at `q n - 1/2`, taking the even order
        # statistic on a tie, which is NumPy's `_closest_observation`.
        var shifted = position - 0.5
        var j = Int(_floor(shifted))
        var g = shifted - Float64(j)
        var gamma = 0 if (g == 0 and j % 2 == 0) else 1
        return _single(min(max(j - 1 + gamma, 0), n - 1))
    raise Error("quantile: unknown method '", method, "'")


def _median3[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype], c: Scalar[dtype]) -> Scalar[dtype]:
    """The middle of three, the pivot every round of `_select_pair` picks."""
    if a < b:
        if b < c:
            return b
        return c if a < c else a
    if a < c:
        return a
    return c if b < c else b


def _select_pair[
    dtype: DType
](mut values: List[Scalar[dtype]], lo: Int, hi: Int) -> Tuple[
    Scalar[dtype], Scalar[dtype]
]:
    """The `lo`-th and `hi`-th smallest of `values`, by selection.

    `hi` is `lo` or `lo + 1`; `values` is permuted in place and is *not*
    left sorted. This is quickselect with a **three-way** (Dutch-flag)
    partition and a median-of-three pivot: one `O(n)` pass per round,
    halving the live range, where the pre-0.2 code sorted all `n`.

    The three-way split is not a refinement, it is the reason this is
    written here rather than calling `std.builtin.sort.partition`. That
    function's split is two-way, so an input whose live range is all one
    value never shrinks and it goes quadratic -- measured at 89 s for
    `2^18` equal elements, and 12 s for `2^18` elements with a *single*
    different one, against 1 ms for distinct values. A constant column is
    ordinary data, so the library cannot have that cliff. Equal elements
    land in the middle block here and the range ends in one round.

    A `rounds` budget of `32 + 3 log2 n` falls back to one full sort, so
    even an adversarial pivot sequence costs `O(n log n)` rather than
    `O(n^2)`.

    The upper neighbour comes out of the same walk: when `lo` lands inside
    the equal block with room to spare it *is* the neighbour, and otherwise
    the neighbour is the smallest of the strictly-greater side, which is
    the last round's right stripe rather than the whole tail.
    """
    var n = len(values)
    var left = 0
    var right = n - 1
    var scan_from = lo + 1
    var scan_to = n - 1
    var rounds = 0
    var budget = 32
    var span = n
    while span > 1:
        span >>= 1
        budget += 3
    while left < right:
        rounds += 1
        if rounds > budget:
            _sort(values)
            scan_from = lo + 1
            scan_to = lo + 1
            break
        var mid = left + ((right - left) >> 1)
        var pivot = _median3(values[left], values[mid], values[right])
        var lt = left
        var i = left
        var gt = right
        while i <= gt:
            var x = values[i]
            if x < pivot:
                values[i] = values[lt]
                values[lt] = x
                lt += 1
                i += 1
            elif pivot < x:
                values[i] = values[gt]
                values[gt] = x
                gt -= 1
            else:
                i += 1
        if lo < lt:
            right = lt - 1
            continue
        if lo > gt:
            left = gt + 1
            continue
        if lo < gt:
            scan_from = lo + 1
            scan_to = lo + 1
        else:
            scan_from = gt + 1
            scan_to = right if gt < right else n - 1
        break
    var low = values[lo]
    if hi <= lo or lo + 1 >= n:
        return (low, low)
    var high = values[scan_from]
    for j in range(scan_from + 1, scan_to + 1):
        if values[j] < high:
            high = values[j]
    return (low, high)


def _blend[
    dtype: DType
](low: Scalar[dtype], high: Scalar[dtype], weight: Float64) -> Scalar[dtype]:
    """`low + weight (high - low)`, evaluated at `Float64` as NumPy
    evaluates it -- the one step that is not at `dtype`, and it touches two
    values rather than `n`."""
    var a = Float64(low)
    return Scalar[dtype](a + weight * (Float64(high) - a))


def _quantile_of[
    dtype: DType
](
    mut values: List[Scalar[dtype]], q: Float64, method: StaticString
) raises -> Scalar[dtype]:
    """One quantile of `values` in any order, selecting the one or two
    order statistics it needs and permuting `values` in place."""
    var at = _positions(q, len(values), method)
    var pair = _select_pair(values, at[0], at[1])
    return _blend(pair[0], pair[1], at[2])


def _quantile_sorted[
    dtype: DType
](
    values: List[Scalar[dtype]], q: Float64, method: StaticString
) raises -> Scalar[dtype]:
    """One quantile of an already ascending `values`: two reads, no
    selection. The vector overloads use this above the threshold
    `_select_route` sets."""
    var at = _positions(q, len(values), method)
    return _blend(values[at[0]], values[at[1]], at[2])


def _select_route(m: Int, n: Int) -> Bool:
    """Whether `m` quantiles of `n` values are cheaper by selection than by
    one sort.

    A `partition` touches about `2 n` elements and the scan for the upper
    neighbour another `n`, so `m` selections cost about `3 m n`; one sort
    costs about `n log2 n`. Selection wins while `3 m < log2 n` -- six
    quantiles of a million values, one of a thousand, none of sixteen.
    Above the threshold the overload sorts once and reads `m` pairs of
    indices, which is what every overload here did before 0.2.
    """
    var bits = 0
    var rest = n
    while rest > 1:
        rest >>= 1
        bits += 1
    return 3 * m < bits


def _host_values[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType], drop_nan: Bool) raises -> List[Scalar[dtype]]:
    """The tensor's values on the host at `dtype`, in no particular order,
    NaNs removed when `drop_nan`. Without `drop_nan` a NaN anywhere empties
    the list, which every caller reads as "the answer is NaN" -- NumPy's
    propagation rule, spelled once.

    `to_host` already hands over an owned `List`, so this takes it and
    compacts in place rather than filtering into a second one: at `2^24`
    elements the copy this avoids is the same size as the download.
    """
    var values = xs.to_host()
    if not drop_nan:
        for i in range(len(values)):
            if values[i] != values[i]:
                return List[Scalar[dtype]]()
        return values^
    var kept = 0
    for i in range(len(values)):
        if values[i] == values[i]:
            values[kept] = values[i]
            kept += 1
    values.resize(kept, Scalar[dtype](0))
    return values^


def _read[
    dtype: DType
](
    mut values: List[Scalar[dtype]], q: Float64, method: StaticString
) raises -> Scalar[dtype]:
    """One quantile, or NaN when `_host_values` reported a NaN."""
    if len(values) == 0:
        return _nan[dtype]()
    return _quantile_of(values, q, method)


def _read_many[
    dtype: DType, m: Int
](
    mut values: List[Scalar[dtype]],
    qs: List[Scalar[dtype]],
    scale: Float64,
    method: StaticString,
    propagate: Bool,
) raises -> List[Scalar[dtype]]:
    """`m` quantiles of one sample, routed by `_select_route`.

    `scale` divides each `q` on the way in, so the percentile overloads
    pass `100` and the quantile ones pass `1`. `propagate` returns NaN for
    an empty `values` instead of raising, which is the plain family's rule;
    the `nan*` family passes `False` and lets `_positions` raise.
    """
    var out = List[Scalar[dtype]](capacity=m)
    if propagate and len(values) == 0:
        for _ in range(m):
            out.append(_nan[dtype]())
        return out^
    if _select_route(m, len(values)):
        for i in range(m):
            out.append(_quantile_of(values, Float64(qs[i]) / scale, method))
        return out^
    _sort(values)
    for i in range(m):
        out.append(_quantile_sorted(values, Float64(qs[i]) / scale, method))
    return out^


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

    One `O(n)` selection on a host copy, not a sort: see the module
    docstring for why the selection is not on the device.
    """
    var values = _host_values(xs, False)
    return _read(values, q, method)


def quantile[
    dtype: DType, LayoutType: TensorLayout, m: Int
](
    xs: Tensor[dtype, LayoutType],
    q: Static[dtype, m],
    method: StaticString = "linear",
) raises -> Static[dtype, m] where (dtype.is_floating_point() and m > 0):
    """Several quantiles at once: `m` selections, or one sort when `m` is
    large enough to pay for it. `numpy.quantile(a, [q0, q1, ...])`.

    `_select_route` documents the threshold -- selection while
    `3 m < log2 n`, one sort above it -- and the answer is the same either
    way.
    """
    var values = _host_values(xs, False)
    var out = _read_many[dtype, m](values, q.to_host(), 1.0, method, True)
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
    """Several percentiles at once, on `quantile`'s selection route."""
    var values = _host_values(xs, False)
    var out = _read_many[dtype, m](values, q.to_host(), 100.0, method, True)
    return Static[dtype, m](q.context(), out^)


def nanquantile[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType], q: Float64, method: StaticString = "linear"
) raises -> Scalar[dtype] where dtype.is_floating_point():
    """`quantile` over the non-NaN elements. `numpy.nanquantile`. Raises
    when every element is NaN, where NumPy warns and returns NaN."""
    var values = _host_values(xs, True)
    return _quantile_of(values, q, method)


def nanquantile[
    dtype: DType, LayoutType: TensorLayout, m: Int
](
    xs: Tensor[dtype, LayoutType],
    q: Static[dtype, m],
    method: StaticString = "linear",
) raises -> Static[dtype, m] where (dtype.is_floating_point() and m > 0):
    """Several NaN-ignoring quantiles at once, on the same route."""
    var values = _host_values(xs, True)
    var out = _read_many[dtype, m](values, q.to_host(), 1.0, method, False)
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
    `"linear"` quantile at `1/2`, which is `median`'s even-count average,
    and like `median` a selection rather than a sort."""
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

    Two selections over one host copy -- the second runs on what the first
    left behind, which is a permutation of the same sample, so the answer
    does not depend on the order they run in.
    """
    if not (nan_policy == "propagate" or nan_policy == "omit"):
        raise Error("iqr: nan_policy must be 'propagate' or 'omit'")
    var values = _host_values(xs, nan_policy == "omit")
    if len(values) == 0:
        return _nan[dtype]()
    var upper = _quantile_of(values, 0.75, interpolation)
    var lower = _quantile_of(values, 0.25, interpolation)
    return upper - lower
