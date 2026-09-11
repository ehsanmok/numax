"""NaN-ignoring reductions over `numax.core.array.Tensor`: `nansum`,
`nanprod`, `nanmean`, `nanvar`, `nanstd`, `nanmin`, `nanmax`, and the
counts they rest on. `nanmedian` and `nanquantile` are in `quantiles.mojo`
with the other order statistics.

**Tier 2, by composition.** None of these has a kernel of its own: each
is `isnan`, a `select` against a fill tensor, and one of the existing
reductions -- `nansum` is `sum(select(isnan(x), 0, x))`, `nanmin` is `min`
with the NaNs replaced by `+inf`, `nanmean` divides `nansum` by the count
of non-NaN elements from `count_nonzero(isnan(x))`. They therefore run
exactly where those primitives run, which today is the host over a
CPU-resident tensor -- the whole-tensor `sum`/`min`/`max`, `isnan` and
`select` all walk host memory, as `numax.stats`' package docstring says of
the reductions -- at the cost of one mask and one fill pass per call, the
same NumPy pays. When the primitives gain a `gpu` parameter, these follow
without a change of their own.

NumPy's edge cases are kept where they are well defined -- a tensor of only
NaNs has `nansum == 0` and `nanprod == 1` -- and raised where NumPy warns
and returns NaN or infinity: `nanmean`, `nanvar`, `nanmin` and `nanmax` of
nothing but NaN have no value to give.

## The MAX gate

Nothing to add to what the plain reductions already do. The mask is
`numax.core.logic.isnan`, the fill `numax.core.sorting.select`.
"""

from std.utils.numerics import inf as _inf

from layout.tile_layout import TensorLayout

from ..core.array import Tensor, full_like
from ..core.elementwise import sqrt as _sqrt
from ..core.logic import isnan
from ..core.ops import multiply, subtract
from ..core.sorting import count_nonzero, select
from .statistics import max as _max, min as _min, sum as _sum


def _filled[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType], fill: Scalar[dtype]) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """`xs` with every NaN replaced by `fill`."""
    var mask = isnan(xs)
    var value = full_like(xs, fill)
    return select(mask, value, xs)


def _nan_count[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> Int where dtype.is_floating_point():
    return count_nonzero(isnan(xs))


def nansum[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The sum of the non-NaN elements; `0` if there are none.
    `numpy.nansum`."""
    return _sum(_filled(xs, Scalar[dtype](0)))


def nanprod[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The product of the non-NaN elements; `1` if there are none.
    `numpy.nanprod`."""
    from .statistics import prod as _prod

    return _prod(_filled(xs, Scalar[dtype](1)))


def nanmean[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The mean of the non-NaN elements. `numpy.nanmean`. Raises when every
    element is NaN, where NumPy warns and returns NaN."""
    var kept = xs.size() - _nan_count(xs)
    if kept == 0:
        raise Error("nanmean: every element is NaN")
    return nansum(xs) / Scalar[dtype](kept)


def nanvar[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType], ddof: Int = 0) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The variance of the non-NaN elements, with `ddof` degrees of freedom
    removed from the count. `numpy.nanvar(a, ddof=ddof)`.

    Two passes -- the mean, then the mean of squared deviations with the
    NaN positions zeroed -- which is the textbook form rather than
    `variance`'s single Welford fold, because the mask has to be applied to
    a deviation the fold never materializes. Raises when fewer than `ddof +
    1` elements are not NaN.
    """
    var kept = xs.size() - _nan_count(xs)
    if kept <= ddof:
        raise Error("nanvar: not enough non-NaN elements for ddof ", ddof)
    var centre = nanmean(xs)
    var deviation = subtract(xs, centre)
    var squares = multiply(deviation, deviation)
    return _sum(_filled(squares, Scalar[dtype](0))) / Scalar[dtype](kept - ddof)


def nanstd[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType], ddof: Int = 0) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The square root of `nanvar`. `numpy.nanstd`."""
    from std.math import sqrt

    return sqrt(nanvar(xs, ddof))


def nanmin[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The smallest non-NaN element. `numpy.nanmin`. Raises when every
    element is NaN."""
    if _nan_count(xs) == xs.size():
        raise Error("nanmin: every element is NaN")
    return _min(_filled(xs, _inf[dtype]()))


def nanmax[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The largest non-NaN element. `numpy.nanmax`. Raises when every
    element is NaN."""
    if _nan_count(xs) == xs.size():
        raise Error("nanmax: every element is NaN")
    return _max(_filled(xs, -_inf[dtype]()))
