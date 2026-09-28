"""NaN-ignoring reductions over `numax.core.tensor.Tensor`: `nansum`,
`nanprod`, `nanmean`, `nanvar`, `nanstd`, `nanmin`, `nanmax`, and the
counts they rest on. `nanmedian` and `nanquantile` are in `quantiles.mojo`
with the other order statistics.

**Tier 2, by composition.** None of these has a kernel of its own: each
is `isnan`, a `select` against a fill tensor, and one of the existing
reductions -- `nansum` is `sum(select(isnan(x), 0, x))`, `nanmin` is `min`
with the NaNs replaced by `+inf`, `nanmean` divides `nansum` by the count
of non-NaN elements from `count_nonzero(isnan(x))`. They therefore run
exactly where those primitives run, at the cost of one mask and one fill
pass per call, the same NumPy pays. The reduction underneath is now a MAX
monoid on either target, and every piece of the composition -- `isnan`,
`select`, the fill, the copy, `count_nonzero` -- takes a `gpu` parameter,
so each of these takes one too and forwards it: at `gpu=True` the whole
call stays on the device and only the scalar answer comes back. `nansum`
and `nanprod` inherit the reassociation the monoid brings with it.

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

from ..core.tensorlike import TensorLike, dim, is_row_major
from ..core.tensor import copy, Tensor, full_like
from ..core.elementwise import sqrt as _sqrt
from ..core.logic import isnan
from ..core.ops import multiply, subtract
from ..core.sorting import count_nonzero, select
from .statistics import max as _max, min as _min, sum as _sum


def _filled[
    T: TensorLike, gpu: Bool = False
](xs: T, fill: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """`xs` with every NaN replaced by `fill`."""
    var mask = isnan[gpu=gpu](xs)
    var value = full_like(xs, fill)
    # `select`'s same-layout form wants both branches at one type, and a
    # generic `T` is not `Tensor[T.dtype, T.LayoutType]` to the checker,
    # and a borrowed `xs` cannot become a `TensorView`; the copy is one pass
    # on `xs`'s own device.
    return select[gpu=gpu](mask, value, copy(xs))


def _nan_count[
    T: TensorLike, gpu: Bool = False
](xs: T) raises -> Int where is_row_major[T] and T.dtype.is_floating_point():
    return count_nonzero[gpu=gpu](isnan[gpu=gpu](xs))


def nansum[
    T: TensorLike, gpu: Bool = False
](xs: T) raises -> Scalar[T.dtype] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The sum of the non-NaN elements; `0` if there are none.
    `numpy.nansum`.

    Parameters:
        T: The tensor type of `xs`, row-major with a floating-point dtype.
        gpu: Run the mask, fill and reduction on the tensor's device; a
            residency mismatch falls back to the host with a notice.

    Args:
        xs: The tensor to reduce, walked flat over all its elements.

    Returns:
        The sum of the non-NaN elements, or `0` if every element is NaN.

    Raises:
        If a residency mismatch occurs under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    return _sum[gpu=gpu](_filled[gpu=gpu](xs, Scalar[dtype](0)))


def nanprod[
    T: TensorLike, gpu: Bool = False
](xs: T) raises -> Scalar[T.dtype] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The product of the non-NaN elements; `1` if there are none.
    `numpy.nanprod`.

    Parameters:
        T: The tensor type of `xs`, row-major with a floating-point dtype.
        gpu: Run the mask, fill and reduction on the tensor's device; a
            residency mismatch falls back to the host with a notice.

    Args:
        xs: The tensor to reduce, walked flat over all its elements.

    Returns:
        The product of the non-NaN elements, or `1` if every element is NaN.

    Raises:
        If a residency mismatch occurs under the `"raise"` fallback policy.
    """
    comptime dtype = T.dtype
    from .statistics import prod as _prod

    return _prod[gpu=gpu](_filled[gpu=gpu](xs, Scalar[dtype](1)))


def nanmean[
    T: TensorLike, gpu: Bool = False
](xs: T) raises -> Scalar[T.dtype] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The mean of the non-NaN elements. `numpy.nanmean`. Raises when every
    element is NaN, where NumPy warns and returns NaN.

    Parameters:
        T: The tensor type of `xs`, row-major with a floating-point dtype.
        gpu: Run the mask, fill and reduction on the tensor's device; a
            residency mismatch falls back to the host with a notice.

    Args:
        xs: The tensor to reduce, walked flat over all its elements.

    Returns:
        The sum of the non-NaN elements divided by their count.

    Raises:
        If every element of `xs` is NaN.
    """
    comptime dtype = T.dtype
    var kept = xs.size() - _nan_count[gpu=gpu](xs)
    if kept == 0:
        raise Error("nanmean: every element is NaN")
    return nansum[gpu=gpu](xs) / Scalar[dtype](kept)


def nanvar[
    T: TensorLike, gpu: Bool = False
](xs: T, ddof: Int = 0) raises -> Scalar[T.dtype] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The variance of the non-NaN elements, with `ddof` degrees of freedom
    removed from the count. `numpy.nanvar(a, ddof=ddof)`.

    Two passes -- the mean, then the mean of squared deviations with the
    NaN positions zeroed -- which is the textbook form rather than
    `variance`'s single Welford fold, because the mask has to be applied to
    a deviation the fold never materializes. Raises when fewer than `ddof +
    1` elements are not NaN.

    Parameters:
        T: The tensor type of `xs`, row-major with a floating-point dtype.
        gpu: Run the mask, fill and reduction on the tensor's device; a
            residency mismatch falls back to the host with a notice.

    Args:
        xs: The tensor to reduce, walked flat over all its elements.
        ddof: Degrees of freedom subtracted from the non-NaN count in the
            divisor.

    Returns:
        The mean squared deviation of the non-NaN elements from their mean,
        over `count - ddof`.

    Raises:
        If `ddof` or fewer elements of `xs` are not NaN.
    """
    comptime dtype = T.dtype
    var kept = xs.size() - _nan_count[gpu=gpu](xs)
    if kept <= ddof:
        raise Error("nanvar: not enough non-NaN elements for ddof ", ddof)
    var center = nanmean[gpu=gpu](xs)
    var deviation = subtract[gpu=gpu](xs, center)
    var squares = multiply[gpu=gpu](deviation, deviation)
    return _sum[gpu=gpu](_filled[gpu=gpu](squares, Scalar[dtype](0))) / Scalar[
        dtype
    ](kept - ddof)


def nanstd[
    T: TensorLike, gpu: Bool = False
](xs: T, ddof: Int = 0) raises -> Scalar[T.dtype] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The square root of `nanvar`. `numpy.nanstd`.

    Parameters:
        T: The tensor type of `xs`, row-major with a floating-point dtype.
        gpu: Run the mask, fill and reduction on the tensor's device; a
            residency mismatch falls back to the host with a notice.

    Args:
        xs: The tensor to reduce, walked flat over all its elements.
        ddof: Degrees of freedom subtracted from the non-NaN count in the
            divisor.

    Returns:
        The square root of `nanvar(xs, ddof)`.

    Raises:
        If `ddof` or fewer elements of `xs` are not NaN.
    """
    from std.math import sqrt

    return sqrt(nanvar[gpu=gpu](xs, ddof))


def nanmin[
    T: TensorLike, gpu: Bool = False
](xs: T) raises -> Scalar[T.dtype] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The smallest non-NaN element. `numpy.nanmin`. Raises when every
    element is NaN.

    Parameters:
        T: The tensor type of `xs`, row-major with a floating-point dtype.
        gpu: Run the mask, fill and reduction on the tensor's device; a
            residency mismatch falls back to the host with a notice.

    Args:
        xs: The tensor to reduce, walked flat over all its elements.

    Returns:
        The smallest non-NaN element of `xs`.

    Raises:
        If every element of `xs` is NaN.
    """
    comptime dtype = T.dtype
    if _nan_count[gpu=gpu](xs) == xs.size():
        raise Error("nanmin: every element is NaN")
    return _min[gpu=gpu](_filled[gpu=gpu](xs, _inf[dtype]()))


def nanmax[
    T: TensorLike, gpu: Bool = False
](xs: T) raises -> Scalar[T.dtype] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The largest non-NaN element. `numpy.nanmax`. Raises when every
    element is NaN.

    Parameters:
        T: The tensor type of `xs`, row-major with a floating-point dtype.
        gpu: Run the mask, fill and reduction on the tensor's device; a
            residency mismatch falls back to the host with a notice.

    Args:
        xs: The tensor to reduce, walked flat over all its elements.

    Returns:
        The largest non-NaN element of `xs`.

    Raises:
        If every element of `xs` is NaN.
    """
    comptime dtype = T.dtype
    if _nan_count[gpu=gpu](xs) == xs.size():
        raise Error("nanmax: every element is NaN")
    return _max[gpu=gpu](_filled[gpu=gpu](xs, -_inf[dtype]()))
