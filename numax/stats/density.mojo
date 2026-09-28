"""Empirical distribution and kernel density estimation over
`numax.core.tensor.Tensor`: `ecdf` and `gaussian_kde`, SciPy's.

**Tier 2.** `ecdf` is `unique_counts` and a running sum of the counts;
`gaussian_kde` evaluates `sum_i N(x; x_i, h^2) / n` at each query point,
one lane per point summing over the data -- `O(n m)`, the direct form
SciPy uses too. Both run where their inputs live at `gpu=True`.

## The MAX gate

Nothing: MAX has neither. **Extend.**
"""

from std.math import exp as _exp, sqrt as _sqrt

from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise

from ..core._drive import _check_device, _notice
from ..core.sorting import unique_counts
from ..core.tensorlike import TensorLike, dim, is_row_major
from ..core.tensor import (
    Dynamic,
    Static,
    _dyn_shape,
    _same_order,
    _scan_device,
    copy as _copy,
)
from ..core.ops import multiply as _multiply, subtract as _subtract
from .statistics import _target, sum as _tsum

comptime _SQRT_2PI = 2.5066282746310002


struct ECDFResult[dtype: DType](Movable):
    """What `ecdf` returns: the distinct sample values and the empirical
    CDF at each, SciPy's `ecdf(...).cdf.quantiles` and `.probabilities`."""

    var quantiles: Dynamic[Self.dtype, 1]
    """The distinct sample values, ascending."""
    var probabilities: Dynamic[Self.dtype, 1]
    """The fraction of the sample at or below each value."""

    def __init__(
        out self,
        var quantiles: Dynamic[Self.dtype, 1],
        var probabilities: Dynamic[Self.dtype, 1],
    ):
        """Build from the two arrays.

        Args:
            quantiles: The distinct values.
            probabilities: The empirical CDF at each.
        """
        self.quantiles = quantiles^
        self.probabilities = probabilities^


def ecdf[
    T: TensorLike, gpu: Bool = False
](sample: T) raises -> ECDFResult[T.dtype] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """The empirical cumulative distribution function of `sample`.
    `scipy.stats.ecdf(sample).cdf`.

    `unique_counts` gives the distinct values and how often each occurs;
    the running sum of the counts over the sample size is the CDF at each.
    At `gpu=True` the sort, the counts and the scan are device work.

    Parameters:
        T: The tensor type of `sample`, row-major and floating-point.
        gpu: Whether to compute on `sample`'s device.

    Args:
        sample: The observations, read flat.

    Returns:
        An `ECDFResult` with the distinct values and the empirical CDF.

    Raises:
        If `sample` is empty, or a device operation fails.
    """
    comptime dtype = T.dtype
    var n = sample.size()
    if n == 0:
        raise Error("ecdf: the sample is empty")
    var counted = unique_counts[gpu=gpu](sample)
    var k = counted.counts.size()
    var ctx = sample.context()
    var probabilities = Dynamic[dtype, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](k))
    )
    var scale = Scalar[dtype](n)
    if _check_device[T, gpu](sample):
        comptime if gpu:
            var running = _scan_device["sum"](
                counted.counts, row_major(_dyn_shape[1](k)), k, 1
            )
            var rp = running.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
            var pp = probabilities.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

            @always_inline
            def divide[
                width: Int, alignment: Int = 1
            ](coord: Coord) {var rp, var pp, var scale}:
                var i = coord_to_index_list(coord)[0]
                pp[unsafe_offset=i] = Scalar[dtype](rp[unsafe_offset=i]) / scale

            elementwise[simd_width=1, target="gpu"](divide, Coord(k), ctx)
            ctx.synchronize()
            _ = running^
            return ECDFResult(_copy(counted.values), probabilities^)
    else:
        _notice[gpu]("ecdf")
    var counts = counted.counts.to_host()
    var values = List[Scalar[dtype]](capacity=k)
    var running = Int64(0)
    for i in range(k):
        running += counts[i]
        values.append(Scalar[dtype](running) / scale)
    probabilities.copy_from_host(values)
    return ECDFResult(_copy(counted.values), probabilities^)


struct gaussian_kde[dtype: DType](Movable):
    """A one-dimensional Gaussian kernel density estimate.
    `scipy.stats.gaussian_kde` for a 1-D dataset.

    The bandwidth is the sample standard deviation (`ddof = 1`) times
    Scott's factor `n^(-1/5)` (the default) or Silverman's
    `(3 n / 4)^(-1/5)`, SciPy's two rules at `d = 1`; `with_factor` takes
    the factor as a number, SciPy's `bw_method=<scalar>`.
    """

    var dataset: Dynamic[Self.dtype, 1]
    """The data the estimate is built on, a copy on its device."""
    var bandwidth: Float64
    """The kernel's standard deviation, `factor * std(dataset)`."""
    var factor: Float64
    """The bandwidth factor the rule gave."""

    def __init__(
        out self,
        var dataset: Dynamic[Self.dtype, 1],
        bandwidth: Float64,
        factor: Float64,
    ):
        """Build from a flat copy of the data and its bandwidth; the
        rules are `create` and `with_factor`.

        Args:
            dataset: The observations.
            bandwidth: The kernel's standard deviation.
            factor: The factor it came from.
        """
        self.dataset = dataset^
        self.bandwidth = bandwidth
        self.factor = factor

    @staticmethod
    def create[
        T: TensorLike, gpu: Bool = False
    ](dataset: T, bw_method: StaticString = "scott") raises -> gaussian_kde[
        Self.dtype
    ] where (
        T.dtype == Self.dtype
        and is_row_major[T]
        and T.dtype.is_floating_point()
    ):
        """The estimate of `dataset` under a named bandwidth rule.

        Parameters:
            T: The tensor type of `dataset`, read flat.
            gpu: Whether to take the moments on `dataset`'s device.

        Args:
            dataset: The observations, at least two.
            bw_method: `"scott"` (the default) or `"silverman"`.

        Returns:
            The estimate.

        Raises:
            If there are fewer than two observations, `bw_method` is not a
            known rule, or the data have no spread.
        """
        var n = dataset.size()
        if n < 2:
            raise Error("gaussian_kde: at least two observations are needed")
        var factor: Float64
        if bw_method == "scott":
            factor = Float64(n) ** (-1.0 / 5.0)
        elif bw_method == "silverman":
            factor = (Float64(n) * 3.0 / 4.0) ** (-1.0 / 5.0)
        else:
            raise Error(
                "gaussian_kde: bw_method is 'scott' or 'silverman'; pass a"
                " number to `gaussian_kde.with_factor`"
            )
        return gaussian_kde[Self.dtype].with_factor[gpu=gpu](dataset, factor)

    @staticmethod
    def with_factor[
        T: TensorLike, gpu: Bool = False
    ](dataset: T, factor: Float64) raises -> gaussian_kde[Self.dtype] where (
        T.dtype == Self.dtype
        and is_row_major[T]
        and T.dtype.is_floating_point()
    ):
        """The estimate with an explicit bandwidth factor, SciPy's
        `bw_method=<number>`.

        Parameters:
            T: The tensor type of `dataset`, read flat.
            gpu: Whether to take the moments on `dataset`'s device.

        Args:
            dataset: The observations, at least two.
            factor: The bandwidth factor.

        Returns:
            The estimate.

        Raises:
            If there are fewer than two observations, or the data have no
            spread.
        """
        var n = dataset.size()
        if n < 2:
            raise Error("gaussian_kde: at least two observations are needed")
        var mean = Float64(_tsum[gpu=gpu](dataset)) / Float64(n)
        var sq = 0.0
        comptime if gpu:
            var d = _subtract[gpu=True](dataset, Scalar[T.dtype](mean))
            sq = Float64(_tsum[gpu=True](_multiply[gpu=True](d, d)))
        else:
            var values = dataset.to_host()
            for i in range(n):
                sq += (Float64(values[i]) - mean) ** 2
        var std = _sqrt(sq / Float64(n - 1))
        if std == 0.0:
            raise Error("gaussian_kde: the data have no spread")
        var copy = rebind_var[Dynamic[Self.dtype, 1]](
            _same_order(dataset, row_major(_dyn_shape[1](n)))
        )
        return gaussian_kde[Self.dtype](copy^, factor * std, factor)

    def evaluate[
        P: TensorLike, gpu: Bool = False
    ](self, points: P) raises -> Dynamic[Self.dtype, 1] where (
        P.dtype == Self.dtype
        and is_row_major[P]
        and Self.dtype.is_floating_point()
    ):
        """The density at each of `points`. `gaussian_kde.evaluate`.

        One lane per point summing the data's kernels, on the points'
        device at `gpu=True`.

        Parameters:
            P: The tensor type of `points`, read flat.
            gpu: Whether to evaluate on the device.

        Args:
            points: Where to evaluate.

        Returns:
            The estimated density at each point.

        Raises:
            If a device operation fails.
        """
        comptime dt = Self.dtype
        var m = points.size()
        var n = self.dataset.size()
        var ctx = points.context()
        var out = Dynamic[dt, 1]._uninitialized(
            ctx, row_major(_dyn_shape[1](m))
        )
        var xp = points.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
        var dp = self.dataset.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
        var op = out.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
        var inv_h = Scalar[dt](1.0 / self.bandwidth)
        var scale = Scalar[dt](1.0 / (Float64(n) * self.bandwidth * _SQRT_2PI))

        @always_inline
        def lane[
            width: Int, alignment: Int = 1
        ](coord: Coord) {var xp, var dp, var op, var inv_h, var scale, var n}:
            var j = coord_to_index_list(coord)[0]
            var x = rebind[Scalar[dt]](xp[unsafe_offset=j])
            var total = Scalar[dt](0)
            for i in range(n):
                var z = (x - rebind[Scalar[dt]](dp[unsafe_offset=i])) * inv_h
                total += _exp(-z * z / 2)
            op[unsafe_offset=j] = total * scale

        elementwise[simd_width=1, target=_target[gpu]()](lane, Coord(m), ctx)
        ctx.synchronize()
        return out^
