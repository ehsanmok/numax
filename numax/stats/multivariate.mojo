"""The multivariate normal distribution over `numax.core.tensor.Tensor`:
`multivariate_normal.pdf`, `logpdf`, `rvs` and `entropy`,
`scipy.stats.multivariate_normal`'s.

**Tier 2**, through `numax.linalg`: the density of each row of `x` is
`-(d ln 2pi + ln det C + |L^-1 (x - mean)|^2) / 2` with `L` the Cholesky
factor of the covariance, so one factorization, one triangular solve
against all the centered rows at once, and one launch folding each
row's squares -- device-resident at `gpu=True`, the covariance's
diagonal read back to form the log-determinant (it is `d` numbers, the
parameter's size).

## The MAX gate

Nothing to delegate for the density itself; the factorization and the
solve are `numax.linalg`'s, which are MAX's `matmul` underneath.
"""

from std.math import log as _std_log

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core.tensorlike import TensorLike, dim, is_row_major
from ..core.tensor import Static
from ..core.elementwise import exp
from ..linalg.cholesky import cholesky
from ..linalg.triangular import solve_triangular
from .random import Generator
from .statistics import _target

comptime _LN_2PI = 1.8378770664093453


def _log_det_from_cholesky[
    L: TensorLike
](lower: L) raises -> Float64 where L.LayoutType.rank == 2:
    """`ln det C = 2 sum ln L_ii`, from the factor's diagonal."""
    var d = lower.dim_at(0)
    var values = lower.to_host()
    var total = 0.0
    for i in range(d):
        var diag = Float64(values[i * d + i])
        if diag <= 0:
            raise Error(
                "multivariate_normal: the covariance is not positive definite"
            )
        total += 2.0 * _std_log(diag)
    return total


struct multivariate_normal:
    """The multivariate normal distribution. `scipy.stats.multivariate_normal`.
    """

    @staticmethod
    def logpdf[
        X: TensorLike, M: TensorLike, C: TensorLike, gpu: Bool = False
    ](x: X, mean: M, cov: C) raises -> Static[X.dtype, dim[X, 0]] where (
        X.dtype.is_floating_point()
        and M.dtype == X.dtype
        and C.dtype.is_floating_point()
        and C.dtype == X.dtype
        and X.LayoutType.rank == 2
        and X.LayoutType.all_dims_known
        and M.LayoutType.rank == 1
        and M.LayoutType.all_dims_known
        and C.LayoutType.rank == 2
        and C.LayoutType.all_dims_known
        and dim[C, 1] == dim[C, 0]
        and dim[M, 0] == dim[X, 1]
        and dim[C, 0] == dim[X, 1]
    ):
        """The log density of each row of `x`.
        `scipy.stats.multivariate_normal.logpdf(x, mean, cov)`.

        Parameters:
            X: The tensor type of `x`, `(m, d)`, one point per row.
            M: The tensor type of `mean`, length `d`.
            C: The tensor type of `cov`, `(d, d)`.
            gpu: Whether to factor, solve and fold on `x`'s device.

        Args:
            x: The points.
            mean: The mean vector.
            cov: The covariance, symmetric positive definite.

        Returns:
            The `m` log densities.

        Raises:
            If `cov` is not positive definite, or a device operation fails.
        """
        comptime dtype = X.dtype
        comptime m = dim[X, 0]
        comptime d = dim[X, 1]
        var ctx = x.context()
        var lower = cholesky[gpu=gpu](cov)
        var log_det = _log_det_from_cholesky(lower)
        # The centered points, transposed: column `j` is `x[j] - mean`.
        var centered = Static[dtype, d, m]._uninitialized(ctx)
        var xp = x.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
        var mp = mean.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
        var cp = centered.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

        @always_inline
        def center[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var xp, var mp, var cp}:
            var f = coord_to_index_list(coord)[0]
            var i = f // m
            var j = f % m
            cp[unsafe_offset=f] = rebind[Scalar[dtype]](
                xp[unsafe_offset=j * d + i]
            ) - rebind[Scalar[dtype]](mp[unsafe_offset=i])

        elementwise[simd_width=1, target=_target[gpu]()](
            center, Coord(d * m), ctx
        )
        # The solve reads `centered` through its own launches; the fill
        # must have landed first.
        ctx.synchronize()
        var z = solve_triangular[gpu=gpu](lower, centered)
        ctx.synchronize()
        var out = Static[dtype, m]._uninitialized(ctx)
        var zp = z.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
        var op = out.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
        var offset = Scalar[dtype](Float64(d) * _LN_2PI + log_det)

        @always_inline
        def fold[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var zp, var op, var offset}:
            var j = coord_to_index_list(coord)[0]
            var total = Scalar[dtype](0)
            for i in range(d):
                var v = rebind[Scalar[dtype]](zp[unsafe_offset=i * m + j])
                total += v * v
            op[unsafe_offset=j] = -(offset + total) / 2

        elementwise[simd_width=1, target=_target[gpu]()](fold, Coord(m), ctx)
        ctx.synchronize()
        # The kernels read these through origin-erased pointers; keep them
        # alive until the launches that read them have finished.
        _ = z^
        _ = centered^
        _ = lower^
        return out^

    @staticmethod
    def pdf[
        X: TensorLike, M: TensorLike, C: TensorLike, gpu: Bool = False
    ](x: X, mean: M, cov: C) raises -> Static[X.dtype, dim[X, 0]] where (
        X.dtype.is_floating_point()
        and M.dtype == X.dtype
        and C.dtype.is_floating_point()
        and C.dtype == X.dtype
        and X.LayoutType.rank == 2
        and X.LayoutType.all_dims_known
        and M.LayoutType.rank == 1
        and M.LayoutType.all_dims_known
        and C.LayoutType.rank == 2
        and C.LayoutType.all_dims_known
        and dim[C, 1] == dim[C, 0]
        and dim[M, 0] == dim[X, 1]
        and dim[C, 0] == dim[X, 1]
    ):
        """The density of each row of `x`, `exp(logpdf)`.
        `scipy.stats.multivariate_normal.pdf(x, mean, cov)`.

        Parameters:
            X: The tensor type of `x`, `(m, d)`, one point per row.
            M: The tensor type of `mean`, length `d`.
            C: The tensor type of `cov`, `(d, d)`.
            gpu: Whether to compute on `x`'s device.

        Args:
            x: The points.
            mean: The mean vector.
            cov: The covariance, symmetric positive definite.

        Returns:
            The `m` densities.

        Raises:
            If `cov` is not positive definite, or a device operation fails.
        """
        return exp[gpu=gpu](multivariate_normal.logpdf[gpu=gpu](x, mean, cov))

    @staticmethod
    def rvs[
        M: TensorLike, C: TensorLike, count: Int, gpu: Bool = False
    ](mean: M, cov: C, mut rng: Generator) raises -> Static[
        M.dtype, count, dim[M, 0]
    ] where (
        M.dtype.is_floating_point()
        and C.dtype.is_floating_point()
        and C.dtype == M.dtype
        and M.LayoutType.rank == 1
        and M.LayoutType.all_dims_known
        and C.LayoutType.rank == 2
        and C.LayoutType.all_dims_known
        and dim[C, 0] == dim[M, 0]
        and dim[C, 1] == dim[C, 0]
    ):
        """`count` draws, one per row. `scipy.stats.multivariate_normal.rvs`,
        through `Generator.multivariate_normal`.

        Parameters:
            M: The tensor type of `mean`, length `d`.
            C: The tensor type of `cov`, `(d, d)`.
            count: How many draws.
            gpu: Whether to draw and compute on `mean`'s device.

        Args:
            mean: The mean vector.
            cov: The covariance, symmetric positive definite.
            rng: The generator; its seed advances.

        Returns:
            A `(count, d)` tensor of draws.

        Raises:
            If `cov` is not positive definite, or a device operation fails.
        """
        return rng.multivariate_normal[count=count, gpu=gpu](mean, cov)

    @staticmethod
    def entropy[
        C: TensorLike
    ](cov: C) raises -> Float64 where (
        C.dtype.is_floating_point()
        and C.LayoutType.rank == 2
        and C.LayoutType.all_dims_known
        and dim[C, 1] == dim[C, 0]
    ):
        """The differential entropy `(d (1 + ln 2pi) + ln det C) / 2`, in
        nats. `scipy.stats.multivariate_normal.entropy`.

        Parameters:
            C: The tensor type of `cov`, `(d, d)`.

        Args:
            cov: The covariance, symmetric positive definite.

        Returns:
            The entropy.

        Raises:
            If `cov` is not positive definite.
        """
        comptime d = dim[C, 0]
        var log_det = _log_det_from_cholesky(cholesky(cov))
        return (Float64(d) * (1.0 + _LN_2PI) + log_det) / 2.0
