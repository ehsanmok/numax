"""Interpolation of scattered data: `NearestNDInterpolator`, `griddata` and
`RBFInterpolator`, SciPy's.

**Tier 2.**

- `NearestNDInterpolator` is `numax.spatial.KDTree` over the data sites
  and a gather of the values at each query's nearest site, both on the
  data's device.
- `griddata` is SciPy's front door; `method="nearest"` is the
  interpolator above. `"linear"` and `"cubic"` interpolate on a Delaunay
  triangulation of the sites, which numax does not build (the Qhull
  family is out of scope), so they raise and say so.
- `RBFInterpolator` is SciPy's radial basis function interpolant: `f(x) =
  sum_j a_j phi(|x - y_j| eps) + sum_p b_p P_p(x_hat)`, with the
  polynomial tail of degree `degree` in coordinates shifted and scaled to
  `[-1, 1]` and the augmented system `[[K + S, P], [P^T, 0]] [a; b] = [d;
  0]` solved dense. SciPy's eight kernels, its default degrees, the same
  shift and scale. The system is assembled on the data's device, one lane
  per entry, and factored by `numax.linalg`'s run-time-order LU; an
  evaluation is one lane per query point summing over the sites.
  SciPy's `neighbors=` (a local interpolant through a `KDTree`) is not
  taken, so the solve is `O(n^3)` in the number of sites.

## The MAX gate

Nothing: MAX has neither a nearest-neighbor interpolator nor radial basis
functions. **Extend.**
"""

from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.math import exp as _exp, log as _log, sqrt as _sqrt

from ..core.tensorlike import TensorLike
from ..core.tensor import Dynamic, _dyn_shape, _same_order, asarray, copy
from ..linalg.lu import _lu_factor_runtime
from ..spatial.kdtree import KDTree


def _target[gpu: Bool]() -> StaticString:
    comptime if gpu:
        return "gpu"
    else:
        return "cpu"


struct NearestNDInterpolator[dtype: DType, gpu: Bool = False](Movable):
    """The value at the nearest data site, in any number of dimensions.
    `scipy.interpolate.NearestNDInterpolator(x, y)`."""

    var tree: KDTree[Self.dtype, Self.gpu]
    var values: Dynamic[Self.dtype, 1]

    def __init__[
        X: TensorLike, Y: TensorLike
    ](out self, x: X, y: Y) raises where (
        X.dtype == Self.dtype
        and Y.dtype == Self.dtype
        and X.LayoutType.rank == 2
        and Y.LayoutType.rank == 1
        and Self.dtype.is_floating_point()
    ):
        """Index the data sites.

        Parameters:
            X: The tensor type of `x`, `n x k`.
            Y: The tensor type of `y`, `n`.

        Args:
            x: The data sites, one per row.
            y: The values at them.

        Raises:
            If `x` and `y` differ in length, or the tree build fails.
        """
        if x.dim_at(0) != y.size():
            raise Error(
                "NearestNDInterpolator: ",
                x.dim_at(0),
                " sites for ",
                y.size(),
                " values",
            )
        self.tree = KDTree[Self.dtype, Self.gpu](x)
        self.values = rebind_var[Dynamic[Self.dtype, 1]](
            _same_order(y, row_major(_dyn_shape[1](y.size())))
        )

    def __call__[
        Q: TensorLike
    ](self, xi: Q) raises -> Dynamic[Self.dtype, 1] where (
        Q.dtype == Self.dtype
        and Q.LayoutType.rank == 2
        and Self.dtype.is_floating_point()
    ):
        """The interpolated values at the rows of `xi`.

        Parameters:
            Q: The tensor type of `xi`, `m x k`.

        Args:
            xi: The query points, on the data's device.

        Returns:
            The value at each point's nearest site.

        Raises:
            If the query or the gather fails.
        """
        var found = self.tree.query[1](xi)
        var m = xi.dim_at(0)
        var ctx = self.values.context()
        var out = Dynamic[Self.dtype, 1](row_major(_dyn_shape[1](m)), ctx)
        var ip = found.indices.tile()
        var vp = self.values.tile()
        var op = out.tile()

        @always_inline
        def gather[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var ip, var vp, var op}:
            var i = coord_to_index_list(coord)[0]
            op.ptr[unsafe_offset=i] = vp.ptr[
                unsafe_offset=Int(ip.ptr[unsafe_offset=i])
            ]

        elementwise[simd_width=1, target=_target[Self.gpu]()](
            gather, Coord(m), ctx
        )
        ctx.synchronize()
        _ = found^
        return out^


def griddata[
    P: TensorLike, V: TensorLike, X: TensorLike, gpu: Bool = False
](
    points: P, values: V, xi: X, method: StaticString = "linear"
) raises -> Dynamic[P.dtype, 1] where (
    P.dtype.is_floating_point()
    and V.dtype == P.dtype
    and X.dtype == P.dtype
    and P.LayoutType.rank == 2
    and V.LayoutType.rank == 1
    and X.LayoutType.rank == 2
):
    """Interpolate scattered data at the points `xi`.
    `scipy.interpolate.griddata(points, values, xi, method)`.

    `method="nearest"` is `NearestNDInterpolator(points, values)(xi)`.
    `"linear"` and `"cubic"` -- SciPy's default is `"linear"` -- interpolate
    on a Delaunay triangulation, which numax does not build, and raise;
    `RBFInterpolator` is the smooth scattered interpolant numax has.

    Parameters:
        P: The tensor type of `points`, `n x k`.
        V: The tensor type of `values`, `n`.
        X: The tensor type of `xi`, `m x k`.
        gpu: Whether the index queries and the gather run on the device.

    Args:
        points: The data sites.
        values: The values at them.
        xi: The query points.
        method: `"nearest"`; `"linear"` and `"cubic"` raise.

    Returns:
        The interpolated values.

    Raises:
        For `"linear"` or `"cubic"`, an unknown method, or a failure in
        the interpolator.
    """
    if method == "nearest":
        var interpolator = NearestNDInterpolator[P.dtype, gpu](points, values)
        return interpolator(xi)
    if method == "linear" or method == "cubic":
        raise Error(
            "griddata: method '",
            method,
            (
                "' needs a Delaunay triangulation,"
                " which numax does not build; use method='nearest' or"
                " RBFInterpolator"
            ),
        )
    raise Error("griddata: method must be 'nearest', got '", method, "'")


comptime _LINEAR = 0
comptime _THIN_PLATE = 1
comptime _CUBIC = 2
comptime _QUINTIC = 3
comptime _MULTIQUADRIC = 4
comptime _INVERSE_MULTIQUADRIC = 5
comptime _INVERSE_QUADRATIC = 6
comptime _GAUSSIAN = 7


def _kernel_code(kernel: StaticString) raises -> Int:
    if kernel == "linear":
        return _LINEAR
    if kernel == "thin_plate_spline":
        return _THIN_PLATE
    if kernel == "cubic":
        return _CUBIC
    if kernel == "quintic":
        return _QUINTIC
    if kernel == "multiquadric":
        return _MULTIQUADRIC
    if kernel == "inverse_multiquadric":
        return _INVERSE_MULTIQUADRIC
    if kernel == "inverse_quadratic":
        return _INVERSE_QUADRATIC
    if kernel == "gaussian":
        return _GAUSSIAN
    raise Error("RBFInterpolator: unknown kernel '", kernel, "'")


def _min_degree(code: Int) -> Int:
    """SciPy's `_NAME_TO_MIN_DEGREE`: the degree below which the
    interpolant is not guaranteed well posed; `-1` for the positive
    definite kernels."""
    if code == _MULTIQUADRIC or code == _LINEAR:
        return 0
    if code == _THIN_PLATE or code == _CUBIC:
        return 1
    if code == _QUINTIC:
        return 2
    return -1


@always_inline
def _phi[
    dtype: DType
](r: Scalar[dtype], code: Int) -> Scalar[dtype] where dtype.is_floating_point():
    """SciPy's kernel functions of the scaled distance `r`."""
    var one = Scalar[dtype](1)
    if code == _LINEAR:
        return -r
    if code == _THIN_PLATE:
        if r == Scalar[dtype](0):
            return Scalar[dtype](0)
        return r * r * _log(r)
    if code == _CUBIC:
        return r * r * r
    if code == _QUINTIC:
        return -(r * r * r * r * r)
    if code == _MULTIQUADRIC:
        return -_sqrt(r * r + one)
    if code == _INVERSE_MULTIQUADRIC:
        return one / _sqrt(r * r + one)
    if code == _INVERSE_QUADRATIC:
        return one / (r * r + one)
    return _exp(-(r * r))


def _monomials(ndim: Int, degree: Int) -> List[Int]:
    """SciPy's `_monomial_powers`: every exponent vector of total degree at
    most `degree`, flattened row by row, `ndim` entries each."""
    var out = List[Int]()
    if degree < 0:
        return out^
    # Combinations with replacement of `deg` variables, lexicographic.
    for deg in range(degree + 1):
        var combo = List[Int](length=deg, fill=0)
        while True:
            var row = List[Int](length=ndim, fill=0)
            for v in combo:
                row[v] += 1
            for e in row:
                out.append(e)
            if deg == 0:
                break
            var i = deg - 1
            while i >= 0 and combo[i] == ndim - 1:
                i -= 1
            if i < 0:
                break
            combo[i] += 1
            for j in range(i + 1, deg):
                combo[j] = combo[i]
    return out^


struct RBFInterpolator[dtype: DType, gpu: Bool = False](Movable):
    """A radial basis function interpolant of scattered data, SciPy's
    `scipy.interpolate.RBFInterpolator(y, d, kernel, epsilon, smoothing,
    degree)`; the module docstring has the system."""

    var sites: Dynamic[Self.dtype, 2]
    """The data sites, `n x k`."""
    var coefficients: Dynamic[Self.dtype, 1]
    """The `n` kernel weights, then the polynomial tail's."""
    var powers: Dynamic[DType.int64, 1]
    """The tail's exponent vectors, `k` per monomial."""
    var shift: Dynamic[Self.dtype, 1]
    var scale: Dynamic[Self.dtype, 1]
    var code: Int
    var epsilon: Float64
    var monomials: Int

    def __init__[
        Y: TensorLike, D: TensorLike
    ](
        out self,
        y: Y,
        d: D,
        kernel: StaticString = "thin_plate_spline",
        epsilon: Float64 = 1.0,
        smoothing: Float64 = 0.0,
        degree: Int = -2,
    ) raises where (
        Y.dtype == Self.dtype
        and D.dtype == Self.dtype
        and Y.LayoutType.rank == 2
        and D.LayoutType.rank == 1
        and Self.dtype.is_floating_point()
    ):
        """Fit the interpolant.

        Parameters:
            Y: The tensor type of `y`, `n x k`.
            D: The tensor type of `d`, `n`.

        Args:
            y: The data sites, one per row.
            d: The values at them.
            kernel: `"thin_plate_spline"` (SciPy's default), `"linear"`,
                `"cubic"`, `"quintic"`, `"multiquadric"`,
                `"inverse_multiquadric"`, `"inverse_quadratic"` or
                `"gaussian"`.
            epsilon: The shape parameter the distances are scaled by;
                SciPy fixes `1` for the scale-invariant kernels and
                requires it for the others.
            smoothing: Added to the kernel matrix's diagonal; `0`
                interpolates.
            degree: The polynomial tail's degree, `-1` for none; SciPy's
                default, `max(min_degree, 0)`, when below `-1`.

        Raises:
            On an unknown kernel, mismatched lengths, fewer sites than
            monomials, a singular system, or a device failure.
        """
        var code = _kernel_code(kernel)
        var n = y.dim_at(0)
        var k = y.dim_at(1)
        if d.size() != n:
            raise Error(
                "RBFInterpolator: ", n, " sites for ", d.size(), " values"
            )
        var deg = degree if degree >= -1 else max(_min_degree(code), 0)
        var powers_host = _monomials(k, deg)
        var q = len(powers_host) // k if k > 0 else 0
        if n < q:
            raise Error(
                "RBFInterpolator: at least ",
                q,
                " data points are required for a degree-",
                deg,
                " tail in ",
                k,
                " dimensions",
            )
        var ctx = y.context()
        var sites = rebind_var[Dynamic[Self.dtype, 2]](
            _same_order(y, row_major(_dyn_shape[2](n, k)))
        )
        var values = rebind_var[Dynamic[Self.dtype, 1]](
            _same_order(d, row_major(_dyn_shape[1](n)))
        )
        var shift = Dynamic[Self.dtype, 1](row_major(_dyn_shape[1](k)), ctx)
        var scale = Dynamic[Self.dtype, 1](row_major(_dyn_shape[1](k)), ctx)
        var sp = sites.tile()
        var hp = shift.tile()
        var cp = scale.tile()

        # The polynomial domain, shifted and scaled into `[-1, 1]` per
        # dimension, a zero spread replaced by one: SciPy's.
        @always_inline
        def bounds[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var sp, var hp, var cp, var n, var k}:
            var dd = coord_to_index_list(coord)[0]
            var lo = sp.ptr[unsafe_offset=dd]
            var hi = lo
            for i in range(1, n):
                var v = sp.ptr[unsafe_offset=i * k + dd]
                lo = min(lo, v)
                hi = max(hi, v)
            hp.ptr[unsafe_offset=dd] = (hi + lo) / Scalar[Self.dtype](2)
            var s = (hi - lo) / Scalar[Self.dtype](2)
            cp.ptr[unsafe_offset=dd] = (
                Scalar[Self.dtype](1) if s == Scalar[Self.dtype](0) else s
            )

        elementwise[simd_width=1, target=_target[Self.gpu]()](
            bounds, Coord(k), ctx
        )
        ctx.synchronize()
        var ph = List[Scalar[DType.int64]](capacity=max(len(powers_host), 1))
        for e in powers_host:
            ph.append(Int64(e))
        if len(ph) == 0:
            ph.append(0)
        var powers = asarray(ph^, ctx)
        var size = n + q
        var lhs = Dynamic[Self.dtype, 2](
            row_major(_dyn_shape[2](size, size)), ctx
        )
        var rhs = Dynamic[Self.dtype, 1](row_major(_dyn_shape[1](size)), ctx)
        var lp = lhs.tile()
        var rp = rhs.tile()
        var pp = powers.tile()
        var vp = values.tile()
        var eps = Scalar[Self.dtype](epsilon)
        var smooth = Scalar[Self.dtype](smoothing)

        @always_inline
        def assemble[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var sp,
            var hp,
            var cp,
            var lp,
            var rp,
            var pp,
            var vp,
            var eps,
            var smooth,
            var n,
            var k,
            var size,
            var code,
        }:
            var e = coord_to_index_list(coord)[0]
            var i = e // size
            var j = e % size
            var value = Scalar[Self.dtype](0)
            if i < n and j < n:
                var acc = Scalar[Self.dtype](0)
                for c in range(k):
                    var diff = (
                        sp.ptr[unsafe_offset=i * k + c]
                        - sp.ptr[unsafe_offset=j * k + c]
                    ) * eps
                    acc += diff * diff
                value = _phi[Self.dtype](_sqrt(acc), code)
                if i == j:
                    value += smooth
            elif i < n or j < n:
                var site = i if i < n else j
                var mono = j - n if i < n else i - n
                var term = Scalar[Self.dtype](1)
                for c in range(k):
                    var xhat = (
                        sp.ptr[unsafe_offset=site * k + c]
                        - hp.ptr[unsafe_offset=c]
                    ) / cp.ptr[unsafe_offset=c]
                    for _ in range(Int(pp.ptr[unsafe_offset=mono * k + c])):
                        term *= xhat
                value = term
            lp.ptr[unsafe_offset=e] = value
            if j == 0:
                rp.ptr[unsafe_offset=i] = vp.ptr[
                    unsafe_offset=i
                ] if i < n else Scalar[Self.dtype](0)

        elementwise[simd_width=1, target=_target[Self.gpu]()](
            assemble, Coord(size * size), ctx
        )
        ctx.synchronize()
        var factor = _lu_factor_runtime[Self.dtype, Self.gpu](lhs)
        if factor.singular():
            raise Error(
                "RBFInterpolator: the system is singular; the polynomial"
                " matrix may not have full column rank"
            )
        self.coefficients = factor.solve(rhs)
        self.sites = sites^
        self.powers = powers^
        self.shift = shift^
        self.scale = scale^
        self.code = code
        self.epsilon = epsilon
        self.monomials = q

    def __call__[
        Q: TensorLike
    ](self, x: Q) raises -> Dynamic[Self.dtype, 1] where (
        Q.dtype == Self.dtype
        and Q.LayoutType.rank == 2
        and Self.dtype.is_floating_point()
    ):
        """The interpolant at the rows of `x`.

        Parameters:
            Q: The tensor type of `x`, `m x k`.

        Args:
            x: The query points, on the data's device.

        Returns:
            The interpolated values.

        Raises:
            If `x` has the wrong width, or a device operation fails.
        """
        var n = self.sites.dim[0]()
        var k = self.sites.dim[1]()
        if x.dim_at(1) != k:
            raise Error(
                "RBFInterpolator: x has ", x.dim_at(1), " columns for ", k
            )
        var m = x.dim_at(0)
        var ctx = self.sites.context()
        var xs = rebind_var[Dynamic[Self.dtype, 2]](
            _same_order(x, row_major(_dyn_shape[2](m, k)))
        )
        var out = Dynamic[Self.dtype, 1](row_major(_dyn_shape[1](m)), ctx)
        var xp = xs.tile()
        var sp = self.sites.tile()
        var ap = self.coefficients.tile()
        var pp = self.powers.tile()
        var hp = self.shift.tile()
        var cp = self.scale.tile()
        var op = out.tile()
        var eps = Scalar[Self.dtype](self.epsilon)
        var code = self.code
        var q = self.monomials

        @always_inline
        def evaluate[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var xp,
            var sp,
            var ap,
            var pp,
            var hp,
            var cp,
            var op,
            var eps,
            var code,
            var q,
            var n,
            var k,
        }:
            var i = coord_to_index_list(coord)[0]
            var total = Scalar[Self.dtype](0)
            for j in range(n):
                var acc = Scalar[Self.dtype](0)
                for c in range(k):
                    var diff = (
                        xp.ptr[unsafe_offset=i * k + c]
                        - sp.ptr[unsafe_offset=j * k + c]
                    ) * eps
                    acc += diff * diff
                total += ap.ptr[unsafe_offset=j] * _phi[Self.dtype](
                    _sqrt(acc), code
                )
            for p in range(q):
                var term = Scalar[Self.dtype](1)
                for c in range(k):
                    var xhat = (
                        xp.ptr[unsafe_offset=i * k + c]
                        - hp.ptr[unsafe_offset=c]
                    ) / cp.ptr[unsafe_offset=c]
                    for _ in range(Int(pp.ptr[unsafe_offset=p * k + c])):
                        term *= xhat
                total += ap.ptr[unsafe_offset=n + p] * term
            op.ptr[unsafe_offset=i] = total

        elementwise[simd_width=1, target=_target[Self.gpu]()](
            evaluate, Coord(m), ctx
        )
        ctx.synchronize()
        _ = xs^
        return out^
