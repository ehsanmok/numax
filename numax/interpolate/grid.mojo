"""Interpolation on a rectilinear grid over `numax.core.tensor.Tensor`:
`RegularGridInterpolator` in two dimensions and `interpn` in up to
eight, linear or nearest. `scipy.interpolate.RegularGridInterpolator` and
`scipy.interpolate.interpn`.

**Tier 2**, like the rest of `numax.interpolate` over `Tensor`: the grid
and its values stay on the device, and a query is one `elementwise` launch
whose lanes bisect each axis and blend the cell's four corners.

## The MAX gate, at the place the comparison is closest

This is where MAX's image kernels look most like the operation and are
not. `nn.resize_linear`, `nn.resize_nearest_neighbor` and
`nn.resize_bicubic` take an NCHW image and *scale factors* and produce the
whole image resampled onto a new uniform grid -- a fixed output, a uniform
input grid, no query points. `RegularGridInterpolator((x, y), values)(pts)`
takes a *rectilinear* grid (axes need not be uniformly spaced) and returns
the value at each of `m` arbitrary points. The resize kernels cannot
express the query and this cannot express a resize cheaply (it would be
`m = H*W` bisections for a grid it knows is uniform), so the two are
different operations rather than one with two spellings. **Extend**, with
the resize family recorded in `docs/parity.md` as what MAX has instead.

## Two dimensions

SciPy's is `n`-dimensional. This takes a `rows x cols` value matrix and
two axis vectors, which is the case a `Static` tensor spells naturally;
the `n`-D form would want a rank-`n` value tensor and `2^n` corner
weights, and nothing in numax asks for it yet. Say so rather than
generalize speculatively.
"""

from algorithm.rowwise_types import RowCoord
from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise
from std.utils.numerics import nan as _nan

from ..core.rowwise import reduce_all

from ..core.tensorlike import TensorLike, TensorView, dim, is_row_major
from ..core.tensor import Dynamic, Static, _dyn_shape, _same_order
from .interp import _interval


struct RegularGridInterpolator[dtype: DType, rows: Int, cols: Int](Movable):
    """Values on a `rows x cols` rectilinear grid, interpolated at
    arbitrary points. `scipy.interpolate.RegularGridInterpolator((x, y),
    values, method, bounds_error, fill_value)`, in two dimensions.

    ```mojo
    var grid = RegularGridInterpolator(x, y, values)         # linear
    var nearest = RegularGridInterpolator(x, y, values, "nearest")
    var at = grid(points)                                    # points: m x 2
    ```

    `x` (ascending, length `rows`) and `y` (ascending, length `cols`) are
    the axes; `values[i, j]` sits at `(x[i], y[j])`. `method` is
    `"linear"` (bilinear within the cell) or `"nearest"` (the closer grid
    point along each axis, the lower one on a tie, as SciPy picks).

    Out-of-range points follow SciPy's three settings: `bounds_error=True`
    (the default) raises; otherwise `fill_value` -- NaN unless given -- is
    returned, or, with `extrapolate=True` (SciPy's `fill_value=None`), the
    boundary cell's rule is applied past the edge.
    """

    var x: Static[Self.dtype, Self.rows]
    var y: Static[Self.dtype, Self.cols]
    var values: Static[Self.dtype, Self.rows, Self.cols]
    var nearest: Bool
    var bounds_error: Bool
    var fill_value: Scalar[Self.dtype]
    var extrapolate: Bool

    def __init__(
        out self,
        mut x: Static[Self.dtype, Self.rows],
        mut y: Static[Self.dtype, Self.cols],
        mut values: Static[Self.dtype, Self.rows, Self.cols],
        method: StaticString = "linear",
        bounds_error: Bool = True,
        fill_value: Optional[Scalar[Self.dtype]] = None,
        extrapolate: Bool = False,
    ) raises where (
        Self.dtype.is_floating_point() and Self.rows >= 2 and Self.cols >= 2
    ):
        """Keep copies of the axes and values on their device.

        Copies rather than moves, so the caller's grid stays usable; the
        three tensors are read on every query and never written.

        Args:
            x: The ascending axis of length `rows`; its device is kept.
            y: The ascending axis of length `cols`.
            values: The `rows x cols` values, `values[i, j]` at
                `(x[i], y[j])`.
            method: `"linear"` for bilinear blending or `"nearest"` for the
                nearest grid point.
            bounds_error: Whether a query point outside the grid raises.
            fill_value: The value returned outside the grid when
                `bounds_error` is false; NaN when not given.
            extrapolate: Whether to apply the boundary cell's rule past the
                edge instead of returning `fill_value`.

        Raises:
            If `method` is neither `"linear"` nor `"nearest"`, or a copy of
            the axes or values fails.
        """
        if not (method == "linear" or method == "nearest"):
            raise Error(
                "RegularGridInterpolator: unknown method '",
                method,
                "'; expected 'linear' or 'nearest'",
            )
        var ctx = x.context()
        self.x = Static[Self.dtype, Self.rows](x.to_host(), ctx)
        self.y = Static[Self.dtype, Self.cols](y.to_host(), ctx)
        self.values = Static[Self.dtype, Self.rows, Self.cols](
            values.to_host(), ctx
        )
        self.nearest = method == "nearest"
        self.bounds_error = bounds_error
        self.fill_value = fill_value.value() if fill_value else _nan[
            Self.dtype
        ]()
        self.extrapolate = extrapolate

    def __call__[
        T: TensorLike,
        gpu: Bool = False,
    ](mut self, points: T) raises -> Static[Self.dtype, dim[T, 0]] where (
        (
            Self.dtype.is_floating_point()
            and Self.rows >= 2
            and Self.cols >= 2
            and dim[T, 0] > 0
        )
        and T.dtype == Self.dtype
        and T.LayoutType.rank == 2
        and T.LayoutType.all_dims_known
        and dim[T, 1] == 2
    ):
        """The grid's value at each of the `m` points, row `q` of `points`
        being `(x_q, y_q)`.

        One launch: each lane bisects both axes, then either blends the
        cell's four corners bilinearly or takes the nearest one. With
        `bounds_error`, the points are first checked on the host and a
        point outside the grid raises, as SciPy's does; the check is a
        pass over `points`, so a caller who knows the points are inside
        can pass `bounds_error=False` and skip it.

        Parameters:
            T: The tensor type of `points`, rank 2 of shape `m x 2`.
            gpu: Whether the query launch targets the GPU on the device of
                `points` rather than the host CPU.

        Args:
            points: The `m x 2` query points, one `(x, y)` pair per row.

        Returns:
            A length-`m` tensor of the interpolated values.

        Raises:
            If `bounds_error` is set and a point lies outside the grid, or
            a copy or the launch fails.
        """
        comptime m = dim[T, 0]
        var ctx = points.context()
        if self.bounds_error:
            var host_points = points.to_host[Self.dtype]()
            var xs = self.x.to_host()
            var ys = self.y.to_host()
            for q in range(m):
                var px = host_points[2 * q]
                var py = host_points[2 * q + 1]
                if (
                    px < xs[0]
                    or px > xs[Self.rows - 1]
                    or py < ys[0]
                    or py > ys[Self.cols - 1]
                ):
                    raise Error(
                        "RegularGridInterpolator: point ",
                        q,
                        " is out of bounds; pass bounds_error=False for a",
                        " fill value or extrapolate=True",
                    )

        var out = Static[Self.dtype, m]._uninitialized(ctx)
        var xg = self.x.tile()
        var yg = self.y.tile()
        var vals = self.values.tile()
        var ps = points.tile_as[Self.dtype]()
        var os = out.tile()
        var nearest = self.nearest
        var extrapolate = self.extrapolate
        var fill = self.fill_value

        @always_inline
        def evaluate[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var xg,
            var yg,
            var vals,
            var ps,
            var os,
            var nearest,
            var extrapolate,
            var fill,
        }:
            var q = coord_to_index_list(coord)[0]
            var px = ps[Coord(q, 0)]
            var py = ps[Coord(q, 1)]
            var result: Scalar[Self.dtype]
            var outside = (
                px < xg[Coord(0)]
                or px > xg[Coord(Self.rows - 1)]
                or py < yg[Coord(0)]
                or py > yg[Coord(Self.cols - 1)]
            )
            if outside and not extrapolate:
                result = fill
            else:
                var i = _interval(xg, Self.rows, px)
                var j = _interval(yg, Self.cols, py)
                var x0 = xg[Coord(i)]
                var x1 = xg[Coord(i + 1)]
                var y0 = yg[Coord(j)]
                var y1 = yg[Coord(j + 1)]
                if nearest:
                    var ii = i if px - x0 <= x1 - px else i + 1
                    var jj = j if py - y0 <= y1 - py else j + 1
                    result = vals[Coord(ii, jj)]
                else:
                    var tx = (px - x0) / (x1 - x0)
                    var ty = (py - y0) / (y1 - y0)
                    var bottom = vals[Coord(i, j)] + tx * (
                        vals[Coord(i + 1, j)] - vals[Coord(i, j)]
                    )
                    var top = vals[Coord(i, j + 1)] + tx * (
                        vals[Coord(i + 1, j + 1)] - vals[Coord(i, j + 1)]
                    )
                    result = bottom + ty * (top - bottom)
            os.store[1](Coord(q), result)

        elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
            evaluate, Coord(m), ctx
        )
        ctx.synchronize()
        return out^


def _grid_target[gpu: Bool]() -> StaticString:
    comptime if gpu:
        return "gpu"
    else:
        return "cpu"


def interpn[
    P: TensorLike,
    V: TensorLike,
    X: TensorLike,
    gpu: Bool = False,
](
    points: P,
    values: V,
    xi: X,
    method: StaticString = "linear",
    bounds_error: Bool = True,
    fill_value: Optional[Scalar[V.dtype]] = None,
    extrapolate: Bool = False,
) raises -> Dynamic[V.dtype, 1] where (
    V.dtype.is_floating_point()
    and P.dtype == V.dtype
    and X.dtype == V.dtype
    and P.LayoutType.rank == 1
    and X.LayoutType.rank == 2
    and V.LayoutType.rank >= 1
    and V.LayoutType.rank <= 8
):
    """Interpolation on a regular grid in any number of dimensions.
    `scipy.interpolate.interpn(points, values, xi, method, bounds_error,
    fill_value)`.

    `values` has rank `d` and shape `(n_0, ..., n_{d-1})`; the grid's axes
    come concatenated in `points`, axis `a`'s `n_a` ascending coordinates
    after the ones before it -- SciPy's tuple of arrays laid end to end,
    their lengths read off `values`' shape. Each row of `xi` is one query
    point. One lane per point bisects each axis and either blends the
    `2^d` corners of its cell (`"linear"`) or takes the nearest grid point
    along each axis, the lower one on a tie (`"nearest"`), as SciPy does.

    Out of range, `bounds_error` raises -- detected on the device, one
    flag per point and one maximum read back; otherwise `fill_value`
    (NaN unless given) is returned, or with `extrapolate` (SciPy's
    `fill_value=None`) the edge cell's rule continues past the edge.

    Parameters:
        P: The tensor type of `points`, rank 1.
        V: The tensor type of `values`, rank `d` from 1 to 8.
        X: The tensor type of `xi`, `m x d`.
        gpu: Whether the queries run on the inputs' device.

    Args:
        points: The axes, concatenated, `n_0 + ... + n_{d-1}` long.
        values: The samples on the grid.
        xi: The query points, one per row.
        method: `"linear"` or `"nearest"`.
        bounds_error: Whether a point outside the grid raises.
        fill_value: The value outside the grid when not raising.
        extrapolate: Continue the edge cells past the grid instead.

    Returns:
        The `m` interpolated values.

    Raises:
        On an unknown method, mismatched lengths, an axis with fewer than
        two points for `"linear"`, a point out of range under
        `bounds_error`, or a device failure.
    """
    comptime dtype = V.dtype
    comptime d = V.LayoutType.rank
    var nearest: Bool
    if method == "linear":
        nearest = False
    elif method == "nearest":
        nearest = True
    else:
        raise Error(
            "interpn: method must be 'linear' or 'nearest', got '", method, "'"
        )
    # Captured by value into the kernel, so register-passable vectors
    # rather than arrays.
    var dims = SIMD[DType.int64, 8](1)
    var starts = SIMD[DType.int64, 8](0)
    var strides = SIMD[DType.int64, 8](1)
    var total = 0
    for a in range(d):
        dims[a] = Int64(values.dim_at(a))
        starts[a] = Int64(total)
        total += Int(dims[a])
        if dims[a] < 2 and not nearest:
            raise Error("interpn: axis ", a, " has ", Int(dims[a]), " points")
    for a in range(d - 2, -1, -1):
        strides[a] = strides[a + 1] * dims[a + 1]
    if points.size() != total:
        raise Error(
            "interpn: points has ",
            points.size(),
            " coordinates for axes of ",
            total,
        )
    if xi.dim_at(1) != d:
        raise Error(
            "interpn: xi has ", xi.dim_at(1), " columns for ", d, " axes"
        )
    var m = xi.dim_at(0)
    var ctx = values.context()
    var ax = rebind_var[Dynamic[dtype, 1]](
        _same_order(points, row_major(_dyn_shape[1](total)))
    )
    var vals = _same_order(values, row_major(_dyn_shape[1](values.size())))
    var qs = rebind_var[Dynamic[dtype, 2]](
        _same_order(xi, row_major(_dyn_shape[2](m, d)))
    )
    var out = Dynamic[dtype, 1](row_major(_dyn_shape[1](m)), ctx)
    var flags = Dynamic[dtype, 1](row_major(_dyn_shape[1](max(m, 1))), ctx)
    var ap = ax.tile()
    var vp = vals.tile()
    var qp = qs.tile()
    var op = out.tile()
    var fp = flags.tile()
    var fill = fill_value.value() if fill_value else _nan[dtype]()

    @always_inline
    def evaluate[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var ap,
        var vp,
        var qp,
        var op,
        var fp,
        var dims,
        var starts,
        var strides,
        var nearest,
        var fill,
        var extrapolate,
    }:
        var i = coord_to_index_list(coord)[0]
        var lower = SIMD[DType.int64, 8](0)
        var frac = SIMD[dtype, 8](0)
        var outside = False
        comptime for a in range(d):
            var s = Int(starts[a])
            var count = Int(dims[a])
            var x = qp.ptr[unsafe_offset=i * d + a]
            var first = ap.ptr[unsafe_offset=s]
            var last = ap.ptr[unsafe_offset=s + count - 1]
            if x < first or x > last:
                outside = True
            if count == 1:
                lower[a] = 0
                frac[a] = Scalar[dtype](0)
            else:
                # The last cell `j` with `axis[j] <= x`, clamped to the cells.
                var lo = 0
                var hi = count - 2
                while lo < hi:
                    var mid = (lo + hi + 1) // 2
                    if ap.ptr[unsafe_offset=s + mid] <= x:
                        lo = mid
                    else:
                        hi = mid - 1
                var x0 = ap.ptr[unsafe_offset=s + lo]
                var x1 = ap.ptr[unsafe_offset=s + lo + 1]
                lower[a] = Int64(lo)
                frac[a] = (x - x0) / (x1 - x0)
        fp.ptr[unsafe_offset=i] = Scalar[dtype](1) if outside else Scalar[
            dtype
        ](0)
        var result = Scalar[dtype](0)
        if nearest:
            var offset = 0
            comptime for a in range(d):
                var j = Int(lower[a])
                if dims[a] > 1 and frac[a] > Scalar[dtype](0.5):
                    j += 1
                offset += j * Int(strides[a])
            result = vp.ptr[unsafe_offset=offset]
        else:
            comptime corners = 1 << d
            comptime for corner in range(corners):
                var weight = Scalar[dtype](1)
                var offset = 0
                comptime for a in range(d):
                    comptime up = (corner >> a) & 1
                    comptime if up == 1:
                        weight *= frac[a]
                        offset += (Int(lower[a]) + 1) * Int(strides[a])
                    else:
                        weight *= Scalar[dtype](1) - frac[a]
                        offset += Int(lower[a]) * Int(strides[a])
                result += weight * vp.ptr[unsafe_offset=offset]
        if outside and not extrapolate:
            result = fill
        op.ptr[unsafe_offset=i] = result

    if m > 0:
        elementwise[simd_width=1, target=_grid_target[gpu]()](
            evaluate, Coord(m), ctx
        )
        ctx.synchronize()
        if bounds_error:
            var worst = Scalar[dtype](0)
            comptime if gpu:
                var peak = Dynamic[dtype, 1](row_major(_dyn_shape[1](1)), ctx)

                @always_inline
                def identity[
                    ww: Int
                ](tile: SIMD[dtype, ww], idx: RowCoord[1]) {} -> SIMD[
                    dtype, ww
                ]:
                    return tile

                reduce_all[monoid="max", gpu=True](
                    flags.tile(), peak.tile(), identity, m, Optional(ctx)
                )
                worst = peak.to_host()[0]
            else:
                var host = flags.to_host()
                for j in range(m):
                    worst = max(worst, host[j])
            if worst > 0:
                raise Error(
                    "interpn: a point is out of bounds; pass bounds_error=False"
                    " for a fill value or extrapolate=True"
                )
    _ = ax^
    _ = vals^
    _ = qs^
    _ = flags^
    return out^
