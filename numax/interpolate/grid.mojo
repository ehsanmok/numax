"""Interpolation on a rectilinear grid over `numax.core.array.Tensor`:
`RegularGridInterpolator` in two dimensions, linear or nearest.
`scipy.interpolate.RegularGridInterpolator`.

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

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from std.utils.numerics import nan as _nan

from ..core.array import Static
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
        """
        if not (method == "linear" or method == "nearest"):
            raise Error(
                "RegularGridInterpolator: unknown method '",
                method,
                "'; expected 'linear' or 'nearest'",
            )
        var ctx = x.context()
        self.x = Static[Self.dtype, Self.rows](ctx, x.to_host())
        self.y = Static[Self.dtype, Self.cols](ctx, y.to_host())
        self.values = Static[Self.dtype, Self.rows, Self.cols](
            ctx, values.to_host()
        )
        self.nearest = method == "nearest"
        self.bounds_error = bounds_error
        self.fill_value = fill_value.value() if fill_value else _nan[
            Self.dtype
        ]()
        self.extrapolate = extrapolate

    def __call__[
        m: Int, gpu: Bool = False
    ](mut self, mut points: Static[Self.dtype, m, 2]) raises -> Static[
        Self.dtype, m
    ] where (
        Self.dtype.is_floating_point()
        and Self.rows >= 2
        and Self.cols >= 2
        and m > 0
    ):
        """The grid's value at each of the `m` points, row `q` of `points`
        being `(x_q, y_q)`.

        One launch: each lane bisects both axes, then either blends the
        cell's four corners bilinearly or takes the nearest one. With
        `bounds_error`, the points are first checked on the host and a
        point outside the grid raises, as SciPy's does; the check is a
        pass over `points`, so a caller who knows the points are inside
        can pass `bounds_error=False` and skip it.
        """
        var ctx = points.context()
        if self.bounds_error:
            var host_points = points.to_host()
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
        var xg = self.x.view()
        var yg = self.y.view()
        var vals = self.values.view()
        var ps = points.view()
        var os = out.view()
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
