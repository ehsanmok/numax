"""Cubic splines over `numax.core.tensor.Tensor`: `CubicHermiteSpline`,
`CubicSpline` with SciPy's boundary conditions, `PchipInterpolator` and
`Akima1DInterpolator`, on knots that need not be uniform.

**This module is tier 2**, like the rest of `numax.interpolate` over
`Tensor`: construction runs on the host, evaluation is one `elementwise`
launch whose lanes bisect the knots. `numax.interpolate`'s `Array` tier has the
`FloatLike` spline that differentiates at `Dual` and runs per lane inside
a kernel, at the price of scanning every interval; the two are
cross-referenced rather than ranked.

## One representation, four constructors

Every spline here is a piecewise cubic in SciPy's `PPoly` form: on the
interval `[x_i, x_{i+1}]`, `c0 + c1 s + c2 s^2 + c3 s^3` with `s = x - x_i`,
the coefficients a `4 x (n-1)` tensor on the device. `CubicHermiteSpline`
owns that form and builds it from knot values and knot *slopes* `d_i`.
The other three differ only in where the slopes come from -- a
tridiagonal solve for `CubicSpline`, local formulas for PCHIP and Akima --
and each hands `(x, y, d)` to `CubicHermiteSpline`. So there is one
evaluation kernel, one `integrate`, and one place the derivative order is
handled; `scipy.interpolate` is arranged the same way, with `PPoly` under
all of them.

## Construction follows the knots

Building a spline is `O(n)` arithmetic on the knots plus, for
`CubicSpline`, one tridiagonal solve -- the slope-form system SciPy's
`_cubic.py` assembles, boundary rows and all, so every `bc_type` here is
SciPy's to the digit. (`solveh_banded` is not used, though the natural
spline's *moment* system is symmetric positive definite: SciPy states its
boundary conditions in slopes, and one formulation that matches SciPy's
rows beats two that would have to be reconciled.)

Knots on the host build on the host, in `Float64`, the solve through
`solve_banded`'s pivoted path. Knots on a device build there, with
nothing downloaded: one launch for the interval widths and secants, one
per slope formula, and for `CubicSpline` one launch assembling the band
and a cyclic-reduction solve (`numax.linalg.banded._tridiagonal_device`).
That solve does not pivot, so the not-a-knot boundary rows -- the one
place SciPy's system is not diagonally dominant -- are folded into their
neighbors first: row 1 minus row 0 no longer names `d_0`, which leaves
rows `1 .. n-2` a dominant system on their own, and `d_0` and `d_{n-1}`
are read back from the boundary rows after it. Akima's "the weights
vanish" threshold is relative to the largest weight, which is one
`reduce_all` on the device. The two- and three-knot not-a-knot splines,
SciPy's special cases, stay on the host: they are three numbers. Like the
operators, the device path sits under `comptime if has_accelerator()` and
is not taken at `float64`, which Metal cannot compile; there the
construction downloads, as before.

## Evaluation

One launch over the query points: bisect to the interval, then Horner on
the local cubic. The derivative order `nu` is a compile-time parameter,
so `spline[nu=1](x)` is the first derivative and `nu > 3` is zero, as
SciPy's `PPoly.__call__(x, nu)` has it.

Outside the knots, `CubicSpline` and `PchipInterpolator` **extrapolate**
with the end intervals' cubics -- SciPy's default, `extrapolate=True` --
and `Akima1DInterpolator` returns NaN, which is SciPy's default for that
one. Each constructor takes `extrapolate` to choose the other behaviour.
The `Array` tier clamps instead, for the reason its docstring gives.

## The MAX gate

Recorded in `interp.mojo` and `docs/parity.md`: MAX has no interpolation
at arbitrary points, only fixed-grid image resizes. An **extend**.
"""

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from algorithm.rowwise_types import RowCoord
from max.gpu.host import DeviceContext
from std.sys import has_accelerator
from std.utils.numerics import nan as _nan

from ..core.rowwise import reduce_all
from ..core.tensorlike import TensorLike, TensorView, dim, is_row_major
from ..core.tensor import Static, copy
from ..linalg.banded import _tridiagonal_device, solve_banded
from .interp import _interval


struct CubicHermiteSpline[dtype: DType, n: Int](Movable):
    """A piecewise cubic through `n` knots with a prescribed slope at
    each: `scipy.interpolate.CubicHermiteSpline(x, y, dydx)`, and the
    `PPoly` form every other spline in this module is built on.

    `x` holds the knots, `c` the `4 x (n-1)` local coefficients -- row `k`
    is the `s^k` coefficient of every interval -- and `extrapolate` says
    whether a query outside the knots takes the end interval's cubic or
    NaN. All three live where the knots did; a spline built on a device
    evaluates there.
    """

    var x: Static[Self.dtype, Self.n]
    var c: Static[Self.dtype, 4, Self.n - 1]
    var extrapolate: Bool

    def __init__(
        out self,
        var x: Static[Self.dtype, Self.n],
        var c: Static[Self.dtype, 4, Self.n - 1],
        extrapolate: Bool,
    ):
        """The `PPoly` form directly: knots and local coefficients already
        on the device. What the constructors below hand in."""
        self.x = x^
        self.c = c^
        self.extrapolate = extrapolate

    def __init__(
        out self,
        mut x: Static[Self.dtype, Self.n],
        mut y: Static[Self.dtype, Self.n],
        mut dydx: Static[Self.dtype, Self.n],
        extrapolate: Bool = True,
    ) raises where Self.dtype.is_floating_point() and Self.n >= 2:
        """The spline through `(x, y)` with slope `dydx` at every knot.
        `x` must be strictly ascending, which is not checked. Built where
        the knots live; see the module docstring."""
        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not x.on_host():
                self = Self._from_device(copy(x), y, dydx, extrapolate)
                return
        self = Self._from_host(
            x.context(),
            _as_float64(x.to_host()),
            _as_float64(y.to_host()),
            _as_float64(dydx.to_host()),
            extrapolate,
        )

    @staticmethod
    def _from_host(
        ctx: DeviceContext,
        xs: List[Float64],
        ys: List[Float64],
        ds: List[Float64],
        extrapolate: Bool,
    ) raises -> Self where Self.dtype.is_floating_point() and Self.n >= 2:
        """Slopes to local cubic coefficients, on the host in `Float64`,
        then one upload each for the knots and the coefficients.

        On `[x_i, x_{i+1}]` with `h = x_{i+1} - x_i` and secant `D`:
        `c0 = y_i`, `c1 = d_i`, `c2 = (3D - 2d_i - d_{i+1}) / h`,
        `c3 = (d_i + d_{i+1} - 2D) / h^2` -- the cubic with the given
        values and slopes at both ends.
        """
        comptime pieces = Self.n - 1
        var knots = List[Scalar[Self.dtype]](capacity=Self.n)
        for i in range(Self.n):
            knots.append(Scalar[Self.dtype](xs[i]))
        var coefficients = List[Scalar[Self.dtype]](
            length=4 * pieces, fill=Scalar[Self.dtype](0)
        )
        for i in range(pieces):
            var h = xs[i + 1] - xs[i]
            var secant = (ys[i + 1] - ys[i]) / h
            coefficients[0 * pieces + i] = Scalar[Self.dtype](ys[i])
            coefficients[1 * pieces + i] = Scalar[Self.dtype](ds[i])
            coefficients[2 * pieces + i] = Scalar[Self.dtype](
                (3.0 * secant - 2.0 * ds[i] - ds[i + 1]) / h
            )
            coefficients[3 * pieces + i] = Scalar[Self.dtype](
                (ds[i] + ds[i + 1] - 2.0 * secant) / (h * h)
            )
        return Self(
            Static[Self.dtype, Self.n](knots^, ctx),
            Static[Self.dtype, 4, pieces](coefficients^, ctx),
            extrapolate,
        )

    @staticmethod
    def _from_device(
        var x: Static[Self.dtype, Self.n],
        mut y: Static[Self.dtype, Self.n],
        mut d: Static[Self.dtype, Self.n],
        extrapolate: Bool,
    ) raises -> Self where Self.dtype.is_floating_point() and Self.n >= 2:
        """`_from_host`'s coefficients, computed on the knots' device: one
        lane per interval, the same four formulas, at `dtype`. Takes the
        knots it keeps."""
        comptime pieces = Self.n - 1
        var ctx = x.context()
        var c = Static[Self.dtype, 4, pieces]._uninitialized(ctx)
        var xv = x.tile()
        var yv = y.tile()
        var dv = d.tile()
        var cv = c.tile()

        @always_inline
        def coefficients[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var xv, var yv, var dv, var cv}:
            var i = coord_to_index_list(coord)[0]
            var h = xv[Coord(i + 1)][0] - xv[Coord(i)][0]
            var secant = (yv[Coord(i + 1)][0] - yv[Coord(i)][0]) / h
            var d0 = dv[Coord(i)][0]
            var d1 = dv[Coord(i + 1)][0]
            cv.store[1](Coord(0, i), yv[Coord(i)][0])
            cv.store[1](Coord(1, i), d0)
            cv.store[1](Coord(2, i), (3 * secant - 2 * d0 - d1) / h)
            cv.store[1](Coord(3, i), (d0 + d1 - 2 * secant) / (h * h))

        elementwise[simd_width=1, target="gpu"](
            coefficients, Coord(pieces), ctx
        )
        ctx.synchronize()
        return Self(x^, c^, extrapolate)

    def __call__[
        T: TensorLike,
        nu: Int = 0,
        gpu: Bool = False,
    ](mut self, points: T) raises -> Static[Self.dtype, dim[T, 0]] where (
        (
            Self.dtype.is_floating_point()
            and Self.n >= 2
            and dim[T, 0] > 0
            and nu >= 0
        )
        and T.dtype == Self.dtype
        and T.LayoutType.rank == 1
        and T.LayoutType.all_dims_known
    ):
        """The spline, or its `nu`-th derivative, at every point.
        `PPoly.__call__(x, nu)`.

        One launch: each lane bisects the knots to its interval (clamped
        to the end intervals when extrapolating) and evaluates the local
        cubic by Horner. `nu` past 3 is identically zero.
        """
        comptime m = dim[T, 0]
        var ctx = points.context()
        var out = Static[Self.dtype, m]._uninitialized(ctx)
        var knots = self.x.tile()
        var coef = self.c.tile()
        var ps = points.tile_as[Self.dtype]()
        var ys = out.tile()
        var extrapolate = self.extrapolate

        @always_inline
        def evaluate[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var knots, var coef, var ps, var ys, var extrapolate}:
            var q = coord_to_index_list(coord)[0]
            var at = ps[Coord(q)]
            var value: Scalar[Self.dtype]
            if not extrapolate and (
                at < knots[Coord(0)] or at > knots[Coord(Self.n - 1)]
            ):
                value = _nan[Self.dtype]()
            else:
                var i = _interval(knots, Self.n, at)
                var s = at - knots[Coord(i)]
                var c1 = coef[Coord(1, i)]
                var c2 = coef[Coord(2, i)]
                var c3 = coef[Coord(3, i)]
                comptime if nu == 0:
                    var c0 = coef[Coord(0, i)]
                    value = c0 + s * (c1 + s * (c2 + s * c3))
                elif nu == 1:
                    value = c1 + s * (2 * c2 + s * (3 * c3))
                elif nu == 2:
                    value = 2 * c2 + s * (6 * c3)
                elif nu == 3:
                    value = 6 * c3
                else:
                    value = Scalar[Self.dtype](0)
            ys.store[1](Coord(q), value)

        elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
            evaluate, Coord(m), ctx
        )
        ctx.synchronize()
        return out^

    def integrate(
        mut self, a: Scalar[Self.dtype], b: Scalar[Self.dtype]
    ) raises -> Scalar[Self.dtype] where Self.n >= 2:
        """The definite integral of the spline over `[a, b]`.
        `PPoly.integrate(a, b)`.

        Exact for the piecewise cubic: each interval's antiderivative is
        evaluated between the interval's overlap with `[a, b]`. With
        `extrapolate`, the end cubics extend past the knots; without it, a
        range that leaves the knots integrates to NaN, as SciPy's does.
        Host-side, `O(n)`, one download of the coefficients -- an integral
        is a scalar and the sum is sequential.
        """
        comptime pieces = Self.n - 1
        var lo = Float64(a)
        var hi = Float64(b)
        var sign = 1.0
        if lo > hi:
            var swap = lo
            lo = hi
            hi = swap
            sign = -1.0
        var knots = self.x.to_host()
        var coef = self.c.to_host()
        var first = Float64(knots[0])
        var last = Float64(knots[Self.n - 1])
        if not self.extrapolate and (lo < first or hi > last):
            return _nan[Self.dtype]()

        var total = 0.0
        for i in range(pieces):
            var left = Float64(knots[i])
            var start = left if i > 0 else lo
            var stop = Float64(knots[i + 1]) if i < pieces - 1 else hi
            var from_x = max(start, lo)
            var to_x = min(stop, hi)
            if to_x <= from_x:
                continue
            var c0 = Float64(coef[0 * pieces + i])
            var c1 = Float64(coef[1 * pieces + i])
            var c2 = Float64(coef[2 * pieces + i])
            var c3 = Float64(coef[3 * pieces + i])
            var s0 = from_x - left
            var s1 = to_x - left
            total += _antiderivative(c0, c1, c2, c3, s1) - _antiderivative(
                c0, c1, c2, c3, s0
            )
        return Scalar[Self.dtype](sign * total)


def _antiderivative(
    c0: Float64, c1: Float64, c2: Float64, c3: Float64, s: Float64
) -> Float64:
    """`int_0^s (c0 + c1 t + c2 t^2 + c3 t^3) dt`."""
    return s * (c0 + s * (c1 / 2.0 + s * (c2 / 3.0 + s * (c3 / 4.0))))


def _as_float64[dtype: DType](values: List[Scalar[dtype]]) -> List[Float64]:
    var out = List[Float64](capacity=len(values))
    for i in range(len(values)):
        out.append(Float64(values[i]))
    return out^


def _sign[dtype: DType](v: Scalar[dtype]) -> Scalar[dtype]:
    return Scalar[dtype](1) if v > 0 else (
        Scalar[dtype](-1) if v < 0 else Scalar[dtype](0)
    )


def _secants_device[
    dtype: DType, n: Int
](mut x: Static[dtype, n], mut y: Static[dtype, n]) raises -> Static[
    dtype, 2, n - 1
] where (n >= 2):
    """Interval widths (row 0) and secants (row 1) of the knots, on their
    device: one launch, one lane per interval."""
    var ctx = x.context()
    var out = Static[dtype, 2, n - 1]._uninitialized(ctx)
    var xv = x.tile()
    var yv = y.tile()
    var ov = out.tile()

    @always_inline
    def secant[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xv, var yv, var ov}:
        var i = coord_to_index_list(coord)[0]
        var h = xv[Coord(i + 1)][0] - xv[Coord(i)][0]
        ov.store[1](Coord(0, i), h)
        ov.store[1](Coord(1, i), (yv[Coord(i + 1)][0] - yv[Coord(i)][0]) / h)

    elementwise[simd_width=1, target="gpu"](secant, Coord(n - 1), ctx)
    return out^


struct CubicSpline[dtype: DType, n: Int](Movable):
    """A cubic spline through `n` knots, twice continuously differentiable,
    with SciPy's boundary conditions. `scipy.interpolate.CubicSpline(x, y,
    bc_type)`.

    ```mojo
    var spline = CubicSpline(x, y)              # not-a-knot, SciPy's default
    var natural = CubicSpline(x, y, "natural")
    var values = spline(points)
    var slopes = spline[nu=1](points)
    ```

    `bc_type` is `"not-a-knot"` (the default: the first two and last two
    pieces are the same cubic, which is the condition that makes the
    spline reproduce a cubic exactly), `"natural"` (zero second derivative
    at both ends) or `"clamped"` (zero first derivative at both ends).
    Any other string raises. `"periodic"` and value-carrying conditions
    are not provided.

    The knot slopes come from the tridiagonal system SciPy's `_cubic.py`
    assembles, row for row, through `numax.linalg.solve_banded`; the
    `n = 2` and `n = 3` not-a-knot cases are its special cases too. The
    knots need not be uniform, which is the limit
    `numax.interpolate`'s `Array` tier's uniform-grid `ArrayCubicSpline` object has and
    its non-uniform `cubic_spline_moments`/`cubic_spline_eval` overloads
    lift.
    """

    var spline: CubicHermiteSpline[Self.dtype, Self.n]

    def __init__(
        out self,
        mut x: Static[Self.dtype, Self.n],
        mut y: Static[Self.dtype, Self.n],
        bc_type: StaticString = "not-a-knot",
        extrapolate: Bool = True,
    ) raises where (
        Self.dtype.is_floating_point() and Self.n >= 2 and Self.n >= 1
    ):
        """Solve for the knot slopes under `bc_type`, then build the
        `PPoly` form."""
        if not (
            bc_type == "not-a-knot"
            or bc_type == "natural"
            or bc_type == "clamped"
        ):
            raise Error(
                "CubicSpline: unknown bc_type '",
                bc_type,
                "'; expected 'not-a-knot', 'natural' or 'clamped'",
            )
        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not x.on_host() and not (
                bc_type == "not-a-knot" and Self.n <= 3
            ):
                var code = Int32(
                    0 if bc_type
                    == "not-a-knot" else (1 if bc_type == "natural" else 2)
                )
                var d = _spline_slopes_device(x, y, code)
                self.spline = CubicHermiteSpline[
                    Self.dtype, Self.n
                ]._from_device(copy(x), y, d, extrapolate)
                return
        var ctx = x.context()
        var xs = _as_float64(x.to_host())
        var ys = _as_float64(y.to_host())
        comptime pieces = Self.n - 1
        var dx = List[Float64](capacity=pieces)
        var slope = List[Float64](capacity=pieces)
        for i in range(pieces):
            dx.append(xs[i + 1] - xs[i])
            slope.append((ys[i + 1] - ys[i]) / dx[i])

        var d = List[Float64](length=Self.n, fill=0.0)
        if Self.n == 2 and bc_type == "not-a-knot":
            # Two points and no curvature condition: the line through them.
            d[0] = slope[0]
            d[1] = slope[0]
        else:
            # The slope-form tridiagonal system, in `solve_banded`'s
            # `(1, 1)` diagonal-ordered storage: row 0 the superdiagonal,
            # row 1 the diagonal, row 2 the subdiagonal.
            var ab = List[Scalar[Self.dtype]](
                length=3 * Self.n, fill=Scalar[Self.dtype](0)
            )
            var rhs = List[Scalar[Self.dtype]](
                length=Self.n, fill=Scalar[Self.dtype](0)
            )

            @always_inline
            def put(
                mut ab: List[Scalar[Self.dtype]], i: Int, j: Int, v: Float64
            ):
                ab[(1 + i - j) * Self.n + j] = Scalar[Self.dtype](v)

            if Self.n == 3 and bc_type == "not-a-knot":
                # The parabola through three points, as SciPy special-cases
                # it: the general not-a-knot rows would be degenerate.
                put(ab, 0, 0, 1.0)
                put(ab, 0, 1, 1.0)
                put(ab, 1, 0, dx[1])
                put(ab, 1, 1, 2.0 * (dx[0] + dx[1]))
                put(ab, 1, 2, dx[0])
                put(ab, 2, 1, 1.0)
                put(ab, 2, 2, 1.0)
                rhs[0] = Scalar[Self.dtype](2.0 * slope[0])
                rhs[1] = Scalar[Self.dtype](
                    3.0 * (dx[1] * slope[0] + dx[0] * slope[1])
                )
                rhs[2] = Scalar[Self.dtype](2.0 * slope[1])
            else:
                for i in range(1, Self.n - 1):
                    put(ab, i, i - 1, dx[i])
                    put(ab, i, i, 2.0 * (dx[i - 1] + dx[i]))
                    put(ab, i, i + 1, dx[i - 1])
                    rhs[i] = Scalar[Self.dtype](
                        3.0 * (dx[i] * slope[i - 1] + dx[i - 1] * slope[i])
                    )
                var last = Self.n - 1
                if bc_type == "not-a-knot":
                    var span = xs[2] - xs[0]
                    put(ab, 0, 0, dx[1])
                    put(ab, 0, 1, span)
                    rhs[0] = Scalar[Self.dtype](
                        (
                            (dx[0] + 2.0 * span) * dx[1] * slope[0]
                            + dx[0] * dx[0] * slope[1]
                        )
                        / span
                    )
                    var end_span = xs[last] - xs[last - 2]
                    put(ab, last, last, dx[last - 2])
                    put(ab, last, last - 1, end_span)
                    rhs[last] = Scalar[Self.dtype](
                        (
                            dx[last - 1] * dx[last - 1] * slope[last - 2]
                            + (2.0 * end_span + dx[last - 1])
                            * dx[last - 2]
                            * slope[last - 1]
                        )
                        / end_span
                    )
                elif bc_type == "natural":
                    put(ab, 0, 0, 2.0 * dx[0])
                    put(ab, 0, 1, dx[0])
                    rhs[0] = Scalar[Self.dtype](3.0 * (ys[1] - ys[0]))
                    put(ab, last, last, 2.0 * dx[last - 1])
                    put(ab, last, last - 1, dx[last - 1])
                    rhs[last] = Scalar[Self.dtype](
                        3.0 * (ys[last] - ys[last - 1])
                    )
                else:
                    put(ab, 0, 0, 1.0)
                    put(ab, last, last, 1.0)

            var band = Static[Self.dtype, 3, Self.n](ab^, ctx)
            var right = Static[Self.dtype, Self.n](rhs^, ctx)
            var solved = solve_banded[l=1, u=1](band, right).to_host()
            for i in range(Self.n):
                d[i] = Float64(solved[i])

        self.spline = CubicHermiteSpline[Self.dtype, Self.n]._from_host(
            ctx, xs, ys, d, extrapolate
        )

    def __call__[
        T: TensorLike,
        nu: Int = 0,
        gpu: Bool = False,
    ](mut self, points: T) raises -> Static[Self.dtype, dim[T, 0]] where (
        (
            Self.dtype.is_floating_point()
            and Self.n >= 2
            and dim[T, 0] > 0
            and nu >= 0
        )
        and T.dtype == Self.dtype
        and T.LayoutType.rank == 1
        and T.LayoutType.all_dims_known
    ):
        """The spline, or its `nu`-th derivative, at every point."""
        comptime m = dim[T, 0]
        return self.spline.__call__[nu=nu, gpu=gpu](points)

    def integrate(
        mut self, a: Scalar[Self.dtype], b: Scalar[Self.dtype]
    ) raises -> Scalar[Self.dtype] where Self.n >= 2:
        """The definite integral over `[a, b]`."""
        return self.spline.integrate(a, b)


def _spline_slopes_device[
    dtype: DType, n: Int
](mut x: Static[dtype, n], mut y: Static[dtype, n], bc: Int32) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n >= 2):
    """`CubicSpline`'s knot slopes on the knots' device, for `bc` 0
    (not-a-knot, `n >= 4`), 1 (natural) or 2 (clamped).

    One launch assembles the three diagonals and the right-hand side of
    SciPy's slope-form system, with the not-a-knot boundary rows folded
    into rows 1 and `n - 2` and replaced by `d = 0` placeholders; the
    cyclic-reduction solve follows, then one two-lane launch recovers
    `d_0` and `d_{n-1}` from the boundary rows. The module docstring says
    why the fold is needed.
    """
    var ctx = x.context()
    var hs = _secants_device(x, y)
    var sub = Static[dtype, n]._uninitialized(ctx)
    var diag = Static[dtype, n]._uninitialized(ctx)
    var sup = Static[dtype, n]._uninitialized(ctx)
    var rhs = Static[dtype, n]._uninitialized(ctx)
    var hv = hs.tile()
    var av = sub.tile()
    var bv = diag.tile()
    var cv = sup.tile()
    var rv = rhs.tile()
    comptime last = n - 1

    @always_inline
    def assemble[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var hv, var av, var bv, var cv, var rv, var bc}:
        var i = coord_to_index_list(coord)[0]
        var zero = Scalar[dtype](0)
        var a = zero
        var b = Scalar[dtype](1)
        var c = zero
        var r = zero
        if i == 0:
            if bc == 1:
                var h0 = hv[Coord(0, 0)][0]
                b = 2 * h0
                c = h0
                r = 3 * h0 * hv[Coord(1, 0)][0]
        elif i == last:
            if bc == 1:
                var h = hv[Coord(0, last - 1)][0]
                a = h
                b = 2 * h
                r = 3 * h * hv[Coord(1, last - 1)][0]
        else:
            var hl = hv[Coord(0, i - 1)][0]
            var hr = hv[Coord(0, i)][0]
            var sl = hv[Coord(1, i - 1)][0]
            var sr = hv[Coord(1, i)][0]
            a = hr
            b = 2 * (hl + hr)
            c = hl
            r = 3 * (hr * sl + hl * sr)
            if bc == 0 and i == 1:
                # Row 1 minus the not-a-knot row 0: `d_0` drops out.
                var span = hl + hr
                a = zero
                b = span
                r -= ((hl + 2 * span) * hr * sl + hl * hl * sr) / span
            if bc == 0 and i == last - 1:
                # Row n-2 minus the not-a-knot row n-1: `d_{n-1}` drops out.
                var h2 = hv[Coord(0, last - 2)][0]
                var h1 = hv[Coord(0, last - 1)][0]
                var s2 = hv[Coord(1, last - 2)][0]
                var s1 = hv[Coord(1, last - 1)][0]
                var span = h2 + h1
                b = span
                c = zero
                r -= (h1 * h1 * s2 + (2 * span + h1) * h2 * s1) / span
        av.store[1](coord, a)
        bv.store[1](coord, b)
        cv.store[1](coord, c)
        rv.store[1](coord, r)

    elementwise[simd_width=1, target="gpu"](assemble, Coord(n), ctx)
    var d = _tridiagonal_device[dtype, n](sub^, diag^, sup^, rhs^, ctx)
    if bc == 0:
        var dv = d.tile()

        @always_inline
        def ends[w: Int, alignment: Int = 1](coord: Coord) {var hv, var dv}:
            var k = coord_to_index_list(coord)[0]
            if k == 0:
                var h0 = hv[Coord(0, 0)][0]
                var h1 = hv[Coord(0, 1)][0]
                var s0 = hv[Coord(1, 0)][0]
                var s1 = hv[Coord(1, 1)][0]
                var span = h0 + h1
                var r0 = ((h0 + 2 * span) * h1 * s0 + h0 * h0 * s1) / span
                dv.store[1](Coord(0), (r0 - span * dv[Coord(1)][0]) / h1)
            else:
                var h2 = hv[Coord(0, last - 2)][0]
                var h1 = hv[Coord(0, last - 1)][0]
                var s2 = hv[Coord(1, last - 2)][0]
                var s1 = hv[Coord(1, last - 1)][0]
                var span = h2 + h1
                var r = (h1 * h1 * s2 + (2 * span + h1) * h2 * s1) / span
                dv.store[1](
                    Coord(last), (r - span * dv[Coord(last - 1)][0]) / h2
                )

        elementwise[simd_width=1, target="gpu"](ends, Coord(2), ctx)
        ctx.synchronize()
    return d^


struct PchipInterpolator[dtype: DType, n: Int](Movable):
    """The shape-preserving piecewise cubic of Fritsch and Butland.
    `scipy.interpolate.PchipInterpolator(x, y)`.

    Monotone data give a monotone interpolant and there is no overshoot
    between knots, at the cost of only one continuous derivative where
    `CubicSpline` has two. The knot slope is the weighted harmonic mean of
    the neighboring secants -- `w1 = 2h_i + h_{i-1}`, `w2 = h_i + 2h_{i-1}`,
    `d_i = (w1 + w2) / (w1/D_{i-1} + w2/D_i)` -- and zero wherever the
    secants change sign or vanish, which is what stops the overshoot. The
    end slopes are SciPy's one-sided three-point estimate with its two
    shape guards. Local formulas, so construction is `O(n)` with no solve.
    """

    var spline: CubicHermiteSpline[Self.dtype, Self.n]

    def __init__(
        out self,
        mut x: Static[Self.dtype, Self.n],
        mut y: Static[Self.dtype, Self.n],
        extrapolate: Bool = True,
    ) raises where Self.dtype.is_floating_point() and Self.n >= 2:
        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not x.on_host():
                var d = _pchip_slopes_device(x, y)
                self.spline = CubicHermiteSpline[
                    Self.dtype, Self.n
                ]._from_device(copy(x), y, d, extrapolate)
                return
        var ctx = x.context()
        var xs = _as_float64(x.to_host())
        var ys = _as_float64(y.to_host())
        comptime pieces = Self.n - 1
        var h = List[Float64](capacity=pieces)
        var secant = List[Float64](capacity=pieces)
        for i in range(pieces):
            h.append(xs[i + 1] - xs[i])
            secant.append((ys[i + 1] - ys[i]) / h[i])

        var d = List[Float64](length=Self.n, fill=0.0)
        if Self.n == 2:
            d[0] = secant[0]
            d[1] = secant[0]
        else:
            for i in range(1, Self.n - 1):
                var m0 = secant[i - 1]
                var m1 = secant[i]
                if _sign(m0) != _sign(m1) or m0 == 0 or m1 == 0:
                    d[i] = 0.0
                else:
                    var w1 = 2.0 * h[i] + h[i - 1]
                    var w2 = h[i] + 2.0 * h[i - 1]
                    d[i] = (w1 + w2) / (w1 / m0 + w2 / m1)
            d[0] = _pchip_edge(h[0], h[1], secant[0], secant[1])
            d[Self.n - 1] = _pchip_edge(
                h[pieces - 1],
                h[pieces - 2],
                secant[pieces - 1],
                secant[pieces - 2],
            )

        self.spline = CubicHermiteSpline[Self.dtype, Self.n]._from_host(
            ctx, xs, ys, d, extrapolate
        )

    def __call__[
        T: TensorLike,
        nu: Int = 0,
        gpu: Bool = False,
    ](mut self, points: T) raises -> Static[Self.dtype, dim[T, 0]] where (
        (
            Self.dtype.is_floating_point()
            and Self.n >= 2
            and dim[T, 0] > 0
            and nu >= 0
        )
        and T.dtype == Self.dtype
        and T.LayoutType.rank == 1
        and T.LayoutType.all_dims_known
    ):
        """The interpolant, or its `nu`-th derivative, at every point."""
        comptime m = dim[T, 0]
        return self.spline.__call__[nu=nu, gpu=gpu](points)

    def integrate(
        mut self, a: Scalar[Self.dtype], b: Scalar[Self.dtype]
    ) raises -> Scalar[Self.dtype] where Self.n >= 2:
        """The definite integral over `[a, b]`."""
        return self.spline.integrate(a, b)


def _pchip_edge[
    dtype: DType
](
    h0: Scalar[dtype], h1: Scalar[dtype], m0: Scalar[dtype], m1: Scalar[dtype]
) -> Scalar[dtype]:
    """SciPy's `_edge_case`: the one-sided three-point slope estimate,
    zeroed if it disagrees in sign with the first secant and capped at
    three times that secant when the two secants disagree."""
    var d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if _sign(d) != _sign(m0):
        return Scalar[dtype](0)
    if _sign(m0) != _sign(m1) and abs(d) > 3 * abs(m0):
        return 3 * m0
    return d


def _pchip_slopes_device[
    dtype: DType, n: Int
](mut x: Static[dtype, n], mut y: Static[dtype, n]) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n >= 2):
    """`PchipInterpolator`'s knot slopes on the knots' device: one lane
    per knot, the host loop's formulas."""
    var ctx = x.context()
    var hs = _secants_device(x, y)
    var d = Static[dtype, n]._uninitialized(ctx)
    var hv = hs.tile()
    var dv = d.tile()
    comptime pieces = n - 1

    @always_inline
    def slope[w: Int, alignment: Int = 1](coord: Coord) {var hv, var dv}:
        var i = coord_to_index_list(coord)[0]
        var zero = Scalar[dtype](0)
        var out: Scalar[dtype]
        comptime if n == 2:
            out = hv[Coord(1, 0)][0]
        else:
            if i == 0:
                out = _pchip_edge(
                    hv[Coord(0, 0)][0],
                    hv[Coord(0, 1)][0],
                    hv[Coord(1, 0)][0],
                    hv[Coord(1, 1)][0],
                )
            elif i == n - 1:
                out = _pchip_edge(
                    hv[Coord(0, pieces - 1)][0],
                    hv[Coord(0, pieces - 2)][0],
                    hv[Coord(1, pieces - 1)][0],
                    hv[Coord(1, pieces - 2)][0],
                )
            else:
                var m0 = hv[Coord(1, i - 1)][0]
                var m1 = hv[Coord(1, i)][0]
                if _sign(m0) != _sign(m1) or m0 == zero or m1 == zero:
                    out = zero
                else:
                    var h0 = hv[Coord(0, i - 1)][0]
                    var h1 = hv[Coord(0, i)][0]
                    var w1 = 2 * h1 + h0
                    var w2 = h1 + 2 * h0
                    out = (w1 + w2) / (w1 / m0 + w2 / m1)
        dv.store[1](coord, out)

    elementwise[simd_width=1, target="gpu"](slope, Coord(n), ctx)
    return d^


def _akima_slopes_device[
    dtype: DType, n: Int
](mut x: Static[dtype, n], mut y: Static[dtype, n]) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n >= 3 and n >= 2):
    """`Akima1DInterpolator`'s knot slopes on the knots' device.

    One launch for the `n + 3` extended secants (the virtual ones by the
    host's linear extension, written out in closed form), one for each
    knot's weight sum, one `reduce_all` max for the threshold, and one for
    the slopes.
    """
    var ctx = x.context()
    var hs = _secants_device(x, y)
    var m = Static[dtype, n + 3]._uninitialized(ctx)
    var f = Static[dtype, n]._uninitialized(ctx)
    var largest = Static[dtype, 1]._uninitialized(ctx)
    var d = Static[dtype, n]._uninitialized(ctx)
    var hv = hs.tile()
    var mv = m.tile()
    var fv = f.tile()
    var lv = largest.tile()
    var dv = d.tile()
    comptime pieces = n - 1

    @always_inline
    def extend[w: Int, alignment: Int = 1](coord: Coord) {var hv, var mv}:
        var k = coord_to_index_list(coord)[0]
        var value: Scalar[dtype]
        if k == 0:
            value = 3 * hv[Coord(1, 0)][0] - 2 * hv[Coord(1, 1)][0]
        elif k == 1:
            value = 2 * hv[Coord(1, 0)][0] - hv[Coord(1, 1)][0]
        elif k == n + 1:
            value = (
                2 * hv[Coord(1, pieces - 1)][0] - hv[Coord(1, pieces - 2)][0]
            )
        elif k == n + 2:
            value = (
                3 * hv[Coord(1, pieces - 1)][0]
                - 2 * hv[Coord(1, pieces - 2)][0]
            )
        else:
            value = hv[Coord(1, k - 2)][0]
        mv.store[1](coord, value)

    elementwise[simd_width=1, target="gpu"](extend, Coord(n + 3), ctx)

    @always_inline
    def weigh[w: Int, alignment: Int = 1](coord: Coord) {var mv, var fv}:
        var i = coord_to_index_list(coord)[0]
        var f1 = abs(mv[Coord(i + 3)][0] - mv[Coord(i + 2)][0])
        var f2 = abs(mv[Coord(i + 1)][0] - mv[Coord(i)][0])
        fv.store[1](coord, f1 + f2)

    elementwise[simd_width=1, target="gpu"](weigh, Coord(n), ctx)

    @always_inline
    def identity[
        w: Int
    ](tile: SIMD[dtype, w], idx: RowCoord[1]) {} -> SIMD[dtype, w]:
        return tile

    reduce_all[monoid="max", gpu=True](
        fv, lv.as_unsafe_any_origin(), identity, n, Optional(ctx)
    )

    @always_inline
    def slope[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var mv, var fv, var lv, var dv}:
        var i = coord_to_index_list(coord)[0]
        var f1 = abs(mv[Coord(i + 3)][0] - mv[Coord(i + 2)][0])
        var f2 = abs(mv[Coord(i + 1)][0] - mv[Coord(i)][0])
        var f12 = fv[coord][0]
        var out: Scalar[dtype]
        if f12 > Scalar[dtype](1e-9) * lv[Coord(0)][0]:
            out = (f1 * mv[Coord(i + 1)][0] + f2 * mv[Coord(i + 2)][0]) / f12
        else:
            out = Scalar[dtype](0.5) * (
                mv[Coord(i + 1)][0] + mv[Coord(i + 2)][0]
            )
        dv.store[1](coord, out)

    elementwise[simd_width=1, target="gpu"](slope, Coord(n), ctx)
    ctx.synchronize()
    return d^


struct Akima1DInterpolator[dtype: DType, n: Int](Movable):
    """Akima's piecewise cubic, whose knot slopes weight the neighboring
    secants by how much the secants on the *other* side change.
    `scipy.interpolate.Akima1DInterpolator(x, y)`, SciPy's `"akima"`
    method.

    Continuous first derivative, no linear solve, and less prone than
    `CubicSpline` to wiggle after a jump in the data because a slope is
    influenced by its four nearest secants and nothing further. The two
    virtual secants past each end are SciPy's linear extension. Where the
    weights both vanish -- the data locally a straight line -- the slope is
    the mean of the two adjacent secants, with SciPy's `1e-9` relative
    threshold deciding "vanish".

    Needs `n >= 3`, and returns NaN outside the knots by default, both as
    SciPy has it; pass `extrapolate=True` for the end cubics instead.
    """

    var spline: CubicHermiteSpline[Self.dtype, Self.n]

    def __init__(
        out self,
        mut x: Static[Self.dtype, Self.n],
        mut y: Static[Self.dtype, Self.n],
        extrapolate: Bool = False,
    ) raises where (
        Self.dtype.is_floating_point() and Self.n >= 3 and Self.n >= 2
    ):
        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not x.on_host():
                var d = _akima_slopes_device(x, y)
                self.spline = CubicHermiteSpline[
                    Self.dtype, Self.n
                ]._from_device(copy(x), y, d, extrapolate)
                return
        var ctx = x.context()
        var xs = _as_float64(x.to_host())
        var ys = _as_float64(y.to_host())
        comptime pieces = Self.n - 1
        # Secants with two virtual ones on each side, at `m[2 .. n]`.
        var m = List[Float64](length=Self.n + 3, fill=0.0)
        for i in range(pieces):
            m[i + 2] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
        m[1] = 2.0 * m[2] - m[3]
        m[0] = 2.0 * m[1] - m[2]
        m[Self.n + 1] = 2.0 * m[Self.n] - m[Self.n - 1]
        m[Self.n + 2] = 2.0 * m[Self.n + 1] - m[Self.n]

        var dm = List[Float64](capacity=Self.n + 2)
        for i in range(Self.n + 2):
            dm.append(abs(m[i + 1] - m[i]))
        var largest = 0.0
        for i in range(Self.n):
            largest = max(largest, dm[i + 2] + dm[i])

        var d = List[Float64](length=Self.n, fill=0.0)
        for i in range(Self.n):
            var f1 = dm[i + 2]
            var f2 = dm[i]
            var f12 = f1 + f2
            if f12 > 1e-9 * largest:
                d[i] = (f1 * m[i + 1] + f2 * m[i + 2]) / f12
            else:
                d[i] = 0.5 * (m[i + 1] + m[i + 2])

        self.spline = CubicHermiteSpline[Self.dtype, Self.n]._from_host(
            ctx, xs, ys, d, extrapolate
        )

    def __call__[
        T: TensorLike,
        nu: Int = 0,
        gpu: Bool = False,
    ](mut self, points: T) raises -> Static[Self.dtype, dim[T, 0]] where (
        (
            Self.dtype.is_floating_point()
            and Self.n >= 2
            and dim[T, 0] > 0
            and nu >= 0
        )
        and T.dtype == Self.dtype
        and T.LayoutType.rank == 1
        and T.LayoutType.all_dims_known
    ):
        """The interpolant, or its `nu`-th derivative, at every point;
        NaN outside the knots unless built with `extrapolate=True`."""
        comptime m = dim[T, 0]
        return self.spline.__call__[nu=nu, gpu=gpu](points)

    def integrate(
        mut self, a: Scalar[Self.dtype], b: Scalar[Self.dtype]
    ) raises -> Scalar[Self.dtype] where Self.n >= 2:
        """The definite integral over `[a, b]`."""
        return self.spline.integrate(a, b)
