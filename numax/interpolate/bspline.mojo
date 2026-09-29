"""B-splines of any degree: `BSpline` and `make_interp_spline`, SciPy's
`scipy.interpolate.BSpline` and `make_interp_spline`.

**Tier 2.** A `BSpline` is SciPy's `(t, c, k)`: knots, coefficients and
degree, both arrays on the device they were made on.

- **Evaluation** is one lane per point: the interval `t[l] <= x < t[l+1]`
  by bisection, clamped to `[k, n - 1]` so a point outside the base
  interval extrapolates the end polynomial, as SciPy's `extrapolate=True`
  does; then the `k + 1` nonzero basis functions by the Cox-de Boor
  recursion, SciPy's `_deBoor_D` at order zero, dotted with their
  coefficients.
- **`derivative`** and **`antiderivative`** are SciPy's `splder` and
  `splantider` coefficient maps -- `k (c_{i+1} - c_i) / (t_{i+k+1} -
  t_{i+1})` on the inner knots, and the running sum of `c_i (t_{i+k+1} -
  t_i) / (k + 1)` on knots extended at both ends -- each a one-lane
  device pass. `integrate(a, b)` is the antiderivative's difference.
- **`make_interp_spline[k]`** picks SciPy's knots (not-a-knot for odd
  `k`, its Greville-based rule at `k = 2`, the data itself at `k = 0, 1`)
  and solves the collocation system `B_j(x_i) c_j = y_i`: banded, `k`
  sub- and superdiagonals, through `solve_banded` on the host, and dense
  through `solve` at `gpu=True`, where the general banded solve has no
  device path.

Degrees up to 15, the size of the in-lane basis array.

## The MAX gate

Nothing: MAX has no spline basis. **Extend.**
"""

from std.collections import Array

from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core._drive import _check_device, _notice
from ..core.tensorlike import TensorLike, dim
from ..core.tensor import Dynamic, Static, _dyn_shape, _same_order, asarray
from ..linalg.banded import solve_banded
from ..linalg.basic import solve

comptime _MAX_DEGREE = 15


def _target[gpu: Bool]() -> StaticString:
    comptime if gpu:
        return "gpu"
    else:
        return "cpu"


def _vec[
    dtype: DType
](count: Int, ctx: DeviceContext) raises -> Dynamic[dtype, 1]:
    return Dynamic[dtype, 1](row_major(_dyn_shape[1](count)), ctx)


@always_inline
def _find_interval[
    dtype: DType
](
    t: Pointer[Scalar[dtype], ImmutAnyOrigin], k: Int, n: Int, x: Scalar[dtype]
) -> Int:
    """SciPy's `find_interval` with extrapolation: the `l` in `[k, n - 1]`
    with `t[l] <= x < t[l + 1]`, the ends taking the end intervals."""
    if x <= t[unsafe_offset=k]:
        return k
    if x >= t[unsafe_offset=n]:
        return n - 1
    var lo = k
    var hi = n - 1
    while lo < hi:
        var mid = (lo + hi + 1) // 2
        if t[unsafe_offset=mid] <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo


@always_inline
def _basis[
    dtype: DType
](
    t: Pointer[Scalar[dtype], ImmutAnyOrigin],
    k: Int,
    ell: Int,
    x: Scalar[dtype],
) -> Array[Scalar[dtype], _MAX_DEGREE + 1] where dtype.is_floating_point():
    """The `k + 1` B-splines nonzero at `x` in interval `ell`, `B_{ell-k}`
    first: SciPy's `_deBoor_D` recursion at order zero."""
    var h = Array[Scalar[dtype], _MAX_DEGREE + 1](fill=Scalar[dtype](0))
    var hh = Array[Scalar[dtype], _MAX_DEGREE + 1](fill=Scalar[dtype](0))
    h[0] = Scalar[dtype](1)
    for j in range(1, k + 1):
        for a in range(j):
            hh[a] = h[a]
        h[0] = Scalar[dtype](0)
        for m in range(1, j + 1):
            var ind = ell + m
            var xb = t[unsafe_offset=ind]
            var xa = t[unsafe_offset=ind - j]
            if xb == xa:
                h[m] = Scalar[dtype](0)
                continue
            var w = hh[m - 1] / (xb - xa)
            h[m - 1] += w * (xb - x)
            h[m] = w * (x - xa)
    return h^


struct BSpline[dtype: DType, gpu: Bool = False](Movable):
    """A spline of degree `k` in the B-spline basis: `S(x) = sum_j c_j
    B_{j,k}(x)` on the knots `t`. `scipy.interpolate.BSpline(t, c, k)`;
    the module docstring has the evaluation and the coefficient maps.

    `c` holds `len(t) - k - 1` coefficients; evaluation extrapolates past
    the base interval `[t[k], t[n]]`, SciPy's default.
    """

    var t: Dynamic[Self.dtype, 1]
    """The knots, non-decreasing."""
    var c: Dynamic[Self.dtype, 1]
    """The coefficients, `len(t) - k - 1` of them."""
    var k: Int
    """The degree."""

    def __init__[
        T: TensorLike, C: TensorLike
    ](out self, t: T, c: C, k: Int) raises where (
        T.dtype == Self.dtype
        and C.dtype == Self.dtype
        and T.LayoutType.rank == 1
        and C.LayoutType.rank == 1
    ):
        """A spline from knots, coefficients and degree.

        Parameters:
            T: The tensor type of `t`.
            C: The tensor type of `c`.

        Args:
            t: The knots, non-decreasing, at least `2 k + 2` of them.
            c: The coefficients; only the first `len(t) - k - 1` are used,
                as SciPy uses them.
            k: The degree, `0 <= k <= 15`.

        Raises:
            If `k` is out of range or there are too few knots or
            coefficients.
        """
        if k < 0 or k > _MAX_DEGREE:
            raise Error("BSpline: the degree must be in [0, 15], got ", k)
        var nt = t.size()
        var n = nt - k - 1
        if n < k + 1:
            raise Error("BSpline: ", nt, " knots are too few for degree ", k)
        if c.size() < n:
            raise Error(
                "BSpline: ",
                c.size(),
                " coefficients for ",
                n,
                " basis functions",
            )
        self.t = rebind_var[Dynamic[Self.dtype, 1]](
            _same_order(t, row_major(_dyn_shape[1](nt)))
        )
        self.c = rebind_var[Dynamic[Self.dtype, 1]](
            _same_order(c, row_major(_dyn_shape[1](c.size())))
        )
        self.k = k

    def __init__(
        out self,
        var t: Dynamic[Self.dtype, 1],
        var c: Dynamic[Self.dtype, 1],
        k: Int,
        *,
        trusted: Bool,
    ):
        """Wrap arrays already checked, as the coefficient maps build them."""
        self.t = t^
        self.c = c^
        self.k = k

    def __call__[
        X: TensorLike
    ](self, x: X, nu: Int = 0) raises -> Dynamic[Self.dtype, 1] where (
        X.dtype == Self.dtype
        and X.LayoutType.rank == 1
        and Self.dtype.is_floating_point()
    ):
        """The spline, or its `nu`-th derivative, at the points `x`.
        `BSpline.__call__(x, nu)`.

        Parameters:
            X: The tensor type of `x`.

        Args:
            x: The points, on the spline's device.
            nu: The derivative order, `0 <= nu <= k`.

        Returns:
            The values, one per point.

        Raises:
            If `nu` is out of range, or a device operation fails.
        """
        if nu < 0 or nu > self.k:
            raise Error(
                "BSpline: derivative order ", nu, " for degree ", self.k
            )
        if nu > 0:
            var d = self.derivative(nu)
            return d(x)
        var q = x.size()
        var ctx = self.t.context()
        var xs = rebind_var[Dynamic[Self.dtype, 1]](
            _same_order(x, row_major(_dyn_shape[1](q)))
        )
        var out = _vec[Self.dtype](q, ctx)
        var tp = self.t.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
        var cp = self.c.tile()
        var xp = xs.tile()
        var op = out.tile()
        var k = self.k
        var n = self.t.size() - k - 1

        @always_inline
        def body[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var tp, var cp, var xp, var op, var k, var n}:
            var i = coord_to_index_list(coord)[0]
            var at = xp.ptr[unsafe_offset=i]
            var ell = _find_interval[Self.dtype](tp, k, n, at)
            var b = _basis[Self.dtype](tp, k, ell, at)
            var acc = Scalar[Self.dtype](0)
            for a in range(k + 1):
                acc += cp.ptr[unsafe_offset=ell - k + a] * b[a]
            op.ptr[unsafe_offset=i] = acc

        elementwise[simd_width=1, target=_target[Self.gpu]()](
            body, Coord(q), ctx
        )
        ctx.synchronize()
        _ = xs^
        return out^

    def derivative(
        self, nu: Int = 1
    ) raises -> Self where Self.dtype.is_floating_point():
        """The `nu`-th derivative as a spline of degree `k - nu`, SciPy's
        `BSpline.derivative` through `splder`.

        Args:
            nu: The order, `0 <= nu <= k`.

        Returns:
            The derivative spline.

        Raises:
            If `nu` is out of range, or a device operation fails.
        """
        if nu < 0 or nu > self.k:
            raise Error(
                "BSpline: derivative order ", nu, " for degree ", self.k
            )
        var t = _same_order(self.t, row_major(_dyn_shape[1](self.t.size())))
        var c = _padded[gpu=Self.gpu](self.c, t.size())
        var k = self.k
        for _ in range(nu):
            var nt = t.size()
            var ctx = t.context()
            var t_new = _vec[Self.dtype](nt - 2, ctx)
            var c_new = _vec[Self.dtype](nt - 2, ctx)
            var tp = t.tile()
            var cp = c.tile()
            var tn = t_new.tile()
            var cn = c_new.tile()

            @always_inline
            def step[
                w: Int, alignment: Int = 1
            ](coord: Coord) {var tp, var cp, var tn, var cn, var nt, var k}:
                for i in range(nt - 2):
                    tn.ptr[unsafe_offset=i] = tp.ptr[unsafe_offset=i + 1]
                    cn.ptr[unsafe_offset=i] = Scalar[Self.dtype](0)
                for i in range(nt - 2 - k):
                    var dt = (
                        tp.ptr[unsafe_offset=i + k + 1]
                        - tp.ptr[unsafe_offset=i + 1]
                    )
                    cn.ptr[unsafe_offset=i] = (
                        (cp.ptr[unsafe_offset=i + 1] - cp.ptr[unsafe_offset=i])
                        * Scalar[Self.dtype](k)
                        / dt
                    )

            elementwise[simd_width=1, target=_target[Self.gpu]()](
                step, Coord(1), ctx
            )
            ctx.synchronize()
            t = t_new^
            c = c_new^
            k -= 1
        return Self(t^, c^, k, trusted=True)

    def antiderivative(
        self, nu: Int = 1
    ) raises -> Self where Self.dtype.is_floating_point():
        """The `nu`-th antiderivative as a spline of degree `k + nu`, zero at
        `t[0]`: SciPy's `BSpline.antiderivative` through `splantider`.

        Args:
            nu: The order, `nu >= 0`, with `k + nu <= 15`.

        Returns:
            The antiderivative spline.

        Raises:
            If the degree would pass 15, or a device operation fails.
        """
        if nu < 0 or self.k + nu > _MAX_DEGREE:
            raise Error(
                "BSpline: antiderivative order ", nu, " for degree ", self.k
            )
        var t = _same_order(self.t, row_major(_dyn_shape[1](self.t.size())))
        var c = _padded[gpu=Self.gpu](self.c, t.size())
        var k = self.k
        for _ in range(nu):
            var nt = t.size()
            var ctx = t.context()
            var t_new = _vec[Self.dtype](nt + 2, ctx)
            var c_new = _vec[Self.dtype](nt + 2, ctx)
            var tp = t.tile()
            var cp = c.tile()
            var tn = t_new.tile()
            var cn = c_new.tile()

            @always_inline
            def step[
                w: Int, alignment: Int = 1
            ](coord: Coord) {var tp, var cp, var tn, var cn, var nt, var k}:
                tn.ptr[unsafe_offset=0] = tp.ptr[unsafe_offset=0]
                for i in range(nt):
                    tn.ptr[unsafe_offset=i + 1] = tp.ptr[unsafe_offset=i]
                tn.ptr[unsafe_offset=nt + 1] = tp.ptr[unsafe_offset=nt - 1]
                var running = Scalar[Self.dtype](0)
                cn.ptr[unsafe_offset=0] = running
                var count = nt - k - 1
                for i in range(count):
                    var dt = (
                        tp.ptr[unsafe_offset=i + k + 1]
                        - tp.ptr[unsafe_offset=i]
                    )
                    running += (
                        cp.ptr[unsafe_offset=i] * dt / Scalar[Self.dtype](k + 1)
                    )
                    cn.ptr[unsafe_offset=i + 1] = running
                for i in range(count + 1, nt + 2):
                    cn.ptr[unsafe_offset=i] = running

            elementwise[simd_width=1, target=_target[Self.gpu]()](
                step, Coord(1), ctx
            )
            ctx.synchronize()
            t = t_new^
            c = c_new^
            k += 1
        return Self(t^, c^, k, trusted=True)

    def integrate(
        self, a: Float64, b: Float64
    ) raises -> Float64 where Self.dtype.is_floating_point():
        """The definite integral over `[a, b]`, extrapolating past the base
        interval, SciPy's `BSpline.integrate(a, b)`: the antiderivative's
        difference.

        Args:
            a: The lower limit.
            b: The upper limit; `b < a` gives the negated integral.

        Returns:
            The integral.

        Raises:
            If the degree is already 15, or a device operation fails.
        """
        var anti = self.antiderivative()
        var ctx = self.t.context()
        var ends = asarray[Self.dtype](
            [Scalar[Self.dtype](a), Scalar[Self.dtype](b)], ctx
        )
        var v = anti(ends).to_host()
        return Float64(v[1]) - Float64(v[0])


def _padded[
    dtype: DType, gpu: Bool
](c: Dynamic[dtype, 1], length: Int) raises -> Dynamic[
    dtype, 1
] where dtype.is_floating_point():
    """`c` extended with zeros to `length`, FITPACK's coefficient
    convention the coefficient maps index against."""
    var ctx = c.context()
    var out = _vec[dtype](length, ctx)
    var count = min(c.size(), length)
    var cp = c.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var op = out.tile()

    @always_inline
    def copy_in[w: Int, alignment: Int = 1](coord: Coord) {var cp, var op}:
        var i = coord_to_index_list(coord)[0]
        op.ptr[unsafe_offset=i] = cp[unsafe_offset=i]

    if count > 0:
        elementwise[simd_width=1, target=_target[gpu]()](
            copy_in, Coord(count), ctx
        )
        ctx.synchronize()
    return out^


def _interp_knots[
    dtype: DType, k: Int
](xs: List[Float64]) -> List[Scalar[dtype]]:
    """SciPy `make_interp_spline`'s default knots for degree `k` without
    boundary conditions."""
    var n = len(xs)
    var t = List[Scalar[dtype]]()
    comptime if k == 0:
        for v in xs:
            t.append(Scalar[dtype](v))
        t.append(Scalar[dtype](xs[n - 1]))
    elif k == 1:
        t.append(Scalar[dtype](xs[0]))
        for v in xs:
            t.append(Scalar[dtype](v))
        t.append(Scalar[dtype](xs[n - 1]))
    elif k == 2:
        for _ in range(3):
            t.append(Scalar[dtype](xs[0]))
        # Greville sites, the second and the second-to-last omitted.
        for i in range(1, n - 2):
            t.append(Scalar[dtype]((xs[i] + xs[i + 1]) / 2.0))
        for _ in range(3):
            t.append(Scalar[dtype](xs[n - 1]))
    else:
        # Not-a-knot.
        comptime m = (k - 1) // 2
        for _ in range(k + 1):
            t.append(Scalar[dtype](xs[0]))
        for i in range(m + 1, n - m - 1):
            t.append(Scalar[dtype](xs[i]))
        for _ in range(k + 1):
            t.append(Scalar[dtype](xs[n - 1]))
    return t^


def make_interp_spline[
    X: TensorLike, Y: TensorLike, k: Int = 3, gpu: Bool = False
](x: X, y: Y) raises -> BSpline[X.dtype, gpu] where (
    X.dtype.is_floating_point()
    and Y.dtype == X.dtype
    and X.LayoutType.rank == 1
    and X.LayoutType.all_dims_known
    and Y.LayoutType.rank == 1
    and Y.LayoutType.all_dims_known
    and dim[Y, 0] == dim[X, 0]
    and dim[X, 0] >= 1
    and k >= 0
    and k <= 15
):
    """The interpolating B-spline of degree `k` through `(x, y)`.
    `scipy.interpolate.make_interp_spline(x, y, k)`, with SciPy's default
    knots (no `bc_type`): not-a-knot for odd `k`.

    The collocation system `B_j(x_i) c_j = y_i` is assembled on the data's
    device, one lane per row writing its `k + 1` nonzeros; per the module
    docstring it is solved banded on the host and dense on the device.

    Parameters:
        X: The tensor type of `x`, strictly increasing, static length.
        Y: The tensor type of `y`, the same length and `dtype`.
        k: The degree, SciPy's default `3`.
        gpu: Whether to assemble and solve on the data's device.

    Args:
        x: The abscissas, strictly increasing, more than `k` of them.
        y: The values.

    Returns:
        The interpolating `BSpline`.

    Raises:
        If there are too few points, or the solve fails.
    """
    comptime dtype = X.dtype
    comptime n = dim[X, 0]
    var xs = _same_order(x, Static[dtype, n]._static_layout())
    var ys = rebind_var[Static[dtype, n]](
        _same_order(y, Static[Y.dtype, n]._static_layout())
    )
    return _make_interp_spline[dtype, n, k, gpu](xs, ys)


def _make_interp_spline[
    dtype: DType, n: Int, k: Int, gpu: Bool
](x: Static[dtype, n], y: Static[dtype, n]) raises -> BSpline[
    dtype, gpu
] where (dtype.is_floating_point() and n >= 1 and k >= 0 and k <= 15):
    """`make_interp_spline` over one concrete type, so the solves see a
    single `dtype` and length."""
    comptime assert n > k, "make_interp_spline: need more than k points"
    var hx = x.to_host()
    var xs = List[Float64](capacity=n)
    for i in range(n):
        xs.append(Float64(hx[i]))
    var ctx = x.context()
    var t = asarray(_interp_knots[dtype, k](xs), ctx)
    var xd = _same_order(x, row_major(_dyn_shape[1](n)))
    var tp = (
        t.tile()
        .ptr.unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    var xp = xd.tile()
    var basis_count = t.size() - k - 1
    comptime if gpu:
        if x.on_host():
            _notice[gpu]("make_interp_spline")
            var cpu = DeviceContext(api="cpu")
            return rebind_var[BSpline[dtype, gpu]](
                _make_interp_spline[dtype, n, k, False](
                    Static[dtype, n](x.to_host(), cpu),
                    Static[dtype, n](y.to_host(), cpu),
                )
            )
        var a = Static[dtype, n, n](ctx)
        var ap = a.tile()

        @always_inline
        def dense[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var tp, var xp, var ap, var basis_count}:
            var i = coord_to_index_list(coord)[0]
            var at = xp.ptr[unsafe_offset=i]
            var ell = _find_interval[dtype](tp, k, basis_count, at)
            var b = _basis[dtype](tp, k, ell, at)
            for a2 in range(k + 1):
                ap.ptr[unsafe_offset=i * n + ell - k + a2] = b[a2]

        elementwise[simd_width=1, target="gpu"](dense, Coord(n), ctx)
        ctx.synchronize()
        var c = solve[gpu=True](a, y)
        return BSpline[dtype, gpu](
            t^,
            _same_order(c, row_major(_dyn_shape[1](n))),
            k,
            trusted=True,
        )
    else:
        if not x.on_host():
            _notice[gpu]("make_interp_spline")
            var cpu = DeviceContext(api="cpu")
            return rebind_var[BSpline[dtype, gpu]](
                _make_interp_spline[dtype, n, k, False](
                    Static[dtype, n](x.to_host(), cpu),
                    Static[dtype, n](y.to_host(), cpu),
                )
            )
        var band = Static[dtype, 2 * k + 1, n](ctx)
        var bp = band.tile()

        @always_inline
        def banded[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var tp, var xp, var bp, var basis_count}:
            # `ab[u + i - j, j] = a[i, j]`, `u = k`.
            var i = coord_to_index_list(coord)[0]
            var at = xp.ptr[unsafe_offset=i]
            var ell = _find_interval[dtype](tp, k, basis_count, at)
            var b = _basis[dtype](tp, k, ell, at)
            for a2 in range(k + 1):
                var j = ell - k + a2
                bp.ptr[unsafe_offset=(k + i - j) * n + j] = b[a2]

        elementwise[simd_width=1, target="cpu"](banded, Coord(n), ctx)
        ctx.synchronize()
        var c = solve_banded[l=k, u=k](band, y)
        return BSpline[dtype, gpu](
            t^,
            _same_order(c, row_major(_dyn_shape[1](n))),
            k,
            trusted=True,
        )
