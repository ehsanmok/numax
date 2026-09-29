"""Boundary-value problems: `solve_bvp`, SciPy's `scipy.integrate.solve_bvp`,
and the `BVPResult` it returns.

**Tier 2.** SciPy's algorithm (Kierzenka and Shampine's, the one in
MATLAB's `bvp4c`), transcribed step for step:

- **Collocation.** The fourth-order Lobatto IIIA method on the mesh `x`:
  on each interval the residual `y_{i+1} - y_i - h/6 (f_i + f_{i+1} + 4
  f_mid)`, with `y_mid` from the cubic through the ends and their slopes.
- **Newton.** SciPy's damped Newton (`solve_newton`): the Jacobian by
  forward differences when none is given, its step accepted by an
  Armijo test on the affine-invariant criterion `|J^-1 r|^2` with up to
  four halvings, at most four Jacobians and eight iterations, reusing the
  Jacobian after a full step. It stops when every collocation residual
  is below `(2/3) h 0.05 tol (1 + |f_mid|)` and the boundary residual
  below `bc_tol`.
- **Error and mesh.** The solution is the cubic Hermite spline through
  `y` and `f`. Its relative residual `y' - f` is estimated on each
  interval by 5-point Lobatto quadrature, and SciPy's rule inserts one
  node where it lies in `(tol, 100 tol)` and two where it is above.

Every phase is a device kernel over the mesh at `gpu=True`: the
collocation residuals and midpoints, the difference Jacobians (`n + 1`
vectorized calls of `fun`), the assembly of the `n m x n m` Newton
system, its factorization (`numax.linalg`'s blocked LU at a run-time
order), the residual estimate, and the mesh insertion, which is a
one-lane offsets pass and a parallel scatter evaluating the spline at the
new nodes. The host drives the iterations and reads one scalar at each
decision -- a cost, a maximum, a node count.

**Ceiling.** The Newton system is assembled and factored dense, `(n m)^2`
storage and `(n m)^3 / 3` flops, where SciPy's is sparse. It is block
bidiagonal apart from the boundary rows, and an almost-block-diagonal
solver is the upgrade; `max_nodes` bounds the size meanwhile. Unknown
parameters (`p`), the singular term `S` and analytic Jacobians are not
taken.

## The MAX gate

Nothing: MAX has no ODE solver of any kind. **Extend.**
"""

from algorithm.rowwise_types import RowCoord
from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.math import sqrt

from ..core.rowwise import reduce_all
from ..core.tensorlike import TensorLike
from ..core.tensor import Dynamic, _dyn_shape, _same_order, copy
from ..linalg.lu import _lu_factor_runtime
from .ode import _target


def _sqrt_eps[dtype: DType]() -> Scalar[dtype]:
    """The forward-difference step's scale, `sqrt(eps)` of `dtype`: SciPy's
    `EPS**0.5` is `float64`'s, and at `float32` that step would round
    away entirely."""
    comptime if dtype == DType.float64:
        return Scalar[dtype](1.4901161193847656e-08)
    else:
        return Scalar[dtype](3.4526698e-04)


def _vec[
    dtype: DType
](count: Int, ctx: DeviceContext) raises -> Dynamic[dtype, 1]:
    return Dynamic[dtype, 1](row_major(_dyn_shape[1](count)), ctx)


def _mat[
    dtype: DType
](rows: Int, cols: Int, ctx: DeviceContext) raises -> Dynamic[dtype, 2]:
    return Dynamic[dtype, 2](row_major(_dyn_shape[2](rows, cols)), ctx)


def _max_of[
    dtype: DType, gpu: Bool
](mut t: Dynamic[dtype, 1]) raises -> Float64 where dtype.is_floating_point():
    """The largest entry, one scalar read back."""
    var count = t.size()
    comptime if gpu:
        var ctx = t.context()
        var out = _vec[dtype](1, ctx)

        @always_inline
        def identity[
            w: Int
        ](tile: SIMD[dtype, w], idx: RowCoord[1]) {} -> SIMD[dtype, w]:
            return tile

        reduce_all[monoid="max", gpu=True](
            t.tile(), out.tile(), identity, count, Optional(ctx)
        )
        return Float64(out.to_host()[0])
    else:
        var values = t.to_host()
        var best = Float64(values[0])
        for i in range(1, count):
            best = max(best, Float64(values[i]))
        return best


def _sum_of[
    dtype: DType, gpu: Bool
](mut t: Dynamic[dtype, 1]) raises -> Float64 where dtype.is_floating_point():
    """The sum of the entries, one scalar read back."""
    var count = t.size()
    comptime if gpu:
        var ctx = t.context()
        var out = _vec[dtype](1, ctx)

        @always_inline
        def identity[
            w: Int
        ](tile: SIMD[dtype, w], idx: RowCoord[1]) {} -> SIMD[dtype, w]:
            return tile

        reduce_all[monoid="sum", gpu=True](
            t.tile(), out.tile(), identity, count, Optional(ctx)
        )
        return Float64(out.to_host()[0])
    else:
        var values = t.to_host()
        var total = 0.0
        for i in range(count):
            total += Float64(values[i])
        return total


struct _Collocation[dtype: DType](Movable):
    """`collocation_fun`'s four outputs, and the midpoints they were taken
    at."""

    var col_res: Dynamic[Self.dtype, 2]
    var y_mid: Dynamic[Self.dtype, 2]
    var f: Dynamic[Self.dtype, 2]
    var f_mid: Dynamic[Self.dtype, 2]
    var x_mid: Dynamic[Self.dtype, 1]

    def __init__(
        out self,
        var col_res: Dynamic[Self.dtype, 2],
        var y_mid: Dynamic[Self.dtype, 2],
        var f: Dynamic[Self.dtype, 2],
        var f_mid: Dynamic[Self.dtype, 2],
        var x_mid: Dynamic[Self.dtype, 1],
    ):
        self.col_res = col_res^
        self.y_mid = y_mid^
        self.f = f^
        self.f_mid = f_mid^
        self.x_mid = x_mid^


def _collocation[
    dtype: DType,
    fun: def(
        Dynamic[dtype, 1], Dynamic[dtype, 2], DeviceContext
    ) raises thin -> Dynamic[dtype, 2],
    gpu: Bool,
](
    mut x: Dynamic[dtype, 1],
    mut h: Dynamic[dtype, 1],
    mut y: Dynamic[dtype, 2],
) raises -> _Collocation[dtype] where dtype.is_floating_point():
    """SciPy's `collocation_fun`: `f` at the nodes, `y_mid` and `f_mid` at
    the interval midpoints, and the Lobatto IIIA residuals."""
    var ctx = y.context()
    var n = y.dim[0]()
    var m = y.dim[1]()
    var f = fun(x, y, ctx)
    var x_mid = _vec[dtype](m - 1, ctx)
    var y_mid = _mat[dtype](n, m - 1, ctx)
    var xp = x.tile()
    var hp = h.tile()
    var yp = y.tile()
    var fp = f.tile()
    var xmp = x_mid.tile()
    var ymp = y_mid.tile()

    @always_inline
    def middle[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var xp, var hp, var yp, var fp, var xmp, var ymp, var n, var m
    }:
        var e = coord_to_index_list(coord)[0]
        var c = e // (m - 1)
        var i = e % (m - 1)
        var hi = hp.ptr[unsafe_offset=i]
        var a = c * m + i
        ymp.ptr[unsafe_offset=e] = Scalar[dtype](0.5) * (
            yp.ptr[unsafe_offset=a + 1] + yp.ptr[unsafe_offset=a]
        ) - Scalar[dtype](0.125) * hi * (
            fp.ptr[unsafe_offset=a + 1] - fp.ptr[unsafe_offset=a]
        )
        if c == 0:
            xmp.ptr[unsafe_offset=i] = (
                xp.ptr[unsafe_offset=i] + Scalar[dtype](0.5) * hi
            )

    elementwise[simd_width=1, target=_target[gpu]()](
        middle, Coord(n * (m - 1)), ctx
    )
    ctx.synchronize()
    var f_mid = fun(x_mid, y_mid, ctx)
    var col_res = _mat[dtype](n, m - 1, ctx)
    var fmp = f_mid.tile()
    var rp = col_res.tile()

    @always_inline
    def residual[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var hp, var yp, var fp, var fmp, var rp, var m}:
        var e = coord_to_index_list(coord)[0]
        var c = e // (m - 1)
        var i = e % (m - 1)
        var a = c * m + i
        rp.ptr[unsafe_offset=e] = (
            yp.ptr[unsafe_offset=a + 1]
            - yp.ptr[unsafe_offset=a]
            - hp.ptr[unsafe_offset=i]
            / Scalar[dtype](6.0)
            * (
                fp.ptr[unsafe_offset=a]
                + fp.ptr[unsafe_offset=a + 1]
                + Scalar[dtype](4.0) * fmp.ptr[unsafe_offset=e]
            )
        )

    elementwise[simd_width=1, target=_target[gpu]()](
        residual, Coord(n * (m - 1)), ctx
    )
    ctx.synchronize()
    return _Collocation[dtype](col_res^, y_mid^, f^, f_mid^, x_mid^)


def _column[
    dtype: DType, gpu: Bool
](mut y: Dynamic[dtype, 2], j: Int) raises -> Dynamic[
    dtype, 1
] where dtype.is_floating_point():
    """Column `j` of `y`, a length-`n` vector on its device."""
    var ctx = y.context()
    var n = y.dim[0]()
    var m = y.dim[1]()
    var out = _vec[dtype](n, ctx)
    var yp = y.tile()
    var op = out.tile()

    @always_inline
    def take[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var yp, var op, var m, var j}:
        var c = coord_to_index_list(coord)[0]
        op.ptr[unsafe_offset=c] = yp.ptr[unsafe_offset=c * m + j]

    elementwise[simd_width=1, target=_target[gpu]()](take, Coord(n), ctx)
    ctx.synchronize()
    return out^


def _residual_vector[
    dtype: DType, gpu: Bool
](
    mut col_res: Dynamic[dtype, 2], mut bc_res: Dynamic[dtype, 1]
) raises -> Dynamic[dtype, 1] where dtype.is_floating_point():
    """SciPy's `hstack((col_res.ravel(order='F'), bc_res))`: interval `i`'s
    `n` residuals at `i n`, the boundary residuals last."""
    var ctx = col_res.context()
    var n = col_res.dim[0]()
    var intervals = col_res.dim[1]()
    var out = _vec[dtype](n * (intervals + 1), ctx)
    var cp = col_res.tile()
    var bp = bc_res.tile()
    var op = out.tile()

    @always_inline
    def gather[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var cp, var bp, var op, var n, var intervals}:
        var e = coord_to_index_list(coord)[0]
        var i = e // n
        var c = e % n
        if i < intervals:
            op.ptr[unsafe_offset=e] = cp.ptr[unsafe_offset=c * intervals + i]
        else:
            op.ptr[unsafe_offset=e] = bp.ptr[unsafe_offset=c]

    elementwise[simd_width=1, target=_target[gpu]()](
        gather, Coord(n * (intervals + 1)), ctx
    )
    ctx.synchronize()
    return out^


def _fun_jac[
    dtype: DType,
    fun: def(
        Dynamic[dtype, 1], Dynamic[dtype, 2], DeviceContext
    ) raises thin -> Dynamic[dtype, 2],
    gpu: Bool,
](
    mut x: Dynamic[dtype, 1],
    mut y: Dynamic[dtype, 2],
    mut f0: Dynamic[dtype, 2],
) raises -> Dynamic[dtype, 1] where dtype.is_floating_point():
    """SciPy's `estimate_fun_jac`: `df/dy` at every node by forward
    differences, `n` vectorized calls of `fun`, stored `[j][r][c]` (node,
    component, variable) in one `m n n` vector."""
    var ctx = y.context()
    var n = y.dim[0]()
    var m = y.dim[1]()
    var jac = _vec[dtype](m * n * n, ctx)
    for c in range(n):
        var y_new = copy(y)
        var steps = _vec[dtype](m, ctx)
        var yp = y_new.tile()
        var sp = steps.tile()

        @always_inline
        def perturb[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var yp, var sp, var m, var c}:
            var j = coord_to_index_list(coord)[0]
            var v = yp.ptr[unsafe_offset=c * m + j]
            var moved = v + _sqrt_eps[dtype]() * (Scalar[dtype](1) + abs(v))
            yp.ptr[unsafe_offset=c * m + j] = moved
            sp.ptr[unsafe_offset=j] = moved - v

        elementwise[simd_width=1, target=_target[gpu]()](perturb, Coord(m), ctx)
        ctx.synchronize()
        var f_new = fun(x, y_new, ctx)
        var fp = f_new.tile()
        var f0p = f0.tile()
        var jp = jac.tile()

        @always_inline
        def difference[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var fp, var f0p, var jp, var sp, var n, var m, var c}:
            var e = coord_to_index_list(coord)[0]
            var r = e // m
            var j = e % m
            jp.ptr[unsafe_offset=(j * n + r) * n + c] = (
                fp.ptr[unsafe_offset=e] - f0p.ptr[unsafe_offset=e]
            ) / sp.ptr[unsafe_offset=j]

        elementwise[simd_width=1, target=_target[gpu]()](
            difference, Coord(n * m), ctx
        )
        ctx.synchronize()
        _ = y_new^
        _ = steps^
    return jac^


def _difference_column[
    dtype: DType, gpu: Bool
](
    mut jac: Dynamic[dtype, 1],
    moved: Dynamic[dtype, 1],
    base: Dynamic[dtype, 1],
    step: Dynamic[dtype, 1],
    n: Int,
    c: Int,
) raises where dtype.is_floating_point():
    """Column `c` of an `n x n` difference Jacobian: `(moved - base) /
    step`."""
    var ctx = jac.context()
    var op = moved.tile()
    var b0 = base.tile()
    var dp = jac.tile()
    var sp = step.tile()

    @always_inline
    def column[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var op, var b0, var dp, var sp, var n, var c}:
        var r = coord_to_index_list(coord)[0]
        dp.ptr[unsafe_offset=r * n + c] = (
            op.ptr[unsafe_offset=r] - b0.ptr[unsafe_offset=r]
        ) / sp.ptr[unsafe_offset=0]

    elementwise[simd_width=1, target=_target[gpu]()](column, Coord(n), ctx)
    ctx.synchronize()


def _bc_jac[
    dtype: DType,
    bc: def(
        Dynamic[dtype, 1], Dynamic[dtype, 1], DeviceContext
    ) raises thin -> Dynamic[dtype, 1],
    gpu: Bool,
](
    mut ya: Dynamic[dtype, 1],
    mut yb: Dynamic[dtype, 1],
    mut bc0: Dynamic[dtype, 1],
) raises -> Tuple[
    Dynamic[dtype, 1], Dynamic[dtype, 1]
] where dtype.is_floating_point():
    """SciPy's `estimate_bc_jac`: `d bc / d ya` and `d bc / d yb` by
    forward differences, each `n x n` row-major."""
    var ctx = ya.context()
    var n = ya.size()
    var dya = _vec[dtype](n * n, ctx)
    var dyb = _vec[dtype](n * n, ctx)
    for side in range(2):
        for c in range(n):
            var moved = copy(ya) if side == 0 else copy(yb)
            var step = _vec[dtype](1, ctx)
            var mp = moved.tile()
            var sp = step.tile()

            @always_inline
            def perturb[
                w: Int, alignment: Int = 1
            ](coord: Coord) {var mp, var sp, var c}:
                var v = mp.ptr[unsafe_offset=c]
                var up = v + _sqrt_eps[dtype]() * (Scalar[dtype](1) + abs(v))
                mp.ptr[unsafe_offset=c] = up
                sp.ptr[unsafe_offset=0] = up - v

            elementwise[simd_width=1, target=_target[gpu]()](
                perturb, Coord(1), ctx
            )
            ctx.synchronize()
            var out = bc(moved, yb, ctx) if side == 0 else bc(ya, moved, ctx)
            if side == 0:
                _difference_column[gpu=gpu](dya, out, bc0, step, n, c)
            else:
                _difference_column[gpu=gpu](dyb, out, bc0, step, n, c)
            _ = moved^
            _ = step^
            _ = out^
    return (dya^, dyb^)


def _assemble[
    dtype: DType, gpu: Bool
](
    h: Dynamic[dtype, 1],
    df: Dynamic[dtype, 1],
    df_mid: Dynamic[dtype, 1],
    dya: Dynamic[dtype, 1],
    dyb: Dynamic[dtype, 1],
    n: Int,
    m: Int,
) raises -> Dynamic[dtype, 2] where dtype.is_floating_point():
    """SciPy's `construct_global_jac`, dense: interval `i`'s rows hold
    `dPhi/dy_i = -I - h/6 (J_i + 2 J_mid) - h^2/12 J_mid J_i` and
    `dPhi/dy_{i+1} = I - h/6 (J_{i+1} + 2 J_mid) + h^2/12 J_mid J_{i+1}`,
    and the last `n` rows the boundary Jacobians at the two ends."""
    var ctx = h.context()
    var size = n * m
    var jac = _mat[dtype](size, size, ctx)
    var hp = h.tile()
    var dp = df.tile()
    var mp = df_mid.tile()
    var ap = dya.tile()
    var bp = dyb.tile()
    var jp = jac.tile()

    @always_inline
    def fill[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var hp, var dp, var mp, var ap, var bp, var jp, var n, var m, var size
    }:
        var e = coord_to_index_list(coord)[0]
        var row = e // size
        var col = e % size
        var blk = col // n
        var b = col % n
        var value = Scalar[dtype](0)
        if row < (m - 1) * n:
            var i = row // n
            var a = row % n
            if blk == i or blk == i + 1:
                var node = blk
                var hi = hp.ptr[unsafe_offset=i]
                var eye = Scalar[dtype](1) if a == b else Scalar[dtype](0)
                var jn = dp.ptr[unsafe_offset=(node * n + a) * n + b]
                var jm = mp.ptr[unsafe_offset=(i * n + a) * n + b]
                var prod = Scalar[dtype](0)
                for q in range(n):
                    prod += (
                        mp.ptr[unsafe_offset=(i * n + a) * n + q]
                        * dp.ptr[unsafe_offset=(node * n + q) * n + b]
                    )
                var lin = (
                    hi / Scalar[dtype](6.0) * (jn + Scalar[dtype](2.0) * jm)
                )
                var quad = hi * hi / Scalar[dtype](12.0) * prod
                if blk == i:
                    value = -eye - lin - quad
                else:
                    value = eye - lin + quad
        else:
            var a = row - (m - 1) * n
            if blk == 0:
                value = ap.ptr[unsafe_offset=a * n + b]
            if blk == m - 1:
                value += bp.ptr[unsafe_offset=a * n + b]
        jp.ptr[unsafe_offset=e] = value

    elementwise[simd_width=1, target=_target[gpu]()](
        fill, Coord(size * size), ctx
    )
    ctx.synchronize()
    return jac^


def _update[
    dtype: DType, gpu: Bool
](
    mut y: Dynamic[dtype, 2], mut step: Dynamic[dtype, 1], alpha: Float64
) raises -> Dynamic[dtype, 2] where dtype.is_floating_point():
    """`y - alpha step`, `step` laid out node-major as the Newton system
    orders its unknowns."""
    var ctx = y.context()
    var n = y.dim[0]()
    var m = y.dim[1]()
    var out = copy(y)
    var op = out.tile()
    var sp = step.tile()
    var a = Scalar[dtype](alpha)

    @always_inline
    def move[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var op, var sp, var a, var n, var m}:
        var e = coord_to_index_list(coord)[0]
        var c = e // m
        var j = e % m
        op.ptr[unsafe_offset=e] = (
            op.ptr[unsafe_offset=e] - a * sp.ptr[unsafe_offset=j * n + c]
        )

    elementwise[simd_width=1, target=_target[gpu]()](move, Coord(n * m), ctx)
    ctx.synchronize()
    return out^


def _dot_self[
    dtype: DType, gpu: Bool
](mut v: Dynamic[dtype, 1]) raises -> Float64 where dtype.is_floating_point():
    var ctx = v.context()
    var count = v.size()
    var sq = _vec[dtype](count, ctx)
    var vp = v.tile()
    var qp = sq.tile()

    @always_inline
    def square[w: Int, alignment: Int = 1](coord: Coord) {var vp, var qp}:
        var e = coord_to_index_list(coord)[0]
        var x = vp.ptr[unsafe_offset=e]
        qp.ptr[unsafe_offset=e] = x * x

    elementwise[simd_width=1, target=_target[gpu]()](square, Coord(count), ctx)
    ctx.synchronize()
    return _sum_of[gpu=gpu](sq)


def _converged[
    dtype: DType, gpu: Bool
](
    mut col_res: Dynamic[dtype, 2],
    mut f_mid: Dynamic[dtype, 2],
    mut h: Dynamic[dtype, 1],
    mut bc_res: Dynamic[dtype, 1],
    tol: Float64,
    bc_tol: Float64,
) raises -> Bool where dtype.is_floating_point():
    """SciPy's stopping test: every `|col_res| < tol_r (1 + |f_mid|)` with
    `tol_r = (2/3) h 0.05 tol`, and every `|bc_res| < bc_tol`."""
    var ctx = col_res.context()
    var n = col_res.dim[0]()
    var intervals = col_res.dim[1]()
    var excess = _vec[dtype](n * intervals, ctx)
    var cp = col_res.tile()
    var fp = f_mid.tile()
    var hp = h.tile()
    var ep = excess.tile()
    var scale = Scalar[dtype](2.0 / 3.0 * 5e-2 * tol)

    @always_inline
    def margin[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var cp, var fp, var hp, var ep, var scale, var intervals}:
        var e = coord_to_index_list(coord)[0]
        var i = e % intervals
        ep.ptr[unsafe_offset=e] = abs(cp.ptr[unsafe_offset=e]) - scale * hp.ptr[
            unsafe_offset=i
        ] * (Scalar[dtype](1) + abs(fp.ptr[unsafe_offset=e]))

    elementwise[simd_width=1, target=_target[gpu]()](
        margin, Coord(n * intervals), ctx
    )
    ctx.synchronize()
    if _max_of[gpu=gpu](excess) >= 0.0:
        return False
    return _max_abs[gpu=gpu](bc_res) < bc_tol


def _max_abs[
    dtype: DType, gpu: Bool
](mut v: Dynamic[dtype, 1]) raises -> Float64 where dtype.is_floating_point():
    var ctx = v.context()
    var count = v.size()
    var out = _vec[dtype](count, ctx)
    var vp = v.tile()
    var op = out.tile()

    @always_inline
    def magnitude[w: Int, alignment: Int = 1](coord: Coord) {var vp, var op}:
        var e = coord_to_index_list(coord)[0]
        op.ptr[unsafe_offset=e] = abs(vp.ptr[unsafe_offset=e])

    elementwise[simd_width=1, target=_target[gpu]()](
        magnitude, Coord(count), ctx
    )
    ctx.synchronize()
    return _max_of[gpu=gpu](out)


def _spline[
    dtype: DType, gpu: Bool
](
    mut x: Dynamic[dtype, 1],
    mut y: Dynamic[dtype, 2],
    mut f: Dynamic[dtype, 2],
    mut xq: Dynamic[dtype, 1],
    derivative: Bool,
) raises -> Dynamic[dtype, 2] where dtype.is_floating_point():
    """SciPy's `create_spline` evaluated at `xq`: the cubic Hermite
    interpolant through `y` with slopes `f`, or its derivative, `n x q`.
    Each query finds its interval by binary search on the device;
    queries outside the mesh extrapolate the end cubic, as SciPy's
    `PPoly(extrapolate=True)` does."""
    var ctx = y.context()
    var n = y.dim[0]()
    var m = y.dim[1]()
    var q = xq.size()
    var out = _mat[dtype](n, q, ctx)
    var xp = x.tile()
    var yp = y.tile()
    var fp = f.tile()
    var qp = xq.tile()
    var op = out.tile()

    @always_inline
    def evaluate[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var xp,
        var yp,
        var fp,
        var qp,
        var op,
        var n,
        var m,
        var q,
        var derivative,
    }:
        var e = coord_to_index_list(coord)[0]
        var c = e // q
        var k = e % q
        var t = qp.ptr[unsafe_offset=k]
        # The last `i` with `x[i] <= t`, clamped to the intervals.
        var lo = 0
        var hi = m - 2
        while lo < hi:
            var mid = (lo + hi + 1) // 2
            if xp.ptr[unsafe_offset=mid] <= t:
                lo = mid
            else:
                hi = mid - 1
        var i = lo
        var hh = xp.ptr[unsafe_offset=i + 1] - xp.ptr[unsafe_offset=i]
        var a = c * m + i
        var y0 = yp.ptr[unsafe_offset=a]
        var y1 = yp.ptr[unsafe_offset=a + 1]
        var d0 = fp.ptr[unsafe_offset=a]
        var d1 = fp.ptr[unsafe_offset=a + 1]
        var slope = (y1 - y0) / hh
        var tt = (d0 + d1 - Scalar[dtype](2) * slope) / hh
        var c0 = tt / hh
        var c1 = (slope - d0) / hh - tt
        var dx = t - xp.ptr[unsafe_offset=i]
        if derivative:
            op.ptr[unsafe_offset=e] = (
                Scalar[dtype](3) * c0 * dx + Scalar[dtype](2) * c1
            ) * dx + d0
        else:
            op.ptr[unsafe_offset=e] = ((c0 * dx + c1) * dx + d0) * dx + y0

    elementwise[simd_width=1, target=_target[gpu]()](
        evaluate, Coord(n * q), ctx
    )
    ctx.synchronize()
    return out^


def _rms_residuals[
    dtype: DType,
    fun: def(
        Dynamic[dtype, 1], Dynamic[dtype, 2], DeviceContext
    ) raises thin -> Dynamic[dtype, 2],
    gpu: Bool,
](
    mut x: Dynamic[dtype, 1],
    mut h: Dynamic[dtype, 1],
    mut y: Dynamic[dtype, 2],
    mut col: _Collocation[dtype],
) raises -> Dynamic[dtype, 1] where dtype.is_floating_point():
    """SciPy's `estimate_rms_residuals`: the relative residual of the spline
    at the midpoint (from `1.5 col_res / h`) and at the two interior
    5-point Lobatto nodes, `x_mid +- (h/2) sqrt(3/7)`, combined by that
    rule's weights."""
    var ctx = y.context()
    var n = y.dim[0]()
    var m = y.dim[1]()
    var intervals = m - 1
    var x1 = _vec[dtype](intervals, ctx)
    var x2 = _vec[dtype](intervals, ctx)
    var xm = col.x_mid.tile()
    var hp = h.tile()
    var p1 = x1.tile()
    var p2 = x2.tile()

    @always_inline
    def nodes[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xm, var hp, var p1, var p2}:
        var i = coord_to_index_list(coord)[0]
        var s = (
            Scalar[dtype](0.5)
            * hp.ptr[unsafe_offset=i]
            * Scalar[dtype](0.6546536707079771)
        )
        p1.ptr[unsafe_offset=i] = xm.ptr[unsafe_offset=i] + s
        p2.ptr[unsafe_offset=i] = xm.ptr[unsafe_offset=i] - s

    elementwise[simd_width=1, target=_target[gpu]()](
        nodes, Coord(intervals), ctx
    )
    ctx.synchronize()
    var y1 = _spline[gpu=gpu](x, y, col.f, x1, False)
    var y2 = _spline[gpu=gpu](x, y, col.f, x2, False)
    var d1 = _spline[gpu=gpu](x, y, col.f, x1, True)
    var d2 = _spline[gpu=gpu](x, y, col.f, x2, True)
    var f1 = fun(x1, y1, ctx)
    var f2 = fun(x2, y2, ctx)
    var out = _vec[dtype](intervals, ctx)
    var cp = col.col_res.tile()
    var fmp = col.f_mid.tile()
    var d1p = d1.tile()
    var d2p = d2.tile()
    var f1p = f1.tile()
    var f2p = f2.tile()
    var op = out.tile()

    @always_inline
    def combine[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var cp,
        var fmp,
        var d1p,
        var d2p,
        var f1p,
        var f2p,
        var hp,
        var op,
        var n,
        var intervals,
    }:
        var i = coord_to_index_list(coord)[0]
        var s_mid = Scalar[dtype](0)
        var s1 = Scalar[dtype](0)
        var s2 = Scalar[dtype](0)
        var one = Scalar[dtype](1)
        for c in range(n):
            var e = c * intervals + i
            var rm = (
                Scalar[dtype](1.5)
                * cp.ptr[unsafe_offset=e]
                / hp.ptr[unsafe_offset=i]
                / (one + abs(fmp.ptr[unsafe_offset=e]))
            )
            var r1 = (d1p.ptr[unsafe_offset=e] - f1p.ptr[unsafe_offset=e]) / (
                one + abs(f1p.ptr[unsafe_offset=e])
            )
            var r2 = (d2p.ptr[unsafe_offset=e] - f2p.ptr[unsafe_offset=e]) / (
                one + abs(f2p.ptr[unsafe_offset=e])
            )
            s_mid += rm * rm
            s1 += r1 * r1
            s2 += r2 * r2
        var total = Scalar[dtype](0.5) * (
            Scalar[dtype](32.0 / 45.0) * s_mid
            + Scalar[dtype](49.0 / 90.0) * (s1 + s2)
        )
        op.ptr[unsafe_offset=i] = sqrt(total)

    elementwise[simd_width=1, target=_target[gpu]()](
        combine, Coord(intervals), ctx
    )
    ctx.synchronize()
    _ = x1^
    _ = x2^
    return out^


def _refine[
    dtype: DType, gpu: Bool
](
    mut x: Dynamic[dtype, 1],
    mut y: Dynamic[dtype, 2],
    mut f: Dynamic[dtype, 2],
    mut rms: Dynamic[dtype, 1],
    tol: Float64,
    max_nodes: Int,
) raises -> Tuple[
    Int, Dynamic[dtype, 1], Dynamic[dtype, 2]
] where dtype.is_floating_point():
    """SciPy's mesh rule and `modify_mesh`: one node in the middle of an
    interval whose residual is in `(tol, 100 tol)`, two at its thirds when
    it is above, and `y` there from the spline. Returns the count added
    and the new mesh and `y`; the count alone when it would pass
    `max_nodes`, with the old mesh."""
    var ctx = y.context()
    var n = y.dim[0]()
    var m = y.dim[1]()
    var intervals = m - 1
    var offsets = Dynamic[DType.int64, 1](row_major(_dyn_shape[1](m + 1)), ctx)
    var rp = rms.tile()
    var op = offsets.tile()
    var lo = Scalar[dtype](tol)
    var hi = Scalar[dtype](100.0 * tol)

    # One lane: the inserts per interval and their running offsets, which
    # the scatter below needs in full; `offsets[m]` is the total.
    @always_inline
    def scan[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var rp, var op, var lo, var hi, var intervals}:
        var running = Int64(0)
        for i in range(intervals):
            op.ptr[unsafe_offset=i] = running
            var r = rp.ptr[unsafe_offset=i]
            if r >= hi:
                running += 2
            elif r > lo:
                running += 1
        op.ptr[unsafe_offset=intervals] = running

    elementwise[simd_width=1, target=_target[gpu]()](scan, Coord(1), ctx)
    ctx.synchronize()
    var added = Int(offsets.to_host()[intervals])
    if added == 0 or m + added > max_nodes:
        return (added, copy(x), copy(y))
    var m_new = m + added
    var x_new = _vec[dtype](m_new, ctx)
    var y_new = _mat[dtype](n, m_new, ctx)
    var xp = x.tile()
    var yp = y.tile()
    var fp = f.tile()
    var xnp = x_new.tile()
    var ynp = y_new.tile()

    @always_inline
    def scatter[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var xp,
        var yp,
        var fp,
        var op,
        var xnp,
        var ynp,
        var rp,
        var lo,
        var hi,
        var n,
        var m,
        var m_new,
    }:
        # Node `i` and the points inserted after it, for one component.
        var e = coord_to_index_list(coord)[0]
        var c = e // m
        var i = e % m
        var base = i + Int(op.ptr[unsafe_offset=i])
        ynp.ptr[unsafe_offset=c * m_new + base] = yp.ptr[
            unsafe_offset=c * m + i
        ]
        if c == 0:
            xnp.ptr[unsafe_offset=base] = xp.ptr[unsafe_offset=i]
        if i == m - 1:
            return
        var r = rp.ptr[unsafe_offset=i]
        var extra = 2 if r >= hi else (1 if r > lo else 0)
        var x0 = xp.ptr[unsafe_offset=i]
        var x1 = xp.ptr[unsafe_offset=i + 1]
        var hh = x1 - x0
        var a = c * m + i
        var y0 = yp.ptr[unsafe_offset=a]
        var y1 = yp.ptr[unsafe_offset=a + 1]
        var d0 = fp.ptr[unsafe_offset=a]
        var d1 = fp.ptr[unsafe_offset=a + 1]
        var slope = (y1 - y0) / hh
        var tt = (d0 + d1 - Scalar[dtype](2) * slope) / hh
        var c0 = tt / hh
        var c1 = (slope - d0) / hh - tt
        for k in range(extra):
            var t: Scalar[dtype]
            if extra == 1:
                t = Scalar[dtype](0.5) * (x0 + x1)
            elif k == 0:
                t = (Scalar[dtype](2) * x0 + x1) / Scalar[dtype](3)
            else:
                t = (x0 + Scalar[dtype](2) * x1) / Scalar[dtype](3)
            var dx = t - x0
            ynp.ptr[unsafe_offset=c * m_new + base + 1 + k] = (
                (c0 * dx + c1) * dx + d0
            ) * dx + y0
            if c == 0:
                xnp.ptr[unsafe_offset=base + 1 + k] = t

    elementwise[simd_width=1, target=_target[gpu]()](scatter, Coord(n * m), ctx)
    ctx.synchronize()
    _ = offsets^
    return (added, x_new^, y_new^)


def _diff[
    dtype: DType, gpu: Bool
](mut x: Dynamic[dtype, 1]) raises -> Dynamic[
    dtype, 1
] where dtype.is_floating_point():
    var ctx = x.context()
    var m = x.size()
    var h = _vec[dtype](m - 1, ctx)
    var xp = x.tile()
    var hp = h.tile()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var xp, var hp}:
        var i = coord_to_index_list(coord)[0]
        hp.ptr[unsafe_offset=i] = (
            xp.ptr[unsafe_offset=i + 1] - xp.ptr[unsafe_offset=i]
        )

    elementwise[simd_width=1, target=_target[gpu]()](step, Coord(m - 1), ctx)
    ctx.synchronize()
    return h^


struct BVPResult[dtype: DType, gpu: Bool = False](Movable):
    """What `solve_bvp` returns, SciPy's `BVPResult`: the final mesh, the
    solution and its derivative there, each interval's relative residual,
    and `sol(xq)` for the spline between the nodes.

    `status` is SciPy's: `0` converged, `1` would have passed `max_nodes`,
    `2` a singular Jacobian, `3` the iteration cap without meeting
    `bc_tol`.
    """

    var x: Dynamic[Self.dtype, 1]
    """The final mesh, `m` nodes."""
    var y: Dynamic[Self.dtype, 2]
    """The solution at the nodes, `n x m`."""
    var yp: Dynamic[Self.dtype, 2]
    """`fun(x, y)`, the derivative at the nodes, `n x m`."""
    var rms_residuals: Dynamic[Self.dtype, 1]
    """Each interval's relative RMS residual, `m - 1`."""
    var niter: Int
    """Outer iterations: Newton solves on successive meshes."""
    var status: Int
    """`0`: converged; `1`: `max_nodes`; `2`: singular; `3`: iterations."""
    var success: Bool
    """Whether `status == 0`."""

    def __init__(
        out self,
        var x: Dynamic[Self.dtype, 1],
        var y: Dynamic[Self.dtype, 2],
        var yp: Dynamic[Self.dtype, 2],
        var rms_residuals: Dynamic[Self.dtype, 1],
        niter: Int,
        status: Int,
    ):
        """Build from the parts.

        Args:
            x: The mesh.
            y: The solution at the mesh.
            yp: Its derivative there.
            rms_residuals: The per-interval residuals.
            niter: The outer iteration count.
            status: SciPy's status code.
        """
        self.x = x^
        self.y = y^
        self.yp = yp^
        self.rms_residuals = rms_residuals^
        self.niter = niter
        self.status = status
        self.success = status == 0

    def sol[
        T: TensorLike
    ](mut self, xq: T) raises -> Dynamic[Self.dtype, 2] where (
        Self.dtype.is_floating_point() and T.dtype == Self.dtype
    ):
        """The cubic Hermite spline through `y` with slopes `yp`, at `xq`:
        SciPy's `sol(xq)`, `n x len(xq)`.

        Parameters:
            T: The tensor type of `xq`, of the solution's `dtype`.

        Args:
            xq: The points to evaluate at; outside the mesh the end cubics
                extrapolate.

        Returns:
            The interpolated solution, one column per point.

        Raises:
            If a device operation fails.
        """
        var q = rebind_var[Dynamic[Self.dtype, 1]](
            _same_order(xq, row_major(_dyn_shape[1](xq.size())))
        )
        return _spline[gpu=Self.gpu](self.x, self.y, self.yp, q, False)


def solve_bvp[
    X: TensorLike,
    Y: TensorLike,
    fun: def(
        Dynamic[Y.dtype, 1], Dynamic[Y.dtype, 2], DeviceContext
    ) raises thin -> Dynamic[Y.dtype, 2],
    bc: def(
        Dynamic[Y.dtype, 1], Dynamic[Y.dtype, 1], DeviceContext
    ) raises thin -> Dynamic[Y.dtype, 1],
    gpu: Bool = False,
](
    x: X,
    y: Y,
    tol: Float64 = 1e-3,
    max_nodes: Int = 1000,
    bc_tol: Float64 = -1.0,
) raises -> BVPResult[Y.dtype, gpu] where (
    Y.dtype.is_floating_point()
    and Y.LayoutType.rank == 2
    and X.LayoutType.rank == 1
    and X.dtype == Y.dtype
):
    """Solve the two-point boundary-value problem `y' = fun(x, y)`,
    `bc(y(a), y(b)) = 0`. `scipy.integrate.solve_bvp(fun, bc, x, y, tol,
    max_nodes, bc_tol)`.

    SciPy's collocation algorithm, per this module's docstring, from the
    initial mesh `x` and guess `y`. `fun` is vectorized as SciPy's is:
    it takes the mesh (`m`) and the states (`n x m`, one column per node)
    and returns the derivatives in the same shape; `bc` takes the two end
    states and returns `n` residuals. Both receive the solver's device
    context and at `gpu=True` are called with device tensors, so a `fun`
    written in device operations keeps the whole solve on the device.

    Parameters:
        X: The tensor type of `x`, rank 1.
        Y: The tensor type of `y`, `n x m`.
        fun: The right-hand side, `fun(x, y, ctx) -> dy/dx`.
        bc: The boundary residuals, `bc(ya, yb, ctx) -> r`.
        gpu: Whether every phase runs on the inputs' device.

    Args:
        x: The initial mesh, strictly increasing, `m >= 2` nodes.
        y: The initial guess at the mesh, `n x m`.
        tol: The relative residual the solution must meet on every
            interval.
        max_nodes: The mesh size that ends the solve with `status = 1`.
        bc_tol: The boundary residual tolerance; `tol` when negative.

    Returns:
        A `BVPResult` with the mesh, the solution, its derivative, the
        residuals, `niter`, `status` and `sol`.

    Raises:
        If `x` and `y` disagree in size or `x` has fewer than two nodes,
        or `fun`, `bc` or a device operation raises.
    """
    comptime dtype = Y.dtype
    var btol = tol if bc_tol < 0 else bc_tol
    var xm = rebind_var[Dynamic[dtype, 1]](
        _same_order(x, row_major(_dyn_shape[1](x.size())))
    )
    var n = y.dim_at(0)
    var m = y.dim_at(1)
    if xm.size() != m or m < 2:
        raise Error(
            "solve_bvp: x has ", xm.size(), " nodes for y's ", m, " columns"
        )
    var ym = _same_order(y, row_major(_dyn_shape[2](n, m)))
    var status = 0
    var niter = 0
    var rms = _vec[dtype](1, xm.context())
    var yp = _mat[dtype](n, m, xm.context())
    while True:
        m = xm.size()
        var h = _diff[gpu=gpu](xm)
        # ---- Newton on this mesh (SciPy's `solve_newton`).
        var col = _collocation[fun=fun, gpu=gpu](xm, h, ym)
        var ya = _column[gpu=gpu](ym, 0)
        var yb = _column[gpu=gpu](ym, m - 1)
        var bc_res = bc(ya, yb, ym.context())
        var res = _residual_vector[gpu=gpu](col.col_res, bc_res)
        var singular = False
        var recompute = True
        var njev = 0
        var cost = 0.0
        var step = copy(res)
        # A placeholder the first iteration replaces: `recompute` starts
        # true, so no solve reads it.
        var factor = _lu_factor_runtime[dtype, gpu](
            _mat[dtype](1, 1, xm.context())
        )
        for _ in range(8):
            if recompute:
                var df = _fun_jac[fun=fun, gpu=gpu](xm, ym, col.f)
                var df_mid = _fun_jac[fun=fun, gpu=gpu](
                    col.x_mid, col.y_mid, col.f_mid
                )
                var dbc = _bc_jac[bc=bc, gpu=gpu](ya, yb, bc_res)
                var jac = _assemble[gpu=gpu](
                    h, df, df_mid, dbc[0], dbc[1], n, m
                )
                njev += 1
                factor = _lu_factor_runtime[dtype, gpu](jac)
                if factor.singular():
                    singular = True
                    break
                step = factor.solve(res)
                cost = _dot_self[gpu=gpu](step)
            var alpha = 1.0
            var y_new = copy(ym)
            var step_new = copy(step)
            var cost_new = cost
            for trial in range(5):
                y_new = _update[gpu=gpu](ym, step, alpha)
                col = _collocation[fun=fun, gpu=gpu](xm, h, y_new)
                ya = _column[gpu=gpu](y_new, 0)
                yb = _column[gpu=gpu](y_new, m - 1)
                bc_res = bc(ya, yb, ym.context())
                res = _residual_vector[gpu=gpu](col.col_res, bc_res)
                step_new = factor.solve(res)
                cost_new = _dot_self[gpu=gpu](step_new)
                if cost_new < (1.0 - 2.0 * alpha * 0.2) * cost:
                    break
                if trial < 4:
                    alpha *= 0.5
            ym = y_new^
            if njev == 4:
                break
            if _converged[gpu=gpu](
                col.col_res, col.f_mid, h, bc_res, tol, btol
            ):
                break
            if alpha == 1.0:
                step = step_new^
                cost = cost_new
                recompute = False
            else:
                recompute = True
        niter += 1
        # ---- Residuals on the converged mesh, and the mesh rule.
        col = _collocation[fun=fun, gpu=gpu](xm, h, ym)
        ya = _column[gpu=gpu](ym, 0)
        yb = _column[gpu=gpu](ym, m - 1)
        bc_res = bc(ya, yb, ym.context())
        var max_bc = _max_abs[gpu=gpu](bc_res)
        rms = _rms_residuals[fun=fun, gpu=gpu](xm, h, ym, col)
        yp = copy(col.f)
        if singular:
            status = 2
            break
        var refined = _refine[gpu=gpu](xm, ym, col.f, rms, tol, max_nodes)
        var added = refined[0]
        if m + added > max_nodes:
            status = 1
            break
        if added > 0:
            xm = copy(refined[1])
            ym = copy(refined[2])
        elif max_bc <= btol:
            status = 0
            break
        elif niter >= 10:
            status = 3
            break
    return BVPResult[dtype, gpu](xm^, ym^, yp^, rms^, niter, status)
