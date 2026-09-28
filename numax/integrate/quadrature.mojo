"""Integration of sampled values: `scipy.integrate`'s `trapezoid`, `simpson`
and `cumulative_trapezoid`, over `Tensor`.

**This module is tier 2**: `Plain`-only, host-side. Each rule walks a host
copy of the samples, on the same terms as `numax.core.elementwise` -- the
arithmetic is a fixed weighted sum, but the walk is not launched on a
device. `ponytail:` a device path would be one reduction through the
`rowwise` scaffolder `numax.stats.sum` already uses, and it waits on
either an `integrate -> stats` edge or that reduction moving into `core`.
The `Array` tier, `numax.integrate.array`, is the one that runs inside a
kernel.

These take *samples* -- a tensor `y` on a uniform grid of spacing `dx`, or
at the points `x` -- which is what `scipy.integrate.trapezoid(y, x=None,
dx=1.0)` takes. `numax.integrate.array.trapezoid[f](a, b)` samples a
function itself; that is the same rule under a different input and is
recorded in `docs/parity.md` as the divergent spelling.

`simpson` follows SciPy 1.11+: composite Simpson over pairs of intervals,
and when the sample count is even the last interval takes Cartwright's
three-point correction rather than a trapezoid, so the rule stays exact for
quadratics however many samples there are. Non-uniform `x` uses the general
three-point formula per pair. Both are checked against `scipy.integrate`'s
digits in the tests, not derived here from scratch.

Rank 1 only. A higher-rank `axis=` form is the `outer`/`length`/`inner`
split `numax.stats` uses and is a follow-up rather than a decision.
"""

from algorithm.rowwise_types import RowCoord
from layout import Coord, coord_to_index_list
from layout.tile_layout import TensorLayout, row_major
from max.algorithm.functional import elementwise

from ..core.tensorlike import TensorLike, TensorView, dim, is_row_major
from ..core.tensor import Dynamic, Static, Tensor, _dyn_shape, _scan_device
from ..core._drive import _check_device, _notice, _require_contiguous
from ..core.rowwise import reduce_all


def _spacings[
    T: TensorLike,
](x: T, n: Int) raises -> List[Scalar[T.dtype]]:
    """`x[i+1] - x[i]`, checking `x` has as many points as `y`."""
    if x.size() != n:
        raise Error(
            "integrate: x has ", x.size(), " points for ", n, " samples"
        )
    var xs = x.to_host()
    var h = List[Scalar[T.dtype]](capacity=n - 1)
    for i in range(n - 1):
        h.append(xs[i + 1] - xs[i])
    return h^


def trapezoid[
    T: TensorLike, gpu: Bool = False
](y: T, dx: Scalar[T.dtype] = 1) raises -> Scalar[T.dtype] where (
    T.dtype.is_floating_point() and T.LayoutType.rank == 1
):
    """The trapezoid rule over samples `y` spaced `dx` apart.
    `scipy.integrate.trapezoid(y, dx=dx)`."""
    if y.size() >= 3 and _check_device[T, gpu](y):
        comptime if gpu:
            return _sampled_device["trapezoid", False](y, y, dx)
    elif y.size() >= 3:
        _notice[gpu]("trapezoid")
    var ys = y.to_host()
    var n = len(ys)
    if n < 2:
        return Scalar[T.dtype](0)
    var total = (ys[0] + ys[n - 1]) / 2
    for i in range(1, n - 1):
        total += ys[i]
    return total * dx


def trapezoid[
    T: TensorLike, XLayout: TensorLayout, gpu: Bool = False
](y: T, x: Tensor[T.dtype, XLayout]) raises -> Scalar[T.dtype] where (
    T.dtype.is_floating_point() and T.LayoutType.rank == 1 and XLayout.rank == 1
):
    """The trapezoid rule over samples `y` at the points `x`, which need
    not be evenly spaced. `scipy.integrate.trapezoid(y, x)`."""
    if y.size() >= 3 and _check_device[T, gpu](y):
        comptime if gpu:
            return _sampled_device["trapezoid", True](y, x, 0)
    elif y.size() >= 3:
        _notice[gpu]("trapezoid")
    var ys = y.to_host()
    var n = len(ys)
    if n < 2:
        return Scalar[T.dtype](0)
    var h = _spacings(x, n)
    var total = Scalar[T.dtype](0)
    for i in range(n - 1):
        total += h[i] * (ys[i] + ys[i + 1]) / 2
    return total


def _sampled_device[
    kind: StaticString, has_x: Bool, T: TensorLike, X: TensorLike
](y: T, x: X, dx: Scalar[T.dtype]) raises -> Scalar[T.dtype] where (
    X.dtype == T.dtype
):
    """`trapezoid` or `simpson` over samples on the device, `n >= 3`.

    One launch writes each term -- an interval's trapezoid, or a pair of
    intervals' three-point Simpson rule plus, for an even sample count,
    Cartwright's correction on the last interval, exactly the host rule --
    and MAX's `ReduceSum` adds them, so one scalar crosses back. The widths
    are `x[i + 1] - x[i]` when `has_x`, else `dx`.
    """
    comptime dtype = T.dtype
    var ctx = y.context()
    var n = y.size()
    _require_contiguous(y)
    comptime if has_x:
        _require_contiguous(x)
        if x.size() != n:
            raise Error(
                "integrate: x has ", x.size(), " points for ", n, " samples"
            )
    var pairs = (n - 1 if n % 2 == 1 else n - 2) // 2
    var count = n - 1
    comptime if kind == "simpson":
        count = pairs + (1 if n % 2 == 0 else 0)
    var terms = Dynamic[dtype, 1]._uninitialized(
        ctx, row_major(_dyn_shape[1](count))
    )
    var yp = y.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var xp = (
        x.tile()
        .ptr.unsafe_bitcast[Scalar[dtype]]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    var tp = terms.tile()

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var yp, var xp, var tp, var dx, var n, var pairs}:
        var q = coord_to_index_list(coord)[0]

        @always_inline
        def h(i: Int) {var xp, var dx} -> Scalar[dtype]:
            comptime if has_x:
                return xp[unsafe_offset=i + 1] - xp[unsafe_offset=i]
            else:
                return dx

        var term: Scalar[dtype]
        comptime if kind == "trapezoid":
            term = h(q) * (yp[unsafe_offset=q] + yp[unsafe_offset=q + 1]) / 2
        else:
            if q < pairs:
                var i = 2 * q
                var h0 = h(i)
                var h1 = h(i + 1)
                var hsum = h0 + h1
                var ratio = h0 / h1
                term = (hsum / 6) * (
                    yp[unsafe_offset=i] * (2 - 1 / ratio)
                    + yp[unsafe_offset=i + 1] * (hsum * hsum / (h0 * h1))
                    + yp[unsafe_offset=i + 2] * (2 - ratio)
                )
            else:
                var h0 = h(n - 3)
                var h1 = h(n - 2)
                var alpha = (2 * h1 * h1 + 3 * h0 * h1) / (6 * (h0 + h1))
                var beta = (h1 * h1 + 3 * h0 * h1) / (6 * h0)
                var eta = h1 * h1 * h1 / (6 * h0 * (h0 + h1))
                term = (
                    alpha * yp[unsafe_offset=n - 1]
                    + beta * yp[unsafe_offset=n - 2]
                    - eta * yp[unsafe_offset=n - 3]
                )
        tp.store[1](coord, term)

    elementwise[simd_width=1, target="gpu"](body, Coord(count), ctx)
    var total = Static[dtype, 1](ctx)

    @always_inline
    def identity[
        w: Int
    ](tile: SIMD[dtype, w], idx: RowCoord[1]) {} -> SIMD[dtype, w]:
        return tile

    reduce_all[monoid="sum", gpu=True](
        terms.tile(), total.tile(), identity, count, Optional(ctx)
    )
    return total.to_host()[0]


def _cumulative_device[
    has_x: Bool, initial: Bool, m: Int, T: TensorLike, X: TensorLike
](y: T, x: X, dx: Scalar[T.dtype]) raises -> Static[T.dtype, m] where (
    X.dtype == T.dtype
):
    """`cumulative_trapezoid` on the device: one launch writes each
    interval's trapezoid (after a leading `0` when `initial`), and the
    device scan accumulates them."""
    comptime dtype = T.dtype
    var ctx = y.context()
    _require_contiguous(y)
    comptime if has_x:
        _require_contiguous(x)
    var terms = Static[dtype, m]._uninitialized(ctx)
    var yp = y.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var xp = (
        x.tile()
        .ptr.unsafe_bitcast[Scalar[dtype]]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    var tp = terms.tile()

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var yp, var xp, var tp, var dx}:
        var q = coord_to_index_list(coord)[0]
        var term = Scalar[dtype](0)
        var i = q - 1 if initial else q
        if i >= 0:
            var h = dx
            comptime if has_x:
                h = xp[unsafe_offset=i + 1] - xp[unsafe_offset=i]
            term = h * (yp[unsafe_offset=i] + yp[unsafe_offset=i + 1]) / 2
        tp.store[1](coord, term)

    elementwise[simd_width=1, target="gpu"](body, Coord(m), ctx)
    return _scan_device["sum"](terms, Static[dtype, m]._static_layout(), m, 1)


def _simpson_general[
    dtype: DType
](ys: List[Scalar[dtype]], h: List[Scalar[dtype]]) -> Scalar[dtype]:
    """Composite Simpson over `ys` with interval widths `h`, SciPy 1.11+'s
    rule: pairs of intervals by the general three-point formula, and an
    even sample count closed by Cartwright's correction on the last
    interval."""
    var n = len(ys)
    if n < 2:
        return Scalar[dtype](0)
    if n == 2:
        return h[0] * (ys[0] + ys[1]) / 2

    var pairs_end = n - 1 if n % 2 == 1 else n - 2
    var total = Scalar[dtype](0)
    var i = 0
    while i < pairs_end:
        var h0 = h[i]
        var h1 = h[i + 1]
        var hsum = h0 + h1
        var hprod = h0 * h1
        var h0divh1 = h0 / h1
        total += (hsum / 6) * (
            ys[i] * (2 - 1 / h0divh1)
            + ys[i + 1] * (hsum * hsum / hprod)
            + ys[i + 2] * (2 - h0divh1)
        )
        i += 2

    if n % 2 == 0:
        var h0 = h[n - 3]
        var h1 = h[n - 2]
        var alpha = (2 * h1 * h1 + 3 * h0 * h1) / (6 * (h0 + h1))
        var beta = (h1 * h1 + 3 * h0 * h1) / (6 * h0)
        var eta = h1 * h1 * h1 / (6 * h0 * (h0 + h1))
        total += alpha * ys[n - 1] + beta * ys[n - 2] - eta * ys[n - 3]
    return total


def simpson[
    T: TensorLike, gpu: Bool = False
](y: T, dx: Scalar[T.dtype] = 1) raises -> Scalar[T.dtype] where (
    T.dtype.is_floating_point() and T.LayoutType.rank == 1
):
    """Composite Simpson's rule over samples `y` spaced `dx` apart.
    `scipy.integrate.simpson(y, dx=dx)`, even sample counts included."""
    if y.size() >= 3 and _check_device[T, gpu](y):
        comptime if gpu:
            return _sampled_device["simpson", False](y, y, dx)
    elif y.size() >= 3:
        _notice[gpu]("simpson")
    var ys = y.to_host()
    var h = List[Scalar[T.dtype]](length=max(len(ys) - 1, 0), fill=dx)
    return _simpson_general(ys, h)


def simpson[
    T: TensorLike, XLayout: TensorLayout, gpu: Bool = False
](y: T, x: Tensor[T.dtype, XLayout]) raises -> Scalar[T.dtype] where (
    T.dtype.is_floating_point() and T.LayoutType.rank == 1 and XLayout.rank == 1
):
    """Composite Simpson's rule over samples `y` at the points `x`.
    `scipy.integrate.simpson(y, x)`."""
    if y.size() >= 3 and _check_device[T, gpu](y):
        comptime if gpu:
            return _sampled_device["simpson", True](y, x, 0)
    elif y.size() >= 3:
        _notice[gpu]("simpson")
    var ys = y.to_host()
    if len(ys) < 2:
        return Scalar[T.dtype](0)
    return _simpson_general(ys, _spacings(x, len(ys)))


def cumulative_trapezoid[
    T: TensorLike,
    initial: Bool = False,
    gpu: Bool = False,
](y: T, dx: Scalar[T.dtype] = 1) raises -> Static[
    T.dtype, dim[T, 0] if initial else dim[T, 0] - 1
] where (
    (T.dtype.is_floating_point() and dim[T, 0] >= 2)
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
):
    """The running trapezoid integral of `y` at spacing `dx`.
    `scipy.integrate.cumulative_trapezoid(y, dx=dx)`.

    `n - 1` values, one per interval, as SciPy returns by default;
    `initial=True` is SciPy's `initial=0`, prepending a zero so the result
    is as long as `y` and `out[i]` is the integral up to `y[i]`. A
    compile-time flag rather than an argument because it changes the
    result's length, which is part of the type.
    """
    comptime n = dim[T, 0]
    comptime m = n if initial else n - 1
    if _check_device[T, gpu](y):
        comptime if gpu:
            return _cumulative_device[False, initial, m](y, y, dx)
    else:
        _notice[gpu]("cumulative_trapezoid")
    var ys = y.to_host()
    var out = List[Scalar[T.dtype]](capacity=m)
    var running = Scalar[T.dtype](0)
    comptime if initial:
        out.append(running)
    for i in range(n - 1):
        running += dx * (ys[i] + ys[i + 1]) / 2
        out.append(running)
    return Static[T.dtype, m](out^, y.context())


def cumulative_trapezoid[
    A: TensorLike,
    B: TensorLike,
    initial: Bool = False,
    gpu: Bool = False,
](y: A, x: B) raises -> Static[
    A.dtype, dim[A, 0] if initial else dim[A, 0] - 1
] where (
    (A.dtype.is_floating_point() and dim[A, 0] >= 2 and B.LayoutType.rank == 1)
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
):
    """The running trapezoid integral of `y` at the points `x`.
    `scipy.integrate.cumulative_trapezoid(y, x)`."""
    comptime n = dim[A, 0]
    comptime m = n if initial else n - 1
    if _check_device[A, gpu](y):
        comptime if gpu:
            if x.size() != n:
                raise Error(
                    "integrate: x has ", x.size(), " points for ", n, " samples"
                )
            return _cumulative_device[True, initial, m](y, x, 0)
    else:
        _notice[gpu]("cumulative_trapezoid")
    var ys = y.to_host()
    var h = _spacings(x, n)
    var out = List[Scalar[A.dtype]](capacity=m)
    var running = Scalar[A.dtype](0)
    comptime if initial:
        out.append(running)
    for i in range(n - 1):
        running += h[i].cast[A.dtype]() * (ys[i] + ys[i + 1]) / 2
        out.append(running)
    return Static[A.dtype, m](out^, y.context())
