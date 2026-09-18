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

from layout.tile_layout import TensorLayout

from ..core.tensorlike import TensorLike, View, dim, is_row_major
from ..core.array import Static, Tensor


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
    T: TensorLike
](y: T, dx: Scalar[T.dtype] = 1) raises -> Scalar[T.dtype] where (
    T.dtype.is_floating_point() and T.LayoutType.rank == 1
):
    """The trapezoid rule over samples `y` spaced `dx` apart.
    `scipy.integrate.trapezoid(y, dx=dx)`."""
    var ys = y.to_host()
    var n = len(ys)
    if n < 2:
        return Scalar[T.dtype](0)
    var total = (ys[0] + ys[n - 1]) / 2
    for i in range(1, n - 1):
        total += ys[i]
    return total * dx


def trapezoid[
    T: TensorLike, XLayout: TensorLayout
](y: T, x: Tensor[T.dtype, XLayout]) raises -> Scalar[T.dtype] where (
    T.dtype.is_floating_point() and T.LayoutType.rank == 1 and XLayout.rank == 1
):
    """The trapezoid rule over samples `y` at the points `x`, which need
    not be evenly spaced. `scipy.integrate.trapezoid(y, x)`."""
    var ys = y.to_host()
    var n = len(ys)
    if n < 2:
        return Scalar[T.dtype](0)
    var h = _spacings(x, n)
    var total = Scalar[T.dtype](0)
    for i in range(n - 1):
        total += h[i] * (ys[i] + ys[i + 1]) / 2
    return total


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
    T: TensorLike
](y: T, dx: Scalar[T.dtype] = 1) raises -> Scalar[T.dtype] where (
    T.dtype.is_floating_point() and T.LayoutType.rank == 1
):
    """Composite Simpson's rule over samples `y` spaced `dx` apart.
    `scipy.integrate.simpson(y, dx=dx)`, even sample counts included."""
    var ys = y.to_host()
    var h = List[Scalar[T.dtype]](length=max(len(ys) - 1, 0), fill=dx)
    return _simpson_general(ys, h)


def simpson[
    T: TensorLike, XLayout: TensorLayout
](y: T, x: Tensor[T.dtype, XLayout]) raises -> Scalar[T.dtype] where (
    T.dtype.is_floating_point() and T.LayoutType.rank == 1 and XLayout.rank == 1
):
    """Composite Simpson's rule over samples `y` at the points `x`.
    `scipy.integrate.simpson(y, x)`."""
    var ys = y.to_host()
    if len(ys) < 2:
        return Scalar[T.dtype](0)
    return _simpson_general(ys, _spacings(x, len(ys)))


def cumulative_trapezoid[
    T: TensorLike,
    initial: Bool = False,
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
    var ys = y.to_host()
    comptime m = n if initial else n - 1
    var out = List[Scalar[T.dtype]](capacity=m)
    var running = Scalar[T.dtype](0)
    comptime if initial:
        out.append(running)
    for i in range(n - 1):
        running += dx * (ys[i] + ys[i + 1]) / 2
        out.append(running)
    return Static[T.dtype, m](y.context(), out^)


def cumulative_trapezoid[
    A: TensorLike,
    B: TensorLike,
    initial: Bool = False,
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
    var ys = y.to_host()
    var h = _spacings(x, n)
    comptime m = n if initial else n - 1
    var out = List[Scalar[A.dtype]](capacity=m)
    var running = Scalar[A.dtype](0)
    comptime if initial:
        out.append(running)
    for i in range(n - 1):
        running += h[i].cast[A.dtype]() * (ys[i] + ys[i + 1]) / 2
        out.append(running)
    return Static[A.dtype, m](y.context(), out^)
