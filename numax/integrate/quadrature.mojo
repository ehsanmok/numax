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

from ..core.array import Static, Tensor


def _spacings[
    dtype: DType, XLayout: TensorLayout
](x: Tensor[dtype, XLayout], n: Int) raises -> List[Scalar[dtype]]:
    """`x[i+1] - x[i]`, checking `x` has as many points as `y`."""
    if x.size() != n:
        raise Error(
            "integrate: x has ", x.size(), " points for ", n, " samples"
        )
    var xs = x.to_host()
    var h = List[Scalar[dtype]](capacity=n - 1)
    for i in range(n - 1):
        h.append(xs[i + 1] - xs[i])
    return h^


def trapezoid[
    dtype: DType, LayoutType: TensorLayout
](y: Tensor[dtype, LayoutType], dx: Scalar[dtype] = 1) raises -> Scalar[
    dtype
] where (dtype.is_floating_point() and LayoutType.rank == 1):
    """The trapezoid rule over samples `y` spaced `dx` apart.
    `scipy.integrate.trapezoid(y, dx=dx)`."""
    var ys = y.to_host()
    var n = len(ys)
    if n < 2:
        return Scalar[dtype](0)
    var total = (ys[0] + ys[n - 1]) / 2
    for i in range(1, n - 1):
        total += ys[i]
    return total * dx


def trapezoid[
    dtype: DType, LayoutType: TensorLayout, XLayout: TensorLayout
](y: Tensor[dtype, LayoutType], x: Tensor[dtype, XLayout]) raises -> Scalar[
    dtype
] where (
    dtype.is_floating_point() and LayoutType.rank == 1 and XLayout.rank == 1
):
    """The trapezoid rule over samples `y` at the points `x`, which need
    not be evenly spaced. `scipy.integrate.trapezoid(y, x)`."""
    var ys = y.to_host()
    var n = len(ys)
    if n < 2:
        return Scalar[dtype](0)
    var h = _spacings(x, n)
    var total = Scalar[dtype](0)
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
    dtype: DType, LayoutType: TensorLayout
](y: Tensor[dtype, LayoutType], dx: Scalar[dtype] = 1) raises -> Scalar[
    dtype
] where (dtype.is_floating_point() and LayoutType.rank == 1):
    """Composite Simpson's rule over samples `y` spaced `dx` apart.
    `scipy.integrate.simpson(y, dx=dx)`, even sample counts included."""
    var ys = y.to_host()
    var h = List[Scalar[dtype]](length=max(len(ys) - 1, 0), fill=dx)
    return _simpson_general(ys, h)


def simpson[
    dtype: DType, LayoutType: TensorLayout, XLayout: TensorLayout
](y: Tensor[dtype, LayoutType], x: Tensor[dtype, XLayout]) raises -> Scalar[
    dtype
] where (
    dtype.is_floating_point() and LayoutType.rank == 1 and XLayout.rank == 1
):
    """Composite Simpson's rule over samples `y` at the points `x`.
    `scipy.integrate.simpson(y, x)`."""
    var ys = y.to_host()
    if len(ys) < 2:
        return Scalar[dtype](0)
    return _simpson_general(ys, _spacings(x, len(ys)))


def cumulative_trapezoid[
    dtype: DType, n: Int, initial: Bool = False
](y: Static[dtype, n], dx: Scalar[dtype] = 1) raises -> Static[
    dtype, n if initial else n - 1
] where (dtype.is_floating_point() and n >= 2):
    """The running trapezoid integral of `y` at spacing `dx`.
    `scipy.integrate.cumulative_trapezoid(y, dx=dx)`.

    `n - 1` values, one per interval, as SciPy returns by default;
    `initial=True` is SciPy's `initial=0`, prepending a zero so the result
    is as long as `y` and `out[i]` is the integral up to `y[i]`. A
    compile-time flag rather than an argument because it changes the
    result's length, which is part of the type.
    """
    var ys = y.to_host()
    comptime m = n if initial else n - 1
    var out = List[Scalar[dtype]](capacity=m)
    var running = Scalar[dtype](0)
    comptime if initial:
        out.append(running)
    for i in range(n - 1):
        running += dx * (ys[i] + ys[i + 1]) / 2
        out.append(running)
    return Static[dtype, m](y.context(), out^)


def cumulative_trapezoid[
    dtype: DType, n: Int, XLayout: TensorLayout, initial: Bool = False
](y: Static[dtype, n], x: Tensor[dtype, XLayout]) raises -> Static[
    dtype, n if initial else n - 1
] where (dtype.is_floating_point() and n >= 2 and XLayout.rank == 1):
    """The running trapezoid integral of `y` at the points `x`.
    `scipy.integrate.cumulative_trapezoid(y, x)`."""
    var ys = y.to_host()
    var h = _spacings(x, n)
    comptime m = n if initial else n - 1
    var out = List[Scalar[dtype]](capacity=m)
    var running = Scalar[dtype](0)
    comptime if initial:
        out.append(running)
    for i in range(n - 1):
        running += h[i] * (ys[i] + ys[i + 1]) / 2
        out.append(running)
    return Static[dtype, m](y.context(), out^)
