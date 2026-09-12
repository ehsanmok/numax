"""How fast is the `Tensor` tier of `numax.interpolate` on this CPU?

Two things are timed, because they scale differently:

1. **Evaluation** -- `interp` (NumPy's 1-D linear, a vectorized
   `searchsorted` and a gather) and `CubicSpline.__call__` at `2^20`
   query points over `n = 1024` knots. Linear in the number of queries,
   so microseconds per call is the number and the `bench/scipy/
   interpolate.py` baseline prints the same columns from `numpy.interp` and
   `scipy.interpolate.CubicSpline`.
2. **Construction** -- `CubicSpline(x, y)` at `n = 1024` and `n = 4096`
   knots: the not-a-knot tridiagonal system through
   `numax.linalg.solve_banded`, `O(n)` on the host.

The error column is the interpolation error against the function the
knots sample (`x^2` for the linear rows, `sin` for the spline rows), which
is the quantity a user of these functions cares about and, at these knot
spacings, sits far above `float32` rounding, so a wrong gather would show
as a jump in it. `float32`, for the reason `bench_linalg.mojo` gives.

CPU only. Run with `pixi run bench-interpolate`.
"""

from max.gpu.host import DeviceContext
from std.benchmark import keep, run
from std.math import sin

from numax.core.array import Static
from numax.interpolate import CubicSpline, interp

comptime dtype = DType.float32
comptime warmup_iters = 2
comptime budget_secs = 1.0
comptime knot_span = 10.0


def _knots[n: Int](ctx: DeviceContext) raises -> Static[dtype, n]:
    """`n` knots on `[0, knot_span]`, uniform except for a deterministic
    jitter so nothing here is a uniform-grid shortcut."""
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        var h = (i * 2654435761 + 12345) % 16777216
        var jitter = 0.3 * (Float64(h) / 16777216.0 - 0.5)
        var t = Float64(i) if (i == 0 or i == n - 1) else Float64(i) + jitter
        values.append(Scalar[dtype](t * knot_span / Float64(n - 1)))
    return Static[dtype, n](ctx, values^)


def _samples[
    n: Int, f: def(Float64) thin -> Float64
](ctx: DeviceContext, mut knots: Static[dtype, n]) raises -> Static[dtype, n]:
    var host = knots.to_host()
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(Scalar[dtype](f(Float64(host[i]))))
    return Static[dtype, n](ctx, values^)


def _queries[m: Int](ctx: DeviceContext) raises -> Static[dtype, m]:
    """`m` points strictly inside the knot span, in hashed order so the
    `searchsorted` sees no monotone run to exploit."""
    var values = List[Scalar[dtype]](capacity=m)
    for i in range(m):
        var h = (i * 2654435761 + 97) % 16777216
        values.append(
            Scalar[dtype](0.01 + 0.98 * knot_span * Float64(h) / 16777216.0)
        )
    return Static[dtype, m](ctx, values^)


def _square(x: Float64) -> Float64:
    return x * x


def _sine(x: Float64) -> Float64:
    return sin(x)


def _row(name: String, n: Int, m: Int, ns: Float64, err: Float64):
    print(name, "\t", n, "\t", m, "\t", ns / 1e3, "\t", err)


def _worst[
    m: Int, f: def(Float64) thin -> Float64
](mut got: Static[dtype, m], mut at: Static[dtype, m]) raises -> Float64:
    var g = got.to_host()
    var a = at.to_host()
    var worst = Float64(0)
    for i in range(m):
        var diff = abs(Float64(g[i]) - f(Float64(a[i])))
        if diff > worst:
            worst = diff
    return worst


def bench_interp[
    n: Int, m: Int
](ctx: DeviceContext) raises where m > 0 and n > 0:
    var xp = _knots[n](ctx)
    var fp = _samples[n, _square](ctx, xp)
    var x = _queries[m](ctx)

    def work() raises {mut x, mut xp, mut fp}:
        var y = interp[dtype, m, n](x, xp, fp)
        keep(y.buffer.unsafe_ptr())

    var ns = (
        run(
            work, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9
    )
    var y = interp[dtype, m, n](x, xp, fp)
    _row("interp", n, m, ns, _worst[m, _square](y, x))


def bench_spline_build[
    n: Int
](ctx: DeviceContext) raises where n >= 2 and n >= 1:
    var xp = _knots[n](ctx)
    var fp = _samples[n, _sine](ctx, xp)

    def work() raises {mut xp, mut fp}:
        var spline = CubicSpline[dtype, n](xp, fp)
        keep(spline.spline.c.buffer.unsafe_ptr())

    var ns = (
        run(
            work, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9
    )
    _row("CubicSpline build", n, 0, ns, 0.0)


def bench_spline_eval[
    n: Int, m: Int
](ctx: DeviceContext) raises where n >= 2 and n >= 1 and m > 0:
    var xp = _knots[n](ctx)
    var fp = _samples[n, _sine](ctx, xp)
    var spline = CubicSpline[dtype, n](xp, fp)
    var x = _queries[m](ctx)

    def work() raises {mut spline, mut x}:
        var y = spline(x)
        keep(y.buffer.unsafe_ptr())

    var ns = (
        run(
            work, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9
    )
    var y = spline(x)
    _row("CubicSpline eval", n, m, ns, _worst[m, _sine](y, x))


def main() raises:
    var ctx = DeviceContext(api="cpu")
    print("dtype =", dtype, " target = cpu")

    print()
    print(
        "Interpolation (us is per call; error is against the sampled function)"
    )
    print("op\tn knots\tm queries\tus\tmax |error|")
    bench_interp[1024, 1 << 20](ctx)
    bench_spline_eval[1024, 1 << 20](ctx)
    bench_spline_build[1024](ctx)
    bench_spline_build[4096](ctx)
