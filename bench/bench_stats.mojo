"""How fast is the `Tensor` tier of `numax.stats` on this CPU?

Four rows, one per shape of work the surface has:

1. `norm.cdf` over a `2^24` tensor -- the most common SciPy call there is,
   and here one `map` through the tier-1 `erf`. Bandwidth-bound, so GB/s
   over one read and one write of `float32` is the number.
2. `histogram` at 64 bins over the same tensor -- a host-side count, as
   the `histograms` module says (MAX ships no scatter-add), against
   NumPy's.
3. `quantile` at `q = 0.5` -- a host sort of a copy, against NumPy's
   partition, and the row that says what a host-side path costs at this
   size.
4. `cov` and `corrcoef` of an `8 x 2^20` matrix -- host-side today, the
   `O(rows^2 n)` in a `Float64` loop; the row measures what that costs
   against the centering `map` and one `matmul` it could be.

The `bench/scipy/stats.py` baseline prints the same columns from
`scipy.stats.norm.cdf`, `numpy.histogram`, `numpy.quantile`, `numpy.cov`
and `numpy.corrcoef`. The error column is against a `float64` host
recomputation of the same quantity. `float32`, for the reason
`bench_linalg.mojo` gives.

CPU only. Run with `pixi run bench-stats`.
"""

from max.gpu.host import DeviceContext
from std.benchmark import keep, run
from std.math import erf, sqrt

from numax.core.array import Static
from numax.stats import corrcoef, cov, histogram, norm, quantile

comptime dtype = DType.float32
comptime warmup_iters = 2
comptime budget_secs = 1.0


def _entry(i: Int) -> Float64:
    """A hashed value on `[-3, 3)`."""
    var h = (i * 2654435761 + 12345) % 16777216
    return 6.0 * (Float64(h) / 16777216.0 - 0.5)


def _data[n: Int](ctx: DeviceContext) raises -> Static[dtype, n]:
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(Scalar[dtype](_entry(i)))
    return Static[dtype, n](ctx, values^)


def _matrix[
    rows: Int, n: Int
](ctx: DeviceContext) raises -> Static[dtype, rows, n]:
    var values = List[Scalar[dtype]](capacity=rows * n)
    for r in range(rows):
        for i in range(n):
            values.append(
                Scalar[dtype](_entry(i) + 0.5 * Float64(r) * _entry(i + r))
            )
    return Static[dtype, rows, n](ctx, values^)


def _row(name: String, n: Int, ns: Float64, err: Float64):
    print(name, "\t", n, "\t", ns / 1e3, "\t", err)


def _band_row(name: String, n: Int, ns: Float64, bytes: Int, err: Float64):
    print(
        name,
        "\t",
        n,
        "\t",
        ns / 1e3,
        "\t",
        Float64(bytes) / (ns / 1e9) / 1e9,
        "\t",
        err,
    )


def bench_norm_cdf[n: Int](ctx: DeviceContext) raises:
    var x = _data[n](ctx)

    def work() raises {mut x}:
        var p = norm.cdf(x, Scalar[dtype](0), Scalar[dtype](1))
        keep(p.buffer.unsafe_ptr())

    var ns = (
        run(
            work, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9
    )
    var p = norm.cdf(x, Scalar[dtype](0), Scalar[dtype](1)).to_host()
    var worst = Float64(0)
    for i in range(0, n, 997):
        var want = 0.5 * (1.0 + erf(_entry(i) / sqrt(2.0)))
        var diff = abs(Float64(p[i]) - want)
        if diff > worst:
            worst = diff
    _band_row("norm.cdf", n, ns, 2 * n * 4, worst)


def bench_histogram[n: Int](ctx: DeviceContext) raises:
    var x = _data[n](ctx)

    def work() raises {x}:
        var h = histogram[bins=64](x, -3.0, 3.0)
        keep(h.counts.buffer.unsafe_ptr())

    var ns = (
        run(
            work, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9
    )
    var h = histogram[bins=64](x, -3.0, 3.0)
    var counts = h.counts.to_host()
    var total = Float64(0)
    for b in range(64):
        total += Float64(counts[b])
    _row("histogram bins=64", n, ns, abs(total - Float64(n)))


def bench_quantile[n: Int](ctx: DeviceContext) raises:
    var x = _data[n](ctx)

    def work() raises {x}:
        keep(quantile(x, 0.5))

    var ns = (
        run(
            work, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9
    )
    # The data is a hashed uniform on [-3, 3), so its median is near 0;
    # the reported error is against 0 and is a statement about the data,
    # not the selection.
    _row("quantile q=0.5", n, ns, abs(Float64(quantile(x, 0.5))))


def bench_cov[
    rows: Int, n: Int
](ctx: DeviceContext) raises where rows > 0 and n > 1:
    var m = _matrix[rows, n](ctx)

    def cov_work() raises {mut m}:
        var c = cov[dtype, rows, n](m)
        keep(c.buffer.unsafe_ptr())

    var cov_ns = (
        run(
            cov_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9
    )

    def corr_work() raises {mut m}:
        var c = corrcoef[dtype, rows, n](m)
        keep(c.buffer.unsafe_ptr())

    var corr_ns = (
        run(
            corr_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9
    )

    # `float64` recomputation of `cov[0, 1]`.
    var host = m.to_host()
    var mean0 = Float64(0)
    var mean1 = Float64(0)
    for i in range(n):
        mean0 += Float64(host[i])
        mean1 += Float64(host[n + i])
    mean0 /= Float64(n)
    mean1 /= Float64(n)
    var acc = Float64(0)
    for i in range(n):
        acc += (Float64(host[i]) - mean0) * (Float64(host[n + i]) - mean1)
    var want = acc / Float64(n - 1)
    var c = cov[dtype, rows, n](m).to_host()
    _row(
        String("cov ") + String(rows) + " x n",
        n,
        cov_ns,
        abs(Float64(c[1]) - want),
    )
    var r = corrcoef[dtype, rows, n](m).to_host()
    _row(
        String("corrcoef ") + String(rows) + " x n",
        n,
        corr_ns,
        abs(Float64(r[0]) - 1.0),
    )


def main() raises:
    var ctx = DeviceContext(api="cpu")
    print("dtype =", dtype, " target = cpu")

    print()
    print("Elementwise (us is per call, GB/s over one read and one write)")
    print("op\tn\tus\tGB/s\tmax |error|")
    bench_norm_cdf[1 << 24](ctx)

    print()
    print("Counting, selection, correlation (us is per call)")
    print("op\tn\tus\terror")
    bench_histogram[1 << 24](ctx)
    bench_quantile[1 << 24](ctx)
    bench_cov[8, 1 << 20](ctx)
