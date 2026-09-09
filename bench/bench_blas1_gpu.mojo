"""The BLAS-1 half of the device linalg sweep, in its own file.

Split out of `bench_linalg_gpu.mojo` for a build reason rather than a
methodological one: the two halves together instantiate more GPU kernels
than Apple's Metal compiler will put in one metallib, and the combined
file failed with "Metal Compiler failed to compile metallib" before a
single measurement ran. Either half builds and runs alone -- established
by bisection, at every size in the sweep and for every factorization --
so the split is the smallest change that makes the device numbers
reachable on Metal at all. CUDA built either way; no measurement changes,
and the two files print the tables they always did.

Same operations, sizes, generators and traffic counts as
`bench_linalg.mojo`'s BLAS-1 table, so the device column stays comparable
to the CPU one and to `../scipy/linalg.py` and `../torch/linalg.py`. Per
`CLAUDE.md` the CPU and GPU numbers never share a table.

These four are bandwidth-bound, so the comparable number is GB/s over the
traffic each operation must move, not GFLOP/s. `dot`, `nrm2` and `asum`
drive MAX's `ReduceSum` monoid under the `rowwise` scaffolder; `axpy` is
one `max.algorithm.elementwise` pass.

Needs a real device -- CUDA or Metal, whichever `DeviceContext` finds. Not
part of CI, which has no GPU runners. Run with `pixi run bench-blas1-gpu`.
"""

from max.gpu.host import DeviceContext
from std.benchmark import keep, run
from std.math import sqrt

from numax.core.array import Static
from numax.linalg import asum, axpy, dot, nrm2

comptime dtype = DType.float32
comptime warmup_iters = 2

comptime budget_secs = 1.0
"""Seconds `std.benchmark.run` may spend on one measurement.

`max_batch_size=1` goes with it, so each sample is one
launch-through-completion round trip rather than a batch amortized across
one synchronize -- the same shape the factorization file and
`../torch/linalg.py` use.
"""


def _ramp[n: Int](ctx: DeviceContext, salt: Int) raises -> Static[dtype, n]:
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(Scalar[dtype](Float64((i * 37 + salt * 11) % 17) - 8.0))
    return Static[dtype, n](ctx, values^)


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


def bench_blas1[n: Int](ctx: DeviceContext) raises:
    var x = _ramp[n](ctx, 1)
    var y = _ramp[n](ctx, 2)

    # `keep` matters most for the three that return a scalar the caller
    # drops, which is the shape an optimizer is likeliest to fold away.
    def dot_work() raises {mut x, mut y}:
        keep(dot[dtype, n, True](x, y))

    def nrm2_work() raises {mut x}:
        keep(nrm2[dtype, n, True](x))

    def asum_work() raises {mut x}:
        keep(asum[dtype, n, True](x))

    def axpy_work() raises {mut x, mut y, imm ctx}:
        var s = axpy[dtype, n, True](Scalar[dtype](2.5), x, y)
        keep(s.buffer.unsafe_ptr())
        ctx.synchronize()

    var dot_ns = (
        run(
            dot_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
            max_batch_size=1,
        ).mean()
        * 1e9
    )
    var nrm2_ns = (
        run(
            nrm2_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
            max_batch_size=1,
        ).mean()
        * 1e9
    )
    var asum_ns = (
        run(
            asum_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
            max_batch_size=1,
        ).mean()
        * 1e9
    )
    var axpy_ns = (
        run(
            axpy_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
            max_batch_size=1,
        ).mean()
        * 1e9
    )

    # The three reductions return a `Scalar`, so they synchronize on their
    # own way out; `axpy` returns a tensor and gets an explicit one above.
    var want_dot = Float64(0)
    var want_dot_abs = Float64(0)
    var want_sq = Float64(0)
    var want_abs = Float64(0)
    for i in range(n):
        var xi = Float64((i * 37 + 11) % 17) - 8.0
        var yi = Float64((i * 37 + 22) % 17) - 8.0
        want_dot += xi * yi
        want_dot_abs += abs(xi * yi)
        want_sq += xi * xi
        want_abs += abs(xi)
    var want_nrm2 = sqrt(want_sq)

    var got_dot = Float64(dot[dtype, n, True](x, y))
    var got_nrm2 = Float64(nrm2[dtype, n, True](x))
    var got_asum = Float64(asum[dtype, n, True](x))
    var summed = axpy[dtype, n, True](Scalar[dtype](2.5), x, y).to_host()
    var worst_axpy = Float64(0)
    for i in range(n):
        var xi = Float64((i * 37 + 11) % 17) - 8.0
        var yi = Float64((i * 37 + 22) % 17) - 8.0
        var diff = abs(Float64(summed[i]) - (2.5 * xi + yi))
        if diff > worst_axpy:
            worst_axpy = diff

    # The three reductions report error relative to the quantity that
    # bounds a floating-point sum -- `sum |x_i y_i|` for `dot`, the value
    # itself for the two positive sums -- rather than to the result, which
    # for a cancelling dot product would divide by nearly nothing.
    _band_row(
        "dot", n, dot_ns, 2 * n * 4, abs(got_dot - want_dot) / want_dot_abs
    )
    _band_row("nrm2", n, nrm2_ns, n * 4, abs(got_nrm2 - want_nrm2) / want_nrm2)
    _band_row("asum", n, asum_ns, n * 4, abs(got_asum - want_abs) / want_abs)
    _band_row("axpy", n, axpy_ns, 3 * n * 4, worst_axpy)


def main() raises:
    var ctx = DeviceContext()
    print("dtype =", dtype, " GPU API:", ctx.api())

    print()
    print("BLAS-1 (us is per call, GB/s over the traffic the op must move)")
    print("op\tn\tus\tGB/s\terror")
    bench_blas1[1 << 16](ctx)
    bench_blas1[1 << 20](ctx)
    bench_blas1[1 << 24](ctx)
    bench_blas1[1 << 26](ctx)
