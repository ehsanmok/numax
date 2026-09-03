"""How much does `numax.core.tensor.map[gpu=True]` buy over the CPU path, for the
same `gaussian(x) = exp(-x^2)` kernel `bench_tensor_map.mojo` benchmarks on
CPU alone -- across a sweep of sizes, not just one.

A single size hides the real story here: launching a GPU kernel has a fixed
dispatch/synchronize cost that a cheap elementwise kernel like `gaussian`
has to pay back out of raw throughput, so which side wins depends on `n`.
This sweeps every size in `sizes` below and reports where the crossover
actually falls on this machine, rather than asserting one from general
principles.

Two GPU columns, because where the synchronize goes changes the answer by
more than 2x at some sizes and there is no single honest number:

- **per-call** wraps `enqueue_function` + `synchronize` together, so each
  iteration is one full launch-through-completion round trip -- the latency
  a caller sees for one call.
- **amortized** enqueues all `timed_iters` launches and synchronizes once,
  so dispatches pipeline and the host round trip is paid once for the batch
  -- steady-state throughput for back-to-back work.

This matters for cross-language comparison specifically: `bench/torch/
gaussian.py` originally reported amortized only while this file reported
per-call only, and `bench/README.md` put the two in one table as though
they were the same measurement. Both files now report both shapes.

The CPU and GPU paths are also timed in *separate* loops rather than
accumulating both inside one. Alternating them measured ~16% lower for the
GPU at 67M (quantified in `bench_gpu_roofline.mojo`): a CPU `map` pass at
that size is ~28ms of bandwidth-saturating work on the same unified memory
the GPU streams from, so interleaving has each path timing the other's
interference. That single change raised the reported 67M GPU figure from
~9,400 to ~13,000 M elem/s without touching a line of `numax`.

Needs a real GPU -- Metal or CUDA, whichever `DeviceContext` finds. Not
part of CI, which has no GPU runners. Run with `pixi run bench-gpu`.
"""

from layout import TileTensor
from layout.tile_layout import row_major
from max.gpu.host import DeviceContext
from std.sys.info import simd_width_of
from std.time import perf_counter_ns

from numax import Plain, gaussian
from numax.core.tensor import map

comptime dtype = DType.float32
comptime cpu_width = simd_width_of[dtype]()
comptime warmup_iters = 3
comptime timed_iters = 10
comptime block_size = 256
comptime sizes = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26]


def gaussian_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(Plain[dtype, w](x)).v


def bench_at_size[n: Int](mut ctx: DeviceContext) raises -> Bool:
    """Times CPU vs GPU at one size `n`, prints one row, and returns whether
    the two paths' outputs agreed (for the sanity check `main` runs on the
    largest size).
    """
    comptime layout = row_major[n]()
    comptime num_blocks = (n + block_size - 1) // block_size

    var xs_cpu_storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        xs_cpu_storage.append(Scalar[dtype](i) * 0.0001 - 50.0)
    var ys_cpu_storage = List[Scalar[dtype]](length=n, fill=0)
    var xs_cpu = TileTensor(xs_cpu_storage, layout)
    var ys_cpu = TileTensor(ys_cpu_storage, layout)

    var xs_gpu_buf = ctx.enqueue_create_buffer[dtype](n)
    var ys_gpu_buf = ctx.enqueue_create_buffer[dtype](n)
    with xs_gpu_buf.map_to_host() as h:
        for i in range(n):
            h[i] = Scalar[dtype](i) * 0.0001 - 50.0
    var xs_gpu = TileTensor(xs_gpu_buf, layout)
    var ys_gpu = TileTensor(ys_gpu_buf, layout)

    def bench_cpu() {imm xs_cpu, imm ys_cpu} -> Int:
        var t0 = perf_counter_ns()
        map[width=cpu_width, step=gaussian_step](xs_cpu, ys_cpu)
        return perf_counter_ns() - t0

    def bench_gpu() raises {imm ctx, imm xs_gpu, imm ys_gpu} -> Int:
        var t0 = perf_counter_ns()
        ctx.enqueue_function[
            map[LayoutType=type_of(layout), step=gaussian_step, gpu=True]
        ](xs_gpu, ys_gpu, grid_dim=num_blocks, block_dim=block_size)
        ctx.synchronize()
        return perf_counter_ns() - t0

    def bench_gpu_amortized() raises {imm ctx, imm xs_gpu, imm ys_gpu} -> Int:
        """All `timed_iters` launches, one synchronize at the end.

        Returns total nanoseconds for the whole batch, not per launch.
        """
        var t0 = perf_counter_ns()
        for _ in range(timed_iters):
            ctx.enqueue_function[
                map[LayoutType=type_of(layout), step=gaussian_step, gpu=True]
            ](xs_gpu, ys_gpu, grid_dim=num_blocks, block_dim=block_size)
        ctx.synchronize()
        return perf_counter_ns() - t0

    # All three measurements get their own warmup. The amortized batch needs
    # it as much as the per-call one does: measured without it, its first
    # batch came in ~2.8x slower than a warmed one at 1M, which would have
    # read as "pipelining barely helps at this size" rather than as a cold
    # pipeline.
    for _ in range(warmup_iters):
        _ = bench_cpu()
        _ = bench_gpu()
        _ = bench_gpu_amortized()

    # Each path gets its own loop, rather than both accumulating inside one.
    # Interleaving them measured ~16% lower for the GPU at 67M (see
    # `bench_gpu_roofline.mojo`'s interleaving section): a CPU `map` pass at
    # that size is ~28ms of bandwidth-saturating work on the same unified
    # memory the GPU is streaming from, so alternating the two has each one
    # timing the other's interference rather than its own throughput.
    var cpu_total: Int = 0
    for _ in range(timed_iters):
        cpu_total += bench_cpu()

    var gpu_total: Int = 0
    for _ in range(timed_iters):
        gpu_total += bench_gpu()

    var gpu_amortized_total = bench_gpu_amortized()

    var cpu_avg_ns = Float64(cpu_total) / Float64(timed_iters)
    var gpu_avg_ns = Float64(gpu_total) / Float64(timed_iters)
    var gpu_amort_avg_ns = Float64(gpu_amortized_total) / Float64(timed_iters)
    var winner = "GPU" if gpu_avg_ns < cpu_avg_ns else "CPU"
    var cpu_elems_per_sec_m = Float64(n) / (cpu_avg_ns / 1e9) / 1e6
    var gpu_elems_per_sec_m = Float64(n) / (gpu_avg_ns / 1e9) / 1e6
    var gpu_amort_elems_per_sec_m = Float64(n) / (gpu_amort_avg_ns / 1e9) / 1e6

    print(
        "n=",
        n,
        " CPU=",
        cpu_avg_ns / 1e6,
        "ms (",
        cpu_elems_per_sec_m,
        "M elem/s)  GPU per-call=",
        gpu_avg_ns / 1e6,
        "ms (",
        gpu_elems_per_sec_m,
        "M elem/s)  GPU amortized=",
        gpu_amort_avg_ns / 1e6,
        "ms (",
        gpu_amort_elems_per_sec_m,
        "M elem/s)  GPU/CPU=",
        gpu_avg_ns / cpu_avg_ns,
        " (",
        winner,
        "wins per-call)",
    )

    with xs_gpu_buf.map_to_host() as xs_h, ys_gpu_buf.map_to_host() as ys_h:
        var max_diff = Float64(0)
        for i in range(n):
            var diff = abs(Float64(ys_cpu_storage[i]) - Float64(ys_h[i]))
            if diff > max_diff:
                max_diff = diff
        return max_diff < 1e-5


def main() raises:
    var ctx = DeviceContext()
    print(
        "dtype =", dtype, " GPU API:", ctx.api(), " CPU SIMD width =", cpu_width
    )
    print("(one row per size; larger n = more work per GPU dispatch)")

    var agreed = True
    agreed = bench_at_size[sizes[0]](ctx) and agreed
    agreed = bench_at_size[sizes[1]](ctx) and agreed
    agreed = bench_at_size[sizes[2]](ctx) and agreed
    agreed = bench_at_size[sizes[3]](ctx) and agreed
    agreed = bench_at_size[sizes[4]](ctx) and agreed
    agreed = bench_at_size[sizes[5]](ctx) and agreed

    print("CPU and GPU outputs agreed at every size:", agreed)
