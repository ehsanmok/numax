"""What does chaining two `map` calls cost versus composing inside `step`?

`bench/README.md`'s cross-language table showed `torch.compile` on MPS well
ahead of everything else, consistent with Inductor fusing an elementwise
chain into one kernel. `numax`'s answer is structural rather than a compiler
pass: a `FloatLike` kernel is already fused, because composition happens
inside `step` before any tensor walk starts. That claim is only worth
anything with a number attached, so this measures the two shapes directly:

1. **chained** -- `map(gaussian)` then `map(scale)`, two full passes over
   the buffer, two GPU dispatches.
2. **fused** -- one `map` whose `step` is `scale(gaussian(x))`, one pass,
   one dispatch.

Both compute `2.5 * exp(-x^2)` and are checked against each other.

One methodology note, since it makes these numbers *not* comparable to the
cross-language table in `bench/README.md`: the GPU half enqueues all timed
iterations and synchronizes once at the end, so per-call host round-trip is
amortized across them. That is the right shape for comparing chained
against fused (both sides get the same treatment) and the wrong shape for
comparing against a benchmark that pays a synchronize per call. Run with
`pixi run bench-fusion`.
"""

from layout import TileTensor
from layout.tile_layout import row_major
from max.gpu.host import DeviceContext
from std.sys.info import simd_width_of
from std.time import perf_counter_ns

from numax import Plain, gaussian
from numax.core.tensor import map, map_threaded

comptime dtype = DType.float32
comptime width = simd_width_of[dtype]()
comptime warmup_iters = 3
comptime timed_iters = 10
comptime block_size = 256


def gaussian_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(Plain[dtype, w](x)).v


def scale_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return (Plain[dtype, w](x) * Plain[dtype, w].constant(2.5)).v


def fused_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """The same two operations, composed as `FloatLike` values rather than
    as two tensor walks. This is the whole point: nothing about `map` had to
    change to fuse them."""
    return (gaussian(Plain[dtype, w](x)) * Plain[dtype, w].constant(2.5)).v


def throughput(n: Int, ns: Float64) -> Int:
    return Int(Float64(n) / 1e6 / (ns / 1e9))


def run_cpu[log2n: Int](ctx: DeviceContext) raises:
    comptime n = 1 << log2n
    comptime lay = row_major[n]()
    var xs = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        xs[i] = Scalar[dtype](i) * 0.0001 - 50.0
    var tmp = List[Scalar[dtype]](length=n, fill=0)
    var chained = List[Scalar[dtype]](length=n, fill=0)
    var fused = List[Scalar[dtype]](length=n, fill=0)

    var xt = TileTensor(xs, lay)
    var tt = TileTensor(tmp, lay)
    var ct = TileTensor(chained, lay)
    var ft = TileTensor(fused, lay)

    for _ in range(warmup_iters):
        map_threaded[width=width, step=gaussian_step](xt, tt, ctx)
        map_threaded[width=width, step=scale_step](tt, ct, ctx)
        map_threaded[width=width, step=fused_step](xt, ft, ctx)

    var t0 = perf_counter_ns()
    for _ in range(timed_iters):
        map_threaded[width=width, step=gaussian_step](xt, tt, ctx)
        map_threaded[width=width, step=scale_step](tt, ct, ctx)
    var chained_ns = Float64(perf_counter_ns() - t0) / Float64(timed_iters)

    t0 = perf_counter_ns()
    for _ in range(timed_iters):
        map_threaded[width=width, step=fused_step](xt, ft, ctx)
    var fused_ns = Float64(perf_counter_ns() - t0) / Float64(timed_iters)

    var max_diff = Float64(0)
    for i in range(n):
        var diff = abs(Float64(chained[i]) - Float64(fused[i]))
        if diff > max_diff:
            max_diff = diff

    print(
        n,
        "\t",
        throughput(n, chained_ns),
        "\t",
        throughput(n, fused_ns),
        "\t",
        chained_ns / fused_ns,
        "\t",
        max_diff,
    )


def run_gpu[log2n: Int](ctx: DeviceContext) raises:
    comptime n = 1 << log2n
    comptime lay = row_major[n]()
    comptime grid = (n + block_size - 1) // block_size

    var xb = ctx.enqueue_create_buffer[dtype](n)
    var tb = ctx.enqueue_create_buffer[dtype](n)
    var cb = ctx.enqueue_create_buffer[dtype](n)
    var fb = ctx.enqueue_create_buffer[dtype](n)
    with xb.map_to_host() as h:
        for i in range(n):
            h[i] = Scalar[dtype](i) * 0.0001 - 50.0

    var xt = TileTensor(xb, lay)
    var tt = TileTensor(tb, lay)
    var ct = TileTensor(cb, lay)
    var ft = TileTensor(fb, lay)

    comptime gaussian_kernel = map[
        LayoutType=type_of(lay), step=gaussian_step, gpu=True
    ]
    comptime scale_kernel = map[
        LayoutType=type_of(lay), step=scale_step, gpu=True
    ]
    comptime fused_kernel = map[
        LayoutType=type_of(lay), step=fused_step, gpu=True
    ]

    for _ in range(warmup_iters):
        ctx.enqueue_function[gaussian_kernel](
            xt, tt, grid_dim=grid, block_dim=block_size
        )
        ctx.enqueue_function[scale_kernel](
            tt, ct, grid_dim=grid, block_dim=block_size
        )
        ctx.enqueue_function[fused_kernel](
            xt, ft, grid_dim=grid, block_dim=block_size
        )
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(timed_iters):
        ctx.enqueue_function[gaussian_kernel](
            xt, tt, grid_dim=grid, block_dim=block_size
        )
        ctx.enqueue_function[scale_kernel](
            tt, ct, grid_dim=grid, block_dim=block_size
        )
    ctx.synchronize()
    var chained_ns = Float64(perf_counter_ns() - t0) / Float64(timed_iters)

    t0 = perf_counter_ns()
    for _ in range(timed_iters):
        ctx.enqueue_function[fused_kernel](
            xt, ft, grid_dim=grid, block_dim=block_size
        )
    ctx.synchronize()
    var fused_ns = Float64(perf_counter_ns() - t0) / Float64(timed_iters)

    var max_diff = Float64(0)
    with cb.map_to_host() as ch:
        with fb.map_to_host() as fh:
            for i in range(n):
                var diff = abs(Float64(ch[i]) - Float64(fh[i]))
                if diff > max_diff:
                    max_diff = diff

    print(
        n,
        "\t",
        throughput(n, chained_ns),
        "\t",
        throughput(n, fused_ns),
        "\t",
        chained_ns / fused_ns,
        "\t",
        max_diff,
    )


def main() raises:
    print("dtype =", dtype, " SIMD width =", width)
    print("\nCPU (map_threaded)")
    print("n\tchained M elem/s\tfused M elem/s\tspeedup\tmax |diff|")
    var cpu_ctx = DeviceContext(api="cpu")
    run_cpu[20](cpu_ctx)
    run_cpu[22](cpu_ctx)
    run_cpu[24](cpu_ctx)
    run_cpu[26](cpu_ctx)

    var gpu_ctx = DeviceContext()
    print("\nGPU (map[gpu=True], api =", gpu_ctx.api(), ")")
    print("n\tchained M elem/s\tfused M elem/s\tspeedup\tmax |diff|")
    run_gpu[20](gpu_ctx)
    run_gpu[22](gpu_ctx)
    run_gpu[24](gpu_ctx)
    run_gpu[26](gpu_ctx)
