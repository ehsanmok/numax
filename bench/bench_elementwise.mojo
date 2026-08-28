"""Would `max.algorithm.elementwise` be a better backend for `numax`'s `map`?

`elementwise[func, simd_width, target=...]` does what `map` does by hand --
walk a shape at native SIMD width -- and additionally distributes the work
across CPU threads, which `map[gpu=False]` does not. This benchmark runs the
same `gaussian(x) = exp(-x^2)` kernel through both, at the same sizes as
`bench_tensor_map_gpu.mojo`, so the numbers are directly comparable to the
existing CPU-vs-GPU table.

Both paths call the *same* `numax.special.gaussian` over `Plain`, so this
measures the walk-and-distribute layer only, not two different kernels.

Run with `pixi run bench-elementwise`.
"""

from layout import TileTensor
from layout.tile_layout import row_major
from max.gpu.host import DeviceContext
from std.sys.info import simd_width_of
from std.time import perf_counter_ns

from numax import Plain, gaussian
from numax.tensor import map, map_threaded

comptime dtype = DType.float32
comptime width = simd_width_of[dtype]()
comptime warmup_iters = 3
comptime timed_iters = 10


def gaussian_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(Plain[dtype, w](x)).v


def run[log2n: Int](ctx: DeviceContext) raises:
    comptime n = 1 << log2n
    comptime lay = row_major[n]()

    var xs = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        xs[i] = Scalar[dtype](i) * 0.0001 - 50.0
    var ys_map = List[Scalar[dtype]](length=n, fill=0)
    var ys_ew = List[Scalar[dtype]](length=n, fill=0)

    var xt = TileTensor(xs, lay)
    var yt_map = TileTensor(ys_map, lay)
    var yt_ew = TileTensor(ys_ew, lay)

    for _ in range(warmup_iters):
        map[width=width, step=gaussian_step](xt, yt_map)
        map_threaded[width=width, step=gaussian_step](xt, yt_ew, ctx)

    var t0 = perf_counter_ns()
    for _ in range(timed_iters):
        map[width=width, step=gaussian_step](xt, yt_map)
    var map_ns = Float64(perf_counter_ns() - t0) / Float64(timed_iters)

    t0 = perf_counter_ns()
    for _ in range(timed_iters):
        map_threaded[width=width, step=gaussian_step](xt, yt_ew, ctx)
    var ew_ns = Float64(perf_counter_ns() - t0) / Float64(timed_iters)

    var max_diff = Float64(0)
    for i in range(n):
        var diff = abs(Float64(ys_map[i]) - Float64(ys_ew[i]))
        if diff > max_diff:
            max_diff = diff

    var scale = Float64(n) / 1e6
    print(
        n,
        "\t",
        Int(scale / (map_ns / 1e9)),
        "\t",
        Int(scale / (ew_ns / 1e9)),
        "\t",
        map_ns / ew_ns,
        "\t",
        max_diff,
    )


def main() raises:
    var ctx = DeviceContext(api="cpu")
    print("dtype =", dtype, " SIMD width =", width, " ctx =", ctx.api())
    print("n\tmap M elem/s\tmap_threaded M elem/s\tspeedup\tmax |diff|")
    run[16](ctx)
    run[18](ctx)
    run[20](ctx)
    run[22](ctx)
    run[24](ctx)
    run[26](ctx)
