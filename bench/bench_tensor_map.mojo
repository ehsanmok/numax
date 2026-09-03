"""Does `numax.core.tensor.map`'s abstraction cost anything over raw SIMD?

Two implementations of the same `gaussian(x) = exp(-x^2)` sweep over the same
`n`-element buffer, at the same native SIMD width:

1. `raw_gaussian_loop` -- a hand-rolled `std.algorithm.functional.vectorize`
   loop with `SIMD` arithmetic and `std.math.exp` inline, no `FloatLike`, no
   `TileTensor`.
2. `numax.core.tensor.map` -- the same math, but reached through `Plain` (a
   `FloatLike` conformer) and `gaussian` from `numax.special`, walking a
   `TileTensor` instead of a raw pointer.

If `numax`'s trait-and-`TileTensor` layer is truly zero-cost, the two should
land within noise of each other. Run with `pixi run bench`.
"""

from layout import TileTensor
from layout.tile_layout import row_major, TensorLayout
from layout.tile_tensor import PointerStorage
from std.algorithm.functional import vectorize
from std.math import exp
from std.sys.info import simd_width_of
from std.time import perf_counter_ns

from numax import Plain, gaussian
from numax.core.tensor import map

comptime dtype = DType.float32
comptime n = 1 << 20
comptime width = simd_width_of[dtype]()
comptime warmup_iters = 3
comptime timed_iters = 20


def raw_gaussian_loop(xs: List[Scalar[dtype]], mut ys: List[Scalar[dtype]]):
    var xs_ptr = xs.unsafe_ptr()
    var ys_ptr = ys.unsafe_ptr()

    def step[w: Int](i: Int) {imm xs_ptr, imm ys_ptr}:
        var x = xs_ptr.unsafe_load[width=w](i)
        ys_ptr.unsafe_store(i, exp(-(x * x)))

    vectorize[width](n, step)


def gaussian_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(Plain[dtype, w](x)).v


def bench_raw(xs: List[Scalar[dtype]], mut ys: List[Scalar[dtype]]) -> Int:
    var t0 = perf_counter_ns()
    raw_gaussian_loop(xs, ys)
    return perf_counter_ns() - t0


def bench_numax[
    LayoutType: TensorLayout
](
    xs: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    ys: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
) -> Int where (
    TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
):
    var t0 = perf_counter_ns()
    map[width=width, step=gaussian_step](xs, ys)
    return perf_counter_ns() - t0


def main() raises:
    print("n =", n, " dtype =", dtype, " SIMD width =", width)

    var xs_storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        xs_storage.append(Scalar[dtype](i) * 0.0001 - 50.0)
    var ys_raw = List[Scalar[dtype]](length=n, fill=0)

    comptime layout = row_major[n]()
    var xs_tensor_storage = xs_storage.copy()
    var xs_tensor = TileTensor(xs_tensor_storage, layout)
    var ys_numax_storage = List[Scalar[dtype]](length=n, fill=0)
    var ys_numax = TileTensor(ys_numax_storage, layout)

    # Warm up both paths -- first calls pay for page faults, cache misses,
    # and (for the raw loop) not much else since there's no lazy init here.
    for _ in range(warmup_iters):
        _ = bench_raw(xs_storage, ys_raw)
        _ = bench_numax(xs_tensor, ys_numax)

    var raw_total: Int = 0
    var numax_total: Int = 0
    for _ in range(timed_iters):
        raw_total += bench_raw(xs_storage, ys_raw)
        numax_total += bench_numax(xs_tensor, ys_numax)

    var raw_avg_ns = Float64(raw_total) / Float64(timed_iters)
    var numax_avg_ns = Float64(numax_total) / Float64(timed_iters)

    print("raw SIMD loop:  ", raw_avg_ns / 1e6, "ms/iter")
    print("numax.core.tensor.map:", numax_avg_ns / 1e6, "ms/iter")
    print("numax / raw ratio:  ", numax_avg_ns / raw_avg_ns)

    # Sanity check: both paths should agree, not just run at the same speed.
    var max_diff = Float64(0)
    for i in range(n):
        var diff = abs(Float64(ys_raw[i]) - Float64(ys_numax[i]))
        if diff > max_diff:
            max_diff = diff
    print("max |raw - numax| =", max_diff)
