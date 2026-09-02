"""Where does `map[gpu=True]`'s throughput actually go? A roofline probe.

`bench_tensor_map_gpu.mojo` reported `gaussian` at ~9,700 M elem/s on an
M3 Pro's Metal GPU at 67M elements. On its own that number says nothing
about whether it's *good*: an elementwise pass over `n` float32 elements
moves `8*n` bytes (one read, one write) no matter how the kernel is
written, so at 9,700 M elem/s it is moving ~78 GB/s -- about half of an
M3 Pro's 150 GB/s. This file exists to find out which half of the machine
is responsible, because the answer picks the fix:

1. **The gate.** An identity `step` (pure copy, no arithmetic at all) run
   the same way as `gaussian`. If copy is also ~78 GB/s, the arithmetic is
   free and the *memory access pattern* is the ceiling. If copy is near
   150 GB/s while `gaussian` is at 78, the memory path is fine and
   `std.math.exp` is the ceiling instead. These two call for completely
   different work, so it's worth one measurement not to guess.
2. **The width x block sweep.** The current path is one element per
   thread: a scalar 4-byte load and a scalar 4-byte store per thread
   (`ys.store[1](Coord(i), step[1](xs.load[1](Coord(i))))` in
   `numax/tensor.mojo`). A bandwidth-bound kernel usually can't saturate
   at that access width, so `_coarse_map` below gives each thread `width`
   *consecutive* elements via one `load[width]`/`store[width]` pair, and
   sweeps that against `block_dim`. `numax` hardcodes `block_dim=256`
   everywhere with no tuning on record, so both axes are open questions.
3. **Sync accounting.** `bench_tensor_map_gpu.mojo` synchronizes once per
   timed iteration (launch-through-completion latency, as a caller
   experiences one call); `bench/torch/gaussian.py` enqueues all of its
   timed iterations and synchronizes once at the end (pipelined
   throughput). Those are different questions with different answers, and
   the cross-language table in `bench/README.md` currently mixes them. The
   last section measures both shapes for the same kernel so the size of
   that discrepancy is a number rather than a suspicion.

`_coarse_map` is deliberately a local copy of `map[gpu=True]`'s body rather
than a call into it: `map` accepts a `width` parameter but ignores it on the
GPU path, so there is nothing to sweep yet. Whichever `width` wins here is
what `map` should be taught to honor.

Every row reports GB/s alongside M elem/s. GB/s is the number that says
whether there is anything left to win, once you divide it by your device's
peak bandwidth -- which this file deliberately does not do for you. Peak is
a hardware constant, is not queryable through `DeviceContext`, and an
assumed one is worse than none: with an M3 Pro's 150 GB/s hardcoded here,
an A10G run printed "394 GB/s, 262 % of peak", which is impossible and
reads as "nothing left to win" when the real figure was ~66 %. The
datasheet number is one lookup away and it is yours, not this file's.

## What it found on an M3 Pro

Recorded here because the conclusions changed what the rest of `bench/`
claims, and because re-running this after a Mojo/MAX bump is the way to
check whether they still hold.

1. **Bandwidth-bound, not compute-bound.** At 67M elements, copy reached
   118 GB/s and `gaussian` 120 GB/s -- equal within run-to-run spread. The
   `exp` is free; the kernel is waiting on memory at every size tested. So
   a cheaper GPU `exp` approximation would buy nothing, and the accuracy
   trade it would cost is not worth paying.
2. **Coarsening does not help on this hardware.** Across `width` in
   `{1, 2, 4, 8}` and `block_dim` in `{128, 256, 512, 1024}`, `width=1` was
   fastest or tied every time, and `width=8` was consistently a few percent
   behind. Neighbouring threads already read neighbouring addresses, so the
   hardware coalesces them into wide transactions and there is nothing for
   a wider per-thread access to recover. `block_dim=256` (the value every
   call site in `numax` already used) is at or within ~1% of the best.
   `map[gpu=True]` was still taught to honor `width` -- it previously
   accepted and silently discarded it, which is worse than either answer --
   but `width=1` remains the default and the recommendation.
3. **The two real problems were in the benchmarks, not in `numax`.** Both
   were worth roughly 40% at 67M combined, and neither needed a library
   change to fix: `bench_tensor_map_gpu.mojo` accumulated its CPU and GPU
   timings in one interleaved loop (section 6 below quantifies it), and it
   compared its own per-call-sync numbers against PyTorch's
   amortized-sync ones. Fixing those moved the reported figure from ~9,400
   to ~15,700 M elem/s, i.e. from "52% of peak, far behind torch.compile"
   to "84% of peak, at parity with it".

So the headline is that `numax`'s GPU elementwise path was already close to
the roofline and the gap to `torch.compile` was mostly measurement. The
remaining ~16% to peak is not addressable by anything in this file.

Needs a real GPU (any backend `DeviceContext` supports); not part of CI,
which has no GPU runners. Run with `pixi run bench-roofline`.
"""

from layout import Coord, TileTensor
from layout.tile_layout import TensorLayout, row_major
from layout.tile_tensor import PointerStorage
from max.gpu.host import DeviceContext
from std.gpu import global_idx
from std.sys.info import simd_width_of
from std.time import perf_counter_ns

from numax import Plain, gaussian
from numax.tensor import map

comptime dtype = DType.float32
# Higher than the 3/10 the other benchmarks use: this one compares
# configurations against each other rather than reporting one headline
# number, and at the smaller sizes run-to-run spread was wide enough at 10
# iterations to invent differences between configurations that repeated runs
# did not reproduce.
comptime warmup_iters = 5
comptime timed_iters = 20
comptime sizes = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26]
comptime block_sizes = [128, 256, 512, 1024]


# One float32 read plus one float32 write per element. Every elementwise
# kernel here moves exactly this much regardless of the arithmetic between.
comptime bytes_per_elem = 8


def gaussian_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(Plain[dtype, w](x)).v


def copy_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """The identity: no arithmetic, so this measures the memory path alone."""
    return x


def _coarse_map[
    dtype: DType,
    LayoutType: TensorLayout,
    step: def[w: Int](SIMD[dtype, w]) thin -> SIMD[dtype, w],
    width: Int,
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
) where (
    TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
):
    """`map[gpu=True]`, but each thread handles `width` consecutive elements.

    Launch with `grid_dim = ceildiv(ceildiv(n, width), block_size)`. A thread
    whose full `width`-wide group fits does one wide load and one wide store;
    the single thread holding a partial group at the end falls back to scalar.

    The scalar tail is an ordinary `if`, which is fine here and is *not* a
    violation of this library's fixed-iteration rule: that rule is about
    SIMD lanes inside a `FloatLike`, which cannot branch independently of
    each other, whereas separate GPU threads diverging is exactly what
    threads do.
    """
    var xs_flat = xs.coalesce()
    var ys_flat = ys.coalesce()
    var n = xs_flat.num_elements()
    var base = Int(global_idx.x) * width
    if base + width <= n:
        ys_flat.store[width](
            Coord(base), step[width](xs_flat.load[width](Coord(base)))
        )
    else:
        for i in range(base, n):
            ys_flat.store[1](Coord(i), step[1](xs_flat.load[1](Coord(i))))


def _report(label: String, n: Int, avg_ns: Float64):
    var secs = avg_ns / 1e9
    var elems_per_sec_m = Float64(n) / secs / 1e6
    var gb_s = Float64(n * bytes_per_elem) / secs / 1e9
    print("    ", label, ": ", elems_per_sec_m, " M elem/s  ", gb_s, " GB/s")


def _time_launch[
    n: Int,
    LayoutType: TensorLayout,
    step: def[w: Int](SIMD[dtype, w]) thin -> SIMD[dtype, w],
    width: Int,
    amortize_sync: Bool = False,
](
    mut ctx: DeviceContext,
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
    block_size: Int,
) raises -> Float64 where (
    TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
):
    """Average nanoseconds per launch of `_coarse_map` at these parameters.

    `amortize_sync=False` synchronizes inside each timed iteration (per-call
    latency); `amortize_sync=True` enqueues every iteration and synchronizes
    once at the end (pipelined throughput). `block_size` is a runtime
    argument to `enqueue_function`, so sweeping it needs no extra
    instantiation -- only `width` does.
    """
    comptime kernel = _coarse_map[LayoutType=LayoutType, step=step, width=width]
    var threads = (n + width - 1) // width
    var grid = (threads + block_size - 1) // block_size

    for _ in range(warmup_iters):
        ctx.enqueue_function[kernel](
            xs, ys, grid_dim=grid, block_dim=block_size
        )
    ctx.synchronize()

    var t0 = perf_counter_ns()
    comptime if amortize_sync:
        for _ in range(timed_iters):
            ctx.enqueue_function[kernel](
                xs, ys, grid_dim=grid, block_dim=block_size
            )
        ctx.synchronize()
    else:
        for _ in range(timed_iters):
            ctx.enqueue_function[kernel](
                xs, ys, grid_dim=grid, block_dim=block_size
            )
            ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / Float64(timed_iters)


def gate_at_size[n: Int](mut ctx: DeviceContext) raises:
    """Section 1: copy vs `gaussian`, both at the current width=1/block=256."""
    comptime layout = row_major[n]()
    var xs_buf = ctx.enqueue_create_buffer[dtype](n)
    var ys_buf = ctx.enqueue_create_buffer[dtype](n)
    with xs_buf.map_to_host() as h:
        for i in range(n):
            h[i] = Scalar[dtype](i) * 0.0001 - 50.0
    var xs = TileTensor(xs_buf, layout)
    var ys = TileTensor(ys_buf, layout)

    print("  n =", n)
    _report(
        "copy    ",
        n,
        _time_launch[n, type_of(layout), copy_step, 1](ctx, xs, ys, 256),
    )
    _report(
        "gaussian",
        n,
        _time_launch[n, type_of(layout), gaussian_step, 1](ctx, xs, ys, 256),
    )


def sweep_at_size[
    n: Int, amortize_sync: Bool = False
](mut ctx: DeviceContext) raises:
    """Section 2: width x block_dim, for both kernels, at one size."""
    comptime layout = row_major[n]()
    var xs_buf = ctx.enqueue_create_buffer[dtype](n)
    var ys_buf = ctx.enqueue_create_buffer[dtype](n)
    with xs_buf.map_to_host() as h:
        for i in range(n):
            h[i] = Scalar[dtype](i) * 0.0001 - 50.0
    var xs = TileTensor(xs_buf, layout)
    var ys = TileTensor(ys_buf, layout)

    print("  n =", n, " (gaussian)")
    for b in materialize[block_sizes]():
        _report(
            "w=1 block=" + String(b),
            n,
            _time_launch[n, type_of(layout), gaussian_step, 1, amortize_sync](
                ctx, xs, ys, b
            ),
        )
        _report(
            "w=2 block=" + String(b),
            n,
            _time_launch[n, type_of(layout), gaussian_step, 2, amortize_sync](
                ctx, xs, ys, b
            ),
        )
        _report(
            "w=4 block=" + String(b),
            n,
            _time_launch[n, type_of(layout), gaussian_step, 4, amortize_sync](
                ctx, xs, ys, b
            ),
        )
        _report(
            "w=8 block=" + String(b),
            n,
            _time_launch[n, type_of(layout), gaussian_step, 8, amortize_sync](
                ctx, xs, ys, b
            ),
        )


def sync_accounting_at_size[n: Int](mut ctx: DeviceContext) raises:
    """Section 3: per-call sync vs sync amortized across the timed loop."""
    comptime layout = row_major[n]()
    var xs_buf = ctx.enqueue_create_buffer[dtype](n)
    var ys_buf = ctx.enqueue_create_buffer[dtype](n)
    with xs_buf.map_to_host() as h:
        for i in range(n):
            h[i] = Scalar[dtype](i) * 0.0001 - 50.0
    var xs = TileTensor(xs_buf, layout)
    var ys = TileTensor(ys_buf, layout)

    print("  n =", n)
    _report(
        "per-call sync ",
        n,
        _time_launch[n, type_of(layout), gaussian_step, 1](ctx, xs, ys, 256),
    )
    _report(
        "amortized sync",
        n,
        _time_launch[n, type_of(layout), gaussian_step, 1, amortize_sync=True](
            ctx, xs, ys, 256
        ),
    )


def interleave_artifact_at_size[n: Int](mut ctx: DeviceContext) raises:
    """Why does `bench_tensor_map_gpu.mojo` report a much lower GPU number?

    That benchmark accumulates both paths in one loop -- `cpu_total +=
    bench_cpu(); gpu_total += bench_gpu()` -- so a full CPU `map` pass runs
    between every timed GPU launch. At 67M elements that CPU pass is ~28 ms
    of memory-saturating work on the *same* unified memory the GPU is trying
    to stream from, and it also doubles the resident footprint (a
    `List`-backed pair of tensors alongside the device-backed pair).

    This section runs the identical GPU launch two ways -- alone, and with a
    CPU `map` pass interleaved exactly as that benchmark does it -- so the
    difference between them is attributable rather than merely suspected.
    """
    comptime layout = row_major[n]()
    comptime cpu_width = simd_width_of[dtype]()

    var xs_buf = ctx.enqueue_create_buffer[dtype](n)
    var ys_buf = ctx.enqueue_create_buffer[dtype](n)
    with xs_buf.map_to_host() as h:
        for i in range(n):
            h[i] = Scalar[dtype](i) * 0.0001 - 50.0
    var xs = TileTensor(xs_buf, layout)
    var ys = TileTensor(ys_buf, layout)

    var xs_cpu_storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        xs_cpu_storage.append(Scalar[dtype](i) * 0.0001 - 50.0)
    var ys_cpu_storage = List[Scalar[dtype]](length=n, fill=0)
    var xs_cpu = TileTensor(xs_cpu_storage, layout)
    var ys_cpu = TileTensor(ys_cpu_storage, layout)

    var grid = (n + 255) // 256
    comptime kernel = _coarse_map[
        LayoutType=type_of(layout), step=gaussian_step, width=1
    ]

    for _ in range(warmup_iters):
        ctx.enqueue_function[kernel](xs, ys, grid_dim=grid, block_dim=256)
    ctx.synchronize()

    var alone_total: Int = 0
    for _ in range(timed_iters):
        var t0 = perf_counter_ns()
        ctx.enqueue_function[kernel](xs, ys, grid_dim=grid, block_dim=256)
        ctx.synchronize()
        alone_total += perf_counter_ns() - t0

    var interleaved_total: Int = 0
    for _ in range(timed_iters):
        map[width=cpu_width, step=gaussian_step](xs_cpu, ys_cpu)
        var t0 = perf_counter_ns()
        ctx.enqueue_function[kernel](xs, ys, grid_dim=grid, block_dim=256)
        ctx.synchronize()
        interleaved_total += perf_counter_ns() - t0

    print("  n =", n)
    _report(
        "GPU alone            ",
        n,
        Float64(alone_total) / Float64(timed_iters),
    )
    _report(
        "GPU + CPU interleaved",
        n,
        Float64(interleaved_total) / Float64(timed_iters),
    )


def check_coarse_correct[n: Int, width: Int](mut ctx: DeviceContext) raises:
    """A coarsened launch has to produce exactly what the scalar one does.

    `n` here is deliberately *not* a multiple of `width` for some callers, so
    the scalar tail is exercised rather than assumed.
    """
    comptime layout = row_major[n]()
    var xs_buf = ctx.enqueue_create_buffer[dtype](n)
    var scalar_buf = ctx.enqueue_create_buffer[dtype](n)
    var coarse_buf = ctx.enqueue_create_buffer[dtype](n)
    with xs_buf.map_to_host() as h:
        for i in range(n):
            h[i] = Scalar[dtype](i) * 0.0001 - 50.0
    var xs = TileTensor(xs_buf, layout)
    var scalar_ys = TileTensor(scalar_buf, layout)
    var coarse_ys = TileTensor(coarse_buf, layout)

    ctx.enqueue_function[
        _coarse_map[LayoutType=type_of(layout), step=gaussian_step, width=1]
    ](xs, scalar_ys, grid_dim=(n + 255) // 256, block_dim=256)
    var threads = (n + width - 1) // width
    ctx.enqueue_function[
        _coarse_map[LayoutType=type_of(layout), step=gaussian_step, width=width]
    ](xs, coarse_ys, grid_dim=(threads + 255) // 256, block_dim=256)
    ctx.synchronize()

    with scalar_buf.map_to_host() as s_h, coarse_buf.map_to_host() as c_h:
        var max_diff = Float64(0)
        for i in range(n):
            max_diff = max(max_diff, abs(Float64(s_h[i]) - Float64(c_h[i])))
        print(
            "    n=",
            n,
            " width=",
            width,
            " max |coarse - scalar| =",
            max_diff,
            " (exact match expected)",
        )


def main() raises:
    var ctx = DeviceContext()
    print("GPU API:", ctx.api(), " device:", ctx.name(), " dtype:", dtype)

    print()
    print("== 1. Gate: is the memory path or std.math.exp the ceiling? ==")
    gate_at_size[sizes[0]](ctx)
    gate_at_size[sizes[1]](ctx)
    gate_at_size[sizes[2]](ctx)
    gate_at_size[sizes[3]](ctx)
    gate_at_size[sizes[4]](ctx)
    gate_at_size[sizes[5]](ctx)

    print()
    print("== 2. Correctness of the coarsened kernel before trusting it ==")
    check_coarse_correct[1 << 20, 4](ctx)
    # Not a multiple of 4 or 8: exercises the scalar tail.
    check_coarse_correct[(1 << 20) + 3, 4](ctx)
    check_coarse_correct[(1 << 20) + 3, 8](ctx)

    print()
    print("== 3. width x block_dim at the largest size, per-call sync ==")
    sweep_at_size[sizes[5]](ctx)

    print()
    print("== 4. width x block_dim at the largest size, amortized sync ==")
    print("==    (this is the shape torch's published number is in)     ==")
    sweep_at_size[sizes[5], amortize_sync=True](ctx)

    print()
    print("== 5. Sync accounting: per-call vs amortized ==")
    sync_accounting_at_size[sizes[0]](ctx)
    sync_accounting_at_size[sizes[2]](ctx)
    sync_accounting_at_size[sizes[5]](ctx)

    print()
    print("== 6. Is bench_tensor_map_gpu.mojo's CPU/GPU interleaving the ==")
    print("==    reason its GPU number is lower than section 1's?       ==")
    interleave_artifact_at_size[sizes[2]](ctx)
    interleave_artifact_at_size[sizes[5]](ctx)
