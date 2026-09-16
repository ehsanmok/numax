"""The same core-surface sweep as `bench_core_surface.mojo`, on the device.

Same operations, same hashed data, same byte counts per element -- but a
separate table and a separate file, because per `CLAUDE.md` a CPU and a
GPU number never share a row. These are different processors and a mixed
table would be a claim about which machine this is.

Five sizes rather than the CPU sweep's seven: `2^16` and `2^18` are there
to bracket `_THREADED_FROM`, and there is no such threshold here. `gpu=True`
is one launch policy at every size, so the extra rows would only be two
more kernel instantiations in the metallib.

Every routine is named with `gpu=True`, so `numax.core._drive` launches
`max.algorithm.elementwise[target="gpu"]` and the data never leaves the
device. The tensors are built on a real `DeviceContext()`, so the device
gate is satisfied and no row is secretly the host fallback.

**Two sync shapes, both reported**, for the reason
`bench_tensor_map_gpu.mojo` gives at length and
`.cursor/rules/findings.mdc` records as a fixed benchmark bug:

- **per-call** synchronizes inside the timed region, so each sample is one
  launch-through-completion round trip -- the latency a caller sees.
- **amortized** enqueues `batch` calls and synchronizes once, so dispatches
  pipeline and the host round trip is paid once -- steady-state throughput.

They differ by more than 2x at small sizes and there is no single honest
number, so neither is reported alone. `sum(a)` is the exception and says so
in its own right: it ends in a one-element `to_host`, which orders against
the launch whatever the caller does, so its two columns are the same
measurement twice.

Each elementwise row allocates its destination per call, the way a
caller's would; in the amortized shape that is `batch` device buffers
alive before the synchronize.

**`float32` only, and not by choice.** Naming `gpu=True` compiles the
device kernels whether or not they run, and `float64` elementwise bodies do
not compile for Metal -- `.cursor/rules/findings.mdc` records the failure.
This is the same constraint `bench_linalg_gpu.mojo` documents.

Needs a real device -- CUDA or Metal, whichever `DeviceContext` finds. Not
part of CI, which has no GPU runners. Run with
`pixi run bench-core-surface-gpu`.
"""

from max.gpu.host import DeviceContext
from std.benchmark import keep, run

from numax.core.array import Static
from numax.core.elementwise import exp
from numax.core.logic import greater
from numax.core.ops import add, multiply
from numax.stats import sum as tensor_sum

comptime dtype = DType.float32
comptime warmup_iters = 2
comptime budget_secs = 1.0

comptime batch = 10
"""Launches enqueued between synchronizes in the amortized shape."""


def _entry(i: Int) -> Float64:
    """A hashed value on `[-3, 3)`, the generator `bench_stats.mojo` uses."""
    var h = (i * 2654435761 + 12345) % 16777216
    return 6.0 * (Float64(h) / 16777216.0 - 0.5)


def _data[n: Int](ctx: DeviceContext, salt: Int) raises -> Static[dtype, n]:
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(Scalar[dtype](_entry(i + salt)))
    return Static[dtype, n](ctx, values^)


def _zeros[n: Int](ctx: DeviceContext) raises -> Static[dtype, n]:
    var values = List[Scalar[dtype]](length=n, fill=Scalar[dtype](0))
    return Static[dtype, n](ctx, values^)


def _row(
    name: String,
    n: Int,
    per_element: Int,
    call_ns: Float64,
    amort_ns: Float64,
):
    print(
        name,
        "\t",
        n,
        "\t",
        per_element,
        "\t",
        call_ns / 1e3,
        "\t",
        Float64(n) / (call_ns / 1e9) / 1e6,
        "\t",
        Float64(n * per_element) / (call_ns / 1e9) / 1e9,
        "\t",
        amort_ns / 1e3,
        "\t",
        Float64(n) / (amort_ns / 1e9) / 1e6,
        "\t",
        Float64(n * per_element) / (amort_ns / 1e9) / 1e9,
    )


def bench_size[n: Int](ctx: DeviceContext) raises where n > 0:
    var a = _data[n](ctx, 0)
    var b = _data[n](ctx, 1)
    var zeros = _zeros[n](ctx)

    def exp_call() raises {mut a, imm ctx}:
        var r = exp[gpu=True](a)
        keep(r.buffer.unsafe_ptr())
        ctx.synchronize()

    def exp_amort() raises {mut a, imm ctx}:
        for _ in range(batch):
            var r = exp[gpu=True](a)
            keep(r.buffer.unsafe_ptr())
        ctx.synchronize()

    def add_call() raises {mut a, mut b, imm ctx}:
        var r = add[gpu=True](a, b)
        keep(r.buffer.unsafe_ptr())
        ctx.synchronize()

    def add_amort() raises {mut a, mut b, imm ctx}:
        for _ in range(batch):
            var r = add[gpu=True](a, b)
            keep(r.buffer.unsafe_ptr())
        ctx.synchronize()

    def scale_call() raises {mut a, imm ctx}:
        var r = multiply[gpu=True](a, Scalar[dtype](2))
        keep(r.buffer.unsafe_ptr())
        ctx.synchronize()

    def scale_amort() raises {mut a, imm ctx}:
        for _ in range(batch):
            var r = multiply[gpu=True](a, Scalar[dtype](2))
            keep(r.buffer.unsafe_ptr())
        ctx.synchronize()

    def compare_call() raises {mut a, mut zeros, imm ctx}:
        var r = greater[gpu=True](a, zeros)
        keep(r.buffer.unsafe_ptr())
        ctx.synchronize()

    def compare_amort() raises {mut a, mut zeros, imm ctx}:
        for _ in range(batch):
            var r = greater[gpu=True](a, zeros)
            keep(r.buffer.unsafe_ptr())
        ctx.synchronize()

    def sum_call() raises {mut a, imm ctx}:
        keep(tensor_sum[gpu=True](a))
        ctx.synchronize()

    def sum_amort() raises {mut a, imm ctx}:
        for _ in range(batch):
            keep(tensor_sum[gpu=True](a))
        ctx.synchronize()

    # Every measurement gets its own warmup, including each amortized
    # batch: measured without one, a cold pipeline reads as "pipelining
    # barely helps" (`bench_tensor_map_gpu.mojo` records the 2.8x).
    _row(
        "exp(a)",
        n,
        8,
        run(
            exp_call,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
            max_batch_size=1,
        ).mean()
        * 1e9,
        run(
            exp_amort,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
            max_batch_size=1,
        ).mean()
        * 1e9
        / Float64(batch),
    )
    _row(
        "a + b",
        n,
        12,
        run(
            add_call,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
            max_batch_size=1,
        ).mean()
        * 1e9,
        run(
            add_amort,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
            max_batch_size=1,
        ).mean()
        * 1e9
        / Float64(batch),
    )
    _row(
        "a * 2",
        n,
        8,
        run(
            scale_call,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
            max_batch_size=1,
        ).mean()
        * 1e9,
        run(
            scale_amort,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
            max_batch_size=1,
        ).mean()
        * 1e9
        / Float64(batch),
    )
    _row(
        "greater(a, zeros)",
        n,
        9,
        run(
            compare_call,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
            max_batch_size=1,
        ).mean()
        * 1e9,
        run(
            compare_amort,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
            max_batch_size=1,
        ).mean()
        * 1e9
        / Float64(batch),
    )
    _row(
        "sum(a)",
        n,
        4,
        run(
            sum_call,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
            max_batch_size=1,
        ).mean()
        * 1e9,
        run(
            sum_amort,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
            max_batch_size=1,
        ).mean()
        * 1e9
        / Float64(batch),
    )


def main() raises:
    var ctx = DeviceContext()
    print(
        "dtype =",
        dtype,
        " GPU API:",
        ctx.api(),
        " amortized batch =",
        batch,
    )

    print()
    print("Core surface on device (us is per call, B/elem bytes per element)")
    print("op\tn\tB/elem\tus/call\tM elem/s\tGB/s\tus/amort\tM elem/s\tGB/s")
    bench_size[1 << 10](ctx)
    bench_size[1 << 14](ctx)
    bench_size[1 << 20](ctx)
    bench_size[1 << 22](ctx)
    bench_size[1 << 24](ctx)
