"""How fast is the NumPy-named core surface over `Tensor` on this CPU?

Five operations, one per shape the surface has, at seven sizes spanning
`numax.core._drive`'s launch policy -- `2^10` and `2^14` below
`_THREADED_FROM`, where the host path is a serial SIMD loop, and `2^16`
through `2^24` above it, where it is `max.algorithm.elementwise` over every
core. The threshold is printed in the header, so a table records the policy
it measured rather than a policy someone has to go and look up.

- `exp(a)` -- one read, one write: **8 bytes per element**.
- `a + b` -- two reads, one write: **12**.
- `a * 2` -- tensor-scalar, so one read and one write again: **8**.
- `greater(a, zeros)` -- two `float32` reads and one `bool` write: **9**.
  numax has no tensor-scalar comparison, so `a > 0` has to name a zero
  tensor and that operand is a real full-width read. The NumPy baseline
  prints both spellings for this reason: `a > 0` at 5 bytes is what a NumPy
  user writes, `np.greater(a, z)` at 9 is the same work numax does.
- `sum(a)` -- a read and nothing else: **4**.

Every row carries its own byte count, because the GB/s column means
nothing without it and the five rows do not agree.

The inputs are built once, outside every timed closure. The destination is
not: each of the four elementwise rows allocates its output the way a
caller's would, so the microseconds include one buffer allocation. `sum`
allocates nothing.

The last table is the abstraction cost, three spellings of the same `exp`
body, run at `2^12` through `2^20` and again at `2^24`. Its first two rows
are what sets `_THREADED_FROM`: they are the same walk over the same two
buffers, so the size at which `map_threaded` overtakes `map` is the
crossover the policy has to encode, measured without the destination
allocation the third row carries. The three spellings:

1. `map` -- the kernel-author primitive, a serial SIMD walk over a
   `TileTensor` the caller already owns.
2. `map_threaded` -- the same walk through `max.algorithm.elementwise`,
   which is what `exp(a)` reaches above `_THREADED_FROM`.
3. `exp(a)` -- the NumPy-named call, which adds the device gate, the
   flatten, and the destination allocation on top of (2).

So (3) against (2) is the surface's overhead and (2) against (1) is what
threading buys, and neither is confounded with the other.

`float32`, for the reason `bench_linalg.mojo` gives, and because the GPU
sibling of this table cannot be anything else. The data is the same hashed
value on `[-3, 3)` that `bench_stats.mojo` uses, so the two files'
elementwise rows are about the same numbers. `bench/numpy/core_surface.py`
prints the same columns from NumPy.

CPU only. Run with `pixi run bench-core-surface`.
"""

from layout import TileTensor
from layout.tile_layout import row_major
from max.gpu.host import DeviceContext
from std.benchmark import keep, run
from std.math import exp as _std_exp
from std.sys.info import simd_width_of

from numax.core._drive import _THREADED_FROM
from numax.core.array import Static
from numax.core.elementwise import exp
from numax.core.logic import greater
from numax.core.ops import add, multiply
from numax.core.tensor import map, map_threaded
from numax.stats import sum as tensor_sum

comptime dtype = DType.float32
comptime width = simd_width_of[dtype]()
comptime warmup_iters = 2
comptime budget_secs = 1.0


def _entry(i: Int) -> Float64:
    """A hashed value on `[-3, 3)`, the generator `bench_stats.mojo` uses."""
    var h = (i * 2654435761 + 12345) % 16777216
    return 6.0 * (Float64(h) / 16777216.0 - 0.5)


def _values[n: Int](salt: Int) -> List[Scalar[dtype]]:
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(Scalar[dtype](_entry(i + salt)))
    return values^


def _data[n: Int](ctx: DeviceContext, salt: Int) raises -> Static[dtype, n]:
    var values = _values[n](salt)
    return Static[dtype, n](ctx, values^)


def _zeros[n: Int](ctx: DeviceContext) raises -> Static[dtype, n]:
    var values = List[Scalar[dtype]](length=n, fill=Scalar[dtype](0))
    return Static[dtype, n](ctx, values^)


def _row(name: String, n: Int, ns: Float64, per_element: Int):
    print(
        name,
        "\t",
        n,
        "\t",
        ns / 1e3,
        "\t",
        Float64(n) / (ns / 1e9) / 1e6,
        "\t",
        per_element,
        "\t",
        Float64(n * per_element) / (ns / 1e9) / 1e9,
    )


def bench_size[n: Int](ctx: DeviceContext) raises where n > 0:
    var a = _data[n](ctx, 0)
    var b = _data[n](ctx, 1)
    var zeros = _zeros[n](ctx)

    def exp_work() raises {mut a}:
        var r = exp(a)
        keep(r.buffer.unsafe_ptr())

    def add_work() raises {mut a, mut b}:
        var r = add(a, b)
        keep(r.buffer.unsafe_ptr())

    def scale_work() raises {mut a}:
        var r = multiply(a, Scalar[dtype](2))
        keep(r.buffer.unsafe_ptr())

    def compare_work() raises {mut a, mut zeros}:
        var r = greater(a, zeros)
        keep(r.buffer.unsafe_ptr())

    def sum_work() raises {mut a}:
        keep(tensor_sum(a))

    _row(
        "exp(a)",
        n,
        run(
            exp_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9,
        8,
    )
    _row(
        "a + b",
        n,
        run(
            add_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9,
        12,
    )
    _row(
        "a * 2",
        n,
        run(
            scale_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9,
        8,
    )
    _row(
        "greater(a, zeros)",
        n,
        run(
            compare_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9,
        9,
    )
    _row(
        "sum(a)",
        n,
        run(
            sum_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9,
        4,
    )


def _exp_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return _std_exp(x)


def bench_abstraction[n: Int](ctx: DeviceContext) raises where n > 0:
    """`map`, `map_threaded` and `exp(a)` over one `exp` body.

    The two primitives write into a destination the caller already owns,
    where `exp(a)` allocates one per call. That difference is part of what
    the third row is measuring and is why it is reported beside them rather
    than subtracted out.
    """
    comptime layout = row_major[n]()
    var xs_storage = _values[n](0)
    var ys_storage = List[Scalar[dtype]](length=n, fill=Scalar[dtype](0))
    var xs = TileTensor(xs_storage, layout)
    var ys = TileTensor(ys_storage, layout)
    var a = _data[n](ctx, 0)

    def map_work() {imm xs, imm ys}:
        map[width=width, step=_exp_step](xs, ys)

    def threaded_work() raises {imm xs, imm ys, imm ctx}:
        map_threaded[width=width, step=_exp_step](xs, ys, ctx)

    def exp_work() raises {mut a}:
        var r = exp(a)
        keep(r.buffer.unsafe_ptr())

    _row(
        "map (serial SIMD)",
        n,
        run(
            map_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9,
        8,
    )
    _row(
        "map_threaded",
        n,
        run(
            threaded_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9,
        8,
    )
    _row(
        "exp(a)",
        n,
        run(
            exp_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9,
        8,
    )


def main() raises:
    var ctx = DeviceContext(api="cpu")
    print(
        "dtype =",
        dtype,
        " target = cpu  SIMD width =",
        width,
        " _THREADED_FROM =",
        _THREADED_FROM,
    )

    print()
    print("Core surface (us is per call, B/elem is bytes moved per element)")
    print("op\tn\tus\tM elem/s\tB/elem\tGB/s")
    bench_size[1 << 10](ctx)
    bench_size[1 << 14](ctx)
    bench_size[1 << 16](ctx)
    bench_size[1 << 18](ctx)
    bench_size[1 << 20](ctx)
    bench_size[1 << 22](ctx)
    bench_size[1 << 24](ctx)

    print()
    print("Abstraction cost: one exp body through three spellings")
    print("op\tn\tus\tM elem/s\tB/elem\tGB/s")
    bench_abstraction[1 << 12](ctx)
    bench_abstraction[1 << 14](ctx)
    bench_abstraction[1 << 15](ctx)
    bench_abstraction[1 << 16](ctx)
    bench_abstraction[1 << 18](ctx)
    bench_abstraction[1 << 20](ctx)
    bench_abstraction[1 << 24](ctx)
