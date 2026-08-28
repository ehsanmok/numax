"""Where does MAX's `matmul` win, and where does `numax`'s own earn its keep?

`max.linalg.matmul` is blocked, vectorized, and multi-threaded over a
`TileTensor` of raw `dtype`. `numax.linalg.matmul` is a naive triple loop
over an `Array[T, n*n]` that is generic in `T: FloatLike`. They are not
competitors at the same job, and the point of this benchmark is to show
where the line falls rather than to declare a winner:

1. **MAX** -- one `n x n` product per call, `target="cpu"`.
2. **numax, scalar** -- the same single product through `Plain[dtype, 1]`.
3. **numax, batched** -- `Plain[dtype, width]` does `width` independent
   products in one call, one per SIMD lane, which is the shape a
   per-element kernel actually needs and the shape MAX's API cannot
   express at all. Reported per matrix, so the column is comparable.

The generic path also differentiates and can run at extra precision, which
no amount of speed makes up for when that is what you need. Run with
`pixi run bench-matmul`.
"""

from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul as max_matmul
from std.collections import Array
from std.sys.info import simd_width_of
from std.time import perf_counter_ns

from numax import Plain
from numax.linalg import matmul as numax_matmul

comptime dtype = DType.float32
comptime width = simd_width_of[dtype]()
comptime warmup_iters = 5


def _entry(i: Int, salt: Int) -> Float64:
    """A cheap deterministic filler, bounded so `float32` accumulation over
    the larger sizes stays comparable between the two implementations."""
    return Float64((i * 37 + salt * 11) % 17) * 0.0625 - 0.5


def _numax_operands[n: Int, w: Int](salt: Int) -> Array[Plain[dtype, w], n * n]:
    var out = Array[Plain[dtype, w], n * n](
        fill=Plain[dtype, w](SIMD[dtype, w](0))
    )
    for i in range(n * n):
        out[i] = Plain[dtype, w](SIMD[dtype, w](Scalar[dtype](_entry(i, salt))))
    return out^


def run[n: Int](timed_iters: Int) raises:
    comptime lay = row_major[n, n]()
    var a_store = List[Scalar[dtype]](length=n * n, fill=0)
    var b_store = List[Scalar[dtype]](length=n * n, fill=0)
    var c_store = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n * n):
        a_store[i] = Scalar[dtype](_entry(i, 0))
        b_store[i] = Scalar[dtype](_entry(i, 1))
    var a = TileTensor(a_store, lay)
    var b = TileTensor(b_store, lay)
    var c = TileTensor(c_store, lay)

    var ea = _numax_operands[n, 1](0)
    var eb = _numax_operands[n, 1](1)
    var ea_wide = _numax_operands[n, width](0)
    var eb_wide = _numax_operands[n, width](1)

    var ec = numax_matmul[Plain[dtype, 1], n](ea, eb)
    for _ in range(warmup_iters):
        max_matmul[target="cpu"](c, a, b)
        _ = numax_matmul[Plain[dtype, 1], n](ea, eb)
        _ = numax_matmul[Plain[dtype, width], n](ea_wide, eb_wide)

    var t0 = perf_counter_ns()
    for _ in range(timed_iters):
        max_matmul[target="cpu"](c, a, b)
    var max_ns = Float64(perf_counter_ns() - t0) / Float64(timed_iters)

    t0 = perf_counter_ns()
    for _ in range(timed_iters):
        _ = numax_matmul[Plain[dtype, 1], n](ea, eb)
    var scalar_ns = Float64(perf_counter_ns() - t0) / Float64(timed_iters)

    t0 = perf_counter_ns()
    for _ in range(timed_iters):
        _ = numax_matmul[Plain[dtype, width], n](ea_wide, eb_wide)
    var batched_ns = (
        Float64(perf_counter_ns() - t0) / Float64(timed_iters) / Float64(width)
    )

    var max_diff = Float64(0)
    for i in range(n * n):
        var diff = abs(Float64(c_store[i]) - Float64(ec[i].v[0]))
        if diff > max_diff:
            max_diff = diff

    print(
        n,
        "\t",
        Int(max_ns),
        "\t",
        Int(scalar_ns),
        "\t",
        Int(batched_ns),
        "\t",
        max_diff,
    )


def main() raises:
    print("dtype =", dtype, " SIMD width =", width, " (batched = per matrix)")
    print("n\tMAX ns\tnumax ns\tbatched ns/matrix\tmax |diff|")
    run[4](5000)
    run[8](2000)
    run[16](1000)
    run[32](200)
    run[64](50)
