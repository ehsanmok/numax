"""The matrix-equation solvers on a device: `solve_sylvester` against the
two `schur` calls it starts with.

Bartels-Stewart is two real Schur forms, two products rotating the
right-hand side into their bases, the quasi-triangular solve and two
products rotating back. The first and last are what `bench_linalg_gpu.mojo`
already measures, so the number this file adds is the difference between
its two rows: the `trsyl_column` launches, one single-block kernel per
column of `b`'s Schur form. A separate file for the reason
`bench_blas1_gpu.mojo` is one: the device `schur` instantiates enough
kernels that adding it to the factorization sweep risks Metal's metallib
limit.

`float32` only, since Metal has no `double`. Each row is the best of a few
full launch-through-completion round trips, synchronizing inside the
timed region, the same shape as `bench_linalg_gpu.mojo`'s.

Needs a real device -- CUDA or Metal, whichever `DeviceContext` finds. Not
part of CI, which has no GPU runners. Run with `pixi run bench-solvers-gpu`.
"""

from std.math import sin
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.linalg import schur, solve_sylvester

comptime dtype = DType.float32
comptime repeats = 4


def _matrix[
    n: Int
](ctx: DeviceContext, seed: Float32, shift: Float32) raises -> Static[
    dtype, n, n
]:
    """A shifted diagonal plus a small dense perturbation: the shifts keep
    the spectra of `a` and `-b` apart, so the equation is well posed."""
    var values = List[Scalar[dtype]](capacity=n * n)
    for i in range(n):
        for j in range(n):
            var diag = shift if i == j else Float32(0)
            values.append(
                diag + 0.3 * sin(Float32(i * n + j) * seed) / Float32(16)
            )
    return Static[dtype, n, n](values^, ctx)


def bench_sylvester[n: Int](ctx: DeviceContext) raises:
    var a = _matrix[n](ctx, 1.3, -2.0)
    var b = _matrix[n](ctx, 0.7, 1.0)
    var q = _matrix[n](ctx, 0.9, 0.0)
    var solve_ns = Int.MAX
    var schur_ns = Int.MAX
    for _ in range(repeats):
        ctx.synchronize()
        var start = perf_counter_ns()
        var x = solve_sylvester[gpu=True](a, b, q)
        ctx.synchronize()
        solve_ns = min(solve_ns, Int(perf_counter_ns() - start))
        _ = x^
        start = perf_counter_ns()
        var sa = schur[gpu=True](a)
        var sb = schur[gpu=True](b)
        ctx.synchronize()
        schur_ns = min(schur_ns, Int(perf_counter_ns() - start))
        _ = sa^
        _ = sb^
    print("solve_sylvester\t", n, "\t", Float64(solve_ns) / 1e6)
    print("two schur calls\t", n, "\t", Float64(schur_ns) / 1e6)


def main() raises:
    var ctx = DeviceContext()
    print("op\t n\t ms")
    bench_sylvester[128](ctx)
    bench_sylvester[256](ctx)
