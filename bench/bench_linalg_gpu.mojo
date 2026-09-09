"""The same linalg sweep as `bench_linalg.mojo`, on the device.

Same operations, same matrices, same flop counts, same residuals -- so the
two files answer one question each and neither answers the other's. Per
`CLAUDE.md` the CPU and GPU numbers never share a table: they are different
processors, and a row that mixed them would be a claim about which machine
this is, not about numax.

What the device changes is *which* part of a blocked factorization hurts.
The trailing update is a GEMM and the GPU is extremely good at it -- the
`matmul (ceiling)` row shows how good. The panel is a single-block kernel
by construction (`numax/linalg/panel.mojo`), so it uses one SM while the
rest of the device idles, and every block step is a host-side launch. The
gap between the ceiling row and the factorization rows is therefore *the*
number to watch here: on CPU it is the panel's serial arithmetic, on GPU it
is that plus launch latency, and the two do not have the same fix.

**`float32` only, and not by choice.** `linalg.matmul` does not compile for
GPU at `float64` -- MAX's GEMV path reduces through `warp.shuffle`, which
has no `float64` case, and because numax's operands are runtime-shaped
every branch of the dispatch gets instantiated whether it runs or not. So
a `float64` factorization on an accelerator is not slow in numax, it is a
compile
error, and this file is `float32` for the same reason the CPU file is.

Every measurement synchronizes inside the timed region, so each row is one
full launch-through-completion round trip -- the latency a caller sees, not
a pipelined steady state. That is the honest shape for a factorization,
which is a sequence of dependent launches and cannot pipeline anyway.

**Read the residual column as a bound, not as the factorization's error.**
It is computed with a `float32` GEMM, so it cannot be smaller than that
GEMM's own error, and on this device that is not small: MAX's `float32`
`matmul` of two `1024 x 1024` matrices has a max residual of 8.4 against a
`float64` recomputation, where cuBLAS's FP32 path gives 3.4 and its TF32
path 18.2. So a Cholesky row reporting ~1 at `n = 1024` is reporting the
check, and the CPU file -- where the same factorization at the same size
reports 1.2e-4 -- is the one to read for accuracy.

**The BLAS-1 table lives in `bench_blas1_gpu.mojo`**, and the reason is a
build limit rather than a methodological one: the two halves together
instantiate more GPU kernels than Apple's Metal compiler will put in one
metallib, so the combined file failed with "Metal Compiler failed to
compile metallib" before running anything. Either half builds alone --
established by bisection, at every size and every factorization -- so
splitting them is what makes the device numbers reachable on Metal at all.
CUDA built either way, and neither table's numbers change.

Needs a real device -- CUDA or Metal, whichever `DeviceContext` finds. Not
part of CI, which has no GPU runners. Run with `pixi run bench-linalg-gpu`.
"""

from max.gpu.host import DeviceContext
from std.time import perf_counter_ns

from numax.core.array import Static, transpose
from numax.linalg import (
    asum,
    axpy,
    cholesky,
    dot,
    lu_factor,
    matmul,
    matvec,
    nrm2,
    qr_factor,
    solve,
)

comptime dtype = DType.float32
comptime warmup_iters = 2


def _iters(work: Int) -> Int:
    """Iteration count scaled to keep each measurement around a tenth of a
    second, floored at three so the smallest sizes still average."""
    var scaled = 400_000_000 // work
    return max(3, min(200, scaled))


def _spd_entry(i: Int, j: Int, n: Int) -> Float64:
    """One entry of a deterministic symmetric positive definite matrix; see
    `bench_linalg.mojo`, which uses the same two generators so the residual
    columns of the two files mean the same thing."""
    if i == j:
        return Float64(n)
    return 1.0 / (1.0 + Float64(abs(i - j)))


def _general_entry(i: Int, j: Int, n: Int) -> Float64:
    if i == j:
        return Float64(n)
    return Float64((i * 37 + j * 11) % 17) * 0.0625 - 0.5


def _spd[n: Int](ctx: DeviceContext) raises -> Static[dtype, n, n]:
    var values = List[Scalar[dtype]](capacity=n * n)
    for i in range(n):
        for j in range(n):
            values.append(Scalar[dtype](_spd_entry(i, j, n)))
    return Static[dtype, n, n](ctx, values^)


def _general[n: Int](ctx: DeviceContext) raises -> Static[dtype, n, n]:
    var values = List[Scalar[dtype]](capacity=n * n)
    for i in range(n):
        for j in range(n):
            values.append(Scalar[dtype](_general_entry(i, j, n)))
    return Static[dtype, n, n](ctx, values^)


def _general_offset[n: Int](ctx: DeviceContext) raises -> Static[dtype, n, n]:
    var values = List[Scalar[dtype]](capacity=n * n)
    for i in range(n):
        for j in range(n):
            values.append(Scalar[dtype](_general_entry(i, j, n) + 1.0))
    return Static[dtype, n, n](ctx, values^)


def _general_rect[
    m: Int, n: Int
](ctx: DeviceContext) raises -> Static[dtype, m, n]:
    var values = List[Scalar[dtype]](capacity=m * n)
    for i in range(m):
        for j in range(n):
            values.append(Scalar[dtype](_general_entry(i, j, m)))
    return Static[dtype, m, n](ctx, values^)


def _ramp[n: Int](ctx: DeviceContext, salt: Int) raises -> Static[dtype, n]:
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(Scalar[dtype](Float64((i * 37 + salt * 11) % 17) - 8.0))
    return Static[dtype, n](ctx, values^)


def _row(name: String, n: Int, ns: Float64, flops: Float64, resid: Float64):
    print(
        name,
        "\t",
        n,
        "\t",
        ns / 1e6,
        "\t",
        flops / (ns / 1e9) / 1e9,
        "\t",
        resid,
    )


def bench_gemm[n: Int](ctx: DeviceContext) raises:
    """`linalg.matmul` with `target="gpu"`, the ceiling every factorization
    row below is measured against. The residual is the product's first row
    against a `float64` recomputation."""
    var a = _general[n](ctx)
    var b = _general_offset[n](ctx)
    var iters = _iters(2 * n * n * n)

    for _ in range(warmup_iters):
        _ = matmul[dtype, n, n, n, True](a, b)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(iters):
        _ = matmul[dtype, n, n, n, True](a, b)
        ctx.synchronize()
    var ns = Float64(perf_counter_ns() - t0) / Float64(iters)

    var product = matmul[dtype, n, n, n, True](a, b).to_host()
    var worst = Float64(0)
    for j in range(n):
        var want = Float64(0)
        for k in range(n):
            want += _general_entry(0, k, n) * (_general_entry(k, j, n) + 1.0)
        var diff = abs(Float64(product[j]) - want)
        if diff > worst:
            worst = diff

    _row("matmul (ceiling)", n, ns, 2.0 * Float64(n) ** 3, worst)


def bench_cholesky[n: Int, block: Int = 32](ctx: DeviceContext) raises:
    var a = _spd[n](ctx)
    var iters = _iters(n * n * n // 3)

    for _ in range(warmup_iters):
        _ = cholesky[dtype, n, True, block](a)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(iters):
        _ = cholesky[dtype, n, True, block](a)
        ctx.synchronize()
    var ns = Float64(perf_counter_ns() - t0) / Float64(iters)

    var lower = cholesky[dtype, n, True, block](a)
    var upper = transpose[dtype, n, n, True](lower)
    var product = matmul[dtype, n, n, n, True](lower, upper).to_host()
    var worst = Float64(0)
    for i in range(n):
        for j in range(n):
            var diff = abs(Float64(product[i * n + j]) - _spd_entry(i, j, n))
            if diff > worst:
                worst = diff

    var label = String("cholesky b=") + String(block)
    _row(label, n, ns, Float64(n) ** 3 / 3.0, worst)


def bench_lu[n: Int, block: Int = 16](ctx: DeviceContext) raises:
    var a = _general[n](ctx)
    var iters = _iters(2 * n * n * n // 3)

    for _ in range(warmup_iters):
        _ = lu_factor[dtype, n, True, block](a)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(iters):
        _ = lu_factor[dtype, n, True, block](a)
        ctx.synchronize()
    var ns = Float64(perf_counter_ns() - t0) / Float64(iters)

    var factored = lu_factor[dtype, n, True, block](a)
    var b = _ramp[n](ctx, 3)
    var x = factored.solve(b)
    var residual = matvec[dtype, n, n, True](a, x).to_host()
    var rhs = b.to_host()
    var worst = Float64(0)
    for i in range(n):
        var diff = abs(Float64(residual[i]) - Float64(rhs[i]))
        if diff > worst:
            worst = diff

    var label = String("lu_factor b=") + String(block)
    _row(label, n, ns, 2.0 * Float64(n) ** 3 / 3.0, worst)


def bench_solve[n: Int, block: Int = 16](ctx: DeviceContext) raises:
    var a = _general[n](ctx)
    var b = _ramp[n](ctx, 5)
    var iters = _iters(2 * n * n * n // 3)

    for _ in range(warmup_iters):
        _ = solve[dtype, n, True, block](a, b)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(iters):
        _ = solve[dtype, n, True, block](a, b)
        ctx.synchronize()
    var ns = Float64(perf_counter_ns() - t0) / Float64(iters)

    var x = solve[dtype, n, True, block](a, b)
    var residual = matvec[dtype, n, n, True](a, x).to_host()
    var rhs = b.to_host()
    var worst = Float64(0)
    for i in range(n):
        var diff = abs(Float64(residual[i]) - Float64(rhs[i]))
        if diff > worst:
            worst = diff

    var flops = 2.0 * Float64(n) ** 3 / 3.0 + 2.0 * Float64(n) ** 2
    _row("solve", n, ns, flops, worst)


def bench_qr[
    m: Int, n: Int, block: Int = 16
](ctx: DeviceContext) raises where m >= n:
    var a = _general_rect[m, n](ctx)
    var iters = _iters(2 * m * n * n - 2 * n * n * n // 3)

    for _ in range(warmup_iters):
        _ = qr_factor[dtype, m, n, True, block](a)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(iters):
        _ = qr_factor[dtype, m, n, True, block](a)
        ctx.synchronize()
    var ns = Float64(perf_counter_ns() - t0) / Float64(iters)

    var factored = qr_factor[dtype, m, n, True, block](a)
    var r = factored.r()
    var q = factored.q()
    var product = matmul[dtype, m, n, n, True](q, r).to_host()
    var worst = Float64(0)
    for i in range(m):
        for j in range(n):
            var diff = abs(
                Float64(product[i * n + j]) - _general_entry(i, j, m)
            )
            if diff > worst:
                worst = diff

    var flops = 2.0 * Float64(n) ** 2 * (Float64(m) - Float64(n) / 3.0)
    var label = String("qr_factor b=") + String(block)
    _row(label, n, ns, flops, worst)


def main() raises:
    var ctx = DeviceContext()
    print("dtype =", dtype, " GPU API:", ctx.api())

    print()
    print("Factorizations (ms is per call, GFLOP/s from the LAPACK count)")
    print("op\tn\tms\tGFLOP/s\tmax |residual|")
    bench_gemm[256](ctx)
    bench_gemm[512](ctx)
    bench_gemm[1024](ctx)
    bench_cholesky[256](ctx)
    bench_cholesky[512](ctx)
    bench_cholesky[1024](ctx)
    bench_lu[256](ctx)
    bench_lu[512](ctx)
    bench_lu[1024](ctx)
    bench_solve[256](ctx)
    bench_solve[512](ctx)
    bench_solve[1024](ctx)
    bench_qr[256, 256](ctx)
    bench_qr[512, 512](ctx)
