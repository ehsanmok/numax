"""How fast is the `Tensor` tier of `numax.linalg` on this CPU, and how much
of its cubic work actually reaches MAX?

Every factorization here is blocked: a panel kernel on the diagonal block,
then a trailing update that is a matrix product and therefore goes to
`linalg.matmul`. So the GFLOP/s column is the whole point of the design.
`bench_matmul.mojo` measures what a bare `linalg.matmul` reaches at these
sizes; a blocked factorization reaching a good fraction of that is one
whose cubic term really is MAX's GEMM, and one that does not is one whose
panel or whose block size is eating the run.

Three tables, because three different things limit them:

1. **Factorizations** -- `cholesky`, `lu_factor`, `solve`, `qr_factor` at
   n = 128 through 1024. Cubic work, so GFLOP/s is the comparable number
   and the flop counts are the standard LAPACK ones (`n^3/3` for Cholesky,
   `2n^3/3` for LU, `4n^3/3` for a square QR, LU plus two substitutions
   for `solve`).
2. **BLAS-1** -- `dot`, `nrm2`, `asum`, `axpy` at n = 64K through 67M.
   Linear work over linear traffic, so these are bandwidth-bound and
   GB/s is the comparable number, not GFLOP/s. The sweep runs past this
   box's 64 MiB of L3 on purpose: at 16M `float32` the vector is exactly
   L3-sized and the reductions read out of cache, so a number there is a
   cache number, not a memory one.
3. **Block size** -- each factorization at its default block against its
   neighbours, at two sizes, because the best block is not the same at
   both: `cholesky` and `lu_factor` hold their defaults (32 and 16) at
   n = 512 and n = 1024, while `qr_factor`'s best block *shrinks* with n,
   so its default of 16 is right at n = 512 and leaves about 2x on the
   table at n = 1024. Pass `block` explicitly for a large QR.

Each row carries a residual so a fast wrong answer cannot hide: `L L^T`
against `A` for Cholesky, `A x - b` for the solves, `Q R` against `A` for
QR, and a `float64` recomputation for the BLAS-1 reductions.

**`float32`, deliberately.** The GPU sibling of this file cannot be
anything else -- `linalg.matmul` does not compile for GPU at `float64`,
because its GEMV path reduces through `warp.shuffle` -- and a CPU table at
`float64` beside a GPU table at `float32` would compare two different
computations. Both files run the same one.

CPU only, and it is a separate file from `bench_linalg_gpu.mojo` for the
reason `bench_tensor_map.mojo` and `bench_tensor_map_gpu.mojo` are
separate: a file that instantiates a GPU kernel needs a device to build,
and this one runs anywhere. Run with `pixi run bench-linalg`.
"""

from max.gpu.host import DeviceContext
from std.benchmark import keep, run
from std.math import sqrt

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

comptime budget_secs = 1.0
"""Seconds `std.benchmark.run` may spend on one measurement.

The iteration count is `run`'s to choose, not this file's. It keeps
halving its estimate until a batch clears `min_runtime_secs` and then
averages until either `max_iters` or this budget runs out, which is what a
hand-rolled count scaled by the cubic work can only approximate -- and
approximated badly at the small sizes, where three iterations of a 0.1 ms
factorization measured mostly timer noise.
"""


def _spd_entry(i: Int, j: Int, n: Int) -> Float64:
    """One entry of a deterministic symmetric positive definite matrix:
    `1 / (1 + |i - j|)` off the diagonal, `n` on it. Diagonal dominance
    makes it positive definite by Gershgorin, so Cholesky cannot fail for
    a reason that is really a bad test matrix."""
    if i == j:
        return Float64(n)
    return 1.0 / (1.0 + Float64(abs(i - j)))


def _general_entry(i: Int, j: Int, n: Int) -> Float64:
    """One entry of a deterministic well-conditioned general matrix, again
    diagonally dominant so the pivoting is exercised without the residual
    being dominated by conditioning."""
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


def _general_rect[
    m: Int, n: Int
](ctx: DeviceContext) raises -> Static[dtype, m, n]:
    var values = List[Scalar[dtype]](capacity=m * n)
    for i in range(m):
        for j in range(n):
            values.append(Scalar[dtype](_general_entry(i, j, m)))
    return Static[dtype, m, n](ctx, values^)


def _general_offset[n: Int](ctx: DeviceContext) raises -> Static[dtype, n, n]:
    """A second general matrix, one greater than `_general` everywhere, so
    the GEMM reference multiplies two different operands."""
    var values = List[Scalar[dtype]](capacity=n * n)
    for i in range(n):
        for j in range(n):
            values.append(Scalar[dtype](_general_entry(i, j, n) + 1.0))
    return Static[dtype, n, n](ctx, values^)


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


def _band_row(name: String, n: Int, ns: Float64, bytes: Int, err: Float64):
    print(
        name,
        "\t",
        n,
        "\t",
        ns / 1e3,
        "\t",
        Float64(bytes) / (ns / 1e9) / 1e9,
        "\t",
        err,
    )


def bench_gemm[n: Int](ctx: DeviceContext) raises:
    """`linalg.matmul` at the same size, which is the ceiling the blocked
    factorizations are trying to reach: their cubic term *is* this call,
    and the fraction of this row a factorization row reaches is how much of
    its work went to MAX rather than to a panel.

    The residual is the first row of the product against a `float64`
    recomputation -- `n` dot products rather than `n^2`, enough to catch a
    wrong shape or a wrong stride without timing a host GEMM.
    """
    var a = _general[n](ctx)
    var b = _general_offset[n](ctx)

    def work() raises {mut a, mut b}:
        var c = matmul[dtype, n, n, n](a, b)
        keep(c.buffer.unsafe_ptr())

    var ns = (
        run(
            work, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9
    )

    var product = matmul[dtype, n, n, n](a, b).to_host()
    var worst = Float64(0)
    for j in range(n):
        var want = Float64(0)
        for k in range(n):
            want += _general_entry(0, k, n) * (_general_entry(k, j, n) + 1.0)
        var diff = abs(Float64(product[j]) - want)
        if diff > worst:
            worst = diff

    _row("matmul (ceiling)", n, ns, 2.0 * Float64(n) ** 3, worst)


def bench_cholesky[n: Int, block: Int = 48](ctx: DeviceContext) raises:
    var a = _spd[n](ctx)

    def work() raises {mut a}:
        var l = cholesky[dtype, n, False, block](a)
        keep(l.buffer.unsafe_ptr())

    var ns = (
        run(
            work, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9
    )

    # `L @ L.T` against `A`, both on the device, so the residual measures
    # the factorization and not a host copy.
    var lower = cholesky[dtype, n, False, block](a)
    var upper = transpose(lower)
    var product = matmul[dtype, n, n, n](lower, upper).to_host()
    var worst = Float64(0)
    for i in range(n):
        for j in range(n):
            var diff = abs(Float64(product[i * n + j]) - _spd_entry(i, j, n))
            if diff > worst:
                worst = diff

    var label = String("cholesky b=") + String(block)
    _row(label, n, ns, Float64(n) ** 3 / 3.0, worst)


def bench_lu[n: Int, block: Int = 24](ctx: DeviceContext) raises:
    var a = _general[n](ctx)

    def work() raises {mut a}:
        var f = lu_factor[dtype, n, False, block](a)
        keep(f.factored.buffer.unsafe_ptr())

    var ns = (
        run(
            work, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9
    )

    # One factorization, one solve, and the residual of the original
    # system -- the only check that exercises both triangular halves and
    # the pivot order together.
    var factored = lu_factor[dtype, n, False, block](a)
    var b = _ramp[n](ctx, 3)
    var x = factored.solve(b)
    var residual = matvec(a, x).to_host()
    var rhs = b.to_host()
    var worst = Float64(0)
    for i in range(n):
        var diff = abs(Float64(residual[i]) - Float64(rhs[i]))
        if diff > worst:
            worst = diff

    var label = String("lu_factor b=") + String(block)
    _row(label, n, ns, 2.0 * Float64(n) ** 3 / 3.0, worst)


def bench_solve[n: Int, block: Int = 24](ctx: DeviceContext) raises:
    var a = _general[n](ctx)
    var b = _ramp[n](ctx, 5)

    def work() raises {mut a, mut b}:
        var x = solve[dtype, n, False, block](a, b)
        keep(x.buffer.unsafe_ptr())

    var ns = (
        run(
            work, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9
    )

    var x = solve[dtype, n, False, block](a, b)
    var residual = matvec(a, x).to_host()
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

    def work() raises {mut a}:
        var f = qr_factor[dtype, m, n, False, block](a)
        keep(f.factored.buffer.unsafe_ptr())

    var ns = (
        run(
            work, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9
    )

    var factored = qr_factor[dtype, m, n, False, block](a)
    var r = factored.r()
    var q = factored.q()
    var product = matmul[dtype, m, n, n](q, r).to_host()
    var worst = Float64(0)
    for i in range(m):
        for j in range(n):
            var diff = abs(
                Float64(product[i * n + j]) - _general_entry(i, j, m)
            )
            if diff > worst:
                worst = diff

    # Householder QR of an `m x n` matrix, LAPACK's `geqrf` count.
    var flops = 2.0 * Float64(n) ** 2 * (Float64(m) - Float64(n) / 3.0)
    var label = String("qr_factor b=") + String(block)
    _row(label, n, ns, flops, worst)


def bench_blas1[n: Int](ctx: DeviceContext) raises:
    var x = _ramp[n](ctx, 1)
    var y = _ramp[n](ctx, 2)

    # `keep` on each result matters more here than above: three of these
    # four return a scalar the caller drops, which is the shape an
    # optimizer is most likely to fold away entirely.
    def dot_work() raises {mut x, mut y}:
        keep(dot(x, y))

    def nrm2_work() raises {mut x}:
        keep(nrm2(x))

    def asum_work() raises {mut x}:
        keep(asum(x))

    def axpy_work() raises {mut x, mut y}:
        var s = axpy(Scalar[dtype](2.5), x, y)
        keep(s.buffer.unsafe_ptr())

    var dot_ns = (
        run(
            dot_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9
    )
    var nrm2_ns = (
        run(
            nrm2_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9
    )
    var asum_ns = (
        run(
            asum_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9
    )
    var axpy_ns = (
        run(
            axpy_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9
    )

    # `float64` recomputations of all four, on the host, from the same
    # entries. The reductions are the ones worth checking: MAX's tree
    # ordering is not the host loop's, so this bounds the disagreement
    # rather than asserting bit equality.
    var want_dot = Float64(0)
    var want_dot_abs = Float64(0)
    var want_sq = Float64(0)
    var want_abs = Float64(0)
    for i in range(n):
        var xi = Float64((i * 37 + 11) % 17) - 8.0
        var yi = Float64((i * 37 + 22) % 17) - 8.0
        want_dot += xi * yi
        want_dot_abs += abs(xi * yi)
        want_sq += xi * xi
        want_abs += abs(xi)
    var want_nrm2 = sqrt(want_sq)

    var got_dot = Float64(dot(x, y))
    var got_nrm2 = Float64(nrm2(x))
    var got_asum = Float64(asum(x))
    var summed = axpy(Scalar[dtype](2.5), x, y).to_host()
    var worst_axpy = Float64(0)
    for i in range(n):
        var xi = Float64((i * 37 + 11) % 17) - 8.0
        var yi = Float64((i * 37 + 22) % 17) - 8.0
        var diff = abs(Float64(summed[i]) - (2.5 * xi + yi))
        if diff > worst_axpy:
            worst_axpy = diff

    # The three reductions report error relative to the quantity that
    # bounds a floating-point sum -- `sum |x_i y_i|` for `dot`, the value
    # itself for the two positive sums -- rather than to the result, which
    # for a cancelling dot product would divide by nearly nothing.
    _band_row(
        "dot", n, dot_ns, 2 * n * 4, abs(got_dot - want_dot) / want_dot_abs
    )
    _band_row("nrm2", n, nrm2_ns, n * 4, abs(got_nrm2 - want_nrm2) / want_nrm2)
    _band_row("asum", n, asum_ns, n * 4, abs(got_asum - want_abs) / want_abs)
    _band_row("axpy", n, axpy_ns, 3 * n * 4, worst_axpy)


def main() raises:
    var ctx = DeviceContext(api="cpu")
    print("dtype =", dtype, " target = cpu")

    print()
    print("Factorizations (ms is per call, GFLOP/s from the LAPACK count)")
    print("op\tn\tms\tGFLOP/s\tmax |residual|")
    bench_gemm[128](ctx)
    bench_gemm[256](ctx)
    bench_gemm[512](ctx)
    bench_gemm[1024](ctx)
    bench_cholesky[128](ctx)
    bench_cholesky[256](ctx)
    bench_cholesky[512](ctx)
    bench_cholesky[1024](ctx)
    bench_lu[128](ctx)
    bench_lu[256](ctx)
    bench_lu[512](ctx)
    bench_lu[1024](ctx)
    bench_solve[128](ctx)
    bench_solve[256](ctx)
    bench_solve[512](ctx)
    bench_solve[1024](ctx)
    bench_qr[128, 128](ctx)
    bench_qr[256, 256](ctx)
    bench_qr[512, 512](ctx)
    bench_qr[1024, 1024](ctx)

    print()
    print("BLAS-1 (us is per call, GB/s over the traffic the op must move)")
    print("op\tn\tus\tGB/s\terror")
    bench_blas1[1 << 16](ctx)
    bench_blas1[1 << 20](ctx)
    bench_blas1[1 << 24](ctx)
    bench_blas1[1 << 26](ctx)

    print()
    print("Block size, at two sizes because the best block is not the same")
    print("op\tn\tms\tGFLOP/s\tmax |residual|")
    bench_cholesky[512, 32](ctx)
    bench_cholesky[512, 48](ctx)
    bench_cholesky[512, 64](ctx)
    bench_cholesky[1024, 32](ctx)
    bench_cholesky[1024, 48](ctx)
    bench_cholesky[1024, 64](ctx)
    bench_lu[512, 16](ctx)
    bench_lu[512, 24](ctx)
    bench_lu[512, 32](ctx)
    bench_lu[1024, 16](ctx)
    bench_lu[1024, 24](ctx)
    bench_lu[1024, 32](ctx)
    bench_qr[512, 512, 4](ctx)
    bench_qr[512, 512, 16](ctx)
    bench_qr[1024, 1024, 4](ctx)
    bench_qr[1024, 1024, 16](ctx)
