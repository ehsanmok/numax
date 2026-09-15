"""4096 Cholesky solves at once, one 4x4 system per lane -- and every
system's sensitivity from the same factorization.

A batch of small dense solves is the shape a `Tensor` kernel cannot
express: `solve` over a 4x4 is a handful of flops against a launch and two
allocations, so doing it 4096 times means 4096 launches. `numax.linalg.array`
already has the answer for one problem -- a register-resident,
`FloatLike`-generic `cholesky` that fits inside a GPU thread -- and
`numax.core.tensor.map_blocks` is what hands it a whole batch: `step`
receives one lane's entire problem and returns its entire answer, so the
factorization runs *inside* the lane and the batch is one launch.

The systems are `A x = b` with `A = M M^T + 4 I` for a hash-generated `M`,
which is symmetric positive definite by construction, so no pivoting is
needed and the tier-1 `cholesky` is exact for it rather than merely
floored. Two questions come out of the same 16 numbers:

- at `Plain`, `x` itself, checked by `max |A x - b|` read back on the host;
- at `Dual`, `dx/dA00` -- the derivative of a factorization *and* its two
  substitutions, with no adjoint rule written anywhere -- checked against a
  central finite difference taken on the host in `float64`.

**Storage is structure-of-arrays, `(k, batch)`.** Row `j` of the input holds
entry `j` of every problem: the 16 entries of `A` row-major, then the 4 of
`b`, so `k_in = 20` and `k_out = 4`. That is what makes one lane load `w`
consecutive addresses and adjacent GPU threads read adjacent memory. A
caller holding the batch the other way round -- one problem per row --
calls `numax.core.array.transpose` once, on either target, instead of
paying a strided gather per block per launch.

Two limits, visible rather than hidden:

- **`n` cannot grow far.** One lane carries the whole problem in registers:
  `k_in + k_out` values at `Plain`, twice that at `Dual`, plus the `n*n`
  factor and the two substitution vectors the kernel builds on top. At
  `n = 4` that is 40 floats per lane at `Dual`; the cost is quadratic in
  `n` and the register file is the ceiling, the same one
  `quantum_well.mojo` hits at a 24x24 eigensolve. Past it the answer is
  the `Tensor` tier, one problem at a time.
- **The dtype is `float32` because Metal has no `double`.** A `float64`
  kernel body is rejected by Apple's compiler before it runs, so the finite
  difference -- host code, not a kernel -- is the only `float64` here.

Run with: `pixi run example-batched-solve` (needs a GPU; the CPU half runs
anywhere).
"""

from max.gpu.host import DeviceContext

from numax.core.tensor import map_blocks
from numax.linalg.array import cholesky, cholesky_solve
from numax.prelude import *

comptime dtype = f32
comptime n = 4
comptime k_in = n * n + n
comptime k_out = n
comptime batch = 4096
comptime Problems = Static[dtype, k_in, batch]
comptime Solutions = Static[dtype, k_out, batch]


def entry(p: Int, j: Int) -> Float64:
    """One entry of problem `p`'s generator matrix `M`, from a hash -- a
    deterministic batch with no two systems alike."""
    var h = (p * n * n + j) * 2654435761 % 100003
    return Float64(h) / 100003.0 - 0.5


def problems(ctx: DeviceContext) raises -> Problems:
    """The batch, packed `(k_in, batch)`: `A = M M^T + 4 I` row-major in
    rows `0..15`, then `b` in rows `16..19`."""
    var values = List[Scalar[dtype]](length=k_in * batch, fill=0)
    for p in range(batch):
        for r in range(n):
            for c in range(n):
                var total = 4.0 if r == c else 0.0
                for k in range(n):
                    total += entry(p, r * n + k) * entry(p, c * n + k)
                values[(r * n + c) * batch + p] = Scalar[dtype](total)
            values[(n * n + r) * batch + p] = Scalar[dtype](
                1.0 + 0.25 * Float64(r)
            )
    return Problems(ctx, values^)


def solve_step[
    w: Int
](block: Array[SIMD[dtype, w], k_in]) -> Array[SIMD[dtype, w], k_out]:
    """`x = A^-1 b` for `w` independent systems at once: the block wrapped
    into `Plain`, factored, and solved inside the lane."""
    comptime P = Plain[dtype, w]
    var a = Array[P, n * n](uninitialized=True)
    comptime for j in range(n * n):
        a[j] = P(block[j])
    var b = Array[P, n](uninitialized=True)
    comptime for j in range(n):
        b[j] = P(block[n * n + j])

    var x = cholesky_solve[P, n](cholesky[P, n](a^), b^)
    var out = Array[SIMD[dtype, w], k_out](uninitialized=True)
    comptime for j in range(n):
        out[j] = x[j].v
    return out^


def sensitivity_step[
    w: Int
](block: Array[SIMD[dtype, w], k_in]) -> Array[SIMD[dtype, w], k_out]:
    """`dx/dA00` for the same systems -- the same two calls, at `Dual` with
    the derivative seeded on the one entry of `A` being varied."""
    comptime P = Plain[dtype, w]
    comptime D = Dual[P]
    var a = Array[D, n * n](uninitialized=True)
    comptime for j in range(n * n):
        a[j] = D(P(block[j]), P.constant(1.0 if j == 0 else 0.0))
    var b = Array[D, n](uninitialized=True)
    comptime for j in range(n):
        b[j] = D(P(block[n * n + j]), P.constant(0.0))

    var x = cholesky_solve[D, n](cholesky[D, n](a^), b^)
    var out = Array[SIMD[dtype, w], k_out](uninitialized=True)
    comptime for j in range(n):
        out[j] = x[j].deriv.v
    return out^


def widest_gap(a: Solutions, b: Solutions) raises -> Float64:
    """The largest absolute difference between two batches of answers."""
    var xs = a.to_host()
    var ys = b.to_host()
    var widest = 0.0
    for i in range(k_out * batch):
        var gap = Float64(abs(xs[i] - ys[i]))
        if gap > widest:
            widest = gap
    return widest


def worst_residual(a: Problems, x: Solutions) raises -> Float64:
    """`max |A x - b|` over the whole batch, on the host."""
    var lhs = a.to_host()
    var rhs = x.to_host()
    var worst = 0.0
    for p in range(batch):
        for r in range(n):
            var total = 0.0
            for c in range(n):
                total += Float64(lhs[(r * n + c) * batch + p]) * Float64(
                    rhs[c * batch + p]
                )
            var gap = abs(total - Float64(lhs[(n * n + r) * batch + p]))
            if gap > worst:
                worst = gap
    return worst


def host_solve(a: List[Float64], b: List[Float64]) -> List[Float64]:
    """The same solve on the host in `float64`, one problem, for the finite
    difference -- the `Array` tier again, at a different conformer."""
    comptime H = Plain[DType.float64]
    var matrix = Array[H, n * n](uninitialized=True)
    for j in range(n * n):
        matrix[j] = H.constant(a[j])
    var vector = Array[H, n](uninitialized=True)
    for j in range(n):
        vector[j] = H.constant(b[j])

    var x = cholesky_solve[H, n](cholesky[H, n](matrix^), vector^)
    var out = List[Float64](length=n, fill=0.0)
    for j in range(n):
        out[j] = Float64(x[j].v)
    return out^


def worst_slope_gap(a: Problems, slopes: Solutions) raises -> Float64:
    """The widest disagreement between the `Dual` derivative and a central
    difference in `A00`, over a sample of the batch."""
    var lhs = a.to_host()
    var ds = slopes.to_host()
    comptime step = 1.0e-4
    var worst = 0.0
    for p in range(0, batch, 337):
        var matrix = List[Float64](length=n * n, fill=0.0)
        var vector = List[Float64](length=n, fill=0.0)
        for j in range(n * n):
            matrix[j] = Float64(lhs[j * batch + p])
        for j in range(n):
            vector[j] = Float64(lhs[(n * n + j) * batch + p])

        matrix[0] += step
        var up = host_solve(matrix, vector)
        matrix[0] -= 2.0 * step
        var down = host_solve(matrix, vector)

        for j in range(n):
            var fd = (up[j] - down[j]) / (2.0 * step)
            var gap = abs(fd - Float64(ds[j * batch + p]))
            if gap > worst:
                worst = gap
    return worst


def main() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    print(batch, "SPD 4x4 systems, on", gpu.api(), "and on the CPU")

    var host_a = problems(cpu)
    var device_a = problems(gpu)
    var host_x = Solutions._uninitialized(cpu)
    var device_x = Solutions._uninitialized(gpu)
    var host_slopes = Solutions._uninitialized(cpu)
    var device_slopes = Solutions._uninitialized(gpu)

    map_blocks[step=solve_step](host_a.view(), host_x.view(), cpu)
    map_blocks[step=sensitivity_step](host_a.view(), host_slopes.view(), cpu)
    map_blocks[step=solve_step, gpu=True](device_a.view(), device_x.view(), gpu)
    map_blocks[step=sensitivity_step, gpu=True](
        device_a.view(), device_slopes.view(), gpu
    )

    print("  max |A x - b|      CPU", worst_residual(host_a, host_x))
    print("  max |A x - b|      GPU", worst_residual(device_a, device_x))
    print(
        "  dx/dA00 vs central difference, widest gap:",
        worst_slope_gap(host_a, host_slopes),
    )
    print("  widest GPU/CPU gap, x      :", widest_gap(device_x, host_x))
    print(
        "  widest GPU/CPU gap, dx/dA00:",
        widest_gap(device_slopes, host_slopes),
    )

    _ = host_a^
    _ = device_a^
    _ = host_x^
    _ = device_x^
    _ = host_slopes^
    _ = device_slopes^
