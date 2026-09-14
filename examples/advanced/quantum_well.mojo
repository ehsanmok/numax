"""The ground state of a quantum harmonic oscillator, four ways from one
function -- and all four again, one well per GPU thread.

`-psi''/2 + w^2 x^2 psi / 2 = E psi`, in units where `hbar = m = 1` so that
`E0 = w/2`, is a symmetric tridiagonal eigenproblem on a grid of `n` points, so
`ground_energy` builds the matrix and takes the lowest eigenvalue of
`numax.linalg.array.eigh`. It is written once against
`FloatLike` and never mentions a dtype, a device, or a derivative rule, which
is what lets the same function answer four different questions here:

- at `Plain`, the ground-state energy itself;
- at `Dual`, `dE0/dw` alongside it -- the Hellmann-Feynman theorem, with no
  perturbation theory written anywhere, checked against the analytic `1/2`;
- inside `numax.optimize.newton`, which frequency puts the ground state at a
  target energy, using a derivative the caller never supplied;
- inside `numax.core.tensor.map`, a sweep over `w`, on the GPU with the whole
  eigensolve running inside a single thread, and on the CPU across SIMD
  lanes. Both come out bit-identical.

The second half runs the other three answers the same way: `dE0/dw` at
`Dual` for every well in the sweep, and `newton` from 256 starting guesses,
each thread carrying its own chain of `Dual` eigensolves. Nothing about the
kernels changes between the two halves except the type they are called
with; the GPU launch is the same `map` the CPU path uses.

Three real limits are visible here rather than hidden:

- **The error is the grid's, not the library's.** 24 points over `[-4, 4]`
  put `E0(w=1)` at about `0.496` against the continuum's `0.5`. The
  derivative is exact for the *discretized* operator, and inherits the same
  offset from the continuum answer. A finer grid closes it at a cost below.
- **`n` cannot grow far.** The matrix is an `Array[T, n*n]` in registers,
  which is exactly what lets the eigensolve run inside a GPU thread, and
  also what caps it: `n = 32` roughly quadruples the build time for a
  0.2 percent accuracy gain, and each thread here already holds 2.3 KB at
  `Plain` and twice that at `Dual`.
- **`eigh` returns eigenvalues in no order**, since sorting is
  data-dependent and this stays tier 1. The minimum comes out of a fold with
  the branchless `min_of` instead, which `Dual` differentiates through to
  whichever eigenvalue actually won.

The dtype is `float32` because Metal has no `double`: a `float64` kernel
body is rejected by Apple's compiler before it runs, so the same source at
`float64` is a CUDA-only example. Everything here, host and device, uses the
one `dtype` so the bit-for-bit comparison is between equals.
"""

from max.gpu.host import DeviceContext

from numax.core.numeric import min_of
from numax.core.tensor import map
from numax.linalg.array import eigh
from numax.optimize.array import newton
from numax.prelude import *

comptime dtype = f32
comptime P = Plain[dtype]
comptime n = 24  # grid points over [-4, 4]
comptime dx = 8.0 / (n - 1)
comptime kinetic = 1.0 / (dx * dx)
comptime sweep = 256
comptime Sweep = Static[dtype, sweep]


def ground_energy[T: FloatLike](w: T) -> T:
    """Lowest eigenvalue of the discretized Hamiltonian. Tier 1."""
    var H = zeros[T, n * n]()
    var x = T.constant(-4.0)
    for i in range(n):
        H[i * n + i] = T.constant(kinetic) + w * w * x * x / T.constant(2.0)
        if i + 1 < n:
            H[i * n + i + 1] = T.constant(-0.5 * kinetic)
            H[(i + 1) * n + i] = T.constant(-0.5 * kinetic)
        x = x + T.constant(dx)

    var energies = eigh[T, n](H)[0].copy()
    var lowest = energies[0].copy()
    for i in range(1, n):
        lowest = min_of(lowest, energies[i])
    return lowest^


def detuning[T: FloatLike](w: T) -> T:
    """Zero exactly where the ground state sits at `E = 1`. Tier 1."""
    return ground_energy(w) - T.one()


def energy_step[w: Int](ws: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """`E0(w)` for a lane-vector of frequencies: one eigensolve per lane."""
    return ground_energy(Plain[dtype, w](ws)).v


def sensitivity_step[w: Int](ws: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """`dE0/dw` for the same frequencies -- the same eigensolve at `Dual`,
    with the derivative seeded to `1`."""
    comptime Pw = Plain[dtype, w]
    return ground_energy(Dual[Pw](Pw(ws), Pw.constant(1.0))).deriv.v


def inversion_step[w: Int](guesses: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """The frequency that puts `E0` at `1`, by Newton from each lane's own
    starting guess: every iteration is a `Dual` eigensolve, inside the
    thread."""
    return newton[f=detuning](Plain[dtype, w](guesses)).v


def widest_gap(a: Sweep, b: Sweep) raises -> Float64:
    """The largest absolute difference between two sweeps, read back to
    the host."""
    var xs = a.to_host()
    var ys = b.to_host()
    var widest = 0.0
    for i in range(sweep):
        var gap = Float64(abs(xs[i] - ys[i]))
        if gap > widest:
            widest = gap
    return widest


def main() raises:
    print("one well, one function")
    print("  E0(w=1)   =", ground_energy(P.constant(1.0)), " continuum 0.5")
    var slope = ground_energy(Dual[P].seed(1.0)).deriv.copy()
    print("  dE0/dw    =", slope, " continuum 0.5")
    print(
        "  E0 = 1 at =", newton[f=detuning](P.constant(1.0)), " continuum 2.0"
    )

    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    print("\n256 wells at once, on", gpu.api(), "and on the CPU")

    # The frequency sweep, on both devices, and the sweep of Newton
    # starting guesses.
    var device_ws = linspace[sweep, dtype](0.5, 2.0, ctx=gpu)
    var host_ws = linspace[sweep, dtype](0.5, 2.0, ctx=cpu)
    var device_guesses = linspace[sweep, dtype](0.5, 4.0, ctx=gpu)
    var host_guesses = linspace[sweep, dtype](0.5, 4.0, ctx=cpu)

    var device_es = zeros[dtype, sweep](gpu)
    var device_slopes = zeros[dtype, sweep](gpu)
    var device_roots = zeros[dtype, sweep](gpu)
    var host_es = zeros[dtype, sweep](cpu)
    var host_slopes = zeros[dtype, sweep](cpu)
    var host_roots = zeros[dtype, sweep](cpu)

    comptime Layout = Sweep.LayoutType
    gpu.enqueue_function[map[LayoutType=Layout, step=energy_step, gpu=True]](
        device_ws.view(), device_es.view(), grid_dim=4, block_dim=64
    )
    gpu.enqueue_function[
        map[LayoutType=Layout, step=sensitivity_step, gpu=True]
    ](device_ws.view(), device_slopes.view(), grid_dim=4, block_dim=64)
    gpu.enqueue_function[map[LayoutType=Layout, step=inversion_step, gpu=True]](
        device_guesses.view(), device_roots.view(), grid_dim=4, block_dim=64
    )
    gpu.synchronize()

    map[step=energy_step](host_ws.view(), host_es.view())
    map[step=sensitivity_step](host_ws.view(), host_slopes.view())
    map[step=inversion_step](host_guesses.view(), host_roots.view())

    var gpu_es = device_es.to_host()
    var gpu_slopes = device_slopes.to_host()
    var gpu_roots = device_roots.to_host()
    print("  E0(w=2)          GPU", gpu_es[sweep - 1], " continuum 1.0")
    print("  dE0/dw at w=2    GPU", gpu_slopes[sweep - 1], " continuum 0.5")
    print(
        "  E0 = 1 at        GPU",
        gpu_roots[0],
        "from w0 = 0.5,",
        gpu_roots[sweep - 1],
        "from w0 = 4.0",
    )
    print("  widest GPU/CPU gap, E0    :", widest_gap(device_es, host_es))
    print(
        "  widest GPU/CPU gap, dE0/dw:", widest_gap(device_slopes, host_slopes)
    )
    print("  widest GPU/CPU gap, root  :", widest_gap(device_roots, host_roots))
