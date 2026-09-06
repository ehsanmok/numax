"""The ground state of a quantum harmonic oscillator, four ways from one
function.

`-psi''/2 + w^2 x^2 psi / 2 = E psi`, in units where `hbar = m = 1` so that
`E0 = w/2`, is a symmetric tridiagonal eigenproblem on a grid of `n` points, so
`ground_energy` builds the matrix and takes the lowest eigenvalue of
`numax.linalg.eigh`. It is written once against
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

Three real limits are visible here rather than hidden:

- **The error is the grid's, not the library's.** 24 points over `[-4, 4]`
  put `E0(w=1)` at about `0.496` against the continuum's `0.5`. The
  derivative is exact for the *discretized* operator, and inherits the same
  offset from the continuum answer. A finer grid closes it at a cost below.
- **`n` cannot grow far.** The matrix is an `Array[T, n*n]` in registers,
  which is exactly what lets the eigensolve run inside a GPU thread, and
  also what caps it: `n = 32` roughly quadruples the build time for a
  0.2 percent accuracy gain, and each thread here already holds 4.6 KB.
- **`eigh` returns eigenvalues in no order**, since sorting is
  data-dependent and this stays tier 1. The minimum comes out of a fold with
  the branchless `min_of` instead, which `Dual` differentiates through to
  whichever eigenvalue actually won.
"""

from max.gpu.host import DeviceContext

from numax.core.numeric import min_of
from numax.core.tensor import map
from numax.prelude import *

comptime P = Plain[f64]
comptime n = 24  # grid points over [-4, 4]
comptime dx = 8.0 / (n - 1)
comptime kinetic = 1.0 / (dx * dx)
comptime sweep = 256
comptime SweepLayout = Shaped[f64, sweep].LayoutType


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


def step[w: Int](ws: SIMD[f64, w]) -> SIMD[f64, w]:
    return ground_energy(Plain[f64, w](ws)).v


def main() raises:
    print("one well, one function")
    print("  E0(w=1)   =", ground_energy(P.constant(1.0)), " continuum 0.5")
    var slope = ground_energy(Dual[P].seed(1.0)).deriv.copy()
    print("  dE0/dw    =", slope, " continuum 0.5")
    print(
        "  E0 = 1 at =", newton[f=detuning](P.constant(1.0)), " continuum 2.0"
    )

    print("\n256 wells at once")
    var gpu = DeviceContext()
    var device_ws = linspace[sweep, f64](0.5, 2.0, ctx=gpu)
    var device_es = zeros[f64, sweep](gpu)
    gpu.enqueue_function[map[LayoutType=SweepLayout, step=step, gpu=True]](
        device_ws.view(), device_es.view(), grid_dim=4, block_dim=64
    )
    gpu.synchronize()

    var cpu = DeviceContext(api="cpu")
    var host_ws = linspace[sweep, f64](0.5, 2.0, ctx=cpu)
    var host_es = zeros[f64, sweep](cpu)
    map[step=step](host_ws.view(), host_es.view())

    var from_gpu = device_es.to_host()
    var from_cpu = host_es.to_host()
    var widest = 0.0
    for i in range(sweep):
        var gap = Float64(abs(from_gpu[i] - from_cpu[i]))
        if gap > widest:
            widest = gap
    print("  GPU E0(w=2) =", from_gpu[sweep - 1], " continuum 1.0")
    print("  CPU E0(w=2) =", from_cpu[sweep - 1])
    print("  widest GPU/CPU gap over the sweep:", widest)
