"""Random initial conditions for an ODE ensemble, on CPU and GPU.

`examples/advanced/ode.mojo` spreads its 1024 trajectories across a fixed
`linspace`-shaped grid of initial conditions -- deterministic, but not what
a Monte Carlo ensemble actually wants. This example closes that gap two
ways:

- **CPU**: `numax.stats.uniform` draws the initial conditions directly
  into a `Tensor`, reproducibly under a fixed `seed`, then
  `numax.integrate.array.rk4` integrates every trajectory via `numax.core.tensor.map` at
  native SIMD width -- no different from `ode.mojo`'s own CPU path once the
  initial conditions exist.
- **GPU**: the same call with `gpu=True` and a device context,
  `uniform[dtype, n, gpu=True](-2, 2, ctx=gpu)`, draws the initial
  conditions *on-device*, one thread per element, with no host round
  trip: `numax.stats.random` fills from `std.random.philox.Random` seeded
  per element from one scalar seed, so every thread's draw is independent
  and the whole ensemble is reproducible. The same `Generator(seed)` on
  the host produces the identical tensor, bit for bit.

Both ensembles integrate the same equation. The initial conditions are
checked element for element -- the host `Generator` and the device fill
share one stream -- and the final states in distribution (sample mean),
since the RK4 steps round differently on the two processors.
"""

from max.gpu.host import DeviceContext

from numax import Plain, Static
from numax.core.numeric import FloatLike
from numax.integrate.array import rk4
from numax.stats import Generator, uniform
from numax.core.tensor import map

comptime dtype = DType.float32
comptime n = 4096
comptime num_steps = 32
comptime t_final = 1.0
comptime rng_seed = UInt64(2026)


def cooling[U: FloatLike](t: U, y: U) -> U:
    """The same Newton-cooling-toward-a-drifting-ambient equation
    `examples/advanced/ode.mojo` uses -- the point here is the initial
    conditions, not the equation."""
    return -(U.constant(1.5) * (y - t.sin()))


def trajectory_step[w: Int](y0: SIMD[dtype, w]) -> SIMD[dtype, w]:
    comptime P = Plain[dtype, w]
    return rk4[f=cooling, num_steps=num_steps](
        P.constant(0.0), P(y0), P.constant(t_final)
    ).v


def sample_mean(xs: List[Scalar[dtype]]) -> Float64:
    var total = Float64(0)
    for i in range(len(xs)):
        total += Float64(xs[i])
    return total / Float64(len(xs))


def main() raises:
    # --- CPU: numax.stats draws the initial conditions ---
    var cpu = DeviceContext(api="cpu")
    var rng = Generator(seed=Int(rng_seed))
    var y0_cpu = rng.uniform[dtype, n](-2, 2, ctx=cpu)

    comptime Ensemble = Static[dtype, n]
    var yt_cpu = Ensemble(cpu)
    map[step=trajectory_step](y0_cpu.view(), yt_cpu.view())

    print("CPU ensemble:", n, "trajectories, initial conditions ~ U(-2, 2)")
    print("  sample mean of y0:  ", sample_mean(y0_cpu.to_host()))
    print("  sample mean of y(1):", sample_mean(yt_cpu.to_host()))
    print()

    # --- GPU: the same draw, filled on the device ---
    var ctx = DeviceContext()
    print("GPU API:", ctx.api())

    var device_rng = Generator(seed=Int(rng_seed))
    var y0_gpu = device_rng.uniform[dtype, n, gpu=True](-2, 2, ctx=ctx)
    var yt_gpu = Ensemble(ctx)

    comptime block_size = 256
    comptime num_blocks = (n + block_size - 1) // block_size
    ctx.enqueue_function[
        map[LayoutType=Ensemble.LayoutType, step=trajectory_step, gpu=True]
    ](y0_gpu.view(), yt_gpu.view(), grid_dim=num_blocks, block_dim=block_size)
    ctx.synchronize()

    var y0_gpu_host = y0_gpu.to_host()
    var y0_cpu_host = y0_cpu.to_host()
    var identical = 0
    for i in range(n):
        if y0_gpu_host[i] == y0_cpu_host[i]:
            identical += 1
    print(
        "initial conditions identical on host and device:", identical, "of", n
    )
    var yt_gpu_host = yt_gpu.to_host()

    print("GPU ensemble:", n, "trajectories, initial conditions ~ U(-2, 2)")
    print("  sample mean of y0:  ", sample_mean(y0_gpu_host))
    print("  sample mean of y(1):", sample_mean(yt_gpu_host))
    print()
    print(
        "The initial conditions are one Philox stream read on two"
        " processors, so they agree bit for bit; the integrated states agree"
        " in distribution, since RK4 rounds differently on each."
    )
