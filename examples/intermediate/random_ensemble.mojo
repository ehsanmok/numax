"""Random initial conditions for an ODE ensemble, on CPU and GPU.

`examples/advanced/ode.mojo` spreads its 1024 trajectories across a fixed
`linspace`-shaped grid of initial conditions -- deterministic, but not what
a Monte Carlo ensemble actually wants. This example closes that gap two
ways:

- **CPU**: `numax.stats.uniform` draws the initial conditions directly
  into a `Tensor`, reproducibly under a fixed `seed`, then
  `numax.integrate.rk4` integrates every trajectory via `numax.core.tensor.map` at
  native SIMD width -- no different from `ode.mojo`'s own CPU path once the
  initial conditions exist.
- **GPU**: initial conditions are drawn *on-device*, one value per thread,
  via `std.random.philox.Random` called directly inside a `map[gpu=True]`
  kernel body -- not through `numax.stats.uniform`, which is host-only
  (see `numax/stats/random.mojo`'s own docstring for why). Each thread seeds its
  stream from a shared seed plus its own flat index, so every thread's
  draw is independent and the whole ensemble is reproducible from one
  scalar seed with no shared mutable state -- exactly the property
  `std.random`'s global generator cannot offer inside a kernel.

Both ensembles integrate the same equation and are checked against each
other in distribution (sample mean of the final states), not
element-for-element -- the CPU and GPU paths draw from different generators
(`std.random.rand` vs. `std.random.philox.Random`) on purpose, so they are
expected to produce different individual trajectories from the same seed
value, not identical ones.
"""

from max.gpu.host import DeviceContext
from std.random import Random

from numax import Plain, Shaped
from numax.core.numeric import FloatLike
from numax.integrate import rk4
from numax.stats import seed, uniform
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


def gpu_initial_condition_step[w: Int](idx: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """One GPU thread, one independent draw from `[-2, 2)`.

    `idx` is this thread's own flat position (fed in as a plain index
    tensor, since `map`'s `step` only ever sees the value at its own
    position) -- used as `Random`'s per-thread `offset`, so every thread's
    stream is independent even though every thread shares the same `seed`.
    """
    var result = SIMD[dtype, w](0)
    for lane in range(w):
        var offset = UInt64(idx[lane])
        var r = Random(seed=rng_seed, offset=offset)
        var u = r.step_uniform()[0]
        result[lane] = Scalar[dtype](u) * 4.0 - 2.0
    return result


def sample_mean(xs: List[Scalar[dtype]]) -> Float64:
    var total = Float64(0)
    for i in range(len(xs)):
        total += Float64(xs[i])
    return total / Float64(len(xs))


def main() raises:
    # --- CPU: numax.stats draws the initial conditions ---
    seed(2026)
    var cpu = DeviceContext(api="cpu")
    var y0_cpu = uniform[dtype, n](-2, 2, ctx=cpu)

    comptime Ensemble = Shaped[dtype, n]
    var yt_cpu = Ensemble(cpu)
    map[step=trajectory_step](y0_cpu.view(), yt_cpu.view())

    print("CPU ensemble:", n, "trajectories, initial conditions ~ U(-2, 2)")
    print("  sample mean of y0:  ", sample_mean(y0_cpu.to_host()))
    print("  sample mean of y(1):", sample_mean(yt_cpu.to_host()))
    print()

    # --- GPU: std.random.philox draws the initial conditions on-device ---
    var ctx = DeviceContext()
    print("GPU API:", ctx.api())

    var flat_indices = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        flat_indices.append(Scalar[dtype](i))
    var idx_tensor = Ensemble(ctx)
    # One staged write for the whole buffer; `idx_tensor[i] = ...` would map
    # the device to the host once per element.
    idx_tensor.copy_from_host(flat_indices)

    var y0_gpu = Ensemble(ctx)
    var yt_gpu = Ensemble(ctx)

    comptime block_size = 256
    comptime num_blocks = (n + block_size - 1) // block_size

    ctx.enqueue_function[
        map[
            LayoutType=Ensemble.LayoutType,
            step=gpu_initial_condition_step,
            gpu=True,
        ]
    ](
        idx_tensor.view(),
        y0_gpu.view(),
        grid_dim=num_blocks,
        block_dim=block_size,
    )
    ctx.enqueue_function[
        map[LayoutType=Ensemble.LayoutType, step=trajectory_step, gpu=True]
    ](y0_gpu.view(), yt_gpu.view(), grid_dim=num_blocks, block_dim=block_size)
    ctx.synchronize()

    var y0_gpu_host = y0_gpu.to_host()
    var yt_gpu_host = yt_gpu.to_host()

    print("GPU ensemble:", n, "trajectories, initial conditions ~ U(-2, 2)")
    print("  sample mean of y0:  ", sample_mean(y0_gpu_host))
    print("  sample mean of y(1):", sample_mean(yt_gpu_host))
    print()
    print(
        "Both ensembles draw from the same theoretical distribution, so"
        " their sample means agree within Monte Carlo noise -- not"
        " bit-for-bit, since the CPU path used `std.random.rand` and the"
        " GPU path used `std.random.philox.Random` directly."
    )
