"""Integrating an ODE ensemble, one GPU thread per trajectory.

A fixed-step integrator has no data-dependent control flow: every
trajectory takes the same number of steps with the same number of stages,
regardless of what the solution does. That makes the whole integration
usable as a `map` kernel body, so an ensemble of initial conditions
parallelizes the obvious way -- one thread, one trajectory, integrated to
completion independently.

The same `step` function runs on both paths here. Storage is a `Tensor` on
either device, and `.view()` is what the kernel takes: the CPU version walks
that view at native SIMD width, so each vector lane is a separate
trajectory; the GPU version launches one thread per element. Neither needed
anything written twice, and the two agree to within float32 rounding.

The example also shows the sensitivity that comes free with the integrator:
seeding the initial condition as a `Dual` returns `dy(t1)/dy0` alongside
`y(t1)`, computed by the same integration rather than by a second pass or a
finite difference.

Run with: `pixi run example-ode` (needs a GPU; the CPU half runs anywhere).
"""

from max.gpu.host import DeviceContext
from std.math import exp as exp_f64

from numax import Dual, FloatLike, Plain, Shaped
from numax.integrate import rk4
from numax.core.tensor import map

comptime dtype = DType.float32
comptime n = 1024
comptime num_steps = 64
comptime t_final = 1.0


def cooling[U: FloatLike](t: U, y: U) -> U:
    """Newton cooling toward an ambient value that itself drifts:
    `dy/dt = -1.5*(y - sin(t))`.

    Nonlinear enough in `t` that no closed form is worth writing, which is
    the point -- the ensemble is what's being demonstrated, not the answer.
    """
    return -(U.constant(1.5) * (y - t.sin()))


def trajectory_step[w: Int](y0: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """One trajectory, integrated to completion, as a `map` kernel body."""
    comptime P = Plain[dtype, w]
    return rk4[f=cooling, num_steps=num_steps](
        P.constant(0.0), P(y0), P.constant(t_final)
    ).v


def sensitivity_step[w: Int](y0: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """`dy(t_final)/dy0` for the same trajectory, from the same
    integrator -- only the type it's called with changed."""
    comptime P = Plain[dtype, w]
    comptime D = Dual[P]
    return rk4[f=cooling, num_steps=num_steps](
        D.constant(0.0), D(P(y0), P.constant(1.0)), D.constant(t_final)
    ).deriv.v


def main() raises:
    var ctx = DeviceContext()
    print("GPU API:", ctx.api())
    print("ensemble:", n, "trajectories,", num_steps, "RK4 steps each")
    print()

    comptime Ensemble = Shaped[dtype, n]

    # Initial conditions spread across [-2, 2].
    var host_y0 = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        host_y0.append(Scalar[dtype](i) * (4.0 / Scalar[dtype](n)) - 2.0)

    var y0 = Ensemble(ctx, host_y0.copy())
    var yt = Ensemble(ctx)
    var dydy0 = Ensemble(ctx)

    comptime block_size = 256
    comptime num_blocks = (n + block_size - 1) // block_size

    ctx.enqueue_function[
        map[
            LayoutType=Ensemble.LayoutType,
            step=trajectory_step,
            gpu=True,
        ]
    ](y0.view(), yt.view(), grid_dim=num_blocks, block_dim=block_size)
    ctx.enqueue_function[
        map[
            LayoutType=Ensemble.LayoutType,
            step=sensitivity_step,
            gpu=True,
        ]
    ](y0.view(), dydy0.view(), grid_dim=num_blocks, block_dim=block_size)
    ctx.synchronize()

    # The same kernel on CPU, at native SIMD width.
    var cpu = DeviceContext(api="cpu")
    var cpu_in = Ensemble(cpu, host_y0.copy())
    var cpu_out = Ensemble(cpu)
    map[step=trajectory_step](cpu_in.view(), cpu_out.view())

    var gpu_yt = yt.to_host()
    var gpu_s = dydy0.to_host()
    var host_yt = cpu_out.to_host()

    var worst = Float64(0.0)
    for i in range(n):
        var diff = abs(Float64(gpu_yt[i]) - Float64(host_yt[i]))
        if diff > worst:
            worst = diff
    print("     y0        y(1) gpu     y(1) cpu    dy(1)/dy0")
    for i in range(0, n, n // 8):
        print(
            "  ",
            host_y0[i],
            "  ",
            gpu_yt[i],
            "  ",
            host_yt[i],
            "  ",
            gpu_s[i],
        )
    print()
    print("worst gpu-vs-cpu difference:", worst)

    # The equation is linear in y, so dy(t)/dy0 = exp(-1.5*t) for every
    # trajectory regardless of where it started -- a closed form the
    # integrator was never told about.
    print(
        "dy(1)/dy0 should be exp(-1.5) =",
        exp_f64(-1.5),
        "for every trajectory",
    )
