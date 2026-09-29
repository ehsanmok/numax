"""`solve_ivp` with `t_eval`, `dense_output` and a terminal event over a
device-resident state at `gpu=True`: the stages, the interpolation at the
requested times, the event's Brent iterations and the dense history all
stay on the device. `y' = -2 y` from 512 starts, against `y0 exp(-2 t)`,
stopped when the mean falls through a threshold."""

from std.math import exp, log
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
)

from max.gpu.host import DeviceContext

from numax.core.ops import multiply
from numax.core.tensor import Static
from numax.integrate import solve_ivp
from numax.stats import mean

comptime f32 = DType.float32
comptime n = 512


def _decay(
    t: Scalar[f32], y: Static[f32, n], ctx: DeviceContext
) raises -> Static[f32, n]:
    return multiply[gpu=True](y, Scalar[f32](-2.0))


def _mean_below_one(
    t: Scalar[f32], y: Static[f32, n], ctx: DeviceContext
) raises -> Scalar[f32]:
    return mean[gpu=True](y) - Scalar[f32](1.0)


def test_recording_solve_ivp_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var starts = List[Scalar[f32]](capacity=n)
    var total = 0.0
    for i in range(n):
        var v = Float32(i % 17) * 0.25 + 0.5
        starts.append(v)
        total += Float64(v)
    var times = Static[f32, 4]([0.0, 0.1, 0.25, 0.5], cpu)
    var r = solve_ivp[
        f=_decay,
        event=_mean_below_one,
        terminal=True,
        direction=-1,
        dense_output=True,
        gpu=True,
    ](0.0, Static[f32, n](starts.copy(), gpu), 1.0, times, rtol=1e-6, atol=1e-8)
    assert_false(r.y.on_host())
    assert_equal(r.status, 1)
    # The mean starts at `total / n` and decays as `exp(-2 t)`.
    var crossing = 0.5 * log(total / Float64(n))
    assert_equal(len(r.t_events), 1)
    assert_almost_equal(r.t_events[0], crossing, atol=1e-4)
    var y = r.y.to_host()
    for j in range(len(r.t)):
        var decay = Float32(exp(-2.0 * r.t[j]))
        for i in range(0, n, 37):
            assert_almost_equal(
                y[i * len(r.t) + j], starts[i] * decay, atol=1e-4, rtol=1e-4
            )
    var mid = r.sol(0.3).to_host()
    for i in range(0, n, 37):
        assert_almost_equal(
            mid[i], starts[i] * Float32(exp(-0.6)), atol=1e-4, rtol=1e-4
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
