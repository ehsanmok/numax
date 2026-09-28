"""`solve_ivp` over a device-resident state at `gpu=True`: every stage and
the step controller's error ratio run on the device, one scalar crossing
back per step. `y' = -2 y` from 512 different starts, against the exact
`y0 exp(-2 t)`."""

from std.math import exp
from std.testing import TestSuite, assert_almost_equal, assert_true

from max.gpu.host import DeviceContext

from numax.core.ops import multiply
from numax.core.tensor import Static
from numax.integrate import solve_ivp

comptime f32 = DType.float32
comptime n = 512


def _decay(
    t: Scalar[f32], y: Static[f32, n], ctx: DeviceContext
) raises -> Static[f32, n]:
    return multiply[gpu=True](y, Scalar[f32](-2.0))


def test_solve_ivp_on_the_device() raises:
    var gpu = DeviceContext()
    var starts = List[Scalar[f32]](capacity=n)
    for i in range(n):
        starts.append(Float32(i % 17) * 0.25 + 0.5)
    var result = solve_ivp[f=_decay, gpu=True](
        0.0, Static[f32, n](starts.copy(), gpu), 1.0, rtol=1e-5, atol=1e-7
    )
    assert_true(result.converged)
    assert_true(not result.y.on_host())
    var got = result.y.to_host()
    var decay = Float32(exp(-2.0))
    for i in range(n):
        assert_almost_equal(got[i], starts[i] * decay, atol=1e-4, rtol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
