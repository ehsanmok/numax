"""`cumulative_simpson` and `romb` at `gpu=True`, against the host, at
`float32`: the shares are one launch and the device scan accumulates
them; Romberg's level sums are one launch, a lane per level."""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.integrate import cumulative_simpson, romb

comptime f32 = DType.float32
comptime n = 257


def _y(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(sin(Float32(i) * 0.05) + 0.1 * Float32(i % 3))
    return Static[f32, n](values^, ctx)


def _x(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(i) * 0.05 + 0.01 * sin(Float32(i)))
    return Static[f32, n](values^, ctx)


def test_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = cumulative_simpson[gpu=True](_y(gpu), 0.05)
    assert_false(d.on_host())
    var got = d.to_host()
    var want = cumulative_simpson(_y(cpu), 0.05).to_host()
    for i in range(n - 1):
        assert_almost_equal(got[i], want[i], atol=1e-4, rtol=1e-5)
    var dx = cumulative_simpson[initial=True, gpu=True](
        _y(gpu), _x(gpu)
    ).to_host()
    var hx = cumulative_simpson[initial=True](_y(cpu), _x(cpu)).to_host()
    for i in range(n):
        assert_almost_equal(dx[i], hx[i], atol=1e-4, rtol=1e-5)
    assert_almost_equal(
        romb[gpu=True](_y(gpu), 0.05), romb(_y(cpu), 0.05), atol=1e-4, rtol=1e-5
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
