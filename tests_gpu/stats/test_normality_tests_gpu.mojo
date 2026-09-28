"""`normaltest` and `jarque_bera` at `gpu=True`, against the host, at
`float32`: the moments are device sums, so the statistics agree to the
`float32` precision of those sums."""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import jarque_bera, normaltest

comptime f32 = DType.float32
comptime n = 2000


def _x(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(sin(0.7 * Float64(i)) * 2 + 0.001 * Float64(i)))
    return Static[f32, n](values^, ctx)


def test_normality_tests_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = normaltest[gpu=True](_x(gpu))
    var h = normaltest(_x(cpu))
    assert_almost_equal(d.statistic, h.statistic, rtol=1e-3)
    var dj = jarque_bera[gpu=True](_x(gpu))
    var hj = jarque_bera(_x(cpu))
    assert_almost_equal(dj.statistic, hj.statistic, rtol=1e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
