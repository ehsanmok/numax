"""`kendalltau` at `gpu=True`, against the host.

The device counts the same pairs and the same tie groups as integers and
shares the host's finish, so tau and the p-value must equal the host's
exactly, with and without ties.
"""

from std.math import sin
from std.testing import TestSuite, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import kendalltau

comptime f32 = DType.float32
comptime n = 300


def _x(ctx: DeviceContext, tied: Bool) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        if tied:
            values.append(Float32((i * 53) % 13))
        else:
            values.append(Float32(sin(Float64(i) * 3.1)) + Float32(i) * 0.01)
    return Static[f32, n](ctx, values^)


def _y(ctx: DeviceContext, tied: Bool) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        if tied:
            values.append(Float32((i * 29) % 7) + Float32(i % 3))
        else:
            values.append(Float32(sin(Float64(i) * 0.7)) + Float32(i) * 0.02)
    return Static[f32, n](ctx, values^)


def test_kendalltau_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    for tied in [False, True]:
        var d = kendalltau[gpu=True](_x(gpu, tied), _y(gpu, tied))
        var h = kendalltau(_x(cpu, tied), _y(cpu, tied))
        assert_equal(d.statistic, h.statistic)
        assert_equal(d.pvalue, h.pvalue)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
