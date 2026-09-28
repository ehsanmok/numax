"""`ptp`, `average`, `moment` and `zscore` at `gpu=True`, against the host.
Compositions of device reductions and elementwise launches; float32
tolerances, since the device sums at the tensor's dtype."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
)

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import average, moment, ptp, zscore

comptime f32 = DType.float32
comptime n = 1024


def _data(ctx: DeviceContext, shift: Float32) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32((i * 37) % 53) * 0.1 + shift)
    return Static[f32, n](ctx, values^)


def test_ptp_average_and_moment_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    assert_equal(ptp[gpu=True](_data(gpu, 0)), ptp(_data(cpu, 0)))
    assert_almost_equal(
        average[gpu=True](_data(gpu, 1)), average(_data(cpu, 1)), rtol=1e-5
    )
    assert_almost_equal(
        average[gpu=True](_data(gpu, 1), _data(gpu, 0.5)),
        average(_data(cpu, 1), _data(cpu, 0.5)),
        rtol=1e-5,
    )
    for order in range(5):
        assert_almost_equal(
            moment[gpu=True](_data(gpu, 0), order),
            moment(_data(cpu, 0), order),
            atol=1e-5,
            rtol=1e-4,
        )


def test_zscore_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = zscore[gpu=True](_data(gpu, 3), 1)
    assert_false(d.on_host())
    var got = d.to_host()
    var want = zscore(_data(cpu, 3), 1).to_host()
    for i in range(n):
        assert_almost_equal(got[i], want[i], atol=1e-5, rtol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
