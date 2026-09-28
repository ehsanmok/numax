"""`lfilter` from an initial state at `gpu=True`, against the host, at
`float32`: the device recurrence from `zi`, and the final state rebuilt
from the last few samples, both to `float32` rounding of the host's."""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.signal import butter, lfilter

comptime f32 = DType.float32
comptime n = 3000


def _x(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(sin(0.03 * Float64(i))))
    return Static[f32, n](values^, ctx)


def test_lfilter_with_zi_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var dtf = butter[f32, 4](0.2, ctx=gpu)
    var htf = butter[f32, 4](0.2, ctx=cpu)
    var d = lfilter[gpu=True](
        dtf.b, dtf.a, _x(gpu), Static[f32, 4]([0.5, -0.2, 0.1, 0.3], gpu)
    )
    assert_false(d.y.on_host())
    var h = lfilter(
        htf.b, htf.a, _x(cpu), Static[f32, 4]([0.5, -0.2, 0.1, 0.3], cpu)
    )
    var dy = d.y.to_host()
    var hy = h.y.to_host()
    for i in range(n):
        assert_almost_equal(dy[i], hy[i], atol=1e-4)
    var dz = d.zf.to_host()
    var hz = h.zf.to_host()
    for i in range(4):
        assert_almost_equal(dz[i], hz[i], atol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
