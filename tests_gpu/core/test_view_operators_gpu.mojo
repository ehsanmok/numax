"""`TensorView`'s operators over device memory follow the view to its
device, as `Tensor`'s do, and match the host."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
)

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.core.tensorlike import TensorView

comptime f32 = DType.float32
comptime n = 512


def _ramp(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(i % 13) * 0.5 - 2.0)
    return Static[f32, n](ctx, values^)


def test_view_operators_stay_on_the_device() raises:
    var gpu = DeviceContext()
    var a = _ramp(gpu)
    var h = _ramp(DeviceContext(api="cpu"))
    var v = TensorView(a.tile(), a.context())
    var hv = TensorView(h.tile(), h.context())
    var s = v * 2.0 + v
    assert_false(s.on_host())
    var got = s.to_host()
    var want = (hv * 2.0 + hv).to_host()
    for i in range(n):
        assert_almost_equal(got[i], want[i], atol=1e-6)
    var mask = v > 0.0
    assert_false(mask.on_host())
    var m = mask.to_host()
    var hm = (hv > 0.0).to_host()
    for i in range(n):
        assert_equal(m[i], hm[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
