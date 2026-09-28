"""`upfirdn` and `resample_poly` at `gpu=True`, against the host, at
`float32`: one lane per output sample summing the same polyphase taps, so
the results agree to `float32` rounding."""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.signal import resample_poly, upfirdn

comptime f32 = DType.float32
comptime n = 1500


def _x(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(sin(0.05 * Float64(i))))
    return Static[f32, n](values^, ctx)


def test_upfirdn_and_resample_poly_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var hg = Static[f32, 5]([0.2, 0.5, 1.0, 0.5, 0.2], gpu)
    var hc = Static[f32, 5]([0.2, 0.5, 1.0, 0.5, 0.2], cpu)
    var d = upfirdn[up=3, down=2, gpu=True](hg, _x(gpu))
    assert_false(d.on_host())
    var got = d.to_host()
    var want = upfirdn[up=3, down=2](hc, _x(cpu)).to_host()
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-5)
    var r = resample_poly[up=3, down=2, gpu=True](_x(gpu)).to_host()
    var rh = resample_poly[up=3, down=2](_x(cpu)).to_host()
    for i in range(len(rh)):
        assert_almost_equal(r[i], rh[i], atol=1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
