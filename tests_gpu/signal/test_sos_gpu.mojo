"""`sosfiltfilt` and `sosfreqz` at `gpu=True`, against the host, at
`float32`.

`sosfiltfilt` runs every section's block-parallel recurrence forward and
then backward over the device extension, so the device answer must match
the host cascade to `float32` rounding; `sosfreqz` is one lane per
frequency and must match to rounding too.
"""

from std.math import cos, sin
from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.signal import OUTPUT_SOS, butter, sosfiltfilt, sosfreqz

comptime f32 = DType.float32
comptime n = 2048


def _signal(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        var t = Float64(i)
        values.append(Float32(sin(0.05 * t) + 0.3 * cos(1.3 * t)))
    return Static[f32, n](values^, ctx)


def test_sosfiltfilt_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = sosfiltfilt[gpu=True](
        butter[f32, 6, output=OUTPUT_SOS](0.2, ctx=gpu), _signal(gpu)
    )
    assert_false(d.on_host())
    var got = d.to_host()
    var want = sosfiltfilt(
        butter[f32, 6, output=OUTPUT_SOS](0.2, ctx=cpu), _signal(cpu)
    ).to_host()
    for i in range(n):
        assert_almost_equal(got[i], want[i], atol=1e-4)


def test_sosfreqz_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = sosfreqz[worN=256, gpu=True](
        butter[f32, 6, output=OUTPUT_SOS](0.2, ctx=gpu)
    )
    var h = sosfreqz[worN=256](butter[f32, 6, output=OUTPUT_SOS](0.2, ctx=cpu))
    var dr = d.real.to_host()
    var hr = h.real.to_host()
    var di = d.imag.to_host()
    var hi = h.imag.to_host()
    for i in range(256):
        assert_almost_equal(dr[i], hr[i], atol=1e-5)
        assert_almost_equal(di[i], hi[i], atol=1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
