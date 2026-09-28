"""`norm=` and `hfft`/`ihfft` at `gpu=True`, against the host, at `float32`.

The scaling pass runs on the device beside the transform, so the device
answer must match the host's to rounding under `"ortho"` and
`"forward"`, and `hfft(ihfft(x))` must return `x` there too.
"""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static, zeros
from numax.fft import fft, hfft, ihfft, irfft, rfft

comptime f32 = DType.float32
comptime n = 256


def _x(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32((i * 37) % 23) * 0.25 - 2.5)
    return Static[f32, n](values^, ctx)


def test_norms_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = fft[gpu=True, norm="ortho"]((_x(gpu), zeros[f32, n](gpu)))
    assert_false(d[0].on_host())
    var h = fft[norm="ortho"]((_x(cpu), zeros[f32, n](cpu)))
    var dr = d[0].to_host()
    var hr = h[0].to_host()
    for i in range(n):
        assert_almost_equal(dr[i], hr[i], atol=1e-4)
    var back = irfft[gpu=True, norm="forward"](
        rfft[gpu=True, norm="forward"](_x(gpu))
    ).to_host()
    var x = _x(cpu).to_host()
    for i in range(n):
        assert_almost_equal(back[i], x[i], atol=1e-5)


def test_hfft_round_trip_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var back = hfft[gpu=True](ihfft[gpu=True](_x(gpu)))
    assert_false(back.on_host())
    var got = back.to_host()
    var x = _x(cpu).to_host()
    for i in range(n):
        assert_almost_equal(got[i], x[i], atol=1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
