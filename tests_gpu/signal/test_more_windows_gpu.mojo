"""The `tukey`, `gaussian`, `flattop` and `nuttall` device fills against
the host at `float32`: the same formula evaluated on either side, so the
values agree to a few `float32` ulp (the device `cos`/`exp` may differ in
the last place). `chebwin` is computed on the host and uploaded, so its
device copy must equal the host one exactly."""

from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.testing import assert_false

from max.gpu.host import DeviceContext

from numax.signal import chebwin, flattop, gaussian, nuttall, tukey

comptime f32 = DType.float32
comptime n = 257


def _close(got: List[Float32], want: List[Float32]) raises:
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=2e-6)


def test_window_fills_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var t = tukey[f32, n](0.4, True, gpu)
    assert_false(t.on_host())
    _close(t.to_host(), tukey[f32, n](0.4, True, cpu).to_host())
    _close(
        gaussian[f32, n](30.0, False, gpu).to_host(),
        gaussian[f32, n](30.0, False, cpu).to_host(),
    )
    _close(
        flattop[f32, n](True, gpu).to_host(),
        flattop[f32, n](True, cpu).to_host(),
    )
    _close(
        nuttall[f32, n](False, gpu).to_host(),
        nuttall[f32, n](False, cpu).to_host(),
    )
    var dc = chebwin[f32, n](80.0, True, gpu).to_host()
    var hc = chebwin[f32, n](80.0, True, cpu).to_host()
    for i in range(n):
        assert_equal(dc[i], hc[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
