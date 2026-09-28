"""`sawtooth`, `square` and `chirp` at `gpu=True`, against the host, at
`float32`: the same formula on both sides, so the values agree to a few
`float32` ulp of the phase (the device `cos` may differ in the last place).
"""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import linspace
from numax.signal import chirp, sawtooth, square

comptime f32 = DType.float32
comptime n = 1000


def test_waveforms_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = sawtooth[gpu=True](linspace[n, f32](-10.0, 10.0, ctx=gpu), 0.3)
    assert_false(d.on_host())
    var got = d.to_host()
    var want = sawtooth(linspace[n, f32](-10.0, 10.0, ctx=cpu), 0.3).to_host()
    var sq = square[gpu=True](linspace[n, f32](-10.0, 10.0, ctx=gpu)).to_host()
    var sq_h = square(linspace[n, f32](-10.0, 10.0, ctx=cpu)).to_host()
    var c = chirp[method="logarithmic", gpu=True](
        linspace[n, f32](0.0, 2.0, ctx=gpu), 1.0, 2.0, 8.0
    ).to_host()
    var c_h = chirp[method="logarithmic"](
        linspace[n, f32](0.0, 2.0, ctx=cpu), 1.0, 2.0, 8.0
    ).to_host()
    for i in range(n):
        assert_almost_equal(got[i], want[i], atol=1e-5)
        assert_almost_equal(sq[i], sq_h[i], atol=1e-6)
        assert_almost_equal(c[i], c_h[i], atol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
