"""`group_delay` at `gpu=True`, against the host, at `float32`: one lane
per frequency evaluating the same two polynomials, so the delays agree to
`float32` rounding of a quotient (looser near the band edge, where the
denominator is small)."""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.signal import butter, group_delay

comptime f32 = DType.float32


def test_group_delay_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var dtf = butter[f32, 4](0.3, ctx=gpu)
    var htf = butter[f32, 4](0.3, ctx=cpu)
    var d = group_delay[worN=128, gpu=True](dtf.b, dtf.a)
    assert_false(d.gd.on_host())
    var got = d.gd.to_host()
    var want = group_delay[worN=128](htf.b, htf.a).gd.to_host()
    for i in range(64):
        assert_almost_equal(got[i], want[i], atol=1e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
