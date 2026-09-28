"""`polyder` and `polyint` at `gpu=True`, against the host.

Each is one multiply or divide per coefficient, so the device answer is
the host's to the last bit at `float32` for these small integers, and the
test also checks `polyint`'s constant lands in the last slot.
"""

from std.testing import TestSuite, assert_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.interpolate import polyder, polyint

comptime f32 = DType.float32


def _p(ctx: DeviceContext) raises -> Static[f32, 6]:
    return Static[f32, 6]([3.0, -2.0, 0.5, 4.0, -1.0, 7.0], ctx)


def test_polyder_on_the_device_matches_the_host() raises:
    var d = polyder[gpu=True](_p(DeviceContext()))
    assert_false(d.on_host())
    var got = d.to_host()
    var want = polyder(_p(DeviceContext(api="cpu"))).to_host()
    for i in range(5):
        assert_equal(got[i], want[i])


def test_polyint_on_the_device_matches_the_host() raises:
    var d = polyint[gpu=True](_p(DeviceContext()), Float32(2.5))
    assert_false(d.on_host())
    var got = d.to_host()
    var want = polyint(_p(DeviceContext(api="cpu")), Float32(2.5)).to_host()
    for i in range(7):
        assert_equal(got[i], want[i])
    assert_equal(got[6], Float32(2.5))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
