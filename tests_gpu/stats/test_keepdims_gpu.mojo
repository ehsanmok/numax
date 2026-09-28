"""`keepdims=True` at `gpu=True`: the kept result stays on the device, has
the reduced axis at extent 1, and holds the host's values exactly (for
`max` and `argmax`, which are exact) or to `float32` rounding (`sum`)."""

from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.testing import assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import argmax, max, sum

comptime f32 = DType.float32


def _m(ctx: DeviceContext) raises -> Static[f32, 64, 48]:
    var values = List[Scalar[f32]](capacity=64 * 48)
    for i in range(64 * 48):
        values.append(Float32((i * 37) % 101) - 50.0)
    return Static[f32, 64, 48](values^, ctx)


def test_keepdims_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var k = max[axis=1, keepdims=True, gpu=True](_m(gpu))
    assert_false(k.on_host())
    assert_equal(k.dim_at(0), 64)
    assert_equal(k.dim_at(1), 1)
    var got = k.to_host()
    var want = max[axis=1](_m(cpu)).to_host()
    for i in range(64):
        assert_equal(got[i], want[i])
    var a = argmax[axis=0, keepdims=True, gpu=True](_m(gpu))
    assert_equal(a.dim_at(0), 1)
    var ga = a.to_host()
    var wa = argmax[axis=0](_m(cpu)).to_host()
    for i in range(48):
        assert_equal(ga[i], wa[i])
    var s = sum[axis=0, keepdims=True, gpu=True](_m(gpu)).to_host()
    var ws = sum[axis=0](_m(cpu)).to_host()
    for i in range(48):
        assert_almost_equal(s[i], ws[i], atol=1e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
