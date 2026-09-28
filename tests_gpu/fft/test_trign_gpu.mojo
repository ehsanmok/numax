"""`dctn` and `idstn` at `gpu=True`, against the host, at `float32`.

A `4 x 8 x 16` volume (powers of two, so each pass is one radix-2 run per
lane): the device transform against the host's to `float32` rounding,
and the device round trip back to the input.
"""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.fft import dctn, dstn, idctn, idstn

comptime f32 = DType.float32


def _volume(ctx: DeviceContext) raises -> Static[f32, 4, 8, 16]:
    var values = List[Scalar[f32]](capacity=512)
    for i in range(512):
        values.append(Float32((i * 37) % 23) * 0.25 - 2.5)
    return Static[f32, 4, 8, 16](values^, ctx)


def test_dctn_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = dctn[gpu=True](_volume(gpu))
    assert_false(d.on_host())
    var got = d.to_host()
    var want = dctn(_volume(cpu)).to_host()
    for i in range(512):
        assert_almost_equal(got[i], want[i], atol=1e-2)
    var back = idctn[gpu=True](d^).to_host()
    var x = _volume(cpu).to_host()
    for i in range(512):
        assert_almost_equal(back[i], x[i], atol=1e-4)


def test_dstn_round_trip_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var back = idstn[gpu=True, norm="ortho"](
        dstn[gpu=True, norm="ortho"](_volume(gpu))
    ).to_host()
    var x = _volume(cpu).to_host()
    for i in range(512):
        assert_almost_equal(back[i], x[i], atol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
