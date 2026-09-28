"""`ediff1d` and `unwrap` at `gpu=True`, against the host, at `float32`.

`ediff1d` is one subtraction per element, so it must agree exactly,
padding included. `unwrap`'s corrections are exact multiples of the
period, but the device scan adds them in a different order than the
host's running sum, so the unwrapped phase is held to a `float32`
tolerance scaled by how far it has wound.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.testing import assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.core.elementwise import ediff1d, unwrap

comptime f32 = DType.float32
comptime n = 700


def _phase(ctx: DeviceContext) raises -> Static[f32, n]:
    # A steadily winding phase, wrapped into `[-pi, pi)`.
    var values = List[Scalar[f32]](capacity=n)
    var pi = 3.141592653589793
    for i in range(n):
        var t = Float64(i) * 0.37
        var wrapped = t - 2 * pi * Float64(Int((t + pi) / (2 * pi)))
        values.append(Float32(wrapped))
    return Static[f32, n](values^, ctx)


def test_ediff1d_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = ediff1d[gpu=True](
        _phase(gpu), to_end=Float32(9.0), to_begin=Float32(-9.0)
    )
    assert_false(d.on_host())
    var got = d.to_host()
    var want = ediff1d(
        _phase(cpu), to_end=Float32(9.0), to_begin=Float32(-9.0)
    ).to_host()
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def test_unwrap_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = unwrap[gpu=True](_phase(gpu))
    assert_false(d.on_host())
    var got = d.to_host()
    var want = unwrap(_phase(cpu)).to_host()
    for i in range(n):
        assert_almost_equal(got[i], want[i], atol=1e-6 * Float64(i + 1))
    # And it did unwind: the last sample is near `0.37 * (n - 1)`.
    assert_almost_equal(got[n - 1], Float32(0.37 * Float64(n - 1)), atol=1e-2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
