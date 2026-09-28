"""`polygamma` inside a device kernel through `map[gpu=True]`, against the
same map on the host, at `float32`, on both sides of zero: the term
counts depend on the order alone, so every lane does the same work."""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax import Plain, polygamma
from numax.core.functional import map
from numax.core.tensor import Static

comptime f32 = DType.float32
comptime n = 256


def _trigamma[w: Int](x: SIMD[f32, w]) -> SIMD[f32, w]:
    return polygamma(1, Plain[f32, w](x)).v


def _tetragamma[w: Int](x: SIMD[f32, w]) -> SIMD[f32, w]:
    return polygamma(2, Plain[f32, w](x)).v


def _values(ctx: DeviceContext) raises -> Static[f32, n]:
    # From -3.97 to 12 in steps of 1/16, offset by 1/32 so no point
    # lands on a pole.
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(i) * 0.0625 - 3.96875)
    return Static[f32, n](values^, ctx)


def _check[
    step: def[w: Int](SIMD[f32, w]) thin -> SIMD[f32, w]
](gpu: DeviceContext, cpu: DeviceContext) raises:
    var ys = Static[f32, n](gpu)
    map[step=step, gpu=True](_values(gpu), ys)
    assert_false(ys.on_host())
    var hy = Static[f32, n](cpu)
    map[step=step](_values(cpu), hy)
    var got = ys.to_host()
    var want = hy.to_host()
    for i in range(n):
        assert_almost_equal(got[i], want[i], atol=1e-5, rtol=1e-4)


def test_polygamma_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    _check[_trigamma](gpu, cpu)
    _check[_tetragamma](gpu, cpu)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
