"""`wofz` inside a device kernel through `map[gpu=True]`, against the same
map on the host, at `float32`, with lanes above and below the real axis:
the reflection is a blend, so both half-planes run in one launch."""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax import Plain, wofz
from numax.core.complex import Complex
from numax.core.functional import map
from numax.core.tensor import Static

comptime f32 = DType.float32
comptime n = 256


def _z[w: Int](x: SIMD[f32, w]) -> Complex[Plain[f32, w]]:
    # `y = 0.3 x`: the lanes with negative `x` sit below the axis.
    return Complex[Plain[f32, w]](Plain[f32, w](x), Plain[f32, w](x * 0.3))


def _re[w: Int](x: SIMD[f32, w]) -> SIMD[f32, w]:
    return wofz(_z(x)).re.v


def _im[w: Int](x: SIMD[f32, w]) -> SIMD[f32, w]:
    return wofz(_z(x)).im.v


def _values(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(i) * (8.0 / Float32(n - 1)) - 4.0)
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


def test_wofz_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    _check[_re](gpu, cpu)
    _check[_im](gpu, cpu)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
