"""`lgamma` and `gamma` at the integers inside a device kernel, against the
host, at `float32`.

The reflection side of `lgamma` takes `ln|sin(pi x)|`, which is `ln 0` at an
integer `x` wherever `sin` rounds to exactly 0. NVIDIA's float32 `sin` does
(`sin.approx.ftz.f32`), and the lane that discards that side still carried
`0 * -inf`, so every integer argument came back NaN."""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax import Plain, gamma, lgamma
from numax.core.functional import map
from numax.core.tensor import Static

comptime f32 = DType.float32
comptime n = 32


def _lgamma[w: Int](x: SIMD[f32, w]) -> SIMD[f32, w]:
    return lgamma(Plain[f32, w](x)).v


def _gamma[w: Int](x: SIMD[f32, w]) -> SIMD[f32, w]:
    return gamma(Plain[f32, w](x)).v


def _values(ctx: DeviceContext) raises -> Static[f32, n]:
    # 1, 1.5, 2, ... : integers and half-integers, all at or above 1.
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(1.0 + Float32(i) * 0.5)
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
        assert_almost_equal(got[i], want[i], atol=1e-4, rtol=1e-4)


def test_gamma_at_the_integers_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    _check[_lgamma](gpu, cpu)
    _check[_gamma](gpu, cpu)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
