"""`ndtr`, `log_ndtr`, `ndtri`, `expit` and `log_expit` inside a device
kernel through `map[gpu=True]`, against the same map on the host, at
`float32`: tier 1 means the regions are blended, not branched, so lanes on
both sides of every split run in one launch."""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax import Plain, expit, log_expit, log_ndtr, ndtr, ndtri
from numax.core.functional import map
from numax.core.tensor import Static

comptime f32 = DType.float32
comptime n = 512


def _ndtr[w: Int](x: SIMD[f32, w]) -> SIMD[f32, w]:
    return ndtr(Plain[f32, w](x)).v


def _log_ndtr[w: Int](x: SIMD[f32, w]) -> SIMD[f32, w]:
    return log_ndtr(Plain[f32, w](x)).v


def _ndtri[w: Int](x: SIMD[f32, w]) -> SIMD[f32, w]:
    # `x` in `[-30, 30]` mapped into `(0, 1)` through `expit`, so the
    # tails and the center are all in the launch.
    return ndtri(expit(Plain[f32, w](x))).v


def _log_expit[w: Int](x: SIMD[f32, w]) -> SIMD[f32, w]:
    return log_expit(Plain[f32, w](x)).v


def _values(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(i) * (60.0 / Float32(n - 1)) - 30.0)
    return Static[f32, n](values^, ctx)


def _check[
    step: def[w: Int](SIMD[f32, w]) thin -> SIMD[f32, w]
](gpu: DeviceContext, cpu: DeviceContext) raises:
    var xs = _values(gpu)
    var ys = Static[f32, n](gpu)
    map[step=step, gpu=True](xs, ys)
    assert_false(ys.on_host())
    var hy = Static[f32, n](cpu)
    map[step=step](_values(cpu), hy)
    var got = ys.to_host()
    var want = hy.to_host()
    for i in range(n):
        assert_almost_equal(got[i], want[i], atol=1e-6, rtol=1e-4)


def test_normal_family_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    _check[_ndtr](gpu, cpu)
    _check[_log_ndtr](gpu, cpu)
    _check[_ndtri](gpu, cpu)
    _check[_log_expit](gpu, cpu)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
