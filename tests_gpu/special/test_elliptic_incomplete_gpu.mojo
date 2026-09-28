"""`ellipkinc`, `ellipeinc`, `elliptic_k` and `elliptic_e` inside a device kernel through
`map[gpu=True]`, against the same map on the host, at `float32`: the
duplication is a fixed step count and the amplitude reduction a `floor`,
so every lane does the same work."""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax import Plain, ellipeinc, ellipkinc, elliptic_e, elliptic_k
from numax.core.functional import map
from numax.core.tensor import Static

comptime f32 = DType.float32
comptime n = 256


def _m[w: Int](phi: SIMD[f32, w]) -> Plain[f32, w]:
    # The parameter sweeps (-2, 0.95) as the amplitude sweeps (-6, 6).
    return Plain[f32, w](
        SIMD[f32, w](0.95) - (SIMD[f32, w](6.0) - phi) * 0.2458
    )


def _f[w: Int](phi: SIMD[f32, w]) -> SIMD[f32, w]:
    return ellipkinc(Plain[f32, w](phi), _m(phi)).v


def _e[w: Int](phi: SIMD[f32, w]) -> SIMD[f32, w]:
    return ellipeinc(Plain[f32, w](phi), _m(phi)).v


def _k[w: Int](phi: SIMD[f32, w]) -> SIMD[f32, w]:
    return elliptic_k(_m(phi)).v


def _ek[w: Int](phi: SIMD[f32, w]) -> SIMD[f32, w]:
    return elliptic_e(_m(phi)).v


def _values(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(i) * (12.0 / Float32(n - 1)) - 6.0)
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


def test_incomplete_elliptic_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    _check[_f](gpu, cpu)
    _check[_e](gpu, cpu)
    _check[_k](gpu, cpu)
    _check[_ek](gpu, cpu)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
