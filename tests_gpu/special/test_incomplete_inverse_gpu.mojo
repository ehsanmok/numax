"""`gammaincinv` and `betaincinv` inside a device kernel through
`map[gpu=True]`, against the same map on the host, at `float32`: the
guesses are blended across NR's splits and the Halley steps are a fixed
count, so lanes on every side run in one launch."""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax import Plain, betaincinv, gammaincinv
from numax.core.functional import map
from numax.core.tensor import Static

comptime f32 = DType.float32
comptime n = 256


def _gammaincinv[w: Int](y: SIMD[f32, w]) -> SIMD[f32, w]:
    # Shapes on both sides of `a = 1`, from the same lanes' `y`.
    var a = Plain[f32, w](SIMD[f32, w](0.4)) + Plain[f32, w](y) * Plain[f32, w](
        SIMD[f32, w](12.0)
    )
    return gammaincinv(a, Plain[f32, w](y)).v


def _betaincinv[w: Int](y: SIMD[f32, w]) -> SIMD[f32, w]:
    # `a` from 0.3 to 6 and `b` the other way, so both sides of
    # `a, b >= 1` appear.
    var py = Plain[f32, w](y)
    var a = Plain[f32, w](SIMD[f32, w](0.3)) + py * Plain[f32, w](
        SIMD[f32, w](5.7)
    )
    var b = Plain[f32, w](SIMD[f32, w](6.0)) - py * Plain[f32, w](
        SIMD[f32, w](5.5)
    )
    return betaincinv(py, a, b).v


def _values(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append((Float32(i) + 0.5) / Float32(n))
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


def test_inverses_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    _check[_gammaincinv](gpu, cpu)
    _check[_betaincinv](gpu, cpu)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
