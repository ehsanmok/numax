"""`RBFInterpolator` and `NearestNDInterpolator` at `gpu=True`, against the
host, at `float32`: the RBF system's assembly, its run-time-order LU and
the evaluation lanes, and the nearest-site tree queries and gather, all
on the device."""

from std.math import cos, sin
from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.interpolate import NearestNDInterpolator, RBFInterpolator

comptime f32 = DType.float32
comptime n = 120
comptime m = 64


def _sites(ctx: DeviceContext) raises -> Static[f32, n, 2]:
    var values = List[Scalar[f32]](capacity=2 * n)
    for i in range(n):
        values.append(0.5 + 0.5 * sin(Float32(i) * 1.37))
        values.append(0.5 + 0.5 * sin(Float32(i) * 0.71 + 0.4))
    return Static[f32, n, 2](values^, ctx)


def _values(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        var a = 0.5 + 0.5 * sin(Float32(i) * 1.37)
        var b = 0.5 + 0.5 * sin(Float32(i) * 0.71 + 0.4)
        values.append(sin(a * 3.0) * cos(b * 2.0))
    return Static[f32, n](values^, ctx)


def _queries(ctx: DeviceContext) raises -> Static[f32, m, 2]:
    var values = List[Scalar[f32]](capacity=2 * m)
    for i in range(m):
        values.append(0.2 + 0.6 * Float32(i % 8) / 7.0)
        values.append(0.2 + 0.6 * Float32(i // 8) / 7.0)
    return Static[f32, m, 2](values^, ctx)


def test_scattered_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = RBFInterpolator[f32, gpu=True](_sites(gpu), _values(gpu))
    var h = RBFInterpolator[f32](_sites(cpu), _values(cpu))
    var dv = d(_queries(gpu))
    assert_false(dv.on_host())
    var got = dv.to_host()
    var want = h(_queries(cpu)).to_host()
    for i in range(m):
        assert_almost_equal(got[i], want[i], atol=2e-3)
    var dn = NearestNDInterpolator[f32, gpu=True](_sites(gpu), _values(gpu))
    var hn = NearestNDInterpolator[f32](_sites(cpu), _values(cpu))
    var gn = dn(_queries(gpu)).to_host()
    var wn = hn(_queries(cpu)).to_host()
    for i in range(m):
        assert_almost_equal(gn[i], wn[i], atol=0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
