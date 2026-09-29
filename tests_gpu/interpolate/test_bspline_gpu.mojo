"""`make_interp_spline` and `BSpline` at `gpu=True`, against the host, at
`float32`: the collocation assembly and its dense solve, the evaluation
lanes and the coefficient maps all run on the device."""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.interpolate import make_interp_spline

comptime f32 = DType.float32
comptime n = 33
comptime q = 100


def _x(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(i) * 0.2 + 0.03 * sin(Float32(i)))
    return Static[f32, n](values^, ctx)


def _y(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        var x = Float32(i) * 0.2 + 0.03 * sin(Float32(i))
        values.append(sin(x) * 2.0 + 0.1 * x)
    return Static[f32, n](values^, ctx)


def _q(ctx: DeviceContext) raises -> Static[f32, q]:
    var values = List[Scalar[f32]](capacity=q)
    for i in range(q):
        values.append(Float32(i) * 0.067 - 0.2)
    return Static[f32, q](values^, ctx)


def test_bspline_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = make_interp_spline[k=3, gpu=True](_x(gpu), _y(gpu))
    var h = make_interp_spline[k=3](_x(cpu), _y(cpu))
    assert_false(d.c.on_host())
    var dv = d(_q(gpu)).to_host()
    var hv = h(_q(cpu)).to_host()
    var dd = d(_q(gpu), 1).to_host()
    var hd = h(_q(cpu), 1).to_host()
    for i in range(q):
        assert_almost_equal(dv[i], hv[i], atol=1e-4)
        assert_almost_equal(dd[i], hd[i], atol=1e-3)
    assert_almost_equal(d.integrate(0.5, 5.5), h.integrate(0.5, 5.5), atol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
