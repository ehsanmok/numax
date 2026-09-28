"""Spline construction on the device, against the host.

Knots on a device build the spline there (secants, slopes, the
`CubicSpline` band solve and the coefficients, nothing downloaded); the
host builds in `Float64`. Each test builds both from the same
non-uniform knots and compares the evaluated spline and its first
derivative at `float32` tolerance.
"""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.interpolate import (
    Akima1DInterpolator,
    CubicHermiteSpline,
    CubicSpline,
    PchipInterpolator,
)

comptime f32 = DType.float32
comptime n = 37
comptime m = 101


def _knots(ctx: DeviceContext) raises -> Static[f32, n]:
    """Strictly ascending, widths varying by a factor of three."""
    var values = List[Scalar[f32]](capacity=n)
    var at = Float32(0)
    for i in range(n):
        values.append(at)
        at += Float32(0.1) + Float32((i * 7) % 5) * 0.05
    return Static[f32, n](ctx, values^)


def _values(ctx: DeviceContext) raises -> Static[f32, n]:
    """A wiggle with a flat stretch and a jump, so PCHIP's zeroed slopes
    and Akima's vanishing weights are both reached."""
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        if 10 <= i and i < 15:
            values.append(1.0)
        elif i >= 25:
            values.append(Float32(2) + Float32(sin(Float64(i) * 0.7)))
        else:
            values.append(Float32(sin(Float64(i) * 0.4)))
    return Static[f32, n](ctx, values^)


def _points(ctx: DeviceContext) raises -> Static[f32, m]:
    var values = List[Scalar[f32]](capacity=m)
    for i in range(m):
        values.append(Float32(-0.2) + Float32(i) * 0.075)
    return Static[f32, m](ctx, values^)


def _assert_close(got: List[Scalar[f32]], want: List[Scalar[f32]]) raises:
    for i in range(len(want)):
        assert_almost_equal(
            Float64(got[i]), Float64(want[i]), atol=2e-4, rtol=2e-4
        )


def _check_cubic(name: StaticString) raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var xg = _knots(gpu)
    var yg = _values(gpu)
    var xc = _knots(cpu)
    var yc = _values(cpu)
    var dev = CubicSpline[f32, n](xg, yg, name)
    var host = CubicSpline[f32, n](xc, yc, name)
    assert_false(dev.spline.c.on_host())
    _assert_close(
        dev[gpu=True](_points(gpu)).to_host(), host(_points(cpu)).to_host()
    )
    _assert_close(
        dev[nu=1, gpu=True](_points(gpu)).to_host(),
        host[nu=1](_points(cpu)).to_host(),
    )


def test_cubic_spline_builds_on_the_device() raises:
    _check_cubic("not-a-knot")
    _check_cubic("natural")
    _check_cubic("clamped")


def test_small_cubic_splines_build_from_device_knots() raises:
    """Four knots is the smallest not-a-knot system the device folds;
    three and two are SciPy's special cases, which stay on the host."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var xs: List[Scalar[f32]] = [0.0, 0.5, 1.5, 2.0]
    var ys: List[Scalar[f32]] = [1.0, -1.0, 2.0, 0.5]
    var xg = Static[f32, 4](gpu, xs.copy())
    var yg = Static[f32, 4](gpu, ys.copy())
    var xc = Static[f32, 4](cpu, xs.copy())
    var yc = Static[f32, 4](cpu, ys.copy())
    var dev = CubicSpline[f32, 4](xg, yg)
    var host = CubicSpline[f32, 4](xc, yc)
    _assert_close(
        dev[gpu=True](_points(gpu)).to_host(), host(_points(cpu)).to_host()
    )
    var x3g = Static[f32, 3](gpu, [0.0, 0.5, 1.5])
    var y3g = Static[f32, 3](gpu, [1.0, -1.0, 2.0])
    var x3c = Static[f32, 3](cpu, [0.0, 0.5, 1.5])
    var y3c = Static[f32, 3](cpu, [1.0, -1.0, 2.0])
    var dev3 = CubicSpline[f32, 3](x3g, y3g)
    var host3 = CubicSpline[f32, 3](x3c, y3c)
    _assert_close(
        dev3[gpu=True](_points(gpu)).to_host(), host3(_points(cpu)).to_host()
    )


def test_pchip_builds_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var xg = _knots(gpu)
    var yg = _values(gpu)
    var xc = _knots(cpu)
    var yc = _values(cpu)
    var dev = PchipInterpolator[f32, n](xg, yg)
    var host = PchipInterpolator[f32, n](xc, yc)
    assert_false(dev.spline.c.on_host())
    _assert_close(
        dev[gpu=True](_points(gpu)).to_host(), host(_points(cpu)).to_host()
    )
    _assert_close(
        dev[nu=1, gpu=True](_points(gpu)).to_host(),
        host[nu=1](_points(cpu)).to_host(),
    )


def test_akima_builds_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var xg = _knots(gpu)
    var yg = _values(gpu)
    var xc = _knots(cpu)
    var yc = _values(cpu)
    var dev = Akima1DInterpolator[f32, n](xg, yg, extrapolate=True)
    var host = Akima1DInterpolator[f32, n](xc, yc, extrapolate=True)
    assert_false(dev.spline.c.on_host())
    _assert_close(
        dev[gpu=True](_points(gpu)).to_host(), host(_points(cpu)).to_host()
    )


def test_hermite_builds_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var xg = _knots(gpu)
    var yg = _values(gpu)
    var dg = _values(gpu)
    var xc = _knots(cpu)
    var yc = _values(cpu)
    var dc = _values(cpu)
    var dev = CubicHermiteSpline[f32, n](xg, yg, dg)
    var host = CubicHermiteSpline[f32, n](xc, yc, dc)
    assert_false(dev.c.on_host())
    _assert_close(dev.c.to_host(), host.c.to_host())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
