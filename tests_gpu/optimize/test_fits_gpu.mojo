"""`least_squares` and `curve_fit` at `gpu=True` on device-resident data:
the residuals, the Jacobian and the damped system stay on the device, and
the fit recovers the parameters the data was generated from, as the host
fit does. `y = a exp(b x)` at 256 points, the model evaluated with device
elementwise launches."""

from std.testing import TestSuite, assert_almost_equal

from max.gpu.host import DeviceContext

from numax.core.elementwise import exp
from numax.core.ops import multiply, subtract
from numax.core.tensor import Static, linspace, stack
from numax.optimize import curve_fit, least_squares

comptime f32 = DType.float32
comptime n = 256


def _model(
    x: Static[f32, n], p: Static[f32, 2], ctx: DeviceContext
) raises -> Static[f32, n]:
    var h = p.to_host()
    return multiply[gpu=True](exp[gpu=True](multiply[gpu=True](x, h[1])), h[0])


def _jacobian(
    x: Static[f32, n], p: Static[f32, 2], ctx: DeviceContext
) raises -> Static[f32, n, 2]:
    var h = p.to_host()
    var e = exp[gpu=True](multiply[gpu=True](x, h[1]))
    var slope = multiply[gpu=True](multiply[gpu=True](x, e), h[0])
    return stack[axis=1, gpu=True](e, slope).as_static[n, 2]()


def _truth(ctx: DeviceContext) raises -> Static[f32, 2]:
    var values: List[Scalar[f32]] = [2.0, 0.5]
    return Static[f32, 2](ctx, values^)


def _residuals(p: Static[f32, 2], ctx: DeviceContext) raises -> Static[f32, n]:
    var x = linspace[n, f32](0.0, 2.0, ctx=ctx)
    return subtract[gpu=True](_model(x, p, ctx), _model(x, _truth(ctx), ctx))


def _residual_jacobian(
    p: Static[f32, 2], ctx: DeviceContext
) raises -> Static[f32, n, 2]:
    return _jacobian(linspace[n, f32](0.0, 2.0, ctx=ctx), p, ctx)


def _start(ctx: DeviceContext) raises -> Static[f32, 2]:
    var values: List[Scalar[f32]] = [1.0, 0.2]
    return Static[f32, 2](ctx, values^)


def test_least_squares_on_the_device() raises:
    var gpu = DeviceContext()
    var fit = least_squares[
        n_resid=n,
        residuals=_residuals,
        jacobian=_residual_jacobian,
        gpu=True,
    ](_start(gpu), 1e-5, 200)
    var got = fit.x.to_host()
    assert_almost_equal(got[0], 2.0, atol=1e-3)
    assert_almost_equal(got[1], 0.5, atol=1e-3)


def test_curve_fit_on_the_device_recovers_the_parameters() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var xs = linspace[n, f32](0.0, 2.0, ctx=gpu)
    var ys = _model(linspace[n, f32](0.0, 2.0, ctx=gpu), _truth(gpu), gpu)
    var fit = curve_fit[model=_model, model_jacobian=_jacobian, gpu=True](
        xs, ys, _start(gpu), 1e-5, 200
    )
    var got = fit.x.to_host()
    assert_almost_equal(got[0], 2.0, atol=1e-3)
    assert_almost_equal(got[1], 0.5, atol=1e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
