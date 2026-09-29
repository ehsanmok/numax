"""`minimize(method="slsqp")` on a device-resident problem: `x0` on the
GPU context, so every evaluation of the objective, its gradient and the
constraints runs there, at `float32`. Hock-Schittkowski 71 against SciPy's
SLSQP optimum."""

from std.math import inf as _inf
from std.testing import TestSuite, assert_almost_equal, assert_true

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.optimize import Bounds, NonlinearConstraint, minimize

comptime f32 = DType.float32
comptime INF = _inf[DType.float64]()


def _hs071(p: Static[f32, 4], ctx: DeviceContext) raises -> Scalar[f32]:
    var v = p.to_host()
    return v[0] * v[3] * (v[0] + v[1] + v[2]) + v[2]


def _hs071_grad(p: Static[f32, 4], ctx: DeviceContext) raises -> Static[f32, 4]:
    var v = p.to_host()
    var s = v[0] + v[1] + v[2]
    return Static[f32, 4](
        [v[3] * s + v[0] * v[3], v[0] * v[3], v[0] * v[3] + 1, v[0] * s],
        ctx,
    )


def _hs071_con(p: Static[f32, 4], ctx: DeviceContext) raises -> Static[f32, 2]:
    var v = p.to_host()
    return Static[f32, 2](
        [
            v[0] * v[1] * v[2] * v[3],
            v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3],
        ],
        ctx,
    )


def _hs071_con_jac(
    p: Static[f32, 4], ctx: DeviceContext
) raises -> Static[f32, 2, 4]:
    var v = p.to_host()
    return Static[f32, 2, 4](
        [
            v[1] * v[2] * v[3],
            v[0] * v[2] * v[3],
            v[0] * v[1] * v[3],
            v[0] * v[1] * v[2],
            2 * v[0],
            2 * v[1],
            2 * v[2],
            2 * v[3],
        ],
        ctx,
    )


def test_hs071_on_device() raises:
    """HS071 from a device `x0` reaches SciPy's `17.0140` at `float32`."""
    var ctx = DeviceContext()
    var x0 = Static[f32, 4]([1.0, 5.0, 5.0, 1.0], ctx)
    var r = minimize[f=_hs071, jac=_hs071_grad](
        x0,
        NonlinearConstraint[f32, 4, 2, _hs071_con, _hs071_con_jac](
            [25.0, 40.0], [INF, 40.0]
        ),
        bounds=Bounds(1.0, 5.0),
    )
    assert_true(r.converged)
    assert_true(not r.x.on_host())
    assert_almost_equal(r.f_x, 17.014017289136994, atol=1e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
