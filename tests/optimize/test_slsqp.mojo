"""Tests for `minimize(method="slsqp")`: SciPy's linear-constraint
tutorial problem, Hock-Schittkowski 71 (a nonlinear equality and
inequality under bounds) and a constrained Rosenbrock against
`scipy.optimize.minimize(method="SLSQP")`'s optimum, the projection onto
a disk and an equality-constrained least norm against their closed
forms, the subproblem solver against `quadprog`'s documented example,
and the argument checks."""

from std.math import inf as _inf, sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    minimize,
)
from numax.optimize._qp import solve_qp

comptime dtype = DType.float64
comptime INF = _inf[DType.float64]()


def _shifted(p: Static[dtype, 2], ctx: DeviceContext) raises -> Scalar[dtype]:
    var v = p.to_host()
    return (v[0] - 1) ** 2 + (v[1] - 2.5) ** 2


def _shifted_grad(
    p: Static[dtype, 2], ctx: DeviceContext
) raises -> Static[dtype, 2]:
    var v = p.to_host()
    return Static[dtype, 2]([2 * (v[0] - 1), 2 * (v[1] - 2.5)], ctx)


def _disk(p: Static[dtype, 2], ctx: DeviceContext) raises -> Static[dtype, 1]:
    var v = p.to_host()
    return Static[dtype, 1]([v[0] * v[0] + v[1] * v[1]], ctx)


def _disk_jac(
    p: Static[dtype, 2], ctx: DeviceContext
) raises -> Static[dtype, 1, 2]:
    var v = p.to_host()
    return Static[dtype, 1, 2]([2 * v[0], 2 * v[1]], ctx)


def test_scipy_linear_tutorial() raises:
    """SciPy's SLSQP tutorial: three linear inequalities and `x >= 0`,
    optimum `(1.4, 1.7)`."""
    var a = Static[dtype, 3, 2]([1.0, -2.0, -1.0, -2.0, -1.0, 2.0])
    var lo = Static[dtype, 3]([-2.0, -6.0, -2.0])
    var hi = Static[dtype, 3]([INF, INF, INF])
    var x0 = Static[dtype, 2]([2.0, 0.0])
    var r = minimize[f=_shifted, jac=_shifted_grad](
        x0, [LinearConstraint(a, lo, hi)], Bounds(0.0, INF)
    )
    assert_true(r.converged)
    var x = r.x.to_host()
    assert_almost_equal(Float64(x[0]), 1.4, atol=1e-8)
    assert_almost_equal(Float64(x[1]), 1.7, atol=1e-8)
    assert_almost_equal(r.f_x, 0.8, atol=1e-10)
    assert_true(r.grad_norm < 1e-6)


def test_disk_projection() raises:
    """Minimizing the distance to `(1, 2.5)` over the disk of radius 2 is
    the radial projection `2 (1, 2.5) / |(1, 2.5)|`."""
    var x0 = Static[dtype, 2]([2.0, 0.0])
    var r = minimize[f=_shifted, jac=_shifted_grad](
        x0, NonlinearConstraint[dtype, 2, 1, _disk, _disk_jac](-INF, 4.0)
    )
    assert_true(r.converged)
    var x = r.x.to_host()
    var scale = 2.0 / sqrt(7.25)
    assert_almost_equal(Float64(x[0]), scale, atol=1e-6)
    assert_almost_equal(Float64(x[1]), 2.5 * scale, atol=1e-6)


def _hs071(p: Static[dtype, 4], ctx: DeviceContext) raises -> Scalar[dtype]:
    var v = p.to_host()
    return v[0] * v[3] * (v[0] + v[1] + v[2]) + v[2]


def _hs071_grad(
    p: Static[dtype, 4], ctx: DeviceContext
) raises -> Static[dtype, 4]:
    var v = p.to_host()
    var s = v[0] + v[1] + v[2]
    return Static[dtype, 4](
        [v[3] * s + v[0] * v[3], v[0] * v[3], v[0] * v[3] + 1, v[0] * s],
        ctx,
    )


def _hs071_con(
    p: Static[dtype, 4], ctx: DeviceContext
) raises -> Static[dtype, 2]:
    var v = p.to_host()
    return Static[dtype, 2](
        [
            v[0] * v[1] * v[2] * v[3],
            v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3],
        ],
        ctx,
    )


def _hs071_con_jac(
    p: Static[dtype, 4], ctx: DeviceContext
) raises -> Static[dtype, 2, 4]:
    var v = p.to_host()
    return Static[dtype, 2, 4](
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


def test_hs071() raises:
    """Hock-Schittkowski 71: `x0 x1 x2 x3 >= 25`, `|x|^2 = 40`, `1 <= x
    <= 5`, against SciPy's SLSQP at `ftol=1e-14`."""
    var x0 = Static[dtype, 4]([1.0, 5.0, 5.0, 1.0])
    var r = minimize[f=_hs071, jac=_hs071_grad](
        x0,
        NonlinearConstraint[dtype, 4, 2, _hs071_con, _hs071_con_jac](
            [25.0, 40.0], [INF, 40.0]
        ),
        bounds=Bounds(1.0, 5.0),
    )
    assert_true(r.converged)
    assert_almost_equal(r.f_x, 17.014017289136994, atol=1e-7)
    var x = r.x.to_host()
    var want: List[Float64] = [
        1.0000000000000606,
        4.742999623794407,
        3.821150001789563,
        1.3794082907335332,
    ]
    for i in range(4):
        assert_almost_equal(Float64(x[i]), want[i], atol=1e-5)


def _rosen(p: Static[dtype, 2], ctx: DeviceContext) raises -> Scalar[dtype]:
    var v = p.to_host()
    return 100 * (v[1] - v[0] * v[0]) ** 2 + (1 - v[0]) ** 2


def _rosen_grad(
    p: Static[dtype, 2], ctx: DeviceContext
) raises -> Static[dtype, 2]:
    var v = p.to_host()
    return Static[dtype, 2](
        [
            -400 * v[0] * (v[1] - v[0] * v[0]) - 2 * (1 - v[0]),
            200 * (v[1] - v[0] * v[0]),
        ],
        ctx,
    )


def _parabolas(
    p: Static[dtype, 2], ctx: DeviceContext
) raises -> Static[dtype, 2]:
    var v = p.to_host()
    return Static[dtype, 2]([v[0] * v[0] + v[1], v[0] * v[0] - v[1]], ctx)


def _parabolas_jac(
    p: Static[dtype, 2], ctx: DeviceContext
) raises -> Static[dtype, 2, 2]:
    var v = p.to_host()
    return Static[dtype, 2, 2]([2 * v[0], 1.0, 2 * v[0], -1.0], ctx)


def test_rosenbrock_mixed() raises:
    """Rosenbrock under a linear row, two nonlinear rows and a box, all
    three kinds together, against SciPy's SLSQP."""
    var x0 = Static[dtype, 2]([0.5, 0.0])
    var a = Static[dtype, 1, 2]([1.0, 2.0])
    var r = minimize[f=_rosen, jac=_rosen_grad](
        x0,
        NonlinearConstraint[dtype, 2, 2, _parabolas, _parabolas_jac](-INF, 1.0),
        linear=[LinearConstraint(a, -INF, 1.0)],
        bounds=Bounds([0.0, -0.5], [1.0, 2.0]),
    )
    assert_true(r.converged)
    assert_almost_equal(r.f_x, 0.24889703505718022, atol=1e-8)
    var x = r.x.to_host()
    assert_almost_equal(Float64(x[0]), 0.5022027172001443, atol=1e-5)
    assert_almost_equal(Float64(x[1]), 0.24889864139992784, atol=1e-5)


def _norm2(p: Static[dtype, 3], ctx: DeviceContext) raises -> Scalar[dtype]:
    var v = p.to_host()
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


def _norm2_grad(
    p: Static[dtype, 3], ctx: DeviceContext
) raises -> Static[dtype, 3]:
    var v = p.to_host()
    return Static[dtype, 3]([2 * v[0], 2 * v[1], 2 * v[2]], ctx)


def test_equality_least_norm() raises:
    """`min |x|^2` with `x0 + x1 + x2 = 1` is `x = 1/3` everywhere: an
    equality row (`lb == ub`) through the linear overload."""
    var x0 = Static[dtype, 3]([1.0, -2.0, 4.0])
    var a = Static[dtype, 1, 3]([1.0, 1.0, 1.0])
    var r = minimize[f=_norm2, jac=_norm2_grad](
        x0, [LinearConstraint(a, 1.0, 1.0)]
    )
    assert_true(r.converged)
    var x = r.x.to_host()
    for i in range(3):
        assert_almost_equal(Float64(x[i]), 1.0 / 3.0, atol=1e-8)


def test_subproblem_quadprog_example() raises:
    """The subproblem solver on `quadprog`'s documented example, with and
    without the first row as an equality."""
    var g: List[Float64] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    var a: List[Float64] = [0.0, 5.0, 0.0]
    var c: List[Float64] = [-4.0, -3.0, 0.0, 2.0, 1.0, 0.0, 0.0, -2.0, 1.0]
    var b: List[Float64] = [-8.0, 2.0, 0.0]
    var r = solve_qp(g, a, c, b, 0, 3)
    assert_true(r.feasible)
    assert_almost_equal(r.x[0], 0.4761904761904762, atol=1e-12)
    assert_almost_equal(r.x[1], 1.0476190476190477, atol=1e-12)
    assert_almost_equal(r.x[2], 2.0952380952380953, atol=1e-12)
    assert_almost_equal(r.multipliers[0], 0.0, atol=1e-12)
    assert_almost_equal(r.multipliers[1], 0.2380952380952381, atol=1e-12)
    assert_almost_equal(r.multipliers[2], 2.0952380952380953, atol=1e-12)
    var e = solve_qp(g, a, c, b, 1, 3)
    assert_almost_equal(e.x[0], 1.1235955056179774, atol=1e-12)
    assert_almost_equal(e.x[1], 1.1685393258426966, atol=1e-12)
    assert_almost_equal(e.x[2], 2.3370786516853934, atol=1e-12)


def test_argument_errors() raises:
    """Inverted bounds and a column mismatch raise, naming `minimize`."""
    var x0 = Static[dtype, 2]([0.0, 0.0])
    with assert_raises(contains="lb <= ub"):
        _ = minimize[f=_shifted, jac=_shifted_grad](
            x0, List[LinearConstraint](), Bounds(1.0, 0.0)
        )
    var a = Static[dtype, 1, 3]([1.0, 1.0, 1.0])
    with assert_raises(contains="columns"):
        _ = minimize[f=_shifted, jac=_shifted_grad](
            x0, [LinearConstraint(a, 0.0, 1.0)]
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
