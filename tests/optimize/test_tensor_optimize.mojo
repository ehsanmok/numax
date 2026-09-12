"""Tests for the `Tensor`-tier optimize surface added in 0.2: `minimize`
with `"l-bfgs"`, `"powell"` and box bounds, `root`, `nnls` and
`lsq_linear` -- each against a known minimizer or `scipy.optimize`'s
answer, and `"l-bfgs"` against `"bfgs"` on the same problem, since limited
memory must reach the same point."""

from std.math import exp as _exp
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax.core.array import Static, zeros
from numax.optimize import lsq_linear, minimize, nnls, root

comptime dtype = DType.float64


def _rosenbrock(
    p: Static[dtype, 2], ctx: DeviceContext
) raises -> Scalar[dtype]:
    var v = p.to_host()
    var a = 1.0 - Float64(v[0])
    var b = Float64(v[1]) - Float64(v[0]) * Float64(v[0])
    return Scalar[dtype](a * a + 100.0 * b * b)


def _rosenbrock_jac(
    p: Static[dtype, 2], ctx: DeviceContext
) raises -> Static[dtype, 2]:
    var v = p.to_host()
    var x = Float64(v[0])
    var y = Float64(v[1])
    var entries = List[Scalar[dtype]](capacity=2)
    entries.append(Scalar[dtype](-2.0 * (1.0 - x) - 400.0 * x * (y - x * x)))
    entries.append(Scalar[dtype](200.0 * (y - x * x)))
    return Static[dtype, 2](ctx, entries^)


def _bowl(p: Static[dtype, 3], ctx: DeviceContext) raises -> Scalar[dtype]:
    """`(x-1)^2 + 2(y+2)^2 + 3(z-3)^2`, minimum `(1, -2, 3)`."""
    var v = p.to_host()
    var a = Float64(v[0]) - 1.0
    var b = Float64(v[1]) + 2.0
    var c = Float64(v[2]) - 3.0
    return Scalar[dtype](a * a + 2.0 * b * b + 3.0 * c * c)


def _bowl_jac(
    p: Static[dtype, 3], ctx: DeviceContext
) raises -> Static[dtype, 3]:
    var v = p.to_host()
    var entries = List[Scalar[dtype]](capacity=3)
    entries.append(Scalar[dtype](2.0 * (Float64(v[0]) - 1.0)))
    entries.append(Scalar[dtype](4.0 * (Float64(v[1]) + 2.0)))
    entries.append(Scalar[dtype](6.0 * (Float64(v[2]) - 3.0)))
    return Static[dtype, 3](ctx, entries^)


def _vec[n: Int](values: List[Float64]) raises -> Static[dtype, n]:
    var ctx = DeviceContext(api="cpu")
    var entries = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        entries.append(Scalar[dtype](values[i]))
    return Static[dtype, n](ctx, entries^)


def test_lbfgs_reaches_the_rosenbrock_minimum() raises:
    var x0 = _vec[2]([-1.2, 1.0])
    var result = minimize[
        dtype, 2, _rosenbrock, _rosenbrock_jac, method="l-bfgs"
    ](x0, max_iter=500)
    assert_true(result.converged)
    var x = result.x.to_host()
    assert_almost_equal(Float64(x[0]), 1.0, atol=1e-6)
    assert_almost_equal(Float64(x[1]), 1.0, atol=1e-6)
    assert_true(result.iterations < 200)


def test_lbfgs_agrees_with_bfgs_on_a_quadratic() raises:
    var x0 = _vec[3]([0.0, 0.0, 0.0])
    var x1 = _vec[3]([0.0, 0.0, 0.0])
    var limited = minimize[dtype, 3, _bowl, _bowl_jac, method="l-bfgs"](
        x0, tol=1e-12
    )
    var full = minimize[dtype, 3, _bowl, _bowl_jac, method="bfgs"](
        x1, tol=1e-12
    )
    assert_true(limited.converged)
    assert_true(full.converged)
    var a = limited.x.to_host()
    var b = full.x.to_host()
    for i in range(3):
        assert_almost_equal(Float64(a[i]), Float64(b[i]), atol=1e-9)
    assert_almost_equal(Float64(a[0]), 1.0, atol=1e-9)
    assert_almost_equal(Float64(a[1]), -2.0, atol=1e-9)
    assert_almost_equal(Float64(a[2]), 3.0, atol=1e-9)


def test_bounded_minimize_stops_on_the_active_bound() raises:
    """Rosenbrock over `x <= 0.5`: the constrained minimum is `(0.5, 0.25)`
    with `f = 0.25`, SciPy's L-BFGS-B answer."""
    var lower = _vec[2]([-2.0, -2.0])
    var upper = _vec[2]([0.5, 2.0])
    var x0 = _vec[2]([0.0, 0.0])
    var result = minimize[
        dtype, 2, _rosenbrock, _rosenbrock_jac, method="l-bfgs"
    ](x0, lower, upper, max_iter=500)
    assert_true(result.converged)
    var x = result.x.to_host()
    assert_almost_equal(Float64(x[0]), 0.5, atol=1e-8)
    assert_almost_equal(Float64(x[1]), 0.25, atol=1e-6)
    assert_almost_equal(result.f_x, 0.25, atol=1e-8)
    var x1 = _vec[2]([0.0, 0.0])
    var lower2 = _vec[2]([-2.0, -2.0])
    var upper2 = _vec[2]([0.5, 2.0])
    var with_bfgs = minimize[
        dtype, 2, _rosenbrock, _rosenbrock_jac, method="bfgs"
    ](x1, lower2, upper2, max_iter=500)
    var y = with_bfgs.x.to_host()
    assert_almost_equal(Float64(y[0]), 0.5, atol=1e-8)
    assert_almost_equal(Float64(y[1]), 0.25, atol=1e-6)


def test_powell_needs_no_gradient_and_finds_the_minimum() raises:
    var x0 = _vec[2]([-1.2, 1.0])
    var result = minimize[dtype, 2, _rosenbrock, method="powell"](
        x0, tol=1e-14, max_iter=500
    )
    var x = result.x.to_host()
    assert_almost_equal(Float64(x[0]), 1.0, atol=1e-5)
    assert_almost_equal(Float64(x[1]), 1.0, atol=1e-5)
    var b0 = _vec[3]([5.0, 5.0, 5.0])
    var bowl = minimize[dtype, 3, _bowl, method="powell"](b0, tol=1e-14)
    var y = bowl.x.to_host()
    assert_almost_equal(Float64(y[0]), 1.0, atol=1e-7)
    assert_almost_equal(Float64(y[1]), -2.0, atol=1e-7)
    assert_almost_equal(Float64(y[2]), 3.0, atol=1e-7)


def test_a_gradient_method_without_jac_raises() raises:
    var x0 = _vec[2]([0.0, 0.0])
    with assert_raises(contains="needs a gradient"):
        _ = minimize[dtype, 2, _rosenbrock, method="bfgs"](x0)


def _system(p: Static[dtype, 2], ctx: DeviceContext) raises -> Static[dtype, 2]:
    """`x^2 + y^2 = 4`, `e^x + y = 1`; SciPy's root from `(-1, -1)` is
    `(-1.8162640688, 0.8373677999)`."""
    var v = p.to_host()
    var x = Float64(v[0])
    var y = Float64(v[1])
    var entries = List[Scalar[dtype]](capacity=2)
    entries.append(Scalar[dtype](x * x + y * y - 4.0))
    entries.append(Scalar[dtype](_exp(x) + y - 1.0))
    return Static[dtype, 2](ctx, entries^)


def _system_jac(
    p: Static[dtype, 2], ctx: DeviceContext
) raises -> Static[dtype, 2, 2]:
    var v = p.to_host()
    var x = Float64(v[0])
    var y = Float64(v[1])
    var entries = List[Scalar[dtype]](capacity=4)
    entries.append(Scalar[dtype](2.0 * x))
    entries.append(Scalar[dtype](2.0 * y))
    entries.append(Scalar[dtype](_exp(x)))
    entries.append(Scalar[dtype](1.0))
    return Static[dtype, 2, 2](ctx, entries^)


def test_root_by_newton_and_lm_both_find_scipys_root() raises:
    var x0 = _vec[2]([-1.0, -1.0])
    var newton = root[dtype, 2, _system, _system_jac](x0)
    assert_true(newton.converged)
    assert_true(newton.residual_norm < 1e-10)
    var x = newton.x.to_host()
    assert_almost_equal(Float64(x[0]), -1.8162640688245402, atol=1e-9)
    assert_almost_equal(Float64(x[1]), 0.837367799891148, atol=1e-9)
    var x1 = _vec[2]([-1.0, -1.0])
    var lm = root[dtype, 2, _system, _system_jac, method="lm"](x1)
    assert_true(lm.residual_norm < 1e-8)
    var y = lm.x.to_host()
    assert_almost_equal(Float64(y[0]), -1.8162640688245402, atol=1e-7)
    assert_almost_equal(Float64(y[1]), 0.837367799891148, atol=1e-7)


def _design() raises -> Static[dtype, 5, 3]:
    var ctx = DeviceContext(api="cpu")
    var values: List[Float64] = [
        1.0,
        2.0,
        0.5,
        2.0,
        1.0,
        1.0,
        0.5,
        1.0,
        3.0,
        1.0,
        0.0,
        2.0,
        0.0,
        1.0,
        0.0,
    ]
    var entries = List[Scalar[dtype]](capacity=15)
    for i in range(15):
        entries.append(Scalar[dtype](values[i]))
    return Static[dtype, 5, 3](ctx, entries^)


def test_nnls_matches_scipy() raises:
    var a = _design()
    var b = _vec[5]([1.0, 2.0, -1.0, 0.5, 3.0])
    var result = nnls(a, b)
    assert_true(result.converged)
    var x = result.x.to_host()
    assert_almost_equal(Float64(x[0]), 0.3404255319148935, atol=1e-10)
    assert_almost_equal(Float64(x[1]), 0.6382978723404256, atol=1e-10)
    assert_equal(Float64(x[2]), 0.0)
    assert_almost_equal(result.residual_norm, 3.1173843372903156, atol=1e-10)


def test_lsq_linear_matches_scipy_with_and_without_active_bounds() raises:
    var a = _design()
    var b = _vec[5]([1.0, 2.0, -1.0, 0.5, 3.0])
    var lower = _vec[3]([-0.5, 0.0, -1.0])
    var upper = _vec[3]([0.5, 0.8, 0.2])
    var bounded = lsq_linear(a, b, lower, upper)
    assert_true(bounded.converged)
    var x = bounded.x.to_host()
    assert_almost_equal(Float64(x[0]), 0.5, atol=1e-10)
    assert_almost_equal(Float64(x[1]), 0.8, atol=1e-10)
    assert_almost_equal(Float64(x[2]), -0.45614035087719296, atol=1e-10)
    # Wide bounds: the unconstrained least-squares solution.
    var a2 = _design()
    var b2 = _vec[5]([1.0, 2.0, -1.0, 0.5, 3.0])
    var wide_lo = _vec[3]([-100.0, -100.0, -100.0])
    var wide_hi = _vec[3]([100.0, 100.0, 100.0])
    var free = lsq_linear(a2, b2, wide_lo, wide_hi)
    var y = free.x.to_host()
    assert_almost_equal(Float64(y[0]), 0.809917355371901, atol=1e-10)
    assert_almost_equal(Float64(y[1]), 0.7406230133502861, atol=1e-10)
    assert_almost_equal(Float64(y[2]), -0.5657978385251112, atol=1e-10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
