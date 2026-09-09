"""Tests for `numax.optimize` over `Tensor`.

The fits here are exactly recoverable -- the data is generated from known
parameters with no noise -- so a converged fit must return those parameters,
not merely a small residual. Two of them also pin the `Tensor` tier against
`numax.optimize.array`, which reads its Jacobian off `Gradient` rather than
taking one, so agreement means the caller-supplied Jacobian and the exact
one drive the same iteration.
"""

from std.math import exp as _exp
from std.testing import TestSuite, assert_almost_equal, assert_true

from max.gpu.host import DeviceContext
from std.collections import Array

from numax.core.array import Static
from numax.core.numeric import FloatLike
from numax.optimize import curve_fit, least_squares
from numax.optimize.array import least_squares as array_least_squares

comptime dtype = DType.float64


# Four points on `y = 2 * exp(0.5 x)`, sampled at x = 0, 1, 2, 3 -- so the
# sample point is just the index, and the data needs no table.
def _sample_x(i: Int) -> Float64:
    return Float64(i)


def _sample_y(i: Int) -> Float64:
    return 2.0 * _exp(0.5 * Float64(i))


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _exp_residuals(
    p: Static[dtype, 2], ctx: DeviceContext
) raises -> Static[dtype, 4]:
    """`a * exp(b x) - y` at the four sample points."""
    var h = p.to_host()
    var a = Float64(h[0])
    var b = Float64(h[1])
    var out = List[Scalar[dtype]](capacity=4)
    for i in range(4):
        out.append(Scalar[dtype](a * _exp(b * _sample_x(i)) - _sample_y(i)))
    return Static[dtype, 4](ctx, out^)


def _exp_jacobian(
    p: Static[dtype, 2], ctx: DeviceContext
) raises -> Static[dtype, 4, 2]:
    """`d/da = exp(b x)`, `d/db = a x exp(b x)`."""
    var h = p.to_host()
    var a = Float64(h[0])
    var b = Float64(h[1])
    var out = List[Scalar[dtype]](capacity=8)
    for i in range(4):
        out.append(Scalar[dtype](_exp(b * _sample_x(i))))
        out.append(Scalar[dtype](a * _sample_x(i) * _exp(b * _sample_x(i))))
    return Static[dtype, 4, 2](ctx, out^)


def _linear_residuals(
    p: Static[dtype, 2], ctx: DeviceContext
) raises -> Static[dtype, 4]:
    """`c + m x - y` against four points on `y = 1 + 2x`. Linear, so
    Levenberg-Marquardt should land in a single accepted step."""
    var h = p.to_host()
    var c = Float64(h[0])
    var m = Float64(h[1])
    var out = List[Scalar[dtype]](capacity=4)
    for i in range(4):
        out.append(
            Scalar[dtype](c + m * _sample_x(i) - (1.0 + 2.0 * _sample_x(i)))
        )
    return Static[dtype, 4](ctx, out^)


def _linear_jacobian(
    p: Static[dtype, 2], ctx: DeviceContext
) raises -> Static[dtype, 4, 2]:
    var out = List[Scalar[dtype]](capacity=8)
    for i in range(4):
        out.append(Scalar[dtype](1.0))
        out.append(Scalar[dtype](_sample_x(i)))
    return Static[dtype, 4, 2](ctx, out^)


def test_least_squares_recovers_an_exact_exponential_fit() raises:
    """Noise-free data from `a = 2, b = 0.5`, started well away at
    `(1, 1)`, so the damping loop has to do real work."""
    var ctx = _cpu()
    var p0 = Static[dtype, 2](ctx, [1.0, 1.0])
    var fit = least_squares[dtype, 2, 4, _exp_residuals, _exp_jacobian](p0)

    assert_true(fit.converged)
    var got = fit.x.to_host()
    assert_almost_equal(Float64(got[0]), 2.0, atol=1e-9)
    assert_almost_equal(Float64(got[1]), 0.5, atol=1e-9)
    assert_almost_equal(fit.cost, 0.0, atol=1e-18)


def test_least_squares_solves_a_linear_fit() raises:
    """A linear model makes the Gauss-Newton step exact, so this pins that
    the augmented system carries the right sign: a wrong one would step
    away from the solution and the damping loop would stall."""
    var ctx = _cpu()
    var p0 = Static[dtype, 2](ctx, [0.0, 0.0])
    var fit = least_squares[dtype, 2, 4, _linear_residuals, _linear_jacobian](
        p0
    )

    assert_true(fit.converged)
    var got = fit.x.to_host()
    assert_almost_equal(Float64(got[0]), 1.0, atol=1e-10)
    assert_almost_equal(Float64(got[1]), 2.0, atol=1e-10)


def test_least_squares_starting_at_the_solution_converges_immediately() raises:
    """The gradient test runs before the first step, so a fit started at
    the answer must report zero iterations rather than taking one."""
    var ctx = _cpu()
    var p0 = Static[dtype, 2](ctx, [2.0, 0.5])
    var fit = least_squares[dtype, 2, 4, _exp_residuals, _exp_jacobian](p0)

    assert_true(fit.converged)
    assert_almost_equal(Float64(fit.iterations), 0.0, atol=0.0)


def test_least_squares_reports_failure_rather_than_raising() raises:
    """Cut off at one iteration, the fit must come back `converged=False`
    carrying the best parameters it reached -- not raise, and not report
    success."""
    var ctx = _cpu()
    var p0 = Static[dtype, 2](ctx, [1.0, 1.0])
    var fit = least_squares[dtype, 2, 4, _exp_residuals, _exp_jacobian](
        p0, 1e-10, 1
    )
    assert_true(not fit.converged)
    assert_true(fit.grad_norm > 0.0)


def _model(
    x: Static[dtype, 4], p: Static[dtype, 2], ctx: DeviceContext
) raises -> Static[dtype, 4]:
    var xs = x.to_host()
    var h = p.to_host()
    var a = Float64(h[0])
    var b = Float64(h[1])
    var out = List[Scalar[dtype]](capacity=4)
    for i in range(4):
        out.append(Scalar[dtype](a * _exp(b * Float64(xs[i]))))
    return Static[dtype, 4](ctx, out^)


def _model_jacobian(
    x: Static[dtype, 4], p: Static[dtype, 2], ctx: DeviceContext
) raises -> Static[dtype, 4, 2]:
    var xs = x.to_host()
    var h = p.to_host()
    var a = Float64(h[0])
    var b = Float64(h[1])
    var out = List[Scalar[dtype]](capacity=8)
    for i in range(4):
        var xi = Float64(xs[i])
        out.append(Scalar[dtype](_exp(b * xi)))
        out.append(Scalar[dtype](a * xi * _exp(b * xi)))
    return Static[dtype, 4, 2](ctx, out^)


def test_curve_fit_matches_least_squares_on_the_same_problem() raises:
    """`curve_fit` repeats the loop rather than wrapping `least_squares`,
    because a compile-time residual parameter cannot reach run-time data.
    Two copies of an algorithm are worth pinning to each other."""
    var ctx = _cpu()
    var xdata = Static[dtype, 4](
        ctx, [_sample_x(0), _sample_x(1), _sample_x(2), _sample_x(3)]
    )
    var ydata = Static[dtype, 4](
        ctx, [_sample_y(0), _sample_y(1), _sample_y(2), _sample_y(3)]
    )
    var p0 = Static[dtype, 2](ctx, [1.0, 1.0])
    var fitted = curve_fit[dtype, 2, 4, _model, _model_jacobian](
        xdata, ydata, p0
    )

    var q0 = Static[dtype, 2](ctx, [1.0, 1.0])
    var direct = least_squares[dtype, 2, 4, _exp_residuals, _exp_jacobian](q0)

    assert_true(fitted.converged)
    var a = fitted.x.to_host()
    var b = direct.x.to_host()
    for j in range(2):
        assert_almost_equal(Float64(a[j]), Float64(b[j]), atol=1e-9)


def _array_residuals[U: FloatLike](p: Array[U, 2]) -> Array[U, 4]:
    """The same model for the `Array` tier, which differentiates it rather
    than being handed a Jacobian."""
    var out = Array[U, 4](fill=U.constant(0.0))
    for i in range(4):
        out[i] = p[0] * (p[1] * U.constant(_sample_x(i))).exp() - U.constant(
            _sample_y(i)
        )
    return out^


def test_the_two_tiers_agree() raises:
    """The caller-supplied Jacobian against the one the `Array` tier reads
    off `Gradient`. Same algorithm, same damping schedule, two entirely
    different routes to `J` -- so landing on the same parameters is a real
    cross-check on the Jacobian this tier has to be given."""
    var ctx = _cpu()
    var p0 = Static[dtype, 2](ctx, [1.0, 1.0])
    var tensor_fit = least_squares[dtype, 2, 4, _exp_residuals, _exp_jacobian](
        p0
    )

    var start = Array[Float64, 2](fill=0.0)
    start[0] = 1.0
    start[1] = 1.0
    var array_fit = array_least_squares[2, 4, _array_residuals](start)

    assert_true(tensor_fit.converged)
    assert_true(array_fit.converged)
    var got = tensor_fit.x.to_host()
    for j in range(2):
        assert_almost_equal(Float64(got[j]), array_fit.x[j], atol=1e-8)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
