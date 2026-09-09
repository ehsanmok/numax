"""Tests for `numax.optimize.minimize`, the `Tensor` tier.

The claims worth pinning are the ones that are not ordinary correctness.

**The two tiers are the same algorithm on different data.** The `Array`
tier reads its gradient off `Gradient`; this one takes `jac` as an argument
because a `Tensor` cannot hold a conformer. Given a hand-written `jac` that
is the same derivative, the two must land on the same minimum.

**The `DeviceContext` lifetime trap must stay fixed.** `findings.mdc`
records that `Tensor.context()` does not outlive the tensor it came from, so
the driver keeps its iterate in a `List[Float64]` and takes `x0` borrowed.
`test_the_context_survives_a_whole_run` is the regression: `x0` is the only
tensor keeping the context alive across every iteration.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from std.collections import Array

from max.gpu.host import DeviceContext

from numax import FloatLike
from numax.core.array import Static
from numax.optimize import TensorMinimizeResult, minimize
from numax.optimize.array import minimize as array_minimize

comptime dtype = DType.float64


def _rosenbrock(
    p: Static[dtype, 2], ctx: DeviceContext
) raises -> Scalar[dtype]:
    """`(1 - x)^2 + 100 (y - x^2)^2`, minimum `(1, 1)`."""
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


def _rosenbrock_array[U: FloatLike](v: Array[U, 2]) -> U:
    """The same objective as an ordinary `FloatLike` kernel, so the `Array`
    tier can differentiate it and the two tiers can be compared."""
    var a = U.one() - v[0]
    var b = v[1] - (v[0] * v[0])
    return a * a + U.constant(100.0) * b * b


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


def _rosenbrock_start(ctx: DeviceContext) raises -> Static[dtype, 2]:
    return Static[dtype, 2](ctx, [-1.2, 1.0])


def _bowl_start(ctx: DeviceContext) raises -> Static[dtype, 3]:
    return Static[dtype, 3](ctx, [-3.0, 4.0, -1.0])


def test_bfgs_minimizes_rosenbrock() raises:
    var ctx = DeviceContext(api="cpu")
    var start = _rosenbrock_start(ctx)
    var result = minimize[dtype, 2, _rosenbrock, _rosenbrock_jac](start)

    assert_true(result.converged)
    var got = result.x.to_host()
    assert_almost_equal(Float64(got[0]), 1.0, atol=1e-5)
    assert_almost_equal(Float64(got[1]), 1.0, atol=1e-5)


def test_cg_minimizes_rosenbrock() raises:
    var ctx = DeviceContext(api="cpu")
    var start = _rosenbrock_start(ctx)
    var result = minimize[
        dtype, 2, _rosenbrock, _rosenbrock_jac, method="cg", gpu=False
    ](start)

    assert_true(result.converged)
    var got = result.x.to_host()
    assert_almost_equal(Float64(got[0]), 1.0, atol=1e-4)
    assert_almost_equal(Float64(got[1]), 1.0, atol=1e-4)


def test_bfgs_minimizes_a_quadratic_bowl() raises:
    var ctx = DeviceContext(api="cpu")
    var start = _bowl_start(ctx)
    var result = minimize[dtype, 3, _bowl, _bowl_jac](start)

    assert_true(result.converged)
    var got = result.x.to_host()
    assert_almost_equal(Float64(got[0]), 1.0, atol=1e-6)
    assert_almost_equal(Float64(got[1]), -2.0, atol=1e-6)
    assert_almost_equal(Float64(got[2]), 3.0, atol=1e-6)


def test_cg_minimizes_a_quadratic_bowl() raises:
    var ctx = DeviceContext(api="cpu")
    var start = _bowl_start(ctx)
    var result = minimize[dtype, 3, _bowl, _bowl_jac, method="cg"](start)

    assert_true(result.converged)
    var got = result.x.to_host()
    assert_almost_equal(Float64(got[0]), 1.0, atol=1e-6)
    assert_almost_equal(Float64(got[1]), -2.0, atol=1e-6)
    assert_almost_equal(Float64(got[2]), 3.0, atol=1e-6)


def test_starting_at_the_minimum_converges_immediately() raises:
    var ctx = DeviceContext(api="cpu")
    var start = Static[dtype, 3](ctx, [1.0, -2.0, 3.0])
    var result = minimize[dtype, 3, _bowl, _bowl_jac](start)

    assert_true(result.converged)
    assert_equal(result.iterations, 0)


def test_the_two_tiers_agree() raises:
    """The `Array` tier reads its gradient off `Gradient`; this tier is
    handed the same derivative as `jac`. Same algorithm, same answer."""
    var ctx = DeviceContext(api="cpu")
    var start = _rosenbrock_start(ctx)
    var tensor_side = minimize[dtype, 2, _rosenbrock, _rosenbrock_jac](start)

    var array_start = Array[Float64, 2](fill=0)
    array_start[0] = -1.2
    array_start[1] = 1.0
    var array_side = array_minimize[2, _rosenbrock_array](array_start^)

    assert_true(tensor_side.converged)
    assert_true(array_side.converged)
    var got = tensor_side.x.to_host()
    assert_almost_equal(Float64(got[0]), array_side.x[0], atol=1e-6)
    assert_almost_equal(Float64(got[1]), array_side.x[1], atol=1e-6)


def test_the_context_survives_a_whole_run() raises:
    """The `findings.mdc:350` regression. `x0` is borrowed and is the only
    thing keeping the `DeviceContext` alive across the loop, so a driver
    that reassigned the tensor it took the context from would fault here
    rather than fail an assertion."""
    var ctx = DeviceContext(api="cpu")
    var start = _bowl_start(ctx)
    var result = minimize[dtype, 3, _bowl, _bowl_jac](start, max_iter=150)

    assert_true(result.converged)
    # `start` must still be usable: the run borrowed it and consumed
    # nothing.
    var original = start.to_host()
    assert_almost_equal(Float64(original[0]), -3.0, atol=0.0)


def test_grad_norm_is_the_quantity_that_converged() raises:
    var ctx = DeviceContext(api="cpu")
    var start = _bowl_start(ctx)
    var result = minimize[dtype, 3, _bowl, _bowl_jac](start)
    assert_true(result.converged)
    assert_true(result.grad_norm < 1e-8)


def test_a_tight_iteration_cap_reports_failure() raises:
    var ctx = DeviceContext(api="cpu")
    var start = _rosenbrock_start(ctx)
    var result = minimize[dtype, 2, _rosenbrock, _rosenbrock_jac](
        start, max_iter=2
    )
    assert_true(not result.converged)
    assert_true(result.iterations <= 2)


def test_an_unknown_method_raises() raises:
    var ctx = DeviceContext(api="cpu")
    var start = _rosenbrock_start(ctx)
    var raised = False
    try:
        _ = minimize[
            dtype, 2, _rosenbrock, _rosenbrock_jac, method="nelder-mead"
        ](start)
    except e:
        raised = True
        assert_true("unknown method" in String(e))
        assert_true("Array-tier only" in String(e))
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
