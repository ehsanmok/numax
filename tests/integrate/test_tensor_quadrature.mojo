"""Tests for `numax.integrate`'s sample-taking rules over `Tensor`.

Checked against `scipy.integrate` 1.18's digits on the same samples --
including the even-count Simpson, where SciPy's Cartwright correction is
the rule and a trapezoid on the last interval would be wrong in the third
digit -- and against exact integrals of polynomials, where each rule's order
says what "exact" means.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.array import Static
from numax.integrate import cumulative_trapezoid, simpson, trapezoid

comptime dtype = DType.float64


def _t[n: Int](values: List[Float64]) raises -> Static[dtype, n]:
    var ctx = DeviceContext(api="cpu")
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype](values[i]))
    return Static[dtype, n](ctx, out^)


def test_trapezoid_with_dx_matches_scipy() raises:
    # x^2 at x = 1..5: scipy.integrate.trapezoid(y, dx=1) == 42.0
    var y = _t[5]([1.0, 4.0, 9.0, 16.0, 25.0])
    assert_almost_equal(trapezoid(y, Scalar[dtype](1.0)), Scalar[dtype](42.0))
    # and at dx = 0.5 over six samples: 36.25
    var y6 = _t[6]([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
    assert_almost_equal(trapezoid(y6, Scalar[dtype](0.5)), Scalar[dtype](36.25))


def test_trapezoid_at_non_uniform_points_matches_scipy() raises:
    # y = x^2 at x = [0, 0.5, 1.5, 2.0, 3.5]: 15.0625
    var x = _t[5]([0.0, 0.5, 1.5, 2.0, 3.5])
    var y = _t[5]([0.0, 0.25, 2.25, 4.0, 12.25])
    assert_almost_equal(trapezoid(y, x), Scalar[dtype](15.0625))


def test_simpson_is_exact_for_a_quadratic_at_an_odd_count() raises:
    # scipy.integrate.simpson(x^2 at 1..5, dx=1) == 41.3333, the integral.
    var y = _t[5]([1.0, 4.0, 9.0, 16.0, 25.0])
    assert_almost_equal(
        simpson(y, Scalar[dtype](1.0)),
        Scalar[dtype](41.33333333333333),
        atol=1e-12,
    )


def test_simpson_at_an_even_count_uses_cartwrights_correction() raises:
    # scipy.integrate.simpson(x^2 at 1..6, dx=1) == 71.6667, still exact.
    # A trapezoid on the last interval would give 71.8333.
    var y = _t[6]([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
    assert_almost_equal(
        simpson(y, Scalar[dtype](1.0)),
        Scalar[dtype](71.66666666666666),
        atol=1e-12,
    )


def test_simpson_at_non_uniform_points_matches_scipy() raises:
    # scipy.integrate.simpson(x^2, x=[0, .5, 1.5, 2, 3.5]) == 14.2917, exact.
    var x = _t[5]([0.0, 0.5, 1.5, 2.0, 3.5])
    var y = _t[5]([0.0, 0.25, 2.25, 4.0, 12.25])
    assert_almost_equal(
        simpson(y, x), Scalar[dtype](14.291666666666666), atol=1e-12
    )


def test_cumulative_trapezoid_returns_one_value_per_interval() raises:
    # scipy: cumulative_trapezoid(x^2 at 1..5, dx=1) == [2.5, 9, 21.5, 42]
    var y = _t[5]([1.0, 4.0, 9.0, 16.0, 25.0])
    var running = cumulative_trapezoid(y, Scalar[dtype](1.0))
    assert_equal(running.size(), 4)
    var out = running.to_host()
    var expected = [2.5, 9.0, 21.5, 42.0]
    for i in range(4):
        assert_almost_equal(out[i], Scalar[dtype](expected[i]))
    # Its last entry is `trapezoid`, which is what "cumulative" means.
    assert_almost_equal(out[3], trapezoid(y, Scalar[dtype](1.0)))


def test_cumulative_trapezoid_with_initial_prepends_a_zero() raises:
    # scipy's initial=0: [0, 2.5, 9, 21.5, 42], as long as y.
    var y = _t[5]([1.0, 4.0, 9.0, 16.0, 25.0])
    var running = cumulative_trapezoid[initial=True](y, Scalar[dtype](1.0))
    assert_equal(running.size(), 5)
    var out = running.to_host()
    assert_almost_equal(out[0], Scalar[dtype](0.0))
    assert_almost_equal(out[4], Scalar[dtype](42.0))


def test_cumulative_trapezoid_at_non_uniform_points_matches_scipy() raises:
    # scipy: [0.0625, 1.3125, 2.875, 15.0625]
    var x = _t[5]([0.0, 0.5, 1.5, 2.0, 3.5])
    var y = _t[5]([0.0, 0.25, 2.25, 4.0, 12.25])
    var out = cumulative_trapezoid(y, x).to_host()
    var expected = [0.0625, 1.3125, 2.875, 15.0625]
    for i in range(4):
        assert_almost_equal(out[i], Scalar[dtype](expected[i]))


def test_a_mismatched_x_raises() raises:
    var x = _t[4]([0.0, 1.0, 2.0, 3.0])
    var y = _t[5]([1.0, 4.0, 9.0, 16.0, 25.0])
    var raised = False
    try:
        _ = trapezoid(y, x)
    except:
        raised = True
    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
