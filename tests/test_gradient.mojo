"""Tests for `numax.Gradient` against closed-form partial derivatives.

`comptime InnerT = Plain[dtype, width]` and `comptime G2 = Gradient[InnerT,
2]` give two-variable gradients a fixed shape without every test spelling
out `Gradient[Plain[dtype, width], 2]`. Every test seeds via `variable`'s
`Float64` overload (`G2.variable(2.0, 0)`) rather than building an `Inner`
by hand first -- `test_variable_float64_overload_matches_inner_overload`
below is the one test checking that overload directly against the other.
"""

from std.math import cos
from std.math import sin
from std.testing import TestSuite, assert_almost_equal

from numax import Gradient, Plain

comptime dtype = DType.float64
comptime width = 1
comptime InnerT = Plain[dtype, width]
comptime G2 = Gradient[InnerT, 2]


def var_x(x: Float64) -> G2:
    return G2.variable(x, 0)


def var_y(y: Float64) -> G2:
    return G2.variable(y, 1)


def test_one_has_zero_gradient() raises:
    var one = G2.one()
    assert_almost_equal(one.value.v, SIMD[dtype, width](1))
    assert_almost_equal(one.grad[0].v, SIMD[dtype, width](0))
    assert_almost_equal(one.grad[1].v, SIMD[dtype, width](0))


def test_variable_is_one_hot() raises:
    var x = var_x(5)
    var y = var_y(7)
    assert_almost_equal(x.value.v, SIMD[dtype, width](5))
    assert_almost_equal(x.grad[0].v, SIMD[dtype, width](1))
    assert_almost_equal(x.grad[1].v, SIMD[dtype, width](0))
    assert_almost_equal(y.value.v, SIMD[dtype, width](7))
    assert_almost_equal(y.grad[0].v, SIMD[dtype, width](0))
    assert_almost_equal(y.grad[1].v, SIMD[dtype, width](1))


def test_variable_float64_overload_matches_inner_overload() raises:
    # G2.variable(2.0, 0) (the Float64 convenience overload) should be
    # identical to seeding the Inner-typed overload with Inner.constant(2.0)
    # by hand.
    var from_literal = G2.variable(2.0, 0)
    var from_inner = G2.variable(InnerT.constant(2.0), 0)
    assert_almost_equal(from_literal.value.v, from_inner.value.v)
    assert_almost_equal(from_literal.grad[0].v, from_inner.grad[0].v)
    assert_almost_equal(from_literal.grad[1].v, from_inner.grad[1].v)


def test_add_sums_gradients() raises:
    # f(x, y) = x + 2y -> grad = (1, 2), independent of the point.
    var x = var_x(3)
    var y = var_y(4)
    var f = x + y + y
    assert_almost_equal(f.value.v, SIMD[dtype, width](11))
    assert_almost_equal(f.grad[0].v, SIMD[dtype, width](1))
    assert_almost_equal(f.grad[1].v, SIMD[dtype, width](2))


def test_mul_is_multivariable_product_rule() raises:
    # f(x, y) = x*y at (2, 3): value 6, df/dx = y = 3, df/dy = x = 2.
    var x = var_x(2)
    var y = var_y(3)
    var f = x * y
    assert_almost_equal(f.value.v, SIMD[dtype, width](6))
    assert_almost_equal(f.grad[0].v, SIMD[dtype, width](3))
    assert_almost_equal(f.grad[1].v, SIMD[dtype, width](2))


def test_neg() raises:
    var x = var_x(2)
    var y = var_y(3)
    var f = -(x * y)
    assert_almost_equal(f.value.v, SIMD[dtype, width](-6))
    assert_almost_equal(f.grad[0].v, SIMD[dtype, width](-3))
    assert_almost_equal(f.grad[1].v, SIMD[dtype, width](-2))


def test_div_is_multivariable_quotient_rule() raises:
    # f(x, y) = x/y at (6, 2): value 3, df/dx = 1/y = 0.5, df/dy = -x/y^2 =
    # -1.5.
    var x = var_x(6)
    var y = var_y(2)
    var f = x / y
    assert_almost_equal(f.value.v, SIMD[dtype, width](3))
    assert_almost_equal(f.grad[0].v, SIMD[dtype, width](0.5))
    assert_almost_equal(f.grad[1].v, SIMD[dtype, width](-1.5))


def test_two_variable_transcendental_kernel_matches_closed_form() raises:
    # f(x, y) = x^2*y + sin(x*y) at (2, 3):
    #   value = 12 + sin(6)
    #   df/dx = 2xy + y*cos(x*y)
    #   df/dy = x^2 + x*cos(x*y)
    var x = var_x(2)
    var y = var_y(3)
    var f = x * x * y + (x * y).sin()

    var x0 = 2.0
    var y0 = 3.0
    var expected_value = x0 * x0 * y0 + sin(x0 * y0)
    var expected_dx = 2 * x0 * y0 + y0 * cos(x0 * y0)
    var expected_dy = x0 * x0 + x0 * cos(x0 * y0)

    assert_almost_equal(f.value.v, SIMD[dtype, width](expected_value))
    assert_almost_equal(f.grad[0].v, SIMD[dtype, width](expected_dx))
    assert_almost_equal(f.grad[1].v, SIMD[dtype, width](expected_dy))


def test_exp_gradient_matches_value_times_input_gradients() raises:
    # d/dx_i[exp(f)] = exp(f) * df/dx_i; with f = x*y, df/dx = y, df/dy = x.
    var x = var_x(0.5)
    var y = var_y(1.5)
    var f = (x * y).exp()
    assert_almost_equal(f.grad[0].v, f.value.v * SIMD[dtype, width](1.5))
    assert_almost_equal(f.grad[1].v, f.value.v * SIMD[dtype, width](0.5))


def test_ln_undoes_exp_with_full_gradient() raises:
    var x = var_x(0.75)
    var y = var_y(1.25)
    var xy = x * y
    var f = xy.exp().ln()
    assert_almost_equal(f.value.v, xy.value.v)
    assert_almost_equal(f.grad[0].v, xy.grad[0].v)
    assert_almost_equal(f.grad[1].v, xy.grad[1].v)


def test_abs_gradient_flips_sign_with_input() raises:
    var x = var_x(-2)
    var y = var_y(3)
    var f = (x * y).abs()
    # x*y = -6 < 0, so |x*y| = -(x*y): gradient is negated too.
    assert_almost_equal(f.value.v, SIMD[dtype, width](6))
    assert_almost_equal(f.grad[0].v, SIMD[dtype, width](-3))
    assert_almost_equal(f.grad[1].v, SIMD[dtype, width](2))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
