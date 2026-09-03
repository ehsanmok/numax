"""Second derivatives across several variables, by nesting `Gradient` and
`Dual` -- no second-order code of their own required.

`Gradient[Dual[Plain], 2]` is `Gradient` whose `Inner` happens to be a
`Dual`. Seed variable `i` with value `Dual(x_i, u_i)` for some direction
`u`, and each component of the resulting gradient carries a derivative
alongside it:

    result.value.value  = f(x)
    result.value.deriv  = the directional derivative of f along u
    result.grad[i].value = df/dx_i
    result.grad[i].deriv = sum_j H[i,j] * u_j

so a one-hot `u = e_k` makes `result.grad[i].deriv` exactly `H[i,k]` -- one
column of the Hessian (equivalently one row, since H is symmetric) per
call. A general `u` gives a Hessian-vector product directly, without ever
forming H.

Both nesting orders are checked here, against each other and against the
closed-form Hessian of `f(x,y) = x^2*y + sin(x*y)`:

    f_xx = 2y - y^2*sin(xy)
    f_xy = f_yx = 2x + cos(xy) - x*y*sin(xy)
    f_yy = -x^2*sin(xy)
"""

from std.math import cos, sin
from std.testing import TestSuite, assert_almost_equal

from numax import Dual, Gradient, Plain

comptime dtype = DType.float64
comptime width = 1
comptime P = Plain[dtype, width]

# Gradient on the outside, Dual on the inside.
comptime DP = Dual[P]
comptime GD2 = Gradient[DP, 2]

# The mirror image: Dual on the outside, Gradient on the inside.
comptime G2 = Gradient[P, 2]
comptime DG2 = Dual[G2]

comptime X0 = 2.0
comptime Y0 = 3.0


def f_xx(x: Float64, y: Float64) -> Float64:
    return 2.0 * y - y * y * sin(x * y)


def f_xy(x: Float64, y: Float64) -> Float64:
    return 2.0 * x + cos(x * y) - x * y * sin(x * y)


def f_yy(x: Float64, y: Float64) -> Float64:
    return -x * x * sin(x * y)


def seed_gd(value: Float64, direction: Float64, index: Int) -> GD2:
    """Variable `index` at `value`, carrying direction component `u_index`."""
    return GD2.variable(DP(P.constant(value), P.constant(direction)), index)


def seed_dg(value: Float64, direction: Float64, index: Int) -> DG2:
    """The same seeding for the mirror nesting order: the `Gradient` is
    what gets the one-hot, the outer `Dual` carries `u_index`."""
    return DG2(G2.variable(value, index), G2.constant(direction))


def test_hessian_first_column_gradient_outside() raises:
    # u = e_0, so grad[i].deriv is H[i,0].
    var x = seed_gd(X0, 1.0, 0)
    var y = seed_gd(Y0, 0.0, 1)
    var f = x * x * y + (x * y).sin()

    assert_almost_equal(
        f.grad[0].deriv.v, SIMD[dtype, width](f_xx(X0, Y0)), atol=1e-10
    )
    assert_almost_equal(
        f.grad[1].deriv.v, SIMD[dtype, width](f_xy(X0, Y0)), atol=1e-10
    )


def test_hessian_second_column_gradient_outside() raises:
    # u = e_1, so grad[i].deriv is H[i,1].
    var x = seed_gd(X0, 0.0, 0)
    var y = seed_gd(Y0, 1.0, 1)
    var f = x * x * y + (x * y).sin()

    assert_almost_equal(
        f.grad[0].deriv.v, SIMD[dtype, width](f_xy(X0, Y0)), atol=1e-10
    )
    assert_almost_equal(
        f.grad[1].deriv.v, SIMD[dtype, width](f_yy(X0, Y0)), atol=1e-10
    )


def test_first_derivatives_still_come_out_of_the_nested_type() raises:
    # Nesting doesn't cost the first-order answer: grad[i].value is still
    # df/dx_i, and value.deriv is the directional derivative along u.
    var x = seed_gd(X0, 1.0, 0)
    var y = seed_gd(Y0, 0.0, 1)
    var f = x * x * y + (x * y).sin()

    var fx = 2.0 * X0 * Y0 + Y0 * cos(X0 * Y0)
    var fy = X0 * X0 + X0 * cos(X0 * Y0)

    assert_almost_equal(
        f.value.value.v,
        SIMD[dtype, width](X0 * X0 * Y0 + sin(X0 * Y0)),
        atol=1e-12,
    )
    assert_almost_equal(f.grad[0].value.v, SIMD[dtype, width](fx), atol=1e-12)
    assert_almost_equal(f.grad[1].value.v, SIMD[dtype, width](fy), atol=1e-12)
    # u = e_0 picks df/dx out as the directional derivative.
    assert_almost_equal(f.value.deriv.v, SIMD[dtype, width](fx), atol=1e-12)


def test_mirror_nesting_order_agrees() raises:
    # Dual[Gradient[...]] instead of Gradient[Dual[...]]: deriv.grad[i] is
    # the same H[i,k] the other order puts in grad[i].deriv.
    var x = seed_dg(X0, 1.0, 0)
    var y = seed_dg(Y0, 0.0, 1)
    var f = x * x * y + (x * y).sin()

    assert_almost_equal(
        f.deriv.grad[0].v, SIMD[dtype, width](f_xx(X0, Y0)), atol=1e-10
    )
    assert_almost_equal(
        f.deriv.grad[1].v, SIMD[dtype, width](f_xy(X0, Y0)), atol=1e-10
    )


def test_hessian_is_symmetric_across_both_orders() raises:
    # H[0,1] computed as "column 1, row 0" by one nesting order and as
    # "column 0, row 1" by the other -- equal by symmetry of mixed
    # partials, and a check that neither order is quietly transposed.
    var xa = seed_gd(X0, 0.0, 0)
    var ya = seed_gd(Y0, 1.0, 1)
    var fa = xa * xa * ya + (xa * ya).sin()

    var xb = seed_dg(X0, 1.0, 0)
    var yb = seed_dg(Y0, 0.0, 1)
    var fb = xb * xb * yb + (xb * yb).sin()

    assert_almost_equal(fa.grad[0].deriv.v, fb.deriv.grad[1].v, atol=1e-10)


def test_hessian_vector_product_without_forming_the_matrix() raises:
    # A general (non-one-hot) u gives H @ u directly in one pass.
    comptime u0 = 0.6
    comptime u1 = -1.4
    var x = seed_gd(X0, u0, 0)
    var y = seed_gd(Y0, u1, 1)
    var f = x * x * y + (x * y).sin()

    assert_almost_equal(
        f.grad[0].deriv.v,
        SIMD[dtype, width](f_xx(X0, Y0) * u0 + f_xy(X0, Y0) * u1),
        atol=1e-10,
    )
    assert_almost_equal(
        f.grad[1].deriv.v,
        SIMD[dtype, width](f_xy(X0, Y0) * u0 + f_yy(X0, Y0) * u1),
        atol=1e-10,
    )


def test_second_derivative_through_a_transcendental_chain() raises:
    # exp/ln/erf all have their own second-order behavior under nesting;
    # g(x,y) = exp(x*y) has H = [[y^2, 1+xy],[1+xy, x^2]] * exp(x*y).
    var x = seed_gd(1.5, 1.0, 0)
    var y = seed_gd(0.7, 0.0, 1)
    var g = (x * y).exp()

    var e = 2.718281828459045 ** (1.5 * 0.7)
    assert_almost_equal(
        g.grad[0].deriv.v, SIMD[dtype, width](0.7 * 0.7 * e), atol=1e-10
    )
    assert_almost_equal(
        g.grad[1].deriv.v,
        SIMD[dtype, width]((1.0 + 1.5 * 0.7) * e),
        atol=1e-10,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
