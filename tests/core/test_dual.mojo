"""Tests for `numax.Dual` against closed-form derivatives.

`comptime InnerT = Plain[dtype, width]` and `comptime D = Dual[InnerT]` give
first-order duals their old, flat shape (`D(value, deriv)` where both fields
are raw-SIMD-backed `Plain`s) without every test spelling out
`Dual[Plain[dtype, width]]`. The nested-`Dual` tests at the bottom build a
second `comptime` alias, `Dual[D]`, on top of that.
"""

from std.testing import TestSuite, assert_almost_equal

from numax import Dual, Plain, f64

comptime dtype = DType.float64
comptime width = 1
comptime InnerT = Plain[dtype, width]
comptime D = Dual[InnerT]


def v(x: Float64) -> InnerT:
    return InnerT(SIMD[dtype, width](x))


def test_one() raises:
    var one = D.one()
    assert_almost_equal(one.value.v, SIMD[dtype, width](1))
    assert_almost_equal(one.deriv.v, SIMD[dtype, width](0))


def test_add_sums_derivatives() raises:
    # f(x) = x, g(x) = 2x -> (f+g)'(x) = 3
    var f = D(v(5), v(1))
    var g = D(v(10), v(2))
    var h = f + g
    assert_almost_equal(h.value.v, SIMD[dtype, width](15))
    assert_almost_equal(h.deriv.v, SIMD[dtype, width](3))


def test_mul_is_product_rule() raises:
    # f(x) = x^2 at x=3: value 9, derivative 2x=6.
    var x = D(v(3), v(1))
    var f = x * x
    assert_almost_equal(f.value.v, SIMD[dtype, width](9))
    assert_almost_equal(f.deriv.v, SIMD[dtype, width](6))


def test_neg() raises:
    var x = D(v(3), v(1))
    var f = -x
    assert_almost_equal(f.value.v, SIMD[dtype, width](-3))
    assert_almost_equal(f.deriv.v, SIMD[dtype, width](-1))


def test_div_is_quotient_rule() raises:
    # f(x) = 1/x at x=2: value 0.5, derivative -1/x^2 = -0.25.
    var one = D.one()
    var x = D(v(2), v(1))
    var f = one / x
    assert_almost_equal(f.value.v, SIMD[dtype, width](0.5))
    assert_almost_equal(f.deriv.v, SIMD[dtype, width](-0.25))


def test_exp_derivative_is_itself() raises:
    # d/dx[exp(x)] = exp(x).
    var x = D(v(1.5), v(1))
    var f = x.exp()
    assert_almost_equal(f.value.v, f.deriv.v)


def test_ln_derivative_is_reciprocal() raises:
    # d/dx[ln(x)] = 1/x.
    var x = D(v(4), v(1))
    var f = x.ln()
    assert_almost_equal(f.deriv.v, SIMD[dtype, width](0.25))


def test_ln_undoes_exp() raises:
    var x = D(v(0.75), v(1))
    var f = x.exp().ln()
    assert_almost_equal(f.value.v, x.value.v)
    assert_almost_equal(f.deriv.v, x.deriv.v)


# --- Second-order autodiff: `Dual[D]`, a `Dual` nested inside a `Dual` ---
#
# Seeding the outer `Dual` at `deriv = D.one()` and the inner `D` at
# `deriv = 1` means the outer `.value` is `(f(x), f'(x))` and the outer
# `.deriv` is `(f'(x), f''(x))` -- so `.deriv.deriv` is the second
# derivative, with no change to `gaussian`, `exp`, or any other kernel.
comptime D2 = Dual[D]


def seed2(x: Float64) -> D2:
    # value = f(x)-tracking dual seeded to compute a first derivative
    # (deriv=1); deriv = a dual seeded at (1, 0), i.e. d/dx of "the first
    # derivative" starts at 1 with its own derivative (the second
    # derivative) starting at 0.
    return D2(D(v(x), v(1)), D(v(1), v(0)))


def test_nested_dual_square_gives_value_first_and_second_derivative() raises:
    # f(x) = x^2: f(3)=9, f'(3)=6, f''(3)=2.
    var x = seed2(3)
    var f = x * x
    assert_almost_equal(f.value.value.v, SIMD[dtype, width](9))
    assert_almost_equal(f.value.deriv.v, SIMD[dtype, width](6))
    assert_almost_equal(f.deriv.deriv.v, SIMD[dtype, width](2))


def test_nested_dual_exp_second_derivative_is_itself() raises:
    # d^2/dx^2[exp(x)] = exp(x).
    var x = seed2(1.2)
    var f = x.exp()
    assert_almost_equal(f.value.value.v, f.deriv.deriv.v)


def test_seed_carries_a_unit_derivative() raises:
    var x = Dual[f64].seed(0.5)
    assert_almost_equal(x.value.v, 0.5)
    assert_almost_equal(x.deriv.v, 1.0)


def test_seed_matches_the_two_argument_constructor() raises:
    comptime P = Plain[DType.float64, 1]
    var seeded = Dual[P].seed(2.0)
    var spelled = Dual[P](P(2.0), P.one())
    assert_almost_equal(seeded.value.v, spelled.value.v)
    assert_almost_equal(seeded.deriv.v, spelled.deriv.v)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
