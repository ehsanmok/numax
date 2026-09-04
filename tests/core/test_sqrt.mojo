"""Tests for `FloatLike.sqrt` across every conformer.

`sqrt` was the trait's last addition, and it went in for two reasons worth
checking separately: correctness on each conformer's own terms, and the
claim that it beats the `exp(0.5*ln(x))` workaround it replaced (checked in
`test_beats_the_exp_ln_workaround` below).
"""

from std.math import sqrt as sqrt_f64
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from numax import Compensated, Complex, Decimal, Dual, Gradient, Plain, f32

comptime dtype = DType.float64
comptime width = 1
comptime P = Plain[dtype, width]
comptime D = Dual[P]
comptime C = Complex[P]
comptime G2 = Gradient[P, 2]


def pv(x: Float64) -> P:
    return P.constant(x)


def test_plain_matches_std_math() raises:
    for x64 in [0.0, 1e-8, 0.25, 1.0, 2.0, 1e6]:
        assert_almost_equal(pv(x64).sqrt().v, SIMD[dtype, width](sqrt_f64(x64)))


def test_plain_squares_back() raises:
    for x64 in [0.3, 7.0, 1234.5]:
        var r = pv(x64).sqrt()
        assert_almost_equal((r * r).v, SIMD[dtype, width](x64), atol=1e-10)


def test_dual_derivative_is_one_over_two_sqrt() raises:
    for x64 in [0.5, 2.0, 9.0]:
        var r = D(pv(x64), pv(1.0)).sqrt()
        assert_almost_equal(r.value.v, SIMD[dtype, width](sqrt_f64(x64)))
        assert_almost_equal(
            r.deriv.v,
            SIMD[dtype, width](0.5 / sqrt_f64(x64)),
            atol=1e-12,
        )


def test_dual_chain_rule_through_a_composite() raises:
    # d/dx[sqrt(1 + x^2)] = x / sqrt(1 + x^2).
    var x64 = 1.7
    var x = D(pv(x64), pv(1.0))
    var f = (D.one() + x * x).sqrt()
    assert_almost_equal(
        f.deriv.v,
        SIMD[dtype, width](x64 / sqrt_f64(1.0 + x64 * x64)),
        atol=1e-12,
    )


def test_gradient_partials() raises:
    # f(x,y) = sqrt(x^2 + y^2); df/dx = x/f, df/dy = y/f.
    var x = G2.variable(3.0, 0)
    var y = G2.variable(4.0, 1)
    var f = (x * x + y * y).sqrt()
    assert_almost_equal(f.value.v, SIMD[dtype, width](5.0), atol=1e-12)
    assert_almost_equal(f.grad[0].v, SIMD[dtype, width](0.6), atol=1e-12)
    assert_almost_equal(f.grad[1].v, SIMD[dtype, width](0.8), atol=1e-12)


def test_compensated_squares_back_to_the_input() raises:
    for x64 in [0.5, 2.0, 10.0, 1e5]:
        var r = Compensated[dtype, width](
            SIMD[dtype, width](x64), SIMD[dtype, width](0)
        ).sqrt()
        var squared = r * r
        var recovered = Float64(squared.value) + Float64(squared.error)
        assert_almost_equal(recovered, x64, atol=1e-12)


def test_compensated_zero_is_zero_not_nan() raises:
    var r = Compensated[dtype, width](
        SIMD[dtype, width](0), SIMD[dtype, width](0)
    ).sqrt()
    assert_almost_equal(r.value, SIMD[dtype, width](0))
    assert_almost_equal(r.error, SIMD[dtype, width](0))


def test_compensated_beats_plain_float32() raises:
    # The reason `Compensated.sqrt` is a from-scratch Newton refinement
    # rather than a widened call: at float32, the double-double result
    # should reconstruct sqrt(2) far more accurately than a single float32
    # can hold.
    var c = Compensated[f32, 1](SIMD[f32, 1](2.0), SIMD[f32, 1](0)).sqrt()
    var reference = 1.4142135623730951
    var comp_err = abs((Float64(c.value) + Float64(c.error)) - reference)
    var plain_err = abs(
        Float64(Plain[f32, 1].constant(2.0).sqrt().v) - reference
    )
    assert_true(comp_err < plain_err)
    assert_true(comp_err < 1e-14)


def test_decimal_squares_back() raises:
    # `Decimal.__mul__`/`__truediv__` floor rather than round to nearest,
    # so the round trip drifts by a few raw units (10^-scale) -- checked
    # against that tolerance, not against exactness.
    comptime scale = 6
    comptime Dec = Decimal[width, scale]
    comptime factor = 1_000_000
    for x64 in [2.0, 9.0, 0.25]:
        var r = Dec.constant(x64).sqrt()
        var squared = r * r
        var drift = abs(
            Int64(squared.raw[0]) - Int64(Int(x64 * Float64(factor)))
        )
        assert_true(
            drift <= 100,
            msg=String("sqrt round trip off by too much: ", squared.raw),
        )


def test_decimal_non_positive_is_zero() raises:
    comptime Dec = Decimal[width, 6]
    assert_equal(Dec.constant(0.0).sqrt().raw, SIMD[DType.int64, width](0))


def test_complex_principal_square_root() raises:
    # sqrt(-4) = 2i on the principal branch.
    var r = C(pv(-4.0), pv(0.0)).sqrt()
    assert_almost_equal(r.re.v, SIMD[dtype, width](0.0), atol=1e-12)
    assert_almost_equal(r.im.v, SIMD[dtype, width](2.0), atol=1e-12)

    # sqrt(i) = (1+i)/sqrt(2).
    var s = C(pv(0.0), pv(1.0)).sqrt()
    var half_root2 = sqrt_f64(0.5)
    assert_almost_equal(s.re.v, SIMD[dtype, width](half_root2), atol=1e-12)
    assert_almost_equal(s.im.v, SIMD[dtype, width](half_root2), atol=1e-12)


def test_complex_squares_back() raises:
    for pair in [(3.0, 4.0), (-2.0, 5.0), (1.5, -0.25)]:
        var z = C(pv(pair[0]), pv(pair[1]))
        var r = z.sqrt()
        var back = r * r
        assert_almost_equal(back.re.v, z.re.v, atol=1e-10)
        assert_almost_equal(back.im.v, z.im.v, atol=1e-10)


def test_complex_of_a_positive_real_is_real() raises:
    var r = C(pv(9.0), pv(0.0)).sqrt()
    assert_almost_equal(r.re.v, SIMD[dtype, width](3.0), atol=1e-12)
    assert_almost_equal(r.im.v, SIMD[dtype, width](0.0), atol=1e-12)


def test_beats_the_exp_ln_workaround() raises:
    # The workaround `sqrt` replaced, kept here only as the comparison:
    # exp(0.5*ln(x)) round-trips through two transcendentals and loses
    # accuracy doing it.
    var worst_direct = 0.0
    var worst_workaround = 0.0
    for i in range(1, 200):
        var x64 = Float64(i) * 0.37
        var reference = sqrt_f64(x64)
        var direct = Float64(pv(x64).sqrt().v)
        var workaround = Float64((pv(0.5) * pv(x64).ln()).exp().v)
        worst_direct = max(worst_direct, abs(direct - reference))
        worst_workaround = max(worst_workaround, abs(workaround - reference))
    assert_almost_equal(worst_direct, 0.0, atol=1e-15)
    assert_true(worst_workaround > worst_direct)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
