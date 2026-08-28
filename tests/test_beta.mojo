"""Tests for `numax.beta`.

Every expected value here is a closed form the incomplete beta reduces to
at particular parameters, not a table copied from another implementation --
`I_x(1,1) = x` (the uniform CDF), `I_x(a,a) = 1/2` at `x = 1/2` (the Beta
distribution's symmetry), `I_x(a,1) = x^a`, and the recurrence relating
`I_x(a,b)` to `I_x(a+1,b)`.
"""

from std.math import exp as exp_f64
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Dual, Plain, beta, betainc, betaincc

comptime dtype = DType.float64
comptime width = 1
comptime D = Dual[Plain[dtype, width]]


def pv(x: Float64) -> Plain[dtype, width]:
    return Plain[dtype, width](SIMD[dtype, width](x))


def test_beta_at_small_integers() raises:
    # B(1,1) = 1, B(2,3) = 1/12, B(3,3) = 1/30.
    assert_almost_equal(
        beta(pv(1), pv(1)).v, SIMD[dtype, width](1.0), atol=1e-10
    )
    assert_almost_equal(
        beta(pv(2), pv(3)).v, SIMD[dtype, width](1.0 / 12.0), atol=1e-10
    )
    assert_almost_equal(
        beta(pv(3), pv(3)).v, SIMD[dtype, width](1.0 / 30.0), atol=1e-10
    )


def test_beta_at_one_half_is_pi() raises:
    # B(1/2,1/2) = Gamma(1/2)^2/Gamma(1) = pi.
    assert_almost_equal(
        beta(pv(0.5), pv(0.5)).v,
        SIMD[dtype, width](3.141592653589793),
        atol=1e-9,
    )


def test_beta_is_symmetric() raises:
    assert_almost_equal(
        beta(pv(2.5), pv(4.25)).v, beta(pv(4.25), pv(2.5)).v, atol=1e-12
    )


def test_betainc_with_both_parameters_one_is_the_identity() raises:
    # I_x(1,1) = x -- the uniform distribution's CDF.
    for x64 in [0.05, 0.25, 0.5, 0.75, 0.95]:
        assert_almost_equal(
            betainc(pv(x64), pv(1), pv(1)).v,
            SIMD[dtype, width](x64),
            atol=1e-10,
        )


def test_betainc_is_one_half_at_the_symmetric_midpoint() raises:
    # I_{1/2}(a,a) = 1/2 for any a, by the Beta distribution's symmetry
    # about 1/2 when its two shape parameters agree.
    for a64 in [0.5, 1.0, 2.0, 7.5]:
        assert_almost_equal(
            betainc(pv(0.5), pv(a64), pv(a64)).v,
            SIMD[dtype, width](0.5),
            atol=1e-9,
        )


def test_betainc_with_b_equal_one_is_a_power() raises:
    # I_x(a,1) = x^a.
    for a64 in [1.5, 3.0, 6.0]:
        for x64 in [0.2, 0.6, 0.9]:
            assert_almost_equal(
                betainc(pv(x64), pv(a64), pv(1)).v,
                SIMD[dtype, width](x64**a64),
                atol=1e-9,
            )


def test_betainc_with_a_equal_one_is_one_minus_a_power() raises:
    # I_x(1,b) = 1 - (1-x)^b.
    for b64 in [2.0, 4.5]:
        for x64 in [0.1, 0.5, 0.85]:
            assert_almost_equal(
                betainc(pv(x64), pv(1), pv(b64)).v,
                SIMD[dtype, width](1.0 - (1.0 - x64) ** b64),
                atol=1e-9,
            )


def test_betainc_at_the_endpoints() raises:
    # I_0 = 0 and I_1 = 1 for any parameters -- the case where the
    # discarded side of the blend would be an infinity if its argument
    # weren't clamped first.
    for a64 in [0.5, 2.0, 9.0]:
        for b64 in [0.75, 3.0]:
            assert_almost_equal(
                betainc(pv(0.0), pv(a64), pv(b64)).v,
                SIMD[dtype, width](0.0),
                atol=1e-12,
            )
            assert_almost_equal(
                betainc(pv(1.0), pv(a64), pv(b64)).v,
                SIMD[dtype, width](1.0),
                atol=1e-12,
            )


def test_betainc_reflection_symmetry() raises:
    # I_x(a,b) + I_{1-x}(b,a) = 1. This crosses the blend's own threshold
    # (the two calls land on opposite sides of it), so it checks the two
    # branches agree rather than just checking one of them twice.
    for x64 in [0.05, 0.3, 0.5, 0.62, 0.97]:
        var lhs = Float64(betainc(pv(x64), pv(2.5), pv(4.0)).v)
        var rhs = Float64(betainc(pv(1.0 - x64), pv(4.0), pv(2.5)).v)
        assert_almost_equal(lhs + rhs, 1.0, atol=1e-9)


def test_betainc_matches_the_parameter_recurrence() raises:
    # I_x(a,b) = I_x(a+1,b) + x^a * (1-x)^b / (a * B(a,b)) -- a standard
    # identity that ties two independent evaluations together.
    var a64 = 3.0
    var b64 = 2.0
    var x64 = 0.4
    var lhs = Float64(betainc(pv(x64), pv(a64), pv(b64)).v)
    var rhs = Float64(betainc(pv(x64), pv(a64 + 1.0), pv(b64)).v) + (
        x64**a64
        * (1.0 - x64) ** b64
        / (a64 * Float64(beta(pv(a64), pv(b64)).v))
    )
    assert_almost_equal(lhs, rhs, atol=1e-9)


def test_betainc_is_monotone() raises:
    var previous = -1.0
    for i in range(20):
        var x64 = Float64(i) / 19.0
        var value = Float64(betainc(pv(x64), pv(2.0), pv(5.0)).v)
        assert_true(value >= previous)
        previous = value


def test_betaincc_is_one_minus_betainc() raises:
    assert_almost_equal(
        betaincc(pv(0.3), pv(2.0), pv(3.0)).v,
        SIMD[dtype, width](1.0) - betainc(pv(0.3), pv(2.0), pv(3.0)).v,
        atol=1e-12,
    )


def test_betainc_derivative_matches_the_density() raises:
    # d/dx[I_x(a,b)] = x^(a-1) * (1-x)^(b-1) / B(a,b) -- the Beta density.
    # `Dual` differentiates the continued fraction, its Lentz guards, and
    # the branchless blend all at once, so agreeing with the closed-form
    # density is a real check on all three.
    var a64 = 2.0
    var b64 = 3.0
    for x64 in [0.2, 0.45, 0.8]:
        var d = Float64(
            betainc(
                D(pv(x64), pv(1.0)), D.constant(a64), D.constant(b64)
            ).deriv.v
        )
        var expected = (
            x64 ** (a64 - 1.0)
            * (1.0 - x64) ** (b64 - 1.0)
            / Float64(beta(pv(a64), pv(b64)).v)
        )
        assert_almost_equal(d, expected, atol=1e-7)


def test_beta_relates_to_the_gamma_ratio_numerically() raises:
    # B(a,b) = exp(lgamma(a)+lgamma(b)-lgamma(a+b)); check one value
    # against a directly computed reference to catch a sign or ordering
    # slip in the log-space arithmetic.
    var expected = exp_f64(0.0) * (1.0 / 12.0)
    assert_almost_equal(
        beta(pv(2.0), pv(3.0)).v, SIMD[dtype, width](expected), atol=1e-10
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
