"""Tests for `numax.special.legendre` against the closed-form low-degree polynomials.
"""

from std.testing import TestSuite, assert_almost_equal

from numax import Dual, Plain, legendre_p

comptime dtype = DType.float64
comptime width = 1
comptime D = Dual[Plain[dtype, width]]


def pv(x: Float64) -> Plain[dtype, width]:
    return Plain[dtype, width].constant(x)


def test_low_degrees_match_closed_forms() raises:
    # P_0 = 1, P_1 = x, P_2 = (3x^2-1)/2, P_3 = (5x^3-3x)/2,
    # P_4 = (35x^4-30x^2+3)/8.
    for x64 in [-0.9, -0.3, 0.0, 0.25, 0.75, 1.0]:
        var x = pv(x64)
        assert_almost_equal(legendre_p(0, x).v, SIMD[dtype, width](1.0))
        assert_almost_equal(legendre_p(1, x).v, SIMD[dtype, width](x64))
        assert_almost_equal(
            legendre_p(2, x).v,
            SIMD[dtype, width]((3.0 * x64 * x64 - 1.0) / 2.0),
            atol=1e-14,
        )
        assert_almost_equal(
            legendre_p(3, x).v,
            SIMD[dtype, width]((5.0 * x64**3 - 3.0 * x64) / 2.0),
            atol=1e-14,
        )
        assert_almost_equal(
            legendre_p(4, x).v,
            SIMD[dtype, width](
                (35.0 * x64**4 - 30.0 * x64 * x64 + 3.0) / 8.0
            ),
            atol=1e-14,
        )


def test_endpoints_are_plus_minus_one() raises:
    # P_n(1) = 1 and P_n(-1) = (-1)^n, exactly, for every n.
    for n in [1, 2, 5, 8, 13]:
        assert_almost_equal(
            legendre_p(n, pv(1.0)).v, SIMD[dtype, width](1.0), atol=1e-12
        )
        var expected = 1.0 if n % 2 == 0 else -1.0
        assert_almost_equal(
            legendre_p(n, pv(-1.0)).v,
            SIMD[dtype, width](expected),
            atol=1e-12,
        )


def test_negative_degree_is_p0() raises:
    assert_almost_equal(legendre_p(-3, pv(0.4)).v, SIMD[dtype, width](1.0))


def test_derivative_matches_the_standard_identity() raises:
    # (x^2 - 1) * P_n'(x) = n * (x*P_n(x) - P_{n-1}(x)) -- an identity
    # independent of how the derivative is obtained, which here is `Dual`
    # differentiating the recurrence itself.
    comptime n = 6
    for x64 in [-0.8, -0.2, 0.35, 0.9]:
        var d = legendre_p(n, D(pv(x64), pv(1.0))).deriv.v
        var lhs = (x64 * x64 - 1.0) * Float64(d)
        var rhs = Float64(n) * (
            x64 * Float64(legendre_p(n, pv(x64)).v)
            - Float64(legendre_p(n - 1, pv(x64)).v)
        )
        assert_almost_equal(lhs, rhs, atol=1e-12)


def test_orthogonality_by_quadrature_on_a_fine_grid() raises:
    # P_2 and P_3 are orthogonal on [-1,1]; a midpoint sum over a fine
    # grid is enough to see the integral vanish, and it checks the
    # recurrence produces genuinely different polynomials rather than
    # rescalings of one another.
    comptime num_points = 20000
    var total = 0.0
    var h = 2.0 / Float64(num_points)
    for i in range(num_points):
        var x64 = -1.0 + (Float64(i) + 0.5) * h
        var x = pv(x64)
        total += Float64(legendre_p(2, x).v) * Float64(legendre_p(3, x).v) * h
    assert_almost_equal(total, 0.0, atol=1e-10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
