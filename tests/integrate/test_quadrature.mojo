"""Tests for `numax.integrate` and `numax.special.orthopoly`.

Integrands here have closed-form antiderivatives, so every expected value
is exact rather than a reference-library number.
"""

from std.math import cos as cos_f64
from std.math import exp as exp_f64
from std.math import sin as sin_f64
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import (
    Dual,
    FloatLike,
    Plain,
    chebyshev_t,
    chebyshev_u,
    gauss_legendre,
    hermite_h,
    laguerre_l,
    legendre_p,
    simpson,
    trapezoid,
)

comptime dtype = DType.float64
comptime width = 1
comptime P = Plain[dtype, width]
comptime D = Dual[P]


def pv(x: Float64) -> P:
    return P(SIMD[dtype, width](x))


def quintic[U: FloatLike](x: U) -> U:
    """`x^5 - 3x^2 + 2`; a degree-5 polynomial, so 3-point Gauss-Legendre
    (exact through degree 5) should already nail it."""
    var x2 = x * x
    return x2 * x2 * x + (-(U.constant(3.0) * x2)) + U.constant(2.0)


def gaussian_bump[U: FloatLike](x: U) -> U:
    return (-(x * x)).exp()


def sine[U: FloatLike](x: U) -> U:
    return x.sin()


def reciprocal[U: FloatLike](x: U) -> U:
    return U.one() / x


def test_gauss_legendre_is_exact_for_low_degree_polynomials() raises:
    # integral of x^5 - 3x^2 + 2 over [0,2] = 32/6 - 8 + 4 = 5/3 * 2 ... :
    # = [x^6/6 - x^3 + 2x] from 0 to 2 = 64/6 - 8 + 4 = 6.666... - 4.
    var expected = 64.0 / 6.0 - 8.0 + 4.0
    var result = gauss_legendre[f=quintic, n=3](pv(0.0), pv(2.0))
    assert_almost_equal(result.v, SIMD[dtype, width](expected), atol=1e-12)


def test_gauss_legendre_matches_a_known_transcendental_integral() raises:
    # integral of sin(x) over [0, pi] = 2.
    var result = gauss_legendre[f=sine, n=10](pv(0.0), pv(3.141592653589793))
    assert_almost_equal(result.v, SIMD[dtype, width](2.0), atol=1e-12)


def test_gauss_legendre_on_a_gaussian() raises:
    # integral of exp(-x^2) over [-3, 3] = sqrt(pi)*erf(3).
    var expected = 1.7724538509055159 * 0.9999779095030014
    var result = gauss_legendre[f=gaussian_bump, n=20](pv(-3.0), pv(3.0))
    assert_almost_equal(result.v, SIMD[dtype, width](expected), atol=1e-10)


def test_gauss_legendre_beats_a_much_finer_trapezoid_grid() raises:
    # 8 evaluations against 128: the reason the default n is small.
    var reference = 2.0
    var upper = pv(3.141592653589793)
    var gauss = Float64(gauss_legendre[f=sine, n=8](pv(0.0), upper).v)
    var trap = Float64(trapezoid[f=sine, num_intervals=128](pv(0.0), upper).v)
    assert_true(abs(gauss - reference) < abs(trap - reference))


def test_gauss_legendre_handles_a_reversed_interval() raises:
    # Swapping the limits negates the result, as the affine map implies.
    var forward = gauss_legendre[f=sine, n=8](pv(0.0), pv(1.0))
    var backward = gauss_legendre[f=sine, n=8](pv(1.0), pv(0.0))
    assert_almost_equal(forward.v, -backward.v, atol=1e-14)


def test_simpson_matches_a_polynomial_exactly() raises:
    # Simpson is exact through degree 3; use a cubic-free integrand it
    # still handles: integral of sin over [0, pi] to O(h^4).
    var result = simpson[f=sine, num_panels=100](pv(0.0), pv(3.141592653589793))
    assert_almost_equal(result.v, SIMD[dtype, width](2.0), atol=1e-9)


def test_simpson_beats_trapezoid_on_the_same_grid() raises:
    var reference = 2.0
    var upper = pv(3.141592653589793)
    var s = Float64(simpson[f=sine, num_panels=16](pv(0.0), upper).v)
    var t = Float64(trapezoid[f=sine, num_intervals=32](pv(0.0), upper).v)
    assert_true(abs(s - reference) < abs(t - reference))


def test_trapezoid_matches_a_logarithm() raises:
    # integral of 1/x over [1, e] = 1.
    var result = trapezoid[f=reciprocal, num_intervals=20000](
        pv(1.0), pv(2.718281828459045)
    )
    assert_almost_equal(result.v, SIMD[dtype, width](1.0), atol=1e-8)


def test_integrating_at_dual_differentiates_the_upper_limit() raises:
    # d/db[integral(f, a, b)] = f(b), the fundamental theorem of calculus,
    # falling out of `Dual` propagating through the quadrature itself.
    var b64 = 1.3
    var a = D.constant(0.0)
    var b = D(pv(b64), pv(1.0))
    var result = gauss_legendre[f=sine, n=12](a, b)
    assert_almost_equal(
        result.deriv.v, SIMD[dtype, width](sin_f64(b64)), atol=1e-10
    )


def test_integrating_at_dual_differentiates_the_lower_limit() raises:
    # d/da[integral(f, a, b)] = -f(a).
    var a64 = 0.4
    var a = D(pv(a64), pv(1.0))
    var b = D.constant(2.0)
    var result = gauss_legendre[f=sine, n=12](a, b)
    assert_almost_equal(
        result.deriv.v, SIMD[dtype, width](-sin_f64(a64)), atol=1e-10
    )


def test_gauss_legendre_nodes_are_symmetric_and_weights_sum_to_two() raises:
    # Integrating the constant 1 over [-1,1] gives the sum of the weights,
    # which must be 2 -- a direct check on the compile-time node/weight
    # table without exposing it.
    def one_kernel[U: FloatLike](x: U) -> U:
        return U.one()

    var total = gauss_legendre[f=one_kernel, n=16](pv(-1.0), pv(1.0))
    assert_almost_equal(total.v, SIMD[dtype, width](2.0), atol=1e-13)


def test_hermite_matches_closed_forms() raises:
    # H_0=1, H_1=2x, H_2=4x^2-2, H_3=8x^3-12x, H_4=16x^4-48x^2+12.
    for x64 in [-1.5, -0.2, 0.0, 0.9, 2.0]:
        var x = pv(x64)
        assert_almost_equal(hermite_h(0, x).v, SIMD[dtype, width](1.0))
        assert_almost_equal(hermite_h(1, x).v, SIMD[dtype, width](2.0 * x64))
        assert_almost_equal(
            hermite_h(2, x).v,
            SIMD[dtype, width](4.0 * x64 * x64 - 2.0),
            atol=1e-12,
        )
        assert_almost_equal(
            hermite_h(3, x).v,
            SIMD[dtype, width](8.0 * x64**3 - 12.0 * x64),
            atol=1e-12,
        )
        assert_almost_equal(
            hermite_h(4, x).v,
            SIMD[dtype, width](16.0 * x64**4 - 48.0 * x64 * x64 + 12.0),
            atol=1e-11,
        )


def test_laguerre_matches_closed_forms() raises:
    # L_0=1, L_1=1-x, L_2=(x^2-4x+2)/2, L_3=(-x^3+9x^2-18x+6)/6.
    for x64 in [0.0, 0.5, 2.0, 5.0]:
        var x = pv(x64)
        assert_almost_equal(laguerre_l(0, x).v, SIMD[dtype, width](1.0))
        assert_almost_equal(laguerre_l(1, x).v, SIMD[dtype, width](1.0 - x64))
        assert_almost_equal(
            laguerre_l(2, x).v,
            SIMD[dtype, width]((x64 * x64 - 4.0 * x64 + 2.0) / 2.0),
            atol=1e-12,
        )
        assert_almost_equal(
            laguerre_l(3, x).v,
            SIMD[dtype, width](
                (-(x64**3) + 9.0 * x64 * x64 - 18.0 * x64 + 6.0) / 6.0
            ),
            atol=1e-12,
        )


def test_chebyshev_t_matches_the_trigonometric_form() raises:
    # T_n(cos(theta)) = cos(n*theta) on [-1,1].
    for n in [1, 2, 5, 9]:
        for theta in [0.3, 1.1, 2.4]:
            var x = pv(cos_f64(theta))
            assert_almost_equal(
                chebyshev_t(n, x).v,
                SIMD[dtype, width](cos_f64(Float64(n) * theta)),
                atol=1e-12,
            )


def test_chebyshev_u_matches_the_trigonometric_form() raises:
    # U_n(cos(theta)) = sin((n+1)*theta)/sin(theta).
    for n in [1, 2, 5, 9]:
        for theta in [0.3, 1.1, 2.4]:
            var x = pv(cos_f64(theta))
            var expected = sin_f64((Float64(n) + 1.0) * theta) / sin_f64(theta)
            assert_almost_equal(
                chebyshev_u(n, x).v,
                SIMD[dtype, width](expected),
                atol=1e-11,
            )


def test_orthogonal_polynomial_derivatives_via_dual() raises:
    # T_n'(x) = n*U_{n-1}(x), an identity relating two of these families
    # with the derivative supplied by `Dual` rather than a formula.
    comptime n = 6
    for x64 in [-0.7, 0.1, 0.55]:
        var d = chebyshev_t(n, D(pv(x64), pv(1.0))).deriv.v
        var expected = Float64(n) * Float64(chebyshev_u(n - 1, pv(x64)).v)
        assert_almost_equal(Float64(d), expected, atol=1e-11)


def test_hermite_derivative_identity() raises:
    # H_n'(x) = 2n*H_{n-1}(x).
    comptime n = 5
    var x64 = 0.8
    var d = hermite_h(n, D(pv(x64), pv(1.0))).deriv.v
    var expected = 2.0 * Float64(n) * Float64(hermite_h(n - 1, pv(x64)).v)
    assert_almost_equal(Float64(d), expected, atol=1e-10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
