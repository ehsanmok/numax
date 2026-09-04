"""Tests for `numax.core.complex` -- arithmetic, the complex-analytic extensions
of `exp`/`ln`/`sin`/`cos`/`erf`, and the `Dual`-nesting property.
"""

from std.math import cos as std_cos
from std.math import cosh as std_cosh
from std.math import erf as std_erf
from std.math import exp as std_exp
from std.math import log as std_log
from std.math import pi
from std.math import sin as std_sin
from std.math import sinh as std_sinh
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Complex, Dual, Plain

comptime dtype = DType.float64
comptime width = 1
comptime CT = Complex[Plain[dtype, width]]


def c(re: Float64, im: Float64) -> CT:
    return CT(
        Plain[dtype, width].constant(re),
        Plain[dtype, width].constant(im),
    )


def assert_complex_almost_equal(
    got: CT, expected_re: Float64, expected_im: Float64, atol: Float64 = 1e-8
) raises:
    assert_almost_equal(got.re.v, SIMD[dtype, width](expected_re), atol=atol)
    assert_almost_equal(got.im.v, SIMD[dtype, width](expected_im), atol=atol)


def test_add() raises:
    var result = c(3.0, 4.0) + c(1.0, -2.0)
    assert_complex_almost_equal(result, 4.0, 2.0)


def test_mul() raises:
    # (3+4i)(1-2i) = 3-6i+4i-8i^2 = 3-2i+8 = 11-2i.
    var result = c(3.0, 4.0) * c(1.0, -2.0)
    assert_complex_almost_equal(result, 11.0, -2.0)


def test_div() raises:
    # (3+4i)/(1-2i) = (3+4i)(1+2i)/5 = (3+10i-8)/5 = -1+2i.
    var result = c(3.0, 4.0) / c(1.0, -2.0)
    assert_complex_almost_equal(result, -1.0, 2.0)


def test_abs_is_modulus_embedded_as_real() raises:
    var result = c(3.0, 4.0).abs()
    assert_complex_almost_equal(result, 5.0, 0.0, atol=1e-6)


def test_copysign_convention() raises:
    var result = c(3.0, 4.0).copysign(c(-1.0, 0.0))
    assert_complex_almost_equal(result, -5.0, 0.0, atol=1e-6)


def test_exp_matches_std_math() raises:
    var result = c(3.0, 4.0).exp()
    var mag = std_exp(Float64(3.0))
    assert_complex_almost_equal(
        result, mag * std_cos(Float64(4.0)), mag * std_sin(Float64(4.0))
    )


def test_ln_matches_std_math() raises:
    var result = c(3.0, 4.0).ln()
    assert_complex_almost_equal(
        result, std_log(Float64(5.0)), 0.9272952180016122
    )


def test_ln_exp_round_trip() raises:
    var z = c(1.5, -2.5)
    var round_trip = z.ln().exp()
    assert_complex_almost_equal(round_trip, 1.5, -2.5, atol=1e-6)


def test_sin_matches_closed_form() raises:
    var result = c(3.0, 4.0).sin()
    assert_complex_almost_equal(
        result,
        std_sin(Float64(3.0)) * std_cosh(Float64(4.0)),
        std_cos(Float64(3.0)) * std_sinh(Float64(4.0)),
    )


def test_cos_matches_closed_form() raises:
    var result = c(3.0, 4.0).cos()
    assert_complex_almost_equal(
        result,
        std_cos(Float64(3.0)) * std_cosh(Float64(4.0)),
        -(std_sin(Float64(3.0)) * std_sinh(Float64(4.0))),
    )


def test_erf_on_real_axis_matches_plain_erf() raises:
    for x in [0.1, 0.5, 1.0, 2.0, 3.0]:
        var result = c(x, 0.0).erf()
        assert_complex_almost_equal(result, std_erf(x), 0.0, atol=1e-8)


def test_erf_at_origin_is_zero() raises:
    assert_complex_almost_equal(c(0.0, 0.0).erf(), 0.0, 0.0)


def test_erfc_plus_erf_is_one() raises:
    var z = c(0.7, 0.3)
    var total = z.erf() + z.erfc()
    assert_complex_almost_equal(total, 1.0, 0.0, atol=1e-7)


def test_atan2_quadrants_via_ln() raises:
    # Each quadrant's `atan2(im, re)`, recovered from `ln`'s imaginary part.
    assert_almost_equal(
        c(1.0, 1.0).ln().im.v, SIMD[dtype, width](Float64(pi) / 4.0), atol=1e-6
    )
    assert_almost_equal(
        c(-1.0, 1.0).ln().im.v,
        SIMD[dtype, width](3.0 * Float64(pi) / 4.0),
        atol=1e-6,
    )
    assert_almost_equal(
        c(-1.0, -1.0).ln().im.v,
        SIMD[dtype, width](-3.0 * Float64(pi) / 4.0),
        atol=1e-6,
    )
    assert_almost_equal(
        c(1.0, -1.0).ln().im.v,
        SIMD[dtype, width](-Float64(pi) / 4.0),
        atol=1e-6,
    )


def test_nests_inside_dual_for_holomorphic_derivatives() raises:
    # Complex[Dual[Plain]] differentiates a complex-valued kernel the same
    # way Dual[Plain] differentiates a real one -- d/dz[z^2] = 2z at
    # z = 3+4i, i.e. 6+8i, with no changes to the kernel (just `z*z`).
    comptime DCT = Complex[Dual[Plain[dtype, width]]]
    var re = Dual[Plain[dtype, width]](
        Plain[dtype, width].constant(3.0),
        Plain[dtype, width].constant(1.0),
    )
    var im = Dual[Plain[dtype, width]](
        Plain[dtype, width].constant(4.0),
        Plain[dtype, width].constant(0.0),
    )
    var z = DCT(re^, im^)
    var w = z * z
    assert_almost_equal(w.re.value.v, SIMD[dtype, width](-7.0), atol=1e-8)
    assert_almost_equal(w.im.value.v, SIMD[dtype, width](24.0), atol=1e-8)
    assert_almost_equal(w.re.deriv.v, SIMD[dtype, width](6.0), atol=1e-8)
    assert_almost_equal(w.im.deriv.v, SIMD[dtype, width](8.0), atol=1e-8)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
