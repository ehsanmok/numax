"""Tests for `numax.distributions`.

Most checks here are identities rather than table lookups: a CDF against
the sum of its own PMF, a distribution against the special case it reduces
to, a PDF against its CDF differentiated by `Dual`, a quantile against the
CDF it inverts. Those catch a wrong constant the same way a reference value
would, and they also catch a wrong *relationship* between two functions
that a per-function table can't.

The few hard numbers used are ones with exact closed forms or ones verified
against independent numerical integration.
"""

from std.math import atan, pi
from std.math import exp as exp_f64
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Dual, FloatLike, Plain
from numax.distributions import (
    beta_cdf,
    beta_pdf,
    beta_quantile,
    binomial_cdf,
    binomial_pmf,
    chi2_cdf,
    chi2_pdf,
    chi2_quantile,
    exponential_cdf,
    exponential_pdf,
    f_cdf,
    f_pdf,
    gamma_cdf,
    gamma_pdf,
    gamma_quantile,
    normal_cdf,
    normal_pdf,
    normal_quantile,
    poisson_cdf,
    poisson_pmf,
    student_t_cdf,
    student_t_pdf,
    student_t_quantile,
)

comptime dtype = DType.float64
comptime width = 1
comptime P = Plain[dtype, width]
comptime D = Dual[P]

# `betainc`'s fixed 100-iteration continued fraction lands around 5e-10, so
# anything routed through it is checked at 1e-8 rather than machine epsilon.
comptime BETA_ATOL = 1e-8


def pv(x: Float64) -> P:
    return P(SIMD[dtype, width](x))


def s(x: P) -> Float64:
    return Float64(x.v)


def dv(x: Float64) -> D:
    """`x` seeded to be differentiated with respect to."""
    return D(pv(x), pv(1.0))


def dc(x: Float64) -> D:
    return D.constant(x)


# ---------------------------------------------------------------- normal


def test_normal_cdf_at_the_mean_is_one_half() raises:
    assert_almost_equal(
        normal_cdf(pv(3.0), pv(3.0), pv(2.0)).v,
        SIMD[dtype, width](0.5),
        atol=1e-15,
    )


def test_normal_cdf_is_symmetric() raises:
    for x in [0.3, 1.4, 2.9]:
        var lower = s(normal_cdf(pv(-x), pv(0.0), pv(1.0)))
        var upper = s(normal_cdf(pv(x), pv(0.0), pv(1.0)))
        assert_almost_equal(lower + upper, 1.0, atol=1e-14)


def test_normal_cdf_matches_known_values() raises:
    assert_almost_equal(
        s(normal_cdf(pv(1.96), pv(0.0), pv(1.0))),
        0.9750021048517795,
        atol=1e-13,
    )
    assert_almost_equal(
        s(normal_cdf(pv(1.0), pv(0.0), pv(1.0))),
        0.8413447460685429,
        atol=1e-13,
    )


def test_normal_cdf_keeps_precision_in_the_far_tail() raises:
    # The reason this is written with `erfc` rather than `1 - erf`: at
    # z = -6 the answer is around 1e-9, which the subtraction form would
    # have to recover from `0.5*(1 - 0.99999...)`.
    assert_almost_equal(
        s(normal_cdf(pv(-6.0), pv(0.0), pv(1.0))),
        9.865876450376946e-10,
        atol=1e-20,
    )


def test_normal_pdf_is_the_derivative_of_normal_cdf() raises:
    for x in [-1.7, 0.0, 0.6, 2.3]:
        var slope = normal_cdf(dv(x), dc(0.5), dc(1.3)).deriv.copy()
        var density = normal_pdf(pv(x), pv(0.5), pv(1.3))
        assert_almost_equal(s(slope), s(density), atol=1e-14)


def test_normal_quantile_inverts_normal_cdf() raises:
    for p in [1e-8, 0.001, 0.1, 0.5, 0.9, 0.975, 0.99999]:
        var x = normal_quantile(pv(p), pv(1.5), pv(2.0))
        var back = s(normal_cdf(x, pv(1.5), pv(2.0)))
        assert_almost_equal(back, p, atol=1e-12)


def test_normal_quantile_matches_a_known_value() raises:
    assert_almost_equal(
        s(normal_quantile(pv(0.975), pv(0.0), pv(1.0))),
        1.959963984540054,
        atol=1e-12,
    )


# ----------------------------------------------------------- exponential


def test_exponential_is_zero_below_its_support() raises:
    assert_almost_equal(s(exponential_pdf(pv(-1.0), pv(2.0))), 0.0)
    assert_almost_equal(s(exponential_cdf(pv(-1.0), pv(2.0))), 0.0)


def test_exponential_cdf_matches_the_closed_form() raises:
    assert_almost_equal(
        s(exponential_cdf(pv(1.0), pv(2.0))),
        1.0 - exp_f64(-2.0),
        atol=1e-14,
    )


def test_exponential_is_memoryless() raises:
    # P(X > s+t) = P(X > s)*P(X > t) is the defining property.
    var rate = pv(0.7)
    var joint = 1.0 - s(exponential_cdf(pv(3.0), rate))
    var first = 1.0 - s(exponential_cdf(pv(1.2), rate))
    var second = 1.0 - s(exponential_cdf(pv(1.8), rate))
    assert_almost_equal(joint, first * second, atol=1e-14)


# ----------------------------------------------------------------- gamma


def test_gamma_with_unit_shape_is_exponential() raises:
    # Gamma(shape=1, scale=1/rate) is Exponential(rate).
    for x in [0.2, 1.0, 4.5]:
        assert_almost_equal(
            s(gamma_cdf(pv(x), pv(1.0), pv(0.5))),
            s(exponential_cdf(pv(x), pv(2.0))),
            atol=1e-12,
        )


def test_gamma_pdf_is_the_derivative_of_gamma_cdf() raises:
    for x in [0.3, 1.1, 5.0]:
        var slope = gamma_cdf(dv(x), dc(2.5), dc(1.7)).deriv.copy()
        var density = gamma_pdf(pv(x), pv(2.5), pv(1.7))
        assert_almost_equal(s(slope), s(density), atol=1e-11)


def test_gamma_cdf_matches_a_closed_form() raises:
    # For integer shape k, P(k, x) = 1 - exp(-x)*sum_{j<k} x^j/j!.
    # k = 2: 1 - exp(-x)*(1 + x).
    var x = 3.0
    assert_almost_equal(
        s(gamma_cdf(pv(x), pv(2.0), pv(1.0))),
        1.0 - exp_f64(-x) * (1.0 + x),
        atol=1e-13,
    )


def test_gamma_quantile_inverts_gamma_cdf() raises:
    for p in [0.01, 0.25, 0.5, 0.9, 0.999]:
        var x = gamma_quantile(pv(p), pv(2.5), pv(1.7))
        assert_almost_equal(s(gamma_cdf(x, pv(2.5), pv(1.7))), p, atol=1e-10)


def test_gamma_quantile_handles_a_small_shape() raises:
    # The Wilson-Hilferty seed is at its worst below shape 1, which is
    # what the Newton refinement is there for.
    var x = gamma_quantile(pv(0.7), pv(0.4), pv(1.0))
    assert_almost_equal(s(gamma_cdf(x, pv(0.4), pv(1.0))), 0.7, atol=1e-9)


# ------------------------------------------------------------ chi-square


def test_chi_square_with_two_df_is_exponential() raises:
    for x in [0.5, 2.0, 6.0]:
        assert_almost_equal(
            s(chi2_cdf(pv(x), pv(2.0))),
            1.0 - exp_f64(-x / 2.0),
            atol=1e-13,
        )


def test_chi_square_critical_value() raises:
    # The 95th percentile at 1 df, the value behind a two-sided z of 1.96.
    assert_almost_equal(
        s(chi2_cdf(pv(3.841458820694124), pv(1.0))), 0.95, atol=1e-9
    )
    assert_almost_equal(
        s(chi2_quantile(pv(0.95), pv(1.0))),
        3.841458820694124,
        atol=1e-7,
    )


def test_chi_square_pdf_is_the_derivative_of_its_cdf() raises:
    var slope = chi2_cdf(dv(2.2), dc(3.0)).deriv.copy()
    assert_almost_equal(s(slope), s(chi2_pdf(pv(2.2), pv(3.0))), atol=1e-12)


# ------------------------------------------------------------------ beta


def test_beta_with_unit_parameters_is_uniform() raises:
    for x in [0.0, 0.25, 0.5, 1.0]:
        assert_almost_equal(s(beta_cdf(pv(x), pv(1.0), pv(1.0))), x, atol=1e-12)
    assert_almost_equal(s(beta_pdf(pv(0.4), pv(1.0), pv(1.0))), 1.0, atol=1e-12)


def test_beta_cdf_matches_an_exact_polynomial() raises:
    # I_x(2,3) = 6x^2/2 - 8x^3 + 3x^4 ... use the known value at x = 1/2.
    assert_almost_equal(
        s(beta_cdf(pv(0.5), pv(2.0), pv(3.0))), 0.6875, atol=BETA_ATOL
    )


def test_beta_cdf_reflection() raises:
    # I_x(a,b) + I_{1-x}(b,a) = 1.
    for x in [0.15, 0.5, 0.83]:
        var forward = s(beta_cdf(pv(x), pv(2.3), pv(4.1)))
        var mirrored = s(beta_cdf(pv(1.0 - x), pv(4.1), pv(2.3)))
        assert_almost_equal(forward + mirrored, 1.0, atol=BETA_ATOL)


def test_beta_pdf_is_the_derivative_of_beta_cdf() raises:
    for x in [0.2, 0.55, 0.9]:
        var slope = beta_cdf(dv(x), dc(2.3), dc(4.1)).deriv.copy()
        assert_almost_equal(
            s(slope), s(beta_pdf(pv(x), pv(2.3), pv(4.1))), atol=1e-7
        )


def test_beta_quantile_inverts_beta_cdf() raises:
    for p in [0.05, 0.3, 0.5, 0.77, 0.99]:
        var x = beta_quantile(pv(p), pv(2.3), pv(4.1))
        assert_almost_equal(s(beta_cdf(x, pv(2.3), pv(4.1))), p, atol=BETA_ATOL)


def test_beta_quantile_is_symmetric_at_the_median() raises:
    # I_{0.5}(a,a) = 0.5 for any a, so the median of a symmetric beta is
    # exactly 1/2.
    assert_almost_equal(
        s(beta_quantile(pv(0.5), pv(3.0), pv(3.0))), 0.5, atol=1e-8
    )


# ------------------------------------------------------------- Student-t


def test_student_t_with_one_df_is_cauchy() raises:
    for x in [-2.0, -0.4, 0.0, 1.3]:
        assert_almost_equal(
            s(student_t_cdf(pv(x), pv(1.0))),
            0.5 + atan(x) / pi,
            atol=BETA_ATOL,
        )


def test_student_t_cdf_is_symmetric() raises:
    for x in [0.5, 1.8, 4.0]:
        var lower = s(student_t_cdf(pv(-x), pv(7.0)))
        var upper = s(student_t_cdf(pv(x), pv(7.0)))
        assert_almost_equal(lower + upper, 1.0, atol=BETA_ATOL)


def test_student_t_cdf_at_zero_is_one_half() raises:
    assert_almost_equal(s(student_t_cdf(pv(0.0), pv(5.0))), 0.5, atol=1e-12)


def test_student_t_pdf_is_the_derivative_of_its_cdf() raises:
    for x in [-1.1, 0.4, 2.6]:
        var slope = student_t_cdf(dv(x), dc(6.0)).deriv.copy()
        assert_almost_equal(
            s(slope), s(student_t_pdf(pv(x), pv(6.0))), atol=1e-7
        )


def test_student_t_quantile_matches_a_table_value() raises:
    # The 97.5th percentile at 10 df -- the multiplier behind a 95%
    # confidence interval from 11 observations.
    assert_almost_equal(
        s(student_t_quantile(pv(0.975), pv(10.0))),
        2.2281388519649385,
        atol=1e-7,
    )


def test_student_t_approaches_the_normal_for_large_df() raises:
    var t = s(student_t_cdf(pv(1.5), pv(2000.0)))
    var z = s(normal_cdf(pv(1.5), pv(0.0), pv(1.0)))
    assert_true(abs(t - z) < 1e-4)


# --------------------------------------------------------------------- F


def test_f_cdf_matches_numerical_integration() raises:
    # Cross-checked against a direct quadrature of the incomplete beta.
    assert_almost_equal(
        s(f_cdf(pv(3.0), pv(4.0), pv(5.0))),
        0.870296515399361,
        atol=BETA_ATOL,
    )


def test_f_relates_to_student_t() raises:
    # A t with nu df, squared, is F(1, nu).
    for x in [0.6, 1.5, 3.2]:
        var from_f = s(f_cdf(pv(x * x), pv(1.0), pv(8.0)))
        var from_t = 2.0 * s(student_t_cdf(pv(x), pv(8.0))) - 1.0
        assert_almost_equal(from_f, from_t, atol=BETA_ATOL)


def test_f_pdf_is_the_derivative_of_f_cdf() raises:
    var slope = f_cdf(dv(2.0), dc(4.0), dc(5.0)).deriv.copy()
    assert_almost_equal(
        s(slope), s(f_pdf(pv(2.0), pv(4.0), pv(5.0))), atol=1e-7
    )


def test_f_is_zero_below_its_support() raises:
    assert_almost_equal(s(f_cdf(pv(-1.0), pv(4.0), pv(5.0))), 0.0)
    assert_almost_equal(s(f_pdf(pv(-1.0), pv(4.0), pv(5.0))), 0.0)


# -------------------------------------------------------------- discrete


def test_poisson_cdf_equals_the_sum_of_its_pmf() raises:
    # The strongest check available for `gammaincc`-as-a-discrete-CDF: the
    # continuous special function must reproduce the finite sum exactly.
    var rate = pv(2.5)
    var running = 0.0
    for k in range(6):
        running += s(poisson_pmf(pv(Float64(k)), rate))
        assert_almost_equal(
            s(poisson_cdf(pv(Float64(k)), rate)), running, atol=1e-11
        )


def test_poisson_pmf_matches_the_closed_form() raises:
    # exp(-2.5) * 2.5^3 / 3!
    assert_almost_equal(
        s(poisson_pmf(pv(3.0), pv(2.5))),
        exp_f64(-2.5) * 15.625 / 6.0,
        atol=1e-13,
    )


def test_binomial_cdf_equals_the_sum_of_its_pmf() raises:
    var n = pv(10.0)
    var p = pv(0.3)
    var running = 0.0
    for k in range(11):
        running += s(binomial_pmf(pv(Float64(k)), n, p))
        assert_almost_equal(
            s(binomial_cdf(pv(Float64(k)), n, p)), running, atol=1e-8
        )


def test_binomial_pmf_sums_to_one() raises:
    var total = 0.0
    for k in range(11):
        total += s(binomial_pmf(pv(Float64(k)), pv(10.0), pv(0.3)))
    assert_almost_equal(total, 1.0, atol=1e-12)


def test_binomial_with_one_trial_is_bernoulli() raises:
    assert_almost_equal(
        s(binomial_pmf(pv(1.0), pv(1.0), pv(0.42))), 0.42, atol=1e-13
    )
    assert_almost_equal(
        s(binomial_pmf(pv(0.0), pv(1.0), pv(0.42))), 0.58, atol=1e-13
    )


# ---------------------------------------------------------------- shared


def test_simd_lanes_are_independent() raises:
    comptime w = 4
    comptime PW = Plain[dtype, w]
    var x = PW(SIMD[dtype, w](-1.0, 0.0, 1.0, 2.0))
    var result = normal_cdf(x, PW.constant(0.0), PW.constant(1.0))
    var expected = SIMD[dtype, w](
        0.15865525393145707,
        0.5,
        0.8413447460685429,
        0.9772498680518208,
    )
    for lane in range(w):
        assert_almost_equal(
            Float64(result.v[lane]), Float64(expected[lane]), atol=1e-13
        )


def test_a_distribution_parameter_can_be_differentiated() raises:
    # Nothing special-cases which argument carries the derivative, so the
    # sensitivity of a tail probability to the scale parameter comes out of
    # the same call.
    var by_scale = gamma_cdf(dc(3.0), dc(2.5), dv(1.7)).deriv.copy()
    # Finite-difference cross-check, since there's no tidy closed form.
    var h = 1e-6
    var up = s(gamma_cdf(pv(3.0), pv(2.5), pv(1.7 + h)))
    var down = s(gamma_cdf(pv(3.0), pv(2.5), pv(1.7 - h)))
    assert_almost_equal(s(by_scale), (up - down) / (2.0 * h), atol=1e-7)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
