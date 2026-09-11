"""Tests for `numax.stats`.

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
from numax.stats import (
    beta,
    binom,
    chi2,
    expon,
    f,
    gamma,
    norm,
    poisson,
    t,
)

comptime dtype = DType.float64
comptime width = 1
comptime P = Plain[dtype, width]
comptime D = Dual[P]

# `betainc`'s fixed 100-iteration continued fraction lands around 5e-10, so
# anything routed through it is checked at 1e-8 rather than machine epsilon.
comptime BETA_ATOL = 1e-8


def pv(x: Float64) -> P:
    return P.constant(x)


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
        norm.cdf(pv(3.0), pv(3.0), pv(2.0)).v,
        SIMD[dtype, width](0.5),
        atol=1e-15,
    )


def test_normal_cdf_is_symmetric() raises:
    for x in [0.3, 1.4, 2.9]:
        var lower = s(norm.cdf(pv(-x), pv(0.0), pv(1.0)))
        var upper = s(norm.cdf(pv(x), pv(0.0), pv(1.0)))
        assert_almost_equal(lower + upper, 1.0, atol=1e-14)


def test_normal_cdf_matches_known_values() raises:
    assert_almost_equal(
        s(norm.cdf(pv(1.96), pv(0.0), pv(1.0))),
        0.9750021048517795,
        atol=1e-13,
    )
    assert_almost_equal(
        s(norm.cdf(pv(1.0), pv(0.0), pv(1.0))),
        0.8413447460685429,
        atol=1e-13,
    )


def test_normal_cdf_keeps_precision_in_the_far_tail() raises:
    # The reason this is written with `erfc` rather than `1 - erf`: at
    # z = -6 the answer is around 1e-9, which the subtraction form would
    # have to recover from `0.5*(1 - 0.99999...)`.
    assert_almost_equal(
        s(norm.cdf(pv(-6.0), pv(0.0), pv(1.0))),
        9.865876450376946e-10,
        atol=1e-20,
    )


def test_normal_pdf_is_the_derivative_of_normal_cdf() raises:
    for x in [-1.7, 0.0, 0.6, 2.3]:
        var slope = norm.cdf(dv(x), dc(0.5), dc(1.3)).deriv.copy()
        var density = norm.pdf(pv(x), pv(0.5), pv(1.3))
        assert_almost_equal(s(slope), s(density), atol=1e-14)


def test_normal_quantile_inverts_normal_cdf() raises:
    for p in [1e-8, 0.001, 0.1, 0.5, 0.9, 0.975, 0.99999]:
        var x = norm.ppf(pv(p), pv(1.5), pv(2.0))
        var back = s(norm.cdf(x, pv(1.5), pv(2.0)))
        assert_almost_equal(back, p, atol=1e-12)


def test_normal_quantile_matches_a_known_value() raises:
    assert_almost_equal(
        s(norm.ppf(pv(0.975), pv(0.0), pv(1.0))),
        1.959963984540054,
        atol=1e-12,
    )


# ----------------------------------------------------------- exponential


def test_exponential_is_zero_below_its_support() raises:
    assert_almost_equal(s(expon.pdf(pv(-1.0), pv(2.0))), 0.0)
    assert_almost_equal(s(expon.cdf(pv(-1.0), pv(2.0))), 0.0)


def test_exponential_cdf_matches_the_closed_form() raises:
    assert_almost_equal(
        s(expon.cdf(pv(1.0), pv(2.0))),
        1.0 - exp_f64(-2.0),
        atol=1e-14,
    )


def test_exponential_is_memoryless() raises:
    # P(X > s+t) = P(X > s)*P(X > t) is the defining property.
    var rate = pv(0.7)
    var joint = 1.0 - s(expon.cdf(pv(3.0), rate))
    var first = 1.0 - s(expon.cdf(pv(1.2), rate))
    var second = 1.0 - s(expon.cdf(pv(1.8), rate))
    assert_almost_equal(joint, first * second, atol=1e-14)


# ----------------------------------------------------------------- gamma


def test_gamma_with_unit_shape_is_exponential() raises:
    # Gamma(shape=1, scale=1/rate) is Exponential(rate).
    for x in [0.2, 1.0, 4.5]:
        assert_almost_equal(
            s(gamma.cdf(pv(x), pv(1.0), pv(0.5))),
            s(expon.cdf(pv(x), pv(2.0))),
            atol=1e-12,
        )


def test_gamma_pdf_is_the_derivative_of_gamma_cdf() raises:
    for x in [0.3, 1.1, 5.0]:
        var slope = gamma.cdf(dv(x), dc(2.5), dc(1.7)).deriv.copy()
        var density = gamma.pdf(pv(x), pv(2.5), pv(1.7))
        assert_almost_equal(s(slope), s(density), atol=1e-11)


def test_gamma_cdf_matches_a_closed_form() raises:
    # For integer shape k, P(k, x) = 1 - exp(-x)*sum_{j<k} x^j/j!.
    # k = 2: 1 - exp(-x)*(1 + x).
    var x = 3.0
    assert_almost_equal(
        s(gamma.cdf(pv(x), pv(2.0), pv(1.0))),
        1.0 - exp_f64(-x) * (1.0 + x),
        atol=1e-13,
    )


def test_gamma_quantile_inverts_gamma_cdf() raises:
    for p in [0.01, 0.25, 0.5, 0.9, 0.999]:
        var x = gamma.ppf(pv(p), pv(2.5), pv(1.7))
        assert_almost_equal(s(gamma.cdf(x, pv(2.5), pv(1.7))), p, atol=1e-10)


def test_gamma_quantile_handles_a_small_shape() raises:
    # The Wilson-Hilferty seed is at its worst below shape 1, which is
    # what the Newton refinement is there for.
    var x = gamma.ppf(pv(0.7), pv(0.4), pv(1.0))
    assert_almost_equal(s(gamma.cdf(x, pv(0.4), pv(1.0))), 0.7, atol=1e-9)


# ------------------------------------------------------------ chi-square


def test_chi_square_with_two_df_is_exponential() raises:
    for x in [0.5, 2.0, 6.0]:
        assert_almost_equal(
            s(chi2.cdf(pv(x), pv(2.0))),
            1.0 - exp_f64(-x / 2.0),
            atol=1e-13,
        )


def test_chi_square_critical_value() raises:
    # The 95th percentile at 1 df, the value behind a two-sided z of 1.96.
    assert_almost_equal(
        s(chi2.cdf(pv(3.841458820694124), pv(1.0))), 0.95, atol=1e-9
    )
    assert_almost_equal(
        s(chi2.ppf(pv(0.95), pv(1.0))),
        3.841458820694124,
        atol=1e-7,
    )


def test_chi_square_pdf_is_the_derivative_of_its_cdf() raises:
    var slope = chi2.cdf(dv(2.2), dc(3.0)).deriv.copy()
    assert_almost_equal(s(slope), s(chi2.pdf(pv(2.2), pv(3.0))), atol=1e-12)


# ------------------------------------------------------------------ beta


def test_beta_with_unit_parameters_is_uniform() raises:
    for x in [0.0, 0.25, 0.5, 1.0]:
        assert_almost_equal(s(beta.cdf(pv(x), pv(1.0), pv(1.0))), x, atol=1e-12)
    assert_almost_equal(s(beta.pdf(pv(0.4), pv(1.0), pv(1.0))), 1.0, atol=1e-12)


def test_beta_cdf_matches_an_exact_polynomial() raises:
    # I_x(2,3) = 6x^2/2 - 8x^3 + 3x^4 ... use the known value at x = 1/2.
    assert_almost_equal(
        s(beta.cdf(pv(0.5), pv(2.0), pv(3.0))), 0.6875, atol=BETA_ATOL
    )


def test_beta_cdf_reflection() raises:
    # I_x(a,b) + I_{1-x}(b,a) = 1.
    for x in [0.15, 0.5, 0.83]:
        var forward = s(beta.cdf(pv(x), pv(2.3), pv(4.1)))
        var mirrored = s(beta.cdf(pv(1.0 - x), pv(4.1), pv(2.3)))
        assert_almost_equal(forward + mirrored, 1.0, atol=BETA_ATOL)


def test_beta_pdf_is_the_derivative_of_beta_cdf() raises:
    for x in [0.2, 0.55, 0.9]:
        var slope = beta.cdf(dv(x), dc(2.3), dc(4.1)).deriv.copy()
        assert_almost_equal(
            s(slope), s(beta.pdf(pv(x), pv(2.3), pv(4.1))), atol=1e-7
        )


def test_beta_quantile_inverts_beta_cdf() raises:
    for p in [0.05, 0.3, 0.5, 0.77, 0.99]:
        var x = beta.ppf(pv(p), pv(2.3), pv(4.1))
        assert_almost_equal(s(beta.cdf(x, pv(2.3), pv(4.1))), p, atol=BETA_ATOL)


def test_beta_quantile_is_symmetric_at_the_median() raises:
    # I_{0.5}(a,a) = 0.5 for any a, so the median of a symmetric beta is
    # exactly 1/2.
    assert_almost_equal(s(beta.ppf(pv(0.5), pv(3.0), pv(3.0))), 0.5, atol=1e-8)


# ------------------------------------------------------------- Student-t


def test_student_t_with_one_df_is_cauchy() raises:
    for x in [-2.0, -0.4, 0.0, 1.3]:
        assert_almost_equal(
            s(t.cdf(pv(x), pv(1.0))),
            0.5 + atan(x) / pi,
            atol=BETA_ATOL,
        )


def test_student_t_cdf_is_symmetric() raises:
    for x in [0.5, 1.8, 4.0]:
        var lower = s(t.cdf(pv(-x), pv(7.0)))
        var upper = s(t.cdf(pv(x), pv(7.0)))
        assert_almost_equal(lower + upper, 1.0, atol=BETA_ATOL)


def test_student_t_cdf_at_zero_is_one_half() raises:
    assert_almost_equal(s(t.cdf(pv(0.0), pv(5.0))), 0.5, atol=1e-12)


def test_student_t_pdf_is_the_derivative_of_its_cdf() raises:
    for x in [-1.1, 0.4, 2.6]:
        var slope = t.cdf(dv(x), dc(6.0)).deriv.copy()
        assert_almost_equal(s(slope), s(t.pdf(pv(x), pv(6.0))), atol=1e-7)


def test_student_t_quantile_matches_a_table_value() raises:
    # The 97.5th percentile at 10 df -- the multiplier behind a 95%
    # confidence interval from 11 observations.
    assert_almost_equal(
        s(t.ppf(pv(0.975), pv(10.0))),
        2.2281388519649385,
        atol=1e-7,
    )


def test_student_t_approaches_the_normal_for_large_df() raises:
    var t = s(t.cdf(pv(1.5), pv(2000.0)))
    var z = s(norm.cdf(pv(1.5), pv(0.0), pv(1.0)))
    assert_true(abs(t - z) < 1e-4)


# --------------------------------------------------------------------- F


def test_f_cdf_matches_numerical_integration() raises:
    # Cross-checked against a direct quadrature of the incomplete beta.
    assert_almost_equal(
        s(f.cdf(pv(3.0), pv(4.0), pv(5.0))),
        0.870296515399361,
        atol=BETA_ATOL,
    )


def test_f_relates_to_student_t() raises:
    # A t with nu df, squared, is F(1, nu).
    for x in [0.6, 1.5, 3.2]:
        var from_f = s(f.cdf(pv(x * x), pv(1.0), pv(8.0)))
        var from_t = 2.0 * s(t.cdf(pv(x), pv(8.0))) - 1.0
        assert_almost_equal(from_f, from_t, atol=BETA_ATOL)


def test_f_pdf_is_the_derivative_of_f_cdf() raises:
    var slope = f.cdf(dv(2.0), dc(4.0), dc(5.0)).deriv.copy()
    assert_almost_equal(
        s(slope), s(f.pdf(pv(2.0), pv(4.0), pv(5.0))), atol=1e-7
    )


def test_f_is_zero_below_its_support() raises:
    assert_almost_equal(s(f.cdf(pv(-1.0), pv(4.0), pv(5.0))), 0.0)
    assert_almost_equal(s(f.pdf(pv(-1.0), pv(4.0), pv(5.0))), 0.0)


# -------------------------------------------------------------- discrete


def test_poisson_cdf_equals_the_sum_of_its_pmf() raises:
    # The strongest check available for `gammaincc`-as-a-discrete-CDF: the
    # continuous special function must reproduce the finite sum exactly.
    var rate = pv(2.5)
    var running = 0.0
    for k in range(6):
        running += s(poisson.pmf(pv(Float64(k)), rate))
        assert_almost_equal(
            s(poisson.cdf(pv(Float64(k)), rate)), running, atol=1e-11
        )


def test_poisson_pmf_matches_the_closed_form() raises:
    # exp(-2.5) * 2.5^3 / 3!
    assert_almost_equal(
        s(poisson.pmf(pv(3.0), pv(2.5))),
        exp_f64(-2.5) * 15.625 / 6.0,
        atol=1e-13,
    )


def test_binomial_cdf_equals_the_sum_of_its_pmf() raises:
    var n = pv(10.0)
    var p = pv(0.3)
    var running = 0.0
    for k in range(11):
        running += s(binom.pmf(pv(Float64(k)), n, p))
        assert_almost_equal(
            s(binom.cdf(pv(Float64(k)), n, p)), running, atol=1e-8
        )


def test_binomial_pmf_sums_to_one() raises:
    var total = 0.0
    for k in range(11):
        total += s(binom.pmf(pv(Float64(k)), pv(10.0), pv(0.3)))
    assert_almost_equal(total, 1.0, atol=1e-12)


def test_binomial_with_one_trial_is_bernoulli() raises:
    assert_almost_equal(
        s(binom.pmf(pv(1.0), pv(1.0), pv(0.42))), 0.42, atol=1e-13
    )
    assert_almost_equal(
        s(binom.pmf(pv(0.0), pv(1.0), pv(0.42))), 0.58, atol=1e-13
    )


# ---------------------------------------------------- the completed set
# `sf`, `isf`, `logpdf`/`logpmf`, `logcdf`, `logsf` on all nine, and the
# four `ppf`s that were missing. Reference digits are scipy.stats 1.18.


def test_survival_functions_complement_their_cdfs() raises:
    # sf + cdf == 1 for every continuous family, at a point in the body.
    assert_almost_equal(
        s(
            norm.sf(pv(1.0), pv(0.0), pv(1.0))
            + norm.cdf(pv(1.0), pv(0.0), pv(1.0))
        ),
        1.0,
        atol=1e-14,
    )
    assert_almost_equal(
        s(expon.sf(pv(1.5), pv(2.0)) + expon.cdf(pv(1.5), pv(2.0))),
        1.0,
        atol=1e-14,
    )
    assert_almost_equal(
        s(
            gamma.sf(pv(2.0), pv(3.0), pv(1.0))
            + gamma.cdf(pv(2.0), pv(3.0), pv(1.0))
        ),
        1.0,
        atol=1e-12,
    )
    assert_almost_equal(
        s(
            beta.sf(pv(0.3), pv(2.0), pv(3.0))
            + beta.cdf(pv(0.3), pv(2.0), pv(3.0))
        ),
        1.0,
        atol=BETA_ATOL,
    )
    assert_almost_equal(
        s(t.sf(pv(1.5), pv(10.0)) + t.cdf(pv(1.5), pv(10.0))),
        1.0,
        atol=BETA_ATOL,
    )
    assert_almost_equal(
        s(f.sf(pv(2.0), pv(5.0), pv(10.0)) + f.cdf(pv(2.0), pv(5.0), pv(10.0))),
        1.0,
        atol=BETA_ATOL,
    )


def test_survival_functions_match_scipy() raises:
    assert_almost_equal(
        s(norm.sf(pv(1.0), pv(0.0), pv(1.0))), 0.15865525393145707, atol=1e-12
    )
    assert_almost_equal(
        s(expon.sf(pv(1.5), pv(2.0))), 0.049787068367863944, atol=1e-14
    )
    assert_almost_equal(
        s(gamma.sf(pv(2.0), pv(3.0), pv(1.0))), 0.6766764161830636, atol=1e-8
    )
    assert_almost_equal(
        s(chi2.sf(pv(7.814727903251179), pv(3.0))), 0.05, atol=1e-8
    )
    assert_almost_equal(
        s(beta.sf(pv(0.3), pv(2.0), pv(3.0))), 0.6517, atol=BETA_ATOL
    )
    assert_almost_equal(
        s(t.sf(pv(1.5), pv(10.0))), 0.08225366322272007, atol=BETA_ATOL
    )
    assert_almost_equal(
        s(f.sf(pv(2.0), pv(5.0), pv(10.0))), 0.1641949508997389, atol=BETA_ATOL
    )


def test_survival_functions_are_one_below_the_support() raises:
    assert_almost_equal(s(expon.sf(pv(-1.0), pv(2.0))), 1.0)
    assert_almost_equal(s(gamma.sf(pv(-1.0), pv(3.0), pv(1.0))), 1.0)
    assert_almost_equal(s(f.sf(pv(-1.0), pv(5.0), pv(10.0))), 1.0)


def test_inverse_survival_functions_match_scipy() raises:
    assert_almost_equal(
        s(norm.isf(pv(0.025), pv(0.0), pv(1.0))), 1.959963984540054, atol=1e-12
    )
    assert_almost_equal(
        s(expon.isf(pv(0.2), pv(2.0))), 0.8047189562170501, atol=1e-14
    )
    assert_almost_equal(
        s(gamma.isf(pv(0.1), pv(3.0), pv(1.0))), 5.322320337834209, atol=1e-6
    )
    assert_almost_equal(
        s(chi2.isf(pv(0.05), pv(3.0))), 7.814727903251182, atol=1e-6
    )
    assert_almost_equal(
        s(beta.isf(pv(0.1), pv(2.0), pv(3.0))), 0.6795394162781817, atol=1e-6
    )
    assert_almost_equal(
        s(t.isf(pv(0.05), pv(10.0))), 1.8124611228116767, atol=1e-6
    )
    assert_almost_equal(
        s(f.isf(pv(0.05), pv(5.0), pv(10.0))), 3.3258345304130104, atol=1e-5
    )


def test_exponential_quantile_is_the_closed_form_and_inverts_its_cdf() raises:
    assert_almost_equal(
        s(expon.ppf(pv(0.5), pv(2.0))), 0.34657359027997264, atol=1e-14
    )
    for p in [0.01, 0.3, 0.9, 0.999]:
        var x = expon.ppf(pv(p), pv(0.7))
        assert_almost_equal(s(expon.cdf(x, pv(0.7))), p, atol=1e-13)


def test_f_quantile_inverts_f_cdf_and_matches_a_table_value() raises:
    # F(0.95; 5, 10) = 3.3258, the value every ANOVA table carries.
    assert_almost_equal(
        s(f.ppf(pv(0.95), pv(5.0), pv(10.0))), 3.3258345304130104, atol=1e-5
    )
    for p in [0.1, 0.5, 0.9]:
        var x = f.ppf(pv(p), pv(4.0), pv(7.0))
        assert_almost_equal(s(f.cdf(x, pv(4.0), pv(7.0))), p, atol=1e-6)


def test_poisson_quantile_is_the_smallest_k_with_cdf_at_least_p() raises:
    # scipy.stats.poisson.ppf([0.01, 0.5, 0.95], 3.0) == [0, 3, 6]; the
    # scan has to land on the integer exactly, not near it.
    assert_almost_equal(s(poisson.ppf(pv(0.01), pv(3.0))), 0.0, atol=1e-12)
    assert_almost_equal(s(poisson.ppf(pv(0.5), pv(3.0))), 3.0, atol=1e-12)
    assert_almost_equal(s(poisson.ppf(pv(0.95), pv(3.0))), 6.0, atol=1e-12)
    # The defining property, at a p strictly between two CDF values:
    # cdf(2) = 0.4232 < 0.5 <= cdf(3) = 0.6472.
    var k = poisson.ppf(pv(0.5), pv(3.0))
    assert_true(s(poisson.cdf(k, pv(3.0))) >= 0.5)
    assert_true(s(poisson.cdf(k - pv(1.0), pv(3.0))) < 0.5)


def test_binomial_quantile_is_the_smallest_k_with_cdf_at_least_p() raises:
    # scipy.stats.binom.ppf([0.02, 0.5, 0.99], 10, 0.3) == [0, 3, 7].
    assert_almost_equal(
        s(binom.ppf(pv(0.02), pv(10.0), pv(0.3))), 0.0, atol=1e-12
    )
    assert_almost_equal(
        s(binom.ppf(pv(0.5), pv(10.0), pv(0.3))), 3.0, atol=1e-12
    )
    assert_almost_equal(
        s(binom.ppf(pv(0.99), pv(10.0), pv(0.3))), 7.0, atol=1e-12
    )


def test_discrete_survival_functions_match_scipy_and_their_cdfs() raises:
    assert_almost_equal(
        s(poisson.sf(pv(3.0), pv(2.5))), 0.2424238668669339, atol=1e-10
    )
    assert_almost_equal(
        s(poisson.sf(pv(3.0), pv(2.5)) + poisson.cdf(pv(3.0), pv(2.5))),
        1.0,
        atol=1e-12,
    )
    assert_almost_equal(
        s(binom.sf(pv(3.0), pv(10.0), pv(0.3))),
        0.3503892815999998,
        atol=BETA_ATOL,
    )
    # The two ends: nothing above n, everything above -1.
    assert_almost_equal(s(binom.sf(pv(10.0), pv(10.0), pv(0.3))), 0.0)
    assert_almost_equal(s(binom.sf(pv(-1.0), pv(10.0), pv(0.3))), 1.0)
    assert_almost_equal(s(poisson.sf(pv(-1.0), pv(2.5))), 1.0)


def test_log_densities_match_scipy_and_are_the_log_of_the_density() raises:
    assert_almost_equal(
        s(norm.logpdf(pv(1.0), pv(0.0), pv(2.0))),
        -1.737085713764618,
        atol=1e-12,
    )
    assert_almost_equal(
        s(gamma.logpdf(pv(2.0), pv(3.0), pv(1.0))),
        -1.3068528194400546,
        atol=1e-10,
    )
    assert_almost_equal(
        s(beta.logpdf(pv(0.3), pv(2.0), pv(3.0))),
        0.5675839575845993,
        atol=1e-10,
    )
    assert_almost_equal(
        s(t.logpdf(pv(1.5), pv(10.0))), -2.0600719941327488, atol=1e-10
    )
    assert_almost_equal(
        s(f.logpdf(pv(2.0), pv(5.0), pv(10.0))), -1.8201234988216655, atol=1e-10
    )
    assert_almost_equal(
        s(poisson.logpmf(pv(3.0), pv(2.5))), -1.5428872736055896, atol=1e-10
    )
    assert_almost_equal(
        s(binom.logpmf(pv(3.0), pv(10.0), pv(0.3))),
        -1.321151277766889,
        atol=1e-10,
    )
    # And on the support, exp(logpdf) is the density itself -- the
    # densities are now *defined* that way, so this pins the refactor.
    assert_almost_equal(
        s(gamma.logpdf(pv(2.0), pv(3.0), pv(1.0)).exp()),
        s(gamma.pdf(pv(2.0), pv(3.0), pv(1.0))),
        atol=1e-15,
    )


def test_log_densities_are_finite_and_exp_to_zero_off_the_support() raises:
    # SciPy says -inf; a tier-1 kernel says a large finite negative whose
    # exp is exactly the 0 the density returns. Both halves are the claim.
    var off = gamma.logpdf(pv(-1.0), pv(3.0), pv(1.0))
    assert_true(s(off) < -1e29)
    assert_almost_equal(s(off.exp()), 0.0)
    assert_almost_equal(s(gamma.pdf(pv(-1.0), pv(3.0), pv(1.0))), 0.0)
    var off_k = binom.logpmf(pv(11.0), pv(10.0), pv(0.3))
    assert_true(s(off_k) < -1e29)


def test_log_cdfs_are_the_log_of_the_cdf() raises:
    assert_almost_equal(
        s(norm.logcdf(pv(-3.0), pv(0.0), pv(1.0))), -6.60772622151035, atol=1e-8
    )
    assert_almost_equal(
        s(t.logsf(pv(1.5), pv(10.0))),
        s(t.sf(pv(1.5), pv(10.0)).ln()),
        atol=1e-14,
    )
    assert_almost_equal(
        s(chi2.logcdf(pv(2.0), pv(3.0))),
        s(chi2.cdf(pv(2.0), pv(3.0)).ln()),
        atol=1e-14,
    )


# ---------------------------------------------------------------- shared


def test_simd_lanes_are_independent() raises:
    comptime w = 4
    comptime PW = Plain[dtype, w]
    var x = PW(SIMD[dtype, w](-1.0, 0.0, 1.0, 2.0))
    var result = norm.cdf(x, PW.constant(0.0), PW.constant(1.0))
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
    var by_scale = gamma.cdf(dc(3.0), dc(2.5), dv(1.7)).deriv.copy()
    # Finite-difference cross-check, since there's no tidy closed form.
    var h = 1e-6
    var up = s(gamma.cdf(pv(3.0), pv(2.5), pv(1.7 + h)))
    var down = s(gamma.cdf(pv(3.0), pv(2.5), pv(1.7 - h)))
    assert_almost_equal(s(by_scale), (up - down) / (2.0 * h), atol=1e-7)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
