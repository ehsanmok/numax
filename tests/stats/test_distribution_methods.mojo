"""Tests for `mean`, `var`, `std`, `interval`, `entropy` and `rvs` on the
nine `numax.stats` distributions.

The moments, the 90% central interval and the entropy against SciPy's
frozen distributions for every family, and each `rvs` by the moments of
100,000 draws -- including `t` and `f`, which draw by inverting their
`ppf`, and the two discrete families through `Generator`'s samplers.
The interval is held to `1e-6`, the `ppf` Newton iterations' own
accuracy; the rest to rounding.
"""

from std.math import sqrt
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax.stats import (
    Generator,
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

comptime f64 = DType.float64


def _near(got: Float64, want: Float64, tol: Float64 = 1e-10) raises:
    assert_almost_equal(got, want, atol=tol, rtol=tol)


def test_moments_intervals_and_entropy_match_scipy() raises:
    _near(norm.mean(1.5, 2.0), 1.5)
    _near(norm.var(1.5, 2.0), 4.0)
    _near(norm.std(1.5, 2.0), 2.0)
    _near(norm.entropy(1.5, 2.0), 2.112085713764618)
    var i_norm = norm.interval(0.9, 1.5, 2.0)
    _near(i_norm[0], -1.7897072539029457, 1e-6)
    _near(i_norm[1], 4.789707253902944, 1e-6)
    _near(expon.mean(0.5), 2.0)
    _near(expon.var(0.5), 4.0)
    _near(expon.std(0.5), 2.0)
    _near(expon.entropy(0.5), 1.6931471805599454)
    var i_expon = expon.interval(0.9, 0.5)
    _near(i_expon[0], 0.10258658877510105, 1e-6)
    _near(i_expon[1], 5.99146454710798, 1e-6)
    _near(gamma.mean(2.5, 1.5), 3.75)
    _near(gamma.var(2.5, 1.5), 5.625)
    _near(gamma.std(2.5, 1.5), 2.3717082451262845)
    _near(gamma.entropy(2.5, 1.5), 2.135413017613219)
    var i_gamma = gamma.interval(0.9, 2.5, 1.5)
    _near(i_gamma[0], 0.8591071695463268, 1e-6)
    _near(i_gamma[1], 8.302873270137264, 1e-6)
    _near(chi2.mean(4.0), 4.0)
    _near(chi2.var(4.0), 8.0)
    _near(chi2.std(4.0), 2.8284271247461903)
    _near(chi2.entropy(4.0), 2.270362845461478)
    var i_chi2 = chi2.interval(0.9, 4.0)
    _near(i_chi2[0], 0.7107230213973239, 1e-6)
    _near(i_chi2[1], 9.487729036781154, 1e-6)
    _near(beta.mean(2.0, 3.0), 0.4)
    _near(beta.var(2.0, 3.0), 0.04)
    _near(beta.std(2.0, 3.0), 0.2)
    _near(beta.entropy(2.0, 3.0), -0.2349066497880008)
    var i_beta = beta.interval(0.9, 2.0, 3.0)
    _near(i_beta[0], 0.09761146288641433, 1e-6)
    _near(i_beta[1], 0.7513953742698181, 1e-6)
    _near(t.mean(5.0), 0.0)
    _near(t.var(5.0), 1.6666666666666667)
    _near(t.std(5.0), 1.2909944487358056)
    _near(t.entropy(5.0), 1.627502672414396)
    var i_t = t.interval(0.9, 5.0)
    _near(i_t[0], -2.0150483733330242, 1e-6)
    _near(i_t[1], 2.015048373333023, 1e-6)
    _near(f.mean(4.0, 9.0), 1.2857142857142858)
    _near(f.var(4.0, 9.0), 1.8183673469387756)
    _near(f.std(4.0, 9.0), 1.3484685190759091)
    _near(f.entropy(4.0, 9.0), 1.1944539447253177)
    var i_f = f.interval(0.9, 4.0, 9.0)
    _near(i_f[0], 0.16670058936947543, 1e-6)
    _near(i_f[1], 3.6330885114190794, 1e-6)
    _near(poisson.mean(3.5), 3.5)
    _near(poisson.var(3.5), 3.5)
    _near(poisson.std(3.5), 1.8708286933869707)
    _near(poisson.entropy(3.5), 2.015172522512972)
    var i_poisson = poisson.interval(0.9, 3.5)
    _near(i_poisson[0], 1.0, 1e-6)
    _near(i_poisson[1], 7.0, 1e-6)
    _near(binom.mean(20.0, 0.3), 6.0)
    _near(binom.var(20.0, 0.3), 4.2)
    _near(binom.std(20.0, 0.3), 2.04939015319192)
    _near(binom.entropy(20.0, 0.3), 2.132538641661489)
    var i_binom = binom.interval(0.9, 20.0, 0.3)
    _near(i_binom[0], 3.0, 1e-6)
    _near(i_binom[1], 9.0, 1e-6)


def _sample_moments[
    dtype: DType
](values: List[Scalar[dtype]]) -> Tuple[Float64, Float64]:
    var total = 0.0
    for i in range(len(values)):
        total += Float64(values[i])
    var m = total / Float64(len(values))
    var sq = 0.0
    for i in range(len(values)):
        sq += (Float64(values[i]) - m) ** 2
    return (m, sq / Float64(len(values) - 1))


def _check_draws[
    dtype: DType
](values: List[Scalar[dtype]], mean: Float64, var_: Float64) raises:
    var got = _sample_moments(values)
    assert_true(abs(got[0] - mean) < 5 * sqrt(var_ / Float64(len(values))))
    assert_true(abs(got[1] - var_) < 0.08 * var_)


def test_rvs_draws_have_the_distribution_moments() raises:
    var rng = Generator(seed=77)
    comptime n = 100000
    _check_draws(norm.rvs[f64, n](1.5, 2.0, rng).to_host(), 1.5, 4.0)
    _check_draws(expon.rvs[f64, n](0.5, rng).to_host(), 2.0, 4.0)
    _check_draws(chi2.rvs[f64, n](4.0, rng).to_host(), 4.0, 8.0)
    _check_draws(t.rvs[f64, n](7.0, rng).to_host(), 0.0, 7.0 / 5.0)
    _check_draws(
        f.rvs[f64, n](6.0, 12.0, rng).to_host(),
        f.mean(6.0, 12.0),
        f.var(6.0, 12.0),
    )
    _check_draws(poisson.rvs[DType.int64, n](3.5, rng).to_host(), 3.5, 3.5)
    _check_draws(binom.rvs[DType.int64, n](20.0, 0.3, rng).to_host(), 6.0, 4.2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
