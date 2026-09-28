"""Tests for `Generator.gamma`, `beta`, `lognormal`, `poisson` and
`binomial`.

Each sample of 200,000 draws against its distribution's mean and variance,
to a tolerance of about five standard errors, across the parameter
regimes the samplers switch between -- gamma below and above shape 1,
Poisson and binomial on both sides of their inversion/rejection splits and
with `p > 1/2` reflected -- plus reproducibility: two generators from one
seed agree draw for draw.
"""

from std.math import exp, sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from numax.stats import Generator

comptime f64 = DType.float64
comptime n = 200000


def _moments(values: List[Float64]) -> Tuple[Float64, Float64]:
    var total = 0.0
    for i in range(len(values)):
        total += values[i]
    var mean = total / Float64(len(values))
    var sq = 0.0
    for i in range(len(values)):
        sq += (values[i] - mean) ** 2
    return (mean, sq / Float64(len(values) - 1))


def _check(values: List[Float64], mean: Float64, variance: Float64) raises:
    var m = _moments(values)
    var se = sqrt(variance / Float64(len(values)))
    assert_true(abs(m[0] - mean) < 5 * se + 1e-12, "mean off")
    # The sample variance's standard error, loosely: a few percent at n.
    assert_true(abs(m[1] - variance) < 0.05 * variance + 1e-12, "variance off")


def _as_f64[dtype: DType](values: List[Scalar[dtype]]) -> List[Float64]:
    var out = List[Float64](capacity=len(values))
    for i in range(len(values)):
        out.append(Float64(values[i]))
    return out^


def test_gamma_and_beta_moments() raises:
    var rng = Generator(seed=11)
    _check(_as_f64(rng.gamma[f64, n](3.5, 2.0).to_host()), 7.0, 14.0)
    _check(_as_f64(rng.gamma[f64, n](0.4, 1.0).to_host()), 0.4, 0.4)
    var a = 2.0
    var b = 5.0
    _check(
        _as_f64(rng.beta[f64, n](a, b).to_host()),
        a / (a + b),
        a * b / ((a + b) ** 2 * (a + b + 1)),
    )


def test_lognormal_moments() raises:
    var rng = Generator(seed=12)
    var mu = 0.3
    var sigma = 0.5
    _check(
        _as_f64(rng.lognormal[f64, n](mu, sigma).to_host()),
        exp(mu + sigma * sigma / 2),
        (exp(sigma * sigma) - 1) * exp(2 * mu + sigma * sigma),
    )


def test_poisson_moments_on_both_sides_of_the_split() raises:
    var rng = Generator(seed=13)
    _check(_as_f64(rng.poisson[DType.int64, n](3.0).to_host()), 3.0, 3.0)
    _check(_as_f64(rng.poisson[DType.int64, n](250.0).to_host()), 250.0, 250.0)


def test_binomial_moments_on_both_sides_and_reflected() raises:
    var rng = Generator(seed=14)
    _check(_as_f64(rng.binomial[DType.int64, n](20, 0.3).to_host()), 6.0, 4.2)
    _check(
        _as_f64(rng.binomial[DType.int64, n](1000, 0.4).to_host()), 400.0, 240.0
    )
    _check(
        _as_f64(rng.binomial[DType.int64, n](1000, 0.9).to_host()), 900.0, 90.0
    )


def test_generators_from_one_seed_agree() raises:
    var a = Generator(seed=5)
    var b = Generator(seed=5)
    var x = a.poisson[DType.int64, 64](40.0).to_host()
    var y = b.poisson[DType.int64, 64](40.0).to_host()
    for i in range(64):
        assert_equal(x[i], y[i])
    var g = a.gamma[f64, 64](2.0).to_host()
    var h = b.gamma[f64, 64](2.0).to_host()
    for i in range(64):
        assert_equal(g[i], h[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
