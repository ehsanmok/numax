"""Tests for the discrete distribution namespaces added in 0.3:
`bernoulli`, `geom`, `nbinom` and `hypergeom`, each against SciPy's frozen
distribution -- `pmf`, `cdf`, `sf` at points across the support, `ppf`
at three probabilities, the moments, the entropy and the 80% interval --
and each `rvs` by the moments of its draws, through `Generator`'s
samplers (`negative_binomial`, `hypergeometric`).
"""

from std.math import sqrt
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax.core.plain import Plain
from numax.stats import Generator, bernoulli, geom, hypergeom, nbinom

comptime _P = Plain[DType.float64, 1]


def _v(x: _P) -> Float64:
    return Float64(x.v[0])


def _near(got: Float64, want: Float64) raises:
    assert_almost_equal(got, want, atol=1e-11, rtol=1e-10)


def test_discrete_families_match_scipy() raises:
    _near(_v(bernoulli.pmf(_P(0.0), _P(0.3))), 0.6999999999999997)
    _near(_v(bernoulli.cdf(_P(0.0), _P(0.3))), 0.7)
    _near(_v(bernoulli.sf(_P(0.0), _P(0.3))), 0.3)
    _near(_v(bernoulli.pmf(_P(1.0), _P(0.3))), 0.3)
    _near(_v(bernoulli.cdf(_P(1.0), _P(0.3))), 1.0)
    _near(_v(bernoulli.sf(_P(1.0), _P(0.3))), 0.0)
    _near(_v(bernoulli.pmf(_P(2.0), _P(0.3))), 0.0)
    _near(_v(bernoulli.cdf(_P(2.0), _P(0.3))), 1.0)
    _near(_v(bernoulli.sf(_P(2.0), _P(0.3))), 0.0)
    _near(_v(bernoulli.ppf(_P(0.2), _P(0.3))), 0.0)
    _near(_v(bernoulli.ppf(_P(0.7), _P(0.3))), 0.0)
    _near(_v(bernoulli.ppf(_P(0.95), _P(0.3))), 1.0)
    _near(bernoulli.mean(0.3), 0.3)
    _near(bernoulli.var(0.3), 0.21)
    _near(bernoulli.entropy(0.3), 0.6108643020548935)
    var i_bernoulli = bernoulli.interval(0.8, 0.3)
    _near(i_bernoulli[0], 0.0)
    _near(i_bernoulli[1], 1.0)

    _near(_v(geom.pmf(_P(1.0), _P(0.25))), 0.25)
    _near(_v(geom.cdf(_P(1.0), _P(0.25))), 0.24999999999999997)
    _near(_v(geom.sf(_P(1.0), _P(0.25))), 0.75)
    _near(_v(geom.pmf(_P(3.0), _P(0.25))), 0.140625)
    _near(_v(geom.cdf(_P(3.0), _P(0.25))), 0.578125)
    _near(_v(geom.sf(_P(3.0), _P(0.25))), 0.42187500000000006)
    _near(_v(geom.pmf(_P(8.0), _P(0.25))), 0.0333709716796875)
    _near(_v(geom.cdf(_P(8.0), _P(0.25))), 0.8998870849609375)
    _near(_v(geom.sf(_P(8.0), _P(0.25))), 0.10011291503906253)
    _near(_v(geom.ppf(_P(0.1), _P(0.25))), 1.0)
    _near(_v(geom.ppf(_P(0.5), _P(0.25))), 3.0)
    _near(_v(geom.ppf(_P(0.9), _P(0.25))), 9.0)
    _near(geom.mean(0.25), 4.0)
    _near(geom.var(0.25), 12.0)
    _near(geom.entropy(0.25), 2.249340578475233)
    var i_geom = geom.interval(0.8, 0.25)
    _near(i_geom[0], 1.0)
    _near(i_geom[1], 9.0)

    _near(_v(nbinom.pmf(_P(0.0), _P(3.5), _P(0.4))), 0.04047715405015527)
    _near(_v(nbinom.cdf(_P(0.0), _P(3.5), _P(0.4))), 0.04047715405015526)
    _near(_v(nbinom.sf(_P(0.0), _P(3.5), _P(0.4))), 0.9595228459498447)
    _near(_v(nbinom.pmf(_P(4.0), _P(3.5), _P(0.4))), 0.12307230478277402)
    _near(_v(nbinom.cdf(_P(4.0), _P(3.5), _P(0.4))), 0.4895322189758545)
    _near(_v(nbinom.sf(_P(4.0), _P(3.5), _P(0.4))), 0.5104677810241455)
    _near(_v(nbinom.pmf(_P(12.0), _P(3.5), _P(0.4))), 0.018533068544838452)
    _near(_v(nbinom.cdf(_P(12.0), _P(3.5), _P(0.4))), 0.9565843049498912)
    _near(_v(nbinom.sf(_P(12.0), _P(3.5), _P(0.4))), 0.04341569505010895)
    _near(_v(nbinom.ppf(_P(0.1), _P(3.5), _P(0.4))), 1.0)
    _near(_v(nbinom.ppf(_P(0.5), _P(3.5), _P(0.4))), 5.0)
    _near(_v(nbinom.ppf(_P(0.9), _P(3.5), _P(0.4))), 10.0)
    _near(nbinom.mean(3.5, 0.4), 5.25)
    _near(nbinom.var(3.5, 0.4), 13.124999999999998)
    _near(nbinom.entropy(3.5, 0.4), 2.58801389486742)
    var i_nbinom = nbinom.interval(0.8, 3.5, 0.4)
    _near(i_nbinom[0], 1.0)
    _near(i_nbinom[1], 10.0)

    _near(
        _v(hypergeom.pmf(_P(1.0), _P(30.0), _P(12.0), _P(8.0))),
        0.06524737631184407,
    )
    _near(
        _v(hypergeom.cdf(_P(1.0), _P(30.0), _P(12.0), _P(8.0))),
        0.07272363818090954,
    )
    _near(
        _v(hypergeom.sf(_P(1.0), _P(30.0), _P(12.0), _P(8.0))),
        0.9272763618190905,
    )
    _near(
        _v(hypergeom.pmf(_P(3.0), _P(30.0), _P(12.0), _P(8.0))),
        0.32205435743666627,
    )
    _near(
        _v(hypergeom.cdf(_P(3.0), _P(30.0), _P(12.0), _P(8.0))),
        0.604113327951409,
    )
    _near(
        _v(hypergeom.sf(_P(3.0), _P(30.0), _P(12.0), _P(8.0))),
        0.395886672048591,
    )
    _near(
        _v(hypergeom.pmf(_P(6.0), _P(30.0), _P(12.0), _P(8.0))),
        0.02415407680774997,
    )
    _near(
        _v(hypergeom.cdf(_P(6.0), _P(30.0), _P(12.0), _P(8.0))),
        0.9974797216776228,
    )
    _near(
        _v(hypergeom.sf(_P(6.0), _P(30.0), _P(12.0), _P(8.0))),
        0.002520278322377273,
    )
    _near(_v(hypergeom.ppf(_P(0.1), _P(30.0), _P(12.0), _P(8.0))), 2.0)
    _near(_v(hypergeom.ppf(_P(0.5), _P(30.0), _P(12.0), _P(8.0))), 3.0)
    _near(_v(hypergeom.ppf(_P(0.9), _P(30.0), _P(12.0), _P(8.0))), 5.0)
    _near(hypergeom.mean(30.0, 12.0, 8.0), 3.2)
    _near(hypergeom.var(30.0, 12.0, 8.0), 1.456551724137931)
    _near(hypergeom.entropy(30.0, 12.0, 8.0), 1.6054678210861735)
    var i_hypergeom = hypergeom.interval(0.8, 30.0, 12.0, 8.0)
    _near(i_hypergeom[0], 2.0)
    _near(i_hypergeom[1], 5.0)


def _check[
    dtype: DType
](values: List[Scalar[dtype]], mean: Float64, var_: Float64) raises:
    var total = 0.0
    for i in range(len(values)):
        total += Float64(values[i])
    var m = total / Float64(len(values))
    var sq = 0.0
    for i in range(len(values)):
        sq += (Float64(values[i]) - m) ** 2
    assert_true(abs(m - mean) < 5 * sqrt(var_ / Float64(len(values))))
    assert_true(abs(sq / Float64(len(values) - 1) - var_) < 0.08 * var_)


def test_rvs_have_the_distribution_moments() raises:
    var rng = Generator(seed=61)
    comptime n = 100000
    _check(bernoulli.rvs[DType.int64, n](0.3, rng).to_host(), 0.3, 0.21)
    _check(geom.rvs[DType.float64, n](0.25, rng).to_host(), 4.0, 12.0)
    _check(
        nbinom.rvs[DType.int64, n](3.5, 0.4, rng).to_host(),
        nbinom.mean(3.5, 0.4),
        nbinom.var(3.5, 0.4),
    )
    _check(
        hypergeom.rvs[DType.int64, n](30, 12, 8, rng).to_host(),
        hypergeom.mean(30.0, 12.0, 8.0),
        hypergeom.var(30.0, 12.0, 8.0),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
