"""Tests for `multivariate_normal.logpdf`, `pdf`, `rvs` and `entropy`
against SciPy on a correlated 3-D normal, four points including the mean,
and `rvs` by its draws' mean."""

from std.testing import TestSuite, assert_almost_equal, assert_true

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import Generator, multivariate_normal

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _mean() raises -> Static[f64, 3]:
    return Static[f64, 3]([1.0, -2.0, 0.5], _cpu())


def _cov() raises -> Static[f64, 3, 3]:
    return Static[f64, 3, 3](
        [2.0, 0.6, 0.2, 0.6, 1.5, -0.3, 0.2, -0.3, 1.0], _cpu()
    )


def _x() raises -> Static[f64, 4, 3]:
    return Static[f64, 4, 3](
        [1.2, -1.5, 0.0, 0.0, -2.0, 1.0, 3.0, 0.5, -1.0, 1.0, -2.0, 0.5], _cpu()
    )


def test_logpdf_pdf_and_entropy_match_scipy() raises:
    var lp = multivariate_normal.logpdf(_x(), _mean(), _cov()).to_host()
    var p = multivariate_normal.pdf(_x(), _mean(), _cov()).to_host()
    var want_lp = [
        -3.3533753473664833,
        -3.727000776919748,
        -6.338684625716997,
        -3.179320364548614,
    ]
    var want_p = [
        0.03496613185144691,
        0.024064903723523157,
        0.0017666244815998026,
        0.041613927813136234,
    ]
    for i in range(4):
        assert_almost_equal(Float64(lp[i]), want_lp[i], atol=1e-12)
        assert_almost_equal(Float64(p[i]), want_p[i], atol=1e-13)
    assert_almost_equal(
        multivariate_normal.entropy(_cov()), 4.679320364548614, atol=1e-12
    )


def test_rvs_have_the_mean() raises:
    var rng = Generator(seed=81)
    comptime n = 50000
    var draws = multivariate_normal.rvs[count=n](_mean(), _cov(), rng).to_host()
    var m = List[Float64](length=3, fill=0.0)
    for i in range(n):
        for j in range(3):
            m[j] += Float64(draws[3 * i + j])
    var want = [1.0, -2.0, 0.5]
    for j in range(3):
        assert_true(abs(m[j] / Float64(n) - want[j]) < 0.03)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
