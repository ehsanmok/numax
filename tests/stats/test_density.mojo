"""Tests for `median_abs_deviation`, `ecdf` and `gaussian_kde` against
SciPy on a sample with ties: the raw and normal-scaled MAD, the ECDF's
distinct values and probabilities, and the KDE's density under Scott's
rule, Silverman's, and an explicit factor."""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import ecdf, gaussian_kde, median_abs_deviation

comptime f64 = DType.float64


def _x() raises -> Static[f64, 12]:
    return Static[f64, 12](
        [2.1, 3.5, 2.1, 4.0, 5.2, 3.5, 3.5, 1.0, 6.3, 2.8, 4.4, 3.9],
        DeviceContext(api="cpu"),
    )


def _pts() raises -> Static[f64, 5]:
    return Static[f64, 5]([0.5, 2.0, 3.5, 5.0, 7.0], DeviceContext(api="cpu"))


def _close(got: List[Float64], want: List[Float64]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-13)


def test_mad_and_ecdf_match_scipy() raises:
    assert_almost_equal(
        median_abs_deviation(_x()), 0.8000000000000003, atol=1e-13
    )
    assert_almost_equal(
        median_abs_deviation(_x(), 0.6744897501960817),
        1.1860817748044818,
        atol=1e-12,
    )
    var e = ecdf(_x())
    _close(e.quantiles.to_host(), [1.0, 2.1, 2.8, 3.5, 3.9, 4.0, 4.4, 5.2, 6.3])
    _close(
        e.probabilities.to_host(),
        [
            0.08333333333333333,
            0.25,
            0.3333333333333333,
            0.5833333333333334,
            0.6666666666666666,
            0.75,
            0.8333333333333334,
            0.9166666666666666,
            1.0,
        ],
    )


def test_gaussian_kde_matches_scipy() raises:
    var scott = gaussian_kde[f64].create(_x())
    _close(
        scott.evaluate(_pts()).to_host(),
        [
            0.04802770130022792,
            0.15378927397109643,
            0.2585972066286555,
            0.14466781587094787,
            0.032808100123950565,
        ],
    )
    var silverman = gaussian_kde[f64].create(_x(), "silverman")
    _close(
        silverman.evaluate(_pts()).to_host(),
        [
            0.04939296973706527,
            0.1543273678748018,
            0.2519572187610566,
            0.14693484640446824,
            0.0334802624538273,
        ],
    )
    var fixed = gaussian_kde[f64].with_factor(_x(), 0.4)
    _close(
        fixed.evaluate(_pts()).to_host(),
        [
            0.04202435230861689,
            0.15484861790977567,
            0.3100839540878431,
            0.12012457183727741,
            0.027941804920762203,
        ],
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
