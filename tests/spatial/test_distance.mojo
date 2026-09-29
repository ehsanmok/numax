"""Tests for `cdist`, `pdist` and `squareform` against SciPy's digits for
every metric, including a constant row, whose correlation distance is NaN
as SciPy's is, and the round trip between the condensed and square
forms."""

from std.math import isnan, nan
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax.core.tensor import Static
from numax.spatial import cdist, pdist, squareform

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _a() raises -> Static[f64, 3, 3]:
    return Static[f64, 3, 3](
        [0.0, 1.0, 2.0, 1.5, -1.0, 0.3, 2.0, 2.0, 2.0], _cpu()
    )


def _b() raises -> Static[f64, 2, 3]:
    return Static[f64, 2, 3]([1.0, 0.0, 0.0, 0.5, 0.5, -1.0], _cpu())


def _close(got: List[Float64], want: List[Float64]) raises:
    for i in range(len(want)):
        if isnan(want[i]):
            assert_true(isnan(got[i]))
        else:
            assert_almost_equal(got[i], want[i], atol=1e-14)


def test_cdist_matches_scipy() raises:
    _close(
        cdist(_a(), _b()).to_host(),
        [
            2.449489742783178,
            3.082207001484488,
            1.1575836902790226,
            2.222611077089287,
            3.0,
            3.6742346141747673,
        ],
    )
    _close(
        cdist(_a(), _b(), "sqeuclidean").to_host(),
        [6.0, 9.5, 1.34, 4.94, 9.0, 13.5],
    )
    _close(
        cdist(_a(), _b(), "cityblock").to_host(), [4.0, 4.0, 1.8, 3.8, 5.0, 6.0]
    )
    _close(
        cdist(_a(), _b(), "chebyshev").to_host(), [2.0, 3.0, 1.0, 1.5, 2.0, 3.0]
    )
    _close(
        cdist(_a(), _b(), "cosine").to_host(),
        [
            1.0,
            1.547722557505166,
            0.17923651725312584,
            1.0223383525804386,
            0.42264973081037416,
            1.0,
        ],
    )
    _close(
        cdist(_a(), _b(), "correlation").to_host(),
        [
            1.8660254037844384,
            1.8660254037844388,
            0.14574937140172128,
            1.0230878548269804,
            nan[DType.float64](),
            nan[DType.float64](),
        ],
    )
    _close(
        cdist(_a(), _b(), "minkowski", 3.0).to_host(),
        [
            2.154434690031884,
            3.0092308274032264,
            1.0482965576835586,
            1.8731210807451883,
            2.571281590658235,
            3.2316520350478255,
        ],
    )


def test_pdist_and_squareform() raises:
    var condensed = pdist(_a())
    _close(
        condensed.to_host(),
        [3.023243291566195, 2.23606797749979, 3.4842502780368694],
    )
    var square = squareform(condensed)
    _close(
        square.to_host(),
        [
            0.0,
            3.023243291566195,
            2.23606797749979,
            3.023243291566195,
            0.0,
            3.4842502780368694,
            2.23606797749979,
            3.4842502780368694,
            0.0,
        ],
    )
    _close(squareform(square).to_host(), condensed.to_host())
    # `pdist` is `cdist` of the set against itself, condensed.
    _close(squareform(cdist(_a(), _a())).to_host(), condensed.to_host())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
