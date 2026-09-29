"""Tests for `RBFInterpolator`, `NearestNDInterpolator` and `griddata`
against SciPy on scattered 2-D data: every kernel at SciPy's default
degree, smoothing, no polynomial tail, and the nearest-site rule; and
that an interpolant reproduces its data."""

from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_almost_equal, assert_raises

from numax.core.tensor import Static
from numax.interpolate import NearestNDInterpolator, RBFInterpolator, griddata

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _y() raises -> Static[f64, 8, 2]:
    return Static[f64, 8, 2](
        [
            0.0,
            0.0,
            1.0,
            0.2,
            0.3,
            1.0,
            1.2,
            1.1,
            0.6,
            0.5,
            0.1,
            0.8,
            0.9,
            0.7,
            0.4,
            0.1,
        ],
        _cpu(),
    )


def _d() raises -> Static[f64, 8]:
    return Static[f64, 8](
        [
            1.0,
            1.73463304173536,
            -0.42535002320541004,
            -0.3120165893577138,
            1.002776287634929,
            -0.5387243847461846,
            0.469001526278338,
            1.6726925800251289,
        ],
        _cpu(),
    )


def _x() raises -> Static[f64, 5, 2]:
    return Static[f64, 5, 2](
        [0.5, 0.5, 0.2, 0.3, 1.0, 1.0, -0.2, 0.4, 0.7, 0.2], _cpu()
    )


def _close(got: List[Float64], want: List[Float64], atol: Float64) raises:
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=atol)


def test_kernels_match_scipy() raises:
    _close(
        RBFInterpolator[f64](_y(), _d(), "thin_plate_spline", 1.0)(
            _x()
        ).to_host(),
        [
            0.9096403224752136,
            0.8373461335963543,
            -0.15791622857669563,
            -0.23484784686799287,
            1.717553112062674,
        ],
        1e-11,
    )
    _close(
        RBFInterpolator[f64](_y(), _d(), "linear", 1.0)(_x()).to_host(),
        [
            0.8674856083725573,
            0.8914234291837937,
            -0.11483746365402439,
            0.2353984749040301,
            1.6157282912014188,
        ],
        1e-11,
    )
    _close(
        RBFInterpolator[f64](_y(), _d(), "cubic", 1.0)(_x()).to_host(),
        [
            0.9178563940077791,
            0.8234921504098751,
            -0.12435431345374348,
            -0.46662363766570597,
            1.751789627112912,
        ],
        1e-11,
    )
    _close(
        RBFInterpolator[f64](_y(), _d(), "quintic", 1.0)(_x()).to_host(),
        [
            0.9280099972901342,
            0.8468406051686916,
            -0.10330628960174604,
            -0.6625368419094002,
            1.7612620291888625,
        ],
        1e-11,
    )
    _close(
        RBFInterpolator[f64](_y(), _d(), "multiquadric", 0.8)(_x()).to_host(),
        [
            0.9177367586584211,
            0.8342283217486539,
            -0.1408092981808693,
            -0.48299590536353776,
            1.7822748116750091,
        ],
        1e-11,
    )
    _close(
        RBFInterpolator[f64](_y(), _d(), "inverse_multiquadric", 1.3)(
            _x()
        ).to_host(),
        [
            0.8829066061757636,
            0.8592862392626945,
            -0.19021282539257034,
            -0.04076656030387049,
            1.8299355804116098,
        ],
        1e-11,
    )
    _close(
        RBFInterpolator[f64](_y(), _d(), "inverse_quadratic", 1.1)(
            _x()
        ).to_host(),
        [
            0.878938238715979,
            0.8599807014034319,
            -0.1929368194290051,
            -0.028307051314378384,
            1.8391799415744272,
        ],
        1e-11,
    )
    _close(
        RBFInterpolator[f64](_y(), _d(), "gaussian", 0.9)(_x()).to_host(),
        [
            0.900296995368663,
            0.8507088375267617,
            -0.14680685542714667,
            -0.30269611857665524,
            1.8234062534959907,
        ],
        1e-11,
    )


def test_smoothing_and_degree() raises:
    _close(
        RBFInterpolator[f64](_y(), _d(), smoothing=0.1)(_x()).to_host(),
        [
            0.8527065991436573,
            0.8390651765683352,
            -0.14592290208935016,
            -0.09655450460860782,
            1.6659027663069592,
        ],
        1e-11,
    )
    _close(
        RBFInterpolator[f64](_y(), _d(), "gaussian", 0.9, degree=-1)(
            _x()
        ).to_host(),
        [
            0.9024025697132245,
            0.8616210572426777,
            -0.14178969586265566,
            -0.3060456252194257,
            1.8217721982249255,
        ],
        1e-11,
    )


def test_interpolates_the_data() raises:
    var r = RBFInterpolator[f64](_y(), _d())
    _close(r(_y()).to_host(), _d().to_host(), 1e-11)


def test_nearest_and_griddata() raises:
    var nearest = NearestNDInterpolator[f64](_y(), _d())
    _close(
        nearest(_x()).to_host(),
        [
            1.002776287634929,
            1.6726925800251289,
            -0.3120165893577138,
            1.0,
            1.73463304173536,
        ],
        0.0,
    )
    _close(
        griddata(_y(), _d(), _x(), "nearest").to_host(),
        [
            1.002776287634929,
            1.6726925800251289,
            -0.3120165893577138,
            1.0,
            1.73463304173536,
        ],
        0.0,
    )
    with assert_raises(contains="Delaunay"):
        _ = griddata(_y(), _d(), _x())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
