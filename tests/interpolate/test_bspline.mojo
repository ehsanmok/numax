"""Tests for `BSpline` and `make_interp_spline` against SciPy at degrees
1, 2, 3 and 5 on non-uniform data: the knots SciPy picks, the solved
coefficients, values and first derivatives at points inside and outside
the base interval, and the integral through the antiderivative; plus the
coefficient maps' identities -- the derivative of the antiderivative is
the spline, and an interpolant reproduces its data."""

from std.math import sin
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_almost_equal

from numax.core.tensor import Static
from numax.interpolate import BSpline, make_interp_spline

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _x() raises -> Static[f64, 9]:
    return Static[f64, 9]([0.0, 0.4, 1.1, 1.5, 2.3, 3.0, 3.2, 4.1, 5.0], _cpu())


def _y() raises -> Static[f64, 9]:
    var xs = _x().to_host()
    var values = List[Float64](capacity=9)
    for i in range(9):
        values.append(sin(xs[i]) + 0.1 * xs[i] * xs[i])
    return Static[f64, 9](values^, _cpu())


def _q() raises -> Static[f64, 7]:
    return Static[f64, 7]([-0.3, 0.2, 1.0, 2.0, 3.1, 4.5, 5.3], _cpu())


def _close(got: List[Float64], want: List[Float64], atol: Float64) raises:
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=atol)


def test_degree_1_matches_scipy() raises:
    var s = make_interp_spline[k=1](_x(), _y())
    _close(
        s.t.to_host(),
        [0.0, 0.0, 0.4, 1.1, 1.5, 2.3, 3.0, 3.2, 4.1, 5.0, 5.0],
        0.0,
    )
    _close(
        s.c.to_host(),
        [
            0.0,
            0.40541834230865054,
            1.0122073600614354,
            1.2224949866040544,
            1.2747052121767202,
            1.0411200080598673,
            0.9656258565724202,
            0.8627228889355896,
            1.5410757253368614,
        ],
        1e-13,
    )
    _close(
        s(_q()).to_host(),
        [
            -0.3040637567314879,
            0.20270917115432527,
            0.9255232146681803,
            1.2551263775869705,
            1.0033729323161438,
            1.1642130384472662,
            1.7671933374706186,
        ],
        1e-13,
    )
    _close(
        s(_q(), 1).to_host(),
        [
            1.0135458557716264,
            1.0135458557716264,
            0.8668414539325496,
            0.06526278196583218,
            -0.37747075743723485,
            0.7537253737791907,
            0.7537253737791907,
        ],
        1e-12,
    )
    assert_almost_equal(s.integrate(0.1, 4.7), 4.5052801335240575, atol=1e-13)


def test_degree_2_matches_scipy() raises:
    var s = make_interp_spline[k=2](_x(), _y())
    _close(
        s.t.to_host(),
        [0.0, 0.0, 0.0, 0.75, 1.3, 1.9, 2.65, 3.1, 3.65, 5.0, 5.0, 5.0],
        0.0,
    )
    _close(
        s.c.to_host(),
        [
            0.0,
            0.3941126585244478,
            0.9860270096679191,
            1.2921799344005311,
            1.3186473880924041,
            1.0894852212180945,
            0.8962278281964221,
            0.7127435969793745,
            1.5410757253368614,
        ],
        1e-13,
    )
    _close(
        s(_q()).to_host(),
        [
            -0.323709904385611,
            0.2064512945170154,
            0.9417725734151681,
            1.305056698154476,
            1.002519394358342,
            1.0590056306920288,
            1.9565666714300456,
        ],
        1e-13,
    )
    _close(
        s(_q(), 1).to_host(),
        [
            1.1070989398388793,
            1.0135458557716264,
            0.7387294659351694,
            -0.01694202246645403,
            -0.3865147860433449,
            0.7011216699015721,
            1.5427809319434698,
        ],
        1e-12,
    )
    assert_almost_equal(s.integrate(0.1, 4.7), 4.470242954933818, atol=1e-13)


def test_degree_3_matches_scipy() raises:
    var s = make_interp_spline[k=3](_x(), _y())
    _close(
        s.t.to_host(),
        [0.0, 0.0, 0.0, 0.0, 1.1, 1.5, 2.3, 3.0, 3.2, 5.0, 5.0, 5.0, 5.0],
        0.0,
    )
    _close(
        s.c.to_host(),
        [
            0.0,
            0.37100242898279395,
            0.9068153274762203,
            1.3159519199137242,
            1.3361202300444213,
            1.105235724098582,
            0.757339506093683,
            0.7122014736734823,
            1.5410757253368614,
        ],
        1e-13,
    )
    _close(
        s(_q()).to_host(),
        [
            -0.29527545441730535,
            0.20353830447554505,
            0.9410345599506472,
            1.3076707713898976,
            1.002459755485217,
            1.0366739363304676,
            2.0311994030733986,
        ],
        1e-13,
    )
    _close(
        s(_q(), 1).to_host(),
        [
            0.9454142926314887,
            1.0185520471051828,
            0.7450895302456636,
            -0.01262877268377898,
            -0.3797409064126202,
            0.666042154809304,
            1.8967952493647848,
        ],
        1e-12,
    )
    assert_almost_equal(s.integrate(0.1, 4.7), 4.464607063127108, atol=1e-13)


def test_degree_5_matches_scipy() raises:
    var s = make_interp_spline[k=5](_x(), _y())
    _close(
        s.t.to_host(),
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.5,
            2.3,
            3.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ],
        0.0,
    )
    _close(
        s.c.to_host(),
        [
            0.0,
            0.29946572841993546,
            0.7963417785100915,
            1.3301177873565764,
            1.4754719810745525,
            0.818738780117089,
            0.6377895065929161,
            1.032397521399574,
            1.5410757253368614,
        ],
        1e-13,
    )
    _close(
        s(_q()).to_host(),
        [
            -0.2845861557425651,
            0.20257033242950284,
            0.9414829437652042,
            1.309274951134915,
            1.0025824580774543,
            1.0482692849507766,
            1.9680873979075981,
        ],
        1e-13,
    )
    _close(
        s(_q(), 1).to_host(),
        [
            0.8819118248349265,
            1.0203831971923922,
            0.7401485009361385,
            -0.016101278328470428,
            -0.37913064719706857,
            0.6918076012692275,
            1.5624094564660298,
        ],
        1e-12,
    )
    assert_almost_equal(s.integrate(0.1, 4.7), 4.468405929636673, atol=1e-13)


def test_identities() raises:
    var s = make_interp_spline[k=3](_x(), _y())
    _close(s(_x()).to_host(), _y().to_host(), 1e-13)
    var round_trip = s.antiderivative().derivative()
    _close(round_trip(_q()).to_host(), s(_q()).to_host(), 1e-12)
    var direct = BSpline[f64](s.t, s.c, 3)
    _close(direct(_q()).to_host(), s(_q()).to_host(), 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
