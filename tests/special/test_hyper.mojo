"""Tests for `zeta` and the hypergeometric functions against mpmath, on
both sides of every blend and at the closed-form values the formulas
must reproduce."""

from std.testing import TestSuite, assert_almost_equal

from numax import Dual, Plain
from numax.special import hyp1f1, hyp2f1, zeta

comptime dtype = DType.float64
comptime P = Plain[dtype, 1]
comptime D = Dual[P]


def pv(x: Float64) -> P:
    return P.constant(x)


def s(x: P) -> Float64:
    return Float64(x.v)


def test_riemann_zeta_matches_mpmath_on_both_sides_of_the_pole() raises:
    var xs: List[Float64] = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0, 30.0]
    var want: List[Float64] = [
        2.612375348685488,
        1.6449340668482264,
        1.341487257250917,
        1.2020569031595942,
        1.0823232337111381,
        1.0173430619844492,
        1.000994575127818,
        1.0000000009313275,
    ]
    for i in range(8):
        assert_almost_equal(s(zeta(pv(xs[i]))), want[i], rtol=1e-12)
    # Below the pole: the same formula, no reflection.
    assert_almost_equal(s(zeta(pv(1.1))), 10.584448464950801, rtol=1e-12)
    assert_almost_equal(s(zeta(pv(0.5))), -1.4603545088095868, rtol=1e-10)
    assert_almost_equal(s(zeta(pv(0.0))), -0.5, atol=1e-9)
    assert_almost_equal(s(zeta(pv(-0.5))), -0.20788622497735457, rtol=1e-10)
    assert_almost_equal(s(zeta(pv(-1.5))), -0.025485201889833036, rtol=1e-9)
    assert_almost_equal(s(zeta(pv(-2.5))), 0.008516928777850331, rtol=1e-9)
    # zeta(-1) = -1/12, zeta(2) = pi^2 / 6.
    assert_almost_equal(s(zeta(pv(-1.0))), -1.0 / 12.0, atol=1e-9)
    assert_almost_equal(s(zeta(pv(2.0))), 1.6449340668482264, rtol=1e-9)


def test_hurwitz_zeta_matches_mpmath_and_reduces_to_riemann() raises:
    assert_almost_equal(
        s(zeta(pv(2.0), pv(3.0))), 0.39493406684822646, rtol=1e-12
    )
    assert_almost_equal(
        s(zeta(pv(3.5), pv(0.5))), 11.620804663441895, rtol=1e-12
    )
    assert_almost_equal(s(zeta(pv(2.5), pv(1.0))), s(zeta(pv(2.5))), rtol=1e-13)


def test_hyp1f1_matches_mpmath_on_both_sides_of_zero() raises:
    """Direct series for `x > 0`, Kummer's transformation for `x < 0`
    where the plain series would cancel to noise."""
    assert_almost_equal(
        s(hyp1f1(pv(1.0), pv(2.0), pv(0.5))), 1.2974425414002564, rtol=1e-13
    )
    assert_almost_equal(
        s(hyp1f1(pv(2.5), pv(3.5), pv(5.0))), 54.52042935197804, rtol=1e-12
    )
    assert_almost_equal(
        s(hyp1f1(pv(0.3), pv(0.7), pv(20.0))), 64463323.49761426, rtol=1e-11
    )
    assert_almost_equal(
        s(hyp1f1(pv(2.0), pv(1.0), pv(10.0))), 242291.12374287387, rtol=1e-11
    )
    assert_almost_equal(
        s(hyp1f1(pv(0.5), pv(1.5), pv(-2.0))), 0.5981440066613041, rtol=1e-12
    )
    assert_almost_equal(
        s(hyp1f1(pv(1.0), pv(3.0), pv(-15.0))), 0.12444444716357618, rtol=1e-11
    )
    # 1F1(a; a; x) = e^x, and 1F1(1; 2; x) = (e^x - 1) / x.
    assert_almost_equal(
        s(hyp1f1(pv(1.7), pv(1.7), pv(3.0))), 20.085536923187668, rtol=1e-12
    )
    assert_almost_equal(
        s(hyp1f1(pv(1.0), pv(2.0), pv(2.0))),
        (7.38905609893065 - 1.0) / 2.0,
        rtol=1e-12,
    )


def test_hyp2f1_matches_mpmath_on_both_sides_of_zero() raises:
    """Direct series for `0 <= x <= 0.9`, Pfaff's transformation for
    `x < 0`, and the closed forms `2F1(1, 1; 2; x) = -ln(1 - x) / x` and
    `2F1(a, b; b; x) = (1 - x)^{-a}`."""
    assert_almost_equal(
        s(hyp2f1(pv(1.0), pv(1.0), pv(2.0), pv(0.5))),
        1.3862943611198906,
        rtol=1e-13,
    )
    assert_almost_equal(
        s(hyp2f1(pv(2.0), pv(3.0), pv(4.0), pv(0.9))),
        21.78942310292967,
        rtol=1e-12,
    )
    assert_almost_equal(
        s(hyp2f1(pv(0.3), pv(1.7), pv(2.2), pv(0.25))),
        1.0675548059414097,
        rtol=1e-13,
    )
    assert_almost_equal(
        s(hyp2f1(pv(0.5), pv(0.5), pv(1.5), pv(-0.7))),
        0.9096164028350779,
        rtol=1e-12,
    )
    assert_almost_equal(
        s(hyp2f1(pv(1.0), pv(2.0), pv(3.0), pv(-0.95))),
        0.6253088696384367,
        rtol=1e-12,
    )
    assert_almost_equal(
        s(hyp2f1(pv(1.0), pv(1.0), pv(2.0), pv(-0.5))),
        0.8109302162163288,
        rtol=1e-12,
    )
    assert_almost_equal(
        s(hyp2f1(pv(1.5), pv(2.0), pv(2.0), pv(0.4))),
        2.1516574145596756,
        rtol=1e-12,
    )
    # A far negative argument through Pfaff: 2F1(1, 1; 2; -9) = ln(10) / 9.
    assert_almost_equal(
        s(hyp2f1(pv(1.0), pv(1.0), pv(2.0), pv(-9.0))),
        2.302585092994046 / 9.0,
        rtol=1e-8,
    )


def test_hypergeometric_functions_differentiate() raises:
    """`d/dx 1F1(a; b; x) = (a/b) 1F1(a+1; b+1; x)` and `d/dx 2F1(a, b; c;
    x) = (ab/c) 2F1(a+1, b+1; c+1; x)` at `Dual`."""
    var m = hyp1f1(
        D(pv(1.5), pv(0.0)), D(pv(2.5), pv(0.0)), D(pv(1.2), pv(1.0))
    )
    var m_expected = 1.5 / 2.5 * s(hyp1f1(pv(2.5), pv(3.5), pv(1.2)))
    assert_almost_equal(Float64(m.deriv.v), m_expected, rtol=1e-12)
    var g = hyp2f1(
        D(pv(0.5), pv(0.0)),
        D(pv(1.5), pv(0.0)),
        D(pv(2.5), pv(0.0)),
        D(pv(0.3), pv(1.0)),
    )
    var g_expected = (
        0.5 * 1.5 / 2.5 * s(hyp2f1(pv(1.5), pv(2.5), pv(3.5), pv(0.3)))
    )
    assert_almost_equal(Float64(g.deriv.v), g_expected, rtol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
