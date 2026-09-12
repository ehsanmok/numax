"""Tests for the Airy functions and the Struve function against mpmath, on
every side of their blends and at the identities that tie them to the
Bessel functions they are built from. Held to `1e-12` relative; `pixi run
accuracy` has `Ai`/`Bi` at `4e-13`/`2e-12` and `struve` at `1e-15`.
"""

from std.math import cos as _cos, sqrt as _sqrt
from std.testing import TestSuite, assert_almost_equal

from numax import Dual, Plain
from numax.special import airy, struve

comptime dtype = DType.float64
comptime P = Plain[dtype, 1]
comptime D = Dual[P]
comptime PI = 3.141592653589793


def pv(x: Float64) -> P:
    return P.constant(x)


def s(x: P) -> Float64:
    return Float64(x.v)


def test_airy_matches_mpmath_in_all_three_regions() raises:
    var xs: List[Float64] = [
        -40.0,
        -10.0,
        -2.0,
        -1.5,
        -0.4,
        0.0,
        0.7,
        1.5,
        3.0,
        10.0,
        25.0,
    ]
    var ai: List[Float64] = [
        -0.04593392343795725,
        0.04024123848644319,
        0.22740742820168558,
        0.4642565777488694,
        0.4542256138886674,
        0.3550280538878172,
        0.18916240039815008,
        0.07174949700810541,
        0.006591139357460719,
        1.1047532552898686e-10,
        8.116026824691387e-38,
    ]
    var aip: List[Float64] = [
        -1.3890908752607183,
        0.99626504413279,
        0.618259020741691,
        0.3091869672024104,
        -0.22503140930241503,
        -0.2588194037928068,
        -0.19985119158228049,
        -0.09738201284230132,
        -0.011912976705951319,
        -3.5206336767389237e-10,
        -4.066089337243281e-37,
    ]
    var bi: List[Float64] = [
        0.2195886242840424,
        -0.3146798296438386,
        -0.4123025879563985,
        -0.19178486115704121,
        0.4300209399485034,
        0.6149266274460007,
        0.9733286558781659,
        1.878941503747895,
        14.037328963730232,
        455641153.54822516,
        3.9220307780413816e35,
    ]
    var bip: List[Float64] = [
        -0.28913994028209195,
        0.11941411339990923,
        0.2787951669211695,
        0.5579081030218973,
        0.48773486404914757,
        0.4482883573538264,
        0.65440591917214,
        1.8862122548481655,
        22.92221496638217,
        1429236134.4828658,
        1.957073508323331e36,
    ]
    for i in range(11):
        var got = airy(pv(xs[i]))
        assert_almost_equal(s(got[0]), ai[i], rtol=1e-12)
        assert_almost_equal(s(got[1]), aip[i], rtol=1e-12)
        assert_almost_equal(s(got[2]), bi[i], rtol=1e-12)
        assert_almost_equal(s(got[3]), bip[i], rtol=1e-12)


def test_airy_derivative_slots_agree_with_dual() raises:
    """`Ai'` returned in the tuple is `d/dx Ai` at `Dual`, on all three
    sides of the blend."""
    var xs: List[Float64] = [-4.0, -1.0, 0.5, 2.5]
    for i in range(4):
        var d = airy(D(pv(xs[i]), pv(1.0)))
        assert_almost_equal(
            Float64(d[0].deriv.v), Float64(d[1].value.v), rtol=1e-11
        )
        assert_almost_equal(
            Float64(d[2].deriv.v), Float64(d[3].value.v), rtol=1e-11
        )


def test_airy_wronskian_is_one_over_pi() raises:
    """`Ai Bi' - Ai' Bi = 1/pi` everywhere."""
    var xs: List[Float64] = [-30.0, -7.0, -1.0, 0.0, 1.0, 4.0, 20.0]
    for i in range(7):
        var a = airy(pv(xs[i]))
        var w = s(a[0]) * s(a[3]) - s(a[1]) * s(a[2])
        assert_almost_equal(w, 1.0 / PI, rtol=1e-12)


def test_struve_matches_mpmath_on_both_sides_of_forty() raises:
    var vs: List[Float64] = [0.0, 0.0, 1.0, 2.5, 0.3, 10.0, 0.0, 0.0, 5.0]
    var xs: List[Float64] = [
        1.0,
        5.0,
        25.0,
        50.0,
        100.0,
        0.5,
        39.9,
        40.1,
        200.0,
    ]
    var want: List[Float64] = [
        0.5686566270482879,
        -0.1852168157766849,
        0.5388036213269295,
        35.42893378014585,
        -0.04656715491301943,
        2.2526619047176753e-14,
        0.14067304106592923,
        0.14175143273425292,
        1078117.4081070055,
    ]
    for i in range(9):
        assert_almost_equal(
            s(struve(pv(vs[i]), pv(xs[i]))), want[i], rtol=1e-12
        )


def test_struve_half_order_is_elementary_and_differentiates() raises:
    """`H_{1/2}(x) = sqrt(2 / pi x) (1 - cos x)`, and `d/dx H_1(5)` from
    mpmath at `Dual`."""
    var xs: List[Float64] = [0.2, 3.0, 30.0, 75.0]
    for i in range(4):
        var x = xs[i]
        assert_almost_equal(
            s(struve(pv(0.5), pv(x))),
            _sqrt(2.0 / (PI * x)) * (1.0 - _cos(x)),
            rtol=1e-12,
        )
    var d = struve(D(pv(1.0), pv(0.0)), D(pv(5.0), pv(1.0)))
    assert_almost_equal(Float64(d.deriv.v), -0.3467792049354978, rtol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
