"""Tests for Owen's T function against mpmath quadrature, on both sides of
the `a = 1` reduction and the `h = 2` quadrature switch, plus its
symmetries and the closed forms at `a = 1` and `h = 0`."""

from std.math import atan as _atan, erfc as _erfc
from std.testing import TestSuite, assert_almost_equal

from numax import Dual, Plain
from numax.special import owens_t

comptime dtype = DType.float64
comptime P = Plain[dtype, 1]
comptime D = Dual[P]
comptime PI = 3.141592653589793


def pv(x: Float64) -> P:
    return P.constant(x)


def s(x: P) -> Float64:
    return Float64(x.v)


def phi(h: Float64) -> Float64:
    return 0.5 * _erfc(-h / 1.4142135623730951)


def test_owens_t_matches_mpmath_quadrature() raises:
    var hs: List[Float64] = [0.5, 2.0, 1.0, 3.0, 5.0, 0.1, 7.0, 2.5, 1.99, 2.01]
    var as_: List[Float64] = [
        0.3,
        0.8,
        3.0,
        20.0,
        1.5,
        0.99,
        0.5,
        100.0,
        1.0,
        1.0,
    ]
    var want: List[Float64] = [
        0.040786707344250106,
        0.010631958144605746,
        0.07929950474887258,
        0.0006749490158150473,
        1.433257859395941e-07,
        0.12341502312505476,
        6.396704462156891e-13,
        0.0031048326628880674,
        0.011376394466255331,
        0.010861030896789251,
    ]
    for i in range(10):
        assert_almost_equal(
            s(owens_t(pv(hs[i]), pv(as_[i]))), want[i], rtol=1e-13
        )


def test_owens_t_symmetries_and_closed_forms() raises:
    # Even in h, odd in a.
    assert_almost_equal(
        s(owens_t(pv(-1.0), pv(0.5))), s(owens_t(pv(1.0), pv(0.5))), rtol=1e-15
    )
    assert_almost_equal(
        s(owens_t(pv(0.5), pv(-0.3))), -s(owens_t(pv(0.5), pv(0.3))), rtol=1e-15
    )
    assert_almost_equal(s(owens_t(pv(1.3), pv(0.0))), 0.0, atol=1e-18)
    # T(0, a) = arctan(a) / 2 pi; T(h, 1) = Phi(h) (1 - Phi(h)) / 2.
    assert_almost_equal(s(owens_t(pv(0.0), pv(1.0))), 0.125, rtol=1e-14)
    assert_almost_equal(
        s(owens_t(pv(0.0), pv(4.0))), _atan(4.0) / (2.0 * PI), rtol=1e-14
    )
    var hs: List[Float64] = [0.3, 1.5, 2.5, 4.0]
    for i in range(4):
        var p = phi(hs[i])
        assert_almost_equal(
            s(owens_t(pv(hs[i]), pv(1.0))), 0.5 * p * (1.0 - p), rtol=1e-13
        )


def test_owens_t_differentiates_in_h() raises:
    var d = owens_t(D(pv(1.2), pv(1.0)), D(pv(0.7), pv(0.0)))
    assert_almost_equal(Float64(d.deriv.v), -0.0581676184975172, rtol=1e-11)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
