"""Tests for the arbitrary-order Bessel functions against mpmath, on both
sides of every blend (`x = 2`, the order-reduction steps, the sign of `v`)
and at the closed forms the half-integer orders reduce to. Held to
`1e-12` relative against mpmath; `pixi run accuracy` has the four
functions at `1e-15` to `4e-13` over `[0.01, 150]`.
"""

from std.math import cos as _cos, sin as _sin, sqrt as _sqrt

# numax's one-ulp `exp`, not `std.math`'s: at `x = 12` the latter is `4e-12`
# off, which is more than the functions under test.
from numax.core.libm import exp as _exp
from std.testing import TestSuite, assert_almost_equal

from numax import Dual, Plain
from numax.special import (
    iv,
    ive,
    j0,
    j1,
    jv,
    kv,
    kve,
    spherical_jn,
    spherical_yn,
    yv,
)

comptime dtype = DType.float64
comptime P = Plain[dtype, 1]
comptime D = Dual[P]
comptime PI = 3.141592653589793


def pv(x: Float64) -> P:
    return P.constant(x)


def s(x: P) -> Float64:
    return Float64(x.v)


def test_jv_matches_mpmath_across_orders_and_both_regions() raises:
    var vs: List[Float64] = [0.0, 0.0, 0.0, 1.0, 2.5, 0.3, 5.0, 29.5, 0.5, 3.0]
    var xs: List[Float64] = [
        0.5,
        10.0,
        150.0,
        3.0,
        7.5,
        0.01,
        2.0,
        12.0,
        200.0,
        199.5,
    ]
    var want: List[Float64] = [
        0.9384698072408129,
        -0.24593576445134835,
        -0.0007740903753942912,
        0.3390589585259365,
        -0.29910405245731303,
        0.22733294197947476,
        0.007039629755871685,
        5.616418901572255e-10,
        -0.04927052384285448,
        0.04115745596751396,
    ]
    for i in range(10):
        assert_almost_equal(
            s(jv(pv(vs[i]), pv(xs[i]))), want[i], rtol=1e-12, atol=1e-18
        )
    # Deep in the small-x, high-order corner the value is 1e-40.
    assert_almost_equal(
        s(jv(pv(10.0), pv(0.001))), 2.6911443943049994e-40, rtol=1e-12
    )


def test_yv_matches_mpmath_across_orders_and_both_regions() raises:
    var vs: List[Float64] = [0.0, 0.0, 0.0, 1.0, 2.5, 0.3, 5.0, 29.5, 0.5, 3.0]
    var xs: List[Float64] = [
        0.5,
        10.0,
        150.0,
        3.0,
        7.5,
        0.01,
        2.0,
        12.0,
        200.0,
        199.5,
    ]
    var want: List[Float64] = [
        -0.44451873350670656,
        0.055671167283599395,
        -0.06514222150903735,
        0.3246744247918,
        -0.013708391569564108,
        -4.501884927725057,
        -9.935989128481975,
        -21034002.66065345,
        -0.02748662114718023,
        -0.038697431333456954,
    ]
    for i in range(10):
        assert_almost_equal(s(yv(pv(vs[i]), pv(xs[i]))), want[i], rtol=1e-12)
    assert_almost_equal(
        s(yv(pv(10.0), pv(0.001))), -1.1828049377990414e38, rtol=1e-12
    )


def test_negative_orders_reflect_and_are_exact_at_integers() raises:
    assert_almost_equal(
        s(jv(pv(-0.5), pv(2.0))), -0.23478571040624846, rtol=1e-12
    )
    assert_almost_equal(
        s(yv(pv(-0.5), pv(2.0))), 0.5130161365618278, rtol=1e-12
    )
    assert_almost_equal(
        s(jv(pv(-1.0 / 3.0), pv(1.2))), 0.45158544421603014, rtol=1e-12
    )
    assert_almost_equal(
        s(yv(pv(-1.0 / 3.0), pv(1.2))), 0.5573537618891734, rtol=1e-12
    )
    # J_{-2} = J_2 and Y_{-2} = Y_2 exactly: the sin(2 pi) is a true zero.
    assert_almost_equal(
        s(jv(pv(-2.0), pv(3.0))), s(jv(pv(2.0), pv(3.0))), rtol=1e-15
    )
    assert_almost_equal(
        s(yv(pv(-2.0), pv(3.0))), s(yv(pv(2.0), pv(3.0))), rtol=1e-15
    )
    assert_almost_equal(
        s(jv(pv(-2.0), pv(3.0))), 0.4860912605858911, rtol=1e-12
    )
    # J_{-3}(0.01) = -J_3(0.01): a Y_3 of size 1e8 would swamp a
    # sin(3 pi) rounding error here.
    assert_almost_equal(
        s(jv(pv(-3.0), pv(0.01))), -s(jv(pv(3.0), pv(0.01))), rtol=1e-14
    )


def test_half_integer_orders_are_elementary() raises:
    """`J_{1/2}(x) = sqrt(2 / pi x) sin x`, `Y_{1/2} = -sqrt(2 / pi x) cos
    x`, `K_{1/2} = sqrt(pi / 2x) e^{-x}`, on both sides of `x = 2`."""
    var xs: List[Float64] = [0.3, 1.9, 2.1, 25.0, 120.0]
    for i in range(5):
        var x = xs[i]
        var amp = _sqrt(2.0 / (PI * x))
        assert_almost_equal(s(jv(pv(0.5), pv(x))), amp * _sin(x), atol=1e-13)
        assert_almost_equal(s(yv(pv(0.5), pv(x))), -amp * _cos(x), atol=1e-13)
        assert_almost_equal(
            s(kv(pv(0.5), pv(x))), _sqrt(PI / (2.0 * x)) * _exp(-x), rtol=1e-12
        )
        assert_almost_equal(
            s(kve(pv(0.5), pv(x))), _sqrt(PI / (2.0 * x)), rtol=1e-12
        )


def test_integer_orders_agree_with_the_polynomial_j0_j1_within_their_bound() raises:
    """`jv(0, x)` and `j0(x)` are two routes to one function; A&S's
    polynomials are good to `5e-8`, and that is the gap."""
    var xs: List[Float64] = [0.1, 1.0, 2.9, 3.1, 8.0, 20.0]
    for i in range(6):
        assert_almost_equal(
            s(jv(pv(0.0), pv(xs[i]))), s(j0(pv(xs[i]))), atol=2e-7
        )
        assert_almost_equal(
            s(jv(pv(1.0), pv(xs[i]))), s(j1(pv(xs[i]))), atol=2e-7
        )


def test_iv_kv_match_mpmath_and_the_scaled_forms_agree() raises:
    var vs: List[Float64] = [0.0, 0.0, 1.0, 2.5, 0.3, 5.0, 29.5]
    var xs: List[Float64] = [0.5, 10.0, 3.0, 7.5, 0.01, 2.0, 12.0]
    var want_i: List[Float64] = [
        1.0634833707413236,
        2815.7166284662544,
        3.9533702174026093,
        172.07689839990607,
        0.2273416857223144,
        0.009825679323131702,
        5.965051840095891e-09,
    ]
    var want_k: List[Float64] = [
        0.9244190712276659,
        1.778006231616765e-05,
        0.040156431128194184,
        0.000367862846522012,
        6.8901026382927695,
        9.431049100596468,
        2631832.691225936,
    ]
    for i in range(7):
        var v = pv(vs[i])
        var x = pv(xs[i])
        assert_almost_equal(s(iv(v, x)), want_i[i], rtol=1e-12)
        assert_almost_equal(s(kv(v, x)), want_k[i], rtol=1e-12)
        assert_almost_equal(s(ive(v, x)), want_i[i] * _exp(-xs[i]), rtol=1e-12)
        assert_almost_equal(s(kve(v, x)), want_k[i] * _exp(xs[i]), rtol=1e-12)
    # Where iv itself would be e^150: the scaled pair is what is returned.
    assert_almost_equal(
        s(ive(pv(2.5), pv(150.0))), 0.031926373911096574, rtol=1e-12
    )
    assert_almost_equal(
        s(kve(pv(2.5), pv(150.0))), 0.10439296856664777, rtol=1e-12
    )
    # Negative order: I_{-v} = I_v + (2/pi) sin(pi v) K_v; K is even.
    assert_almost_equal(s(iv(pv(-0.5), pv(2.0))), 2.122591620177637, rtol=1e-12)
    assert_almost_equal(
        s(iv(pv(-1.0 / 3.0), pv(1.2))), 1.4018033311387195, rtol=1e-12
    )
    assert_almost_equal(
        s(kv(pv(-0.5), pv(2.0))), s(kv(pv(0.5), pv(2.0))), rtol=1e-15
    )
    assert_almost_equal(s(iv(pv(0.0), pv(0.0))), 1.0, rtol=1e-14)


def test_spherical_bessel_functions() raises:
    assert_almost_equal(
        s(spherical_jn(2, pv(4.0))), 0.27628368577135015, rtol=1e-12
    )
    assert_almost_equal(
        s(spherical_jn(5, pv(0.5))), 2.9774668754574457e-06, rtol=1e-12
    )
    assert_almost_equal(
        s(spherical_yn(1, pv(2.0))), -0.35061200427605527, rtol=1e-12
    )
    assert_almost_equal(
        s(spherical_yn(3, pv(10.0))), -0.09532747887656891, rtol=1e-12
    )
    # j_0(x) = sin(x) / x, including the limit at zero.
    assert_almost_equal(
        s(spherical_jn(0, pv(1.7))), _sin(1.7) / 1.7, rtol=1e-14
    )
    assert_almost_equal(s(spherical_jn(0, pv(1e-9))), 1.0, rtol=1e-12)
    assert_almost_equal(s(spherical_jn(0, pv(0.0))), 1.0, rtol=1e-12)


def test_bessel_functions_differentiate() raises:
    """`d/dx J_v = (J_{v-1} - J_{v+1}) / 2`, `d/dx I_v = (I_{v-1} +
    I_{v+1}) / 2`, and the mpmath derivatives of `Y_1` and `K_{0.3}`, at
    `Dual`."""
    var j = jv(D(pv(2.5), pv(0.0)), D(pv(7.5), pv(1.0)))
    assert_almost_equal(Float64(j.deriv.v), 0.035148154689586764, rtol=1e-12)
    var i = iv(D(pv(2.5), pv(0.0)), D(pv(3.0), pv(1.0)))
    assert_almost_equal(Float64(i.deriv.v), 1.8367005844906648, rtol=1e-12)
    var y = yv(D(pv(1.0), pv(0.0)), D(pv(3.0), pv(1.0)))
    assert_almost_equal(Float64(y.deriv.v), 0.26862520174885707, rtol=1e-12)
    # Not at `x = 2` itself: that is the blend seam, where the clamps'
    # `|x - 2|` has no derivative and `Dual` reads the kink.
    var k = kv(D(pv(0.3), pv(0.0)), D(pv(2.3), pv(1.0)))
    assert_almost_equal(Float64(k.deriv.v), -0.09706608589578762, rtol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
