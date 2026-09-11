"""Tests for the exponential, sine, cosine and Fresnel integrals against
mpmath at 25 digits, on both sides of every blend threshold, plus the
symmetries and the derivatives the definitions imply.

The tolerances differ by function on purpose. `Si` and the Fresnel pair
never call `ln` and are held to `1e-13`; `exp1`, `expi`, `expn` and `Ci`
have a `ln x` term in their series region and inherit `std.math.log`'s
`2e-9` relative floor (`bench/accuracy/README.md`), so they are held to
`1e-8` relative -- which is where the algorithm's own error would be
visible if it were larger.
"""

from std.math import exp as _exp, sin as _sin
from std.testing import TestSuite, assert_almost_equal

from numax import Dual, Plain
from numax.special import exp1, expi, expn, fresnel, sici

comptime dtype = DType.float64
comptime P = Plain[dtype, 1]
comptime D = Dual[P]


def pv(x: Float64) -> P:
    return P.constant(x)


def s(x: P) -> Float64:
    return Float64(x.v)


def test_exp1_matches_mpmath_across_the_blend() raises:
    var xs: List[Float64] = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0]
    var want: List[Float64] = [
        1.8229239584193906,
        0.5597735947761608,
        0.21938393439552029,
        0.024914917870269736,
        0.0011482955912753257,
        4.156968929685325e-06,
        9.835525290649882e-11,
    ]
    for i in range(7):
        assert_almost_equal(s(exp1(pv(xs[i]))), want[i], rtol=1e-8, atol=1e-20)


def test_expi_matches_mpmath_on_both_sides_of_zero() raises:
    var xs: List[Float64] = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0]
    var positive: List[Float64] = [
        -1.6228128139692766,
        0.4542199048631736,
        1.8951178163559368,
        7.0737658945786,
        40.18527535580318,
        2492.2289762418777,
        25615652.664056588,
    ]
    var negative: List[Float64] = [
        -1.8229239584193906,
        -0.5597735947761608,
        -0.21938393439552029,
        -0.024914917870269736,
        -0.0011482955912753257,
        -4.156968929685325e-06,
        -9.835525290649882e-11,
    ]
    for i in range(7):
        assert_almost_equal(
            s(expi(pv(xs[i]))), positive[i], rtol=1e-8, atol=1e-9
        )
        assert_almost_equal(
            s(expi(pv(-xs[i]))), negative[i], rtol=1e-8, atol=1e-20
        )


def test_expi_asymptotic_region_and_derivative() raises:
    """Past the `x = 40` threshold the asymptotic sum takes over; and
    `d/dx Ei(x) = e^x / x` at `Dual`."""
    # mpmath: ei(45) = 7.943916035704454e17
    var far = s(expi(pv(45.0)))
    var reference = 7.943916035704454e17
    assert_almost_equal(far / reference, 1.0, atol=1e-11)
    var d = expi(D(pv(2.5), pv(1.0)))
    assert_almost_equal(Float64(d.deriv.v), _exp(2.5) / 2.5, rtol=1e-12)


def test_expn_matches_mpmath() raises:
    var xs: List[Float64] = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0]
    var n2: List[Float64] = [
        0.7225450221940205,
        0.326643862324553,
        0.14849550677592205,
        0.019797703948224457,
        0.000996469042708838,
        3.830240465631609e-06,
        9.404856430858148e-11,
    ]
    var n5: List[Float64] = [
        0.21901595224028045,
        0.13097731169586485,
        0.0704542374617204,
        0.011907379826344131,
        0.0007057606934245853,
        3.0897289142536863e-06,
        8.307130599417691e-11,
    ]
    for i in range(7):
        assert_almost_equal(s(expn(2, pv(xs[i]))), n2[i], rtol=1e-8, atol=1e-20)
        assert_almost_equal(s(expn(5, pv(xs[i]))), n5[i], rtol=1e-8, atol=1e-20)
    assert_almost_equal(s(expn(0, pv(2.0))), 0.06766764161830635, rtol=1e-13)
    assert_almost_equal(s(expn(1, pv(2.0))), 0.04890051070806112, rtol=1e-9)
    # `expn(1, x)` is `exp1(x)`: two routes to one function agree.
    assert_almost_equal(s(expn(1, pv(0.7))), s(exp1(pv(0.7))), rtol=1e-9)


def test_sici_matches_mpmath_and_is_odd_in_si() raises:
    var xs: List[Float64] = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 50.0]
    var si: List[Float64] = [
        0.09994446110827696,
        0.4931074180430667,
        0.946083070367183,
        1.7785201734438267,
        1.549931244944674,
        1.6583475942188741,
        1.54824170104344,
        1.551617072485936,
    ]
    var ci: List[Float64] = [
        -1.7278683866572966,
        -0.1777840788066129,
        0.33740392290096816,
        0.2858711963653835,
        -0.19002974965664388,
        -0.04545643300445537,
        0.044419820845353314,
        -0.005628386324116306,
    ]
    for i in range(8):
        var both = sici(pv(xs[i]))
        assert_almost_equal(s(both[0]), si[i], atol=1e-13)
        assert_almost_equal(s(both[1]), ci[i], rtol=1e-8, atol=1e-9)
    var negative = sici(pv(-2.5))
    assert_almost_equal(s(negative[0]), -1.7785201734438267, atol=1e-13)
    assert_almost_equal(s(negative[1]), 0.2858711963653835, rtol=1e-8)
    # d/dx Si(x) = sin(x) / x, on both sides of the threshold.
    var below = sici(D(pv(1.5), pv(1.0)))
    assert_almost_equal(Float64(below[0].deriv.v), _sin(1.5) / 1.5, atol=1e-12)
    var above = sici(D(pv(3.0), pv(1.0)))
    assert_almost_equal(Float64(above[0].deriv.v), _sin(3.0) / 3.0, atol=1e-12)


def test_fresnel_matches_mpmath_and_is_odd() raises:
    var xs: List[Float64] = [0.1, 0.5, 1.0, 1.5, 2.5, 4.0, 8.0]
    var fs: List[Float64] = [
        0.0005235895476122107,
        0.06473243285999927,
        0.43825914739035476,
        0.6975049600820931,
        0.6191817558195929,
        0.42051575424692844,
        0.46021421439301446,
    ]
    var fc: List[Float64] = [
        0.09999753262708508,
        0.4923442258714464,
        0.7798934003768229,
        0.4452611760398215,
        0.45741300964177706,
        0.4984260330381776,
        0.49980218037719715,
    ]
    for i in range(7):
        var both = fresnel(pv(xs[i]))
        assert_almost_equal(s(both[0]), fs[i], atol=1e-13)
        assert_almost_equal(s(both[1]), fc[i], atol=1e-13)
    var negative = fresnel(pv(-1.5))
    assert_almost_equal(s(negative[0]), -0.6975049600820931, atol=1e-13)
    assert_almost_equal(s(negative[1]), -0.4452611760398215, atol=1e-13)
    var at_zero = fresnel(pv(0.0))
    assert_almost_equal(s(at_zero[0]), 0.0, atol=1e-15)
    assert_almost_equal(s(at_zero[1]), 0.0, atol=1e-15)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
