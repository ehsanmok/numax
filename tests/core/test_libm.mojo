"""Tests for `numax.core.libm`'s float64 `exp`, `log`, `erf`, `log1p`,
`log2`, `exp2` and `cosh`: within one or two ulp of mpmath at points
spanning each domain (including the ranges where `std.math`'s versions
are off by `1e4` to `1e9` ulp), exactness at the powers of two where it
is claimed, the special values,
the odd symmetry of `erf`, and the same answers at SIMD width four as at
width one."""

from std.memory import bitcast
from std.utils.numerics import inf, nan
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from numax import Plain
from numax.core.libm import cosh, erf, exp, exp2, log, log1p, log2

comptime F = Float64
comptime ULP = 2.3e-16  # one ulp relative, generously: |x| * 2^-52


def within_ulps(got: Float64, want: Float64, ulps: Float64) raises:
    assert_almost_equal(got, want, rtol=ulps * ULP, atol=0.0)


def test_exp_is_within_one_ulp_across_the_range() raises:
    var xs: List[Float64] = [
        -700.0,
        -30.3,
        -29.4863,
        -1.0,
        -0.5,
        1e-10,
        0.5,
        1.0,
        29.7637,
        30.3,
        700.0,
        709.7,
    ]
    var want: List[Float64] = [
        9.85967654375977e-305,
        6.932297597586547e-14,
        1.5640931652491285e-13,
        0.36787944117144233,
        0.6065306597126334,
        1.0000000001,
        1.6487212707001282,
        2.718281828459045,
        8437439485393.79,
        14425231835807.887,
        1.0142320547350045e304,
        1.6549840276802644e308,
    ]
    for i in range(12):
        within_ulps(exp(F(xs[i])), want[i], 1.5)
    assert_equal(exp(F(0.0)), 1.0)
    # Gradual underflow: 5e-324 is the smallest denormal.
    assert_equal(exp(F(-745.0)), 5e-324)
    assert_equal(exp(F(-746.0)), 0.0)
    assert_equal(exp(F(710.0)), inf[DType.float64]())
    assert_equal(exp(inf[DType.float64]()), inf[DType.float64]())
    assert_equal(exp(-inf[DType.float64]()), 0.0)
    var n = exp(nan[DType.float64]())
    assert_true(n != n)


def test_log_is_within_one_ulp_down_to_the_denormals() raises:
    var xs: List[Float64] = [
        1e-310,
        1e-300,
        0.1,
        0.19315210964763310,
        0.5,
        0.7824,
        0.9999999,
        1.0000001,
        1.313,
        2.0,
        10.0,
        1e300,
    ]
    var want: List[Float64] = [
        -713.8013788281542,
        -690.7755278982137,
        -2.3025850929940455,
        -1.644277267601599,
        -0.6931471805599453,
        -0.2453891602615295,
        -1.0000000494736474e-07,
        9.999999505838704e-08,
        0.2723145953206591,
        0.6931471805599453,
        2.302585092994046,
        690.7755278982137,
    ]
    for i in range(12):
        within_ulps(log(F(xs[i])), want[i], 1.5)
    assert_equal(log(F(1.0)), 0.0)
    assert_equal(log(F(0.0)), -inf[DType.float64]())
    assert_equal(log(inf[DType.float64]()), inf[DType.float64]())
    var n = log(F(-1.0))
    assert_true(n != n)
    var m = log(nan[DType.float64]())
    assert_true(m != m)


def test_erf_is_within_one_ulp_and_odd() raises:
    var xs: List[Float64] = [
        1e-10,
        0.1,
        -0.3404410015858561,
        0.5,
        0.6363,
        0.84,
        0.85,
        1.964,
        3.0,
    ]
    var want: List[Float64] = [
        1.1283791670955126e-10,
        0.1124629160182849,
        -0.369807755747788,
        0.5204998778130465,
        0.6318074174509493,
        0.7651427114549945,
        0.7706680576083526,
        0.9945223761849636,
        0.9999779095030014,
    ]
    for i in range(9):
        within_ulps(erf(F(xs[i])), want[i], 2.0)
        assert_equal(erf(F(-xs[i])), -erf(F(xs[i])))
    assert_equal(erf(F(0.0)), 0.0)
    assert_equal(erf(F(-6.0)), -1.0)
    assert_equal(erf(F(30.0)), 1.0)
    assert_equal(erf(inf[DType.float64]()), 1.0)
    assert_equal(erf(-inf[DType.float64]()), -1.0)
    var n = erf(nan[DType.float64]())
    assert_true(n != n)


def test_width_four_matches_width_one_and_plain_uses_libm() raises:
    var v = SIMD[DType.float64, 4](-30.3, 0.5, 1.313, 29.7637)
    var e = exp(v)
    var l = log(SIMD[DType.float64, 4](0.7824, 1.313, 10.0, 1e-310))
    var r = erf(SIMD[DType.float64, 4](0.1, 0.85, -0.3404410015858561, 3.0))
    for i in range(4):
        assert_equal(e[i], exp(v[i]))
    assert_equal(l[0], log(F(0.7824)))
    assert_equal(l[3], log(F(1e-310)))
    assert_equal(r[1], erf(F(0.85)))
    assert_equal(r[2], erf(F(-0.3404410015858561)))
    # `Plain` routes through the same functions.
    comptime P = Plain[DType.float64, 1]
    assert_equal(Float64(P.constant(29.7637).exp().v), exp(F(29.7637)))
    assert_equal(Float64(P.constant(0.7824).ln().v), log(F(0.7824)))
    assert_equal(Float64(P.constant(0.6363).erf().v), erf(F(0.6363)))
    # And float32 still goes to the standard library, which is fine there.
    comptime P32 = Plain[DType.float32, 1]
    assert_almost_equal(
        Float64(P32.constant(1.0).exp().v), 2.718281828459045, rtol=2e-7
    )


def test_log1p_is_within_two_ulps_including_near_zero() raises:
    # mpmath at 200 bits. `std.math.log1p` is 5e-7 relative off at float64.
    var xs: List[Float64] = [
        -0.9999999,
        -0.5,
        -1e-300,
        1e-18,
        1e-10,
        0.3678794411714423,
        0.7,
        2.5,
        100000.0,
        1e300,
    ]
    var want: List[Float64] = [
        -16.118095651484676,
        -0.6931471805599453,
        -1e-300,
        1e-18,
        9.999999999500001e-11,
        0.3132616875182228,
        0.5306282510621704,
        1.252762968495368,
        11.51293546492023,
        690.7755278982137,
    ]
    for i in range(len(xs)):
        within_ulps(log1p(xs[i]), want[i], 2.0)
    assert_equal(log1p(F(-1.0)), -inf[DType.float64]())
    assert_equal(log1p(inf[DType.float64]()), inf[DType.float64]())
    var below = log1p(F(-2.0))
    assert_true(below != below)


def test_log2_is_within_one_ulp_and_exact_at_powers_of_two() raises:
    # mpmath at 200 bits. `std.math.log2` is 2.5e-9 relative off at float64.
    var xs: List[Float64] = [
        5e-324,
        1e-310,
        1e-05,
        0.3678794411714423,
        0.7,
        1.0000001,
        3.0,
        10000000000.0,
        1.7e308,
    ]
    var want: List[Float64] = [
        -1074.0,
        -1029.7977094150824,
        -16.609640474436812,
        -1.4426950408889636,
        -0.5145731728297583,
        1.4426949695965583e-07,
        1.584962500721156,
        33.219280948873624,
        1023.9193879716706,
    ]
    for i in range(len(xs)):
        within_ulps(log2(xs[i]), want[i], 1.0)
    for k in range(-1022, 1024, 7):
        var power = bitcast[DType.float64](Int64(k + 1023) << 52)
        assert_equal(log2(power), F(k))
    assert_equal(log2(F(0.0)), -inf[DType.float64]())
    assert_equal(log2(inf[DType.float64]()), inf[DType.float64]())


def test_exp2_is_within_one_ulp_and_exact_at_the_integers() raises:
    # mpmath at 200 bits. `std.math.exp2` is 4.8e-12 relative off at float64.
    var xs: List[Float64] = [
        -30.7,
        -0.4,
        0.3678794411714423,
        0.7,
        17.25,
        1023.5,
    ]
    var want: List[Float64] = [
        5.732962923799255e-10,
        0.757858283255199,
        1.2904546490875854,
        1.624504792712471,
        155871.75497763665,
        1.2711610061536464e308,
    ]
    for i in range(len(xs)):
        within_ulps(exp2(xs[i]), want[i], 1.0)
    for k in range(-1022, 1024, 7):
        var power = bitcast[DType.float64](Int64(k + 1023) << 52)
        assert_equal(exp2(F(k)), power)
    # Into the denormals, the answer has fewer bits; hold it absolutely.
    assert_equal(exp2(F(-1074.0)), 5e-324)
    assert_almost_equal(exp2(F(-1022.5)), 1.5733648139913585e-308, atol=5e-324)
    assert_equal(exp2(F(1024.0)), inf[DType.float64]())
    assert_equal(exp2(F(-1076.0)), 0.0)


def test_cosh_is_within_two_ulps_to_the_overflow() raises:
    # mpmath at 200 bits. `std.math.cosh` is 7e-12 relative off at float64.
    var xs: List[Float64] = [
        0.0,
        1e-09,
        0.3678794411714423,
        -2.5,
        17.25,
        300.0,
        -709.5,
        710.4,
    ]
    var want: List[Float64] = [
        1.0,
        1.0,
        1.0684342442825563,
        6.132289479663686,
        15507786.63724113,
        9.712131976206279e129,
        6.774931596573164e307,
        1.6663642832806496e308,
    ]
    for i in range(len(xs)):
        within_ulps(cosh(xs[i]), want[i], 2.0)
    assert_equal(cosh(F(711.0)), inf[DType.float64]())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
