"""Tests for `numax.core.libm`'s float64 `exp`, `log` and `erf`: within one
ulp of mpmath at points spanning each domain (including the ranges where
`std.math`'s versions are off by `1e5` to `1e8` ulp), the special values,
the odd symmetry of `erf`, and the same answers at SIMD width four as at
width one."""

from std.utils.numerics import inf, nan
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from numax import Plain
from numax.core.libm import erf, exp, log

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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
