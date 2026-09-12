"""`exp`, `log` and `erf` at float64 to within one unit in the last place,
for `Plain` to stand on.

**This module is tier 1**: straight-line SIMD arithmetic and bit
manipulation, no data-dependent loops, launchable inside a GPU thread.

`std.math`'s float64 `exp`, `log` and `erf` are not accurate to double
precision at the pinned release, and measured against mpmath on a grid
they come out at

| function | worst ulp | worst relative | why |
| --- | --- | --- | --- |
| `exp`, [-30, 30] | 105,000 | 1.2e-11 | reduces with `ln 2` rounded to float32, so the error grows with `|x|` |
| `log`, [1e-6, 1e3] | 9,300,000 | 1.6e-9 | an absolute floor of `2e-10`, which is `8e-10` relative on `ln 0.8` |
| `erf`, [-6, 6] | 196,000,000 | 2.3e-8 | float32 polynomial coefficients |
| `erfc`, `sin`, `cos` | 3 | 5e-16 | fine, and used as they are |

Every tier-1 kernel in numax is built on these three, and every row in
`bench/accuracy/README.md` above `1e-13` was one of them seen through an
algorithm that was itself at `1e-14`. The three functions here are Sun's
fdlibm algorithms (`e_exp.c`, `e_log.c`, `s_erf.c`), transcribed to SIMD
with masks in place of branches, and checked in Python against mpmath
at 30,000 random points each before transcription: under one ulp for
`log` and `erf`, under one ulp for `exp` across `[-745, 709.78]`. At any
other `dtype` they hand straight to `std.math`, whose float32 versions
are adequate for float32.

- `exp`: `k = round(x / ln 2)`, `r = x - k ln2_hi - k ln2_lo` with the
  split constant so `k ln2_hi` is exact, `e^r` by fdlibm's degree-five
  rational correction, and `2^k` applied as two half-powers built from
  exponent bits so `k` down to `-1074` (the gradual underflow range) and
  up to `1024` never form an invalid exponent.
- `log`: `x = 2^k m` with `m` in `[sqrt2/2, sqrt2)` read off the bits
  (denormals scaled up by `2^54` first), `f = m - 1`, `s = f / (2 + f)`,
  and fdlibm's seven-term odd series in `s` with the `k ln 2` split the
  same way.
- `erf`: fdlibm's rational approximation `x + x P(x^2) / Q(x^2)` below
  `|x| = 0.84375`, and `1 - erfc(|x|)` with the sign restored above it,
  where `erfc` is small enough that the standard library's three ulp on
  it are below one ulp of the result.
"""

from std.math import erf as _std_erf
from std.math import erfc as _std_erfc
from std.math import exp as _std_exp
from std.math import floor, isnan
from std.math import log as _std_log
from std.memory import bitcast
from std.utils.numerics import inf, nan

comptime _EXP_OVERFLOW = 709.782712893384
comptime _EXP_UNDERFLOW = -745.1332191019412
comptime _INV_LN2 = 1.44269504088896338700e00
comptime _LN2_HI = 6.93147180369123816490e-01
comptime _LN2_LO = 1.90821492927058770002e-10
comptime _P1 = 1.66666666666666019037e-01
comptime _P2 = -2.77777777770155933842e-03
comptime _P3 = 6.61375632143793436117e-05
comptime _P4 = -1.65339022054652515390e-06
comptime _P5 = 4.13813679705723846039e-08

comptime _LG1 = 6.666666666666735130e-01
comptime _LG2 = 3.999999999940941908e-01
comptime _LG3 = 2.857142874366239149e-01
comptime _LG4 = 2.222219843214978396e-01
comptime _LG5 = 1.818357216161805012e-01
comptime _LG6 = 1.531383769920937332e-01
comptime _LG7 = 1.479819860511658591e-01
comptime _SQRT2 = 1.4142135623730951
comptime _MIN_NORMAL = 2.2250738585072014e-308
comptime _TWO54 = 18014398509481984.0

comptime _PP0 = 1.28379167095512558561e-01
comptime _PP1 = -3.25042107247001499370e-01
comptime _PP2 = -2.84817495755985104766e-02
comptime _PP3 = -5.77027029648944159157e-03
comptime _PP4 = -2.37630166566501626084e-05
comptime _QQ1 = 3.97917223959155352819e-01
comptime _QQ2 = 6.50222499887672944485e-02
comptime _QQ3 = 5.08130628187576562776e-03
comptime _QQ4 = 1.32494738004321644526e-04
comptime _QQ5 = -3.96022827877536812320e-06
comptime _ERF_SPLIT = 0.84375


def exp[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """`e^x`, correctly rounded to within one ulp at float64; `std.math.exp`
    at every other dtype."""
    comptime if dtype != DType.float64:
        return _std_exp(x)
    else:
        var is_nan = isnan(x)
        var over = x.gt(_EXP_OVERFLOW)
        var under = x.lt(_EXP_UNDERFLOW)
        var xs = is_nan.select(SIMD[dtype, width](0.0), x).clamp(
            _EXP_UNDERFLOW, _EXP_OVERFLOW
        )
        var k = floor(xs * _INV_LN2 + 0.5)
        var hi = xs - k * _LN2_HI
        var lo = k * _LN2_LO
        var r = hi - lo
        var t = r * r
        var c = r - t * (_P1 + t * (_P2 + t * (_P3 + t * (_P4 + t * _P5))))
        var y = 1.0 - ((lo - (r * c) / (2.0 - c)) - hi)
        # `2^k` as two half-powers built from exponent bits: `k` runs from
        # `-1074` (the last denormal) to `1024`, and neither half leaves the
        # exponent field's `[1, 2046]`.
        var k_int = k.cast[DType.int64]()
        var a = k_int >> 1
        var two_a = bitcast[DType.float64, width]((a + 1023) << 52).cast[
            dtype
        ]()
        var two_b = bitcast[DType.float64, width](
            (k_int - a + 1023) << 52
        ).cast[dtype]()
        var result = (y * two_a) * two_b
        result = over.select(SIMD[dtype, width](inf[dtype]()), result)
        result = under.select(SIMD[dtype, width](0.0), result)
        return is_nan.select(x, result)


def log[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """`ln x`, correctly rounded to within one ulp at float64 for every
    positive finite `x` including the denormals, with `ln 0 = -inf`, `ln
    inf = inf` and NaN for `x < 0`; `std.math.log` at every other dtype."""
    comptime if dtype != DType.float64:
        return _std_log(x)
    else:
        var regular = x.gt(0.0) & x.lt(inf[dtype]())
        var safe = regular.select(x, SIMD[dtype, width](1.0))
        var invalid = x.lt(0.0) | isnan(x)
        var denormal = safe.lt(_MIN_NORMAL)
        var scaled = denormal.select(safe * _TWO54, safe)
        var bits = bitcast[DType.int64, width](scaled)
        var exponent = ((bits >> 52) & 0x7FF) - 1023
        var mantissa = bitcast[DType.float64, width](
            (bits & 0x000FFFFFFFFFFFFF) | (SIMD[DType.int64, width](1023) << 52)
        ).cast[dtype]()
        var high = mantissa.gt(_SQRT2)
        var m = high.select(mantissa * 0.5, mantissa)
        var k_int = (
            exponent
            + high.select(
                SIMD[DType.int64, width](1), SIMD[DType.int64, width](0)
            )
            + denormal.select(
                SIMD[DType.int64, width](-54), SIMD[DType.int64, width](0)
            )
        )
        var k = k_int.cast[dtype]()
        var f = m - 1.0
        var s = f / (2.0 + f)
        var z = s * s
        var w = z * z
        var t1 = w * (_LG2 + w * (_LG4 + w * _LG6))
        var t2 = z * (_LG1 + w * (_LG3 + w * (_LG5 + w * _LG7)))
        var big_r = t1 + t2
        var hfsq = 0.5 * f * f
        var result = k * _LN2_HI - (
            (hfsq - (s * (hfsq + big_r) + k * _LN2_LO)) - f
        )
        # IEEE specials, spelled out rather than borrowed: `std.math.log`
        # returns `709.78` (`ln` of the largest finite) for infinity.
        result = x.eq(inf[dtype]()).select(
            SIMD[dtype, width](inf[dtype]()), result
        )
        result = x.eq(0.0).select(SIMD[dtype, width](-inf[dtype]()), result)
        return invalid.select(SIMD[dtype, width](nan[dtype]()), result)


def erf[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """`erf x`, to within one ulp at float64; `std.math.erf` at every other
    dtype."""
    comptime if dtype != DType.float64:
        return _std_erf(x)
    else:
        var ax = abs(x)
        var small = ax.lt(_ERF_SPLIT)
        var z = x * x
        var r = _PP0 + z * (_PP1 + z * (_PP2 + z * (_PP3 + z * _PP4)))
        var s = 1.0 + z * (
            _QQ1 + z * (_QQ2 + z * (_QQ3 + z * (_QQ4 + z * _QQ5)))
        )
        var near = x + x * (r / s)
        var far = 1.0 - _std_erfc(ax)
        var far_signed = x.lt(0.0).select(-far, far)
        return small.select(near, far_signed)
