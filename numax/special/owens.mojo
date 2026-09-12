"""Owen's T function `T(h, a) = (1/2pi) int_0^a exp(-h^2 (1+x^2) / 2) /
(1+x^2) dx`. `scipy.special.owens_t(h, a)`.

**This module is tier 1.** SciPy's Patefield-Tandy implementation picks one
of six methods per point; here two fixed 64-node Gauss-Legendre rules do
the work, blended, after the argument is reduced to `h >= 0` (`T` is even
in `h`), `a >= 0` (odd in `a`) and `a <= 1`:

- `a > 1`: `T(h, a) = (Phi(h) Q(ah) + Phi(ah) Q(h)) / 2 - T(ah, 1/a)`, with
  `Phi` the normal CDF and `Q = 1 - Phi` from `erfc` directly, so the
  `1/2 Phi(h) + 1/2 Phi(ah) - Phi(h) Phi(ah)` of the textbook form never
  cancels: at `h = 7` that form loses every digit of a `6e-13` answer.
- `h < 2`: the integral as written over `[0, a]`; the integrand is a
  Gaussian of width `1/h >= 1/2` times a rational function, and the
  degree-127 rule has it to `1e-16` absolute.
- `h >= 2`: substituting `u = h x`, `T = (h / 2pi) e^{-h^2/2} int_0^{ha}
  e^{-u^2/2} / (h^2 + u^2) du`, the upper limit clamped at `8.5` where
  `e^{-u^2/2}` is below `2e-16`. The `e^{-h^2/2}` outside the integral is
  what keeps the result relatively accurate as `T` falls to `1e-16` at `h =
  8`, where the first form would return the rounding noise of a sum of
  `O(1)` weights.

Checked against mpmath quadrature at `1e-15` relative for `h <= 8`, every
`a`; below `T ~ 1e-30` (`h > 11`) the exponential's own floor shows.

## The MAX gate

Nothing: MAX has no Owen's T. **Extend.**
"""

from std.collections import Array

from ..core.numeric import FloatLike, blend, ge_indicator, max_of, min_of

comptime _TWO_PI = 6.283185307179586
comptime _INV_SQRT2 = 0.7071067811865476
comptime _NODES = 64

# Gauss-Legendre nodes and weights on `[0, 1]`, 64 points, mpmath at 30
# digits.
comptime _GL_X: Array[Float64, 64] = [
    0.000347479132113930272,
    0.00182994161402236033,
    0.00449331426162783963,
    0.00833187305768702153,
    0.0133365861050445181,
    0.0194956001739731405,
    0.026794312570798592,
    0.0352154139340302121,
    0.0447389314607485971,
    0.0553422770024429471,
    0.0670003009229535901,
    0.0796853518737098186,
    0.0933673424386012201,
    0.108013820528329296,
    0.123590046369734052,
    0.140059074914194587,
    0.157381843472883379,
    0.17551726437267133,
    0.194422322413803375,
    0.214052176898682983,
    0.234360267990052727,
    0.255298427146473521,
    0.276816991373267956,
    0.298864921018004198,
    0.321389920831165942,
    0.344338564004894522,
    0.367656418895616292,
    0.391288178129996458,
    0.415177789788003591,
    0.439268590351939723,
    0.46350343910610048,
    0.487824853668287784,
    0.512175146331712216,
    0.53649656089389952,
    0.560731409648060277,
    0.584822210211996409,
    0.608711821870003542,
    0.632343581104383708,
    0.655661435995105478,
    0.678610079168834058,
    0.701135078981995802,
    0.723183008626732044,
    0.744701572853526479,
    0.765639732009947273,
    0.785947823101317017,
    0.805577677586196625,
    0.82448273562732867,
    0.842618156527116621,
    0.859940925085805413,
    0.876409953630265948,
    0.891986179471670704,
    0.90663265756139878,
    0.920314648126290181,
    0.93299969907704641,
    0.944657722997557053,
    0.955261068539251403,
    0.964784586065969788,
    0.973205687429201408,
    0.980504399826026859,
    0.986663413894955482,
    0.991668126942312978,
    0.99550668573837216,
    0.99817005838597764,
    0.99965252086788607,
]

comptime _GL_W: Array[Float64, 64] = [
    0.000891640360848216474,
    0.00207351663028123382,
    0.00325222898448918143,
    0.00442337991318197386,
    0.00558406973006556441,
    0.0067315239483593213,
    0.00786301523801235966,
    0.00897585788784867154,
    0.0100674115767651047,
    0.0111350869041916271,
    0.0121763512843554367,
    0.0131887348575273293,
    0.0141698363071297416,
    0.0151173285362012394,
    0.0160289641774257768,
    0.0169025809185708047,
    0.0177361066284411919,
    0.018527564270120023,
    0.0192750765893078146,
    0.0199768705663601707,
    0.0206312816213117643,
    0.0212367575618267945,
    0.0217918622646617267,
    0.0222952790818782815,
    0.0227458139637090722,
    0.0231423982906572086,
    0.0234840914081050087,
    0.0237700828574151543,
    0.0239996942982291539,
    0.0241723811174014786,
    0.0242877337207517135,
    0.0243454785045698602,
    0.0243454785045698602,
    0.0242877337207517135,
    0.0241723811174014786,
    0.0239996942982291539,
    0.0237700828574151543,
    0.0234840914081050087,
    0.0231423982906572086,
    0.0227458139637090722,
    0.0222952790818782815,
    0.0217918622646617267,
    0.0212367575618267945,
    0.0206312816213117643,
    0.0199768705663601707,
    0.0192750765893078146,
    0.018527564270120023,
    0.0177361066284411919,
    0.0169025809185708047,
    0.0160289641774257768,
    0.0151173285362012394,
    0.0141698363071297416,
    0.0131887348575273293,
    0.0121763512843554367,
    0.0111350869041916271,
    0.0100674115767651047,
    0.00897585788784867154,
    0.00786301523801235966,
    0.0067315239483593213,
    0.00558406973006556441,
    0.00442337991318197386,
    0.00325222898448918143,
    0.00207351663028123382,
    0.000891640360848216474,
]


def _owens_core[T: FloatLike](h: T, a: T) -> T:
    """`T(h, a)` for `h >= 0`, `0 <= a <= 1`: the two quadratures blended
    at `h = 2`."""
    var hs = min_of(h, T.constant(2.0))
    var hl = max_of(h, T.constant(2.0))
    var hs2 = hs * hs
    var hl2 = hl * hl
    var upper = min_of(hl * a, T.constant(8.5))
    var direct = T.constant(0.0)
    var substituted = T.constant(0.0)
    comptime for i in range(_NODES):
        comptime node = _GL_X[i]
        comptime weight = _GL_W[i]
        var xx = a * T.constant(node)
        var one_plus = T.one() + xx * xx
        direct = (
            direct
            + T.constant(weight)
            * (-(hs2 * one_plus / T.constant(2.0))).exp()
            / one_plus
        )
        var u = upper * T.constant(node)
        var u2 = u * u
        substituted = substituted + T.constant(weight) * (
            -(u2 / T.constant(2.0))
        ).exp() / (hl2 + u2)
    direct = a * direct / T.constant(_TWO_PI)
    substituted = (
        hl
        * upper
        * (-(hl2 / T.constant(2.0))).exp()
        * substituted
        / T.constant(_TWO_PI)
    )
    return blend(ge_indicator(T.constant(2.0), h), direct, substituted)


def _normal_cdf_pair[T: FloatLike](h: T) -> Tuple[T, T]:
    """`(Phi(h), 1 - Phi(h))`, each from its own `erfc` so neither is a
    difference."""
    var z = h * T.constant(_INV_SQRT2)
    return (T.constant(0.5) * (-z).erfc(), T.constant(0.5) * z.erfc())


def owens_t[T: FloatLike](h: T, a: T) -> T:
    """Owen's T function `T(h, a)` for any real `h` and `a`.
    `scipy.special.owens_t(h, a)`. Even in `h`, odd in `a`; the module
    docstring has the reduction and the two quadratures.
    """
    var ah = h.abs()
    var aa = a.abs()
    var sign_a = T.one().copysign(a)
    var small = min_of(aa, T.one())
    var big = max_of(aa, T.one())
    var direct = _owens_core(ah, small)
    var bh = big * ah
    var ph = _normal_cdf_pair(ah)
    var pbh = _normal_cdf_pair(bh)
    var reduced = T.constant(0.5) * (
        ph[0] * pbh[1] + pbh[0] * ph[1]
    ) - _owens_core(bh, T.one() / big)
    return sign_a * blend(ge_indicator(T.one(), aa), direct, reduced)
