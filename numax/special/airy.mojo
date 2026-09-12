"""The Airy functions `Ai`, `Ai'`, `Bi`, `Bi'`, as `scipy.special.airy(x)`
returns them.

**This module is tier 1.** Three regions, each clamped to its own side and
blended by two indicators, so every lane does the same work:

- `|x| <= 1.5`: the Maclaurin series `Ai = c_1 f - c_2 g`, `Bi = sqrt(3)
  (c_1 f + c_2 g)` with `f = sum 3^k (1/3)_k x^{3k} / (3k)!`, `g = sum 3^k
  (2/3)_k x^{3k+1} / (3k+1)!`, fourteen terms (the fifteenth is below
  `1e-18` at `|x| = 1.5`) and the derivatives as their own running
  products. The largest term is `1` here, so the series never cancels.
- `x >= 1.5`: `Ai = (1/pi) sqrt(x/3) K_{1/3}(zeta)`, `Ai' = -(x / pi
  sqrt 3) K_{2/3}(zeta)`, `Bi = sqrt(x/3) (I_{-1/3} + I_{1/3})(zeta)`, `Bi'
  = (x / sqrt 3) (I_{-2/3} + I_{2/3})(zeta)`, `zeta = (2/3) x^{3/2}`, from
  `numax.special.bessel`'s scaled `(I e^{-zeta}, K e^{zeta})` pair with
  the exponential put back once.
- `x <= -1.5`: the same with `J` and `Y` at `|x|`: `Ai = (sqrt|x| / 3)
  (J_{1/3} + J_{-1/3})`, `Ai' = (|x| / 3) (J_{2/3} - J_{-2/3})`, `Bi =
  sqrt(|x|/3) (J_{-1/3} - J_{1/3})`, `Bi' = (|x| / sqrt 3) (J_{-2/3} +
  J_{2/3})`, the negative orders through the reflection formula.

Domain `|x| <= 44`, where `zeta` reaches the `200` the Bessel continued
fraction is deep enough for; `Ai(44)` is already `1e-89`. Checked against
mpmath at `5e-14` relative or better, all four functions, both sides.

## The MAX gate

Nothing: MAX has no Airy functions. **Extend.**
"""

from ..core.numeric import FloatLike, ge_indicator, max_of, min_of
from .bessel import _bessel_ik, _bessel_jy

comptime _PI = 3.141592653589793
comptime _TWO_OVER_PI = 0.6366197723675814
comptime _SQRT3 = 1.7320508075688772
comptime _SIN_PI_3 = 0.8660254037844386
comptime _AI0 = 0.35502805388781724
comptime _AIP0 = 0.2588194037928068
comptime _SERIES_TERMS = 14


def airy[T: FloatLike](x: T) -> Tuple[T, T, T, T]:
    """`(Ai(x), Ai'(x), Bi(x), Bi'(x))` for `|x| <= 44`.
    `scipy.special.airy(x)`. The module docstring has the three regions.
    """
    var ax = x.abs()
    var far = max_of(ax, T.constant(1.5))
    var zeta = T.constant(2.0 / 3.0) * far * far.sqrt()
    var positive = ge_indicator(x, T.constant(1.5))
    var negative = ge_indicator(T.constant(-1.5), x)
    var middle = T.one() - positive - negative

    # Series at the clamped argument.
    var xs = max_of(min_of(x, T.constant(1.5)), T.constant(-1.5))
    var x3 = xs * xs * xs
    var f = T.constant(0.0)
    var g = T.constant(0.0)
    var fp = T.constant(0.0)
    var gp = T.constant(0.0)
    var tf = T.one()
    var tg = xs.copy()
    var tfp = xs * xs / T.constant(2.0)
    var tgp = T.one()
    comptime for k in range(_SERIES_TERMS):
        f = f + tf
        g = g + tg
        gp = gp + tgp
        comptime if k >= 1:
            fp = fp + tfp
            tfp = tfp * x3 / T.constant(Float64((3 * k) * (3 * k + 2)))
        tf = tf * x3 / T.constant(Float64((3 * k + 2) * (3 * k + 3)))
        tg = tg * x3 / T.constant(Float64((3 * k + 3) * (3 * k + 4)))
        tgp = tgp * x3 / T.constant(Float64((3 * k + 1) * (3 * k + 3)))
    var c1 = T.constant(_AI0)
    var c2 = T.constant(_AIP0)
    var ai_mid = c1 * f - c2 * g
    var aip_mid = c1 * fp - c2 * gp
    var bi_mid = T.constant(_SQRT3) * (c1 * f + c2 * g)
    var bip_mid = T.constant(_SQRT3) * (c1 * fp + c2 * gp)

    # `x >= 1.5`: modified Bessel at `zeta`, scaled pair.
    var ik13 = _bessel_ik(T.constant(1.0 / 3.0), zeta)
    var ik23 = _bessel_ik(T.constant(2.0 / 3.0), zeta)
    var e_minus_2z = (-(T.constant(2.0) * zeta)).exp()
    var e_z = zeta.exp()
    var reflect = T.constant(_TWO_OVER_PI * _SIN_PI_3)
    var im13 = ik13[0] + reflect * ik13[1] * e_minus_2z
    var im23 = ik23[0] + reflect * ik23[1] * e_minus_2z
    var root_third = (far / T.constant(3.0)).sqrt()
    var ai_pos = root_third / T.constant(_PI) * ik13[1] / e_z
    var aip_pos = -(far / T.constant(_PI * _SQRT3)) * ik23[1] / e_z
    var bi_pos = root_third * (im13 + ik13[0]) * e_z
    var bip_pos = far / T.constant(_SQRT3) * (im23 + ik23[0]) * e_z

    # `x <= -1.5`: ordinary Bessel at `zeta`; `J_{-1/3} = J_{1/3} / 2 -
    # sin(pi/3) Y_{1/3}`, `J_{-2/3} = -J_{2/3} / 2 - sin(pi/3) Y_{2/3}`.
    var jy13 = _bessel_jy(T.constant(1.0 / 3.0), zeta)
    var jy23 = _bessel_jy(T.constant(2.0 / 3.0), zeta)
    var jm13 = T.constant(0.5) * jy13[0] - T.constant(_SIN_PI_3) * jy13[1]
    var jm23 = T.constant(-0.5) * jy23[0] - T.constant(_SIN_PI_3) * jy23[1]
    var ai_neg = far.sqrt() / T.constant(3.0) * (jy13[0] + jm13)
    var aip_neg = far / T.constant(3.0) * (jy23[0] - jm23)
    var bi_neg = root_third * (jm13 - jy13[0])
    var bip_neg = far / T.constant(_SQRT3) * (jm23 + jy23[0])

    var ai = ai_mid * middle + ai_pos * positive + ai_neg * negative
    var aip = aip_mid * middle + aip_pos * positive + aip_neg * negative
    var bi = bi_mid * middle + bi_pos * positive + bi_neg * negative
    var bip = bip_mid * middle + bip_pos * positive + bip_neg * negative
    return (ai^, aip^, bi^, bip^)
