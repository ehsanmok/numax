"""The Struve function `H_v(x)`. `scipy.special.struve(v, x)`.

**This module is tier 1.** The power series `sum (-1)^k (x/2)^{2k+v+1} /
(Gamma(k+3/2) Gamma(k+v+3/2))` alternates and its terms grow to `e^{x}`
before they shrink, so past `x ~ 6` it has lost digits and past `x ~ 20`
all of them. Two regions replace it, blended at `x = 40`:

- `x <= 40`: the Neumann-type series in Bessel functions, DLMF 11.4.18,
  `H_v = 4 / (sqrt(pi) Gamma(v+1/2)) sum_k (2k+v+1) Gamma(k+v+1) /
  (k! (2k+1)(2k+2v+1)) J_{2k+v+1}(x)`. Its coefficients are bounded and
  the Bessel functions are at most `1`, so nothing cancels; and every
  `J_{v+n}(x)` it needs comes out of one backward recurrence (Miller's
  algorithm) from order `v + 100`, normalized by the Neumann identity
  `(x/2)^v / Gamma(v+1) = sum_k (v+2k) Gamma(v+k) / (k! Gamma(v+1))
  J_{v+2k}(x)` accumulated in the same pass. The pair is renormalized to
  unit L1 norm at every step so `x -> 0` cannot overflow it. Forty Bessel
  terms, orders up to `v + 81`; the recurrence starts far enough above
  `x = 40` that `J_{v+100}(40)` is below `1e-26`.
- `x > 40`: `H_v = Y_v + (1/pi) sum_{k<20} Gamma(k+1/2) (x/2)^{v-2k-1} /
  Gamma(v+1/2-k)`, DLMF 11.6.1, the terms as a running ratio `(k+1/2)(v -
  1/2 - k) / (x/2)^2` so `Gamma` at its poles (half-integer `v`, where the
  sum terminates) never has to be evaluated. The twentieth term is `2e-19`
  relative at `x = 40`, `v = 0`.

Domain `v >= 0`, `0 < x <= 200` (the `Y_v` continued fraction's ceiling);
checked against mpmath at `4e-14` relative or better for `v <= 20`.

## The MAX gate

Nothing: MAX has no Struve function. **Extend.**
"""

from ..core.numeric import FloatLike, blend, ge_indicator, max_of, min_of
from .bessel import _bessel_jy
from .gamma import lgamma

comptime _PI = 3.141592653589793
comptime _SQRT_PI = 1.7724538509055159
comptime _TINY = 1e-30
comptime _MILLER = 100
comptime _BESSEL_TERMS = 40
comptime _ASYMPTOTIC_TERMS = 20


def _struve_bessel_series[T: FloatLike](v: T, x: T) -> T:
    """DLMF 11.4.18 over Miller's backward recurrence, `x <= 40`."""
    var a = T.one()
    var b = T.constant(0.0)
    # `P_k = (v+1)_k / k!`, walked down from `P_{49}` as the recurrence
    # passes each order; the first order it meets is `v + 99`, odd offset
    # `2 * 49 + 1`.
    var p = T.one()
    comptime for k in range(1, _MILLER // 2):
        p = p * (v + T.constant(Float64(k))) / T.constant(Float64(k))
    var neumann = T.constant(0.0)
    var struve_sum = T.constant(0.0)
    var two_over_x = T.constant(2.0) / x
    comptime for n in range(_MILLER, 0, -1):
        # `j_{v+n-1} = (2 (v+n) / x) j_{v+n} - j_{v+n+1}`.
        var below = (v + T.constant(Float64(n))) * two_over_x * a - b
        b = a^
        a = below^
        comptime m = n - 1
        comptime if m % 2 == 0:
            comptime k = m // 2
            comptime if k == 0:
                neumann = neumann + a
            else:
                neumann = (
                    neumann
                    + (v + T.constant(Float64(2 * k)))
                    / (v + T.constant(Float64(k)))
                    * p
                    * a
                )
                p = p * T.constant(Float64(k)) / (v + T.constant(Float64(k)))
        else:
            comptime k = (m - 1) // 2
            comptime if k <= _BESSEL_TERMS:
                struve_sum = struve_sum + (
                    T.constant(Float64(2 * k + 1)) + v
                ) * p * a / (
                    T.constant(Float64(2 * k + 1))
                    * (T.constant(Float64(2 * k + 1)) + T.constant(2.0) * v)
                )
        var scale = a.abs() + b.abs()
        a = a / scale
        b = b / scale
        neumann = neumann / scale
        struve_sum = struve_sum / scale
    # `4 Gamma(v+1) / (sqrt(pi) Gamma(v+1/2))` times `(x/2)^v / Gamma(v+1)`.
    var prefactor = (
        v * (x / T.constant(2.0)).ln() - lgamma(v + T.constant(0.5))
    ).exp() * T.constant(4.0 / _SQRT_PI)
    return prefactor * struve_sum / neumann


def _struve_asymptotic[T: FloatLike](v: T, x: T) -> T:
    """DLMF 11.6.1, `x >= 40`."""
    var half_x = x / T.constant(2.0)
    var term = (
        T.constant(_SQRT_PI)
        * ((v - T.one()) * half_x.ln() - lgamma(v + T.constant(0.5))).exp()
    )
    var total = term.copy()
    var inv_sq = T.one() / (half_x * half_x)
    comptime for k in range(1, _ASYMPTOTIC_TERMS):
        term = (
            term
            * T.constant(Float64(k) - 0.5)
            * (v - T.constant(Float64(k) - 0.5))
            * inv_sq
        )
        total = total + term
    return _bessel_jy(v, x)[1] + total / T.constant(_PI)


def struve[T: FloatLike](v: T, x: T) -> T:
    """The Struve function `H_v(x)` for `v >= 0` and `0 < x <= 200`.
    `scipy.special.struve(v, x)`. The Bessel-function series to `x = 40`
    and the asymptotic expansion past it, blended; the module docstring has
    both.
    """
    var near = ge_indicator(T.constant(40.0), x)
    var small = max_of(min_of(x, T.constant(40.0)), T.constant(_TINY))
    var large = max_of(x, T.constant(40.0))
    return blend(
        near, _struve_bessel_series(v, small), _struve_asymptotic(v, large)
    )
