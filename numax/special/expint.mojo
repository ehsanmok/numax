"""Exponential, sine, cosine and Fresnel integrals: `exp1`, `expi`,
`expn`, `sici` and `fresnel`, with `scipy.special`'s definitions.

**This module is tier 1.** Every function is a fixed-depth series blended
against a fixed-depth continued fraction at a threshold, with each side's
argument clamped into the region where it is accurate, so both sides are
finite everywhere and the blend is the arithmetic `0`/`1` select
`numax.special.gamma` describes. No `if` on data, no tolerance loop:
launchable per lane inside a kernel, differentiable at `Dual`.

## One continued fraction, three functions

`E_1(z) = e^{-z} / (z + 1 - 1^2 / (z + 3 - 2^2 / (z + 5 - ...)))` is the
even contraction of the exponential integral's S-fraction, and it
converges for every `z` off the negative real axis -- including the
imaginary axis. Evaluated from the tail at a fixed depth over any
`FloatLike`, it gives `E_1(x)` for real `x` (the tail of `exp1`, and of
`expi` through `Ei(-x) = -E_1(x)`), and at `z = i x` over `Complex[T]` it
gives `Ci(x) = -Re E_1(ix)` and `Si(x) = pi/2 + Im E_1(ix)` -- the
conformer layer doing the complex arithmetic the classic auxiliary
functions `f` and `g` would otherwise need rational fits for. The Fresnel
integrals take the same route through `erfc`'s continued fraction at `z =
(1 - i) sqrt(pi) x / 2`, since `C(x) + i S(x) = (1 + i) / 2 * erf(z)`.
`expn` is the generalization `E_n(x) = e^{-x} / (x + n - 1 n / (x + n + 2
- 2 (n+1) / ...))`, Numerical Recipes' fraction, with `n` a scalar `Int`
shared by every lane.

## Error bounds, from the harness

The algorithms' own errors, checked in mpmath before transcription, sit at
the blend thresholds where each side is at its worst: `E_1` (series to 40
terms below `1.5`, fraction to depth 60 above) within `2e-15` relative;
`Si`/`Ci` (series to 30 terms below `2`, fraction to depth 80 above) within
`2e-15` absolute; the Fresnel integrals (series to 25 terms below `2`,
`erfc` fraction to depth 80 above) within `3e-16`; `Ei` on the positive
axis (series to 120 terms below `40`, the asymptotic sum to 20 terms
above) within `5e-14` relative.

What `pixi run accuracy` measures is larger for four of them, and the
reason is the primitive rather than the algorithm: every series region
carries a `ln x` term, and `std.math.log` is off by up to `8e-10` absolute
(`bench/accuracy/README.md`), so `exp1`, `expi`, `expn` and `Ci` read
`~3e-9` relative at their worst point while `Si` and the Fresnel pair,
which never take a logarithm, read `1e-15`. The floor is inherited, not
introduced, and a correctly rounded `ln` would collapse the four to the
others' level with no change here.

## The MAX gate

Nothing: MAX ships no exponential or trigonometric integrals. **Extend.**
"""

from std.collections import Array

from ..core.complex import Complex
from ..core.numeric import FloatLike, blend, ge_indicator, max_of, min_of

comptime _EULER = 0.5772156649015329
comptime _PI = 3.141592653589793
comptime _HALF_PI = 1.5707963267948966
comptime _SQRT_PI = 1.7724538509055159
comptime _TINY = 1e-300


def _e1_fraction[U: FloatLike, depth: Int](z: U) -> U:
    """`E_1(z)` by the even-contracted continued fraction from the tail,
    `depth` levels deep. Valid off the negative real axis; over
    `Complex[T]` it is the sine and cosine integrals' engine."""
    var t = z + U.constant(Float64(2 * depth + 1))
    comptime for step in range(depth):
        comptime k = depth - step
        t = z + U.constant(Float64(2 * k - 1)) - U.constant(Float64(k * k)) / t
    return (-z).exp() / t


def _e1_series[T: FloatLike, terms: Int](x: T) -> T:
    """`E_1(x) = -gamma - ln x - sum_{k>=1} (-x)^k / (k k!)`, for small
    positive `x`."""
    var total = T.constant(0.0)
    var power = T.one()
    comptime for k in range(1, terms + 1):
        power = power * (-x) / T.constant(Float64(k))
        total = total + power / T.constant(Float64(k))
    return T.constant(-_EULER) - x.ln() - total


def exp1[T: FloatLike](x: T) -> T:
    """The exponential integral `E_1(x) = int_x^inf e^{-t} / t dt` for
    `x > 0`. `scipy.special.exp1(x)`.

    The power series below `x = 1.5` and the continued fraction above it,
    each clamped to its side and blended; `x <= 0` gives NaN or infinity
    from the logarithm, as SciPy's real `exp1` does off its domain.
    """
    var near = ge_indicator(T.constant(1.5), x)
    var small = min_of(x, T.constant(1.5))
    var large = max_of(x, T.constant(1.5))
    return blend(near, _e1_series[T, 40](small), _e1_fraction[T, 60](large))


def expi[T: FloatLike](x: T) -> T:
    """The exponential integral `Ei(x) = -int_{-x}^inf e^{-t} / t dt`, for
    any real `x != 0`. `scipy.special.expi(x)`.

    Three regions blended: `Ei(x) = -E_1(-x)` for `x < 0`; the power
    series `gamma + ln x + sum x^k / (k k!)` for `0 < x <= 40`, whose terms
    are all positive so it never cancels; and the asymptotic `e^x / x sum
    k! / x^k` beyond, where the series would need more terms than it is
    worth and the asymptotic sum is already at `5e-14`. Overflows past
    `x ~ 700`, as `e^x` does.
    """
    var positive = ge_indicator(x, T.constant(0.0))
    var ax = x.abs()

    # `x > 0`: series below 40, asymptotic above.
    var series_arg = max_of(min_of(ax, T.constant(40.0)), T.constant(_TINY))
    var total = T.constant(0.0)
    var power = T.one()
    comptime for k in range(1, 121):
        power = power * series_arg / T.constant(Float64(k))
        total = total + power / T.constant(Float64(k))
    var series = T.constant(_EULER) + series_arg.ln() + total

    var asym_arg = max_of(ax, T.constant(40.0))
    var asym_total = T.one()
    var term = T.one()
    comptime for k in range(1, 20):
        term = term * T.constant(Float64(k)) / asym_arg
        asym_total = asym_total + term
    var asymptotic = asym_arg.exp() / asym_arg * asym_total
    var positive_value = blend(
        ge_indicator(T.constant(40.0), ax), series, asymptotic
    )

    # `x < 0`: `-E_1(|x|)`, with `exp1`'s own two regions.
    var negative_value = -exp1(max_of(ax, T.constant(_TINY)))
    return blend(positive, positive_value, negative_value)


def expn[T: FloatLike](n: Int, x: T) -> T:
    """The generalized exponential integral `E_n(x) = int_1^inf e^{-x t} /
    t^n dt` for integer `n >= 0` and `x > 0`. `scipy.special.expn(n, x)`.

    `n` is one scalar shared by every lane -- the order, not data -- so a
    loop over it is fixed work. `E_0` is `e^{-x} / x`; for `n >= 1`, the
    series with its digamma term below `x = 1` (Numerical Recipes 6.3) and
    the continued fraction above, blended.
    """
    if n == 0:
        return (-x).exp() / x
    var near = ge_indicator(T.one(), x)
    var small = max_of(min_of(x, T.one()), T.constant(_TINY))
    var large = max_of(x, T.one())

    # Continued fraction: `E_n(x) = e^{-x} / (x + n - 1 n / (x + n + 2 -
    # 2 (n+1) / ...))`, from the tail at depth 60.
    comptime depth = 60
    var t = large + T.constant(Float64(n + 2 * depth))
    for step in range(depth):
        var k = depth - step
        t = (
            large
            + T.constant(Float64(n + 2 * (k - 1)))
            - T.constant(Float64(k * (n + k - 1))) / t
        )
    var fraction = (-large).exp() / t

    # Series: `(-x)^{n-1} / (n-1)! (-ln x + psi(n)) - sum_{k != n-1}
    # (-x)^k / ((k - n + 1) k!)`, to 60 terms.
    var psi = -_EULER
    for m in range(1, n):
        psi += 1.0 / Float64(m)
    var lead = T.one()
    for k in range(1, n):
        lead = lead * (-small) / T.constant(Float64(k))
    var series = lead * (T.constant(psi) - small.ln())
    var power = T.one()
    for k in range(60):
        if k > 0:
            power = power * (-small) / T.constant(Float64(k))
        if k != n - 1:
            series = series - power / T.constant(Float64(k - n + 1))
    return blend(near, series, fraction)


def sici[T: FloatLike](x: T) -> Tuple[T, T]:
    """The sine and cosine integrals `Si(x) = int_0^x sin t / t dt` and
    `Ci(x) = gamma + ln x + int_0^x (cos t - 1) / t dt`, as `(Si, Ci)`.
    `scipy.special.sici(x)`.

    `Si` is odd and computed at `|x|` with the sign restored; `Ci` is even
    in its real part, which is what is returned for `x < 0` (SciPy drops
    the `i pi` of the complex value too). Power series below `|x| = 2`,
    `E_1(i|x|)` over `Complex[T]` by the continued fraction above it: `Ci
    = -Re`, `Si = pi/2 + Im`.
    """
    var ax = x.abs()
    var sign = T.one().copysign(x)
    var near = ge_indicator(T.constant(2.0), ax)
    var small = max_of(min_of(ax, T.constant(2.0)), T.constant(_TINY))
    var large = max_of(ax, T.constant(2.0))

    # Series: Si = sum (-1)^k x^{2k+1} / ((2k+1)(2k+1)!),
    # Ci = gamma + ln x + sum (-1)^k x^{2k} / (2k (2k)!).
    var x2 = small * small
    var si_series = T.constant(0.0)
    var ci_series = T.constant(0.0)
    var odd_power = small.copy()
    var even_power = T.one()
    comptime for k in range(30):
        comptime sign_k = -1.0 if k % 2 == 1 else 1.0
        si_series = si_series + odd_power * T.constant(
            sign_k / Float64(2 * k + 1)
        )
        odd_power = (
            odd_power * x2 / T.constant(Float64((2 * k + 2) * (2 * k + 3)))
        )
        even_power = (
            even_power * x2 / T.constant(Float64((2 * k + 1) * (2 * k + 2)))
        )
        ci_series = ci_series + even_power * T.constant(
            -sign_k / Float64(2 * k + 2)
        )
    ci_series = T.constant(_EULER) + small.ln() + ci_series

    var e1 = _e1_fraction[Complex[T], 80](
        Complex[T](T.constant(0.0), large.copy())
    )
    var si_fraction = T.constant(_HALF_PI) + e1.im
    var ci_fraction = -e1.re

    var si = blend(near, si_series, si_fraction)
    var ci = blend(near, ci_series, ci_fraction)
    return (si * sign, ci^)


def fresnel[T: FloatLike](x: T) -> Tuple[T, T]:
    """The Fresnel integrals `S(x) = int_0^x sin(pi t^2 / 2) dt` and `C(x)
    = int_0^x cos(pi t^2 / 2) dt`, as `(S, C)`. `scipy.special.fresnel(x)`.

    Both odd, computed at `|x|` with the sign restored. Power series below
    `|x| = 2`; above it, `C + i S = (1 + i) / 2 * erf(z)` at `z = (1 - i)
    sqrt(pi) x / 2`, with `erfc(z)` by its continued fraction over
    `Complex[T]` from the tail at depth 80.
    """
    var ax = x.abs()
    var sign = T.one().copysign(x)
    var near = ge_indicator(T.constant(2.0), ax)
    var small = min_of(ax, T.constant(2.0))
    var large = max_of(ax, T.constant(2.0))

    # Series: S = sum (-1)^k (pi/2)^{2k+1} x^{4k+3} / ((2k+1)! (4k+3)),
    # C = sum (-1)^k (pi/2)^{2k} x^{4k+1} / ((2k)! (4k+1)).
    var u = T.constant(_HALF_PI) * small * small
    var u2 = u * u
    var s_series = T.constant(0.0)
    var c_series = T.constant(0.0)
    var s_power = u * small
    var c_power = small.copy()
    comptime for k in range(25):
        comptime sign_k = -1.0 if k % 2 == 1 else 1.0
        c_series = c_series + c_power * T.constant(sign_k / Float64(4 * k + 1))
        s_series = s_series + s_power * T.constant(sign_k / Float64(4 * k + 3))
        c_power = c_power * u2 / T.constant(Float64((2 * k + 1) * (2 * k + 2)))
        s_power = s_power * u2 / T.constant(Float64((2 * k + 2) * (2 * k + 3)))

    # erfc(z) = e^{-z^2} / (sqrt(pi) (z + (1/2) / (z + 1 / (z + (3/2) / ...))))
    var half_root_pi = T.constant(_SQRT_PI / 2.0) * large
    var z = Complex[T](half_root_pi.copy(), -half_root_pi)
    var t = z.copy()
    comptime for step in range(80):
        comptime k = 80 - step
        t = z + Complex[T].constant(Float64(k) / 2.0) / t
    var erfc_z = (-(z * z)).exp() / (Complex[T].constant(_SQRT_PI) * t)
    var erf_z = Complex[T].constant(1.0) - erfc_z
    var w = Complex[T](T.constant(0.5), T.constant(0.5)) * erf_z

    var s = blend(near, s_series, w.im)
    var c = blend(near, c_series, w.re)
    return (s * sign, c * sign)
