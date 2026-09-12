"""Bessel functions: `j0`/`j1`/`y0`/`y1` by Abramowitz & Stegun's polynomials,
and `jv`/`yv`/`iv`/`kv` of arbitrary real order by Temme's method.

**This module is tier 1.** The near/far split is a `0`/`1` blend built from
`copysign` rather than a branch, and the far branch's argument is clamped
so both sides are finite everywhere -- both sides are always evaluated.

`std.math` ships full-domain `j0`/`j1`/`y0`/`y1`, but -- same story as
`numax.special.gamma`'s docstring -- they're CPU-only libm calls, rejected outright
inside a GPU kernel. (On this platform's libm, the `float32` symbols for
all four are missing entirely, so they're not even usable at `numax`'s
usual `dtype` on CPU alone.) Not adopted here for the same reason `gamma`
wasn't: it would make these functions CPU-only for `Plain` specifically,
regressing GPU support this module already has.

Every function below blends a "near" branch (a polynomial, valid at small
`|x|`) with a "far" branch (an oscillating asymptotic form, valid at large
`|x|`) at `|x| = 3`, the same threshold Abramowitz & Stegun's own 9.4.1-9.4.6
polynomials split at. Both branches are always computed and blended
arithmetically, the same way `numax.special.gamma`'s reflection is: no `if` (`Self`
may hold a SIMD vector with lanes on both sides of the threshold), and the
far branch's argument is clamped to `max(|x|, 3.0)` first (the same
"clamp so the discarded side stays finite" trick `gamma` uses `max(x, 1-x)`
for) so it never divides by a near-zero `|x|` -- the raw asymptotic formula
would otherwise blow up (and eventually produce `cos`/`sin` of an
overflowing argument, i.e. NaN) well before reaching the actual blend.
`1/sqrt(ax)` in the far branch is `exp(-0.5 * ln(ax))`, since `FloatLike`
has `ln`/`exp` but no `sqrt` of its own -- not worth adding a primitive for
one call site.

Coefficients for `J0`/`J1`'s near and far branches, and `Y0`'s near branch,
are Abramowitz & Stegun 9.4.1-9.4.5, transcribed and then checked
numerically against `std.math`'s `j0`/`j1`/`y0` (float64, CPU-only --
used here purely as a test oracle, see `tests/special/test_bessel.mojo`) to within
each formula's documented error bound. `Y1`'s near-branch polynomial
(9.4.6) doesn't get the same transcription treatment: rather than risk
mistranscribing its coefficients from a source with no clean, non-OCR
digitization, they're a direct least-squares fit of `Y1(x) - (2/pi)*
(ln(x/2)*J1(x) - 1/x)` against `std.math.y1` over `x` in `(0, 3]` -- checked
to agree with the reference to ~1e-9, comparable to or better than the
other branches' A&S-quoted error bounds, so nothing here is any less
trustworthy for being fit rather than transcribed.

## Arbitrary real order

`jv`, `yv`, `iv`, `kv` (with the exponentially scaled `ive`, `kve`) and the
spherical `spherical_jn`/`spherical_yn` are one algorithm, Numerical
Recipes' `bessjy`/`bessik` (Temme's method) recast without a single
data-dependent loop. For `|v|` the order is split `v = nl + mu` with `nl =
floor(v + 1/2)` and `|mu| <= 1/2`; the ratio `J_{v+1}/J_v` (or
`I_{v+1}/I_v`) comes from its continued fraction evaluated backward from a
fixed depth of 260 (100 for `I`), enough for `x <= 200`; the pair `(J_v,
J_{v+1})` is recurred down to `mu` in a fixed 30 steps, each step taken or
held by a `0`/`1` indicator so the loop count is the same for every lane
and the order is capped at `|v| <= 30`; at `mu` the second solution comes
from Temme's series for `x < 2` and from Steed's continued fraction (depth
60, complex for `Y`, depth 100 for `K`) for `x >= 2`, blended; the
Wronskian `J_{mu+1} Y_mu - J_mu Y_{mu+1} = 2/(pi x)` (`I_mu K_{mu+1} +
I_{mu+1} K_mu = 1/x`) fixes the scale of the `J` chain with no division by
a value that could be a zero of `J`; and `Y`, `K` are recurred upward,
again in 30 held-or-taken steps. The sign of `J_v` is the product of the
continued fraction's denominator signs, since `J_{v+260}(x) > 0` for every
`x` in range. Negative orders go through the reflection formulas with
`cos(pi v)` and `sin(pi v)` computed from the fractional part of `v`, so
they are exact at the integers. Every step is a blend, so the whole family
is tier 1 and was compiled into a Metal kernel before it was called that.

Domain: `0 < x <= 200`, `|v| <= 30`. `x` is floored at `1e-30`, so
`spherical_jn(0, 0)` is `1` rather than `0/0`. One soft spot: an order a
hair above a half-integer (`v = 2.5 + 1e-8`) at `x` below `1e-6` reduces
to `mu` just above `-1/2`, where `Y_mu` is nearly the small solution and
Temme's series cancels by `(2/x)^{2|mu|}`; exact half-integers are routed
to `+1/2` and do not. The continued fraction for
`J` is the ceiling on `x`: past 200 its depth is not enough and the
result is wrong rather than slow, which is why the bound is stated; the
upgrade path is Hankel's asymptotic expansion above it. Measured in `pixi
run accuracy` at `1e-13` relative or better across the domain, so `jv(0,
x)` is the full-precision route where `j0`'s A&S polynomial (`5e-8`) is
not enough.
"""

from std.collections import Array

from ..core.complex import Complex
from ..core.numeric import (
    FloatLike,
    blend,
    ge_indicator,
    guard_nonzero,
    max_of,
    min_of,
)

comptime _PI = 3.14159265358979323846
comptime _TWO_OVER_PI = 0.6366197723675814
comptime _HALF_PI = 1.5707963267948966
comptime _TINY = 1e-30
comptime _CF1_J_DEPTH = 260
comptime _CF1_I_DEPTH = 100
comptime _CF2_JY_DEPTH = 60
comptime _CF2_K_DEPTH = 100
comptime _TEMME_TERMS = 20
comptime _ORDER_STEPS = 30

# `1/Gamma(z) = sum_{k>=1} c_k z^k`, `c_1 .. c_26` (A&S 6.1.34, regenerated
# in mpmath at 40 digits). Temme's `Gamma_1(mu) = (1/Gamma(1-mu) -
# 1/Gamma(1+mu)) / (2 mu)` and `Gamma_2 = (1/Gamma(1-mu) + 1/Gamma(1+mu)) /
# 2` fall out of it as the even and odd halves, `-sum c_{2j+2} mu^{2j}` and
# `sum c_{2j+1} mu^{2j}`, with no cancellation at `mu -> 0` where the
# quotient form loses every digit. `c_26 mu^26` is `2e-24` at `|mu| = 1/2`.
comptime _RGAMMA: Array[Float64, 26] = [
    1.0,
    0.577215664901532861,
    -0.655878071520253881,
    -0.0420026350340952355,
    0.16653861138229149,
    -0.0421977345555443367,
    -0.00962197152787697356,
    0.00721894324666309954,
    -0.00116516759185906511,
    -0.000215241674114950973,
    0.000128050282388116186,
    -0.0000201348547807882387,
    -1.25049348214267066e-6,
    1.13302723198169588e-6,
    -2.0563384169776071e-7,
    6.11609510448141582e-9,
    5.00200764446922293e-9,
    -1.18127457048702014e-9,
    1.04342671169110051e-10,
    7.78226343990507125e-12,
    -3.69680561864220571e-12,
    5.10037028745447598e-13,
    -2.05832605356650678e-14,
    -5.34812253942301798e-15,
    1.22677862823826079e-15,
    -1.18125930169745877e-16,
]


def _far_branch_indicator[T: FloatLike](ax: T) -> T:
    """`1` where `ax <= 3`, `0` where `ax > 3` -- branchless."""
    return ge_indicator(T.constant(3.0) - ax, T.constant(0.0))


def _clamp_to_far_domain[T: FloatLike](ax: T) -> T:
    """`max(ax, 3.0)`: keeps the far branch's `3/ax` finite even when the
    real `ax` is near zero and the near branch is about to be selected
    instead.
    """
    return max_of(ax, T.constant(3.0))


def _inv_sqrt[T: FloatLike](ax_safe: T) -> T:
    """`1/sqrt(ax_safe)`. This used to be `exp(-0.5*ln(ax_safe))`, from
    before `FloatLike` had a `sqrt` -- one hardware instruction now instead
    of two transcendental calls, and it drops the accuracy the round trip
    through `ln`/`exp` was costing.
    """
    return T.one() / ax_safe.sqrt()


def _j0_near[T: FloatLike](x: T) -> T:
    """A&S 9.4.1: `J0(x)` for `|x| <= 3`, max absolute error ~5e-8.

    `x` only appears as `x^2`, so no `abs()` is needed -- already even.
    """
    var t = (x * x) / T.constant(9.0)
    return (
        (
            (
                (
                    (T.constant(0.0002100) * t + T.constant(-0.0039444)) * t
                    + T.constant(0.0444479)
                )
                * t
                + T.constant(-0.3163866)
            )
            * t
            + T.constant(1.2656208)
        )
        * t
        + T.constant(-2.2499997)
    ) * t + T.one()


def _j1_near[T: FloatLike](x: T) -> T:
    """A&S 9.4.2: `J1(x)` for `|x| <= 3`, max absolute error ~1.3e-8.

    The polynomial is a function of `t = (x/3)^2` (even), multiplied by
    `x` itself -- odd overall, matching `J1`, with no explicit sign
    handling needed.
    """
    var t = (x * x) / T.constant(9.0)
    var poly = (
        (
            (
                (
                    (T.constant(0.00001109) * t + T.constant(-0.00031761)) * t
                    + T.constant(0.00443319)
                )
                * t
                + T.constant(-0.03954289)
            )
            * t
            + T.constant(0.21093573)
        )
        * t
        + T.constant(-0.56249985)
    ) * t + T.constant(0.5)
    return x * poly


def _f0[T: FloatLike](p: T) -> T:
    """A&S 9.4.3's `f0(p)`, shared by `J0`'s and `Y0`'s far branch."""
    return (
        (
            (
                (
                    (T.constant(0.00014476) * p + T.constant(-0.00072805)) * p
                    + T.constant(0.00137237)
                )
                * p
                + T.constant(-0.00009512)
            )
            * p
            + T.constant(-0.00552740)
        )
        * p
        + T.constant(-0.00000077)
    ) * p + T.constant(0.79788456)


def _theta0[T: FloatLike](ax_safe: T, p: T) -> T:
    """A&S 9.4.3's `theta0(p)`, shared by `J0`'s and `Y0`'s far branch."""
    return (
        ax_safe
        + (
            (
                (
                    (
                        (T.constant(0.00013558) * p + T.constant(-0.00029333))
                        * p
                        + T.constant(-0.00054125)
                    )
                    * p
                    + T.constant(0.00262573)
                )
                * p
                + T.constant(-0.00003954)
            )
            * p
            + T.constant(-0.04166397)
        )
        * p
        + T.constant(-0.78539816)
    )


def _f1[T: FloatLike](p: T) -> T:
    """A&S 9.4.4's `f1(p)`, shared by `J1`'s and `Y1`'s far branch."""
    return (
        (
            (
                (
                    (T.constant(-0.00020033) * p + T.constant(0.00113653)) * p
                    + T.constant(-0.00249511)
                )
                * p
                + T.constant(0.00017105)
            )
            * p
            + T.constant(0.01659667)
        )
        * p
        + T.constant(0.00000156)
    ) * p + T.constant(0.79788456)


def _theta1[T: FloatLike](ax_safe: T, p: T) -> T:
    """A&S 9.4.4's `theta1(p)`, shared by `J1`'s and `Y1`'s far branch."""
    return (
        ax_safe
        + (
            (
                (
                    (
                        (T.constant(-0.00029166) * p + T.constant(0.00079824))
                        * p
                        + T.constant(0.00074348)
                    )
                    * p
                    + T.constant(-0.00637879)
                )
                * p
                + T.constant(0.00005650)
            )
            * p
            + T.constant(0.12499612)
        )
        * p
        + T.constant(-2.35619449)
    )


def j0[T: FloatLike](x: T) -> T:
    """`J0(x)`, the order-zero Bessel function of the first kind, for any
    real `x`. Even, so only `|x|` ever reaches either branch.
    """
    var ax = x.abs()
    var s = _far_branch_indicator(ax)

    var near = _j0_near(x)

    var ax_safe = _clamp_to_far_domain(ax)
    var p = T.constant(3.0) / ax_safe
    var far = _f0(p) * _inv_sqrt(ax_safe) * _theta0(ax_safe, p).cos()

    return near * s + far * (T.one() - s)


def j1[T: FloatLike](x: T) -> T:
    """`J1(x)`, the order-one Bessel function of the first kind, for any
    real `x`. Odd: the far branch is computed at `|x|` and then given
    `x`'s sign back via `copysign` (`_j1_near`'s `x * poly(x^2)` shape
    already makes the near branch odd on its own).
    """
    var ax = x.abs()
    var s = _far_branch_indicator(ax)

    var near = _j1_near(x)

    var ax_safe = _clamp_to_far_domain(ax)
    var p = T.constant(3.0) / ax_safe
    var far_at_ax = _f1(p) * _inv_sqrt(ax_safe) * _theta1(ax_safe, p).cos()
    # `far_at_ax` is `J1`'s far-branch value at the *positive* mirror
    # point `ax`, already carrying whatever sign `J1` actually has there
    # (`J1` oscillates through both signs for positive arguments, same as
    # `J0` -- this is a genuine sign flip for `J1`'s oddness, not a
    # magnitude/sign split, so it's a `*`, not a `copysign`).
    var far = far_at_ax * T.one().copysign(x)

    return near * s + far * (T.one() - s)


def y0[T: FloatLike](x: T) -> T:
    """`Y0(x)`, the order-zero Bessel function of the second kind, for
    `x > 0` (`Y0` itself has a logarithmic singularity at `0` and isn't
    real-valued for negative `x`).
    """
    var s = _far_branch_indicator(x)

    var t = (x * x) / T.constant(9.0)
    var near_poly = (
        (
            (
                (
                    (T.constant(-0.00024846) * t + T.constant(0.00427916)) * t
                    + T.constant(-0.04261214)
                )
                * t
                + T.constant(0.25300117)
            )
            * t
            + T.constant(-0.74350384)
        )
        * t
        + T.constant(0.60559366)
    ) * t + T.constant(0.36746691)
    var near = (
        T.constant(_TWO_OVER_PI) * (x * T.constant(0.5)).ln() * _j0_near(x)
        + near_poly
    )

    var x_safe = _clamp_to_far_domain(x)
    var p = T.constant(3.0) / x_safe
    var far = _f0(p) * _inv_sqrt(x_safe) * _theta0(x_safe, p).sin()

    return near * s + far * (T.one() - s)


def y1[T: FloatLike](x: T) -> T:
    """`Y1(x)`, the order-one Bessel function of the second kind, for
    `x > 0` (same domain restriction as `Y0`, plus its own `-2/(pi*x)`
    singularity at `0`).

    The near branch's polynomial correction (see this module's docstring)
    is a least-squares fit rather than a transcription; everything else is
    A&S 9.4.2/9.4.4/9.4.6's shared `ln(x/2)*J1(x) - 1/x` structure.
    """
    var s = _far_branch_indicator(x)

    var t = (x * x) / T.constant(9.0)
    var near_poly = (
        (
            (
                (
                    (
                        T.constant(-1.9114694908021805e-05) * t
                        + T.constant(0.0003762544910492209)
                    )
                    * t
                    + T.constant(-0.00454698544460594)
                )
                * t
                + T.constant(0.03477406025093513)
            )
            * t
            + T.constant(-0.14629895727503786)
        )
        * t
        + T.constant(0.24092313569092993)
    ) * t + T.constant(0.024578509592272924)
    var near = (
        T.constant(_TWO_OVER_PI)
        * ((x * T.constant(0.5)).ln() * _j1_near(x) - (T.one() / x))
        + x * near_poly
    )

    var x_safe = _clamp_to_far_domain(x)
    var p = T.constant(3.0) / x_safe
    var far = _f1(p) * _inv_sqrt(x_safe) * _theta1(x_safe, p).sin()

    return near * s + far * (T.one() - s)


# --- arbitrary real order --------------------------------------------------


def _gamma_pair[T: FloatLike](mu: T) -> Tuple[T, T, T, T]:
    """Temme's `(Gamma_1(mu), Gamma_2(mu), 1/Gamma(1+mu), 1/Gamma(1-mu))`
    for `|mu| <= 1/2`, from the Taylor series of `1/Gamma`."""
    var mu2 = mu * mu
    var gam1 = T.constant(0.0)
    var gam2 = T.constant(0.0)
    var power = T.one()
    comptime for j in range(13):
        comptime c_odd = _RGAMMA[2 * j]
        comptime c_even = _RGAMMA[2 * j + 1]
        gam2 = gam2 + T.constant(c_odd) * power
        gam1 = gam1 - T.constant(c_even) * power
        power = power * mu2
    var gampl = gam2 - mu * gam1
    var gammi = gam2 + mu * gam1
    return (gam1^, gam2^, gampl^, gammi^)


def _sinhc[T: FloatLike](e: T) -> T:
    """`sinh(e) / e`: the series below `|e| = 0.1` (truncation `2e-22`), the
    exponentials above it, blended -- the quotient alone cancels to noise
    at small `e`."""
    var ae = e.abs()
    var near = ge_indicator(T.constant(0.1), ae)
    var e2 = e * e
    var series = T.one() + e2 / T.constant(6.0) * (
        T.one()
        + e2
        / T.constant(20.0)
        * (
            T.one()
            + e2
            / T.constant(42.0)
            * (
                T.one()
                + e2 / T.constant(72.0) * (T.one() + e2 / T.constant(110.0))
            )
        )
    )
    var big = max_of(ae, T.constant(0.1))
    var direct = (big.exp() - (-big).exp()) / (T.constant(2.0) * big)
    return blend(near, series, direct)


def _cosh[T: FloatLike](e: T) -> T:
    return (e.exp() + (-e).exp()) / T.constant(2.0)


def _cf1_j[T: FloatLike](v: T, x: T) -> Tuple[T, T]:
    """`J_{v+1}(x) / J_v(x)` by `x / (2(v+1) - x^2 / (2(v+2) - ...))` from
    the tail at depth 260, and the sign of `J_v`: `J_{v+260}(x)` is
    positive for `x <= 200`, and each level's denominator carries the sign
    of one ratio `J_{v+k} / J_{v+k-1}`."""
    var t = T.constant(0.0)
    var sign = T.one()
    comptime for step in range(_CF1_J_DEPTH):
        comptime k = _CF1_J_DEPTH - step
        var den = guard_nonzero(
            T.constant(Float64(2 * k)) + T.constant(2.0) * v - x * t,
            T.constant(_TINY),
        )
        sign = sign * T.one().copysign(den)
        t = x / den
    return (t^, sign^)


def _cf1_i[T: FloatLike](v: T, x: T) -> T:
    """`I_{v+1}(x) / I_v(x)` by `x / (2(v+1) + x^2 / (2(v+2) + ...))` from
    the tail at depth 100; every denominator is positive."""
    var t = T.constant(0.0)
    comptime for step in range(_CF1_I_DEPTH):
        comptime k = _CF1_I_DEPTH - step
        t = x / (T.constant(Float64(2 * k)) + T.constant(2.0) * v + x * t)
    return t^


def _chain_down[
    T: FloatLike, modified: Bool
](v: T, x: T, ratio: T, sign: T) -> Tuple[T, T, T, T]:
    """From `(f_v, f_{v+1}) = (sign, sign * ratio)`, recur the pair down
    `nl = floor(v + 1/2)` orders to `mu`, renormalizing to unit L1 norm at
    every step so nothing overflows. Thirty steps always run; step `k` is
    taken where `nl >= k + 1` and held elsewhere. Returns `(f_mu, f_{mu+1},
    f_v, nl)` with `f_v` rescaled alongside, so any one true value fixes
    them all."""
    # `ceil(v - 1/2)`, not `floor(v + 1/2)`: both put `mu` within `1/2` of
    # zero, but the half-integers -- every spherical Bessel order -- land
    # on `mu = +1/2` this way and `-1/2` the other, and at `-1/2` Temme's
    # series produces `Y_{-1/2} = J_{1/2}`, the small solution, as the
    # difference of two terms of size `(2/x)^{1/2}`: nine digits gone at
    # `x = 1e-9`. At `+1/2` the large solution is what it computes.
    var nl = -((T.constant(0.5) - v).floor())
    var xi = T.one() / x
    var a = sign.copy()
    var b = sign * ratio
    var saved = sign.copy()
    var order = v.copy()
    comptime for step in range(_ORDER_STEPS):
        var take = ge_indicator(nl, T.constant(Float64(step + 1)))
        var below: T
        comptime if modified:
            below = T.constant(2.0) * order * xi * a + b
        else:
            below = T.constant(2.0) * order * xi * a - b
        var new_b = blend(take, a, b)
        var new_a = blend(take, below, a)
        a = new_a^
        b = new_b^
        order = order - take
        var scale = a.abs() + b.abs()
        a = a / scale
        b = b / scale
        saved = saved / scale
    return (a^, b^, saved^, nl^)


def _temme_y[T: FloatLike](mu: T, x: T) -> Tuple[T, T]:
    """Temme's series for `(Y_mu(x), Y_{mu+1}(x))`, `|mu| <= 1/2`, `x <=
    2`; twenty terms, the last `1/(20!)^2`."""
    var x2 = x / T.constant(2.0)
    var pimu = T.constant(_PI) * mu
    var pg = guard_nonzero(pimu, T.constant(_TINY))
    var fact = pg / pg.sin()
    var d = -(x2.ln())
    var e = mu * d
    var fact2 = _sinhc(e)
    var g = _gamma_pair(mu)
    var ff = (
        T.constant(_TWO_OVER_PI) * fact * (g[0] * _cosh(e) + g[1] * fact2 * d)
    )
    var ee = e.exp()
    var p = ee / (g[2] * T.constant(_PI))
    var q = T.one() / (ee * T.constant(_PI) * g[3])
    var pimu2 = pimu / T.constant(2.0)
    var pg2 = guard_nonzero(pimu2, T.constant(_TINY))
    var fact3 = pg2.sin() / pg2
    var r = T.constant(_PI) * pimu2 * fact3 * fact3
    var c = T.one()
    var dd = -(x2 * x2)
    var total = ff + r * q
    var total1 = p.copy()
    var mu2 = mu * mu
    comptime for i in range(1, _TEMME_TERMS + 1):
        ff = (T.constant(Float64(i)) * ff + p + q) / (
            T.constant(Float64(i * i)) - mu2
        )
        c = c * dd / T.constant(Float64(i))
        p = p / (T.constant(Float64(i)) - mu)
        q = q / (T.constant(Float64(i)) + mu)
        var delta = c * (ff + r * q)
        total = total + delta
        total1 = total1 + (c * p - T.constant(Float64(i)) * delta)
    return (-total, -total1 * T.constant(2.0) / x)


def _cf2_jy[T: FloatLike](mu: T, x: T) -> Tuple[T, T]:
    """Steed's continued fraction for `p + i q = H'_mu / H_mu` (the Hankel
    function `J + iY`) at `x >= 2`, over `Complex[T]` from the tail at
    depth 60."""
    var mu2 = mu * mu
    var t = Complex[T](T.constant(0.0), T.constant(0.0))
    comptime for step in range(_CF2_JY_DEPTH):
        comptime k = _CF2_JY_DEPTH - step
        comptime half_sq = (Float64(k) - 0.5) * (Float64(k) - 0.5)
        var a = Complex[T](T.constant(half_sq) - mu2, T.constant(0.0))
        var b = Complex[T](T.constant(2.0) * x, T.constant(Float64(2 * k)))
        t = a / (b + t)
    var xi = T.one() / x
    # -1/(2x) + i + (i/x) t
    var p = -(xi / T.constant(2.0)) - t.im * xi
    var q = T.one() + t.re * xi
    return (p^, q^)


def _temme_k[T: FloatLike](mu: T, x: T) -> Tuple[T, T]:
    """Temme's series for `(K_mu(x), K_{mu+1}(x))`, `|mu| <= 1/2`, `x <= 2`."""
    var x2 = x / T.constant(2.0)
    var pimu = T.constant(_PI) * mu
    var pg = guard_nonzero(pimu, T.constant(_TINY))
    var fact = pg / pg.sin()
    var d = -(x2.ln())
    var e = mu * d
    var fact2 = _sinhc(e)
    var g = _gamma_pair(mu)
    var ff = fact * (g[0] * _cosh(e) + g[1] * fact2 * d)
    var total = ff.copy()
    var ee = e.exp()
    var p = ee / (T.constant(2.0) * g[2])
    var q = T.one() / (T.constant(2.0) * ee * g[3])
    var c = T.one()
    var dd = x2 * x2
    var total1 = p.copy()
    var mu2 = mu * mu
    comptime for i in range(1, _TEMME_TERMS + 1):
        ff = (T.constant(Float64(i)) * ff + p + q) / (
            T.constant(Float64(i * i)) - mu2
        )
        c = c * dd / T.constant(Float64(i))
        p = p / (T.constant(Float64(i)) - mu)
        q = q / (T.constant(Float64(i)) + mu)
        total = total + c * ff
        total1 = total1 + c * (p - T.constant(Float64(i)) * ff)
    return (total^, total1 * T.constant(2.0) / x)


def _cf2_k[T: FloatLike](mu: T, x: T) -> Tuple[T, T]:
    """Temme's continued fraction for `K` at `x >= 2`, run forward for a
    fixed 100 levels: returns `(h, s)` with `K_mu = sqrt(pi / 2x) e^{-x} /
    s` and `K_{mu+1} = K_mu (mu + x + 1/2 - h) / x`.

    Numerical Recipes' `bessik` carries an auxiliary `q` and the factor
    `c = prod a_k / k!` separately and adds `q * delh` to `s`; past the
    point where `s` has converged (level 10 at `x = 60`) `q` keeps growing
    like `(2x)^k / k!` while `delh` shrinks faster, so at a fixed depth of
    100 `q` passes `1e69` -- harmless in float64, `inf * 0 = NaN` in
    float32, where this was first seen as `iv(1, x)` returning NaN on
    Metal. Every quantity here is therefore carried already multiplied by
    `delh`: `U = c q_k delh`, `W = c q_{k-1} delh`, `T = q delh`, all of
    which decay.
    """
    var b = T.constant(2.0) * (T.one() + x)
    var d = T.one() / b
    var h = d.copy()
    var delh = d.copy()
    var a1 = T.constant(0.25) - mu * mu
    var a = -a1
    var u = a1 * d
    var w = T.constant(0.0)
    var t = a1 * d
    var s = T.one() + t
    comptime for i in range(2, _CF2_K_DEPTH + 1):
        a = a - T.constant(Float64(2 * (i - 1)))
        var b_old = b.copy()
        b = b + T.constant(2.0)
        d = T.one() / guard_nonzero(b + a * d, T.constant(_TINY))
        var rho = b * d - T.one()
        var u_new = -rho * (w - b_old * u) / T.constant(Float64(i))
        w = -rho * a * u / T.constant(Float64(i))
        u = u_new^
        t = rho * t + u
        delh = rho * delh
        h = h + delh
        s = s + t
    return (a1 * h, s^)


def _bessel_jy[T: FloatLike](v: T, x: T) -> Tuple[T, T]:
    """`(J_v(x), Y_v(x))` for `0 <= v <= 30`, `0 < x <= 200`; the module
    docstring has the algorithm."""
    var xs = max_of(x, T.constant(_TINY))
    var xi = T.one() / xs
    var cf = _cf1_j(v, xs)
    var chain = _chain_down[T, False](v, xs, cf[0], cf[1])
    var jmu = chain[0].copy()
    var jmu1 = chain[1].copy()
    var jv_scaled = chain[2].copy()
    var nl = chain[3].copy()
    var mu = v - nl

    var near = ge_indicator(T.constant(2.0), xs)
    var small = min_of(xs, T.constant(2.0))
    var large = max_of(xs, T.constant(2.0))

    # `x < 2`: Temme's `Y` pair, and the Wronskian fixes the `J` chain.
    var ty = _temme_y(mu, small)
    var c_small = (T.constant(_TWO_OVER_PI) / small) / guard_nonzero(
        jmu1 * ty[0] - jmu * ty[1], T.constant(_TINY)
    )

    # `x >= 2`: `p + iq`, with `f = J'/J = mu/x - J_{mu+1}/J_mu`, gives
    # `J_mu = sqrt(W q / ((p - f)^2 + q^2))` and `Y_mu = J_mu (p - f) / q`;
    # written with `A = (p - mu/x) j_mu + j_{mu+1} = (p - f) j_mu` so no
    # step divides by `j_mu`.
    var pq = _cf2_jy(mu, large)
    var p = pq[0].copy()
    var q = pq[1].copy()
    var big_a = (p - mu / large) * jmu + jmu1
    var c_large = (
        T.constant(_TWO_OVER_PI)
        / large
        * q
        / (big_a * big_a + q * q * jmu * jmu)
    ).sqrt()
    var ymu_large = big_a / q * c_large
    var jmu_large = jmu * c_large
    var ymup_large = q * jmu_large + p * ymu_large
    var ymu1_large = mu / large * ymu_large - ymup_large

    var c = blend(near, c_small, c_large)
    var j = jv_scaled * c

    # `Y` upward from `mu` to `v`, the same held-or-taken thirty steps.
    var a = blend(near, ty[0], ymu_large)
    var b = blend(near, ty[1], ymu1_large)
    var order = mu.copy()
    comptime for step in range(_ORDER_STEPS):
        var take = ge_indicator(nl, T.constant(Float64(step + 1)))
        var above = T.constant(2.0) * (order + T.one()) * xi * b - a
        var new_a = blend(take, b, a)
        var new_b = blend(take, above, b)
        a = new_a^
        b = new_b^
        order = order + take
    return (j^, a^)


def _bessel_ik[T: FloatLike](v: T, x: T) -> Tuple[T, T]:
    """`(I_v(x) e^{-x}, K_v(x) e^{x})` for `0 <= v <= 30`, `0 < x <= 200`:
    the scaled pair, so neither overflows before `x = 200`."""
    var xs = max_of(x, T.constant(_TINY))
    var xi = T.one() / xs
    var ratio = _cf1_i(v, xs)
    var chain = _chain_down[T, True](v, xs, ratio, T.one())
    var imu = chain[0].copy()
    var imu1 = chain[1].copy()
    var iv_scaled = chain[2].copy()
    var nl = chain[3].copy()
    var mu = v - nl

    var near = ge_indicator(T.constant(2.0), xs)
    var small = min_of(xs, T.constant(2.0))
    var large = max_of(xs, T.constant(2.0))

    var tk = _temme_k(mu, small)
    var es = small.exp()
    var hs = _cf2_k(mu, large)
    var kmu_large = (T.constant(_HALF_PI) / large).sqrt() / hs[1]
    var kmu1_large = kmu_large * (mu + large + T.constant(0.5) - hs[0]) / large
    var kmu = blend(near, tk[0] * es, kmu_large)
    var kmu1 = blend(near, tk[1] * es, kmu1_large)

    # Wronskian `I_mu K_{mu+1} + I_{mu+1} K_mu = 1/x`, every term positive.
    var c = xi / (imu1 * kmu + imu * kmu1)
    var i_scaled = iv_scaled * c

    var a = kmu^
    var b = kmu1^
    var order = mu.copy()
    comptime for step in range(_ORDER_STEPS):
        var take = ge_indicator(nl, T.constant(Float64(step + 1)))
        var above = T.constant(2.0) * (order + T.one()) * xi * b + a
        var new_a = blend(take, b, a)
        var new_b = blend(take, above, b)
        a = new_a^
        b = new_b^
        order = order + take
    return (i_scaled^, a^)


def _parity_sign[T: FloatLike](n: T) -> T:
    """`(-1)^n` for an integer-valued `n >= 0`."""
    var half = (n / T.constant(2.0)).floor()
    return T.one() - T.constant(2.0) * (n - T.constant(2.0) * half)


def _order_trig[T: FloatLike](av: T) -> Tuple[T, T]:
    """`(cos(pi v), sin(pi v))` for `v >= 0`, from the fractional part of
    `v` so both are exact where `v` is an integer -- `sin(2 pi)` evaluated
    directly is `-2.4e-16`, and multiplied by a `Y_v` of `1e4` that is a
    visible error in `J_{-v}`."""
    var n = (av + T.constant(0.5)).floor()
    var frac = T.constant(_PI) * (av - n)
    var s = _parity_sign(n)
    return (s * frac.cos(), s * frac.sin())


def jv[T: FloatLike](v: T, x: T) -> T:
    """`J_v(x)`, the Bessel function of the first kind of real order `v`,
    for `|v| <= 30` and `0 < x <= 200`. `scipy.special.jv(v, x)`.

    Negative orders through `J_{-v} = cos(pi v) J_v - sin(pi v) Y_v`, so
    `jv(-2, x)` is `jv(2, x)` exactly. The module docstring has the
    algorithm and the bound.
    """
    var av = v.abs()
    var jy = _bessel_jy(av, x)
    var trig = _order_trig(av)
    var reflected = trig[0] * jy[0] - trig[1] * jy[1]
    return blend(ge_indicator(v, T.constant(0.0)), jy[0], reflected)


def yv[T: FloatLike](v: T, x: T) -> T:
    """`Y_v(x)`, the Bessel function of the second kind of real order `v`,
    for `|v| <= 30` and `0 < x <= 200`. `scipy.special.yv(v, x)`.

    Negative orders through `Y_{-v} = sin(pi v) J_v + cos(pi v) Y_v`.
    """
    var av = v.abs()
    var jy = _bessel_jy(av, x)
    var trig = _order_trig(av)
    var reflected = trig[1] * jy[0] + trig[0] * jy[1]
    return blend(ge_indicator(v, T.constant(0.0)), jy[1], reflected)


def ive[T: FloatLike](v: T, x: T) -> T:
    """`I_v(x) e^{-x}`, the exponentially scaled modified Bessel function
    of the first kind, for `|v| <= 30` and `0 < x <= 200`.
    `scipy.special.ive(v, x)`. The scaled form is what the algorithm
    produces; `iv` multiplies the exponential back on.

    Negative orders through `I_{-v} = I_v + (2/pi) sin(pi v) K_v`.
    """
    var av = v.abs()
    var ik = _bessel_ik(av, x)
    var trig = _order_trig(av)
    var reflected = (
        ik[0]
        + T.constant(_TWO_OVER_PI)
        * trig[1]
        * ik[1]
        * (-(T.constant(2.0) * x)).exp()
    )
    return blend(ge_indicator(v, T.constant(0.0)), ik[0], reflected)


def kve[T: FloatLike](v: T, x: T) -> T:
    """`K_v(x) e^{x}`, the exponentially scaled modified Bessel function of
    the second kind, for `|v| <= 30` and `0 < x <= 200`.
    `scipy.special.kve(v, x)`. `K_{-v} = K_v`."""
    return _bessel_ik(v.abs(), x)[1].copy()


def iv[T: FloatLike](v: T, x: T) -> T:
    """`I_v(x)`, the modified Bessel function of the first kind of real
    order `v`, for `|v| <= 30` and `0 < x <= 200`. `scipy.special.iv(v,
    x)`. `ive(v, x) e^{x}`; overflows where `e^x` does."""
    return ive(v, x) * x.exp()


def kv[T: FloatLike](v: T, x: T) -> T:
    """`K_v(x)`, the modified Bessel function of the second kind of real
    order `v`, for `|v| <= 30` and `0 < x <= 200`. `scipy.special.kv(v,
    x)`. `kve(v, x) e^{-x}`."""
    return kve(v, x) * (-x).exp()


def spherical_jn[T: FloatLike](n: Int, x: T) -> T:
    """The spherical Bessel function `j_n(x) = sqrt(pi / 2x) J_{n+1/2}(x)`
    for integer `0 <= n <= 29` and `0 <= x <= 200`.
    `scipy.special.spherical_jn(n, x)`. `x` is floored at `1e-30`, so
    `spherical_jn(0, 0)` is `1`."""
    var xs = max_of(x, T.constant(_TINY))
    var v = T.constant(Float64(n) + 0.5)
    return (T.constant(_HALF_PI) / xs).sqrt() * _bessel_jy(v, xs)[0]


def spherical_yn[T: FloatLike](n: Int, x: T) -> T:
    """The spherical Bessel function `y_n(x) = sqrt(pi / 2x) Y_{n+1/2}(x)`
    for integer `0 <= n <= 29` and `0 < x <= 200`.
    `scipy.special.spherical_yn(n, x)`."""
    var xs = max_of(x, T.constant(_TINY))
    var v = T.constant(Float64(n) + 0.5)
    return (T.constant(_HALF_PI) / xs).sqrt() * _bessel_jy(v, xs)[1]
