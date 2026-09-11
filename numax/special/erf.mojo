"""The error function, its complement, and their inverses.

**This module is tier 1.** Two forwarding calls to the trait method, and
two inverses at a fixed iteration count.

`erf`/`erfc` are `FloatLike` trait methods (see `numax.core.numeric`), not
freestanding formulas, so each conformer can supply whatever's fastest or
most precise for it: `Plain` delegates straight to `std.math.erf`/`erfc`
(a genuinely different, GPU-compatible implementation that doesn't share
`default_erf_approx`'s cancellation error -- see this module's history and
`tests/core/test_compensated.mojo`), `Dual` differentiates through the chain
rule over its `Inner`'s own `erf()`, and `Compensated`/`Decimal` fall back
to `numax.core.numeric.default_erf_approx`, the Abramowitz & Stegun 7.1.26
approximation this module used to implement directly for every type.

The free functions below just forward to those trait methods, so
`from numax import erf, erfc` keeps working the same way it always has.

`erfinv`/`erfcinv` are numax's own. `std.math` has no inverse error
function, and neither does any MAX root; the initial guess is Giles's
rational approximation and two Newton steps against the trait's own `erf`
finish it, so every conformer gets an inverse consistent with its own
forward function -- `Dual` differentiates it through the Newton steps and
comes out with `sqrt(pi)/2 * exp(erfinv(y)^2)`, the closed-form derivative.
"""

from ..core.numeric import FloatLike, blend, ge_indicator, max_of, min_of


def erf[T: FloatLike](x: T) -> T:
    """The error function, `(2/sqrt(pi)) * integral(exp(-t^2), 0, x)`."""
    return x.erf()


def erfc[T: FloatLike](x: T) -> T:
    """The complementary error function, `1 - erf(x)`."""
    return x.erfc()


def erfinv[T: FloatLike](y: T) -> T:
    """The inverse error function: the `x` with `erf(x) == y`, for `y` in
    `(-1, 1)`. `scipy.special.erfinv`.

    Tier 1. A starting guess, then three Newton steps against the trait's
    own `erf`/`erfc`, so the result is limited by those two functions'
    accuracy rather than the guess's -- at `Plain` that is `std.math`'s
    floor, about `2e-8` absolute for `erf` at `float64` (the inherited
    floor `findings.mdc` records), a few ULP for `erfc` in the tail.

    The guess has two regions on `w = -ln((1 - y)(1 + y))`. Below `w = 5`
    it is Giles's central polynomial (*Approximating the erfinv function*,
    GPU Computing Gems, 2010), good to about `1e-7`. Past it, it is the
    asymptotic inversion of `erfc(x) ~ exp(-x^2) / (x sqrt(pi))`: with
    `L = -ln(gap * sqrt(pi))`, `x ~ sqrt(L - ln(L) / 2)`, good to about
    `1e-3` uniformly however close to 1 `y` gets. Giles's own tail
    polynomial is *not* used: it is a `float32`-range fit, and past
    `w ~ 37` -- reachable at `float64` -- it extrapolates to `x ~ -82`,
    which Newton turns into an infinity. Three Newton steps take `1e-3` to
    below `1e-15`, where two would take `1e-7` there; the third is the
    price of the deep tail and is fixed work, not a tolerance.

    The two regions are blended, not branched: a `ge_indicator` on `w`
    picks the guess, and each side is evaluated at a *clamped* argument so
    both stay finite everywhere -- the central polynomial at `w` held
    below 5, the asymptotic form at `L` held above 2. That is what makes
    this launchable inside a GPU thread with lanes on both sides of the
    split.

    One clamp keeps every intermediate finite, and its placement is shaped
    by the `max_of`/`min_of` trap `findings.mdc` records: the *gap*
    `1 - |y|` is computed first, exactly, and held to at least `1e-30`;
    `|y|` is then recovered from it. Clamping `|y|` itself does not
    survive the blend -- `1 + (1 - 2^-53)` rounds to exactly `2` -- and
    the gap's lost bits become `x`'s error in the tail. So `erfinv(+-1)`
    returns `erfinv(+-(1 - 1e-30))`, about `+-8.15`, rather than the
    `+-inf` SciPy returns; a tier-1 kernel clamps where it cannot branch.
    Outside `[-1, 1]` the gap is negative, the clamp yields zero, and the
    result is NaN; nothing raises.

    The Newton residual is `erf(x) - |y|` in the central region and
    `gap - erfc(x)` in the tail, blended by the same indicator. Past about
    `x = 5.8` at `float64`, `erf(x)` is exactly `1` and can no longer tell
    two targets apart, while `erfc(x)` still carries the tail to full
    relative precision -- `std.math.erfc` is accurate to a few ULP out to
    `x = 6`. Near `y = 0` the roles reverse, which is why neither form is
    used alone.
    """
    var one = T.one()
    var two = T.constant(2.0)
    var five = T.constant(5.0)

    # `gap` first and exactly -- `1 - |y|` is exact for `|y|` in `[0.5, 1]`
    # -- and `mag` recovered from it, so the last bit of a `y` near 1 is
    # never rounded away by the clamp. Clamping `|y|` itself does not
    # survive `min_of`: `1 + (1 - 2^-53)` rounds to exactly `2`, and the
    # gap's relative error then becomes `x`'s.
    var sign_y = one.copysign(y)
    var gap = max_of(one - y.abs(), T.constant(1e-30))
    var mag = one - gap
    var w = -(gap * (one + mag)).ln()

    # Central polynomial at `w` held in `[0, 5]`; asymptotic inversion at
    # `L` held in `[2, inf)`; `blend` then picks by which side `w` is on.
    var in_tail = ge_indicator(w, five)
    var wc = min_of(w, five) - T.constant(2.5)

    var pc = T.constant(2.81022636e-08)
    pc = T.constant(3.43273939e-07) + pc * wc
    pc = T.constant(-3.5233877e-06) + pc * wc
    pc = T.constant(-4.39150654e-06) + pc * wc
    pc = T.constant(0.00021858087) + pc * wc
    pc = T.constant(-0.00125372503) + pc * wc
    pc = T.constant(-0.00417768164) + pc * wc
    pc = T.constant(0.246640727) + pc * wc
    pc = T.constant(1.50140941) + pc * wc

    var big_l = max_of(-(gap * T.constant(1.7724538509055159)).ln(), two)
    var x_tail = (big_l - T.constant(0.5) * big_l.ln()).sqrt()

    var x = blend(in_tail, x_tail, pc * mag)

    # Three Newton steps; `erf' = 2/sqrt(pi) exp(-x^2)`. The residual is
    # read off `erf` centrally and off `erfc` in the tail, where `erf` is 1.
    var half_sqrt_pi = T.constant(0.8862269254527580)
    comptime for _ in range(3):
        var central = x.erf() - mag
        var tail = gap - x.erfc()
        var residual = blend(in_tail, tail, central)
        x = x - residual * half_sqrt_pi * (x * x).exp()

    return x * sign_y


def erfcinv[T: FloatLike](y: T) -> T:
    """The inverse complementary error function: the `x` with
    `erfc(x) == y`, for `y` in `(0, 2)`. `scipy.special.erfcinv`.

    `erfinv(1 - y)`, and honestly so: this loses the digits `1 - y` loses,
    so for `y` below about `1e-8` the answer is `erfinv`'s clamped value
    rather than the large `x` SciPy would return. A tail formula in `y`
    itself is what would fix that; no caller in numax needs it yet.
    """
    return erfinv(T.one() - y)
