"""The confluent and Gauss hypergeometric functions, `hyp1f1(a, b, x)` and
`hyp2f1(a, b, c, x)`. `scipy.special.hyp1f1` and `hyp2f1`.

**This module is tier 1.** Both are their defining series to a fixed term
count, with one transformation each that turns the argument range where
the series cancels into one where every term has the same sign -- and the
two sides blended by the sign of `x` rather than branched on.

`hyp1f1(a, b, x) = sum (a)_k / (b)_k x^k / k!` converges for every `x`,
but for `x < 0` its terms alternate and grow before they shrink, and the
double-precision sum loses most of its digits past `x ~ -10`. Kummer's
transformation `1F1(a, b, x) = e^x 1F1(b - a, b, -x)` moves the argument
to `|x|`, where the terms are positive, so the negative side is `e^{-|x|}`
times a well-behaved series. Two hundred terms cover `|x| <= 100`.

`hyp2f1(a, b, c, x) = sum (a)_k (b)_k / (c)_k x^k / k!` converges for `|x|
< 1`, and has three arms rather than two. For `x < 0` Pfaff's
transformation `2F1(a, b, c, x) = (1 - x)^{-a} 2F1(a, c - b, c, x / (x -
1))` maps `(-inf, 0)` onto `(0, 1)` with a positive argument, so the
alternating cancellation never happens. On `[0, 0.9)` the series is summed
directly; four hundred terms are exact to double precision there, and lose
accuracy as `x -> 1` (the tail is `x^400`, `4e-5` at `x = 0.975`). On
`[0.9, 1)` that tail is replaced by Abramowitz and Stegun 15.3.6, which
runs two series at `1 - x` -- an argument of at most `0.1`, so their tails
are nothing -- against the `Gamma` ratios `G(c)G(d)/(G(c-a)G(c-b))` and
`G(c)G(-d)/(G(a)G(b))` with `d = c - a - b`.

The pole in `d` is what kept 15.3.6 out until now, and what remains of it
is the module's one ceiling. `G(d)` and `G(-d)` pole together at every
integer `d`, and a blend evaluates both sides everywhere, so the arm has
to be finite even where nobody wants it. Two guards do that: `d` is nudged
by `1e-5` whenever it is within `1e-6` of an integer, which keeps every
`Gamma` finite on the discarded side, and within that same `1e-6` the
*direct* series stays selected, so an integer `d` past `x = 0.9` is
answered at the series' accuracy and not at 15.3.6's. That is the residual
ceiling: the limit case, where the two terms cancel and a logarithm
appears, is not implemented. `|x| > 1` is outside the series' domain; the
direct arm's argument is capped at `0.99` to keep the discarded side
finite, so nothing diverges and nothing is checked.

## The MAX gate

Nothing: MAX has no hypergeometric functions. **Extend.**
"""

from ..core.numeric import FloatLike, blend, ge_indicator, max_of, min_of
from .gamma import gammasgn, lgamma

comptime _1F1_TERMS = 200
comptime _2F1_TERMS = 400

# Where `hyp2f1` leaves the direct series for the `1 - x` transformation,
# and the two clamps that keep the *unselected* side of that blend finite
# (both sides are always evaluated, and `0 * inf` is `NaN`).
comptime _2F1_SWITCH = 0.9
comptime _2F1_X_CAP = 0.99
comptime _2F1_Y_CAP = 0.1
comptime _2F1_Y_FLOOR = 1e-30

# `d = c - a - b` sits on a `Gamma` pole at every integer. Within
# `_2F1_D_TOL` of one the direct series stays selected; the transformation's
# own `d` is nudged by `_2F1_D_NUDGE` regardless, so the arm nobody selected
# still evaluates to a finite number.
comptime _2F1_D_TOL = 1e-6
comptime _2F1_D_NUDGE = 1e-5

# `exp` of anything past this overflows `float32`, so every exponent built
# from a `Gamma` ratio is capped here rather than allowed to reach `inf`.
comptime _2F1_LOG_CAP = 80.0


def _series_1f1[T: FloatLike](a: T, b: T, x: T) -> T:
    """`sum_{k<terms} (a)_k / (b)_k x^k / k!`, the term carried as a
    running product."""
    var term = T.one()
    var total = T.one()
    comptime for k in range(_1F1_TERMS):
        term = (
            term
            * (a + T.constant(Float64(k)))
            / (b + T.constant(Float64(k)))
            * x
            / T.constant(Float64(k + 1))
        )
        total = total + term
    return total^


def hyp1f1[T: FloatLike](a: T, b: T, x: T) -> T:
    """Kummer's confluent hypergeometric function `1F1(a; b; x) = M(a, b,
    x)`, for `b` not a non-positive integer. `scipy.special.hyp1f1(a, b,
    x)`.

    The series at `|x|`, through Kummer's transformation on the negative
    side, blended by the sign of `x`; accurate to double precision for
    `|x| <= 100`, the module docstring has the reasoning.
    """
    var positive = ge_indicator(x, T.constant(0.0))
    var ax = x.abs()
    var direct = _series_1f1(a, b, ax)
    var reflected = (-ax).exp() * _series_1f1(b - a, b, ax)
    return blend(positive, direct, reflected)


def _series_2f1[T: FloatLike](a: T, b: T, c: T, x: T) -> T:
    """`sum_{k<terms} (a)_k (b)_k / (c)_k x^k / k!`."""
    var term = T.one()
    var total = T.one()
    comptime for k in range(_2F1_TERMS):
        term = (
            term
            * (a + T.constant(Float64(k)))
            * (b + T.constant(Float64(k)))
            / (c + T.constant(Float64(k)))
            * x
            / T.constant(Float64(k + 1))
        )
        total = total + term
    return total^


def _near_integer[T: FloatLike](d: T) -> T:
    """`1` where `d` lies within `_2F1_D_TOL` of an integer, `0` elsewhere
    -- branchless, off `floor`.

    `Gamma` has a pole at every non-positive integer and `hyp2f1`'s `1 - x`
    transformation needs `Gamma(d)` *and* `Gamma(-d)`, so an integer `d`
    poles both of them at once. This is the indicator that keeps the direct
    series selected there.
    """
    var frac = d - d.floor()
    var gap = min_of(frac, T.one() - frac)
    return T.one() - ge_indicator(gap, T.constant(_2F1_D_TOL))


def hyp2f1[T: FloatLike](a: T, b: T, c: T, x: T) -> T:
    """Gauss's hypergeometric function `2F1(a, b; c; x)` for `|x| < 1` and
    `c` not a non-positive integer. `scipy.special.hyp2f1(a, b, c, x)`.

    Three arms of one blend, selected by `x` alone: Pfaff's transformation
    for `x < 0`, the series directly on `[0, 0.9)`, and Abramowitz &
    Stegun 15.3.6's `1 - x` transformation on `[0.9, 1)`, where the series'
    `x^400` tail is no longer negligible. Accurate to double precision
    across all three, with one exception: 15.3.6 carries `Gamma(c-a-b)` and
    `Gamma(a+b-c)`, which pole together at every integer `d = c - a - b`,
    so within `1e-6` of an integer `d` the direct series stays selected and
    its tail is the error -- `4e-5` at `x = 0.975`, worse as `x -> 1`. The
    limit case that replaces 15.3.6 at integer `d` adds a logarithm and is
    not implemented; the module docstring has the reasoning.

    `pixi run accuracy` measures the new arm at `3.2e-15` relative over
    `[0.9, 0.999]` at `d = 0.5`, against the `4e-5` the series it replaced
    reached by `x = 0.975`.

    `|x| > 1` is outside the series' domain. The direct arm's argument is
    capped at `0.99` so the discarded side of the blend stays finite rather
    than reaching `inf` and turning `0 * inf` into `NaN`, which means a
    caller who passes `x > 1` gets the value at `0.99`, not a diverging
    sum. Neither is an answer; nothing here checks the domain.
    """
    var positive = ge_indicator(x, T.constant(0.0))
    var forward = min_of(max_of(x, T.constant(0.0)), T.constant(_2F1_X_CAP))
    var backward = min_of(x, T.constant(0.0))
    var direct = _series_2f1(a, b, c, forward)
    # Pfaff: (1 - x)^{-a} 2F1(a, c - b; c; x / (x - 1)), argument in [0, 1).
    var one_minus = T.one() - backward
    var mapped = backward / (backward - T.one())
    var reflected = (-(a * one_minus.ln())).exp() * _series_2f1(
        a, c - b, c, mapped
    )

    # A&S 15.3.6. Both series run at `1 - x`, clamped into `[tiny, 0.1]` so
    # the arm evaluates finitely at every `x`, not only the selecting ones.
    # The `Gamma` ratios go through `lgamma`/`gammasgn`: a sum of logarithms
    # cannot overflow on its way to a ratio that does not.
    var d = c - a - b
    var near = _near_integer(d)
    var ds = d + T.constant(_2F1_D_NUDGE) * near
    var y = min_of(
        max_of(T.one() - x, T.constant(_2F1_Y_FLOOR)),
        T.constant(_2F1_Y_CAP),
    )
    var log_c = lgamma(c)
    var sgn_c = gammasgn(c)

    var log_first = log_c + lgamma(ds) - lgamma(c - a) - lgamma(c - b)
    var first = (
        sgn_c
        * gammasgn(ds)
        * gammasgn(c - a)
        * gammasgn(c - b)
        * min_of(log_first, T.constant(_2F1_LOG_CAP)).exp()
        * _series_2f1(a, b, T.one() - ds, y)
    )

    # `(1 - x)^{c-a-b}` folded into the same exponent as its `Gamma` ratio.
    var log_second = log_c + lgamma(-ds) - lgamma(a) - lgamma(b) + ds * y.ln()
    var second = (
        sgn_c
        * gammasgn(-ds)
        * gammasgn(a)
        * gammasgn(b)
        * min_of(log_second, T.constant(_2F1_LOG_CAP)).exp()
        * _series_2f1(c - a, c - b, T.one() + ds, y)
    )

    var near_one = ge_indicator(x, T.constant(_2F1_SWITCH)) * (T.one() - near)
    return blend(positive, blend(near_one, first + second, direct), reflected)
