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
< 1`. For `x < 0` Pfaff's transformation `2F1(a, b, c, x) = (1 - x)^{-a}
2F1(a, c - b, c, x / (x - 1))` maps `(-inf, 0)` onto `(0, 1)` with a
positive argument, so the alternating cancellation never happens; for `0
<= x < 1` the series is summed directly. Four hundred terms are exact to
double precision for `x <= 0.9` and lose accuracy as `x -> 1` (the tail
is `x^400`, `4e-5` at `x = 0.975`), which is the stated ceiling: SciPy's
`1 - x` transformations for that last corner need `Gamma` at `c - a - b`,
which has poles at the integers the common cases sit on, and a blend
cannot carry a pole on its unselected side. `|x| > 1` is outside the
series' domain and returns whatever the diverging sum reaches, as
documented rather than checked.

## The MAX gate

Nothing: MAX has no hypergeometric functions. **Extend.**
"""

from ..core.numeric import FloatLike, blend, ge_indicator, max_of, min_of

comptime _1F1_TERMS = 200
comptime _2F1_TERMS = 400


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


def hyp2f1[T: FloatLike](a: T, b: T, c: T, x: T) -> T:
    """Gauss's hypergeometric function `2F1(a, b; c; x)` for `|x| < 1` and
    `c` not a non-positive integer. `scipy.special.hyp2f1(a, b, c, x)`.

    The series directly for `0 <= x < 1` and through Pfaff's
    transformation for `x < 0`, blended by the sign of `x`. Exact to double
    precision for `x <= 0.9` and every `x < 0`; the module docstring
    states the ceiling as `x -> 1`.
    """
    var positive = ge_indicator(x, T.constant(0.0))
    var forward = max_of(x, T.constant(0.0))
    var backward = min_of(x, T.constant(0.0))
    var direct = _series_2f1(a, b, c, forward)
    # Pfaff: (1 - x)^{-a} 2F1(a, c - b; c; x / (x - 1)), argument in [0, 1).
    var one_minus = T.one() - backward
    var mapped = backward / (backward - T.one())
    var reflected = (-(a * one_minus.ln())).exp() * _series_2f1(
        a, c - b, c, mapped
    )
    return blend(positive, direct, reflected)
