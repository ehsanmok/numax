"""The standard normal distribution function, its logarithm and its
inverse: `ndtr`, `log_ndtr` and `ndtri`, SciPy's.

**This module is tier 1.** Each is a fixed set of regions evaluated in
full and blended, every region fed an argument clamped into its own
range, so a SIMD vector whose lanes fall in different regions is one
evaluation and the functions run inside a GPU kernel.

- `ndtr(x)` is `erfc(-x / sqrt(2)) / 2`, read off `erfc` so the lower
  tail keeps its relative precision where `(1 + erf) / 2` would cancel.
- `log_ndtr(x)` has three regions, SciPy's: `log1p(-ndtr(-x))` above 0,
  where `ndtr` is within rounding of 1; `log(ndtr(x))` on `[-20, 0]`;
  and below `-20` (`-10` at `float32`, where `ndtr` underflows first)
  the asymptotic series of `log(erfc)`,
  `-x^2/2 - log(-x) - log(2 pi)/2 + log(1 + sum_k (-1)^k (2k-1)!! / x^2k)`
  to ten terms, whose truncation there is below `1e-17`, so it stays
  finite far past where `ndtr` underflows.
- `ndtri(p)` is Wichura's AS 241 (`PPND16`, *Applied Statistics* 37,
  1988), three rational minimax fits: one in `p - 1/2` for
  `|p - 1/2| <= 0.425`, and two in `r = sqrt(-log(min(p, 1 - p)))`
  split at `r = 5`, together good to about `1e-16` relative for every
  `p` down to `exp(-729)`. It is the one piece of the family that is
  not built on `erfc`, which is why it keeps full precision in the far
  tail where `numax.special.erfcinv` does not. `0` and `1` map to `-inf`
  and `inf`, and anything outside `[0, 1]` to NaN.

## The MAX gate

Nothing: MAX has no normal distribution function. `std.math` has `erfc`,
which `ndtr` is. **Extend.**
"""

from ..core.numeric import FloatLike, blend, ge_indicator, max_of, min_of
from .information import _log1p

comptime _SQRT1_2 = 0.7071067811865476
comptime _HALF_LN_2PI = 0.9189385332046728


def ndtr[T: FloatLike](x: T) -> T:
    """The standard normal CDF, `P(Z <= x)`. `scipy.special.ndtr(x)`.

    `erfc(-x / sqrt(2)) / 2`, SciPy's own form. `erfc` itself is a few
    ulp at `Plain`, but the rounding of `x / sqrt(2)` is amplified by
    `erfc`'s slope to a relative error near `x^2 eps` in the lower tail --
    `pixi run accuracy` reads `5.6e-15` over `[-30, 8]` -- as it is in
    SciPy.

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the input.

    Args:
        x: The point to evaluate at.

    Returns:
        `Phi(x)`, in `[0, 1]`.
    """
    return T.constant(0.5) * (-(x * T.constant(_SQRT1_2))).erfc()


def log_ndtr[T: FloatLike](x: T) -> T:
    """The logarithm of the standard normal CDF. `scipy.special.log_ndtr(x)`.

    Three regions blended, per this module's docstring: `log1p(-ndtr(-x))`
    for `x > 0`, `log(ndtr(x))` on `[-20, 0]`, and the asymptotic series
    below `-20`, which carries the result to any finite `x` where `ndtr`
    itself underflows below `x = -38`.

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the input.

    Args:
        x: The point to evaluate at.

    Returns:
        `log(Phi(x))`, at most `0`.
    """
    var zero = T.constant(0.0)
    # The series takes over at -20, SciPy's split, where `float64`'s
    # `ndtr` is still `3e-89`. `float32`'s underflows near -14, so a
    # narrow conformer -- one where `1 + 2^-30` rounds to 1 -- switches at
    # -10 instead, where the ten-term series is still good to `1e-11`.
    var narrow = ge_indicator(
        zero, (T.one() + T.constant(9.313225746154785e-10)) - T.one()
    )
    var cut = blend(narrow, T.constant(-10.0), T.constant(-20.0))
    var positive = ge_indicator(x, zero)
    var deep = T.one() - ge_indicator(x, cut)

    # Above 0: `ndtr(-x)` is at most 1/2, so `log1p` of its negative is
    # finite; the argument is held at 0 for the lanes below it.
    var xa = max_of(x, zero)
    var upper = _log1p(-ndtr(-xa))

    # The middle: `ndtr` of an argument held at -20 or above never
    # underflows, so its logarithm is finite on every lane.
    var middle = ndtr(max_of(x, cut)).ln()

    # Below -20: the series in `1 / x^2`, at `x` held at -20 or below.
    var xc = min_of(x, cut)
    var inv2 = T.one() / (xc * xc)
    var term = T.one()
    var total = T.one()
    comptime for k in range(1, 11):
        term = -term * T.constant(Float64(2 * k - 1)) * inv2
        total = total + term
    var tail = (
        -(T.constant(0.5) * xc * xc)
        - (-xc).ln()
        - T.constant(_HALF_LN_2PI)
        + total.ln()
    )
    return blend(positive, upper, blend(deep, tail, middle))


def _as241_central[T: FloatLike](q: T) -> T:
    """AS 241's central fit, `q R(0.180625 - q^2)` for `|q| <= 0.425`."""
    var r = T.constant(0.180625) - q * q
    var num = T.constant(2509.0809287301226727)
    num = num * r + T.constant(33430.575583588128105)
    num = num * r + T.constant(67265.770927008700853)
    num = num * r + T.constant(45921.953931549871457)
    num = num * r + T.constant(13731.693765509461125)
    num = num * r + T.constant(1971.5909503065514427)
    num = num * r + T.constant(133.14166789178437745)
    num = num * r + T.constant(3.387132872796366608)
    var den = T.constant(5226.495278852854561)
    den = den * r + T.constant(28729.085735721942674)
    den = den * r + T.constant(39307.89580009271061)
    den = den * r + T.constant(21213.794301586595867)
    den = den * r + T.constant(5394.1960214247511077)
    den = den * r + T.constant(687.1870074920579083)
    den = den * r + T.constant(42.313330701600911252)
    den = den * r + T.one()
    return q * num / den


def _as241_near[T: FloatLike](r: T) -> T:
    """AS 241's first tail fit, for `r = sqrt(-log p)` in `(0, 5]`."""
    var s = r - T.constant(1.6)
    var num = T.constant(7.7454501427834140764e-4)
    num = num * s + T.constant(0.0227238449892691845833)
    num = num * s + T.constant(0.24178072517745061177)
    num = num * s + T.constant(1.27045825245236838258)
    num = num * s + T.constant(3.64784832476320460504)
    num = num * s + T.constant(5.7694972214606914055)
    num = num * s + T.constant(4.6303378461565452959)
    num = num * s + T.constant(1.42343711074968357734)
    var den = T.constant(1.05075007164441684324e-9)
    den = den * s + T.constant(5.475938084995344946e-4)
    den = den * s + T.constant(0.0151986665636164571966)
    den = den * s + T.constant(0.14810397642748007459)
    den = den * s + T.constant(0.68976733498510000455)
    den = den * s + T.constant(1.6763848301838038494)
    den = den * s + T.constant(2.05319162663775882187)
    den = den * s + T.one()
    return num / den


def _as241_far[T: FloatLike](r: T) -> T:
    """AS 241's second tail fit, for `r` in `(5, 27]`."""
    var s = r - T.constant(5.0)
    var num = T.constant(2.01033439929228813265e-7)
    num = num * s + T.constant(2.71155556874348757815e-5)
    num = num * s + T.constant(0.0012426609473880784386)
    num = num * s + T.constant(0.026532189526576123093)
    num = num * s + T.constant(0.29656057182850489123)
    num = num * s + T.constant(1.7848265399172913358)
    num = num * s + T.constant(5.4637849111641143699)
    num = num * s + T.constant(6.6579046435011037772)
    var den = T.constant(2.04426310338993978564e-15)
    den = den * s + T.constant(1.4215117583164458887e-7)
    den = den * s + T.constant(1.8463183175100546818e-5)
    den = den * s + T.constant(7.868691311456132591e-4)
    den = den * s + T.constant(0.0148753612908506148525)
    den = den * s + T.constant(0.13692988092273580531)
    den = den * s + T.constant(0.59983220655588793769)
    den = den * s + T.one()
    return num / den


def ndtri[T: FloatLike](p: T) -> T:
    """The inverse of the standard normal CDF: the `x` with
    `ndtr(x) == p`. `scipy.special.ndtri(p)`.

    Wichura's AS 241, per this module's docstring: about `1e-16` relative
    for every `p` in `[exp(-729), 1)`, the tails included. `ndtri(0)` is
    `-inf` and `ndtri(1)` is `inf`, as SciPy's; outside `[0, 1]` the
    result is NaN.

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the input.

    Args:
        p: The probability, in `[0, 1]`.

    Returns:
        `Phi^-1(p)`.
    """
    var zero = T.constant(0.0)
    var half = T.constant(0.5)
    var limit = T.constant(0.425)
    var q = p - half
    var central = ge_indicator(limit, q.abs())
    var inner = _as241_central(max_of(min_of(q, limit), -limit))

    # The tail's distance from the boundary is `min(p, 1 - p)` formed
    # directly -- `1/2 - |q|` would lose a small `p` to `p - 1/2`. At
    # exactly 0 it is replaced by `1e-30` before the logarithm, which only
    # has to give the edge lane a finite, positive magnitude for the
    # division below to sign; it is not a floor under small `p`, since a
    # floor small enough for `float64` rounds to 0 at `float32`, and the
    # infinite `r` that follows turns `min_of` into NaN.
    var p_min = min_of(p, T.one() - p)
    var at_edge = ge_indicator(zero, p_min.abs())
    var r = (-(p_min + at_edge * T.constant(1e-30)).ln()).sqrt()
    var far = ge_indicator(r, T.constant(5.0))
    var near_value = _as241_near(min_of(r, T.constant(5.0)))
    var far_value = _as241_far(
        max_of(min_of(r, T.constant(27.0)), T.constant(5.0))
    )
    var magnitude = blend(far, far_value, near_value)
    var x = blend(central, inner, T.one().copysign(q) * magnitude)

    # `p` at exactly 0 or 1 divides by zero to the signed infinity, and a
    # `p` outside `[0, 1]` makes `sqrt(p_min)` NaN; both are added as
    # terms, not blended, so they reach the result rather than being
    # multiplied by an indicator's zero.
    return x / (T.one() - at_edge) + zero * p_min.sqrt()
