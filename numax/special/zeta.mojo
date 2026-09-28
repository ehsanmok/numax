"""The Riemann and Hurwitz zeta functions, `zeta(s)` and `zeta(s, q)`,
`scipy.special.zeta`, and the polygamma functions built on the Hurwitz
one, `polygamma(n, x)`.

**This module is tier 1.** One Euler-Maclaurin formula, at a fixed `N =
10` direct terms and `M = 8` Bernoulli corrections, evaluates the analytic
continuation on both sides of the pole at once: `sum_{k<N} (k+q)^{-s} +
(N+q)^{1-s} / (s-1) + (N+q)^{-s} / 2 + sum_{j<=M} B_{2j} / (2j)! s (s+1)
... (s+2j-2) (N+q)^{-s-2j+1}`. The `(N+q)^{1-s} / (s-1)` term is what makes
it valid below `s = 1` as well as above, and the only data-dependent hazard
is the pole at `s = 1`, where `1 / (s - 1)` is guarded away from `0 / 0`
and comes out infinite as it should.

Checked in mpmath before transcription at `N = 10, M = 8`: within `1e-18`
of `zeta(s)` for `s` in `[-2.5, 30]` and of `zeta(s, q)` at `q = 0.5` and
`q = 3`. Below `-2.5` the Bernoulli corrections grow and `M = 8` stops
being enough, so the **Riemann** overload leaves the formula there for
Riemann's functional equation, `zeta(s) = 2^s pi^{s-1} sin(pi s / 2)
Gamma(1 - s) zeta(1 - s)`, blended in at `s < -0.5` -- one more arm of the
same branchless shape every kernel here is built from, evaluated at
`min_of(s, -0.5)` so that the discarded side keeps `1 - s >= 1.5` and
`Gamma` never reaches a pole. That takes the Riemann form to `-20` and
past it at rounding, and the trivial zeros at the even negative integers
come out of `sin(pi s / 2)` rather than a special case.

What `pixi run accuracy` measures is therefore rounding through the
`(k+q)^{-s} = exp(-s ln(k+q))` powers and, below `-0.5`, through `Gamma`:
`6e-16` relative above the pole and `3e-14` over `[-20, 0.9]`, where it
used to read `8e-11` over the eighth of that range it covered. Near a
trivial zero read the absolute column, which is `1e-15`; relative error at
a zero is meaningless, the same reading the Bessel rows need. The Riemann
overload takes its logarithms from an exact compile-time table; the
Hurwitz one takes them at run time.

The **Hurwitz** overload keeps Euler-Maclaurin alone and so keeps the
`-2.5` floor: the functional equation has no one-term Hurwitz counterpart
at a general `q`. SciPy returns NaN for `s < 1` in that form, so there is
nothing below to match against either.

## The MAX gate

Nothing: MAX has no zeta function. **Extend.**
"""

from std.collections import Array

from ..core.numeric import (
    FloatLike,
    blend,
    ge_indicator,
    guard_nonzero,
    max_of,
    min_of,
)
from .gamma import digamma, gamma

comptime _N = 10
comptime _M = 8
comptime _TINY = 1e-300

# Where the Riemann overload leaves Euler-Maclaurin for the functional
# equation. Both clamps keep the *unselected* side finite: the reflection
# runs at `min_of(s, -0.5)` so `gamma` never reaches a pole, and
# Euler-Maclaurin at `max_of(s, -3)`, the bottom `M = 8` covers.
comptime _REFLECT_AT = -0.5
comptime _EULER_FLOOR = -3.0

comptime _PI = 3.141592653589793
comptime _LN_2 = 0.6931471805599453
comptime _LN_PI = 1.1447298858494002

# `ln k` for `k = 1 .. 11`, so the Riemann overload's powers `k^{-s} =
# exp(-s ln k)` take a logarithm correct to the last bit rather than a
# rounded one: at `s = -2.5` each term is near 300 and the answer is
# `0.0085`, so half an ulp of `ln k` is already `4e-14` of the answer,
# and the table costs nothing.
comptime _LN: Array[Float64, 11] = [
    0.0,
    0.6931471805599453,
    1.0986122886681098,
    1.3862943611198906,
    1.6094379124341003,
    1.791759469228055,
    1.9459101490553132,
    2.0794415416798357,
    2.1972245773362196,
    2.302585092994046,
    2.3978952727983707,
]

# `B_{2j} / (2j)!` for `j = 1 .. 8`: 1/12, -1/720, 1/30240, -1/1209600,
# 1/47900160, -691/1307674368000, 1/74724249600, -3617/10670622842880000.
# A compile-time table like `_LN`, not a runtime `List[Float64]`: the list
# put `double` stores in the kernel body and Metal rejected it.
comptime _BERNOULLI: Array[Float64, _M] = [
    0.08333333333333333,
    -0.001388888888888889,
    3.306878306878307e-05,
    -8.267195767195768e-07,
    2.08767569878681e-08,
    -5.284190138687493e-10,
    1.3382536530684679e-11,
    -3.3896802963225829e-13,
]


def _euler_maclaurin[T: FloatLike, riemann: Bool](s: T, q: T) -> T:
    """The Euler-Maclaurin sum for `zeta(s, q)` at fixed `N` and `M`. With
    `riemann`, `q` is `1` and the logarithms of `k + 1` are the exact
    compile-time table above; otherwise they are taken at run time."""
    var total = T.constant(0.0)
    comptime for k in range(_N):
        comptime if riemann:
            # Bound as a `comptime` alias so the table read folds to a
            # literal, the idiom `numax.interpolate`'s `Array` tier uses for its nodes.
            comptime ln_k = _LN[k]
            total = total + (-(s * T.constant(ln_k))).exp()
        else:
            total = total + (-(s * (q + T.constant(Float64(k))).ln())).exp()

    var edge = q + T.constant(Float64(_N))
    var log_edge: T
    comptime if riemann:
        comptime ln_edge = _LN[_N]
        log_edge = T.constant(ln_edge)
    else:
        log_edge = edge.ln()
    var edge_pow = (-(s * log_edge)).exp()
    var denominator = guard_nonzero(s - T.one(), T.constant(_TINY))
    total = total + edge * edge_pow / denominator + edge_pow / T.constant(2.0)

    # `B_{2j} / (2j)! * s (s+1) ... (s+2j-2) * edge^{-s-2j+1}`, the rising
    # product and the power carried along from one correction to the next.
    var rising = s.copy()
    var power = edge_pow / edge
    comptime for j in range(1, _M + 1):
        comptime b_j = _BERNOULLI[j - 1]
        total = total + T.constant(b_j) * rising * power
        rising = (
            rising
            * (s + T.constant(Float64(2 * j - 1)))
            * (s + T.constant(Float64(2 * j)))
        )
        power = power / (edge * edge)
    return total^


def zeta[T: FloatLike](s: T) -> T:
    """The Riemann zeta function `sum_{k>=1} k^{-s}` and its analytic
    continuation, for `s != 1`. `scipy.special.zeta(s)`.

    Two arms of one blend, selected by `ge_indicator(s, -0.5)`. Above
    `-0.5` it is the Euler-Maclaurin sum, valid on both sides of the pole:
    `zeta(0) = -1/2`, `zeta(-1) = -1/12`, `zeta(2) = pi^2 / 6` all come out
    of the same formula. Below it, Riemann's functional equation `zeta(s) =
    2^s pi^{s-1} sin(pi s / 2) Gamma(1 - s) zeta(1 - s)` maps the argument
    back into the convergent half-plane, which is what takes the function
    past the `s = -2.5` the Bernoulli corrections used to stop at.

    Each arm evaluates at a clamped argument so the *other* one stays
    finite everywhere: the reflection at `min_of(s, -0.5)`, where `1 - s >=
    1.5` keeps `Gamma` off its poles and `zeta(1 - s)` in the half-plane
    Euler-Maclaurin is exact in, and Euler-Maclaurin at `max_of(s, -3)`.
    Where an arm is selected its clamp is the identity, so nothing is
    approximated by the clamping.

    The trivial zeros at the even negative integers fall out of
    `sin(pi s / 2)` rather than being special-cased, and land at `1e-18`
    absolute rather than exactly zero, since the sine of a large multiple
    of `pi` is not exactly zero in floating point. Read the absolute error
    there and the relative error away from it, as the Bessel rows do.
    `pixi run accuracy` measures `3e-14` relative over `[-20, 0.9]`, all of
    it rounding through `Gamma` and the `(k+q)^{-s}` powers.

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the input.

    Args:
        s: The exponent, `s != 1`.

    Returns:
        The Riemann `zeta(s)`, analytically continued for `s < 1`.
    """
    var above = ge_indicator(s, T.constant(_REFLECT_AT))

    var direct = _euler_maclaurin[T, True](
        max_of(s, T.constant(_EULER_FLOOR)), T.one()
    )

    # `2^s pi^{s-1}` as one exponential of compile-time logarithms, so the
    # kernel body carries no runtime `Float64` table Metal would reject.
    var sr = min_of(s, T.constant(_REFLECT_AT))
    var scale = (
        sr * T.constant(_LN_2) + (sr - T.one()) * T.constant(_LN_PI)
    ).exp()
    var reflected = (
        scale
        * (T.constant(_PI) * sr / T.constant(2.0)).sin()
        * gamma(T.one() - sr)
        * _euler_maclaurin[T, True](T.one() - sr, T.one())
    )

    return blend(above, direct, reflected)


def zeta[T: FloatLike](s: T, q: T) -> T:
    """The Hurwitz zeta function `sum_{k>=0} (k + q)^{-s}` for `q > 0`,
    continued below `s = 1` as the Riemann form is. `scipy.special.zeta(s,
    q)`. `zeta(s, 1)` is `zeta(s)`.

    Euler-Maclaurin only, so this one still stops near `s = -2.5`: the
    functional equation the Riemann overload gained has no Hurwitz
    counterpart at a general `q` (Hurwitz's own reflection is a sum of two
    Lerch transcendents, not one term). SciPy returns NaN for `s < 1` in
    this form; numax returns the continuation while it is good and loses
    digits below `-2.5` rather than refusing.

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the inputs.

    Args:
        s: The exponent, `s != 1`; accurate down to about `s = -2.5`.
        q: The shift, `q > 0`.

    Returns:
        The Hurwitz `zeta(s, q)`.
    """
    return _euler_maclaurin[T, False](s, q)


def _int_pow[T: FloatLike](base: T, e: Int) -> T:
    """`base^e` for `e >= 0` by binary exponentiation: `log2(e)` roundings
    rather than `e`, and exact in sign for a negative `base`, which
    `exp(e ln(base))` is not. `e` is one value for every lane, so the
    branch on its bits is uniform."""
    var acc = T.one()
    var b = base.copy()
    var k = e
    while k > 0:
        if k & 1 == 1:
            acc = acc * b
        b = b * b
        k >>= 1
    return acc^


def polygamma[T: FloatLike](n: Int, x: T) -> T:
    """The `n`-th derivative of the digamma function,
    `psi^(n)(x) = d^(n+1)/dx^(n+1) ln Gamma(x)`. `scipy.special.polygamma(n,
    x)`.

    `n = 0` is `digamma`. For `n >= 1` it is `(-1)^(n+1) n! zeta(n + 1,
    x)`, the Hurwitz zeta summed here rather than through `zeta(s, q)`:
    the exponent is the integer `n + 1`, so the direct terms
    `(x + k)^-(n+1)` are integer powers (`_int_pow`), exact in sign for a
    negative `x + k`, where `zeta`'s `exp(-s ln(k + q))` would be NaN. That
    is what makes the function valid on the negative axis, as SciPy's is.
    `N = 30 + 2n` direct terms are followed by the Euler-Maclaurin tail at
    `q = x + N`, `q^-n / n + q^-(n+1) / 2 + sum_{j<=8} B_2j / (2j)! (n+1)
    ... (n+2j-1) q^-(n+2j)`; the first omitted correction is below `1e-16`
    relative for `n <= 20` at `x >= 0`, and the extra 30 terms keep `q` that
    large for `x` down to `-20`. The term counts depend on `n` alone,
    which is the same for every lane, so the function stays tier 1.
    `pixi run accuracy` reads `7e-16` relative at `n = 1` and `3` over
    `[0.2, 30]`, `1.5e-15` at `n = 10`, and `7e-16` at `n = 1` between the
    poles on `(-5, 0)`.

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the input.

    Args:
        n: The derivative order, `0 <= n <= 20`.
        x: The point to evaluate at, `x > -20` and not a non-positive
            integer (the poles).

    Returns:
        `psi^(n)(x)`.
    """
    if n == 0:
        return digamma(x)
    var one = T.one()
    var terms = 30 + 2 * n
    var total = T.constant(0.0)
    var shifted = x.copy()
    for _ in range(terms):
        total = total + _int_pow(one / shifted, n + 1)
        shifted = shifted + one

    # The tail at `q = x + N`, with the order and its rising product
    # carried as `T` values: a runtime `Int` converted to `Float64` inside
    # the body would be a `double` Metal rejects.
    var order = T.constant(0.0)
    var factorial = one.copy()
    for _ in range(n):
        order = order + one
        factorial = factorial * order
    var inv_q = one / shifted
    var power = _int_pow(inv_q, n)
    total = total + power / order + power * inv_q / T.constant(2.0)
    var rising = order + one
    var next = order + T.constant(2.0)
    power = power * inv_q * inv_q
    comptime for j in range(_M):
        comptime b_j = _BERNOULLI[j]
        total = total + T.constant(b_j) * rising * power
        rising = rising * next * (next + one)
        next = next + T.constant(2.0)
        power = power * inv_q * inv_q
    var sign = one.copy() if n % 2 == 1 else -one
    return sign * factorial * total
