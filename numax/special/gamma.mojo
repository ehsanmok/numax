"""The gamma family: `gamma`, `lgamma`, the incomplete gamma functions and
`gammainc`'s inverse.

**This module is tier 1.** Reflection across `x = 0.5` is a `0`/`1`
indicator built from `copysign`, and `gammainc`'s series runs a fixed 100
terms with no convergence test, and `gammaincinv` runs twelve Halley
steps against it.

Every kernel here uses a **fixed** number of terms or iterations rather than
a data-dependent convergence check: a GPU thread can't branch per-lane on
"has this series converged yet" the way a scalar loop could (every lane in
a SIMD/SIMT group has to do the same amount of work), so accuracy here is
traded for a bounded, uniform amount of work instead of adaptive precision.
`lgamma`/`gamma`'s reflection for `x <= 0` below follows the same
discipline for a different reason: `Self` may hold a SIMD vector with lanes
on both sides of the reflection boundary, and there's no `select`-like
primitive on `FloatLike` to pick a per-lane branch (an ordinary `if` would
branch on the whole vector, not each lane) -- so both "sides" are always
computed and blended arithmetically instead, using only ops `FloatLike`
already has (`abs`, `copysign`, `+`, `*`). See `lgamma`'s docstring for the
identity that makes this possible without ever evaluating an invalid
expression on the "wrong" side.

`std.math` ships a full-domain `gamma`/`lgamma` (correctly reflecting to
negative non-integer `x`, same domain as this module now), but it's a
CPU-only libm call -- compiling it into a GPU kernel fails outright with
`"constraint failed: libm operations are only available on CPU targets"`.
`numax.special.erf` does delegate to `std.math` for `Plain`, but only because
`std.math.erf` is GPU-compatible (verified by launching it on Metal and on
CUDA) *and* measurably better conditioned than the approximation it
replaced. Neither holds here, so this is not a trade worth making: this
module's own Lanczos approximation runs on both CPU and GPU today, and
swapping in `std.math` for `Plain` specifically would make `gamma`/`lgamma`
CPU-only the moment a caller's `T` happens to be `Plain` inside a
`map[gpu=True]` kernel.
"""

from std.collections import Array

from ..core.dual import Dual
from ..core.numeric import FloatLike, blend, ge_indicator, max_of, min_of


def _lgamma_positive[T: FloatLike](x: T) -> T:
    """`ln(Gamma(x))` via the Lanczos approximation (`g=7`, 9 terms).

    Valid for `x` roughly `> -6.5` (`t = z + 7.5` needs to stay positive
    for `t.ln()` below); `lgamma`/`gamma` only ever call this at `y = max(x,
    1-x) >= 0.5`, comfortably inside that range.
    """
    comptime coef: Array[Float64, 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.1385710952657201,
        9.984369578019572e-06,
        1.5056327351493116e-07,
    ]
    comptime g = 7.0

    comptime c0 = coef[0]
    var z = x - T.one()
    var a = T.constant(c0)

    comptime for i in range(1, 9):
        comptime ci = coef[i]
        a = a + T.constant(ci) / (z + T.constant(Float64(i)))

    var t = z + T.constant(g + 0.5)

    # ln(sqrt(2*pi) * t^(z+0.5) * exp(-t) * a)
    #   = 0.5*ln(2*pi) + (z+0.5)*ln(t) - t + ln(a)
    var half_ln_2pi = T.constant(0.9189385332046727)
    return half_ln_2pi + (z + T.constant(0.5)) * t.ln() - t + a.ln()


def _ge_half_indicator[T: FloatLike](x: T) -> T:
    """`1` where `x >= 0.5`, `0` where `x < 0.5` -- branchless.

    `ge_indicator`'s `x == threshold` convention puts the `x == 0.5`
    boundary on the `>= 0.5` side, which matches this being used as "pick
    `x` itself as the Lanczos argument" vs. "pick `1 - x` instead" below.
    """
    return ge_indicator(x, T.constant(0.5))


def lgamma[T: FloatLike](x: T) -> T:
    """`ln|Gamma(x)|`, valid for any `x` except the non-positive integers
    (`Gamma`'s poles).

    For `x >= 0.5` this is `_lgamma_positive(x)` directly. For `x < 0.5`,
    `Gamma(x)*Gamma(1-x) = pi/sin(pi*x)` gives `ln|Gamma(x)| = ln(pi) -
    ln|sin(pi*x)| - ln(Gamma(1-x))`, and `1 - x > 0.5` there so
    `_lgamma_positive(1-x)` is valid too. Both `y = max_of(x, 1-x)` (always
    `>= 0.5`, branchless) and the reflection formula itself
    (`sin`/`abs`/`ln` have no domain issue at `x >= 0.5` either -- they
    just compute a number this function ends up discarding) are always
    numerically valid regardless of which side `x` is actually on, which is
    what lets `_ge_half_indicator`'s `0`/`1` blend stand in for a real
    per-lane branch without ever multiplying anything by (or discarding) a
    NaN.

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the input.

    Args:
        x: The point to evaluate at, not a non-positive integer.

    Returns:
        `ln|Gamma(x)|`.
    """
    var s = _ge_half_indicator(x)

    var one_minus_x = T.one() - x
    var y = max_of(x, one_minus_x)
    var lp_y = _lgamma_positive(y)

    var sin_pix = (T.constant(3.14159265358979323846) * x).sin()
    var reflected = T.constant(1.1447298858494002) - sin_pix.abs().ln() - lp_y

    return lp_y * s + reflected * (T.one() - s)


def _gamma_sign[T: FloatLike](x: T) -> T:
    """`+1` where `Gamma(x) > 0`, `-1` where `Gamma(x) < 0` -- branchless.

    `+1` for `x >= 0.5` (`Gamma` is always positive there) and
    `sign(sin(pi*x))` for `x < 0.5`, from the same reflection identity
    `lgamma` uses: `Gamma(x) = pi / (sin(pi*x) * Gamma(1-x))`, and
    `Gamma(1-x) > 0` since `1-x > 0.5` there. `lgamma(x).exp()` only ever
    recovers `|Gamma(x)|` (`lgamma` is `ln|Gamma(x)|`), so `gamma` and
    `gammainc` below both need this to recover the actual sign.
    """
    var s = _ge_half_indicator(x)
    var sin_pix = (T.constant(3.14159265358979323846) * x).sin()
    return s + (T.one() - s) * T.one().copysign(sin_pix)


def gamma[T: FloatLike](x: T) -> T:
    """`Gamma(x)`, valid for any `x` except the non-positive integers.

    `exp(lgamma(x))` recovers the magnitude; `_gamma_sign` recovers the
    sign `lgamma` (`ln|Gamma(x)|`) discards.

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the input.

    Args:
        x: The point to evaluate at, not a non-positive integer.

    Returns:
        `Gamma(x)`, with its sign.
    """
    return lgamma(x).exp() * _gamma_sign(x)


def gammainc[T: FloatLike](a: T, x: T) -> T:
    """The regularized lower incomplete gamma function, `P(a,x) = γ(a,x)/Γ(a)`.

    `x^a * exp(-x) / Gamma(a)` times a fixed 100-term series (`sum_{n=0}^{99}
    x^n / (a*(a+1)*...*(a+n))`, accumulated term-by-term rather than as a
    ratio of factorials to avoid overflow) -- accurate to close to `dtype`
    precision when `x` isn't much larger than `a`, since that's when the
    series converges fastest; the further `x` exceeds `a`, the more of the
    fixed 100 terms it takes to reach the same accuracy; there's no
    continued-fraction branch here to compensate for that (see this
    module's docstring on why a data-dependent branch is avoided).

    `a` can be any real except the non-positive integers, now that `lgamma`
    reflects there too -- `_gamma_sign(a)` recovers `Gamma(a)`'s actual
    sign, since `lgamma(a).exp()` alone is only ever `|Gamma(a)|`. Checked
    against the standard recurrence `P(a,x) - P(a+1,x) = x^a*exp(-x) /
    Gamma(a+1)` at negative non-integer `a` (see `tests/special/test_gamma.mojo`).
    Only the `a > 0` case is a CDF bounded to `[0, 1]` -- at negative `a`
    this is still the mathematically consistent extension of the same
    series (that's what the recurrence check above confirms), but "P(a,x)"
    can then land above `1` or below `0`, since the probabilistic
    "regularized" interpretation only holds for `a > 0`.
    """
    comptime num_terms = 100

    var log_prefactor = a * x.ln() - x - lgamma(a)
    var prefactor = log_prefactor.exp() * _gamma_sign(a)

    var term = T.one() / a
    var total = term.copy()
    # The term index is carried as a running `T` rather than converted from
    # the loop's `Int`: `T.constant(Float64(n))` with a runtime `n` emits an
    # int64-to-double instruction Metal rejects, which kept this function
    # (and everything built on it) CPU-only. See `numax.special.orthopoly`'s module
    # docstring.
    var nf = T.one()
    for _ in range(1, num_terms):
        term = term * x / (a + nf)
        total = total + term
        nf = nf + T.one()

    return prefactor * total


def gammaincc[T: FloatLike](a: T, x: T) -> T:
    """The regularized upper incomplete gamma function, `Q(a,x) = 1 - P(a,x)`.
    """
    return T.one() - gammainc(a, x)


def gammaincinv[T: FloatLike](a: T, y: T) -> T:
    """The inverse of `gammainc` in its second argument: the `x` with
    `gammainc(a, x) == y`, for `a > 0` and `y` in `[0, 1]`.
    `scipy.special.gammaincinv(a, y)`.

    Tier 1. Numerical Recipes' starting guess (*Numerical Recipes*, 3rd
    ed., 6.2.1) -- Wilson-Hilferty's cube-root normal approximation for
    `a > 1`, raised to the inverted leading term `(y Gamma(a + 1))^(1/a)`
    where that is larger, and for `a <= 1` the power law `(y / t)^(1/a)` of the
    series' leading term below `t = 1 - a (0.253 + 0.12 a)` and an
    exponential tail above -- then twelve Halley steps against `gammainc`
    itself, with `dP/dx = x^(a-1) e^-x / Gamma(a)`. A step that would
    leave `x <= 0` halves `x` instead, NR's safeguard. Each guess is
    evaluated at an `a` and a `y` clamped into its own region, so both are
    finite on every lane and the choice between them is a blend.

    The result is as accurate as `gammainc` is at it: to about `1e-15`
    relative through the lower tail, and in the upper tail the absolute
    error of `gammainc` near 1 divided by the density there --
    `pixi run accuracy` reads `7e-14` relative at `y = 0.99`, `a = 10` --
    since `gammaincc` is `1 - gammainc` and carries no more digits. `gammainc`'s fixed 100-term series converges
    for `x` up to about `a + 30` at moderate `a`, which bounds how far
    into the upper tail the inverse is meaningful at large `a`. `gammaincinv(a, 0)` is `0`
    and `gammaincinv(a, 1)` is `inf`, as SciPy's.

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the inputs.

    Args:
        a: The shape, `a > 0`.
        y: The target probability, in `[0, 1]`.

    Returns:
        The `x >= 0` with `gammainc(a, x) == y`.
    """
    var zero = T.constant(0.0)
    var one = T.one()
    var half = T.constant(0.5)
    # The edges: `y = 0` is answered by multiplying by zero and `y = 1` by
    # dividing by it, with the iteration run at a stand-in `y` so every
    # intermediate stays finite on those lanes.
    var at_zero = ge_indicator(zero, y.abs())
    var at_one = ge_indicator(y, one)
    var p = y + at_zero * T.constant(0.25) - at_one * half

    # `a > 1`: Wilson-Hilferty, from a normal quantile of the tail mass.
    var big = one - ge_indicator(one, a)
    var ab = max_of(a, one)
    var pp = min_of(p, one - p)
    var t = (-T.constant(2.0) * pp.ln()).sqrt()
    var z = (T.constant(2.30753) + t * T.constant(0.27061)) / (
        one + t * (T.constant(0.99229) + t * T.constant(0.04481))
    ) - t
    z = blend(ge_indicator(p, half), z, -z)
    var cube = (
        one - one / (T.constant(9.0) * ab) - z / (T.constant(3.0) * ab.sqrt())
    )
    var guess_wh = ab * cube * cube * cube
    # Deep in the lower tail Wilson-Hilferty's cube goes negative; there
    # the series' leading term, `P ~ x^a / Gamma(a + 1)`, inverts
    # directly. It never overshoots the root -- `P(a, x) <= x^a / Gamma(a
    # + 1)`, since the rest of the series is Kummer's `M(a, a + 1, -x) <= 1`
    # -- so the larger of the two is the better guess.
    var guess_lead = ((p.ln() + lgamma(ab + one)) / ab).exp()
    var guess_big = max_of(guess_wh, guess_lead)

    # `a <= 1`: the series' leading term below `ts`, an exponential tail
    # above it, at `a` held in `(0, 1]`.
    var asm = min_of(a, one)
    var ts = one - asm * (T.constant(0.253) + asm * T.constant(0.12))
    var low = one - ge_indicator(p, ts)
    var ratio = min_of(p / ts, one)
    var guess_head = (ratio.ln() / asm).exp()
    var rest = max_of((p - ts) / (one - ts), zero)
    var guess_tail = one - (max_of(one - rest, T.constant(1e-30))).ln()
    var guess_small = blend(low, guess_head, guess_tail)

    var x = blend(big, guess_big, guess_small)
    var a1 = a - one
    var log_gamma_a = lgamma(a)
    comptime for _ in range(12):
        var err = gammainc(a, x) - p
        var density = (a1 * x.ln() - x - log_gamma_a).exp()
        var u = err / density
        var step = u / (one - half * min_of(one, u * (a1 / x - one)))
        var next = x - step
        var fallen = ge_indicator(zero, next)
        x = blend(fallen, half * x, next)
    return x * (one - at_zero) / (one - at_one)


def gammasgn[T: FloatLike](x: T) -> T:
    """The sign of `Gamma(x)`: `+1` for `x > 0` and for `x` in `(-2, -1)`,
    `(-4, -3)`, ..., `-1` on the other negative unit intervals.
    `scipy.special.gammasgn(x)`. What `lgamma`'s `ln|Gamma|` discards and
    `gamma` puts back; public so a caller doing the same
    `exp(lgamma)`-and-sign trick on a ratio can."""
    return _gamma_sign(x)


def factorial[T: FloatLike](n: T) -> T:
    """`n!` as `Gamma(n + 1)`, so a non-integer `n` gets the analytic
    continuation SciPy's `factorial(n, exact=False)` returns."""
    return gamma(n + T.one())


def comb[T: FloatLike](n: T, k: T) -> T:
    """The binomial coefficient `n choose k` as `Gamma(n+1) / (Gamma(k+1)
    Gamma(n-k+1))`, evaluated through `lgamma` so `comb(1000, 500)` does
    not overflow on the way. `scipy.special.comb(n, k, exact=False)`, and
    `scipy.special.binom` for non-integer arguments. Defined for `n >= k >=
    0`; `k > n` gives `Gamma` of a non-positive argument rather than
    SciPy's `0`."""
    var one = T.one()
    var log_value = lgamma(n + one) - lgamma(k + one) - lgamma(n - k + one)
    return log_value.exp()


def perm[T: FloatLike](n: T, k: T) -> T:
    """The number of `k`-permutations of `n`, `n! / (n - k)!`, through
    `lgamma`. `scipy.special.perm(n, k, exact=False)`. Defined for `n >= k
    >= 0`."""
    var one = T.one()
    return (lgamma(n + one) - lgamma(n - k + one)).exp()


def poch[T: FloatLike](z: T, m: T) -> T:
    """The Pochhammer symbol `(z)_m = Gamma(z + m) / Gamma(z)`, the rising
    factorial, for any real `z` and `m` off `Gamma`'s poles.
    `scipy.special.poch(z, m)`. `exp(lgamma(z+m) - lgamma(z))` with both
    signs restored through `gammasgn`, so `poch(-2.5, 3)` comes out
    `-1.875` and not its magnitude."""
    var magnitude = (lgamma(z + m) - lgamma(z)).exp()
    return magnitude * _gamma_sign(z + m) * _gamma_sign(z)


def digamma[T: FloatLike](x: T) -> T:
    """The digamma function, `psi(x) = d/dx[ln(Gamma(x))]`.

    Implemented as literally that: `lgamma` evaluated at a `Dual` seeded
    with derivative `1`, reading the derivative back off. There is no
    separate series or asymptotic expansion here, and there doesn't need to
    be -- `lgamma` is built entirely from `FloatLike` operations that
    `Dual` already knows the chain rule for, so differentiating it is
    exactly the "write the kernel once, get the derivative from the type"
    pattern the rest of the library sells, turned on the library itself.

    Valid wherever `lgamma` is (any `x` except the non-positive integers),
    including the reflected `x < 0.5` side, since `Dual` differentiates
    through whichever branch `lgamma`'s blend selects rather than needing
    its own reflection formula. `tests/special/test_gamma.mojo` was already
    checking this exact expression against `psi(5)` before it had a name.

    Note this costs one `lgamma` evaluation carrying a derivative
    alongside, not two evaluations -- forward-mode propagates the
    derivative through the same pass that computes the value.

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the input.

    Args:
        x: The point to evaluate at, not a non-positive integer.

    Returns:
        `psi(x)`, the derivative of `lgamma` at `x`.
    """
    var seeded = lgamma(Dual[T](x.copy(), T.one()))
    return seeded.deriv.copy()
