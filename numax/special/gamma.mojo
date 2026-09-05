"""The gamma family: `gamma`, `lgamma`, and the incomplete gamma functions.

**This module is tier 1.** Reflection across `x = 0.5` is a `0`/`1`
indicator built from `copysign`, and `gammainc`'s series runs a fixed 100
terms with no convergence test.

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
from ..core.numeric import FloatLike, ge_indicator, max_of


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
    """
    var seeded = lgamma(Dual[T](x.copy(), T.one()))
    return seeded.deriv.copy()
