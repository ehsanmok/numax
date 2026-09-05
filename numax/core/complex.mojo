"""`Complex[Inner: FloatLike]`: a complex number over any `FloatLike` base
type, itself conforming to `FloatLike` -- the same nesting trick `Dual`
uses (`Complex[Dual[Plain[...]]]` differentiates a complex-valued kernel
the same way `Dual[Plain[...]]` differentiates a real one, with no changes
to the kernel itself).

**This module is tier 1.** `erf` is a fixed 40-term series and `_atan2` a
fixed-degree polynomial with a branchless quadrant correction, so a
complex-valued kernel launches like any other.

Most of `FloatLike`'s contract has an unambiguous complex-analytic
extension (`+`, `-`, `*`, `/`, `exp`, `sin`, `cos` -- all standard, `sin`/
`cos` via `exp` alone, since `FloatLike` has no `sinh`/`cosh` of its own and
one call site here isn't reason enough to add them: `cosh(b) =
(exp(b)+exp(-b))/2`, `sinh(b) = (exp(b)-exp(-b))/2`). Three methods don't:

- **`ln`.** `ln(z) = ln(|z|) + i*atan2(im, re)` needs the complex
  *argument*, which needs `atan2` -- not in `FloatLike` at all (no
  conformer has needed it before this one). Rather than grow the trait for
  a capability only `Complex` uses, `_atan2` below is a private,
  from-scratch helper built entirely from ops `FloatLike` already has: a
  degree-15 odd polynomial fit for `atan(a)` on `a` in `[0, 1]` (fit
  in Python against `math.atan`, max error ~9e-8 -- there's no `std.math`
  polynomial-approximation source to transcribe here, `atan` itself is
  simple enough that fitting directly against the trusted reference was
  more reliable than hunting for a citable one), combined into a full
  `atan2` via the standard branchless min/max identity (swap the
  larger/smaller of `|re|`/`|im|` so the polynomial only ever sees an
  argument in `[0, 1]`, then correct the quadrant with the same
  `copysign`-built indicator blend `numax.special.gamma`/`numax.special.bessel` already
  use for their own branchless reflections).
- **`abs`/`copysign`.** `FloatLike.abs(self) -> Self` forces the return
  type to stay `Self` -- but a complex number's true magnitude,
  `sqrt(re^2+im^2)`, is a *real* number, not a complex one, so there's no
  way to return it as a `Complex` without picking a convention. This
  implementation embeds it as `Complex(magnitude, 0)`; `copysign` follows
  the same convention, embedding `Complex(magnitude(self).copysign
  (sign_source.re), 0)`. Neither has a canonical mathematical meaning for
  complex numbers (there's no total order on `C`, so "sign" is already a
  choice, not a fact) -- these exist so `Complex` satisfies the trait's
  contract, not because a kernel written for real `FloatLike` values is
  expected to call them meaningfully on a genuinely complex one.
- **`erf`/`erfc`.** No closed form from elementary functions; this uses
  the defining power series (`erf(z) = (2/sqrt(pi)) * sum_n (-1)^n *
  z^(2n+1) / (n! * (2n+1))`) directly, fixed at 40 terms (`numax`'s usual
  "bounded, uniform work" discipline -- see `numax.special.gamma`'s docstring for
  why). Checked against `Plain.erf()` along the real axis (where the
  series should reduce to it exactly) and cross-checked for stability
  against a much higher term count off-axis in Python before writing this
  -- accurate for moderate `|z|`, but the series converges roughly like a
  Taylor expansion of `exp(|z|^2)` (`erf` restricted to the imaginary axis
  grows like `exp(y^2)`, the same "Dawson/Faddeeva function" territory a
  robust complex `erf` needs a genuinely different algorithm for), so 40
  fixed terms stop being enough well before `|z|` gets large -- a real
  scoping gap, not an oversight, and not worth a more elaborate algorithm
  until something downstream actually needs `erf` at large complex
  arguments. `erfc` is `one() - erf()` (no complex `erfc`-specific
  identity is used here to dodge cancellation the way `Plain.erfc()`
  does for real arguments).
"""

from .numeric import FloatLike, ge_indicator, max_of

comptime _PI = 3.14159265358979323846
comptime _TWO_OVER_SQRT_PI = 1.1283791670955126

# Coefficients of `atan(a)/a` as an even polynomial in `a^2`, fit in Python
# against `math.atan` over `a` in `(0, 1]` (Chebyshev-node least squares),
# max absolute error in `atan(a)` itself ~8.8e-8.
comptime _ATAN_C0 = 0.9999999581834507
comptime _ATAN_C1 = -0.333323026315408
comptime _ATAN_C2 = 0.1997367798272199
comptime _ATAN_C3 = -0.1404011815243807
comptime _ATAN_C4 = 0.09967863968125344
comptime _ATAN_C5 = -0.06021825244673024
comptime _ATAN_C6 = 0.024756145042650144
comptime _ATAN_C7 = -0.004830987285053649


def _atan_poly[T: FloatLike](a: T) -> T:
    """`atan(a)` for `a` in `[0, 1]`, via the fit in this module's
    docstring, Horner's method in `a^2`.
    """
    var a2 = a * a
    var s = (
        (
            (
                (
                    (
                        (T.constant(_ATAN_C7) * a2 + T.constant(_ATAN_C6)) * a2
                        + T.constant(_ATAN_C5)
                    )
                    * a2
                    + T.constant(_ATAN_C4)
                )
                * a2
                + T.constant(_ATAN_C3)
            )
            * a2
            + T.constant(_ATAN_C2)
        )
        * a2
        + T.constant(_ATAN_C1)
    ) * a2 + T.constant(_ATAN_C0)
    return a * s


def _atan2[T: FloatLike](y: T, x: T) -> T:
    """`atan2(y, x)`, branchless throughout (every lane may hold a
    different quadrant): reduces to `atan` of `min(|x|,|y|)/max(|x|,|y|)`
    (always in `[0, 1]`, so `_atan_poly` above is always in its valid
    domain), then blends in the two quadrant corrections and finally
    `y`'s sign, all via the same `copysign`-indicator pattern as the rest
    of `numax`'s special functions.
    """
    var ax = x.abs()
    var ay = y.abs()
    var diff = ax - ay
    var big = (ax + ay + diff.abs()) / T.constant(2.0)
    var small = (ax + ay - diff.abs()) / T.constant(2.0)
    var big_safe = max_of(big, T.constant(1e-300))

    var r = _atan_poly(small / big_safe)

    var ay_gt_ax = ge_indicator(-diff, T.constant(0.0))
    var r1 = r + ay_gt_ax * (T.constant(_PI / 2.0) - (T.constant(2.0) * r))

    var x_lt_0 = T.one() - ge_indicator(x, T.constant(0.0))
    var r2 = r1 + x_lt_0 * (T.constant(_PI) - (T.constant(2.0) * r1))

    return r2.copysign(y)


@fieldwise_init
struct Complex[Inner: FloatLike](Copyable, FloatLike, Movable):
    """A complex number, `re + i*im`, over any `FloatLike` base type."""

    var re: Self.Inner
    var im: Self.Inner

    @staticmethod
    def i() -> Self:
        """The imaginary unit -- a convenience, not part of `FloatLike`."""
        return Self(Self.Inner.constant(0.0), Self.Inner.one())

    @staticmethod
    def one() -> Self:
        return Self(Self.Inner.one(), Self.Inner.constant(0.0))

    def __add__(self, rhs: Self) -> Self:
        return Self(self.re + rhs.re, self.im + rhs.im)

    def __mul__(self, rhs: Self) -> Self:
        # (a+bi)(c+di) = (ac-bd) + (ad+bc)i.
        return Self(
            self.re * rhs.re - (self.im * rhs.im),
            self.re * rhs.im + self.im * rhs.re,
        )

    def __neg__(self) -> Self:
        return Self(-self.re, -self.im)

    def __truediv__(self, rhs: Self) -> Self:
        # (a+bi)/(c+di) = (a+bi)(c-di) / (c^2+d^2).
        var denom = rhs.re * rhs.re + rhs.im * rhs.im
        var num_re = self.re * rhs.re + self.im * rhs.im
        var num_im = self.im * rhs.re - (self.re * rhs.im)
        return Self(num_re / denom, num_im / denom)

    def _modulus(self) -> Self.Inner:
        """`sqrt(re^2 + im^2)`, straight off `Inner.sqrt` now that
        `FloatLike` has one (this used to be `exp(0.5*ln(...))`, and needed
        a floor on its argument so `ln` never saw a literal zero; `sqrt(0)`
        is just `0`, so the floor is gone with it).
        """
        return (self.re * self.re + self.im * self.im).sqrt()

    def sqrt(self) -> Self:
        # The principal square root, via the standard half-angle-free form
        # sqrt(z) = sqrt((|z|+a)/2) + i*sign(b)*sqrt((|z|-a)/2), which
        # avoids computing an argument and halving it. Both radicands are
        # non-negative mathematically (|z| >= |a|); `max_of` against zero
        # keeps a lane that rounded a hair below from reaching `sqrt` as a
        # negative. `copysign`'s `+1`-at-zero convention puts the branch cut
        # on the negative real axis approached from below, matching the
        # usual principal-branch convention (`sqrt(-4) = 2i`).
        var modulus = self._modulus()
        var half = Self.Inner.constant(0.5)
        var re_part = (half * (modulus + self.re)).sqrt()
        var im_magnitude = (
            max_of(half * (modulus - self.re), Self.Inner.constant(0.0))
        ).sqrt()
        return Self(re_part^, im_magnitude.copysign(self.im))

    def exp(self) -> Self:
        # exp(a+bi) = exp(a) * (cos(b) + i*sin(b)).
        var mag = self.re.exp()
        return Self(mag * self.im.cos(), mag * self.im.sin())

    def ln(self) -> Self:
        # ln(z) = ln(|z|) + i*atan2(im, re) -- see this module's docstring
        # for why `atan2` is a private helper here rather than a trait
        # method.
        var log_mag = self._modulus().ln()
        var theta = _atan2(self.im, self.re)
        return Self(log_mag^, theta^)

    def erf(self) -> Self:
        # The defining power series -- see this module's docstring for the
        # term count and its scope.
        comptime num_terms = 40

        var z2 = self * self
        var term = self.copy()
        var total = term.copy()
        # `comptime for`, not an ordinary loop: `n` has to be a compile-time
        # value for `Float64(n)` to fold into a `dtype` literal. A runtime
        # index would instead emit the int64-to-double conversion Metal
        # rejects, as `numax.special.orthopoly`'s module docstring records.
        comptime for n in range(1, num_terms):
            var factor = Self.Inner.constant(
                -(Float64(2 * n - 1)) / Float64(n * (2 * n + 1))
            )
            term = term * z2 * Self(factor^, Self.Inner.constant(0.0))
            total = total + term
        var scale = Self(
            Self.Inner.constant(_TWO_OVER_SQRT_PI), Self.Inner.constant(0.0)
        )
        return total * scale

    def erfc(self) -> Self:
        return Self.one() - self.erf()

    def sin(self) -> Self:
        # sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b).
        var eb = self.im.exp()
        var e_neg_b = (-self.im).exp()
        var cosh_b = (eb + e_neg_b) * Self.Inner.constant(0.5)
        var sinh_b = (eb - e_neg_b) * Self.Inner.constant(0.5)
        return Self(self.re.sin() * cosh_b, self.re.cos() * sinh_b)

    def cos(self) -> Self:
        # cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b).
        var eb = self.im.exp()
        var e_neg_b = (-self.im).exp()
        var cosh_b = (eb + e_neg_b) * Self.Inner.constant(0.5)
        var sinh_b = (eb - e_neg_b) * Self.Inner.constant(0.5)
        return Self(self.re.cos() * cosh_b, -(self.re.sin() * sinh_b))

    @staticmethod
    def constant(v: Float64) -> Self:
        return Self(Self.Inner.constant(v), Self.Inner.constant(0.0))

    def abs(self) -> Self:
        # See this module's docstring for the embedding convention.
        return Self(self._modulus(), Self.Inner.constant(0.0))

    def copysign(self, sign_source: Self) -> Self:
        # See this module's docstring for the embedding convention.
        return Self(
            self._modulus().copysign(sign_source.re),
            Self.Inner.constant(0.0),
        )

    def floor(self) -> Self:
        # Same "no canonical complex meaning" embedding as `abs`/`copysign`
        # above: `self._modulus()` is a real `Inner`, `floor`ed on that real
        # axis, then re-embedded as `Complex(x, 0)`.
        return Self(self._modulus().floor(), Self.Inner.constant(0.0))

    def ceil(self) -> Self:
        return Self(self._modulus().ceil(), Self.Inner.constant(0.0))

    def trunc(self) -> Self:
        return Self(self._modulus().trunc(), Self.Inner.constant(0.0))
