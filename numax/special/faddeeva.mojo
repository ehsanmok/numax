"""The Faddeeva function `w(z) = exp(-z^2) erfc(-i z)`, `wofz`, SciPy's.

**This module is tier 1.** `w` is Weideman's rational approximation
(*Computation of the complex error function*, SIAM J. Numer. Anal. 31,
1994) at `N = 40`:

    w(z) = 2 p(Z) / (L - i z)^2 + (1 / sqrt(pi)) / (L - i z),
    Z = (L + i z) / (L - i z),  L = sqrt(N / sqrt(2)),

`p` a degree-40 polynomial with real coefficients, valid on the closed
upper half-plane. The coefficients are the discrete Fourier coefficients
of `exp(-t^2) (L^2 + t^2)` at `t = L tan(theta / 2)`, computed at 40
digits in mpmath and rounded once. Below the real axis, `w(z) = 2
exp(-z^2) - w(-z)`, blended in with the exponential's argument zeroed on
the upper lanes so it never overflows where it is discarded. There is no
region split in the upper half-plane at all: one rational function covers
it, which is what makes this cheaper to launch than the continued fraction
plus series pair most implementations use.

Measured against SciPy's `wofz` (Johnson's Faddeeva package) over `x`
in `+-[1e-3, 1e3]` and `y` in `[0, 1e3]`: `6e-15` relative in modulus.
`pixi run accuracy` reads each component against mpmath along lines of
the plane -- `x + i`, the imaginary axis (`erfcx`), the real axis
(Dawson's integral) and `x - 0.5 i` -- at `1.3e-15` relative or better. The real part on the real axis is `exp(-x^2)`, which for large
`|x|` is far below the imaginary part and is accurate only to that
modulus bound, as in any such evaluation. Below the axis `w` grows as
`exp(y^2 - x^2)` and overflows where it truly does.

## The MAX gate

Nothing: neither `std.math` nor any MAX root has a complex error
function. **Extend.**
"""

from std.collections import Array

from ..core.complex import Complex
from ..core.numeric import FloatLike, ge_indicator

comptime _L = 5.3182958969449885
comptime _INV_SQRT_PI = 0.5641895835477563

comptime _WEIDEMAN: Array[Float64, 40] = [
    -1.899694947394927e-15,
    1.128073562364402e-15,
    1.1357687198999241e-14,
    -5.409310282882142e-15,
    -7.074086260286855e-14,
    1.37256205867155e-14,
    4.5329666782606727e-13,
    1.2031458219387989e-13,
    -2.907688342182867e-12,
    -2.7276023158200452e-12,
    1.7714495214011192e-11,
    3.47272670930455e-11,
    -9.055124450928292e-11,
    -3.5632339865976533e-10,
    2.1086006347066517e-10,
    3.0177805400090707e-09,
    3.2497465180436973e-09,
    -1.8315616783040462e-08,
    -6.35177348504429e-08,
    1.4198642399935674e-08,
    5.912136951899494e-07,
    1.483566113220078e-06,
    -1.0660138984947143e-06,
    -1.8007447144750956e-05,
    -5.591309264248318e-05,
    -3.939363145489569e-05,
    0.0004398070159869668,
    0.0027054056330737914,
    0.010048186242783424,
    0.029202916471241867,
    0.07182361779074337,
    0.15504263802479495,
    0.29989437996150065,
    0.5266528988277086,
    0.8472174576593818,
    1.2563815675765133,
    1.7253830848179779,
    2.201513794878312,
    2.61605415276186,
    2.8996245093897053,
]
"""Weideman's `p` at `N = 40`, highest degree first, for Horner."""


def _upper[T: FloatLike](z: Complex[T]) -> Complex[T]:
    """`w(z)` for `Im z >= 0`, Weideman's form."""
    var iz = Complex[T](-z.im, z.re.copy())
    var denom = Complex[T].constant(_L) - iz
    var big_z = (Complex[T].constant(_L) + iz) / denom
    var p = Complex[T].constant(0.0)
    comptime for k in range(40):
        comptime c = _WEIDEMAN[k]
        p = p * big_z + Complex[T].constant(c)
    return (
        Complex[T].constant(2.0) * p / (denom * denom)
        + Complex[T].constant(_INV_SQRT_PI) / denom
    )


def wofz[T: FloatLike](z: Complex[T]) -> Complex[T]:
    """The Faddeeva function `w(z) = exp(-z^2) erfc(-i z)`.
    `scipy.special.wofz(z)`.

    Weideman's `N = 40` rational form in the upper half-plane and the
    reflection `w(z) = 2 exp(-z^2) - w(-z)` below it, per this module's
    docstring: `6e-15` relative in modulus against SciPy, `1.3e-15` per
    component in `pixi run accuracy`. For real `x`,
    `Re w(x) = exp(-x^2)` and `Im w(x)` is `2 / sqrt(pi)` times Dawson's
    integral; `Re w(i y) = erfcx(y)`; and `Re w` is the Voigt profile's
    kernel.

    Parameters:
        T: The `FloatLike` conformer of the real and imaginary parts.

    Args:
        z: The point to evaluate at.

    Returns:
        `w(z)`.
    """
    var zero = T.constant(0.0)
    # 1 below the real axis, 0 on and above it.
    var lower = T.one() - ge_indicator(z.im, zero)
    var flip = T.one() - T.constant(2.0) * lower
    var zu = Complex[T](z.re * flip, z.im * flip)
    var wu = _upper(zu)
    #  for the lower lanes, as  plus a
    # correction the indicator scales to zero elsewhere; the exponential's
    # argument is zeroed on the upper lanes so it stays finite there.
    var zl = Complex[T](z.re * lower, z.im * lower)
    var expo = (-(zl * zl)).exp()
    var corr = Complex[T].constant(2.0) * (expo - wu)
    return Complex[T](wu.re + lower * corr.re, wu.im + lower * corr.im)
