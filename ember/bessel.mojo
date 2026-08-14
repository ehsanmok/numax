"""The Bessel function of the first kind, order zero."""

from .numeric import FloatLike


def bessel_j0[T: FloatLike](x: T) -> T:
    """`J0(x)`, via the Abramowitz & Stegun 9.4.1 polynomial approximation.

    Valid for `|x| <= 3` (max absolute error ~5e-8 there); `ember` doesn't
    implement A&S 9.4.3's asymptotic form for `|x| > 3` yet, since blending
    the two branch-free needs a `select`-like primitive `FloatLike` doesn't
    have (a data-dependent branch on `Self` isn't safe when `Self` may hold
    a SIMD vector with lanes on both sides of the threshold). `x` only
    appears here as `x^2`, so no explicit `abs()` is needed -- the
    polynomial is already even.
    """
    var t = (x * x) / T.constant(9.0)

    # Horner's method on t = (x/3)^2.
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
