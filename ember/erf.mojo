"""The error function and its complement, generic over `FloatLike`."""

from .numeric import FloatLike


def erf[T: FloatLike](x: T) -> T:
    """The error function, `(2/sqrt(pi)) * integral(exp(-t^2), 0, x)`.

    Uses the Abramowitz & Stegun 7.1.26 rational approximation (max absolute
    error ~1.5e-7 for `x >= 0`), extended to negative `x` via `copysign`
    since `erf` is odd -- the formula itself is only derived for `x >= 0`.
    """
    var ax = x.abs()

    var t = T.one() / (T.one() + T.constant(0.3275911) * ax)

    # Horner's method: ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t.
    var poly = (
        (
            (
                (T.constant(1.061405429) * t + T.constant(-1.453152027)) * t
                + T.constant(1.421413741)
            )
            * t
            + T.constant(-0.284496736)
        )
        * t
        + T.constant(0.254829592)
    ) * t

    var erf_ax = T.one() + (-(poly * (-(ax * ax)).exp()))
    return erf_ax.copysign(x)


def erfc[T: FloatLike](x: T) -> T:
    """The complementary error function, `1 - erf(x)`."""
    return T.one() + (-erf(x))
