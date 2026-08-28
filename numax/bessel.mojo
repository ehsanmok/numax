"""Bessel functions of the first and second kind, orders zero and one.

`std.math` ships full-domain `j0`/`j1`/`y0`/`y1`, but -- same story as
`numax.gamma`'s docstring -- they're CPU-only libm calls, rejected outright
inside a GPU kernel. (On this platform's libm, the `float32` symbols for
all four are missing entirely, so they're not even usable at `numax`'s
usual `dtype` on CPU alone.) Not adopted here for the same reason `gamma`
wasn't: it would make these functions CPU-only for `Plain` specifically,
regressing GPU support this module already has.

Every function below blends a "near" branch (a polynomial, valid at small
`|x|`) with a "far" branch (an oscillating asymptotic form, valid at large
`|x|`) at `|x| = 3`, the same threshold Abramowitz & Stegun's own 9.4.1-9.4.6
polynomials split at. Both branches are always computed and blended
arithmetically, the same way `numax.gamma`'s reflection is: no `if` (`Self`
may hold a SIMD vector with lanes on both sides of the threshold), and the
far branch's argument is clamped to `max(|x|, 3.0)` first (the same
"clamp so the discarded side stays finite" trick `gamma` uses `max(x, 1-x)`
for) so it never divides by a near-zero `|x|` -- the raw asymptotic formula
would otherwise blow up (and eventually produce `cos`/`sin` of an
overflowing argument, i.e. NaN) well before reaching the actual blend.
`1/sqrt(ax)` in the far branch is `exp(-0.5 * ln(ax))`, since `FloatLike`
has `ln`/`exp` but no `sqrt` of its own -- not worth adding a primitive for
one call site.

Coefficients for `J0`/`J1`'s near and far branches, and `Y0`'s near branch,
are Abramowitz & Stegun 9.4.1-9.4.5, transcribed and then checked
numerically against `std.math`'s `j0`/`j1`/`y0` (float64, CPU-only --
used here purely as a test oracle, see `tests/test_bessel.mojo`) to within
each formula's documented error bound. `Y1`'s near-branch polynomial
(9.4.6) doesn't get the same transcription treatment: rather than risk
mistranscribing its coefficients from a source with no clean, non-OCR
digitization, they're a direct least-squares fit of `Y1(x) - (2/pi)*
(ln(x/2)*J1(x) - 1/x)` against `std.math.y1` over `x` in `(0, 3]` -- checked
to agree with the reference to ~1e-9, comparable to or better than the
other branches' A&S-quoted error bounds, so nothing here is any less
trustworthy for being fit rather than transcribed.
"""

from .numeric import FloatLike, ge_indicator, max_of

comptime _PI = 3.14159265358979323846
comptime _TWO_OVER_PI = 0.6366197723675814


def _far_branch_indicator[T: FloatLike](ax: T) -> T:
    """`1` where `ax <= 3`, `0` where `ax > 3` -- branchless."""
    return ge_indicator(T.constant(3.0) + (-ax), T.constant(0.0))


def _clamp_to_far_domain[T: FloatLike](ax: T) -> T:
    """`max(ax, 3.0)`: keeps the far branch's `3/ax` finite even when the
    real `ax` is near zero and the near branch is about to be selected
    instead.
    """
    return max_of(ax, T.constant(3.0))


def _inv_sqrt[T: FloatLike](ax_safe: T) -> T:
    """`1/sqrt(ax_safe)`. This used to be `exp(-0.5*ln(ax_safe))`, from
    before `FloatLike` had a `sqrt` -- one hardware instruction now instead
    of two transcendental calls, and it drops the accuracy the round trip
    through `ln`/`exp` was costing.
    """
    return T.one() / ax_safe.sqrt()


def _j0_near[T: FloatLike](x: T) -> T:
    """A&S 9.4.1: `J0(x)` for `|x| <= 3`, max absolute error ~5e-8.

    `x` only appears as `x^2`, so no `abs()` is needed -- already even.
    """
    var t = (x * x) / T.constant(9.0)
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


def _j1_near[T: FloatLike](x: T) -> T:
    """A&S 9.4.2: `J1(x)` for `|x| <= 3`, max absolute error ~1.3e-8.

    The polynomial is a function of `t = (x/3)^2` (even), multiplied by
    `x` itself -- odd overall, matching `J1`, with no explicit sign
    handling needed.
    """
    var t = (x * x) / T.constant(9.0)
    var poly = (
        (
            (
                (
                    (T.constant(0.00001109) * t + T.constant(-0.00031761)) * t
                    + T.constant(0.00443319)
                )
                * t
                + T.constant(-0.03954289)
            )
            * t
            + T.constant(0.21093573)
        )
        * t
        + T.constant(-0.56249985)
    ) * t + T.constant(0.5)
    return x * poly


def _f0[T: FloatLike](p: T) -> T:
    """A&S 9.4.3's `f0(p)`, shared by `J0`'s and `Y0`'s far branch."""
    return (
        (
            (
                (
                    (T.constant(0.00014476) * p + T.constant(-0.00072805)) * p
                    + T.constant(0.00137237)
                )
                * p
                + T.constant(-0.00009512)
            )
            * p
            + T.constant(-0.00552740)
        )
        * p
        + T.constant(-0.00000077)
    ) * p + T.constant(0.79788456)


def _theta0[T: FloatLike](ax_safe: T, p: T) -> T:
    """A&S 9.4.3's `theta0(p)`, shared by `J0`'s and `Y0`'s far branch."""
    return (
        ax_safe
        + (
            (
                (
                    (
                        (T.constant(0.00013558) * p + T.constant(-0.00029333))
                        * p
                        + T.constant(-0.00054125)
                    )
                    * p
                    + T.constant(0.00262573)
                )
                * p
                + T.constant(-0.00003954)
            )
            * p
            + T.constant(-0.04166397)
        )
        * p
        + T.constant(-0.78539816)
    )


def _f1[T: FloatLike](p: T) -> T:
    """A&S 9.4.4's `f1(p)`, shared by `J1`'s and `Y1`'s far branch."""
    return (
        (
            (
                (
                    (T.constant(-0.00020033) * p + T.constant(0.00113653)) * p
                    + T.constant(-0.00249511)
                )
                * p
                + T.constant(0.00017105)
            )
            * p
            + T.constant(0.01659667)
        )
        * p
        + T.constant(0.00000156)
    ) * p + T.constant(0.79788456)


def _theta1[T: FloatLike](ax_safe: T, p: T) -> T:
    """A&S 9.4.4's `theta1(p)`, shared by `J1`'s and `Y1`'s far branch."""
    return (
        ax_safe
        + (
            (
                (
                    (
                        (T.constant(-0.00029166) * p + T.constant(0.00079824))
                        * p
                        + T.constant(0.00074348)
                    )
                    * p
                    + T.constant(-0.00637879)
                )
                * p
                + T.constant(0.00005650)
            )
            * p
            + T.constant(0.12499612)
        )
        * p
        + T.constant(-2.35619449)
    )


def bessel_j0[T: FloatLike](x: T) -> T:
    """`J0(x)`, the order-zero Bessel function of the first kind, for any
    real `x`. Even, so only `|x|` ever reaches either branch.
    """
    var ax = x.abs()
    var s = _far_branch_indicator(ax)

    var near = _j0_near(x)

    var ax_safe = _clamp_to_far_domain(ax)
    var p = T.constant(3.0) / ax_safe
    var far = _f0(p) * _inv_sqrt(ax_safe) * _theta0(ax_safe, p).cos()

    return near * s + far * (T.one() + (-s))


def bessel_j1[T: FloatLike](x: T) -> T:
    """`J1(x)`, the order-one Bessel function of the first kind, for any
    real `x`. Odd: the far branch is computed at `|x|` and then given
    `x`'s sign back via `copysign` (`_j1_near`'s `x * poly(x^2)` shape
    already makes the near branch odd on its own).
    """
    var ax = x.abs()
    var s = _far_branch_indicator(ax)

    var near = _j1_near(x)

    var ax_safe = _clamp_to_far_domain(ax)
    var p = T.constant(3.0) / ax_safe
    var far_at_ax = _f1(p) * _inv_sqrt(ax_safe) * _theta1(ax_safe, p).cos()
    # `far_at_ax` is `J1`'s far-branch value at the *positive* mirror
    # point `ax`, already carrying whatever sign `J1` actually has there
    # (`J1` oscillates through both signs for positive arguments, same as
    # `J0` -- this is a genuine sign flip for `J1`'s oddness, not a
    # magnitude/sign split, so it's a `*`, not a `copysign`).
    var far = far_at_ax * T.one().copysign(x)

    return near * s + far * (T.one() + (-s))


def bessel_y0[T: FloatLike](x: T) -> T:
    """`Y0(x)`, the order-zero Bessel function of the second kind, for
    `x > 0` (`Y0` itself has a logarithmic singularity at `0` and isn't
    real-valued for negative `x`).
    """
    var s = _far_branch_indicator(x)

    var t = (x * x) / T.constant(9.0)
    var near_poly = (
        (
            (
                (
                    (T.constant(-0.00024846) * t + T.constant(0.00427916)) * t
                    + T.constant(-0.04261214)
                )
                * t
                + T.constant(0.25300117)
            )
            * t
            + T.constant(-0.74350384)
        )
        * t
        + T.constant(0.60559366)
    ) * t + T.constant(0.36746691)
    var near = (
        T.constant(_TWO_OVER_PI) * (x * T.constant(0.5)).ln() * _j0_near(x)
        + near_poly
    )

    var x_safe = _clamp_to_far_domain(x)
    var p = T.constant(3.0) / x_safe
    var far = _f0(p) * _inv_sqrt(x_safe) * _theta0(x_safe, p).sin()

    return near * s + far * (T.one() + (-s))


def bessel_y1[T: FloatLike](x: T) -> T:
    """`Y1(x)`, the order-one Bessel function of the second kind, for
    `x > 0` (same domain restriction as `Y0`, plus its own `-2/(pi*x)`
    singularity at `0`).

    The near branch's polynomial correction (see this module's docstring)
    is a least-squares fit rather than a transcription; everything else is
    A&S 9.4.2/9.4.4/9.4.6's shared `ln(x/2)*J1(x) - 1/x` structure.
    """
    var s = _far_branch_indicator(x)

    var t = (x * x) / T.constant(9.0)
    var near_poly = (
        (
            (
                (
                    (
                        T.constant(-1.9114694908021805e-05) * t
                        + T.constant(0.0003762544910492209)
                    )
                    * t
                    + T.constant(-0.00454698544460594)
                )
                * t
                + T.constant(0.03477406025093513)
            )
            * t
            + T.constant(-0.14629895727503786)
        )
        * t
        + T.constant(0.24092313569092993)
    ) * t + T.constant(0.024578509592272924)
    var near = (
        T.constant(_TWO_OVER_PI)
        * ((x * T.constant(0.5)).ln() * _j1_near(x) + (-(T.one() / x)))
        + x * near_poly
    )

    var x_safe = _clamp_to_far_domain(x)
    var p = T.constant(3.0) / x_safe
    var far = _f1(p) * _inv_sqrt(x_safe) * _theta1(x_safe, p).sin()

    return near * s + far * (T.one() + (-s))
