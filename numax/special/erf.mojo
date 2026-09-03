"""The error function and its complement.

`erf`/`erfc` are `FloatLike` trait methods (see `numax.core.numeric`), not
freestanding formulas, so each conformer can supply whatever's fastest or
most precise for it: `Plain` delegates straight to `std.math.erf`/`erfc`
(a genuinely different, GPU-compatible implementation that doesn't share
`default_erf_approx`'s cancellation error -- see this module's history and
`tests/core/test_compensated.mojo`), `Dual` differentiates through the chain
rule over its `Inner`'s own `erf()`, and `Compensated`/`Decimal` fall back
to `numax.core.numeric.default_erf_approx`, the Abramowitz & Stegun 7.1.26
approximation this module used to implement directly for every type.

The free functions below just forward to those trait methods, so
`from numax import erf, erfc` keeps working the same way it always has.
"""

from ..core.numeric import FloatLike


def erf[T: FloatLike](x: T) -> T:
    """The error function, `(2/sqrt(pi)) * integral(exp(-t^2), 0, x)`."""
    return x.erf()


def erfc[T: FloatLike](x: T) -> T:
    """The complementary error function, `1 - erf(x)`."""
    return x.erfc()
