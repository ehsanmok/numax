"""Complete elliptic integrals of the first and second kind, `K(m)` and
`E(m)` (parameterized by `m = k^2`, following `std`'s and A&S's own
convention -- not the modulus `k` itself).

Both are Abramowitz & Stegun 17.3.34/17.3.36's polynomial-plus-log
("Hastings") approximations in `m1 = 1 - m`, each accurate to ~2e-8. Neither
is in `std.math` or MAX's accelerator library at all, so there's nothing to
delegate to for either function, and both were checked against a
from-scratch Gauss-AGM reference implementation written in Python before
this module (see `tests/special/test_elliptic.mojo`, which ports it).

That reference was not a formality: A&S 17.3.36's table, as digitized,
has a known-bad digit group for `E(m)`'s `b4` coefficient. Every OCR'd copy
found while researching this returns `b4` mangled -- differently each time,
but always wrong by roughly 40%, producing a ~3e-4 error instead of the
documented ~2e-8. It was recovered by holding the other seven coefficients
at their literature values and least-squares fitting `b4` alone against the
AGM reference across `m` in `(0, 1)`, landing on the `0.00526449639` used
below. That value also resolves the "449639" tail visible in the mangled
text once the leading digits are corrected, so it's the number the table
meant rather than an independent invention. `K`'s 17.3.34 table needed none
of this: all ten of its coefficients checked out exactly against the same
reference.

`K(m)` has a genuine logarithmic singularity at `m = 1` (`K(m) ->
+infinity`); `E(m)` is finite there (`E(1) = 1`) but its own formula has a
`0 * (-infinity)` indeterminate form at exactly `m1 = 0`, since `E`'s log
coefficient `Q(m1)` has no constant term (`Q(m1) = b1*m1 + ...`, `Q(0) =
0`) while `ln(m1) -> -infinity` there. Both get the same fix: `m1` is
floored at a small epsilon (branchless, via `numax.core.tensor.max_op`'s
`max(a,b) = (a+b+|a-b|)/2` identity, same as `numax.special.bessel`'s domain
clamps) before it's handed to `ln` -- for `K`, this caps the singularity at
a large-but-finite value rather than reaching a true `+infinity`; for `E`,
it turns the indeterminate `0 * (-infinity)` into an ordinary `(tiny) *
(large but finite)`, which is what the limit actually evaluates to.
"""

from ..core.numeric import FloatLike, max_of


def elliptic_k[T: FloatLike](m: T) -> T:
    """The complete elliptic integral of the first kind, `K(m) =
    integral(0, pi/2, dtheta / sqrt(1 - m*sin(theta)^2))`.

    Valid for `0 <= m < 1`; diverges (to a large-but-finite value, per this
    module's docstring) as `m` approaches `1`.
    """
    var m1 = T.one() - m
    var log_m1 = max_of(m1, T.constant(1e-15)).ln()

    var p = (
        (
            (T.constant(0.01451196212) * m1 + T.constant(0.03742563713)) * m1
            + T.constant(0.03590092383)
        )
        * m1
        + T.constant(0.09666344259)
    ) * m1 + T.constant(1.38629436112)
    var q = (
        (
            (T.constant(0.00441787012) * m1 + T.constant(0.03328355346)) * m1
            + T.constant(0.06880248576)
        )
        * m1
        + T.constant(0.12498593597)
    ) * m1 + T.constant(0.5)

    return p - (q * log_m1)


def elliptic_e[T: FloatLike](m: T) -> T:
    """The complete elliptic integral of the second kind, `E(m) =
    integral(0, pi/2, sqrt(1 - m*sin(theta)^2) dtheta)`.

    Valid for `0 <= m <= 1` (unlike `K`, `E(1) = 1` is finite -- see this
    module's docstring for the `0 * (-infinity)` indeterminate form that
    needs guarding against right there).
    """
    var m1 = T.one() - m
    var log_m1 = max_of(m1, T.constant(1e-15)).ln()

    var p = (
        (
            (T.constant(0.01736506451) * m1 + T.constant(0.04757383546)) * m1
            + T.constant(0.06260601220)
        )
        * m1
        + T.constant(0.44325141463)
    ) * m1 + T.one()
    var q = (
        (
            (T.constant(0.00526449639) * m1 + T.constant(0.04069697526)) * m1
            + T.constant(0.09200180037)
        )
        * m1
        + T.constant(0.24998368310)
    ) * m1

    return p - (q * log_m1)
