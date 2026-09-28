"""Complete elliptic integrals of the first and second kind, `K(m)` and
`E(m)` (parameterized by `m = k^2`, following `std`'s and A&S's own
convention -- not the modulus `k` itself), the incomplete `F(phi | m)` and
`E(phi | m)` as `ellipkinc`/`ellipeinc`, and Carlson's symmetric forms
`elliprf` and `elliprd` they are built on.

The incomplete pair is Carlson's: `F = sin(phi) R_F(cos^2, 1 - m sin^2,
1)` and `E` that minus `(m/3) sin^3 R_D(...)`, after reducing the
amplitude by the period `F(phi + pi) = F(phi) + 2 K(m)`. `R_F` and `R_D`
run a fixed sixteen duplication steps and Carlson's fifth-order series,
which `pixi run accuracy` reads at a few ulp, `1e-15` relative, including
`m = 0.99` and negative `m`. The polynomial `K`/`E` below are the older,
`2e-8` pair.

**This module is tier 1.** `m1` is floored at a small positive epsilon
before it reaches `ln`, branchlessly, which is what keeps both functions
finite at their shared singular point.

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
floored at a small epsilon (branchless, via `numax.core.functional.max_op`,
the exact selection `numax.core.numeric.max_of` is, same as
`numax.special.bessel`'s domain clamps) before it's handed to `ln` -- for `K`, this caps the singularity at
a large-but-finite value rather than reaching a true `+infinity`; for `E`,
it turns the indeterminate `0 * (-infinity)` into an ordinary `(tiny) *
(large but finite)`, which is what the limit actually evaluates to.
"""

from ..core.numeric import FloatLike, max_of, min_of


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


comptime _DUPLICATIONS = 16
"""Carlson duplication steps. Each divides the arguments' spread by 4, and
four take comparable arguments to the point where the fifth-order series
is exact at `float64`. The other twelve are for arguments far apart, whose
smallest takes a step per halving of its exponent to catch up: `1e-300`
against `1` needs about nine."""


def elliprf[T: FloatLike](x: T, y: T, z: T) -> T:
    """Carlson's symmetric elliptic integral of the first kind,
    `R_F(x, y, z) = (1/2) integral(0, inf, dt / sqrt((t+x)(t+y)(t+z)))`,
    for `x, y, z >= 0` with at most one of them zero.
    `scipy.special.elliprf`.

    Tier 1. `_DUPLICATIONS` duplication steps, `v -> (v + lambda) / 4` with
    `lambda = sqrt(xy) + sqrt(xz) + sqrt(yz)`, then Carlson's fifth-order
    series about the mean (*Numerical Recipes*, 3rd ed., 6.12, without its
    convergence test).

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the inputs.

    Args:
        x: The first argument, `x >= 0`.
        y: The second argument, `y >= 0`.
        z: The third argument, `z >= 0`.

    Returns:
        `R_F(x, y, z)`.
    """
    var quarter = T.constant(0.25)
    var xt = x.copy()
    var yt = y.copy()
    var zt = z.copy()
    for _ in range(_DUPLICATIONS):
        var sx = xt.sqrt()
        var sy = yt.sqrt()
        var sz = zt.sqrt()
        var lam = sx * (sy + sz) + sy * sz
        xt = quarter * (xt + lam)
        yt = quarter * (yt + lam)
        zt = quarter * (zt + lam)
    var ave = (xt + yt + zt) / T.constant(3.0)
    var dx = (ave - xt) / ave
    var dy = (ave - yt) / ave
    var dz = (ave - zt) / ave
    var e2 = dx * dy - dz * dz
    var e3 = dx * dy * dz
    return (
        T.one()
        + (
            T.constant(1.0 / 24.0) * e2
            - T.constant(0.1)
            - T.constant(3.0 / 44.0) * e3
        )
        * e2
        + T.constant(1.0 / 14.0) * e3
    ) / ave.sqrt()


def elliprd[T: FloatLike](x: T, y: T, z: T) -> T:
    """Carlson's elliptic integral of the second kind,
    `R_D(x, y, z) = (3/2) integral(0, inf, dt / ((t+z) sqrt((t+x)(t+y)(t+z))))`,
    for `x, y >= 0` not both zero and `z > 0`. `scipy.special.elliprd`.

    Tier 1: the duplication `elliprf` uses, accumulating the `R_D` sum at
    each step, then Carlson's fifth-order series (*Numerical Recipes*,
    3rd ed., 6.12, without its convergence test).

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the inputs.

    Args:
        x: The first argument, `x >= 0`.
        y: The second argument, `y >= 0`.
        z: The third argument, `z > 0`.

    Returns:
        `R_D(x, y, z)`.
    """
    var quarter = T.constant(0.25)
    var xt = x.copy()
    var yt = y.copy()
    var zt = z.copy()
    var total = T.constant(0.0)
    var fac = T.one()
    for _ in range(_DUPLICATIONS):
        var sx = xt.sqrt()
        var sy = yt.sqrt()
        var sz = zt.sqrt()
        var lam = sx * (sy + sz) + sy * sz
        total = total + fac / (sz * (zt + lam))
        fac = quarter * fac
        xt = quarter * (xt + lam)
        yt = quarter * (yt + lam)
        zt = quarter * (zt + lam)
    var ave = T.constant(0.2) * (xt + yt + T.constant(3.0) * zt)
    var dx = (ave - xt) / ave
    var dy = (ave - yt) / ave
    var dz = (ave - zt) / ave
    var ea = dx * dy
    var eb = dz * dz
    var ec = ea - eb
    var ed = ea - T.constant(6.0) * eb
    var ee = ed + ec + ec
    comptime c1 = 3.0 / 14.0
    comptime c2 = 1.0 / 6.0
    comptime c3 = 9.0 / 22.0
    comptime c4 = 3.0 / 26.0
    var series = (
        T.one()
        + ed
        * (
            -T.constant(c1)
            + T.constant(0.25 * c3) * ed
            - T.constant(1.5 * c4) * dz * ee
        )
        + dz
        * (
            T.constant(c2) * ee
            + dz * (-T.constant(c3) * ec + dz * T.constant(c4) * ea)
        )
    )
    return T.constant(3.0) * total + fac * series / (ave * ave.sqrt())


def _elliprf0[T: FloatLike](y: T, z: T) -> T:
    """`R_F(0, y, z)`, the complete integral's form, with the first
    duplication step taken by hand: `lambda = sqrt(yz)`, since `sqrt(0)`
    would put `0/0` in a `Dual`'s derivative even though the zero is a
    constant. `R_F` is invariant under the step."""
    var lam = (y * z).sqrt()
    var quarter = T.constant(0.25)
    return elliprf(quarter * lam, quarter * (y + lam), quarter * (z + lam))


def _elliprd0[T: FloatLike](y: T, z: T) -> T:
    """`R_D(0, y, z)`, the first duplication step taken by hand as for
    `_elliprf0`: `R_D(x, y, z) = 3 / (sqrt(z) (z + lambda)) + R_D(x', y',
    z') / 4`."""
    var lam = (y * z).sqrt()
    var quarter = T.constant(0.25)
    return T.constant(3.0) / (z.sqrt() * (z + lam)) + quarter * elliprd(
        quarter * lam, quarter * (y + lam), quarter * (z + lam)
    )


def _reduce_amplitude[T: FloatLike](phi: T) -> Tuple[T, T]:
    """`phi = k pi + r` with `r` in `[-pi/2, pi/2]`: `(k, r)`, `k` as a
    `T` holding an integer."""
    comptime pi = 3.141592653589793
    var k = (phi / T.constant(pi) + T.constant(0.5)).floor()
    return (k.copy(), phi - k * T.constant(pi))


def ellipkinc[T: FloatLike](phi: T, m: T) -> T:
    """The incomplete elliptic integral of the first kind,
    `F(phi | m) = integral(0, phi, dt / sqrt(1 - m sin(t)^2))`.
    `scipy.special.ellipkinc(phi, m)`, in the parameter `m = k^2` as
    `elliptic_k` takes it.

    Tier 1. The amplitude is reduced to `r` in `[-pi/2, pi/2]` with `phi =
    k pi + r`, and `F = 2 k K(m) + sin(r) R_F(cos(r)^2, 1 - m sin(r)^2,
    1)`, the complete integral `K(m) = R_F(0, 1 - m, 1)` taken from the
    same `elliprf` rather than `elliptic_k`'s `2e-8` polynomial.

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the inputs.

    Args:
        phi: The amplitude, in radians.
        m: The parameter, `m < 1`; `m = 1` is allowed for `|phi| < pi/2`,
            where `F = atanh(sin(phi))`.

    Returns:
        `F(phi | m)`.
    """
    var reduced = _reduce_amplitude(phi)
    var k = reduced[0].copy()
    var r = reduced[1].copy()
    var s = r.sin()
    var c = r.cos()
    var one = T.one()
    var partial = s * elliprf(c * c, one - m * s * s, one)
    # The period term at `m` held below 1, so `K(m)` stays finite on the
    # lanes whose `k` is 0 and discards it; at `m >= 1` a nonzero `k` has no
    # finite answer.
    var mc = min_of(m, T.constant(0.9999999999999999))
    var complete = _elliprf0(one - mc, one)
    return T.constant(2.0) * k * complete + partial


def ellipeinc[T: FloatLike](phi: T, m: T) -> T:
    """The incomplete elliptic integral of the second kind,
    `E(phi | m) = integral(0, phi, sqrt(1 - m sin(t)^2) dt)`.
    `scipy.special.ellipeinc(phi, m)`, in the parameter `m = k^2`.

    Tier 1. As `ellipkinc`, with `E = 2 k E(m) + sin(r) R_F(c^2, d, 1) - (m /
    3) sin(r)^3 R_D(c^2, d, 1)` for `c = cos(r)` and `d = 1 - m sin(r)^2`,
    and the complete `E(m) = R_F(0, 1 - m, 1) - (m / 3) R_D(0, 1 - m, 1)`.

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the inputs.

    Args:
        phi: The amplitude, in radians.
        m: The parameter, `m <= 1`.

    Returns:
        `E(phi | m)`.
    """
    var reduced = _reduce_amplitude(phi)
    var k = reduced[0].copy()
    var r = reduced[1].copy()
    var s = r.sin()
    var c = r.cos()
    var one = T.one()
    var third = T.constant(1.0 / 3.0)
    var cc = c * c
    var d = one - m * s * s
    var partial = s * elliprf(cc, d, one) - third * m * s * s * s * elliprd(
        cc, d, one
    )
    var q = one - m
    var complete = _elliprf0(q, one) - third * m * _elliprd0(q, one)
    return T.constant(2.0) * k * complete + partial
