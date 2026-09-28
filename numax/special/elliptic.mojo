"""Complete elliptic integrals of the first and second kind, `K(m)` and
`E(m)` (parameterized by `m = k^2`, following `std`'s and A&S's own
convention -- not the modulus `k` itself), the incomplete `F(phi | m)` and
`E(phi | m)` as `ellipkinc`/`ellipeinc`, and Carlson's symmetric forms
`elliprf` and `elliprd` all of them are built on.

**This module is tier 1.** `R_F` and `R_D` run a fixed sixteen
duplication steps and Carlson's fifth-order series, with no convergence
test; everything else is a closed form over them, and the edge cases are
substitutions and blends rather than branches.

- `K(m) = R_F(0, 1 - m, 1)` and `E(m) = R_F(0, 1 - m, 1) - (m/3) R_D(0,
  1 - m, 1)`, with the first duplication step taken by hand for the zero
  argument, since `sqrt(0)` would put `0/0` in a `Dual`'s derivative.
  `K(1)` is `inf` and `E(1)` is `1`, as SciPy's.
- `F(phi | m) = sin(phi) R_F(cos^2, 1 - m sin^2, 1)` and `E(phi | m)`
  that minus `(m/3) sin^3 R_D(...)`, after reducing the amplitude by the
  period `F(phi + pi) = F(phi) + 2 K(m)`.

`pixi run accuracy` reads all of them at a few ulp, `1e-15` relative,
including `m = 0.99` and negative `m`. Before 0.3 `K`/`E` were Abramowitz
and Stegun 17.3.34/17.3.36's polynomial-plus-log approximations at
`~2e-8` -- the second only after its misdigitized `b4` coefficient was
recovered by fitting against an AGM reference -- and the Carlson forms
replaced them outright.

## The MAX gate

Nothing: neither `std.math` nor any MAX root has an elliptic integral.
**Extend.**
"""

from ..core.numeric import FloatLike, blend, ge_indicator, min_of


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
    same `elliprf` that `elliptic_k` is.

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


def elliptic_k[T: FloatLike](m: T) -> T:
    """The complete elliptic integral of the first kind, `K(m) =
    integral(0, pi/2, dtheta / sqrt(1 - m sin(theta)^2))`.
    `scipy.special.ellipk(m)`.

    `R_F(0, 1 - m, 1)`, per this module's docstring: `1e-15` relative for
    every `m < 1`, negative `m` included, `inf` at `m = 1` and NaN above
    it.

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the input.

    Args:
        m: The parameter, `m <= 1`.

    Returns:
        `K(m)`.
    """
    var one = T.one()
    var q = one - m
    # `m = 1` runs at a stand-in `q` and is divided to `inf` afterwards,
    # since the duplication of `R_F(0, 0, 1)` is finite but meaningless.
    var at_one = ge_indicator(T.constant(0.0), q.abs())
    var k = _elliprf0(q + at_one * T.constant(0.5), one)
    return k / (one - at_one)


def elliptic_e[T: FloatLike](m: T) -> T:
    """The complete elliptic integral of the second kind, `E(m) =
    integral(0, pi/2, sqrt(1 - m sin(theta)^2) dtheta)`.
    `scipy.special.ellipe(m)`.

    `R_F(0, 1 - m, 1) - (m/3) R_D(0, 1 - m, 1)`, per this module's
    docstring: `1e-15` relative for `m < 1`, and exactly `1` at `m = 1`,
    where the two terms would each be infinite.

    Parameters:
        T: The `FloatLike` conformer, scalar or SIMD, of the input.

    Args:
        m: The parameter, `m <= 1`.

    Returns:
        `E(m)`.
    """
    var one = T.one()
    var q = one - m
    var at_one = ge_indicator(T.constant(0.0), q.abs())
    var qs = q + at_one * T.constant(0.5)
    var e = _elliprf0(qs, one) - T.constant(1.0 / 3.0) * m * _elliprd0(qs, one)
    return blend(at_one, one, e)
