"""The Lambert W function: the principal branch (`lambertw`, `x >= -1/e`)
and the other real branch (`lambertw_m1`, `-1/e <= x < 0`).

Both branches solve the same equation, `w*exp(w) = x`, which has one real
solution for `x > 0` and two for `x` in `[-1/e, 0)` -- `lambertw` is the one
that's `>= -1`, continuous with `x >= 0`'s single solution; `lambertw_m1` is
the other one, `<= -1`, diverging to `-infinity` as `x` approaches `0` from
below. They meet at the branch point `x = -1/e`, `w = -1`.

Both use Halley's method (see `numax.special.gamma`'s docstring for why a fixed
iteration count, not a data-dependent tolerance) -- the interesting part is
the seed, since a seed that's excellent away from the branch point can be
badly wrong (even wrong-branch-shaped) near it, and vice versa:

- `lambertw`'s existing `ln(1+x)` seed turns out not to need any change to
  cover the extended domain at all: `ln(1+x)` stays defined all the way
  down to `x = -1/e` (`1+x` only reaches `0` at `x = -1`, well outside this
  domain), and empirically (checked in Python before writing this, no
  independent Mojo-side seed logic needed) the same fixed 20 Halley
  iterations already used for `x >= 0` converge to full precision even at
  `x` a nanometer above `-1/e`, where the seed error is largest (Halley's
  cubic convergence recovers fast once it's in the right basin, even from a
  seed that's off by more than half a unit in `w`).
- `lambertw_m1` has no such luck with a single seed shape: the branch-point
  series below (accurate near `-1/e`, where `w` is close to `-1`) is
  numerically *terrible* far from it (checked directly: at `x = -1e-30`,
  its seed error is ~70 and 30 fixed Halley iterations aren't enough to
  claw back from that), while the standard `ln(-x) - ln(-ln(-x))`
  double-log seed (accurate for small `|x|`, where `w` is very negative) is
  terrible *near* the branch point for the opposite reason. `lambertw_m1`
  picks between them with the same branchless blend `numax.special.gamma`'s
  reflection and `numax.special.bessel`'s near/far split use: both seeds are always
  valid to compute anywhere in `[-1/e, 0)` (neither one's formula has a
  domain restriction inside this range, only an accuracy cliff), so a
  `0`/`1` indicator can pick the right one per SIMD lane with no risk of
  selecting a NaN.
"""

from ..core.numeric import FloatLike, max_of

comptime _E = 2.718281828459045235360287471352662497757


def lambertw[T: FloatLike](x: T) -> T:
    """The principal branch of the Lambert W function: solves `w*exp(w) = x`
    for `w >= -1`. Valid for `x >= -1/e` (the branch point where this and
    `lambertw_m1` meet); undefined below it, since there's no real solution
    there at all.

    Halley's method, seeded from `ln(1 + x)` and iterated a **fixed** number
    of times rather than to a data-dependent tolerance (see `numax.special.gamma`'s
    docstring for why). See this module's docstring for why the same seed
    and iteration count that work for `x >= 0` turn out to already cover
    `x` down to `-1/e` too, with no extra branch-point handling needed.
    """
    comptime num_iters = 20

    var w = (T.one() + x).ln()

    for _ in range(num_iters):
        var ew = w.exp()
        var f = w * ew + (-x)
        var fprime = ew * (w + T.one())
        var fprime2 = ew * (w + T.constant(2.0))
        var numerator = T.constant(2.0) * f * fprime
        var denominator = T.constant(2.0) * (fprime * fprime) + (-(f * fprime2))
        w = w + (-(numerator / denominator))

    return w^


def _branch_point_arg[T: FloatLike](x: T) -> T:
    """`2*(e*x + 1)`, clamped to `>= 0` so a lane where `x` is a hair below
    `-1/e` from rounding noise doesn't hand a negative value to `sqrt`.
    """
    return max_of(
        T.constant(2.0) * (T.constant(_E) * x + T.one()), T.constant(0.0)
    )


def _branch_point_seed[T: FloatLike](x: T, negate_odd_terms: T) -> T:
    """The Lambert W branch-point series (Corless et al. 1996): with `p =
    sqrt(2*(e*x+1))`, `W0(x) = -1 + p - p^2/3 + 11p^3/72 - 43p^4/540 + ...`
    and `W_{-1}(x)` is the same series with every odd power of `p` negated.
    `negate_odd_terms` is the sign multiplying `p` and `p^3`: `T.constant
    (1.0)` reproduces `W0`'s series, `T.constant(-1.0)` gives `W_{-1}`'s
    (`lambertw` above doesn't use this seed at all -- this exists only for
    `lambertw_m1`'s blend below, but takes a sign parameter rather than
    hardcoding `W_{-1}`'s so it could be checked against the plain
    `ln(1+x)` seed near the branch point during development).
    """
    var p = _branch_point_arg(x).sqrt()
    var p2 = p * p
    var p3 = p2 * p
    var p4 = p3 * p
    return (
        -T.one()
        + negate_odd_terms * p
        + (-(p2 / T.constant(3.0)))
        + negate_odd_terms * (T.constant(11.0) / T.constant(72.0)) * p3
        + (-(T.constant(43.0) / T.constant(540.0)) * p4)
    )


def lambertw_m1[T: FloatLike](x: T) -> T:
    """The other real branch of the Lambert W function: solves `w*exp(w) =
    x` for `w <= -1`. Valid for `-1/e <= x < 0` -- `w` diverges to
    `-infinity` as `x` approaches `0` (a true singularity, not just a
    scoping choice, so `x = 0` itself isn't included).

    Halley's method again, but the seed is a branchless blend of two
    formulas rather than one (see this module's docstring for why a single
    seed doesn't cover this branch's whole domain the way `lambertw`'s
    does): the branch-point series (`_branch_point_seed`, accurate near
    `x = -1/e`) where `2*(e*x+1) < 0.5`, and the standard double-log seed
    `ln(-x) - ln(-ln(-x))` (accurate for small `|x|`) otherwise. The `0.5`
    cutoff was picked empirically in Python: it's comfortably inside where
    each seed is still accurate enough for 30 fixed Halley iterations to
    reach full precision, checked from `x = -1/e + 1e-9` out to `x = -1e-30`.
    """
    comptime num_iters = 30

    var arg = _branch_point_arg(x)
    var near_branch_point = T.one().copysign(T.constant(0.5) + (-arg))
    var s = (near_branch_point + T.one()) / T.constant(2.0)

    var series_seed = _branch_point_seed(x, T.constant(-1.0))
    var neg_x = -x
    var loglog_seed = neg_x.ln() + (-((-(neg_x.ln())).ln()))

    var w = series_seed * s + loglog_seed * (T.one() + (-s))

    for _ in range(num_iters):
        var ew = w.exp()
        var f = w * ew + (-x)
        var fprime = ew * (w + T.one())
        var fprime2 = ew * (w + T.constant(2.0))
        var numerator = T.constant(2.0) * f * fprime
        var denominator = T.constant(2.0) * (fprime * fprime) + (-(f * fprime2))
        w = w + (-(numerator / denominator))

    return w^
