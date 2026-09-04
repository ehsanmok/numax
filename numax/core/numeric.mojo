"""The trait every numeric kernel in `numax` is written against.

A kernel bounded on `FloatLike` names no SIMD width, no dtype, and no
instruction set -- only the handful of operations it actually needs. The
_type_ it gets called with then decides what you get out of it: plain SIMD
(`Plain`), a value paired with its derivative (`Dual`), or a value carried
to roughly double precision (`Compensated`). Swap the type, not the kernel.

The trait is intentionally small: enough to build the `special` module's
activations and any similar closed-form kernel on it, and no more. Add an
operation here only when a kernel genuinely needs it -- every method added
is one more thing every future `FloatLike` type must implement.
"""


trait FloatLike(Copyable, Deinitable, Movable):
    """The minimal arithmetic a real-valued vectorized kernel needs."""

    @staticmethod
    def one() -> Self:
        """The multiplicative identity, at whatever width/precision `Self` is.
        """
        ...

    def __add__(self, rhs: Self) -> Self:
        ...

    def __mul__(self, rhs: Self) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __sub__(self, rhs: Self) -> Self:
        """`self - rhs`.

        The one method here with a body rather than `...`: it is exactly
        `self + (-rhs)`, so no conformer implements it and adding it did not
        pay the usual "every present and future conformer must implement
        this" cost. It buys back the 191 call sites that used to spell
        subtraction the long way -- including `Interval.width`, whose
        docstring said `hi - lo` while its body could not.

        A conformer that can subtract more accurately than it can negate and
        add is free to override; none of the ones here can, since negation is
        exact at every one of them.
        """
        return self + (-rhs)

    def __truediv__(self, rhs: Self) -> Self:
        ...

    def exp(self) -> Self:
        ...

    def ln(self) -> Self:
        """The natural logarithm. Only meaningful for `self > 0`."""
        ...

    def sqrt(self) -> Self:
        """The square root. Only meaningful for `self >= 0`.

        Held out of the trait for a long time on the "one call site isn't
        reason enough" rule -- `numax.special.bessel`'s `_inv_sqrt` and
        `numax.special.lambertw`'s branch-point seed each spelled it
        `exp(0.5 * ln(x))` instead. Two things changed: the call sites
        multiplied (Cholesky, complex modulus, distribution normalizers,
        interval bounds, FFT magnitudes), and the workaround turned out to
        be genuinely worse rather than merely verbose. `exp(0.5*ln(x))`
        costs two transcendental calls where hardware has a single `sqrt`
        instruction, and it loses accuracy doing it, since the `ln` result's
        rounding error gets amplified back through `exp`.
        """
        ...

    def erf(self) -> Self:
        """The error function, `(2/sqrt(pi)) * integral(exp(-t^2), 0, self)`."""
        ...

    def erfc(self) -> Self:
        """The complementary error function, `1 - erf(self)`.

        A trait method rather than `Self.one() - self.erf()` uniformly:
        that expression cancels catastrophically for large `self` (`erf`
        is already within rounding distance of `1` there), which is
        exactly the case a conformer with a real `erfc` -- `Plain`, via
        `std.math.erfc` -- exists to avoid. See `default_erf_approx`
        below for the conformers using it as their `erfc`, too.
        """
        ...

    def sin(self) -> Self:
        ...

    def cos(self) -> Self:
        ...

    @staticmethod
    def constant(v: Float64) -> Self:
        """Build a literal coefficient as `Self`.

        Lets a kernel embed constants (polynomial coefficients, etc.)
        without knowing how `Self` represents a value -- `Dual` gives the
        constant a zero derivative, `Compensated` gets to decide how much
        of `v` survives the trip through a single `dtype` lane.
        """
        ...

    def abs(self) -> Self:
        ...

    def copysign(self, sign_source: Self) -> Self:
        """Return `self`'s magnitude with `sign_source`'s sign.

        Lets a kernel derived for `x >= 0` extend to all `x` via
        `result.copysign(x)`, without ever branching on `Self` itself --
        which may hold a SIMD vector with mixed-sign lanes.
        """
        ...

    def floor(self) -> Self:
        """The largest integer value `<= self`, as a `Self`.

        Held out of the trait for a long time on the same "one call site
        isn't reason enough" rule that delayed `sqrt` -- until
        `numax.core.interval`'s `sin`/`cos` needed it for real: a tight
        enclosure has to know whether an input interval contains a peak or
        trough, which is a `floor` of a scaled argument (see
        `numax/core/interval.mojo`'s own docstring). That's a genuine kernel
        that needed a genuinely new capability, the same bar `sqrt`'s
        promotion was held to.
        """
        ...

    def ceil(self) -> Self:
        """The smallest integer value `>= self`, as a `Self`.

        Added alongside `floor` for the same kernel (`numax.core.interval`'s
        tight `sin`/`cos` enclosure needs both directions of rounding, one
        per bound of the interval it's enclosing).
        """
        ...

    def trunc(self) -> Self:
        """`self` rounded toward zero, as a `Self`.

        The third of the pair, added for the same kernel:
        `numax.core.interval`'s peak/trough detection composes `floor`/`ceil`
        of a *shifted* argument, but the branchless index computation
        underneath (turning "does this interval contain a peak" into an
        arithmetic comparison of two rounded quantities) is most directly
        expressed with a round-toward-zero, which `floor`/`ceil` alone
        don't give without an extra sign check.
        """
        ...


def max_of[T: FloatLike](a: T, b: T) -> T:
    """The larger of `a` and `b`, lane-wise and branchless.

    `max(a, b) = (a + b + |a - b|) / 2`, an identity needing only
    operations the trait already has. Lives here, next to the trait, rather
    than in any one kernel: clamping an argument into a function's valid
    domain before it reaches `ln` (so that *both* sides of a blend are safe
    to evaluate everywhere) is the single most common thing every kernel in
    `numax` does, and it shouldn't be re-derived per module.

    One trap, because the identity is arithmetic rather than a hardware
    `max`: it needs `a + b` and `a - b` to both be representable without
    losing the smaller operand. Clamping with a floor many orders of
    magnitude below the other argument doesn't work --
    `max_of(-1.0, 1e-30)` returns exactly `0.0` in float64, not `1e-30`,
    because `-1 + 1e-30` rounds back to `-1` and the two terms cancel. So a
    tiny floor guards a value that's *already near zero* (which is the
    usual case: an argument clamped to its support first) but does not
    rescue an arbitrarily negative one. Clamp to `0` first, or use a floor
    within range of the values actually being clamped.
    """
    var diff = a - b
    return (a + b + diff.abs()) / T.constant(2.0)


def min_of[T: FloatLike](a: T, b: T) -> T:
    """The smaller of `a` and `b`, lane-wise and branchless -- `max_of`'s
    identity with the `|a - b|` term subtracted instead of added."""
    var diff = a - b
    return (a + b - diff.abs()) / T.constant(2.0)


def ge_indicator[T: FloatLike](x: T, threshold: T) -> T:
    """`1` where `x >= threshold`, `0` where `x < threshold` -- branchless.

    Reads the sign of `x - threshold` off via `copysign` (`+1`/`-1`) rather
    than a comparison, then remaps `{-1, +1}` to `{0, 1}`. `copysign`'s
    `+1`-at-exactly-zero convention puts the `x == threshold` boundary on
    the `>=` side.

    This is the primitive that stands in for a per-lane branch throughout
    `numax`: a `Self` may hold a SIMD vector whose lanes disagree about
    which side of `threshold` they're on, and an ordinary `if` would branch
    on the whole vector. Pair it with `blend` below.
    """
    var sign = T.one().copysign(x - threshold)
    return (sign + T.one()) / T.constant(2.0)


def guard_nonzero[T: FloatLike](x: T, floor: T) -> T:
    """`x` with its magnitude raised to at least `floor`, sign preserved.

    The branchless form of "if this denominator has landed on zero, nudge
    it" -- Lentz's continued-fraction guard in `numax.special.beta`, and the
    zero-derivative guard in `numax.optimize`. `copysign`'s `+1`-at-zero
    convention means an exactly-zero `x` comes back as `+floor`.
    """
    return max_of(x.abs(), floor).copysign(x)


def blend[T: FloatLike](indicator: T, if_one: T, if_zero: T) -> T:
    """Pick `if_one` where `indicator` is `1` and `if_zero` where it's `0`.

    The arithmetic stand-in for a per-lane `select`. Both arguments are
    always evaluated by the caller before they get here, so both must be
    numerically valid over the whole domain -- multiplying a NaN by a `0`
    indicator yields NaN, not zero. Clamping the "wrong" side's argument
    into a safe range with `max_of`/`min_of` first is how the kernels here
    guarantee that.
    """
    return if_one * indicator + if_zero * (T.one() - indicator)


def default_erf_approx[T: FloatLike](x: T) -> T:
    """Abramowitz & Stegun 7.1.26's rational approximation of `erf`.

    The shared fallback for `FloatLike.erf()` on conformers with no faster
    or more precise native source -- `Compensated` and `Decimal` both call
    this directly as their own `erf()`. Lives here, next to the trait,
    rather than in `numax.special.erf`, so those base numeric types don't have to
    depend on a special-function module built on top of `FloatLike` --
    `Plain.erf()` skips this entirely in favor of `std.math.erf` (which is
    GPU-compatible and doesn't share this approximation's cancellation
    error near `x = 0`; see `tests/core/test_compensated.mojo`), and `Dual.erf()`
    skips it via the chain rule over its `Inner`'s own `erf()`.

    Max absolute error ~1.5e-7 for `x >= 0`, extended to negative `x` via
    `copysign` since `erf` is odd -- the formula itself is only derived for
    `x >= 0`.
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

    var erf_ax = T.one() - (poly * (-(ax * ax)).exp())
    return erf_ax.copysign(x)
