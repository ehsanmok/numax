"""`Interval[Inner]` -- a value known only to lie between two bounds.

Every operation maps a set of possible inputs to a set of possible outputs,
so running an existing kernel at `Interval` answers "what range can this
produce over that range of inputs" in one call, instead of sampling and
hoping. Nothing in the kernel changes; `Interval` conforms to `FloatLike`
like every other type here, and nests the same way (`Interval[Dual[...]]`
carries interval-valued derivatives).

## How rigorous is this, exactly

Not fully, and the reason is worth stating precisely rather than burying.

A genuine interval library rounds each lower bound *down* and each upper
bound *up*, so the computed interval provably contains the true one. That
needs directed rounding modes per operation, which SIMD and GPU targets do
not reliably expose -- and even if they did, `FloatLike` has no way to
express "round this operation toward negative infinity". Every bound here
is therefore computed in the ambient round-to-nearest mode, and can be off
by a few ULP in the unsafe direction.

The standard fallback is to widen each result by an ULP. That is available
here as `inflate`, but it is *opt-in* rather than automatic, for a second
reason specific to this design: `Inner` is any `FloatLike`, including
`Dual` and `Compensated`, so there is no machine epsilon the type can
consult -- the caller knows their precision and this type does not.

So: treat these bounds as tight-to-a-few-ULP range analysis, which is what
most callers actually want, and call `inflate` with a slack appropriate to
your precision when you need the enclosure to be sound.

## What the operations guarantee

Monotone functions (`exp`, `ln`, `sqrt`, `erf`) map endpoints to endpoints,
`erfc` does the same reversed, and the arithmetic operations take the
extremes over all combinations of endpoints -- all exact enclosures up to
the rounding caveat above.

`sin` and `cos` used to return the trivial `[-1, 1]` rather than a tight
bound, because a tight enclosure needs to know whether the input interval
contains a peak or trough -- a `floor` of a scaled argument -- and
`FloatLike` didn't have one. Now that it does (see `numax/numeric.mojo`'s
`floor` docstring, added specifically for this), `sin`/`cos` below detect
the peak/trough case directly and only fall back to `[-1, 1]` when the
interval genuinely spans both (or more) -- see their own docstrings.

## The dependency problem, which will bite you first

Bounds get looser than the true range whenever a variable appears more than
once, because each occurrence is treated as independent. `x*x` over
`[-1, 2]` returns `[-2, 4]`, not `[0, 4]`, since the multiply sees two
unrelated intervals and `-1 * 2` is a reachable corner of that box even
though `x * x` can never be negative. Run `exp(-(x*x))` over `[-1, 2]` and
you get `[0.018, 7.39]` where the true range is `[0.018, 1]` -- still a
correct enclosure, just a weak one.

This is inherent to interval arithmetic rather than anything specific to
this implementation. The standard mitigations are to rewrite the expression
so each variable appears once, or to split the input into subintervals and
take the union of the results, which tightens quadratically as the
subintervals shrink.

Division by an interval containing zero has no finite enclosure. This
returns whatever the underlying arithmetic produces (an infinity, or a NaN
where the infinities cancel) rather than signalling, because signalling
would mean branching on data, and lanes can disagree about whether their
own divisor straddles zero.
"""

from .numeric import FloatLike, blend, ge_indicator, max_of, min_of

comptime _PI = 3.14159265358979323846
comptime _TWO_PI = 2.0 * _PI
comptime _HALF_PI = _PI / 2.0
comptime _THREE_HALF_PI = 3.0 * _PI / 2.0


def _period_contains[T: FloatLike](lo: T, hi: T, phase: T) -> T:
    """`1` if `[lo, hi]` contains `phase + 2*pi*k` for some integer `k`,
    else `0` -- the branchless building block `sin`/`cos`'s tight
    enclosure is built from below.

    `[lo, hi]` contains such a point exactly when
    `ceil((lo - phase) / (2*pi)) <= floor((hi - phase) / (2*pi))`: the
    smallest candidate `k` at or after `lo` doesn't overshoot past `hi`.
    `ceil` is expressed as `-floor(-x)` (`FloatLike.ceil()` exists too, but
    reusing `floor` for both keeps this to one new capability rather than
    two independent ones, mirroring how `numax.gamma`'s reflection and
    `numax.bessel`'s near/far blend each lean on a single primitive twice).
    """
    var two_pi = T.constant(_TWO_PI)
    var a = (lo + (-phase)) / two_pi
    var b = (hi + (-phase)) / two_pi
    var ceil_a = -((-a).floor())
    var floor_b = b.floor()
    return ge_indicator(floor_b + (-ceil_a), T.constant(0.0))


@fieldwise_init
struct Interval[Inner: FloatLike](Copyable, FloatLike, Movable):
    """A closed interval `[lo, hi]` over any other `FloatLike` type."""

    var lo: Self.Inner
    var hi: Self.Inner

    @staticmethod
    def degenerate(value: Self.Inner) -> Self:
        """The single-point interval `[value, value]`.

        Not part of `FloatLike` -- it's the constructor a caller needs when
        an input is known exactly and only some other input is uncertain.
        """
        return Self(value.copy(), value.copy())

    @staticmethod
    def one() -> Self:
        return Self(Self.Inner.one(), Self.Inner.one())

    @staticmethod
    def constant(v: Float64) -> Self:
        return Self(Self.Inner.constant(v), Self.Inner.constant(v))

    def width(self) -> Self.Inner:
        """`hi - lo`, the size of the interval -- an `Inner`, not an
        `Interval`, since the width of a range is a single number."""
        return self.hi + (-self.lo)

    def midpoint(self) -> Self.Inner:
        return (self.lo + self.hi) / Self.Inner.constant(2.0)

    def inflate(self, relative: Float64) -> Self:
        """Widen both bounds by `relative` times the current width, plus the
        same fraction of each bound's own magnitude.

        The manual stand-in for directed rounding described in this module's
        docstring. Pass something on the order of a few times your
        precision's machine epsilon (about `1e-7` for `float32`, `2e-16` for
        `float64`) to absorb the rounding error accumulated so far.
        """
        var pad_from_width = self.width() * Self.Inner.constant(relative)
        var pad_lo = pad_from_width + self.lo.abs() * Self.Inner.constant(
            relative
        )
        var pad_hi = pad_from_width + self.hi.abs() * Self.Inner.constant(
            relative
        )
        return Self(self.lo + (-pad_lo), self.hi + pad_hi)

    def __add__(self, rhs: Self) -> Self:
        return Self(self.lo + rhs.lo, self.hi + rhs.hi)

    def __neg__(self) -> Self:
        # Negation flips the interval end for end.
        return Self(-self.hi, -self.lo)

    def __mul__(self, rhs: Self) -> Self:
        # The extremes of a product over a box are attained at its corners,
        # so all four are computed and the smallest and largest kept. Doing
        # it by sign analysis instead would need per-lane branching for no
        # gain -- four multiplies is cheaper than the branches would be.
        var ll = self.lo * rhs.lo
        var lh = self.lo * rhs.hi
        var hl = self.hi * rhs.lo
        var hh = self.hi * rhs.hi
        return Self(
            min_of(min_of(ll, lh), min_of(hl, hh)),
            max_of(max_of(ll, lh), max_of(hl, hh)),
        )

    def __truediv__(self, rhs: Self) -> Self:
        # Multiplication by the reciprocal interval. If `rhs` straddles
        # zero, its reciprocal endpoints are infinite and the result is
        # meaningless -- see this module's docstring for why that isn't
        # detected here.
        var one = Self.Inner.one()
        return self * Self(one / rhs.hi, one / rhs.lo)

    def exp(self) -> Self:
        return Self(self.lo.exp(), self.hi.exp())

    def ln(self) -> Self:
        return Self(self.lo.ln(), self.hi.ln())

    def sqrt(self) -> Self:
        return Self(self.lo.sqrt(), self.hi.sqrt())

    def erf(self) -> Self:
        return Self(self.lo.erf(), self.hi.erf())

    def erfc(self) -> Self:
        # Decreasing, so the endpoints swap.
        return Self(self.hi.erfc(), self.lo.erfc())

    def sin(self) -> Self:
        """A tight enclosure of `{sin(x) : x in [lo, hi]}`.

        `sin`'s extrema alternate every `pi`: a peak (`sin = 1`) at
        `pi/2 + 2*k*pi`, a trough (`sin = -1`) at `3*pi/2 + 2*k*pi`, for
        every integer `k`. `_period_contains` (module-level, above) tests
        each branchlessly; three cases collapse into two independent
        `blend`s because each bound only ever depends on one flag:

        - the lower bound is `-1` whenever a trough is enclosed, and
          `min(sin(lo), sin(hi))` otherwise (whether or not a peak is also
          enclosed doesn't change which value is smaller);
        - the upper bound is symmetric, `1` whenever a peak is enclosed,
          `max(sin(lo), sin(hi))` otherwise.

        Neither flag set means `[lo, hi]` contains no extremum at all,
        which (since extrema are spaced `pi` apart) also proves `sin` is
        monotonic on `[lo, hi]` -- so the two endpoint values already
        bracket every value in between.
        """
        var sin_lo = self.lo.sin()
        var sin_hi = self.hi.sin()
        var has_peak = _period_contains(
            self.lo, self.hi, Self.Inner.constant(_HALF_PI)
        )
        var has_trough = _period_contains(
            self.lo, self.hi, Self.Inner.constant(_THREE_HALF_PI)
        )
        var one = Self.Inner.one()
        var lo = blend(has_trough, -one, min_of(sin_lo.copy(), sin_hi.copy()))
        var hi = blend(has_peak, Self.Inner.one(), max_of(sin_lo, sin_hi))
        return Self(lo^, hi^)

    def cos(self) -> Self:
        """A tight enclosure of `{cos(x) : x in [lo, hi]}`, the same
        construction as `sin` shifted by a quarter period: `cos`'s peaks
        are at `2*k*pi`, its troughs at `pi + 2*k*pi`."""
        var cos_lo = self.lo.cos()
        var cos_hi = self.hi.cos()
        var has_peak = _period_contains(
            self.lo, self.hi, Self.Inner.constant(0.0)
        )
        var has_trough = _period_contains(
            self.lo, self.hi, Self.Inner.constant(_PI)
        )
        var one = Self.Inner.one()
        var lo = blend(has_trough, -one, min_of(cos_lo.copy(), cos_hi.copy()))
        var hi = blend(has_peak, Self.Inner.one(), max_of(cos_lo, cos_hi))
        return Self(lo^, hi^)

    def abs(self) -> Self:
        """`[mig, mag]`: the smallest and largest magnitude in the interval.

        The lower bound is zero whenever the interval straddles zero, which
        `max_of(0, max_of(lo, -hi))` produces without a branch -- the inner
        `max_of` is negative exactly when `lo < 0 < hi`.
        """
        var zero = Self.Inner.constant(0.0)
        var mig = max_of(zero, max_of(self.lo.copy(), -self.hi))
        var mag = max_of(self.lo.abs(), self.hi.abs())
        return Self(mig^, mag^)

    def copysign(self, sign_source: Self) -> Self:
        """The set of `copysign(x, s)` for `x` in `self` and `s` in
        `sign_source`.

        Three cases, blended arithmetically rather than branched: a source
        entirely at or above zero gives `[mig, mag]`, one entirely below
        gives `[-mag, -mig]`, and one straddling zero gives `[-mag, mag]`,
        since both signs are then reachable. Zero counts as positive here,
        matching `copysign`'s own convention throughout `numax`.
        """
        var zero = Self.Inner.constant(0.0)
        var magnitudes = self.abs()
        var all_positive = ge_indicator(sign_source.lo.copy(), zero)
        var all_negative = ge_indicator(-sign_source.hi, zero) * (
            Self.Inner.one() + (-all_positive)
        )
        var straddles = Self.Inner.one() + (-all_positive) + (-all_negative)

        var lo = (
            all_positive * magnitudes.lo
            + all_negative * (-magnitudes.hi)
            + straddles * (-magnitudes.hi)
        )
        var hi = (
            all_positive * magnitudes.hi
            + all_negative * (-magnitudes.lo)
            + straddles * magnitudes.hi
        )
        return Self(lo^, hi^)

    def floor(self) -> Self:
        """`[floor(lo), floor(hi)]` -- `floor` is monotonic non-decreasing,
        so applying it to each bound separately still encloses every
        `floor(x)` for `x` in `self`, and does so tightly."""
        return Self(self.lo.floor(), self.hi.floor())

    def ceil(self) -> Self:
        """`[ceil(lo), ceil(hi)]`, for the same monotonicity reason as
        `floor`."""
        return Self(self.lo.ceil(), self.hi.ceil())

    def trunc(self) -> Self:
        """`[min(trunc(lo), trunc(hi)), max(trunc(lo), trunc(hi))]`.

        Unlike `floor`/`ceil`, `trunc` (round toward zero) is not monotonic
        across zero -- `trunc(-0.5) = 0 > trunc(-1.5) = -1`, so applying it
        to `lo` and `hi` in order doesn't guarantee `lo_result <=
        hi_result` the way `floor`/`ceil` do. Sorting the two `trunc`ed
        bounds keeps this a valid interval (`lo <= hi`) in every case,
        including one that straddles zero.
        """
        var t_lo = self.lo.trunc()
        var t_hi = self.hi.trunc()
        return Self(min_of(t_lo.copy(), t_hi.copy()), max_of(t_lo, t_hi))
