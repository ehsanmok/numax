"""Double-double compensated arithmetic: a value plus its rounding error.

An ordinary float discards the bits below its own precision on every
operation. `Compensated` recovers them: every operation here is an
error-free transformation (`two_sum`, `two_prod`, built on `fma`) that
computes both the rounded result and the exact remainder that rounding threw
away, then carries that remainder into the next operation. The pair
`(value, error)` together represents a number to roughly twice `dtype`'s
precision -- a double-double.

This is a `dtype`-relative doubling, not a fixed bit count: `Compensated`
over `float32` recovers ~48 bits (roughly `float64`-like precision out of
`float32` lanes), and over `float64` it recovers ~106 bits.

Every renormalization step below (folding a correction back into `(value,
error)`) is itself a `two_sum`, spelled out inline rather than factored into
a shared helper -- Mojo tuples add friction here, and the four call sites
differ enough (some already start from an error-free sum, some need it
computed from scratch) that a shared helper wouldn't stay simple.
"""

from std.collections import Array
from std.math import copysign, exp2, fma, log, round

from .numeric import FloatLike


def _split_f64[
    dtype: DType, width: Int
](v: Float64) -> Tuple[SIMD[dtype, width], SIMD[dtype, width]]:
    """Split a `Float64` literal into a `dtype`-native hi/lo pair.

    Evaluated entirely at compile time (see its call sites in `exp()`, all
    under `comptime for`), so the float64 arithmetic here never reaches
    device code -- only the resulting `dtype`-native `hi`/`lo` constants do.
    This mirrors `Compensated.constant()`'s split, done once per coefficient
    instead of once per call.
    """
    var v64 = SIMD[DType.float64, width](v)
    var hi = v64.cast[dtype]()
    var lo = (v64 - hi.cast[DType.float64]()).cast[dtype]()
    return (hi, lo)


@fieldwise_init
struct Compensated[dtype: DType, width: Int](
    Deinitable,
    FloatLike where dtype.is_floating_point(),
    ImplicitlyCopyable,
    Movable,
):
    """`value` is the rounded result; `error` is what rounding it discarded."""

    var value: SIMD[Self.dtype, Self.width]
    var error: SIMD[Self.dtype, Self.width]

    @staticmethod
    def one() -> Self:
        return Self(
            SIMD[Self.dtype, Self.width](1), SIMD[Self.dtype, Self.width](0)
        )

    def __add__(self, rhs: Self) -> Self:
        # two_sum(self.value, rhs.value), then fold in both error terms and
        # renormalize with a second two_sum.
        var s = self.value + rhs.value
        var bb = s - self.value
        var err = (self.value - (s - bb)) + (rhs.value - bb)

        var e2 = err + self.error + rhs.error

        var s2 = s + e2
        var bb2 = s2 - s
        var err2 = (s - (s2 - bb2)) + (e2 - bb2)

        return Self(s2, err2)

    def __neg__(self) -> Self:
        return Self(-self.value, -self.error)

    def __mul__(self, rhs: Self) -> Self:
        # two_prod(self.value, rhs.value), using FMA for the exact remainder,
        # then fold in the cross terms and renormalize.
        var p = self.value * rhs.value
        var p_err = fma(self.value, rhs.value, -p)

        var e2 = p_err + self.value * rhs.error + self.error * rhs.value

        var p2 = p + e2
        var bb = p2 - p
        var err2 = (p - (p2 - bb)) + (e2 - bb)

        return Self(p2, err2)

    def __truediv__(self, rhs: Self) -> Self:
        # Standard double-double division: one float division for a first
        # estimate, one compensated multiply-and-subtract to find its
        # residual, one more float division to correct for that residual,
        # then a two_sum to fold the two estimates back into one pair.
        var q1 = self.value / rhs.value
        var residual = self + (
            -(Self(q1, SIMD[Self.dtype, Self.width](0)) * rhs)
        )
        var q2 = residual.value / rhs.value

        var s = q1 + q2
        var bb = s - q1
        var err = (q1 - (s - bb)) + (q2 - bb)

        return Self(s, err)

    def exp(self) -> Self where Self.dtype.is_floating_point():
        comptime DT = Self.dtype
        comptime W = Self.width

        # log2(e) and a hi/lo split of ln(2), each exact once rounded to `DT`.
        comptime log2e = SIMD[DT, W](1.4426950408889634)
        comptime ln2_hi = SIMD[DT, W](0.6931471824645996)
        comptime ln2_lo = SIMD[DT, W](-1.904654323148236e-09)

        # Range reduction: x = m*ln2 + r, |r| <= ln2/2, kept as a
        # compensated pair so the reduced argument doesn't lose precision.
        var m = round(self.value * log2e)

        var t = m * ln2_hi
        var t_err = fma(m, ln2_hi, -t)

        var s = self.value - t
        var s_bb = s - self.value
        var s_err = (self.value - (s - s_bb)) + (-t - s_bb)

        var r = Self(s, (self.error - t_err) - m * ln2_lo + s_err)

        # exp(r) = sum_k r^k / k!, accumulated in compensated arithmetic.
        # Coefficients are 1/k! for k=1..14 -- 14 terms lands comfortably
        # past float32-double-double precision for |r| <= ln(2)/2. Each is
        # a `Float64` literal only at compile time: `_split_f64` runs under
        # `comptime for`, so every `hi`/`lo` pair below is baked in as a
        # `DT`-native constant, with no runtime float64 arithmetic (or
        # storage) reaching device code -- see `_split_f64`'s docstring.
        comptime num_terms = 14
        comptime coef: Array[Float64, num_terms] = [
            1.0,
            0.5,
            0.16666666666666666,
            0.041666666666666664,
            0.008333333333333333,
            0.001388888888888889,
            0.0001984126984126984,
            2.48015873015873e-05,
            2.7557319223985893e-06,
            2.755731922398589e-07,
            2.505210838544172e-08,
            2.08767569878681e-09,
            1.6059043836821613e-10,
            1.1470745597729725e-11,
        ]

        var term = Self(SIMD[DT, W](1), SIMD[DT, W](0))
        var total = term

        comptime for k in range(num_terms):
            comptime split = _split_f64[DT, W](coef[k])
            term = term * r
            total = total + term * Self(split[0], split[1])

        # 2**m is exact in float, so scaling by it introduces no new error.
        var scale = exp2(m)
        return Self(total.value * scale, total.error * scale)

    def ln(self) -> Self where Self.dtype.is_floating_point():
        # Newton's method on f(y) = exp(y) - self, i.e. y_{n+1} = y_n - 1 +
        # self*exp(-y_n) -- quadratically convergent, so seeding it with an
        # ordinary (single-`dtype`) `log` estimate and iterating a couple of
        # fixed times roughly squares the number of correct digits each
        # time, using nothing but the compensated `exp`/`__add__`/`__mul__`
        # already implemented above.
        comptime DT = Self.dtype
        comptime W = Self.width
        var y = Self(log(self.value), SIMD[DT, W](0))

        comptime num_iters = 3
        comptime for _ in range(num_iters):
            y = y + (-Self.one()) + self * (-y).exp()

        return y

    @staticmethod
    def constant(v: Float64) -> Self:
        # Split the literal into a hi/lo pair at `dtype`'s precision, rather
        # than just rounding it once and calling the residual zero -- a
        # kernel's coefficients get the same double-`dtype` treatment as
        # everything computed from them.
        var v64 = SIMD[DType.float64, Self.width](v)
        var hi = v64.cast[Self.dtype]()
        var lo = (v64 - hi.cast[DType.float64]()).cast[Self.dtype]()
        return Self(hi, lo)

    def abs(self) -> Self where Self.dtype.is_floating_point():
        # Multiplying both fields by the same +-1 is exact, so this stays an
        # error-free transformation.
        var sign = copysign(SIMD[Self.dtype, Self.width](1), self.value)
        return Self(self.value * sign, self.error * sign)

    def copysign(
        self, sign_source: Self
    ) -> Self where Self.dtype.is_floating_point():
        var flip = copysign(
            SIMD[Self.dtype, Self.width](1), self.value
        ) * copysign(SIMD[Self.dtype, Self.width](1), sign_source.value)
        return Self(self.value * flip, self.error * flip)
