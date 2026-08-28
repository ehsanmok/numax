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
from std.math import ceil, copysign, exp2, floor, fma, log, round, sqrt, trunc

from .numeric import FloatLike, default_erf_approx


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

    def sqrt(self) -> Self where Self.dtype.is_floating_point():
        # Newton's method on f(y) = y^2 - self, in the form
        # y_{n+1} = (y_n + self/y_n) / 2 -- the same "seed from an ordinary
        # single-`dtype` estimate, then refine a fixed number of times in
        # compensated arithmetic" shape `ln` uses. Quadratic convergence
        # means the ~24 correct bits of a float32 seed become ~48 after one
        # step and past double-double precision after two; three is the
        # same margin `ln` keeps.
        #
        # `self <= 0` is outside the domain, but `self == 0` exactly would
        # divide by zero in the refinement rather than returning 0, so it
        # short-circuits below -- branchlessly, since `Self` may be a SIMD
        # vector with only some lanes at zero.
        comptime DT = Self.dtype
        comptime W = Self.width

        var seed = sqrt(self.value)
        var is_positive = seed.gt(SIMD[DT, W](0)).select(
            SIMD[DT, W](1), SIMD[DT, W](0)
        )
        # A zero lane's divisor becomes 1, so its refinement is a no-op on
        # the 0 seed instead of a division by zero.
        var safe = Self(seed + (SIMD[DT, W](1) - is_positive), SIMD[DT, W](0))
        var y = safe

        comptime num_iters = 3
        comptime for _ in range(num_iters):
            y = (y + self / y) * Self(SIMD[DT, W](0.5), SIMD[DT, W](0))

        return Self(y.value * is_positive, y.error * is_positive)

    def erf(self) -> Self where Self.dtype.is_floating_point():
        # No double-double-precision `erf` to port here (see
        # `numax.numeric.default_erf_approx`'s docstring), so this reuses
        # the same rational approximation `Plain` used to -- run through
        # compensated `+`/`*`/`exp`, which still recovers precision the
        # formula's own internal cancellation (near `x = 0`) would
        # otherwise lose to a single `dtype`; see
        # `tests/test_compensated.mojo`.
        return default_erf_approx(self)

    def erfc(self) -> Self where Self.dtype.is_floating_point():
        return Self.one() + (-self.erf())

    def _range_reduce_2pi(self) -> Self where Self.dtype.is_floating_point():
        """Shared range reduction for `sin`/`cos`: `self = m*2*pi + r`.

        Same shape as `exp()`'s reduction by `ln2` above -- `m` is rounded
        to the nearest integer so `|r| <= pi`, kept as a compensated pair
        so the reduced argument doesn't lose precision. Reducing only to
        `|r| <= pi` (rather than `|r| <= pi/4` via a quadrant table) avoids
        needing a `select`-like primitive to pick a quadrant per-lane --
        `FloatLike` doesn't have one yet (see `numax/bessel.mojo`'s own
        note on the same gap) -- at the cost of needing more Taylor terms
        below to converge at the wider boundary.
        """
        comptime DT = Self.dtype
        comptime W = Self.width
        comptime inv_two_pi = SIMD[DT, W](0.15915494309189535)
        comptime two_pi_hi = SIMD[DT, W](6.2831854820251465)
        comptime two_pi_lo = SIMD[DT, W](-1.7484556025237907e-07)

        var m = round(self.value * inv_two_pi)

        var t = m * two_pi_hi
        var t_err = fma(m, two_pi_hi, -t)

        var s = self.value - t
        var s_bb = s - self.value
        var s_err = (self.value - (s - s_bb)) + (-t - s_bb)

        return Self(s, (self.error - t_err) - m * two_pi_lo + s_err)

    def sin(self) -> Self where Self.dtype.is_floating_point():
        # sin(r) = r * sum_k (-1)^k r^(2k) / (2k+1)!, evaluated in `r^2`.
        # 24 terms converge to well past double-double precision even at
        # the reduction's `|r| = pi` boundary (checked numerically: the
        # k=21 term is already ~4e-32, and every later term is smaller).
        comptime DT = Self.dtype
        comptime W = Self.width
        var r = self._range_reduce_2pi()
        var r2 = r * r

        comptime num_terms = 24
        comptime coef: Array[Float64, num_terms] = [
            1.0,
            -0.16666666666666666,
            0.008333333333333333,
            -0.0001984126984126984,
            2.7557319223985893e-06,
            -2.505210838544172e-08,
            1.6059043836821613e-10,
            -7.647163731819816e-13,
            2.8114572543455206e-15,
            -8.22063524662433e-18,
            1.9572941063391263e-20,
            -3.868170170630684e-23,
            6.446950284384474e-26,
            -9.183689863795546e-29,
            1.1309962886447716e-31,
            -1.216125041553518e-34,
            1.151633562077195e-37,
            -9.67759295863189e-41,
            7.265460179153071e-44,
            -4.902469756513544e-47,
            2.9893108271424046e-50,
            -1.6552108677421951e-53,
            8.359650847182804e-57,
            -3.866628513960594e-60,
        ]

        var power = Self.one()
        var total = Self(SIMD[DT, W](0), SIMD[DT, W](0))
        comptime for k in range(num_terms):
            comptime split = _split_f64[DT, W](coef[k])
            total = total + power * Self(split[0], split[1])
            power = power * r2

        return total * r

    def cos(self) -> Self where Self.dtype.is_floating_point():
        # cos(r) = sum_k (-1)^k r^(2k) / (2k)!, evaluated in `r^2` -- same
        # reduction and convergence margin as `sin()` above.
        comptime DT = Self.dtype
        comptime W = Self.width
        var r = self._range_reduce_2pi()
        var r2 = r * r

        comptime num_terms = 24
        comptime coef: Array[Float64, num_terms] = [
            1.0,
            -0.5,
            0.041666666666666664,
            -0.001388888888888889,
            2.48015873015873e-05,
            -2.755731922398589e-07,
            2.08767569878681e-09,
            -1.1470745597729725e-11,
            4.779477332387385e-14,
            -1.5619206968586225e-16,
            4.110317623312165e-19,
            -8.896791392450574e-22,
            1.6117375710961184e-24,
            -2.4795962632247976e-27,
            3.279889237069838e-30,
            -3.7699876288159054e-33,
            3.8003907548547434e-36,
            -3.387157535521162e-39,
            2.6882202662866363e-42,
            -1.911963205040282e-45,
            1.2256174391283858e-48,
            -7.117406731291439e-52,
            3.7618428812322616e-55,
            -1.817315401561479e-58,
        ]

        var power = Self.one()
        var total = Self(SIMD[DT, W](0), SIMD[DT, W](0))
        comptime for k in range(num_terms):
            comptime split = _split_f64[DT, W](coef[k])
            total = total + power * Self(split[0], split[1])
            power = power * r2

        return total

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

    def floor(self) -> Self where Self.dtype.is_floating_point():
        # Floored from `value` alone, with `error` zeroed rather than
        # refined -- an honest, documented gap rather than a claim of
        # exactness: a `value` that rounds to just above an integer while
        # `error` is negative enough to put the true `value + error` just
        # below it produces the wrong integer here, since nothing folds
        # `error` back in before the floor. Rare in practice (`error` is
        # many orders of magnitude smaller than `value` whenever `value`
        # itself isn't already near the rounding boundary), but real.
        return Self(floor(self.value), SIMD[Self.dtype, Self.width](0))

    def ceil(self) -> Self where Self.dtype.is_floating_point():
        return Self(ceil(self.value), SIMD[Self.dtype, Self.width](0))

    def trunc(self) -> Self where Self.dtype.is_floating_point():
        return Self(trunc(self.value), SIMD[Self.dtype, Self.width](0))
