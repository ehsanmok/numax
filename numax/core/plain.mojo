"""Ordinary SIMD, wrapped so it can conform to `FloatLike`.

**This module is tier 1.** Every method is a fixed amount of straight-line
work on a SIMD register, so a kernel instantiated at `Plain` launches
inside a GPU thread unmodified.

Mojo can't retroactively add a trait to a type from outside its defining
module, so a bare `SIMD` can't conform to `FloatLike` directly -- `Plain` is
the thin wrapper that lets it. Instantiating a `FloatLike` kernel with `Plain`
is the baseline: no derivative, no extra precision, just the hardware.

`exp`, `ln` and `erf` come from `numax.core.libm` rather than `std.math`:
at float64 the standard library's are `1e5`, `9e6` and `2e8` ulp off at
the pinned release (its own `erfc`, `sin`, `cos` and `sqrt` are within a
few ulp and are used directly), and every accuracy row in the library
sat on that floor. `libm` is fdlibm's algorithms in SIMD form, within one
ulp, and defers to `std.math` at every other dtype.
"""

from std.math import ceil, copysign, cos, erfc, floor, sin, sqrt, trunc

from .libm import erf, exp, log
from .numeric import FloatLike


@fieldwise_init
struct Plain[dtype: DType, width: Int = 1](
    Copyable, FloatLike where dtype.is_floating_point(), Movable, Writable
):
    """A `SIMD[dtype, width]` value, viewed as a `FloatLike`.

    Conforms to `Writable`, so `print(x)` writes the number and `x.v` is
    needed only when the raw `SIMD` is what the caller actually wants.
    """

    var v: SIMD[Self.dtype, Self.width]

    def write_to(self, mut writer: Some[Writer]):
        """Write the wrapped `SIMD` value, as `print(x.v)` would.

        Args:
            writer: The destination the lanes are written to.
        """
        writer.write(self.v)

    @staticmethod
    def one() -> Self:
        """The value `1` in every lane.

        Returns:
            A `Plain` holding `1` in each of its `width` lanes.
        """
        return Self(SIMD[Self.dtype, Self.width](1))

    def __add__(self, rhs: Self) -> Self:
        """Lane-wise `SIMD` addition.

        Args:
            rhs: The addend.

        Returns:
            `self.v + rhs.v`, rounded once.
        """
        return Self(self.v + rhs.v)

    def __mul__(self, rhs: Self) -> Self:
        """Lane-wise `SIMD` multiplication.

        Args:
            rhs: The multiplier.

        Returns:
            `self.v * rhs.v`, rounded once.
        """
        return Self(self.v * rhs.v)

    def __neg__(self) -> Self:
        """Lane-wise negation.

        Returns:
            `-self.v`, exact.
        """
        return Self(-self.v)

    def __truediv__(self, rhs: Self) -> Self:
        """Lane-wise `SIMD` division.

        Args:
            rhs: The divisor.

        Returns:
            `self.v / rhs.v`, rounded once; IEEE infinities or NaN for a zero
            divisor.
        """
        return Self(self.v / rhs.v)

    def exp(self) -> Self where Self.dtype.is_floating_point():
        """`e^self`, through `numax.core.libm.exp`.

        Returns:
            `e^self.v`, within one ulp at float64.
        """
        return Self(exp(self.v))

    def ln(self) -> Self where Self.dtype.is_floating_point():
        """The natural logarithm, through `numax.core.libm.log`.

        Returns:
            `ln(self.v)`, within one ulp at float64; NaN for negative lanes.
        """
        return Self(log(self.v))

    def sqrt(self) -> Self where Self.dtype.is_floating_point():
        """The square root, through `std.math.sqrt`.

        Returns:
            `sqrt(self.v)`, NaN for negative lanes.
        """
        return Self(sqrt(self.v))

    def erf(self) -> Self where Self.dtype.is_floating_point():
        """The error function, through `numax.core.libm.erf`.

        Returns:
            `erf(self.v)`, within one ulp at float64.
        """
        return Self(erf(self.v))

    def erfc(self) -> Self where Self.dtype.is_floating_point():
        """The complementary error function, through `std.math.erfc`.

        Returns:
            `erfc(self.v)`, computed directly rather than as `1 - erf`.
        """
        return Self(erfc(self.v))

    def sin(self) -> Self where Self.dtype.is_floating_point():
        """The sine in radians, through `std.math.sin`.

        Returns:
            `sin(self.v)`.
        """
        return Self(sin(self.v))

    def cos(self) -> Self where Self.dtype.is_floating_point():
        """The cosine in radians, through `std.math.cos`.

        Returns:
            `cos(self.v)`.
        """
        return Self(cos(self.v))

    @staticmethod
    def constant(v: Float64) -> Self:
        """Broadcast a `Float64` literal to every lane.

        Args:
            v: The literal, rounded once to `dtype`.

        Returns:
            A `Plain` holding `v` in each of its `width` lanes.
        """
        return Self(SIMD[Self.dtype, Self.width](v))

    def abs(self) -> Self:
        """The lane-wise absolute value.

        Returns:
            `|self.v|`.
        """
        return Self(abs(self.v))

    def copysign(
        self, sign_source: Self
    ) -> Self where Self.dtype.is_floating_point():
        """The magnitude of `self` with the sign of `sign_source`.

        Args:
            sign_source: The value whose per-lane sign bit is copied.

        Returns:
            `std.math.copysign(self.v, sign_source.v)`.
        """
        return Self(copysign(self.v, sign_source.v))

    def floor(self) -> Self where Self.dtype.is_floating_point():
        """Lane-wise rounding toward negative infinity.

        Returns:
            `floor(self.v)`.
        """
        return Self(floor(self.v))

    def ceil(self) -> Self where Self.dtype.is_floating_point():
        """Lane-wise rounding toward positive infinity.

        Returns:
            `ceil(self.v)`.
        """
        return Self(ceil(self.v))

    def trunc(self) -> Self where Self.dtype.is_floating_point():
        """Lane-wise rounding toward zero.

        Returns:
            `trunc(self.v)`.
        """
        return Self(trunc(self.v))
