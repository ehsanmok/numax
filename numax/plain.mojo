"""Ordinary SIMD, wrapped so it can conform to `FloatLike`.

Mojo can't retroactively add a trait to a type from outside its defining
module, so a bare `SIMD` can't conform to `FloatLike` directly -- `Plain` is
the thin wrapper that lets it. Instantiating a `FloatLike` kernel with `Plain`
is the baseline: no derivative, no extra precision, just the hardware.
"""

from std.math import copysign, cos, erf, erfc, exp, log, sin, sqrt

from .numeric import FloatLike


@fieldwise_init
struct Plain[dtype: DType, width: Int](
    Copyable, FloatLike where dtype.is_floating_point(), Movable
):
    """A `SIMD[dtype, width]` value, viewed as a `FloatLike`."""

    var v: SIMD[Self.dtype, Self.width]

    @staticmethod
    def one() -> Self:
        return Self(SIMD[Self.dtype, Self.width](1))

    def __add__(self, rhs: Self) -> Self:
        return Self(self.v + rhs.v)

    def __mul__(self, rhs: Self) -> Self:
        return Self(self.v * rhs.v)

    def __neg__(self) -> Self:
        return Self(-self.v)

    def __truediv__(self, rhs: Self) -> Self:
        return Self(self.v / rhs.v)

    def exp(self) -> Self where Self.dtype.is_floating_point():
        return Self(exp(self.v))

    def ln(self) -> Self where Self.dtype.is_floating_point():
        return Self(log(self.v))

    def sqrt(self) -> Self where Self.dtype.is_floating_point():
        return Self(sqrt(self.v))

    def erf(self) -> Self where Self.dtype.is_floating_point():
        return Self(erf(self.v))

    def erfc(self) -> Self where Self.dtype.is_floating_point():
        return Self(erfc(self.v))

    def sin(self) -> Self where Self.dtype.is_floating_point():
        return Self(sin(self.v))

    def cos(self) -> Self where Self.dtype.is_floating_point():
        return Self(cos(self.v))

    @staticmethod
    def constant(v: Float64) -> Self:
        return Self(SIMD[Self.dtype, Self.width](v))

    def abs(self) -> Self:
        return Self(abs(self.v))

    def copysign(
        self, sign_source: Self
    ) -> Self where Self.dtype.is_floating_point():
        return Self(copysign(self.v, sign_source.v))
