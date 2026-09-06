"""Ordinary SIMD, wrapped so it can conform to `FloatLike`.

**This module is tier 1.** Every method is a fixed amount of straight-line
work on a SIMD register, so a kernel instantiated at `Plain` launches
inside a GPU thread unmodified.

Mojo can't retroactively add a trait to a type from outside its defining
module, so a bare `SIMD` can't conform to `FloatLike` directly -- `Plain` is
the thin wrapper that lets it. Instantiating a `FloatLike` kernel with `Plain`
is the baseline: no derivative, no extra precision, just the hardware.
"""

from std.math import (
    ceil,
    copysign,
    cos,
    erf,
    erfc,
    exp,
    floor,
    log,
    sin,
    sqrt,
    trunc,
)

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
        writer.write(self.v)

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

    def floor(self) -> Self where Self.dtype.is_floating_point():
        return Self(floor(self.v))

    def ceil(self) -> Self where Self.dtype.is_floating_point():
        return Self(ceil(self.v))

    def trunc(self) -> Self where Self.dtype.is_floating_point():
        return Self(trunc(self.v))


comptime f32 = DType.float32
"""`DType.float32`, spelled short.

A dtype, not a conformer -- the same name works at every layer, which is the
point: `linspace[5, f32](...)` and `Shaped[f32, 4, 4](ctx)` on the tensor
side, `Plain[f32]` and `Dual[Plain[f32]]` on the kernel side. `Plain`'s
`width` defaults to 1, so `Plain[f32]` is the single-lane scalar; name
`Plain[f32, 8]` for a vector width.
"""

comptime f64 = DType.float64
"""`DType.float64`. See `f32` above."""
