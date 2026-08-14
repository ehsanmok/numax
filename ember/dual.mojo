"""Forward-mode automatic differentiation: value paired with derivative.

`Dual[Inner: FloatLike]` wraps *any* `FloatLike` type in its `value`/`deriv`
fields, not just a raw `SIMD` pair -- which is what lets it nest. Seeding a
plain `Dual[Plain[dtype, width]]` with `deriv = 1` gets a first derivative,
same as before. Seeding a `Dual[Dual[Plain[dtype, width]]]` -- a `Dual` whose
`Inner` is itself a `Dual` -- gets a first *and* second derivative from one
kernel call: the outer `Dual` differentiates its `Inner`'s `value` by the
chain rule, and since `Inner` is itself being differentiated (it's a `Dual`
too), that's the second derivative falling out of the same rule applied
twice, with no change to the kernel itself.

Every operation below is written against `Inner`'s own `FloatLike` methods
rather than raw `SIMD` arithmetic -- that's what makes nesting `Dual` inside
itself actually track a second derivative, instead of just producing a
differently-shaped first derivative.
"""

from .numeric import FloatLike


@fieldwise_init
struct Dual[Inner: FloatLike](Copyable, FloatLike, Movable):
    """`value` is `f(x)`, `deriv` is `df/dx` at the same point."""

    var value: Self.Inner
    var deriv: Self.Inner

    @staticmethod
    def one() -> Self:
        # The derivative of a constant is zero.
        return Self(Self.Inner.one(), Self.Inner.constant(0.0))

    def __add__(self, rhs: Self) -> Self:
        return Self(self.value + rhs.value, self.deriv + rhs.deriv)

    def __mul__(self, rhs: Self) -> Self:
        # Product rule: (fg)' = f'g + fg'.
        return Self(
            self.value * rhs.value,
            self.deriv * rhs.value + self.value * rhs.deriv,
        )

    def __neg__(self) -> Self:
        return Self(-self.value, -self.deriv)

    def __truediv__(self, rhs: Self) -> Self:
        # Quotient rule: (f/g)' = (f'g - fg') / g^2.
        var value = self.value / rhs.value
        var deriv = (self.deriv * rhs.value + (-(self.value * rhs.deriv))) / (
            rhs.value * rhs.value
        )
        return Self(value^, deriv^)

    def exp(self) -> Self:
        # d/dx[exp(f)] = f' * exp(f).
        var e = self.value.exp()
        var d = e.copy() * self.deriv
        return Self(e^, d^)

    def ln(self) -> Self:
        # d/dx[ln(f)] = f' / f.
        return Self(self.value.ln(), self.deriv / self.value)

    @staticmethod
    def constant(v: Float64) -> Self:
        # The derivative of a constant is zero.
        return Self(Self.Inner.constant(v), Self.Inner.constant(0.0))

    def abs(self) -> Self:
        # d/dx|f| = sign(f) * f'.
        var sign = Self.Inner.one().copysign(self.value)
        return Self(self.value.abs(), self.deriv * sign)

    def copysign(self, sign_source: Self) -> Self:
        # copysign(f, s) = |f| * sign(s), so its derivative is f' scaled by
        # +1 if f and s already agree in sign, -1 if copysign has to flip f.
        var flip = Self.Inner.one().copysign(
            self.value
        ) * Self.Inner.one().copysign(sign_source.value)
        return Self(self.value.copysign(sign_source.value), self.deriv * flip)
