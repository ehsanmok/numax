"""Forward-mode automatic differentiation: value paired with derivative.

**This module is tier 1.** Each operation is its `Inner`'s operation plus
the chain rule -- fixed work, no branching -- so differentiating a kernel
costs it nothing in launchability.

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
struct Dual[Inner: FloatLike](
    Copyable, FloatLike, Movable, Writable where conforms_to(Inner, Writable)
):
    """`value` is `f(x)`, `deriv` is `df/dx` at the same point.

    Conforms to `Writable` when `Inner` does, so `print(d)` writes both
    components: `Dual(f(x), f'(x))`.
    """

    var value: Self.Inner
    var deriv: Self.Inner

    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.Inner, Writable):
        """Write `Dual(value, deriv)`.

        Args:
            writer: The destination both components are written to.
        """
        writer.write("Dual(", self.value, ", ", self.deriv, ")")

    @staticmethod
    def one() -> Self:
        """The constant `1`, with a zero derivative.

        Returns:
            `Dual(Inner.one(), 0)`.
        """
        # The derivative of a constant is zero.
        return Self(Self.Inner.one(), Self.Inner.constant(0.0))

    @staticmethod
    def seed(x: Float64) -> Self:
        """`x`, with its derivative seeded to 1 -- the variable being
        differentiated with respect to.

        `Dual.seed(0.5)` is `Dual[P](P(0.5), P.one())`, which is what
        nearly every call differentiating a single-variable function
        starts with. Use the two-argument constructor directly for the
        cases that are not that: a constant carries a zero derivative
        (`Dual[P](P(c), P.constant(0.0))`), and a nested `Dual` for a
        second derivative seeds an inner one instead.

        Args:
            x: The point at which the derivative is taken.

        Returns:
            `Dual(Inner.constant(x), Inner.one())`.
        """
        return Self(Self.Inner.constant(x), Self.Inner.one())

    def __add__(self, rhs: Self) -> Self:
        """The sum rule: values and derivatives add.

        Args:
            rhs: The addend.

        Returns:
            `(f + g, f' + g')`.
        """
        return Self(self.value + rhs.value, self.deriv + rhs.deriv)

    def __mul__(self, rhs: Self) -> Self:
        """The product rule: `(fg)' = f'g + fg'`.

        Args:
            rhs: The multiplier `g`.

        Returns:
            `(f * g, f' * g + f * g')`.
        """
        # Product rule: (fg)' = f'g + fg'.
        return Self(
            self.value * rhs.value,
            self.deriv * rhs.value + self.value * rhs.deriv,
        )

    def __neg__(self) -> Self:
        """Negation of both the value and the derivative.

        Returns:
            `(-f, -f')`.
        """
        return Self(-self.value, -self.deriv)

    def __truediv__(self, rhs: Self) -> Self:
        """The quotient rule: `(f/g)' = (f'g - fg') / g^2`.

        Args:
            rhs: The divisor `g`.

        Returns:
            `(f / g, (f' * g - f * g') / g^2)`.
        """
        # Quotient rule: (f/g)' = (f'g - fg') / g^2.
        var value = self.value / rhs.value
        var deriv = (self.deriv * rhs.value - (self.value * rhs.deriv)) / (
            rhs.value * rhs.value
        )
        return Self(value^, deriv^)

    def exp(self) -> Self:
        """The chain rule for `exp`: `d/dx[exp(f)] = f' * exp(f)`.

        Returns:
            `(exp(f), f' * exp(f))`.
        """
        # d/dx[exp(f)] = f' * exp(f).
        var e = self.value.exp()
        var d = e.copy() * self.deriv
        return Self(e^, d^)

    def ln(self) -> Self:
        """The chain rule for `ln`: `d/dx[ln(f)] = f' / f`.

        Returns:
            `(ln(f), f' / f)`; only meaningful for `f > 0`.
        """
        # d/dx[ln(f)] = f' / f.
        return Self(self.value.ln(), self.deriv / self.value)

    def sqrt(self) -> Self:
        """The chain rule for `sqrt`: `f' / (2*sqrt(f))`.

        Returns:
            `(sqrt(f), f' / (2*sqrt(f)))`, with an infinite derivative at `f =
            0`.
        """
        # d/dx[sqrt(f)] = f' / (2*sqrt(f)), dividing by the value already
        # computed. The derivative is genuinely infinite at f = 0, which
        # this reproduces.
        var v = self.value.sqrt()
        var d = self.deriv / (Self.Inner.constant(2.0) * v)
        return Self(v^, d^)

    def erf(self) -> Self:
        """The chain rule for `erf`: `(2/sqrt(pi)) * exp(-f^2) * f'`.

        Returns:
            `(erf(f), (2/sqrt(pi)) * exp(-f^2) * f')`, the value from
            `Inner.erf()`.
        """
        # d/dx[erf(f)] = (2/sqrt(pi)) * exp(-f^2) * f'.
        var v = self.value.erf()
        var scale = (
            Self.Inner.constant(1.1283791670955126)
            * (-(self.value * self.value)).exp()
        )
        var d = self.deriv * scale
        return Self(v^, d^)

    def erfc(self) -> Self:
        """The chain rule for `erfc`: `-(2/sqrt(pi)) * exp(-f^2) * f'`.

        Returns:
            `(erfc(f), -(2/sqrt(pi)) * exp(-f^2) * f')`, the value from
            `Inner.erfc()`.
        """
        # d/dx[erfc(f)] = -(2/sqrt(pi)) * exp(-f^2) * f' -- computed via
        # `Inner.erfc()` rather than `one() - erf()`, so a large `f`
        # doesn't force a cancelling subtraction here either.
        var v = self.value.erfc()
        var scale = (
            Self.Inner.constant(1.1283791670955126)
            * (-(self.value * self.value)).exp()
        )
        var d = -(self.deriv * scale)
        return Self(v^, d^)

    def sin(self) -> Self:
        """The chain rule for `sin`: `d/dx[sin(f)] = cos(f) * f'`.

        Returns:
            `(sin(f), cos(f) * f')`.
        """
        # d/dx[sin(f)] = cos(f) * f'.
        return Self(self.value.sin(), self.value.cos() * self.deriv)

    def cos(self) -> Self:
        """The chain rule for `cos`: `d/dx[cos(f)] = -sin(f) * f'`.

        Returns:
            `(cos(f), -sin(f) * f')`.
        """
        # d/dx[cos(f)] = -sin(f) * f'.
        return Self(self.value.cos(), -(self.value.sin() * self.deriv))

    @staticmethod
    def constant(v: Float64) -> Self:
        """A literal, with a zero derivative.

        Args:
            v: The literal value.

        Returns:
            `(Inner.constant(v), 0)`.
        """
        # The derivative of a constant is zero.
        return Self(Self.Inner.constant(v), Self.Inner.constant(0.0))

    def abs(self) -> Self:
        """The derivative of `|f|`: `sign(f) * f'`.

        Returns:
            `(|f|, sign(f) * f')`, taking `sign(0) = +1`.
        """
        # d/dx|f| = sign(f) * f'.
        var sign = Self.Inner.one().copysign(self.value)
        return Self(self.value.abs(), self.deriv * sign)

    def copysign(self, sign_source: Self) -> Self:
        """`copysign(f, s)`, whose derivative is `f'` times `+1` or `-1`.

        Args:
            sign_source: The value `s` whose sign is copied; its derivative is
                ignored.

        Returns:
            `(copysign(f, s), f' * sign(f) * sign(s))`.
        """
        # copysign(f, s) = |f| * sign(s), so its derivative is f' scaled by
        # +1 if f and s already agree in sign, -1 if copysign has to flip f.
        var flip = Self.Inner.one().copysign(
            self.value
        ) * Self.Inner.one().copysign(sign_source.value)
        return Self(self.value.copysign(sign_source.value), self.deriv * flip)

    def floor(self) -> Self:
        """`floor(f)`, a step function with zero derivative.

        Returns:
            `(floor(f), 0)`; the derivative is undefined at the integers and
            reported as zero.
        """
        # A step function, so the derivative is zero almost everywhere.
        # Undefined at the integers, which this does not special-case.
        return Self(self.value.floor(), Self.Inner.constant(0.0))

    def ceil(self) -> Self:
        """`ceil(f)`, a step function with zero derivative.

        Returns:
            `(ceil(f), 0)`; the derivative is undefined at the integers and
            reported as zero.
        """
        return Self(self.value.ceil(), Self.Inner.constant(0.0))

    def trunc(self) -> Self:
        """`trunc(f)`, a step function with zero derivative.

        Returns:
            `(trunc(f), 0)`; the derivative is undefined at the nonzero integers
            and reported as zero.
        """
        return Self(self.value.trunc(), Self.Inner.constant(0.0))
