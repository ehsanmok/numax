"""Multi-variable forward-mode automatic differentiation.

**This module is tier 1.** Each operation is the `Dual` rule run once per
gradient component, over a loop whose bound is the compile-time `n_vars`
and identical in every lane.

`Dual[Inner]` tracks one derivative -- fine for a kernel of a single
variable, or differentiated one variable at a time by calling it `n_vars`
times with a different seed each call. `Gradient[Inner, n_vars]` tracks all
`n_vars` partial derivatives in a single pass instead: `value` is `f(x)`
same as `Dual`, and `grad` is the full gradient vector `[df/dx_0, ...,
df/dx_{n_vars-1}]`, propagated by the multivariate chain rule through every
operation below. Seed each input variable with `Gradient.variable(x_i, i)`
(value `x_i`, a one-hot gradient at position `i`) and a kernel of several
inputs -- built purely from `+`/`*`/`/`/`exp`/etc., composed by the caller,
same as any other `FloatLike` kernel -- returns every partial derivative at
once, still just one call.

Every operation is written against `Inner`'s own `FloatLike` methods,
component-wise across `grad`, the same discipline `Dual` uses for nesting
-- so `Gradient[Dual[Plain[...]]]` would (in principle) get a Hessian-row's
worth of information the way nested `Dual` gets a second derivative,
though that combination isn't exercised here; 0.1.0 stops at first-order,
multi-variable gradients.
"""

from std.collections import Array

from .numeric import FloatLike


def _zero_grad[T: FloatLike, n_vars: Int]() -> Array[T, n_vars]:
    return Array[T, n_vars](fill=T.constant(0.0))


@fieldwise_init
struct Gradient[Inner: FloatLike, n_vars: Int](
    Copyable, FloatLike, Movable, Writable where conforms_to(Inner, Writable)
):
    """`value` is `f(x_0, ..., x_{n_vars-1})`; `grad[i]` is `df/dx_i` at the
    same point.

    Conforms to `Writable` when `Inner` does, so `print(g)` writes the
    value and every partial derivative.
    """

    var value: Self.Inner
    var grad: Array[Self.Inner, Self.n_vars]

    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.Inner, Writable):
        """Write `Gradient(value, [df/dx_0, ...])`.

        Args:
            writer: The destination the value and partials are written to.
        """
        writer.write("Gradient(", self.value, ", [")
        for i in range(Self.n_vars):
            if i > 0:
                writer.write(", ")
            writer.write(self.grad[i])
        writer.write("])")

    @staticmethod
    def variable(var value: Self.Inner, index: Int) -> Self:
        """Seed the `index`-th input variable: `value` as given, `grad` a
        one-hot vector (`1` at `index`, `0` elsewhere).

        Args:
            value: The input's value, as an `Inner`.
            index: Which of the `n_vars` inputs this is, in `[0, n_vars)`.

        Returns:
            `(value, e_index)`, the one-hot gradient at `index`.
        """
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        g[index] = Self.Inner.one()
        return Self(value^, g^)

    @staticmethod
    def variable(value: Float64, index: Int) -> Self:
        """Convenience overload of the above for the common case where the
        seed is just a number: `Self.Inner.constant(value)` builds `Inner`
        from it, the same conversion every `FloatLike` kernel already uses
        to embed a literal without knowing `Inner`'s representation. Skips
        a `SIMD`/`Inner`-construction helper at the call site entirely --
        see `examples/gradient.mojo`.

        Args:
            value: The input's value, converted by `Inner.constant`.
            index: Which of the `n_vars` inputs this is, in `[0, n_vars)`.

        Returns:
            `(Inner.constant(value), e_index)`, the one-hot gradient at `index`.
        """
        return Self.variable(Self.Inner.constant(value), index)

    @staticmethod
    def one() -> Self:
        """The constant `1`, with a zero gradient.

        Returns:
            `(Inner.one(), 0)`.
        """
        # A constant's gradient is zero in every direction.
        return Self(Self.Inner.one(), _zero_grad[Self.Inner, Self.n_vars]())

    def __add__(self, rhs: Self) -> Self:
        """The sum rule, component by component.

        Args:
            rhs: The addend.

        Returns:
            `(f + g, grad f + grad g)`.
        """
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = self.grad[i] + rhs.grad[i]
        return Self(self.value + rhs.value, g^)

    def __mul__(self, rhs: Self) -> Self:
        """The product rule per component: `d(fg)/dx_i = f_i g + f g_i`.

        Args:
            rhs: The multiplier `g`.

        Returns:
            `(f * g, grad f * g + f * grad g)`.
        """
        # Product rule per component: d(fg)/dx_i = (df/dx_i)*g + f*(dg/dx_i).
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = self.grad[i] * rhs.value + self.value * rhs.grad[i]
        return Self(self.value * rhs.value, g^)

    def __neg__(self) -> Self:
        """Negation of the value and every partial.

        Returns:
            `(-f, -grad f)`.
        """
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = -self.grad[i]
        return Self(-self.value, g^)

    def __truediv__(self, rhs: Self) -> Self:
        """The quotient rule per component: `(f_i g - f g_i) / g^2`.

        Args:
            rhs: The divisor `g`.

        Returns:
            `(f / g, (grad f * g - f * grad g) / g^2)`.
        """
        # Quotient rule per component: d(f/g)/dx_i = ((df/dx_i)*g -
        # f*(dg/dx_i)) / g^2.
        var value = self.value / rhs.value
        var denom = rhs.value * rhs.value
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = (
                self.grad[i] * rhs.value - (self.value * rhs.grad[i])
            ) / denom
        return Self(value^, g^)

    def exp(self) -> Self:
        """The chain rule for `exp`: every partial scaled by `exp(f)`.

        Returns:
            `(exp(f), exp(f) * grad f)`.
        """
        # d(exp(f))/dx_i = exp(f) * df/dx_i.
        var e = self.value.exp()
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = e.copy() * self.grad[i]
        return Self(e^, g^)

    def ln(self) -> Self:
        """The chain rule for `ln`: every partial divided by `f`.

        Returns:
            `(ln(f), grad f / f)`; only meaningful for `f > 0`.
        """
        # d(ln(f))/dx_i = (df/dx_i) / f.
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = self.grad[i] / self.value
        return Self(self.value.ln(), g^)

    def sqrt(self) -> Self:
        """The chain rule for `sqrt`: every partial divided by `2*sqrt(f)`.

        Returns:
            `(sqrt(f), grad f / (2*sqrt(f)))`, infinite partials at `f = 0`.
        """
        # d(sqrt(f))/dx_i = (df/dx_i) / (2*sqrt(f)).
        var v = self.value.sqrt()
        var scale = Self.Inner.constant(2.0) * v
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = self.grad[i] / scale.copy()
        return Self(v^, g^)

    def erf(self) -> Self:
        """The chain rule for `erf`: partials scaled by `(2/sqrt(pi))e^(-f^2)`.

        Returns:
            `(erf(f), (2/sqrt(pi)) * exp(-f^2) * grad f)`.
        """
        # d(erf(f))/dx_i = (2/sqrt(pi)) * exp(-f^2) * df/dx_i.
        var v = self.value.erf()
        var scale = (
            Self.Inner.constant(1.1283791670955126)
            * (-(self.value * self.value)).exp()
        )
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = self.grad[i] * scale.copy()
        return Self(v^, g^)

    def erfc(self) -> Self:
        """The chain rule for `erfc`: partials times `-(2/sqrt(pi))e^(-f^2)`.

        Returns:
            `(erfc(f), -(2/sqrt(pi)) * exp(-f^2) * grad f)`, the value from
            `Inner.erfc()`.
        """
        # Computed via `Inner.erfc()` rather than `one() - erf()`, same
        # cancellation-avoidance reason as `Dual.erfc()`.
        var v = self.value.erfc()
        var scale = (
            Self.Inner.constant(1.1283791670955126)
            * (-(self.value * self.value)).exp()
        )
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = -(self.grad[i] * scale.copy())
        return Self(v^, g^)

    def sin(self) -> Self:
        """The chain rule for `sin`: every partial scaled by `cos(f)`.

        Returns:
            `(sin(f), cos(f) * grad f)`.
        """
        # d(sin(f))/dx_i = cos(f) * df/dx_i.
        var c = self.value.cos()
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = c.copy() * self.grad[i]
        return Self(self.value.sin(), g^)

    def cos(self) -> Self:
        """The chain rule for `cos`: every partial scaled by `-sin(f)`.

        Returns:
            `(cos(f), -sin(f) * grad f)`.
        """
        # d(cos(f))/dx_i = -sin(f) * df/dx_i.
        var s = self.value.sin()
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = -(s.copy() * self.grad[i])
        return Self(self.value.cos(), g^)

    @staticmethod
    def constant(v: Float64) -> Self:
        """A literal, with a zero gradient.

        Args:
            v: The literal value.

        Returns:
            `(Inner.constant(v), 0)`.
        """
        # A constant's gradient is zero in every direction.
        return Self(
            Self.Inner.constant(v), _zero_grad[Self.Inner, Self.n_vars]()
        )

    def abs(self) -> Self:
        """The derivative of `|f|`: every partial scaled by `sign(f)`.

        Returns:
            `(|f|, sign(f) * grad f)`, taking `sign(0) = +1`.
        """
        # d|f|/dx_i = sign(f) * df/dx_i.
        var sign = Self.Inner.one().copysign(self.value)
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = self.grad[i] * sign.copy()
        return Self(self.value.abs(), g^)

    def copysign(self, sign_source: Self) -> Self:
        """`copysign(f, s)`, every partial scaled by `+1` or `-1`.

        Args:
            sign_source: The value `s` whose sign is copied; its gradient is
                ignored.

        Returns:
            `(copysign(f, s), sign(f) * sign(s) * grad f)`.
        """
        # Same reasoning as `Dual.copysign`: the derivative is `df/dx_i`
        # scaled by +1 if `f` and `sign_source` already agree in sign, -1
        # if `copysign` has to flip `f`.
        var flip = Self.Inner.one().copysign(
            self.value
        ) * Self.Inner.one().copysign(sign_source.value)
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = self.grad[i] * flip.copy()
        return Self(self.value.copysign(sign_source.value), g^)

    def floor(self) -> Self:
        """`floor(f)`, a step function with zero gradient.

        Returns:
            `(floor(f), 0)`; undefined at the integers and reported as zero.
        """
        # A step function, same as `Dual.floor()` -- the gradient is zero
        # in every direction almost everywhere.
        return Self(self.value.floor(), _zero_grad[Self.Inner, Self.n_vars]())

    def ceil(self) -> Self:
        """`ceil(f)`, a step function with zero gradient.

        Returns:
            `(ceil(f), 0)`; undefined at the integers and reported as zero.
        """
        return Self(self.value.ceil(), _zero_grad[Self.Inner, Self.n_vars]())

    def trunc(self) -> Self:
        """`trunc(f)`, a step function with zero gradient.

        Returns:
            `(trunc(f), 0)`; undefined at the nonzero integers and reported as
            zero.
        """
        return Self(self.value.trunc(), _zero_grad[Self.Inner, Self.n_vars]())
