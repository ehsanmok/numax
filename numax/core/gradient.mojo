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
struct Gradient[Inner: FloatLike, n_vars: Int](Copyable, FloatLike, Movable):
    """`value` is `f(x_0, ..., x_{n_vars-1})`; `grad[i]` is `df/dx_i` at the
    same point.
    """

    var value: Self.Inner
    var grad: Array[Self.Inner, Self.n_vars]

    @staticmethod
    def variable(var value: Self.Inner, index: Int) -> Self:
        """Seed the `index`-th input variable: `value` as given, `grad` a
        one-hot vector (`1` at `index`, `0` elsewhere).
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
        """
        return Self.variable(Self.Inner.constant(value), index)

    @staticmethod
    def one() -> Self:
        # A constant's gradient is zero in every direction.
        return Self(Self.Inner.one(), _zero_grad[Self.Inner, Self.n_vars]())

    def __add__(self, rhs: Self) -> Self:
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = self.grad[i] + rhs.grad[i]
        return Self(self.value + rhs.value, g^)

    def __mul__(self, rhs: Self) -> Self:
        # Product rule per component: d(fg)/dx_i = (df/dx_i)*g + f*(dg/dx_i).
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = self.grad[i] * rhs.value + self.value * rhs.grad[i]
        return Self(self.value * rhs.value, g^)

    def __neg__(self) -> Self:
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = -self.grad[i]
        return Self(-self.value, g^)

    def __truediv__(self, rhs: Self) -> Self:
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
        # d(exp(f))/dx_i = exp(f) * df/dx_i.
        var e = self.value.exp()
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = e.copy() * self.grad[i]
        return Self(e^, g^)

    def ln(self) -> Self:
        # d(ln(f))/dx_i = (df/dx_i) / f.
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = self.grad[i] / self.value
        return Self(self.value.ln(), g^)

    def sqrt(self) -> Self:
        # d(sqrt(f))/dx_i = (df/dx_i) / (2*sqrt(f)).
        var v = self.value.sqrt()
        var scale = Self.Inner.constant(2.0) * v
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = self.grad[i] / scale.copy()
        return Self(v^, g^)

    def erf(self) -> Self:
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
        # d(sin(f))/dx_i = cos(f) * df/dx_i.
        var c = self.value.cos()
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = c.copy() * self.grad[i]
        return Self(self.value.sin(), g^)

    def cos(self) -> Self:
        # d(cos(f))/dx_i = -sin(f) * df/dx_i.
        var s = self.value.sin()
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = -(s.copy() * self.grad[i])
        return Self(self.value.cos(), g^)

    @staticmethod
    def constant(v: Float64) -> Self:
        # A constant's gradient is zero in every direction.
        return Self(
            Self.Inner.constant(v), _zero_grad[Self.Inner, Self.n_vars]()
        )

    def abs(self) -> Self:
        # d|f|/dx_i = sign(f) * df/dx_i.
        var sign = Self.Inner.one().copysign(self.value)
        var g = _zero_grad[Self.Inner, Self.n_vars]()
        for i in range(Self.n_vars):
            g[i] = self.grad[i] * sign.copy()
        return Self(self.value.abs(), g^)

    def copysign(self, sign_source: Self) -> Self:
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
        # A step function, same as `Dual.floor()` -- the gradient is zero
        # in every direction almost everywhere.
        return Self(self.value.floor(), _zero_grad[Self.Inner, Self.n_vars]())

    def ceil(self) -> Self:
        return Self(self.value.ceil(), _zero_grad[Self.Inner, Self.n_vars]())

    def trunc(self) -> Self:
        return Self(self.value.trunc(), _zero_grad[Self.Inner, Self.n_vars]())
