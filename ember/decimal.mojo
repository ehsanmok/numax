"""A base-10 fixed-point `FloatLike` conformer: exact decimal arithmetic.

`Decimal[width, scale]` stores a value as an integer scaled by `10^scale`
(`SIMD[DType.int64, width]`) -- the same representation `decimo`-style
projects use for base-10 exactness a binary `float` can't offer: `0.1 + 0.2`
is exactly `0.3` here, never `0.30000000000000004`. This is a different
problem than `Compensated` solves (base-10 exactness for decimal literals
and their sums, vs. extra *binary* significand bits), so it sits next to
`Compensated` as a fourth `FloatLike` conformer, not a replacement for it.

`scale` decimal digits after the point means every raw value is bounded by
`int64`'s range divided by `10^scale`; `__mul__` in particular squares that
bound before dividing back out, so this is scoped to modest magnitudes and
a modest `scale` (single digits), not arbitrary-precision decimal -- see
`gammainc`'s and `bessel_j0`'s docstrings in `ember.gamma`/`ember.bessel`
for the same kind of explicit, documented scope limit rather than silent
overflow.

`exp()`/`ln()` use fixed-iteration series/Newton's method, the same
GPU-motivated constraint as `ember.gamma`'s kernels (see its docstring):
no data-dependent convergence check, so a bounded amount of work replaces
an adaptive one.
"""

from std.math import copysign as _copysign_f64
from std.math import log

from .numeric import FloatLike


def _pow10[n: Int]() -> Int64:
    var r: Int64 = 1
    comptime for _ in range(n):
        r *= 10
    return r


@fieldwise_init
struct Decimal[width: Int, scale: Int](
    Copyable, Deinitable, FloatLike, Movable
):
    """`raw` is the value times `10^scale`, stored exactly as an integer."""

    var raw: SIMD[DType.int64, Self.width]

    @staticmethod
    def one() -> Self:
        return Self(SIMD[DType.int64, Self.width](_pow10[Self.scale]()))

    def __add__(self, rhs: Self) -> Self:
        # Same scale on both sides, so this is exact -- no rounding at all.
        return Self(self.raw + rhs.raw)

    def __neg__(self) -> Self:
        return Self(-self.raw)

    def __mul__(self, rhs: Self) -> Self:
        # (a*10^s) * (b*10^s) = (a*b)*10^s * 10^s, so dividing back out by
        # 10^s recovers a value scaled by 10^s again. `//` floors rather
        # than rounding to nearest, so this can be off by up to one raw
        # unit (`10^-scale`) versus a round-to-nearest product.
        comptime factor = _pow10[Self.scale]()
        return Self((self.raw * rhs.raw) // factor)

    def __truediv__(self, rhs: Self) -> Self:
        # (a*10^s) * 10^s / (b*10^s) = (a/b)*10^s -- one extra factor of
        # 10^s before dividing keeps the quotient at the right scale.
        comptime factor = _pow10[Self.scale]()
        return Self((self.raw * factor) // rhs.raw)

    def exp(self) -> Self:
        # sum_{n=0}^{29} self^n / n!, built term-by-term (term_n =
        # term_{n-1} * self / n) so no factorial ever needs its own
        # representation. No range reduction (unlike `Compensated.exp()`),
        # so this converges well for `self` within a few units of zero and
        # progressively needs more of the fixed 30 terms the larger
        # `|self|` gets.
        comptime num_terms = 30
        var term = Self.one()
        var total = term.copy()
        for n in range(1, num_terms):
            term = term * self / Self.constant(Float64(n))
            total = total + term
        return total^

    def ln(self) -> Self:
        # Newton's method on f(y) = exp(y) - self, exactly like
        # `Compensated.ln()` -- seeded from an ordinary `float64` `log` of
        # `self`'s value (going through `float64` here, unlike the rest of
        # this type's arithmetic, purely to get a starting guess) and
        # refined with this type's own exact decimal `exp`/`__add__`/`__mul__`.
        comptime factor = _pow10[Self.scale]()
        var x_f64 = self.raw.cast[DType.float64]() / SIMD[
            DType.float64, Self.width
        ](Float64(factor))
        var y0_f64 = log(x_f64) * SIMD[DType.float64, Self.width](
            Float64(factor)
        )
        var y = Self(y0_f64.cast[DType.int64]())

        comptime num_iters = 8
        comptime for _ in range(num_iters):
            y = y + (-Self.one()) + self * (-y).exp()

        return y^

    @staticmethod
    def constant(v: Float64) -> Self:
        comptime factor = _pow10[Self.scale]()
        var scaled = v * Float64(factor)
        var rounded = Int64(scaled + 0.5) if scaled >= 0 else Int64(
            scaled - 0.5
        )
        return Self(SIMD[DType.int64, Self.width](rounded))

    def abs(self) -> Self:
        return Self(abs(self.raw))

    def copysign(self, sign_source: Self) -> Self:
        # Round-trips through `float64` rather than a boolean mask/select,
        # since `SIMD[DType.int64, width]` comparisons don't expose the
        # same lane-select API `Plain`/`Dual`/`Compensated` use for
        # `copysign` on their floating-point fields.
        var mag_f = abs(self.raw).cast[DType.float64]()
        var sign_f = sign_source.raw.cast[DType.float64]()
        return Self(_copysign_f64(mag_f, sign_f).cast[DType.int64]())
