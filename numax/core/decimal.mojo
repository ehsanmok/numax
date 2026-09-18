"""A base-10 fixed-point `FloatLike` conformer: exact decimal arithmetic.

**This module is tier 1.** `exp` is a fixed 30-term series and `ln` and
`sqrt` are fixed-count Newton refinements, the same trade the other
conformers make.

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
`gammainc`'s and `j0`'s docstrings in
`numax.special.gamma`/`numax.special.bessel`
for the same kind of explicit, documented scope limit rather than silent
overflow.

`exp()`/`ln()` use fixed-iteration series/Newton's method, the same
GPU-motivated constraint as `numax.special.gamma`'s kernels (see its docstring):
no data-dependent convergence check, so a bounded amount of work replaces
an adaptive one.
"""

from std.math import copysign as _copysign_f64
from std.math import log, sqrt

from .numeric import FloatLike, default_erf_approx


def _pow10[n: Int]() -> Int64:
    var r: Int64 = 1
    comptime for _ in range(n):
        r *= 10
    return r


@fieldwise_init
struct Decimal[width: Int, scale: Int](
    Copyable, Deinitable, FloatLike, Movable, Writable
):
    """`raw` is the value times `10^scale`, stored exactly as an integer.

    Conforms to `Writable` through `to_float64`, so `print(x)` shows the
    number rather than its scaled integer. `raw` is still the field to
    assert against when exactness is what is under test.
    """

    var raw: SIMD[DType.int64, Self.width]

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.to_float64())

    @staticmethod
    def one() -> Self:
        return Self(SIMD[DType.int64, Self.width](_pow10[Self.scale]()))

    def to_float64(self) -> SIMD[DType.float64, Self.width]:
        """The value as an ordinary float -- `raw / 10^scale`.

        The read-out every other conformer has and this one did not, which
        left callers doing the descaling by hand. Lossy on purpose and named
        for it: the whole point of `Decimal` is that `raw` is exact, so
        `raw` stays the field to assert against when exactness is what is
        under test. This is for printing, plotting, and handing a value to
        something that only speaks binary floating point.
        """
        return self.raw.cast[DType.float64]() / Float64(_pow10[Self.scale]())

    def __add__(self, rhs: Self) -> Self:
        # Same scale on both sides, so this is exact -- no rounding at all.
        return Self(self.raw + rhs.raw)

    def __neg__(self) -> Self:
        return Self(-self.raw)

    def __mul__(self, rhs: Self) -> Self:
        # (a*10^s)(b*10^s) = (ab*10^s)*10^s, so dividing 10^s back out
        # rescales correctly. `//` floors, so this can sit one raw unit
        # (`10^-scale`) below a round-to-nearest product.
        comptime factor = _pow10[Self.scale]()
        return Self((self.raw * rhs.raw) // factor)

    def __truediv__(self, rhs: Self) -> Self:
        # (a*10^s) * 10^s / (b*10^s) = (a/b)*10^s -- one extra factor of
        # 10^s before dividing keeps the quotient at the right scale.
        comptime factor = _pow10[Self.scale]()
        return Self((self.raw * factor) // rhs.raw)

    def exp(self) -> Self:
        # sum_{n=0}^{29} self^n / n!, term-by-term so no factorial needs a
        # representation. No range reduction, so accuracy falls off as
        # `|self|` grows past a few units.
        comptime num_terms = 30
        var term = Self.one()
        var total = term.copy()
        for n in range(1, num_terms):
            term = term * self / Self.constant(Float64(n))
            total = total + term
        return total^

    def ln(self) -> Self:
        # Newton on f(y) = exp(y) - self, as `Compensated.ln()`: a
        # `float64` `log` for the seed only, then this type's own exact
        # decimal `exp`/`+`/`*` for the refinement.
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
            y = y - Self.one() + self * (-y).exp()

        return y^

    def sqrt(self) -> Self:
        # Newton on f(y) = y^2 - self, seeded from a `float64` sqrt and
        # refined in this type's exact arithmetic, as `ln` does. A
        # non-positive `self` returns zero rather than dividing.
        comptime factor = _pow10[Self.scale]()
        var x_f64 = self.raw.cast[DType.float64]() / SIMD[
            DType.float64, Self.width
        ](Float64(factor))
        var positive = self.raw.gt(SIMD[DType.int64, Self.width](0)).select(
            SIMD[DType.int64, Self.width](1),
            SIMD[DType.int64, Self.width](0),
        )
        var y0_f64 = sqrt(x_f64) * SIMD[DType.float64, Self.width](
            Float64(factor)
        )
        # Non-positive lanes get a raw seed of 1 so the division below is
        # defined; their result is masked back to zero at the end.
        var y = Self(
            y0_f64.cast[DType.int64]() * positive
            + (SIMD[DType.int64, Self.width](1) - positive)
        )

        comptime num_iters = 4
        comptime for _ in range(num_iters):
            y = (y + self / y) * Self.constant(0.5)

        return Self(y.raw * positive)

    def erf(self) -> Self:
        # Same rationale as `Compensated.erf()`: no exact-decimal `erf` to
        # port, so this reuses `default_erf_approx` over this type's own
        # exact `+`/`*`/`exp`.
        return default_erf_approx(self)

    def erfc(self) -> Self:
        return Self.one() - self.erf()

    def sin(self) -> Self:
        # sin(x) = sum_{k=0}^{29} term_k, term_k = term_{k-1} * (-x^2) /
        # ((2k)(2k+1)). Fixed 30 terms and no range reduction, as `exp()`
        # above, with the same falloff as `|self|` grows.
        comptime num_terms = 30
        var neg_x2 = -(self * self)
        var term = self.copy()
        var total = term.copy()
        for k in range(1, num_terms):
            var denom = Self.constant(Float64(2 * k) * Float64(2 * k + 1))
            term = term * neg_x2 / denom
            total = total + term
        return total^

    def cos(self) -> Self:
        # cos(x) = sum_{k=0}^{29} term_k, term_0 = 1, term_k = term_{k-1} *
        # (-x^2) / ((2k-1)*(2k)).
        comptime num_terms = 30
        var neg_x2 = -(self * self)
        var term = Self.one()
        var total = term.copy()
        for k in range(1, num_terms):
            var denom = Self.constant(Float64(2 * k - 1) * Float64(2 * k))
            term = term * neg_x2 / denom
            total = total + term
        return total^

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
        # Through `float64` rather than a mask: `SIMD[DType.int64, w]`
        # comparisons expose no lane-select like the one `copysign` uses on
        # the floating-point conformers.
        var mag_f = abs(self.raw).cast[DType.float64]()
        var sign_f = sign_source.raw.cast[DType.float64]()
        return Self(_copysign_f64(mag_f, sign_f).cast[DType.int64]())

    def floor(self) -> Self:
        # Exact and float-free: `//` on `SIMD[DType.int64, ...]` already
        # rounds toward negative infinity (`-7 // 10 == -1`), which is what
        # `floor` of `raw/10^scale` needs.
        comptime factor = _pow10[Self.scale]()
        return Self((self.raw // factor) * factor)

    def ceil(self) -> Self:
        # `ceil(a/b) = -floor(-a/b)`, reusing the same floor-division `//`.
        comptime factor = _pow10[Self.scale]()
        return Self(-((-self.raw) // factor) * factor)

    def trunc(self) -> Self:
        # Round toward zero: floor of the magnitude, sign reattached --
        # `//` on a non-negative numerator is already truncation, so this
        # only needs the sign split `copysign` above also needs.
        comptime factor = _pow10[Self.scale]()
        var mag = (abs(self.raw) // factor) * factor
        var is_neg = self.raw.lt(SIMD[DType.int64, Self.width](0)).select(
            SIMD[DType.int64, Self.width](-1),
            SIMD[DType.int64, Self.width](1),
        )
        return Self(mag * is_neg)
