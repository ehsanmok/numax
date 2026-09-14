"""The special functions at `Interval` and `Decimal`: the same `erf`, `j0`
and `gaussian` the rest of the library calls, returning an enclosure of
every value over an input range, or exact base-10 arithmetic.

No function here was written for either type. `erf[T: FloatLike]` is one
definition, and the conformer decides what comes back: at `Interval` a
`[lo, hi]` that contains `erf(x)` for every `x` in the input interval, at
`Decimal` a value whose arithmetic is exact in base 10. This is the
"one kernel, several meanings" claim for the two conformers that carry no
derivative -- and, in the second half, where that claim stops.

What an interval buys, and what it costs:

- **Monotone functions enclose tightly.** `erf` over `[0.2, 0.9]` is
  exactly `[erf(0.2), erf(0.9)]`.
- **Non-monotone ones enclose loosely.** `j0` crosses zero and turns
  inside `[2, 3]`; the bound is valid, and wider than the true range,
  because each occurrence of `x` in the polynomial is treated as
  independent -- the dependency problem. Splitting the input into
  subintervals and taking the union tightens it quadratically, shown
  below with four pieces.
- **Alternating approximations do not enclose at all.** `gamma` is Lanczos'
  nine-term alternating sum, and over `[1.5, 2.5]` that sum's interval
  crosses zero, so `ln` of it is NaN. This is a property of interval
  arithmetic through a rational approximation, not a defect in the
  example, and the honest thing to do is show it.

`Decimal` has the complementary shape: `0.1 + 0.2` is exactly `0.3` and
stays exact through any arithmetic, while the transcendentals (`exp`,
`ln`, the series behind `erf` and `j0`) are fixed-iteration approximations
rounded to the fixed-point grid, so `gamma(4.5)` lands within a few parts
in ten thousand of the `Plain` value at six decimal places.
"""

from numax.core.interval import Interval
from numax.prelude import *
from numax.special import erf, gamma, gaussian, j0

comptime P = Plain[f64]
comptime I = Interval[P]
comptime Dc = Decimal[1, 6]


def interval(lo: Float64, hi: Float64) -> I:
    return I(P.constant(lo), P.constant(hi))


def sampled_range[
    f: def[T: FloatLike](T) thin -> T
](lo: Float64, hi: Float64) -> Tuple[Float64, Float64]:
    """The true range of `f` over `[lo, hi]`, by dense sampling -- the
    thing an enclosure must contain."""
    var smallest = f(P.constant(lo)).v[0]
    var largest = smallest
    for i in range(1, 1001):
        var y = f(P.constant(lo + (hi - lo) * Float64(i) / 1000.0)).v[0]
        if y < smallest:
            smallest = y
        if y > largest:
            largest = y
    return (smallest, largest)


def main() raises:
    print("erf over [0.2, 0.9]")
    print("  enclosure   ", erf(interval(0.2, 0.9)))
    var erf_range = sampled_range[erf](0.2, 0.9)
    print("  sampled     [", erf_range[0], ",", erf_range[1], "]")

    print("\nj0 over [2, 3], which crosses zero at 2.4048")
    var whole = j0(interval(2.0, 3.0))
    print("  enclosure   ", whole)
    var j0_range = sampled_range[j0](2.0, 3.0)
    print("  sampled     [", j0_range[0], ",", j0_range[1], "]")
    # Four subintervals, united: the dependency problem shrinks with the
    # width of each piece.
    var lo = j0(interval(2.0, 2.25)).lo.v[0]
    var hi = j0(interval(2.0, 2.25)).hi.v[0]
    for k in range(1, 4):
        var piece = j0(
            interval(2.0 + 0.25 * Float64(k), 2.25 + 0.25 * Float64(k))
        )
        if piece.lo.v[0] < lo:
            lo = piece.lo.v[0]
        if piece.hi.v[0] > hi:
            hi = piece.hi.v[0]
    print("  four pieces [", lo, ",", hi, "]")

    print("\ngaussian over [-1, 2], the dependency problem in one line")
    print("  enclosure   ", gaussian(interval(-1.0, 2.0)))
    var g_range = sampled_range[gaussian](-1.0, 2.0)
    print("  sampled     [", g_range[0], ",", g_range[1], "]")

    print("\ngamma over [1.5, 2.5]: where an enclosure stops being one")
    print("  enclosure   ", gamma(interval(1.5, 2.5)))
    var gamma_range = sampled_range[gamma](1.5, 2.5)
    print("  sampled     [", gamma_range[0], ",", gamma_range[1], "]")

    print("\nDecimal[1, 6]")
    var total = Dc.constant(0.1) + Dc.constant(0.2)
    print(
        "  0.1 + 0.2   =", total, " raw", total.raw, " (binary:", 0.1 + 0.2, ")"
    )
    print(
        "  erf(0.5)    =", erf(Dc.constant(0.5)), " Plain", erf(P.constant(0.5))
    )
    print(
        "  gamma(4.5)  =",
        gamma(Dc.constant(4.5)),
        " Plain",
        gamma(P.constant(4.5)),
    )
    print(
        "  j0(1.0)     =", j0(Dc.constant(1.0)), " Plain", j0(P.constant(1.0))
    )
