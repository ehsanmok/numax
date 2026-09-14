"""The special functions at `Interval` and `Decimal`: the two conformers
that carry no derivative and are the easiest to leave untested.

The claim under test is the library's: a kernel written once against
`FloatLike` means an enclosure when called at `Interval` and exact decimal
arithmetic when called at `Decimal`. So these tests instantiate the special
functions at both and check the property each conformer promises --
containment for `Interval`, agreement with `Plain` to the conformer's own
precision for `Decimal` -- rather than a value table.

What is deliberately not here: `gamma`, `lgamma` and `expi` at `Interval`.
Lanczos' alternating nine-term sum and the series/asymptotic blend in
`expi` both evaluate a "discarded" side whose interval reaches infinity,
and `0 * inf` is NaN. Interval arithmetic through a rational approximation
with alternating coefficients is not an enclosure of anything useful, and
the suite records that boundary here instead of asserting a NaN.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from numax import Decimal, FloatLike, Plain
from numax.core.interval import Interval
from numax.special import erf, erfc, expi, gamma, gaussian, j0, j1, lgamma

comptime P = Plain[DType.float64]
comptime I = Interval[P]
comptime scale = 6
comptime Dc = Decimal[1, scale]


def _interval(lo: Float64, hi: Float64) -> I:
    return I(P.constant(lo), P.constant(hi))


def _assert_encloses[
    f: def[T: FloatLike](T) thin -> T
](lo: Float64, hi: Float64, samples: Int = 64) raises:
    """Every sampled `f(x)` for `x` in `[lo, hi]` lands inside `f` of the
    interval -- the containment property, sampled densely so an endpoint
    swap or a dropped corner shows up."""
    var bounds = f(_interval(lo, hi))
    var blo = bounds.lo.v[0]
    var bhi = bounds.hi.v[0]
    assert_true(blo <= bhi, "bounds are not ordered")
    for i in range(samples + 1):
        var x = lo + (hi - lo) * Float64(i) / Float64(samples)
        var y = f(P.constant(x)).v[0]
        assert_true(
            blo <= y and y <= bhi,
            "f(" + String(x) + ") = " + String(y) + " escaped its bounds",
        )


def test_erf_encloses_its_range() raises:
    # Monotone, so the enclosure is exact: the endpoints map to the
    # endpoints and every sample sits strictly inside.
    _assert_encloses[erf](0.2, 0.9)
    _assert_encloses[erf](-0.5, 0.5)
    _assert_encloses[erf](-3.0, 3.0)


def test_erf_enclosure_is_tight_when_monotone() raises:
    var bounds = erf(_interval(0.2, 0.9))
    assert_almost_equal(bounds.lo.v, erf(P.constant(0.2)).v)
    assert_almost_equal(bounds.hi.v, erf(P.constant(0.9)).v)


def test_erfc_encloses_and_reverses_the_endpoints() raises:
    _assert_encloses[erfc](0.2, 0.9)
    var bounds = erfc(_interval(0.2, 0.9))
    assert_almost_equal(bounds.lo.v, erfc(P.constant(0.9)).v)
    assert_almost_equal(bounds.hi.v, erfc(P.constant(0.2)).v)


def test_gaussian_encloses_across_its_peak() raises:
    # `exp(-x*x)` is not monotone over an interval containing zero, and
    # the `x * x` inside is the textbook dependency problem, so the
    # enclosure is valid but loose. Containment is the property; tightness
    # is not promised.
    _assert_encloses[gaussian](-1.0, 2.0)
    _assert_encloses[gaussian](0.5, 1.5)


def test_bessel_j0_and_j1_enclose_inside_the_near_branch() raises:
    # Both are polynomial-in-`x^2` on `|x| < 8`, and `j0` crosses zero at
    # 2.4048 inside the second interval, so this covers a sign change.
    _assert_encloses[j0](0.5, 1.5)
    _assert_encloses[j0](2.0, 3.0)
    _assert_encloses[j0](5.0, 6.0)
    _assert_encloses[j1](0.5, 1.5)
    _assert_encloses[j1](3.0, 4.0)


def test_degenerate_interval_is_the_point_value() raises:
    # `[x, x]` through any of these is `[f(x), f(x)]` up to rounding.
    var x = I.degenerate(P.constant(0.7))
    assert_almost_equal(erf(x).lo.v, erf(P.constant(0.7)).v)
    assert_almost_equal(erf(x).hi.v, erf(P.constant(0.7)).v)
    assert_almost_equal(j0(x).lo.v, j0(P.constant(0.7)).v, atol=1e-12)
    assert_almost_equal(j0(x).hi.v, j0(P.constant(0.7)).v, atol=1e-12)


def _dec(x: Float64) -> Float64:
    return Dc.constant(x).to_float64()[0]


def test_decimal_arithmetic_is_exact_through_a_kernel() raises:
    # `0.1 + 0.2` is exactly `0.3` here, and stays exact through the
    # arithmetic a kernel does with it -- the property `Decimal` exists for.
    var total = Dc.constant(0.1) + Dc.constant(0.2)
    assert_equal(total.raw, SIMD[DType.int64, 1](300_000))
    var scaled = total * Dc.constant(3.0)
    assert_equal(scaled.raw, SIMD[DType.int64, 1](900_000))
    assert_equal(scaled.to_float64(), SIMD[DType.float64, 1](0.9))


def test_decimal_erf_agrees_with_plain_to_the_approximation() raises:
    # `Decimal.erf` is the shared A&S 7.1.26 fallback (`1.5e-7`), read out
    # at `10^-6`: so agreement to `1e-5` is the honest bound.
    var xs = [0.25, 0.5, 1.0, 1.75]
    for i in range(4):
        var dec = erf(Dc.constant(xs[i])).to_float64()[0]
        var expected = erf(P.constant(xs[i])).v[0]
        assert_almost_equal(dec, expected, atol=1e-5)


def test_decimal_bessel_and_expi_agree_with_plain() raises:
    assert_almost_equal(
        j0(Dc.constant(1.0)).to_float64()[0],
        j0(P.constant(1.0)).v[0],
        atol=1e-5,
    )
    assert_almost_equal(
        j1(Dc.constant(2.0)).to_float64()[0],
        j1(P.constant(2.0)).v[0],
        atol=1e-5,
    )
    assert_almost_equal(
        expi(Dc.constant(1.5)).to_float64()[0],
        expi(P.constant(1.5)).v[0],
        atol=1e-5,
    )


def test_decimal_gamma_agrees_with_plain_to_fixed_point_precision() raises:
    # `lgamma` at `Decimal` is Lanczos through a fixed-iteration `ln` and
    # `exp` at six decimal places, and every intermediate rounds to that
    # grid, so the result carries a few parts in ten thousand of error --
    # a scope limit of fixed-point transcendentals, stated rather than
    # hidden. The arithmetic around them stays exact.
    var xs = [2.5, 3.0, 4.5]
    for i in range(3):
        var dec = gamma(Dc.constant(xs[i])).to_float64()[0]
        var expected = gamma(P.constant(xs[i])).v[0]
        assert_almost_equal(dec, expected, rtol=1e-3)
    assert_almost_equal(
        lgamma(Dc.constant(3.0)).to_float64()[0],
        lgamma(P.constant(3.0)).v[0],
        atol=1e-3,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
