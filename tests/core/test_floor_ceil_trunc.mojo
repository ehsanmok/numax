"""Tests for `FloatLike.floor()`/`ceil()`/`trunc()` across every conformer.

The kernel that motivated this addition -- `numax.core.interval`'s tight
`sin`/`cos` enclosure -- has its own tests in `tests/core/test_interval.mojo`;
this file checks the trait methods themselves, one conformer at a time,
matching the `test_sqrt.mojo`/`test_sincos.mojo` precedent of one
cross-conformer file per trait addition.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from numax import Compensated, Complex, Decimal, Dual, Gradient, Interval, Plain

comptime dtype = DType.float64
comptime width = 1
comptime P = Plain[dtype, width]
comptime D = Dual[P]
comptime C = Complex[P]
comptime G1 = Gradient[P, 1]
comptime I = Interval[P]


def pv(x: Float64) -> P:
    return P(SIMD[dtype, width](x))


def test_plain_floor_ceil_trunc_on_positive_and_negative() raises:
    assert_almost_equal(pv(2.7).floor().v, SIMD[dtype, width](2.0))
    assert_almost_equal(pv(2.7).ceil().v, SIMD[dtype, width](3.0))
    assert_almost_equal(pv(2.7).trunc().v, SIMD[dtype, width](2.0))

    assert_almost_equal(pv(-2.7).floor().v, SIMD[dtype, width](-3.0))
    assert_almost_equal(pv(-2.7).ceil().v, SIMD[dtype, width](-2.0))
    assert_almost_equal(pv(-2.7).trunc().v, SIMD[dtype, width](-2.0))


def test_plain_floor_ceil_trunc_on_an_exact_integer_is_a_no_op() raises:
    assert_almost_equal(pv(4.0).floor().v, SIMD[dtype, width](4.0))
    assert_almost_equal(pv(4.0).ceil().v, SIMD[dtype, width](4.0))
    assert_almost_equal(pv(4.0).trunc().v, SIMD[dtype, width](4.0))


def test_dual_floor_ceil_trunc_zero_the_derivative() raises:
    # Step functions: the value follows Plain, the derivative is zero
    # (almost everywhere -- the discontinuity at each integer isn't
    # represented specially, matching every other blend in this library
    # that clamps rather than special-cases a measure-zero set).
    var x = D(pv(2.7), pv(1.0))
    var f = x.floor()
    assert_almost_equal(f.value.v, SIMD[dtype, width](2.0))
    assert_almost_equal(f.deriv.v, SIMD[dtype, width](0.0))

    var c = x.ceil()
    assert_almost_equal(c.value.v, SIMD[dtype, width](3.0))
    assert_almost_equal(c.deriv.v, SIMD[dtype, width](0.0))

    var t = x.trunc()
    assert_almost_equal(t.value.v, SIMD[dtype, width](2.0))
    assert_almost_equal(t.deriv.v, SIMD[dtype, width](0.0))


def test_gradient_floor_ceil_trunc_zero_every_partial() raises:
    var x = G1.variable(2.7, 0)
    var f = x.floor()
    assert_almost_equal(f.value.v, SIMD[dtype, width](2.0))
    assert_almost_equal(f.grad[0].v, SIMD[dtype, width](0.0))


def test_compensated_floor_ceil_trunc_zero_the_error_term() raises:
    # No refinement needed on an already-exact operation: `floor`/`ceil`/
    # `trunc` of a double-double value only ever need the `value` half,
    # since the true mathematical result is itself always exactly
    # representable in a single `dtype` lane.
    var x = Compensated[dtype, width](
        SIMD[dtype, width](2.7), SIMD[dtype, width](1e-20)
    )
    var f = x.floor()
    assert_almost_equal(f.value, SIMD[dtype, width](2.0))
    assert_almost_equal(f.error, SIMD[dtype, width](0.0))

    var c = x.ceil()
    assert_almost_equal(c.value, SIMD[dtype, width](3.0))

    var t = x.trunc()
    assert_almost_equal(t.value, SIMD[dtype, width](2.0))

    var neg = Compensated[dtype, width](
        SIMD[dtype, width](-2.7), SIMD[dtype, width](0)
    )
    assert_almost_equal(neg.floor().value, SIMD[dtype, width](-3.0))
    assert_almost_equal(neg.ceil().value, SIMD[dtype, width](-2.0))
    assert_almost_equal(neg.trunc().value, SIMD[dtype, width](-2.0))


def test_decimal_floor_ceil_trunc_are_exact_integer_arithmetic() raises:
    # Decimal's whole reason to exist: no float round-trip anywhere here,
    # just integer division on the raw fixed-point representation.
    comptime Dec = Decimal[width, 4]
    var x = Dec.constant(2.7)
    assert_equal(x.floor().raw, Dec.constant(2.0).raw)
    assert_equal(x.ceil().raw, Dec.constant(3.0).raw)
    assert_equal(x.trunc().raw, Dec.constant(2.0).raw)

    var neg = Dec.constant(-2.7)
    assert_equal(neg.floor().raw, Dec.constant(-3.0).raw)
    assert_equal(neg.ceil().raw, Dec.constant(-2.0).raw)
    assert_equal(neg.trunc().raw, Dec.constant(-2.0).raw)

    var exact = Dec.constant(4.0)
    assert_equal(exact.floor().raw, exact.raw)
    assert_equal(exact.ceil().raw, exact.raw)
    assert_equal(exact.trunc().raw, exact.raw)


def test_complex_floor_ceil_trunc_use_the_abs_copysign_embedding() raises:
    # No canonical complex meaning, so -- matching `abs`/`copysign` --
    # the modulus is floored/ceiled/trunc'd and re-embedded on the real
    # axis as Complex(x, 0).
    var z = C(pv(2.7), pv(0.0))
    var f = z.floor()
    assert_almost_equal(f.re.v, SIMD[dtype, width](2.0))
    assert_almost_equal(f.im.v, SIMD[dtype, width](0.0))

    var c = z.ceil()
    assert_almost_equal(c.re.v, SIMD[dtype, width](3.0))
    assert_almost_equal(c.im.v, SIMD[dtype, width](0.0))

    var t = z.trunc()
    assert_almost_equal(t.re.v, SIMD[dtype, width](2.0))
    assert_almost_equal(t.im.v, SIMD[dtype, width](0.0))


def test_interval_floor_ceil_map_each_bound_separately() raises:
    # floor/ceil are both monotone, so mapping each bound separately stays
    # a tight enclosure -- floor([2.7, 3.2]) = [2, 3], ceil([2.7, 3.2])
    # = [3, 4].
    var bounds = I(pv(2.7), pv(3.2))
    var f = bounds.floor()
    assert_almost_equal(f.lo.v, SIMD[dtype, width](2.0))
    assert_almost_equal(f.hi.v, SIMD[dtype, width](3.0))

    var c = bounds.ceil()
    assert_almost_equal(c.lo.v, SIMD[dtype, width](3.0))
    assert_almost_equal(c.hi.v, SIMD[dtype, width](4.0))


def test_interval_trunc_sorts_the_bounds_across_zero() raises:
    # trunc is not monotone across zero (trunc(-0.5) = 0 > trunc(-1.5) =
    # -1), so the straddling case is the one worth pinning: [-1.5, 0.5]
    # trunc's endpoints to {-1, 0}, already ordered, but the interior
    # value trunc(-0.5) = 0 must not escape the returned bounds either.
    var bounds = I(pv(-1.5), pv(0.5))
    var t = bounds.trunc()
    assert_almost_equal(t.lo.v, SIMD[dtype, width](-1.0))
    assert_almost_equal(t.hi.v, SIMD[dtype, width](0.0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
