"""Tests for `numax.core.interval`.

The central property is containment, not equality: whatever the true range
of a kernel is over an input interval, the computed interval must contain
it. Most tests here sample the input densely and assert every sample's
result falls inside the returned bounds, which catches an endpoint swap or
a dropped corner that a spot check of two numbers would not.
"""

from std.math import cos as cos_f64
from std.math import sin as sin_f64
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Compensated, Dual, FloatLike, Plain
from numax.core.interval import Interval

comptime dtype = DType.float64
comptime P = Plain[dtype, 1]
comptime I = Interval[P]


def _interval(lo: Float64, hi: Float64) -> I:
    return I(P.constant(lo), P.constant(hi))


def _contains(bounds: I, value: Float64) -> Bool:
    return bounds.lo.v[0] <= value and value <= bounds.hi.v[0]


def _assert_encloses[
    f: def[T: FloatLike](T) thin -> T
](lo: Float64, hi: Float64, samples: Int) raises:
    """Every sampled value of `f` must land inside `f` of the interval."""
    var bounds = f(_interval(lo, hi))
    for i in range(samples + 1):
        var x = lo + (hi - lo) * Float64(i) / Float64(samples)
        var y = f(P.constant(x)).v[0]
        assert_true(
            _contains(bounds, y),
            "f(" + String(x) + ") = " + String(y) + " escaped its bounds",
        )


def square[T: FloatLike](x: T) -> T:
    return x * x


def gaussian_like[T: FloatLike](x: T) -> T:
    return (-(x * x)).exp()


def scaled_exp[T: FloatLike](x: T) -> T:
    return (x * T.constant(0.5)).exp() + T.constant(1.0)


def ratio[T: FloatLike](x: T) -> T:
    return T.one() / (x + T.constant(3.0))


def test_addition_adds_endpoints() raises:
    var got = _interval(1.0, 2.0) + _interval(-1.0, 3.0)
    assert_almost_equal(got.lo.v[0], 0.0)
    assert_almost_equal(got.hi.v[0], 5.0)


def test_negation_flips_the_interval() raises:
    var got = -_interval(1.0, 2.0)
    assert_almost_equal(got.lo.v[0], -2.0)
    assert_almost_equal(got.hi.v[0], -1.0)


def test_multiplication_takes_the_extreme_corner() raises:
    # [1,2] * [-1,3] -- the minimum is 2*(-1), which a sign-naive
    # implementation multiplying lo*lo and hi*hi would miss entirely.
    var got = _interval(1.0, 2.0) * _interval(-1.0, 3.0)
    assert_almost_equal(got.lo.v[0], -2.0)
    assert_almost_equal(got.hi.v[0], 6.0)


def test_multiplication_of_two_negative_intervals_is_positive() raises:
    var got = _interval(-3.0, -1.0) * _interval(-2.0, -0.5)
    assert_almost_equal(got.lo.v[0], 0.5)
    assert_almost_equal(got.hi.v[0], 6.0)


def test_division_by_a_positive_interval() raises:
    var got = _interval(1.0, 2.0) / _interval(2.0, 4.0)
    assert_almost_equal(got.lo.v[0], 0.25)
    assert_almost_equal(got.hi.v[0], 1.0)


def test_monotone_functions_map_endpoints() raises:
    var e = _interval(1.0, 2.0).exp()
    assert_almost_equal(e.lo.v[0], 2.718281828459045)
    assert_almost_equal(e.hi.v[0], 7.38905609893065)

    var l = _interval(1.0, 8.0).ln()
    assert_almost_equal(l.lo.v[0], 0.0)
    assert_almost_equal(l.hi.v[0], 2.0794415416798357)

    var s = _interval(4.0, 9.0).sqrt()
    assert_almost_equal(s.lo.v[0], 2.0)
    assert_almost_equal(s.hi.v[0], 3.0)


def test_erfc_swaps_endpoints_because_it_decreases() raises:
    var bounds = _interval(0.0, 1.0).erfc()
    assert_true(
        bounds.lo.v[0] < bounds.hi.v[0], "erfc bounds came back inverted"
    )
    assert_almost_equal(bounds.hi.v[0], 1.0)
    assert_almost_equal(bounds.lo.v[0], 0.15729920705028513)


def test_abs_of_a_straddling_interval_reaches_zero() raises:
    var straddling = _interval(-1.0, 3.0).abs()
    assert_almost_equal(straddling.lo.v[0], 0.0)
    assert_almost_equal(straddling.hi.v[0], 3.0)

    var positive = _interval(2.0, 5.0).abs()
    assert_almost_equal(positive.lo.v[0], 2.0)
    assert_almost_equal(positive.hi.v[0], 5.0)

    var negative = _interval(-5.0, -2.0).abs()
    assert_almost_equal(negative.lo.v[0], 2.0)
    assert_almost_equal(negative.hi.v[0], 5.0)


def test_copysign_covers_both_signs_when_the_source_straddles() raises:
    var magnitude = _interval(1.0, 2.0)
    var from_positive = magnitude.copysign(_interval(1.0, 2.0))
    assert_almost_equal(from_positive.lo.v[0], 1.0)
    assert_almost_equal(from_positive.hi.v[0], 2.0)

    var from_negative = magnitude.copysign(_interval(-2.0, -1.0))
    assert_almost_equal(from_negative.lo.v[0], -2.0)
    assert_almost_equal(from_negative.hi.v[0], -1.0)

    var from_both = magnitude.copysign(_interval(-1.0, 3.0))
    assert_almost_equal(from_both.lo.v[0], -2.0)
    assert_almost_equal(from_both.hi.v[0], 2.0)


def test_enclosure_holds_for_sampled_kernels() raises:
    _assert_encloses[square](-2.0, 3.0, 200)
    _assert_encloses[gaussian_like](-1.5, 2.0, 200)
    _assert_encloses[scaled_exp](-3.0, 1.0, 200)
    # The divisor stays clear of zero, which is the documented requirement.
    _assert_encloses[ratio](-1.0, 4.0, 200)


def test_the_dependency_problem_is_real_and_documented() raises:
    # x*x over [-1,2] returns [-2,4] rather than [0,4], because the multiply
    # cannot know both operands are the same variable. Pinned as a test
    # because it is the first surprise any user of this type will hit, and
    # because a future "improvement" that tightened it would have to be a
    # square() special case rather than a change to __mul__.
    var loose = square(_interval(-1.0, 2.0))
    assert_almost_equal(loose.lo.v[0], -2.0)
    assert_almost_equal(loose.hi.v[0], 4.0)


def test_splitting_the_input_tightens_the_bound() raises:
    # The standard mitigation for the above: subdivide and take the union.
    # Over 64 pieces the spurious negative part of x*x nearly vanishes.
    var lo_bound = 1e30
    var hi_bound = -1e30
    comptime pieces = 64
    for i in range(pieces):
        var a = -1.0 + 3.0 * Float64(i) / Float64(pieces)
        var b = -1.0 + 3.0 * Float64(i + 1) / Float64(pieces)
        var piece = square(_interval(a, b))
        if piece.lo.v[0] < lo_bound:
            lo_bound = piece.lo.v[0]
        if piece.hi.v[0] > hi_bound:
            hi_bound = piece.hi.v[0]
    assert_true(lo_bound > -0.01, "subdivision failed to tighten the low end")
    assert_almost_equal(hi_bound, 4.0)


comptime _PI = 3.14159265358979323846


def test_sin_tightly_encloses_a_range_containing_the_peak() raises:
    # [0, pi/2] contains sin's peak at pi/2, so the tight enclosure is
    # [sin(0), 1] = [0, 1] -- not the trivial [-1, 1].
    var s = _interval(0.0, _PI / 2.0).sin()
    assert_almost_equal(s.lo.v[0], 0.0, atol=1e-12)
    assert_almost_equal(s.hi.v[0], 1.0, atol=1e-12)


def test_sin_tightly_encloses_a_range_containing_the_trough() raises:
    # [pi, 3*pi/2] contains sin's trough at 3*pi/2, so the tight enclosure
    # is [-1, sin(pi)] = [-1, 0].
    var s = _interval(_PI, 3.0 * _PI / 2.0).sin()
    assert_almost_equal(s.lo.v[0], -1.0, atol=1e-12)
    assert_almost_equal(s.hi.v[0], 0.0, atol=1e-12)


def test_sin_falls_back_to_the_trivial_enclosure_when_it_spans_both() raises:
    # A wide interval spanning many periods contains both a peak and a
    # trough, so [-1, 1] is the correct (and now the only sound) answer.
    var s = _interval(0.0, 100.0 * _PI).sin()
    assert_almost_equal(s.lo.v[0], -1.0, atol=1e-12)
    assert_almost_equal(s.hi.v[0], 1.0, atol=1e-12)


def test_sin_is_tight_on_a_monotone_piece() raises:
    # [0.1, 0.2] contains neither extremum, so sin is monotone there and
    # the tight enclosure is exactly the two endpoint values -- no [-1, 1]
    # looseness left at all.
    var s = _interval(0.1, 0.2).sin()
    assert_almost_equal(s.lo.v[0], sin_f64(0.1), atol=1e-12)
    assert_almost_equal(s.hi.v[0], sin_f64(0.2), atol=1e-12)


def test_cos_tightly_encloses_a_range_containing_the_peak() raises:
    # cos's peak is at 0; [-0.1, 0.1] contains it.
    var c = _interval(-0.1, 0.1).cos()
    assert_almost_equal(c.hi.v[0], 1.0, atol=1e-12)
    assert_almost_equal(c.lo.v[0], cos_f64(0.1), atol=1e-12)


def test_cos_tightly_encloses_a_range_containing_the_trough() raises:
    # cos's trough is at pi; [pi - 0.1, pi + 0.1] contains it.
    var c = _interval(_PI - 0.1, _PI + 0.1).cos()
    assert_almost_equal(c.lo.v[0], -1.0, atol=1e-12)
    assert_almost_equal(c.hi.v[0], cos_f64(_PI - 0.1), atol=1e-12)


def test_cos_falls_back_to_the_trivial_enclosure_when_it_spans_both() raises:
    var c = _interval(0.0, 100.0 * _PI).cos()
    assert_almost_equal(c.lo.v[0], -1.0, atol=1e-12)
    assert_almost_equal(c.hi.v[0], 1.0, atol=1e-12)


def test_inflate_widens_both_ends() raises:
    var widened = _interval(1.0, 2.0).inflate(0.01)
    assert_true(widened.lo.v[0] < 1.0, "lower bound did not move down")
    assert_true(widened.hi.v[0] > 2.0, "upper bound did not move up")
    assert_true(_contains(widened, 1.0) and _contains(widened, 2.0))


def test_degenerate_interval_behaves_like_a_scalar() raises:
    var point = I.degenerate(P.constant(3.0))
    var got = gaussian_like(point)
    var want = gaussian_like(P.constant(3.0)).v[0]
    assert_almost_equal(got.lo.v[0], want)
    assert_almost_equal(got.hi.v[0], want)
    assert_almost_equal(got.width().v[0], 0.0)


def test_midpoint_and_width() raises:
    var span = _interval(-1.0, 4.0)
    assert_almost_equal(span.midpoint().v[0], 1.5)
    assert_almost_equal(span.width().v[0], 5.0)


def test_interval_nests_inside_dual() raises:
    # Dual[Interval[Plain]] gives an interval-valued value and derivative:
    # d/dx exp(-x^2) = -2x*exp(-x^2), and the derivative's bounds must
    # contain the true derivative at every point of the input range.
    comptime DI = Dual[I]
    var seeded = DI(_interval(0.5, 1.0), I.one())
    var result = gaussian_like(seeded)

    for i in range(21):
        var x = 0.5 + 0.5 * Float64(i) / 20.0
        var value = gaussian_like(P.constant(x)).v[0]
        var deriv = -2.0 * x * value
        assert_true(_contains(result.value, value), "value escaped its bounds")
        assert_true(
            _contains(result.deriv, deriv), "derivative escaped its bounds"
        )


def test_interval_over_compensated_inner() raises:
    # The other nesting direction: bounds carried at double-double
    # precision. Checks only that it composes and stays ordered, since the
    # precision claim belongs to Compensated's own tests.
    comptime IC = Interval[Compensated[DType.float32, 1]]
    var bounds = IC(
        Compensated[DType.float32, 1].constant(1.0),
        Compensated[DType.float32, 1].constant(2.0),
    )
    var got = gaussian_like(bounds)
    assert_true(
        Float64(got.lo.value[0]) <= Float64(got.hi.value[0]),
        "compensated interval came back inverted",
    )


def test_lanes_are_independent() raises:
    comptime PW = Plain[dtype, 4]
    comptime IW = Interval[PW]
    var lo = SIMD[dtype, 4](0)
    var hi = SIMD[dtype, 4](0)
    for lane in range(4):
        lo[lane] = Float64(lane)
        hi[lane] = Float64(lane) + 2.0
    var bounds = square(IW(PW(lo), PW(hi)))
    for lane in range(4):
        var a = Float64(lane)
        var b = a + 2.0
        assert_almost_equal(bounds.lo.v[lane], a * a)
        assert_almost_equal(bounds.hi.v[lane], b * b)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
