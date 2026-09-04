"""Tests for `numax.Decimal`, the base-10 fixed-point `FloatLike` conformer."""

from std.testing import TestSuite, assert_equal, assert_true

from numax import Decimal, gaussian, sigmoid

comptime width = 1
comptime scale = 6
comptime D = Decimal[width, scale]
comptime factor = 1_000_000


def test_one_is_scaled_by_factor() raises:
    assert_equal(D.one().raw, SIMD[DType.int64, width](factor))


def test_to_float64_descales_the_raw_integer() raises:
    # The read-out is exactly `raw / 10^scale`, and round-trips a value the
    # type can represent exactly.
    var a = D.constant(0.1) + D.constant(0.2)
    assert_equal(a.raw, SIMD[DType.int64, width](300_000))
    assert_equal(a.to_float64(), SIMD[DType.float64, width](0.3))


def test_add_is_exact_for_decimal_literals() raises:
    # 0.1 + 0.2 is exactly 0.3 in base-10 fixed point -- the whole point of
    # this type versus a binary float.
    var a = D.constant(0.1)
    var b = D.constant(0.2)
    var c = a + b
    assert_equal(c.raw, SIMD[DType.int64, width](300_000))


def test_mul_matches_expected_product() raises:
    var a = D.constant(3.0)
    var b = D.constant(4.0)
    assert_equal((a * b).raw, SIMD[DType.int64, width](12_000_000))


def test_div_matches_expected_quotient() raises:
    var a = D.constant(3.0)
    var b = D.constant(4.0)
    assert_equal((a / b).raw, SIMD[DType.int64, width](750_000))


def test_neg() raises:
    var a = D.constant(2.5)
    assert_equal((-a).raw, SIMD[DType.int64, width](-2_500_000))


def test_abs() raises:
    var a = D.constant(-5.0)
    assert_equal(a.abs().raw, SIMD[DType.int64, width](5_000_000))


def test_copysign() raises:
    var mag = D.constant(5.0)
    var neg_source = D.constant(-1.0)
    assert_equal(
        mag.copysign(neg_source).raw, SIMD[DType.int64, width](-5_000_000)
    )


def test_exp_matches_f64_reference_within_a_few_raw_units() raises:
    var one = D.constant(1.0)
    var e = one.exp()
    var reference = SIMD[DType.int64, width](2_718_282)
    var diff = abs(Int64(e.raw[0]) - Int64(reference[0]))
    assert_true(diff <= 10, msg=String("exp(1) raw off by too much: ", e.raw))


def test_ln_undoes_exp() raises:
    # `exp()`'s fixed 30-term series and `ln()`'s truncating-division
    # arithmetic each drift by a handful of raw units (`10^-scale`) on
    # their own; round-tripping through both compounds that, so this
    # checks the result stays within a small fraction of a percent of `x`
    # rather than expecting near-exact agreement.
    var x = D.constant(2.5)
    var roundtrip = x.exp().ln()
    var diff = abs(Int64(roundtrip.raw[0]) - Int64(x.raw[0]))
    assert_true(
        diff <= 200, msg=String("ln(exp(x)) drifted too far from x: ", diff)
    )


def test_gaussian_kernel_works_over_decimal() raises:
    # The same `FloatLike` kernel `special.mojo` uses for `Plain`/`Dual`/
    # `Compensated`, called with `Decimal` -- no changes to `gaussian`
    # itself.
    var x = D.constant(0.0)
    var g = gaussian(x)
    assert_equal(g.raw, SIMD[DType.int64, width](factor))


def test_sigmoid_kernel_works_over_decimal() raises:
    var x = D.constant(0.0)
    var s = sigmoid(x)
    assert_equal(s.raw, SIMD[DType.int64, width](factor // 2))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
