"""Tests for `ember.Compensated` against float64 references.

`dtype` is `float32` throughout, so a plain `float32` result is only good to
about `1e-7`. Every assertion here checks that `value + error` -- the
compensated pair, folded back together -- lands far closer to a float64
reference than that, confirming the error-free transformations actually
recover the bits a bare `float32` would discard.
"""

from std.math import exp, log
from std.testing import TestSuite, assert_almost_equal, assert_true

from ember import Compensated, Plain, erf

comptime dtype = DType.float32
comptime width = 1


def combined(c: Compensated[dtype, width]) -> Float64:
    return Float64(c.value) + Float64(c.error)


def test_one() raises:
    var one = Compensated[dtype, width].one()
    assert_almost_equal(one.value, SIMD[dtype, width](1))
    assert_almost_equal(one.error, SIMD[dtype, width](0))


def test_add_recovers_precision_addition_alone_would_lose() raises:
    # 1.0 + 1e-8 rounds away entirely in float32; the compensated pair
    # should still recover it in the error term.
    var a = Compensated[dtype, width](
        SIMD[dtype, width](1.0), SIMD[dtype, width](0)
    )
    var b = Compensated[dtype, width](
        SIMD[dtype, width](1e-8), SIMD[dtype, width](0)
    )
    var c = a + b
    var err = abs(combined(c) - 1.00000001)
    assert_true(err < 1e-12, msg=String("addition error too large: ", err))


def test_mul_recovers_precision_a_plain_f32_product_would_lose() raises:
    # Both factors are exact in float32 (integers under 2^24, scaled by an
    # exact power of two), so their product's true value -- exact integer
    # times exact integer, scaled -- is itself exact in float64. A plain
    # float32 multiply can only keep ~24 bits of that ~48-bit product;
    # the compensated pair should keep (nearly) all of it.
    comptime scale = SIMD[dtype, width](1.0 / 16777216.0)  # 2^-24, exact
    var a_int = SIMD[dtype, width](16777213.0)  # 2^24 - 3, exact in float32
    var b_int = SIMD[dtype, width](16777211.0)  # 2^24 - 5, exact in float32

    var a = Compensated[dtype, width](a_int * scale, SIMD[dtype, width](0))
    var b = Compensated[dtype, width](b_int * scale, SIMD[dtype, width](0))
    var c = a * b

    var reference = (Float64(16777213.0) * Float64(16777211.0)) / Float64(
        16777216.0 * 16777216.0
    )
    var compensated_err = abs(combined(c) - reference)

    assert_true(
        compensated_err < 1e-13,
        msg=String("multiplication error too large: ", compensated_err),
    )


def test_div_matches_f64_reference() raises:
    var a = Compensated[dtype, width](
        SIMD[dtype, width](1.0), SIMD[dtype, width](0)
    )
    var b = Compensated[dtype, width](
        SIMD[dtype, width](3.0), SIMD[dtype, width](0)
    )
    var c = a / b
    var err = abs(combined(c) - (1.0 / 3.0))
    assert_true(err < 1e-13, msg=String("division error too large: ", err))


def test_exp_beats_plain_f32() raises:
    # Widen the float32 input to float64 exactly (no new rounding) before
    # computing the reference, so both `plain_err` and `compensated_err`
    # measure the precision of `exp` itself, not the input's own float32
    # representation error against an arbitrary decimal literal.
    var x = SIMD[dtype, width](0.7)
    var reference = exp(Float64(x))

    var plain_err = abs(Float64(exp(x)) - reference)

    var c = Compensated[dtype, width](x, SIMD[dtype, width](0)).exp()
    var compensated_err = abs(combined(c) - reference)

    assert_true(
        compensated_err < plain_err,
        msg=String(
            "compensated exp should beat plain f32: compensated=",
            compensated_err,
            " plain=",
            plain_err,
        ),
    )
    assert_true(compensated_err < 1e-12)


def test_ln_beats_plain_f32() raises:
    # Same shape as `test_exp_beats_plain_f32`: widen to float64 exactly
    # before computing the reference so both errors measure `ln` itself.
    var x = SIMD[dtype, width](12345.6789)
    var reference = log(Float64(x))

    var plain_err = abs(Float64(log(x)) - reference)

    var c = Compensated[dtype, width](x, SIMD[dtype, width](0)).ln()
    var compensated_err = abs(combined(c) - reference)

    assert_true(
        compensated_err < plain_err,
        msg=String(
            "compensated ln should beat plain f32: compensated=",
            compensated_err,
            " plain=",
            plain_err,
        ),
    )
    assert_true(compensated_err < 1e-10)


def test_ln_undoes_exp() raises:
    var x = Compensated[dtype, width](
        SIMD[dtype, width](2.5), SIMD[dtype, width](0)
    )
    var roundtrip = x.exp().ln()
    assert_true(abs(combined(roundtrip) - 2.5) < 1e-6)


def test_erf_compensated_beats_plain_for_small_x() raises:
    # `erf`'s rational approximation computes `1 - poly(x) * exp(-x^2)`,
    # which is a near-total cancellation for small `x` -- exactly the
    # pattern compensated arithmetic is built to survive. At x=1e-5, plain
    # float32 loses most of its significant digits in that subtraction;
    # the compensated pair should not.
    var x = SIMD[dtype, width](1e-05)
    var reference: Float64 = 1.1283791670579e-05

    var plain_err = abs(Float64(erf(Plain[dtype, width](x)).v) - reference)

    var c = erf(Compensated[dtype, width](x, SIMD[dtype, width](0)))
    var compensated_err = abs(combined(c) - reference)

    assert_true(
        compensated_err < plain_err,
        msg=String(
            "compensated erf should beat plain f32 for small x: compensated=",
            compensated_err,
            " plain=",
            plain_err,
        ),
    )
    assert_true(compensated_err < 1e-8)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
