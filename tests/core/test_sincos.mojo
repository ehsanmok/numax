"""Tests for `FloatLike.sin()`/`cos()` across every conformer.

`Plain` delegates straight to `std.math`, so it's the reference the other
three are checked against (or, for `Compensated`, checked to beat).
"""

from std.math import cos as cos_f64
from std.math import pi
from std.math import sin as sin_f64
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Compensated, Decimal, Dual, Plain

comptime dtype = DType.float32
comptime width = 1
comptime D = Dual[Plain[dtype, width]]


def pv(x: Float64) -> Plain[dtype, width]:
    return Plain[dtype, width].constant(x)


def test_plain_sin_cos_match_known_values() raises:
    assert_almost_equal(Plain[dtype, width](0).sin().v, SIMD[dtype, width](0))
    assert_almost_equal(Plain[dtype, width](0).cos().v, SIMD[dtype, width](1))
    assert_almost_equal(
        pv(Float64(pi) / 2.0).sin().v, SIMD[dtype, width](1), atol=1e-6
    )
    assert_almost_equal(pv(Float64(pi)).cos().v, SIMD[dtype, width](-1))


def test_plain_sin_squared_plus_cos_squared_is_one() raises:
    var x = pv(1.3)
    var s = x.sin()
    var c = x.cos()
    assert_almost_equal(s.v * s.v + c.v * c.v, SIMD[dtype, width](1))


def test_dual_derivative_of_sin_is_cos() raises:
    var x = D(pv(0.6), pv(1))
    var s = x.sin()
    assert_almost_equal(s.deriv.v, pv(0.6).cos().v)


def test_dual_derivative_of_cos_is_negative_sin() raises:
    var x = D(pv(0.6), pv(1))
    var c = x.cos()
    assert_almost_equal(c.deriv.v, -pv(0.6).sin().v)


def test_compensated_sin_beats_plain_for_large_x() raises:
    # A large-ish x forces `Compensated`'s range reduction to do real work
    # (several multiples of 2*pi); plain float32 loses precision the same
    # way it does for `exp`/`ln` at inputs that aren't near zero.
    var x = SIMD[dtype, width](100.5)
    var reference = sin_f64(Float64(x))

    var plain_err = abs(Float64(Plain[dtype, width](x).sin().v) - reference)

    var c = Compensated[dtype, width](x, SIMD[dtype, width](0)).sin()
    var compensated_err = abs((Float64(c.value) + Float64(c.error)) - reference)

    assert_true(
        compensated_err < plain_err,
        msg=String(
            "compensated sin should beat plain f32: compensated=",
            compensated_err,
            " plain=",
            plain_err,
        ),
    )
    assert_true(compensated_err < 1e-9)


def test_compensated_sin_squared_plus_cos_squared_is_one() raises:
    var x = Compensated[dtype, width](
        SIMD[dtype, width](7.25), SIMD[dtype, width](0)
    )
    var s = x.sin()
    var c = x.cos()
    var total = s * s + c * c
    var combined = Float64(total.value) + Float64(total.error)
    assert_true(abs(combined - 1.0) < 1e-9)


def test_decimal_sin_cos_identity() raises:
    comptime Dec = Decimal[width, 4]
    var x = Dec.constant(0.5)
    var s = x.sin()
    var c = x.cos()
    var total = s * s + c * c
    assert_true(
        abs(Float64(total.raw) / 10000.0 - 1.0) < 1e-3,
        msg="decimal sin^2+cos^2 should be close to 1",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
