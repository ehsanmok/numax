"""Tests for the `Writable` conformance every `FloatLike` type now carries.

The claim under test is that a caller who wants to look at a number writes
`print(x)` rather than reaching past the type for a raw field, and that a
nested type prints both of its levels rather than only the outer one.
"""

from std.collections import Array
from std.testing import TestSuite, assert_equal, assert_true

from numax import (
    Compensated,
    Complex,
    Decimal,
    Dual,
    Gradient,
    Interval,
    Plain,
    f64,
)

comptime P = Plain[f64, 1]


def test_plain_prints_the_number_alone() raises:
    assert_equal(String(P(2.5)), "2.5")


def test_dual_prints_both_components() raises:
    assert_equal(String(Dual[P](P(1.5), P(-2.0))), "Dual(1.5, -2.0)")


def test_a_nested_dual_prints_both_levels() raises:
    # A second-order value is a `Dual` whose components are `Dual`s, and
    # printing only the outer one would hide the second derivative.
    var inner = Dual[P](P(1.0), P(2.0))
    var outer = Dual[Dual[P]](inner.copy(), Dual[P](P(3.0), P(4.0)))
    assert_equal(String(outer), "Dual(Dual(1.0, 2.0), Dual(3.0, 4.0))")


def test_gradient_prints_every_partial() raises:
    assert_equal(
        String(Gradient[P, 2].variable(P(3.0), 0)), "Gradient(3.0, [1.0, 0.0])"
    )


def test_complex_prints_in_the_usual_notation() raises:
    assert_equal(String(Complex[P](P(1.0), P(2.0))), "1.0 + 2.0i")


def test_interval_prints_its_bounds() raises:
    assert_equal(String(Interval[P](P(0.0), P(1.0))), "[0.0, 1.0]")


def test_compensated_prints_both_limbs() raises:
    # The number is the sum, so printing `value` alone would claim a
    # precision the type spent its whole design carrying.
    var c = Compensated[f64, 1](1.5, 0.25)
    assert_equal(String(c), "1.5 + 0.25")


def test_decimal_prints_the_value_not_the_scaled_integer() raises:
    assert_equal(String(Decimal[1, 2](250)), "2.5")


def test_a_conformer_prints_the_same_value_its_field_holds() raises:
    # The point of the conformance: `print(x)` and `print(x.v)` agree, so
    # the field is only needed when the raw SIMD itself is wanted.
    var x = P(0.125)
    assert_equal(String(x), String(x.v))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
