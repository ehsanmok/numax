"""Tests for the branchless helpers every tier-1 kernel is built from:
`max_of`, `min_of`, `ge_indicator`, `guard_nonzero`, `blend`. The claims
are exactness -- a clamp returns one of its operands bit for bit -- and
the boundary conventions the kernels rely on."""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from numax import Dual, Plain
from numax.core.numeric import (
    blend,
    ge_indicator,
    guard_nonzero,
    max_of,
    min_of,
)
from numax.special import sici

comptime dtype = DType.float64
comptime P = Plain[dtype, 1]
comptime D = Dual[P]


def pv(x: Float64) -> P:
    return P.constant(x)


def s(x: P) -> Float64:
    return Float64(x.v)


def test_clamps_are_exact_selections_across_magnitudes() raises:
    """The old `(a + b -+ |a - b|) / 2` identity lost the small operand
    against the large one; the selection form returns it unchanged."""
    assert_equal(s(min_of(pv(1e-9), pv(2.0))), 1e-9)
    assert_equal(s(min_of(pv(1e-30), pv(2.0))), 1e-30)
    assert_equal(s(max_of(pv(-1.0), pv(1e-30))), 1e-30)
    assert_equal(s(max_of(pv(1e-300), pv(1e-320))), 1e-300)
    assert_equal(s(max_of(pv(3.0), pv(-7.0))), 3.0)
    assert_equal(s(min_of(pv(3.0), pv(-7.0))), -7.0)
    assert_equal(s(max_of(pv(2.5), pv(2.5))), 2.5)
    # A clamp of a tiny argument before a blend no longer moves it.
    assert_almost_equal(s(sici(pv(1e-9))[0]), 1e-9, rtol=1e-15)


def test_ge_indicator_boundary_and_guard_convention() raises:
    assert_equal(s(ge_indicator(pv(2.0), pv(2.0))), 1.0)
    assert_equal(s(ge_indicator(pv(1.9999999), pv(2.0))), 0.0)
    # `-0.0 - 0.0` is `-0.0`, whose sign `copysign` reads as negative: a
    # negative zero sits on the `<` side. Kernels that test `x >= 0` see
    # `-0.0` as negative, which every blend in `numax` tolerates because
    # both sides agree at zero.
    assert_equal(s(ge_indicator(pv(-0.0), pv(0.0))), 0.0)
    assert_equal(s(ge_indicator(pv(0.0), pv(0.0))), 1.0)
    assert_equal(s(guard_nonzero(pv(0.0), pv(1e-30))), 1e-30)
    assert_equal(s(guard_nonzero(pv(-1e-40), pv(1e-30))), -1e-30)
    assert_equal(s(guard_nonzero(pv(-3.0), pv(1e-30))), -3.0)
    assert_equal(s(blend(pv(1.0), pv(7.0), pv(-7.0))), 7.0)
    assert_equal(s(blend(pv(0.0), pv(7.0), pv(-7.0))), -7.0)


def test_clamps_select_the_derivative_too() raises:
    """At `Dual`, `max_of(x, c)` differentiates to `1` where `x` wins and
    `0` where the constant does."""
    var wins = max_of(D(pv(3.0), pv(1.0)), D(pv(1.0), pv(0.0)))
    assert_equal(Float64(wins.value.v), 3.0)
    assert_equal(Float64(wins.deriv.v), 1.0)
    var loses = max_of(D(pv(-3.0), pv(1.0)), D(pv(1.0), pv(0.0)))
    assert_equal(Float64(loses.value.v), 1.0)
    assert_equal(Float64(loses.deriv.v), 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
