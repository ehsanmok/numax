"""Tests for `polygamma` against SciPy's digits at orders 0 to 20 and on
the negative axis, the recurrence `psi^(n)(x + 1) = psi^(n)(x) + (-1)^n n!
/ x^(n+1)`, the closed form `psi^(1)(1) = pi^2 / 6`, and `Dual`'s
derivative of order `n` being order `n + 1`."""

from std.testing import TestSuite, assert_almost_equal

from numax import Dual, Plain, digamma, polygamma

comptime P = Plain[DType.float64, 1]
comptime D = Dual[P]


def _pg(n: Int, x: Float64) -> Float64:
    return polygamma(n, P(x)).v[0]


def test_matches_scipy() raises:
    # scipy.special.polygamma.
    var ns: List[Int] = [1, 1, 1, 2, 3, 5, 10, 20, 1, 3]
    var xs: List[Float64] = [
        0.2,
        5.0,
        -0.5,
        1.0,
        0.7,
        12.0,
        3.0,
        4.0,
        -7.3,
        -2.6,
    ]
    var want: List[Float64] = [
        26.26737720542378,
        0.22132295573711533,
        8.934802200544679,
        -2.404113806319188,
        25.879149678427744,
        0.0001182082905108149,
        -21.43660662871299,
        -558395.5858139333,
        14.951383181433922,
        283.5383805977357,
    ]
    for i in range(len(ns)):
        assert_almost_equal(_pg(ns[i], xs[i]), want[i], atol=0.0, rtol=4e-15)


def test_order_zero_is_digamma() raises:
    assert_almost_equal(_pg(0, 2.5), digamma(P(2.5)).v[0], atol=0.0)


def test_recurrence_and_closed_form() raises:
    assert_almost_equal(_pg(1, 1.0), 1.6449340668482264, atol=0.0, rtol=2e-16)
    var x = 1.7
    # n = 3: psi'''(x + 1) = psi'''(x) - 6 / x^4.
    assert_almost_equal(
        _pg(3, x + 1.0), _pg(3, x) - 6.0 / (x * x * x * x), rtol=1e-14
    )


def test_dual_derivative_is_the_next_order() raises:
    var r = polygamma(2, D(P(3.2), P(1.0)))
    assert_almost_equal(r.deriv.v[0], _pg(3, 3.2), atol=0.0, rtol=1e-14)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
