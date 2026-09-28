"""Tests for `gammaincinv` against SciPy's digits from `1e-10` to `0.999`
at shapes on both sides of NR's `a = 1` split, its round trip through
`gammainc`, its edges, and its derivative through `Dual`, which the
Halley steps carry to `1 / (dP/dx)`."""

from std.math import exp as _exp, isinf
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Dual, Plain, gammainc, gammaincinv, lgamma

comptime P = Plain[DType.float64, 1]
comptime D = Dual[P]


def _inv(a: Float64, y: Float64) -> Float64:
    return gammaincinv(P(a), P(y)).v[0]


def test_matches_scipy() raises:
    # scipy.special.gammaincinv.
    assert_almost_equal(_inv(0.1, 1e-10), 6.073048362408172e-101, rtol=1e-13)
    assert_almost_equal(_inv(0.1, 0.5), 0.0005933911044602284, rtol=1e-13)
    assert_almost_equal(_inv(0.5, 1e-10), 7.85398163397448e-21, rtol=1e-13)
    assert_almost_equal(_inv(0.5, 0.9), 1.352771727047709, rtol=1e-13)
    assert_almost_equal(_inv(1.0, 0.5), 0.6931471805599455, rtol=1e-13)
    assert_almost_equal(_inv(2.5, 1e-10), 0.00016167785731248486, rtol=1e-13)
    assert_almost_equal(_inv(2.5, 0.1), 0.8051539934811613, rtol=1e-13)
    assert_almost_equal(_inv(10.0, 1e-10), 0.4727220926063524, rtol=1e-13)
    assert_almost_equal(_inv(10.0, 0.5), 9.66871461471413, rtol=1e-13)
    assert_almost_equal(_inv(30.0, 1e-10), 7.000511700351831, rtol=1e-13)
    assert_almost_equal(_inv(30.0, 0.9), 37.1985028596843, rtol=1e-13)
    # The upper tail carries `gammainc`'s absolute error near 1 over the
    # density, so it is checked at the looser bound the docstring states.
    assert_almost_equal(_inv(10.0, 0.999), 22.65737330906293, rtol=1e-12)


def test_round_trip() raises:
    var shapes: List[Float64] = [0.3, 1.0, 4.0, 15.0]
    var ys: List[Float64] = [1e-8, 0.02, 0.4, 0.8]
    for a in shapes:
        for y in ys:
            var x = _inv(a, y)
            assert_almost_equal(gammainc(P(a), P(x)).v[0], y, rtol=1e-13)


def test_edges() raises:
    assert_almost_equal(_inv(2.0, 0.0), 0.0, atol=0.0)
    assert_true(isinf(_inv(2.0, 1.0)))


def test_derivative_is_the_reciprocal_density() raises:
    var a = 3.0
    var y = 0.35
    var r = gammaincinv(D(P(a), P(0.0)), D(P(y), P(1.0)))
    var x = r.value.v[0]
    var density = _exp((a - 1.0) * _log(x) - x - lgamma(P(a)).v[0])
    assert_almost_equal(r.deriv.v[0], 1.0 / density, rtol=1e-10)


def _log(x: Float64) -> Float64:
    return P(x).ln().v[0]


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
