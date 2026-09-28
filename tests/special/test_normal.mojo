"""Tests for `ndtr`, `log_ndtr`, `ndtri`, `expit` and `log_expit` against
SciPy's digits -- the far tails included, where the naive forms underflow
or cancel -- and for the identities between them: the round trip, the
derivatives through `Dual`, and `expit` being `sigmoid`."""

from std.math import exp as _exp, isinf, isnan, sqrt as _sqrt
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import (
    Dual,
    Plain,
    expit,
    log_expit,
    log_ndtr,
    ndtr,
    ndtri,
    sigmoid,
)

comptime P = Plain[DType.float64, 1]
comptime D = Dual[P]


def _v(x: P) -> Float64:
    return x.v[0]


def _close(got: Float64, want: Float64, rel: Float64 = 4e-15) raises:
    assert_almost_equal(got, want, atol=0.0, rtol=rel)


def test_ndtri_matches_scipy_into_the_tails() raises:
    var ps: List[Float64] = [1e-300, 1e-100, 1e-20, 1e-12, 1e-05, 0.01, 0.075]
    var want: List[Float64] = [
        -37.0470962993612,
        -21.273453560965322,
        -9.262340089798409,
        -7.034483825301131,
        -4.264890793922825,
        -2.3263478740408408,
        -1.4395314709384563,
    ]
    for i in range(len(ps)):
        _close(_v(ndtri(P(ps[i]))), want[i])
    _close(_v(ndtri(P(0.7))), 0.5244005127080407)
    _close(_v(ndtri(P(0.99))), 2.3263478740408408)
    _close(_v(ndtri(P(1.0 - 1e-10))), 6.361340889697422)
    assert_almost_equal(_v(ndtri(P(0.5))), 0.0, atol=1e-300)


def test_ndtri_edges() raises:
    var lo = _v(ndtri(P(0.0)))
    var hi = _v(ndtri(P(1.0)))
    assert_true(isinf(lo) and lo < 0.0)
    assert_true(isinf(hi) and hi > 0.0)
    assert_true(isnan(_v(ndtri(P(-0.1)))))
    assert_true(isnan(_v(ndtri(P(1.5)))))


def test_ndtr_and_log_ndtr_match_scipy() raises:
    var xs: List[Float64] = [-50.0, -20.5, -19.5, -5.0, -1.0, 1.0, 5.0, 10.0]
    var logs: List[Float64] = [
        -1254.83136113942,
        -214.06672896326384,
        -194.01696577749755,
        -15.064998393988727,
        -1.8410216450092634,
        -0.1727537790234499,
        -2.866516129637631e-07,
        -7.619853024160474e-24,
    ]
    for i in range(len(xs)):
        _close(_v(log_ndtr(P(xs[i]))), logs[i], 5e-15)
    _close(_v(ndtr(P(-20.5))), 1.0764673258790346e-93, 2e-14)
    _close(_v(ndtr(P(-5.0))), 2.8665157187919344e-07)
    _close(_v(ndtr(P(1.0))), 0.8413447460685429)
    assert_almost_equal(_v(ndtr(P(0.0))), 0.5, atol=0.0)


def test_round_trip() raises:
    # The upper side stops at 3: `ndtr(7.5)` is within `3e-14` of 1, so
    # no inverse could recover 7.5 from the rounded value.
    var xs: List[Float64] = [-30.0, -8.0, -2.5, -0.3, 0.0, 0.4, 1.5, 3.0]
    for x in xs:
        assert_almost_equal(_v(ndtri(ndtr(P(x)))), x, atol=2e-14, rtol=1e-13)


def test_derivatives_through_dual() raises:
    # d ndtr / dx = phi(x); d log_ndtr / dx = phi(x) / ndtr(x);
    # d ndtri / dp = 1 / phi(ndtri(p)).
    var inv_sqrt_2pi = 0.3989422804014327
    var x = 0.8
    var phi = inv_sqrt_2pi * _exp(-0.5 * x * x)
    var d = ndtr(D(P(x), P(1.0)))
    _close(d.deriv.v[0], phi, 1e-14)
    var l = log_ndtr(D(P(x), P(1.0)))
    _close(l.deriv.v[0], phi / _v(ndtr(P(x))), 1e-14)
    var tail = log_ndtr(D(P(-30.0), P(1.0)))
    # Mills' ratio: phi(x) / Phi(x) ~ -x - 1/x + 2/x^3 far below.
    assert_almost_equal(tail.deriv.v[0], 30.0333, atol=1e-3)
    var q = ndtri(D(P(0.2), P(1.0)))
    var at = q.value.v[0]
    _close(q.deriv.v[0], 1.0 / (inv_sqrt_2pi * _exp(-0.5 * at * at)), 1e-9)


def test_expit_is_sigmoid_and_log_expit_is_stable() raises:
    var xs: List[Float64] = [-40.0, -5.0, 0.0, 1.0, 40.0]
    for x in xs:
        assert_almost_equal(_v(expit(P(x))), _v(sigmoid(P(x))), atol=0.0)
    _close(_v(log_expit(P(-5.0))), -5.006715348489118)
    _close(_v(log_expit(P(10.0))), -4.539889921686465e-05)
    # `log(expit(40))` rounds to 0; the stable form keeps `-exp(-40)`.
    _close(_v(log_expit(P(40.0))), -4.248354255291589e-18)
    _close(_v(log_expit(P(-800.0))), -800.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
