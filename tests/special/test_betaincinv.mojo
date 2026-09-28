"""Tests for `betaincinv` against SciPy's digits -- small shapes whose
roots sit a hundred decades below 1, both sides of NR's `a, b >= 1`
split, and a root that rounds to 1 -- its round trip through `betainc`,
its edges, and its derivative through `Dual`."""

from std.math import exp as _exp
from std.testing import TestSuite, assert_almost_equal

from numax import Dual, Plain, betainc, betaincinv, betaln

comptime P = Plain[DType.float64, 1]
comptime D = Dual[P]


def _inv(y: Float64, a: Float64, b: Float64) -> Float64:
    return betaincinv(P(y), P(a), P(b)).v[0]


def test_matches_scipy() raises:
    # scipy.special.betaincinv(a, b, y); the target comes first here.
    assert_almost_equal(
        _inv(1e-12, 0.1, 0.1), 8.86928065555035e-118, atol=0.0, rtol=1e-13
    )
    assert_almost_equal(
        _inv(0.9, 0.1, 0.1), 0.9999999113071999, atol=0.0, rtol=1e-15
    )
    assert_almost_equal(_inv(0.9999, 0.1, 0.1), 1.0, atol=0.0)
    assert_almost_equal(
        _inv(1e-12, 0.5, 0.5), 2.4674011002723395e-24, atol=0.0, rtol=1e-14
    )
    assert_almost_equal(
        _inv(0.9, 0.5, 0.5), 0.9755282581475768, atol=0.0, rtol=1e-15
    )
    assert_almost_equal(
        _inv(0.1, 2.0, 3.0), 0.14255931671003072, atol=0.0, rtol=1e-14
    )
    assert_almost_equal(
        _inv(1e-12, 5.0, 1.5), 0.003263018994535392, atol=0.0, rtol=1e-14
    )
    assert_almost_equal(
        _inv(0.9999, 20.0, 20.0), 0.7717120709518093, atol=0.0, rtol=1e-14
    )
    assert_almost_equal(
        _inv(0.5, 0.3, 50.0), 0.0014718251233310777, atol=0.0, rtol=5e-13
    )


def test_round_trip() raises:
    var shapes: List[Float64] = [0.2, 1.0, 3.5, 12.0]
    var ys: List[Float64] = [1e-9, 0.05, 0.5, 0.95]
    for a in shapes:
        for b in shapes:
            for y in ys:
                var x = _inv(y, a, b)
                # A root within `1e-4` of 1 is skipped: there an ulp of
                # `x` is a visible fraction of `1 - x`, and `b < 1` turns
                # it into an error in `y` no representable `x` can avoid
                # (`b = 0.2` at `y = 0.95` puts the root at `1 - 7e-8`).
                if x > 1.0 - 1e-4:
                    continue
                assert_almost_equal(
                    betainc(P(x), P(a), P(b)).v[0], y, atol=1e-15, rtol=1e-12
                )


def test_edges() raises:
    assert_almost_equal(_inv(0.0, 2.0, 3.0), 0.0, atol=0.0)
    assert_almost_equal(_inv(1.0, 2.0, 3.0), 1.0, atol=0.0)


def test_derivative_is_the_reciprocal_density() raises:
    var a = 2.5
    var b = 4.0
    var r = betaincinv(D(P(0.3), P(1.0)), D(P(a), P(0.0)), D(P(b), P(0.0)))
    var x = r.value.v[0]
    var log_density = (
        (a - 1.0) * P(x).ln().v[0]
        + (b - 1.0) * P(1.0 - x).ln().v[0]
        - betaln(P(a), P(b)).v[0]
    )
    assert_almost_equal(r.deriv.v[0], 1.0 / _exp(log_density), rtol=1e-10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
