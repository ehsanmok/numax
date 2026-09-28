"""Tests for `wofz` against SciPy's digits in all four quadrants, on both
axes and far out, and against its identities: `w(x) = exp(-x^2) + (2i /
sqrt(pi)) D(x)` on the real axis, `w(i y) = erfcx(y)`, the reflection
`w(-z) = 2 exp(-z^2) - w(z)`, and `w'(z) = -2 z w(z) + 2i / sqrt(pi)`
through `Dual`."""

from std.math import exp as _exp
from std.testing import TestSuite, assert_almost_equal

from numax import Dual, Plain, erfc, wofz
from numax.core.complex import Complex

comptime P = Plain[DType.float64, 1]
comptime C = Complex[P]
comptime D = Dual[P]
comptime CD = Complex[D]


def _w(x: Float64, y: Float64) -> Tuple[Float64, Float64]:
    var w = wofz(C(P(x), P(y)))
    return (w.re.v[0], w.im.v[0])


def _close(got: Tuple[Float64, Float64], re: Float64, im: Float64) raises:
    # Relative in modulus, the measure the module docstring states.
    var mod = (re * re + im * im) ** 0.5
    assert_almost_equal(got[0], re, atol=4e-15 * mod)
    assert_almost_equal(got[1], im, atol=4e-15 * mod)


def test_matches_scipy() raises:
    # scipy.special.wofz.
    _close(_w(1.0, 1.0), 0.30474420525691254, 0.2082189382028316)
    _close(_w(-2.5, 0.3), 0.038226506260685265, -0.24304200853097793)
    _close(_w(0.5, 10.0), 0.05600435223166482, 0.00277295478096162)
    _close(_w(1.0, -0.5), 0.15554114245433115, 1.1378372157816865)
    _close(_w(-3.0, -1.0), -0.06467357479385974, -0.17373084850174422)
    _close(_w(0.1, -2.0), 99.32072206313673, 42.11059502293406)
    _close(_w(200.0, 50.0), 0.0006637740414818436, 0.0026550336912172443)
    _close(_w(2.0, 1e-8), 0.018315641205991173, 0.3400262163334408)
    _close(_w(30.0, 0.0), 0.0, 0.01881678486866075)
    _close(_w(0.0, 0.0), 1.0, 0.0)


def test_axes() raises:
    # Real axis: Re w = exp(-x^2). Imaginary axis: w = erfcx(y), real.
    var real_axis = _w(1.3, 0.0)
    assert_almost_equal(real_axis[0], _exp(-1.69), rtol=2e-15)
    var up = _w(0.0, 0.8)
    assert_almost_equal(up[0], _exp(0.64) * erfc(P(0.8)).v[0], rtol=4e-15)
    assert_almost_equal(up[1], 0.0, atol=1e-16)


def test_reflection() raises:
    # w(-z) = 2 exp(-z^2) - w(z), through the upper-half-plane branch for
    # one side and the reflected branch for the other.
    var x = 0.7
    var y = 0.4
    var w = _w(x, y)
    var neg = _w(-x, -y)
    var ez = C(P(-(x * x - y * y)), P(-2.0 * x * y)).exp()
    assert_almost_equal(neg[0], 2.0 * ez.re.v[0] - w[0], rtol=1e-14)
    assert_almost_equal(neg[1], 2.0 * ez.im.v[0] - w[1], rtol=1e-14)


def test_derivative_through_dual() raises:
    # d/dx w(x + iy) = w'(z) = -2 z w + 2i / sqrt(pi).
    var x = 0.9
    var y = 0.6
    var z = CD(D(P(x), P(1.0)), D(P(y), P(0.0)))
    var w = wofz(z)
    var wv = _w(x, y)
    var want_re = -2.0 * (x * wv[0] - y * wv[1])
    var want_im = -2.0 * (x * wv[1] + y * wv[0]) + 1.1283791670955126
    assert_almost_equal(w.re.deriv.v[0], want_re, rtol=1e-13)
    assert_almost_equal(w.im.deriv.v[0], want_im, rtol=1e-13)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
