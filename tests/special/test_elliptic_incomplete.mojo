"""Tests for `ellipkinc`, `ellipeinc`, `elliprf` and `elliprd` against
SciPy's digits -- amplitudes past `pi/2`, negative ones, a negative
parameter and `m = 1` -- and the identities between them: the amplitude
`pi/2` gives the complete integrals, `m = 0` gives `phi`, `m = 1` gives
`atanh(sin(phi))`, and `Dual`'s derivative in `phi` is the integrand."""

from std.math import sin as _sin, sqrt as _sqrt
from std.testing import TestSuite, assert_almost_equal

from numax import Dual, Plain, ellipeinc, ellipkinc, elliprd, elliprf

comptime P = Plain[DType.float64, 1]
comptime D = Dual[P]
comptime HALF_PI = 1.5707963267948966


def _f(phi: Float64, m: Float64) -> Float64:
    return ellipkinc(P(phi), P(m)).v[0]


def _e(phi: Float64, m: Float64) -> Float64:
    return ellipeinc(P(phi), P(m)).v[0]


def test_matches_scipy() raises:
    var ph: List[Float64] = [0.3, 1.0, HALF_PI, 2.5, -4.0, 10.0, 1.2, 1.0]
    var ms: List[Float64] = [0.5, 0.9, 0.99, 0.3, 0.7, -2.0, 1.0, 0.999999]
    var fk: List[Float64] = [
        0.30225466857501754,
        1.1885008994681585,
        3.695637362989875,
        2.77338117755762,
        -5.088975077596994,
        7.556383748300228,
        1.6736992495582428,
        1.2261907568130581,
    ]
    var fe: List[Float64] = [
        0.2977753719531601,
        0.8601912677655394,
        1.015993545025224,
        2.261502532164161,
        -3.273334011969113,
        13.736746862266909,
        0.9320390859672263,
        0.8414711771679343,
    ]
    for i in range(len(ph)):
        assert_almost_equal(_f(ph[i], ms[i]), fk[i], atol=0.0, rtol=2e-15)
        assert_almost_equal(_e(ph[i], ms[i]), fe[i], atol=0.0, rtol=2e-15)


def test_carlson_forms_match_scipy() raises:
    assert_almost_equal(
        elliprf(P(1.0), P(2.0), P(0.0)).v[0], 1.3110287771460598, rtol=2e-15
    )
    assert_almost_equal(
        elliprd(P(0.0), P(2.0), P(1.0)).v[0], 1.7972103521033884, rtol=2e-15
    )
    # Arguments three hundred decades apart, which the duplication count
    # is sized for.
    assert_almost_equal(
        elliprf(P(1e-300), P(1.0), P(1.0)).v[0], HALF_PI, rtol=2e-15
    )
    assert_almost_equal(
        elliprd(P(1e-300), P(1e-300), P(1.0)).v[0],
        1035.2427333890003,
        rtol=2e-15,
    )


def test_special_cases() raises:
    # m = 0: the integrand is 1.
    assert_almost_equal(_f(0.8, 0.0), 0.8, rtol=1e-15)
    assert_almost_equal(_e(0.8, 0.0), 0.8, rtol=1e-15)
    # m = 1 below pi/2: F = atanh(sin(phi)), E = sin(phi).
    var s = _sin(0.9)
    assert_almost_equal(
        _f(0.9, 1.0), 0.5 * P((1.0 + s) / (1.0 - s)).ln().v[0], rtol=1e-14
    )
    assert_almost_equal(_e(0.9, 1.0), s, rtol=1e-15)
    # Periodicity: one more half-turn adds twice the complete integral.
    assert_almost_equal(
        _f(0.7 + 3.141592653589793, 0.4) - _f(0.7, 0.4),
        2.0 * _f(HALF_PI, 0.4),
        rtol=1e-14,
    )


def test_derivative_in_phi_is_the_integrand() raises:
    var m = 0.6
    var phi = 1.1
    var s = _sin(phi)
    var root = _sqrt(1.0 - m * s * s)
    var f = ellipkinc(D(P(phi), P(1.0)), D(P(m), P(0.0)))
    assert_almost_equal(f.deriv.v[0], 1.0 / root, rtol=1e-14)
    var e = ellipeinc(D(P(phi), P(1.0)), D(P(m), P(0.0)))
    assert_almost_equal(e.deriv.v[0], root, rtol=1e-14)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
