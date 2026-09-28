"""Tests for `buttord`, `cheb1ord`, `cheb2ord` and `ellipord` against
SciPy: lowpass, highpass, bandpass and bandstop specifications in Nyquist
units, and a lowpass given in hertz through `fs`. The bandstop cases run
SciPy's bounded Brent search for the passband edges, so they check its
transcription too.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from numax.signal import FilterOrder, buttord, cheb1ord, cheb2ord, ellipord


def _check(got: FilterOrder, order: Int, wn: Float64, wn_high: Float64) raises:
    assert_equal(got.order, order)
    assert_almost_equal(got.wn, wn, atol=1e-9)
    assert_almost_equal(got.wn_high, wn_high, atol=1e-9)


def test_order_estimators_match_scipy() raises:
    _check(buttord(0.2, 0.3, 3.0, 40.0), 11, 0.2000403906692605, 0.0)
    _check(buttord(0.3, 0.2, 1.0, 60.0), 17, 0.2898861317859975, 0.0)
    _check(
        buttord((0.2, 0.5), (0.1, 0.6), 3.0, 40.0),
        9,
        0.19997484768392573,
        0.5000427940030838,
    )
    _check(
        buttord((0.1, 0.6), (0.2, 0.5), 3.0, 40.0),
        9,
        0.14760893156979507,
        0.5999425870213491,
    )
    _check(buttord(100.0, 150.0, 2.0, 30.0, 1000.0), 9, 102.82109368678904, 0.0)
    _check(cheb1ord(0.2, 0.3, 3.0, 40.0), 6, 0.2, 0.0)
    _check(cheb1ord(0.3, 0.2, 1.0, 60.0), 9, 0.3, 0.0)
    _check(cheb1ord((0.2, 0.5), (0.1, 0.6), 3.0, 40.0), 5, 0.2, 0.5)
    _check(
        cheb1ord((0.1, 0.6), (0.2, 0.5), 3.0, 40.0),
        5,
        0.14758236056266263,
        0.5999987080915621,
    )
    _check(cheb1ord(100.0, 150.0, 2.0, 30.0, 1000.0), 5, 100.0, 0.0)
    _check(cheb2ord(0.2, 0.3, 3.0, 40.0), 6, 0.2745644373777229, 0.0)
    _check(cheb2ord(0.3, 0.2, 1.0, 60.0), 9, 0.21464673601510123, 0.0)
    _check(
        cheb2ord((0.2, 0.5), (0.1, 0.6), 3.0, 40.0),
        5,
        0.15201672851009923,
        0.5906581070614803,
    )
    _check(
        cheb2ord((0.1, 0.6), (0.2, 0.5), 3.0, 40.0),
        5,
        0.19578223886719018,
        0.5072373464396942,
    )
    _check(
        cheb2ord(100.0, 150.0, 2.0, 30.0, 1000.0), 5, 137.23704236506342, 0.0
    )
    _check(ellipord(0.2, 0.3, 3.0, 40.0), 4, 0.2, 0.0)
    _check(ellipord(0.3, 0.2, 1.0, 60.0), 6, 0.3, 0.0)
    _check(ellipord((0.2, 0.5), (0.1, 0.6), 3.0, 40.0), 4, 0.2, 0.5)
    _check(
        ellipord((0.1, 0.6), (0.2, 0.5), 3.0, 40.0),
        4,
        0.14758232794342988,
        0.5999987080915621,
    )
    _check(ellipord(100.0, 150.0, 2.0, 30.0, 1000.0), 3, 100.0, 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
