"""Tests for `numax.special.elliptic` against known closed-form values, a
from-scratch Gauss-AGM reference, and the standard derivative identities.
"""

from std.math import pi
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Dual, Plain, elliptic_e, elliptic_k

comptime dtype = DType.float64
comptime width = 1
comptime D = Dual[Plain[dtype, width]]


def v(x: Float64) -> Plain[dtype, width]:
    return Plain[dtype, width](SIMD[dtype, width](x))


def agm_reference(m: Float64) -> Tuple[Float64, Float64]:
    """A from-scratch Gauss-AGM computation of `K(m)`/`E(m)`, independent of
    `numax.special.elliptic`'s own A&S polynomial approximation.

    This is the reference rather than a `std.math` call because `std.math`
    has no elliptic integrals at all. The AGM iteration converges
    quadratically and is a completely different algorithm from the
    polynomial-plus-log approximation under test, so agreement between them
    is real evidence. It is also what caught and then recovered A&S
    17.3.36's misdigitized `b4` coefficient -- see `numax/special/elliptic.mojo`.
    """
    var a = Float64(1.0)
    var b = (Float64(1.0) - m) ** 0.5
    var c = m**0.5
    var two_pow = Float64(1.0)
    var sum_c2 = c * c
    # AGM converges quadratically, so a small fixed iteration count already
    # reaches full `float64` precision -- deliberately *not* iterating
    # further or breaking on a tiny `|c|` threshold: `a` and `b` become so
    # close after ~6-7 iterations that `a - b` loses almost all its
    # significant digits to cancellation, so `c` stagnates at a noise floor
    # rather than continuing to shrink, and `two_pow` (still doubling every
    # iteration) would amplify that noise into real error in `sum_c2` if
    # this ran anywhere near as long as a naive "keep going until `c` is
    # tiny" loop suggests (confirmed directly: 60 iterations here corrupts
    # `E(0.3)` by ~7.5e-5, while 12 matches `numax.elliptic_e` to ~1e-8).
    for _ in range(12):
        var a_next = (a + b) / 2.0
        var b_next = (a * b) ** 0.5
        var c_next = (a - b) / 2.0
        a = a_next
        b = b_next
        c = c_next
        two_pow *= 2.0
        sum_c2 += two_pow * c * c
    var k = Float64(pi) / (2.0 * a)
    var e = k * (1.0 - sum_c2 / 2.0)
    return (k, e)


def test_elliptic_k_e_at_zero_are_pi_over_two() raises:
    assert_almost_equal(
        elliptic_k(v(0.0)).v, SIMD[dtype, width](Float64(pi) / 2.0), atol=1e-9
    )
    assert_almost_equal(
        elliptic_e(v(0.0)).v, SIMD[dtype, width](Float64(pi) / 2.0), atol=1e-9
    )


def test_elliptic_e_at_one_is_one() raises:
    # E(1) = 1 exactly, despite the 0*(-infinity) indeterminate form in the
    # formula right there -- see `numax/special/elliptic.mojo`'s module docstring.
    assert_almost_equal(
        elliptic_e(v(1.0)).v, SIMD[dtype, width](1.0), atol=1e-9
    )


def test_elliptic_k_diverges_but_stays_finite_at_one() raises:
    # K(1) is a true mathematical singularity; `numax.special.elliptic` caps it at
    # a large-but-finite value (via the log-argument floor) rather than
    # producing an actual infinity or NaN.
    var k_at_one = elliptic_k(v(1.0)).v[0]
    assert_true(k_at_one > 15.0)
    assert_true(k_at_one < 1e6)


def test_elliptic_k_matches_agm_reference() raises:
    for m64 in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999]:
        var expected = agm_reference(m64)[0]
        assert_almost_equal(
            elliptic_k(v(m64)).v, SIMD[dtype, width](expected), atol=1e-7
        )


def test_elliptic_e_matches_agm_reference() raises:
    for m64 in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999]:
        var expected = agm_reference(m64)[1]
        assert_almost_equal(
            elliptic_e(v(m64)).v, SIMD[dtype, width](expected), atol=1e-7
        )


def test_elliptic_k_derivative_matches_closed_form() raises:
    # dK/dm = E(m)/(2*m*(1-m)) - K(m)/(2*m), for m != 0.
    var m64 = 0.5
    var x = D(v(m64), v(1))
    var k = elliptic_k(x)
    var e_ref = agm_reference(m64)[1]
    var k_ref = agm_reference(m64)[0]
    var expected = e_ref / (2.0 * m64 * (1.0 - m64)) - k_ref / (2.0 * m64)
    assert_almost_equal(k.deriv.v, SIMD[dtype, width](expected), atol=1e-6)


def test_elliptic_e_derivative_matches_closed_form() raises:
    # dE/dm = (E(m) - K(m)) / (2*m), for m != 0.
    var m64 = 0.5
    var x = D(v(m64), v(1))
    var e = elliptic_e(x)
    var e_ref = agm_reference(m64)[1]
    var k_ref = agm_reference(m64)[0]
    var expected = (e_ref - k_ref) / (2.0 * m64)
    assert_almost_equal(e.deriv.v, SIMD[dtype, width](expected), atol=1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
