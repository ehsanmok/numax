"""Tests for `numax.special.elliptic`'s complete integrals against known
closed-form values, SciPy's digits, a from-scratch Gauss-AGM reference,
and the standard derivative identities."""

from std.math import isinf, pi
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Dual, Plain, elliptic_e, elliptic_k

comptime dtype = DType.float64
comptime width = 1
comptime D = Dual[Plain[dtype, width]]


def pv(x: Float64) -> Plain[dtype, width]:
    return Plain[dtype, width].constant(x)


def agm_reference(m: Float64) -> Tuple[Float64, Float64]:
    """A from-scratch Gauss-AGM computation of `K(m)`/`E(m)`, independent of
    `numax.special.elliptic`'s Carlson forms.

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
    # A fixed count, deliberately not a `|c|` threshold: after ~6-7
    # iterations `a - b` has lost its digits to cancellation and `c`
    # stagnates at a noise floor, which `two_pow`'s doubling then amplifies
    # into `sum_c2`. Measured: 60 iterations corrupts `E(0.3)` by 7.5e-5
    # where 12 matches `numax.elliptic_e` to about 1e-8 -- the noise floor
    # of this reference's `E`, not of the function under test.
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
        elliptic_k(pv(0.0)).v, SIMD[dtype, width](Float64(pi) / 2.0), atol=1e-9
    )
    assert_almost_equal(
        elliptic_e(pv(0.0)).v, SIMD[dtype, width](Float64(pi) / 2.0), atol=1e-9
    )


def test_elliptic_e_at_one_is_one() raises:
    # E(1) = 1 exactly, despite the 0*(-infinity) indeterminate form in the
    # formula right there -- see `numax/special/elliptic.mojo`'s module docstring.
    assert_almost_equal(
        elliptic_e(pv(1.0)).v, SIMD[dtype, width](1.0), atol=1e-9
    )


def test_elliptic_k_is_infinite_at_one() raises:
    # K(1) is a true singularity, and SciPy returns `inf` there.
    assert_true(isinf(elliptic_k(pv(1.0)).v[0]))


def test_complete_integrals_match_scipy() raises:
    # scipy.special.ellipk / ellipe.
    var ms: List[Float64] = [0.5, 0.9, 0.999999, 1.0 - 1e-15, -5.0]
    var ks: List[Float64] = [
        1.8540746773013719,
        2.5780921133481733,
        8.294051463601061,
        18.656082357290334,
        0.9555039270640441,
    ]
    var es: List[Float64] = [
        1.3506438810476755,
        1.1047747327040733,
        1.0000038970261722,
        1.000000000000009,
        2.830198246345877,
    ]
    for i in range(len(ms)):
        assert_almost_equal(
            elliptic_k(pv(ms[i])).v[0], ks[i], atol=0.0, rtol=2e-15
        )
        # Below `1 - 1e-15` the `R_F - (m/3) R_D` difference cancels a
        # digit and a half, so `E` there is `5e-15`.
        assert_almost_equal(
            elliptic_e(pv(ms[i])).v[0], es[i], atol=0.0, rtol=6e-15
        )


def test_elliptic_k_matches_agm_reference() raises:
    for m64 in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999]:
        var expected = agm_reference(m64)[0]
        assert_almost_equal(
            elliptic_k(pv(m64)).v, SIMD[dtype, width](expected), atol=1e-13
        )


def test_elliptic_e_matches_agm_reference() raises:
    for m64 in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999]:
        var expected = agm_reference(m64)[1]
        assert_almost_equal(
            elliptic_e(pv(m64)).v, SIMD[dtype, width](expected), atol=1e-7
        )


def test_elliptic_k_derivative_matches_closed_form() raises:
    # dK/dm = E(m)/(2*m*(1-m)) - K(m)/(2*m), for m != 0.
    var m64 = 0.5
    var x = D(pv(m64), pv(1))
    var k = elliptic_k(x)
    var e_ref = 1.3506438810476755
    var k_ref = 1.8540746773013719
    var expected = e_ref / (2.0 * m64 * (1.0 - m64)) - k_ref / (2.0 * m64)
    assert_almost_equal(k.deriv.v, SIMD[dtype, width](expected), atol=1e-14)


def test_elliptic_e_derivative_matches_closed_form() raises:
    # dE/dm = (E(m) - K(m)) / (2*m), for m != 0.
    var m64 = 0.5
    var x = D(pv(m64), pv(1))
    var e = elliptic_e(x)
    var e_ref = 1.3506438810476755
    var k_ref = 1.8540746773013719
    var expected = (e_ref - k_ref) / (2.0 * m64)
    assert_almost_equal(e.deriv.v, SIMD[dtype, width](expected), atol=1e-14)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
