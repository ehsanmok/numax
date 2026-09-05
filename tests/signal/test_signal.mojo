"""Tests for `numax.signal`.

Convolution and correlation are checked against NumPy's exact outputs on
small inputs -- the offset conventions are the part that goes wrong, and
they only show up on asymmetric data. The windows are checked against their
closed forms and against the properties that distinguish them (endpoint
value, symmetry, coefficient sum).
"""

from std.collections import Array
from std.math import cos
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Complex, Dual, FloatLike, Plain
from numax.fft import circular_convolve, fft, ifft
from numax.signal import (
    firwin,
    lfilter,
    apply_window,
    blackman,
    convolve,
    same,
    correlate,
    hamming,
    hann,
)

comptime dtype = DType.float64
comptime P = Plain[dtype]
comptime D = Dual[P]
comptime C = Complex[P]


def pv(x: Float64) -> P:
    return P.constant(x)


def s(x: P) -> Float64:
    return Float64(x.v)


def filled[n: Int](values: List[Float64]) -> Array[P, n]:
    var out = Array[P, n](fill=pv(0.0))
    for i in range(n):
        out[i] = pv(values[i])
    return out^


# ------------------------------------------------------------------
# convolve
# ------------------------------------------------------------------


def test_convolve_matches_numpy_on_an_asymmetric_pair() raises:
    # numpy.convolve([1, 2, 3], [4, 5]) == [4, 13, 22, 15]
    var a = filled[3]([1.0, 2.0, 3.0])
    var b = filled[2]([4.0, 5.0])
    var out = convolve[P, 3, 2](a, b)
    var expected = [4.0, 13.0, 22.0, 15.0]
    for i in range(4):
        assert_almost_equal(s(out[i]), expected[i])


def test_convolve_with_a_delta_is_the_identity() raises:
    var a = filled[4]([1.0, -2.0, 3.5, 0.25])
    var delta = filled[1]([1.0])
    var out = convolve[P, 4, 1](a, delta)
    for i in range(4):
        assert_almost_equal(s(out[i]), s(a[i]))


def test_convolve_with_a_shifted_delta_shifts_the_signal() raises:
    # [0, 0, 1] convolved in delays by two samples.
    var a = filled[3]([1.0, 2.0, 3.0])
    var delayed = filled[3]([0.0, 0.0, 1.0])
    var out = convolve[P, 3, 3](a, delayed)
    assert_almost_equal(s(out[0]), 0.0)
    assert_almost_equal(s(out[1]), 0.0)
    assert_almost_equal(s(out[2]), 1.0)
    assert_almost_equal(s(out[3]), 2.0)
    assert_almost_equal(s(out[4]), 3.0)


def test_convolve_is_commutative() raises:
    var a = filled[3]([1.0, 2.0, 3.0])
    var b = filled[2]([4.0, 5.0])
    var forward = convolve[P, 3, 2](a, b)
    var reversed_order = convolve[P, 2, 3](b, a)
    for i in range(4):
        assert_almost_equal(s(forward[i]), s(reversed_order[i]))


def test_convolve_sums_to_the_product_of_the_sums() raises:
    # sum(conv(a, b)) == sum(a) * sum(b), which catches a dropped term
    # anywhere in the overlap bookkeeping.
    var a = filled[4]([1.0, 2.0, 3.0, 4.0])
    var b = filled[3]([2.0, -1.0, 0.5])
    var out = convolve[P, 4, 3](a, b)
    var total = 0.0
    for i in range(6):
        total += s(out[i])
    assert_almost_equal(total, 10.0 * 1.5)


def test_convolve_mode_same_takes_the_central_window() raises:
    # numpy.convolve([1, 2, 3], [4, 5], "same") == [4, 13, 22]
    var a = filled[3]([1.0, 2.0, 3.0])
    var b = filled[2]([4.0, 5.0])
    var out = convolve[P, 3, 2, same](a, b)
    var expected = [4.0, 13.0, 22.0]
    for i in range(3):
        assert_almost_equal(s(out[i]), expected[i])


def test_convolve_mode_same_centres_an_odd_kernel() raises:
    # numpy.convolve([1, 2, 3, 4], [1, 1, 1], "same") == [3, 6, 9, 7]
    var a = filled[4]([1.0, 2.0, 3.0, 4.0])
    var kernel = filled[3]([1.0, 1.0, 1.0])
    var out = convolve[P, 4, 3, same](a, kernel)
    var expected = [3.0, 6.0, 9.0, 7.0]
    for i in range(4):
        assert_almost_equal(s(out[i]), expected[i])


# ------------------------------------------------------------------
# correlate
# ------------------------------------------------------------------


def test_correlate_matches_numpy_on_an_asymmetric_pair() raises:
    # numpy.correlate([1, 2, 3], [4, 5], "full") == [5, 14, 23, 12]
    var a = filled[3]([1.0, 2.0, 3.0])
    var b = filled[2]([4.0, 5.0])
    var out = correlate[P, 3, 2](a, b)
    var expected = [5.0, 14.0, 23.0, 12.0]
    for i in range(4):
        assert_almost_equal(s(out[i]), expected[i])


def test_correlate_equals_convolve_with_a_reversed_kernel() raises:
    # The identity the implementation is built on, checked against an
    # explicit reversal rather than assumed.
    var a = filled[4]([1.0, -2.0, 3.0, 0.5])
    var b = filled[3]([2.0, 1.0, -1.0])
    var b_reversed = filled[3]([-1.0, 1.0, 2.0])

    var correlated = correlate[P, 4, 3](a, b)
    var convolved = convolve[P, 4, 3](a, b_reversed)
    for i in range(6):
        assert_almost_equal(s(correlated[i]), s(convolved[i]))


def test_autocorrelation_peaks_at_zero_lag() raises:
    # A signal correlates with itself most strongly when aligned, which
    # for the "full" layout is index k - 1 = m - 1.
    var a = filled[5]([1.0, 2.0, -1.0, 0.5, 3.0])
    var out = correlate[P, 5, 5](a, a)
    var zero_lag = s(out[4])
    for i in range(9):
        if i != 4:
            assert_true(s(out[i]) <= zero_lag)


def test_autocorrelation_at_zero_lag_is_the_energy() raises:
    var a = filled[4]([1.0, 2.0, -1.0, 0.5])
    var out = correlate[P, 4, 4](a, a)
    var energy = 0.0
    for i in range(4):
        energy += s(a[i]) * s(a[i])
    assert_almost_equal(s(out[3]), energy)


# ------------------------------------------------------------------
# Windows
# ------------------------------------------------------------------


def test_hann_matches_its_closed_form() raises:
    comptime n = 8
    var w = hann[P, n]()
    for i in range(n):
        var theta = 2.0 * 3.141592653589793 * Float64(i) / Float64(n - 1)
        assert_almost_equal(s(w[i]), 0.5 - 0.5 * cos(theta))


def test_hann_endpoints_are_exactly_zero() raises:
    comptime n = 16
    var w = hann[P, n]()
    assert_almost_equal(s(w[0]), 0.0, atol=1e-15)
    assert_almost_equal(s(w[n - 1]), 0.0, atol=1e-15)


def test_hann_is_symmetric() raises:
    comptime n = 9
    var w = hann[P, n]()
    for i in range(n):
        assert_almost_equal(s(w[i]), s(w[n - 1 - i]), atol=1e-15)


def test_hann_peaks_at_one_for_odd_length() raises:
    # An odd-length symmetric window has a sample exactly at the centre.
    comptime n = 9
    var w = hann[P, n]()
    assert_almost_equal(s(w[n // 2]), 1.0, atol=1e-15)


def test_hamming_endpoints_are_the_pedestal() raises:
    # 0.54 - 0.46 = 0.08, the feature that distinguishes Hamming from Hann.
    comptime n = 16
    var w = hamming[P, n]()
    assert_almost_equal(s(w[0]), 0.08, atol=1e-14)
    assert_almost_equal(s(w[n - 1]), 0.08, atol=1e-14)


def test_hamming_is_symmetric() raises:
    comptime n = 12
    var w = hamming[P, n]()
    for i in range(n):
        assert_almost_equal(s(w[i]), s(w[n - 1 - i]), atol=1e-15)


def test_blackman_endpoints_are_zero() raises:
    # 0.42 - 0.5 + 0.08 = 0 exactly, in real arithmetic.
    comptime n = 16
    var w = blackman[P, n]()
    assert_almost_equal(s(w[0]), 0.0, atol=1e-15)
    assert_almost_equal(s(w[n - 1]), 0.0, atol=1e-15)


def test_blackman_is_narrower_at_the_shoulders_than_hann() raises:
    # Blackman's extra cosine term pulls the shoulders down; that is what
    # buys its lower sidelobes.
    comptime n = 32
    var b = blackman[P, n]()
    var h = hann[P, n]()
    assert_true(s(b[n // 4]) < s(h[n // 4]))


def test_apply_window_multiplies_elementwise() raises:
    comptime n = 8
    var x = Array[P, n](fill=pv(2.0))
    var w = hann[P, n]()
    var windowed = apply_window[P, n](x, w)
    for i in range(n):
        assert_almost_equal(s(windowed[i]), 2.0 * s(w[i]))


def test_windowing_suppresses_spectral_leakage() raises:
    """The reason windows exist, as a measurement.

    A tone whose frequency sits *between* two bins leaks across the whole
    spectrum when transformed rectangularly, because the finite record
    looks like a discontinuous signal. A Hann window tapers the ends to
    zero and confines the leakage to the neighbouring bins.
    """
    comptime log2n = 5
    comptime n = 1 << log2n
    # 4.5 cycles across the record: maximally between bins.
    comptime cycles = 4.5

    var rectangular = Array[C, n](fill=C.constant(0.0))
    var windowed = Array[C, n](fill=C.constant(0.0))
    var w = hann[P, n]()
    for i in range(n):
        var sample = cos(
            2.0 * 3.141592653589793 * cycles * Float64(i) / Float64(n)
        )
        rectangular[i] = C(pv(sample), pv(0.0))
        windowed[i] = C(pv(sample) * w[i], pv(0.0))

    var rect_spectrum = fft[P, log2n](rectangular)
    var hann_spectrum = fft[P, log2n](windowed)

    # Energy far from the tone (more than 3 bins away) should be much
    # smaller once windowed.
    var rect_far = 0.0
    var hann_far = 0.0
    for k in range(n // 2 + 1):
        if abs(Float64(k) - cycles) > 3.0:
            rect_far += (
                Float64(rect_spectrum[k].re.v) ** 2
                + Float64(rect_spectrum[k].im.v) ** 2
            )
            hann_far += (
                Float64(hann_spectrum[k].re.v) ** 2
                + Float64(hann_spectrum[k].im.v) ** 2
            )
    assert_true(hann_far < rect_far / 10.0)


# ------------------------------------------------------------------
# The composable-type spine still applies
# ------------------------------------------------------------------


def test_convolve_differentiates_at_dual() raises:
    """Convolution is bilinear, so `d/da[0] conv(a, b)[0] == b[0]`. Being
    `FloatLike`-generic, that comes out of `Dual` with nothing added."""
    var a = Array[D, 3](fill=D.constant(0.0))
    a[0] = D(P.constant(1.0), P.one())
    a[1] = D.constant(2.0)
    a[2] = D.constant(3.0)
    var b = Array[D, 2](fill=D.constant(0.0))
    b[0] = D.constant(4.0)
    b[1] = D.constant(5.0)

    var out = convolve[D, 3, 2](a, b)
    # out[0] = a[0]*b[0], so d/da[0] = b[0] = 4.
    assert_almost_equal(Float64(out[0].deriv.v), 4.0)
    # out[1] = a[0]*b[1] + a[1]*b[0], so d/da[0] = b[1] = 5.
    assert_almost_equal(Float64(out[1].deriv.v), 5.0)


def test_convolve_lanes_are_independent() raises:
    """Two SIMD lanes convolving different signals must not interact --
    the property that makes this usable inside a vectorized kernel."""
    comptime W = Plain[dtype, 2]
    var a = Array[W, 2](fill=W.constant(0.0))
    a[0] = W(SIMD[dtype, 2](1.0, 10.0))
    a[1] = W(SIMD[dtype, 2](2.0, 20.0))
    var b = Array[W, 2](fill=W.constant(0.0))
    b[0] = W(SIMD[dtype, 2](3.0, 30.0))
    b[1] = W(SIMD[dtype, 2](4.0, 40.0))

    var out = convolve[W, 2, 2](a, b)
    # Lane 0: [1,2]*[3,4] = [3, 10, 8]. Lane 1: [10,20]*[30,40] =
    # [300, 1000, 800].
    assert_almost_equal(out[0].v[0], 3.0)
    assert_almost_equal(out[1].v[0], 10.0)
    assert_almost_equal(out[2].v[0], 8.0)
    assert_almost_equal(out[0].v[1], 300.0)
    assert_almost_equal(out[1].v[1], 1000.0)
    assert_almost_equal(out[2].v[1], 800.0)


def test_linear_convolution_matches_zero_padded_circular() raises:
    """`convolve` and `numax.fft.circular_convolve` are the same operation
    once the wraparound is padded away. Checking that pins the offset
    convention of both against each other."""
    comptime log2n = 3
    comptime n = 1 << log2n

    var a = filled[3]([1.0, 2.0, 3.0])
    var b = filled[2]([4.0, 5.0])
    var direct = convolve[P, 3, 2](a, b)

    # Zero-pad both to n = 8 >= 3 + 2 - 1, so nothing wraps.
    var a_padded = Array[C, n](fill=C.constant(0.0))
    var b_padded = Array[C, n](fill=C.constant(0.0))
    for i in range(3):
        a_padded[i] = C(a[i].copy(), pv(0.0))
    for i in range(2):
        b_padded[i] = C(b[i].copy(), pv(0.0))

    var via_fft = circular_convolve[P, log2n](a_padded, b_padded)
    for i in range(4):
        assert_almost_equal(s(direct[i]), Float64(via_fft[i].re.v), atol=1e-10)


def test_lfilter_with_no_recursion_is_a_convolution() raises:
    # `na == 1` makes the difference equation depend on the input alone,
    # so it must agree with the truncated linear convolution.
    var x = filled[6]([1.0, -2.0, 3.0, 0.5, -1.5, 2.0])
    var b = filled[3]([0.25, 0.5, 0.25])
    var a = filled[1]([1.0])

    var filtered = lfilter[P, 6, 3, 1](b, a, x)
    var expected = convolve[P, 6, 3](x, b)
    for i in range(6):
        assert_almost_equal(s(filtered[i]), s(expected[i]), atol=1e-12)


def test_lfilter_runs_a_recursion_no_convolution_can() raises:
    # One-pole lowpass: y[i] = 0.5*x[i] + 0.5*y[i-1]. With a unit step in,
    # the output approaches 1 as 1 - 0.5**(i+1), which never terminates --
    # exactly what a finite kernel cannot produce.
    var x = filled[6]([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    var b = filled[1]([0.5])
    var a = filled[2]([1.0, -0.5])

    var filtered = lfilter[P, 6, 1, 2](b, a, x)
    for i in range(6):
        var expected = 1.0 - 0.5 ** Float64(i + 1)
        assert_almost_equal(s(filtered[i]), expected, atol=1e-12)


def test_lfilter_divides_by_the_leading_coefficient() raises:
    var x = filled[3]([1.0, 2.0, 3.0])
    var b = filled[1]([1.0])
    var scaled = lfilter[P, 3, 1, 1](b, filled[1]([2.0]), x)
    for i in range(3):
        assert_almost_equal(s(scaled[i]), s(x[i]) / 2.0, atol=1e-12)


def test_firwin_passes_a_constant_unchanged() raises:
    # Unit DC gain is what the normalization is for: the taps sum to one,
    # so a constant signal comes through at its own level.
    var taps = firwin[P, 21](0.25)
    var total = 0.0
    for i in range(21):
        total += s(taps[i])
    assert_almost_equal(total, 1.0, atol=1e-12)


def test_firwin_is_symmetric_so_the_phase_is_linear() raises:
    var taps = firwin[P, 21](0.3)
    for i in range(21):
        assert_almost_equal(s(taps[i]), s(taps[20 - i]), atol=1e-12)


def test_firwin_attenuates_above_its_cutoff() raises:
    # Run a slow and a fast sinusoid through the filter and compare what
    # survives. The cutoff is a quarter of Nyquist, so the second is well
    # inside the stopband.
    var taps = firwin[P, 31](0.25)

    var slow = Array[P, 64](fill=pv(0.0))
    var fast = Array[P, 64](fill=pv(0.0))
    for i in range(64):
        slow[i] = pv(cos(2.0 * 3.141592653589793 * 0.03 * Float64(i)))
        fast[i] = pv(cos(2.0 * 3.141592653589793 * 0.4 * Float64(i)))

    var ones = filled[1]([1.0])
    var slow_out = lfilter[P, 64, 31, 1](taps, ones, slow)
    var fast_out = lfilter[P, 64, 31, 1](taps, ones, fast)

    # Past the filter's own length the transient is over.
    var slow_peak = 0.0
    var fast_peak = 0.0
    for i in range(31, 64):
        slow_peak = max(slow_peak, abs(s(slow_out[i])))
        fast_peak = max(fast_peak, abs(s(fast_out[i])))

    assert_true(slow_peak > 0.9)
    assert_true(fast_peak < 0.01)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
