from std.collections import Array
from std.math import cos, sin
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Complex, Dual, FloatLike, Plain
from numax.fft import (
    circular_convolve,
    fft,
    fft2,
    fftfreq,
    fftshift,
    ifft,
    ifft2,
    ifftshift,
    irfft,
    rfft,
    rfftfreq,
)

comptime dtype = DType.float64
comptime P = Plain[dtype, 1]
comptime C = Complex[P]
comptime TWO_PI = 6.283185307179586


def _real_signal[
    log2n: Int
](samples: List[Float64],) -> Array[C, 1 << log2n]:
    comptime n = 1 << log2n
    var out = Array[C, n](fill=C.constant(0.0))
    for i in range(n):
        out[i] = C(P.constant(samples[i]), P.constant(0.0))
    return out^


def _dft_reference(samples: List[Float64], k: Int) -> List[Float64]:
    """One bin of the naive `O(n^2)` DFT, in host `Float64`.

    Deliberately computed outside `numax` -- comparing the FFT against a
    from-scratch reference is the point, so it can't share the FFT's own
    twiddles. Returns `[re, im]`.
    """
    var n = len(samples)
    var re = Float64(0)
    var im = Float64(0)
    for j in range(n):
        var angle = -TWO_PI * Float64(j) * Float64(k) / Float64(n)
        re += samples[j] * cos(angle)
        im += samples[j] * sin(angle)
    return [re, im]


def test_fft_matches_a_naive_dft() raises:
    comptime log2n = 4
    comptime n = 1 << log2n
    var samples = List[Float64]()
    for i in range(n):
        samples.append(cos(0.7 * Float64(i)) + 0.3 * Float64(i % 5))

    var spectrum = fft[P, log2n](_real_signal[log2n](samples))
    for k in range(n):
        var want = _dft_reference(samples, k)
        assert_almost_equal(spectrum[k].re.v[0], want[0], atol=1e-11)
        assert_almost_equal(spectrum[k].im.v[0], want[1], atol=1e-11)


def test_fft_of_a_constant_is_a_single_spike() raises:
    # A constant signal has all its energy at DC: X[0] = n*c, X[k>0] = 0.
    comptime log2n = 5
    comptime n = 1 << log2n
    var xs = Array[C, n](fill=C(P.constant(2.5), P.constant(0.0)))
    var spectrum = fft[P, log2n](xs)

    assert_almost_equal(spectrum[0].re.v[0], 2.5 * Float64(n), atol=1e-12)
    assert_almost_equal(spectrum[0].im.v[0], 0.0, atol=1e-12)
    for k in range(1, n):
        assert_almost_equal(spectrum[k].re.v[0], 0.0, atol=1e-12)
        assert_almost_equal(spectrum[k].im.v[0], 0.0, atol=1e-12)


def test_fft_of_a_pure_tone_lands_in_two_bins() raises:
    # cos(2*pi*m*j/n) = (exp(+i...) + exp(-i...))/2, so a real cosine at an
    # exact bin frequency puts n/2 at bin m and n/2 at bin n-m, nothing
    # anywhere else. This is the check that catches an off-by-one in the
    # bit-reversal permutation, which a constant signal can't see.
    comptime log2n = 4
    comptime n = 1 << log2n
    comptime m = 3
    var samples = List[Float64]()
    for j in range(n):
        samples.append(cos(TWO_PI * Float64(m) * Float64(j) / Float64(n)))

    var spectrum = fft[P, log2n](_real_signal[log2n](samples))
    for k in range(n):
        var expected = 0.0
        if k == m or k == n - m:
            expected = Float64(n) / 2.0
        assert_almost_equal(spectrum[k].re.v[0], expected, atol=1e-11)
        assert_almost_equal(spectrum[k].im.v[0], 0.0, atol=1e-11)


def test_ifft_inverts_fft() raises:
    comptime log2n = 6
    comptime n = 1 << log2n
    var xs = Array[C, n](fill=C.constant(0.0))
    for i in range(n):
        xs[i] = C(
            P.constant(sin(0.31 * Float64(i))),
            P.constant(cos(0.17 * Float64(i)) * 0.5),
        )

    var round_trip = ifft[P, log2n](fft[P, log2n](xs))
    for i in range(n):
        assert_almost_equal(round_trip[i].re.v[0], xs[i].re.v[0], atol=1e-12)
        assert_almost_equal(round_trip[i].im.v[0], xs[i].im.v[0], atol=1e-12)


def test_fft_is_linear() raises:
    comptime log2n = 4
    comptime n = 1 << log2n
    var a = Array[C, n](fill=C.constant(0.0))
    var b = Array[C, n](fill=C.constant(0.0))
    var sum_input = Array[C, n](fill=C.constant(0.0))
    for i in range(n):
        a[i] = C(P.constant(Float64(i) * 0.1), P.constant(0.0))
        b[i] = C(P.constant(0.0), P.constant(sin(Float64(i))))
        sum_input[i] = a[i] + b[i]

    var fa = fft[P, log2n](a)
    var fb = fft[P, log2n](b)
    var fsum = fft[P, log2n](sum_input)
    for k in range(n):
        assert_almost_equal(
            fsum[k].re.v[0], (fa[k] + fb[k]).re.v[0], atol=1e-12
        )
        assert_almost_equal(
            fsum[k].im.v[0], (fa[k] + fb[k]).im.v[0], atol=1e-12
        )


def test_parseval_identity_holds() raises:
    # sum |x[j]|^2 = (1/n) * sum |X[k]|^2. An energy identity checks every
    # bin at once, which a spot check of a few bins doesn't.
    comptime log2n = 5
    comptime n = 1 << log2n
    var xs = Array[C, n](fill=C.constant(0.0))
    var time_energy = Float64(0)
    for i in range(n):
        var re = cos(0.4 * Float64(i)) + 0.2
        var im = sin(0.9 * Float64(i))
        xs[i] = C(P.constant(re), P.constant(im))
        time_energy += re * re + im * im

    var spectrum = fft[P, log2n](xs)
    var freq_energy = Float64(0)
    for k in range(n):
        freq_energy += (
            spectrum[k].re.v[0] * spectrum[k].re.v[0]
            + spectrum[k].im.v[0] * spectrum[k].im.v[0]
        )
    assert_almost_equal(freq_energy / Float64(n), time_energy, atol=1e-10)


def test_real_input_spectrum_is_conjugate_symmetric() raises:
    comptime log2n = 4
    comptime n = 1 << log2n
    var samples = List[Float64]()
    for i in range(n):
        samples.append(Float64((i * 7) % 11) - 5.0)

    var spectrum = fft[P, log2n](_real_signal[log2n](samples))
    for k in range(1, n // 2):
        assert_almost_equal(
            spectrum[k].re.v[0], spectrum[n - k].re.v[0], atol=1e-12
        )
        assert_almost_equal(
            spectrum[k].im.v[0], -spectrum[n - k].im.v[0], atol=1e-12
        )


def test_circular_convolution_matches_the_direct_sum() raises:
    comptime log2n = 4
    comptime n = 1 << log2n
    var av = List[Float64]()
    var bv = List[Float64]()
    for i in range(n):
        av.append(Float64((i * 3) % 5) - 2.0)
        bv.append(Float64((i * 5) % 7) - 3.0)

    var got = circular_convolve[P, log2n](
        _real_signal[log2n](av), _real_signal[log2n](bv)
    )
    for i in range(n):
        var want = Float64(0)
        for j in range(n):
            want += av[j] * bv[(i - j + n) % n]
        assert_almost_equal(got[i].re.v[0], want, atol=1e-10)
        assert_almost_equal(got[i].im.v[0], 0.0, atol=1e-10)


def test_fft_differentiates_through_dual() raises:
    # Seed one sample's real part with a derivative. Since X[k] is linear in
    # the input, dX[k]/dx[j] is exactly exp(-2*pi*i*j*k/n) -- so the
    # derivative of the whole transform is a known closed form, and any
    # mishandled copy or dropped term in the butterfly would show up.
    comptime log2n = 3
    comptime n = 1 << log2n
    comptime D = Dual[P]
    comptime CD = Complex[D]
    comptime seeded_index = 2

    var xs = Array[CD, n](fill=CD.constant(0.0))
    for i in range(n):
        var deriv = 1.0 if i == seeded_index else 0.0
        xs[i] = CD(
            D(P.constant(Float64(i) + 1.0), P.constant(deriv)),
            D.constant(0.0),
        )

    var spectrum = fft[D, log2n](xs)
    for k in range(n):
        var angle = -TWO_PI * Float64(seeded_index) * Float64(k) / Float64(n)
        assert_almost_equal(spectrum[k].re.deriv.v[0], cos(angle), atol=1e-12)
        assert_almost_equal(spectrum[k].im.deriv.v[0], sin(angle), atol=1e-12)


def test_lanes_are_independent() raises:
    comptime log2n = 3
    comptime n = 1 << log2n
    comptime PW = Plain[dtype, 4]
    comptime CW = Complex[PW]

    var xs = Array[CW, n](fill=CW.constant(0.0))
    for i in range(n):
        var lanes = SIMD[dtype, 4](0)
        for lane in range(4):
            lanes[lane] = Float64(i) * (Float64(lane) + 1.0)
        xs[i] = CW(PW(lanes), PW.constant(0.0))

    var spectrum = fft[PW, log2n](xs)
    # DC bin of a ramp scaled by (lane+1): (lane+1) * sum(0..n-1).
    var ramp_sum = Float64(n * (n - 1)) / 2.0
    for lane in range(4):
        assert_almost_equal(
            spectrum[0].re.v[lane],
            ramp_sum * (Float64(lane) + 1.0),
            atol=1e-11,
        )


def test_single_point_transform_is_the_identity() raises:
    # log2n = 0 means n = 1: no stages run, and the empty twiddle table has
    # to be constructible anyway. An edge case worth pinning, since the
    # butterfly loop and the table size both degenerate here.
    var xs = Array[C, 1](fill=C(P.constant(3.0), P.constant(-1.0)))
    var spectrum = fft[P, 0](xs)
    assert_almost_equal(spectrum[0].re.v[0], 3.0, atol=1e-12)
    assert_almost_equal(spectrum[0].im.v[0], -1.0, atol=1e-12)


def test_two_point_transform_is_sum_and_difference() raises:
    var xs = Array[C, 2](fill=C.constant(0.0))
    xs[0] = C(P.constant(5.0), P.constant(0.0))
    xs[1] = C(P.constant(2.0), P.constant(0.0))
    var spectrum = fft[P, 1](xs)
    assert_almost_equal(spectrum[0].re.v[0], 7.0, atol=1e-12)
    assert_almost_equal(spectrum[1].re.v[0], 3.0, atol=1e-12)


# ------------------------------------------------------------------
# Real-input transforms, frequency axes, and the 2-D transform
# ------------------------------------------------------------------


def test_rfft_returns_the_non_redundant_half() raises:
    comptime log2n = 3
    comptime n = 1 << log2n
    var x = Array[P, n](fill=P.constant(0.0))
    for i in range(n):
        x[i] = P.constant(Float64(i) * 0.5 - 1.0)

    var half = rfft[P, log2n](x)
    # n/2 + 1 bins: 0 through Nyquist inclusive.
    assert_true(half.__len__() == n // 2 + 1)


def test_rfft_matches_the_full_transform_on_its_first_half() raises:
    comptime log2n = 3
    comptime n = 1 << log2n
    var real = Array[P, n](fill=P.constant(0.0))
    var embedded = Array[C, n](fill=C.constant(0.0))
    for i in range(n):
        var v = Float64(i) * 0.5 - 1.0
        real[i] = P.constant(v)
        embedded[i] = C(P.constant(v), P.constant(0.0))

    var half = rfft[P, log2n](real)
    var full = fft[P, log2n](embedded)
    for k in range(n // 2 + 1):
        assert_almost_equal(Float64(half[k].re.v), Float64(full[k].re.v))
        assert_almost_equal(Float64(half[k].im.v), Float64(full[k].im.v))


def test_rfft_dc_bin_is_the_sum_of_the_samples() raises:
    comptime log2n = 3
    comptime n = 1 << log2n
    var x = Array[P, n](fill=P.constant(0.0))
    var expected = 0.0
    for i in range(n):
        var v = Float64(i) * 0.5 - 1.0
        x[i] = P.constant(v)
        expected += v

    var half = rfft[P, log2n](x)
    assert_almost_equal(Float64(half[0].re.v), expected)
    # A real input's DC bin has no imaginary part.
    assert_almost_equal(Float64(half[0].im.v), 0.0)


def test_irfft_inverts_rfft() raises:
    comptime log2n = 4
    comptime n = 1 << log2n
    var x = Array[P, n](fill=P.constant(0.0))
    for i in range(n):
        x[i] = P.constant(cos(TWO_PI * Float64(i) * 3.0 / Float64(n)))

    var recovered = irfft[P, log2n](rfft[P, log2n](x))
    for i in range(n):
        assert_almost_equal(Float64(recovered[i].v), Float64(x[i].v))


def test_rfft_of_a_pure_tone_puts_all_energy_in_one_bin() raises:
    comptime log2n = 4
    comptime n = 1 << log2n
    comptime bin = 3
    var x = Array[P, n](fill=P.constant(0.0))
    for i in range(n):
        x[i] = P.constant(cos(TWO_PI * Float64(bin) * Float64(i) / Float64(n)))

    var half = rfft[P, log2n](x)
    # A cosine at an exact bin frequency: n/2 in that bin, ~0 elsewhere.
    assert_almost_equal(Float64(half[bin].re.v), Float64(n) / 2.0, atol=1e-10)
    for k in range(n // 2 + 1):
        if k != bin:
            var magnitude = (
                Float64(half[k].re.v) ** 2 + Float64(half[k].im.v) ** 2
            ) ** 0.5
            assert_true(magnitude < 1e-10)


def test_fftfreq_matches_numpys_layout() raises:
    # numpy.fft.fftfreq(8, 0.1) is
    #   [0, 1.25, 2.5, 3.75, -5, -3.75, -2.5, -1.25]
    comptime log2n = 3
    var freqs = fftfreq[P, log2n](P.constant(0.1))
    var expected = [0.0, 1.25, 2.5, 3.75, -5.0, -3.75, -2.5, -1.25]
    for k in range(8):
        assert_almost_equal(Float64(freqs[k].v), expected[k], atol=1e-12)


def test_rfftfreq_is_all_non_negative() raises:
    # numpy.fft.rfftfreq(8, 0.1) is [0, 1.25, 2.5, 3.75, 5]
    comptime log2n = 3
    var freqs = rfftfreq[P, log2n](P.constant(0.1))
    var expected = [0.0, 1.25, 2.5, 3.75, 5.0]
    for k in range(5):
        assert_almost_equal(Float64(freqs[k].v), expected[k], atol=1e-12)


def test_fftshift_moves_dc_to_the_middle() raises:
    comptime log2n = 3
    comptime n = 1 << log2n
    var spectrum = Array[C, n](fill=C.constant(0.0))
    for k in range(n):
        spectrum[k] = C(P.constant(Float64(k)), P.constant(0.0))

    var shifted = fftshift[P, log2n](spectrum)
    # Bin 0 lands at index n/2.
    assert_almost_equal(Float64(shifted[n // 2].re.v), 0.0)
    # And the negative-frequency half comes first.
    assert_almost_equal(Float64(shifted[0].re.v), Float64(n // 2))


def test_fftshift_is_its_own_inverse_for_even_n() raises:
    comptime log2n = 3
    comptime n = 1 << log2n
    var spectrum = Array[C, n](fill=C.constant(0.0))
    for k in range(n):
        spectrum[k] = C(P.constant(Float64(k)), P.constant(-Float64(k)))

    var twice = fftshift[P, log2n](fftshift[P, log2n](spectrum))
    for k in range(n):
        assert_almost_equal(Float64(twice[k].re.v), Float64(k))
        assert_almost_equal(Float64(twice[k].im.v), -Float64(k))


def test_ifftshift_undoes_fftshift() raises:
    comptime log2n = 3
    comptime n = 1 << log2n
    var spectrum = Array[C, n](fill=C.constant(0.0))
    for k in range(n):
        spectrum[k] = C(P.constant(Float64(k)), P.constant(-Float64(k)))

    var restored = ifftshift[P, log2n](fftshift[P, log2n](spectrum))
    for k in range(n):
        assert_almost_equal(Float64(restored[k].re.v), Float64(k))
        assert_almost_equal(Float64(restored[k].im.v), -Float64(k))


def test_ifft2_inverts_fft2() raises:
    comptime log2n = 2
    comptime n = 1 << log2n
    var image = Array[C, n * n](fill=C.constant(0.0))
    for i in range(n * n):
        image[i] = C(P.constant(Float64(i % 5)), P.constant(Float64(i % 3)))

    var recovered = ifft2[P, log2n](fft2[P, log2n](image))
    for i in range(n * n):
        assert_almost_equal(Float64(recovered[i].re.v), Float64(image[i].re.v))
        assert_almost_equal(Float64(recovered[i].im.v), Float64(image[i].im.v))


def test_fft2_of_a_constant_image_is_a_single_spike() raises:
    # A constant image has all its energy at DC and nothing elsewhere --
    # the 2-D analogue of fft of a constant sequence.
    comptime log2n = 2
    comptime n = 1 << log2n
    var image = Array[C, n * n](fill=C(P.constant(2.0), P.constant(0.0)))

    var spectrum = fft2[P, log2n](image)
    assert_almost_equal(
        Float64(spectrum[0].re.v), 2.0 * Float64(n * n), atol=1e-10
    )
    for i in range(1, n * n):
        var magnitude = (
            Float64(spectrum[i].re.v) ** 2 + Float64(spectrum[i].im.v) ** 2
        ) ** 0.5
        assert_true(magnitude < 1e-10)


def test_fft2_separates_into_row_then_column_transforms() raises:
    # The 2-D DFT separates exactly, so transforming rows with `fft` and
    # then columns with `fft` by hand must reproduce `fft2`. This is the
    # property `fft2` is implemented from, checked independently.
    comptime log2n = 2
    comptime n = 1 << log2n
    var image = Array[C, n * n](fill=C.constant(0.0))
    for i in range(n * n):
        image[i] = C(P.constant(Float64(i) * 0.25), P.constant(0.0))

    var by_hand = Array[C, n * n](fill=C.constant(0.0))
    var row = Array[C, n](fill=C.constant(0.0))
    for i in range(n):
        for j in range(n):
            row[j] = image[i * n + j].copy()
        var transformed = fft[P, log2n](row)
        for j in range(n):
            by_hand[i * n + j] = transformed[j].copy()
    var column = Array[C, n](fill=C.constant(0.0))
    for j in range(n):
        for i in range(n):
            column[i] = by_hand[i * n + j].copy()
        var transformed = fft[P, log2n](column)
        for i in range(n):
            by_hand[i * n + j] = transformed[i].copy()

    var direct = fft2[P, log2n](image)
    for i in range(n * n):
        assert_almost_equal(Float64(direct[i].re.v), Float64(by_hand[i].re.v))
        assert_almost_equal(Float64(direct[i].im.v), Float64(by_hand[i].im.v))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
