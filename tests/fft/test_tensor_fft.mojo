"""Tests for `numax.fft` over `Tensor`.

The load-bearing check is the last group: the `Tensor` tier is pinned
against `numax.fft.array`, which is separately tested against NumPy's exact
outputs. Two tiers computing the same transform by different routes -- a
register-resident `comptime` butterfly against `log2(n) + 1` device
launches over a bit-reversed buffer -- agreeing to `1e-12` is a much
stronger statement than either matching a table of constants alone.
"""

from std.math import cos as _cos, sin as _sin
from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext
from std.collections import Array

from numax.core.array import Static, copy, zeros
from numax.core.complex import Complex
from numax.core.plain import Plain
from numax.fft import (
    fft,
    fft2,
    fftfreq,
    fftshift,
    ifft,
    ifft2,
    ifftshift,
    irfft,
    next_fast_len,
    rfft,
    rfft2,
    rfftfreq,
)
from numax.fft.fft import _as_matrix, _bluestein
from numax.fft.array import fft as array_fft
from numax.fft.array import fft2 as array_fft2

comptime dtype = DType.float64
comptime P = Plain[dtype]


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _ramp[n: Int]() raises -> Static[dtype, n]:
    """`[1, 2, ..., n]`, the sequence NumPy's docs use for `fft`."""
    var ctx = _cpu()
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(Scalar[dtype](i + 1))
    return Static[dtype, n](ctx, values^)


def _from[n: Int](values: List[Float64]) raises -> Static[dtype, n]:
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype](values[i]))
    return Static[dtype, n](_cpu(), out^)


def _matrix[
    rows: Int, cols: Int
](values: List[Float64]) raises -> Static[dtype, rows, cols]:
    var out = List[Scalar[dtype]](capacity=rows * cols)
    for i in range(rows * cols):
        out.append(Scalar[dtype](values[i]))
    return Static[dtype, rows, cols](_cpu(), out^)


def _signal() -> List[Float64]:
    """Eight samples with no symmetry, so every bin is non-trivial."""
    return [1.0, 2.0, 0.5, -1.0, 3.0, 0.25, -2.0, 1.5]


def _image() -> List[Float64]:
    """`arange(16) * 0.5 - 3`, as a 4x4: the matrix NumPy's `fft2`
    reference values below were taken on."""
    var out = List[Float64]()
    for i in range(16):
        out.append(Float64(i) * 0.5 - 3.0)
    return out^


def test_fft_matches_numpy_on_the_ramp() raises:
    """`numpy.fft.fft([1..8])` -- `36` for the DC term and `-4` for every
    other real part, which is the closed form for an arithmetic ramp."""
    var ctx = _cpu()
    var real = _ramp[8]()
    var imag = zeros[dtype, 8](ctx)
    var out = fft((real^, imag^))
    var re = out[0].to_host()
    var im = out[1].to_host()

    assert_almost_equal(Float64(re[0]), 36.0, atol=1e-12)
    assert_almost_equal(Float64(im[0]), 0.0, atol=1e-12)
    for k in range(1, 8):
        assert_almost_equal(Float64(re[k]), -4.0, atol=1e-12)
    assert_almost_equal(Float64(im[1]), 9.65685424949238, atol=1e-12)
    assert_almost_equal(Float64(im[2]), 4.0, atol=1e-12)
    assert_almost_equal(Float64(im[3]), 1.6568542494923797, atol=1e-12)
    assert_almost_equal(Float64(im[4]), 0.0, atol=1e-12)
    # the spectrum of a real sequence is conjugate-symmetric
    assert_almost_equal(Float64(im[5]), -1.6568542494923797, atol=1e-12)
    assert_almost_equal(Float64(im[7]), -9.65685424949238, atol=1e-12)


def test_ifft_inverts_fft() raises:
    """`ifft(fft(x)) == x`, which is what the `1/n` and the conjugated
    twiddle table together have to get right."""
    var ctx = _cpu()
    var original = _ramp[16]().to_host()
    var real = _ramp[16]()
    var imag = zeros[dtype, 16](ctx)
    var back = ifft(fft((real^, imag^)))
    var re = back[0].to_host()
    var im = back[1].to_host()
    for i in range(16):
        assert_almost_equal(Float64(re[i]), Float64(original[i]), atol=1e-12)
        assert_almost_equal(Float64(im[i]), 0.0, atol=1e-12)


def test_fft_of_a_constant_is_a_single_spike() raises:
    """A constant sequence has all its energy at DC: `X[0] == n * c` and
    every other bin zero. The cleanest check that no twiddle is misindexed,
    since every off-DC bin is a cancelling sum."""
    var ctx = _cpu()
    var values = List[Scalar[dtype]](length=32, fill=Scalar[dtype](2.5))
    var real = Static[dtype, 32](ctx, values^)
    var imag = zeros[dtype, 32](ctx)
    var out = fft((real^, imag^))
    var re = out[0].to_host()
    var im = out[1].to_host()

    assert_almost_equal(Float64(re[0]), 80.0, atol=1e-12)
    for k in range(1, 32):
        assert_almost_equal(Float64(re[k]), 0.0, atol=1e-12)
        assert_almost_equal(Float64(im[k]), 0.0, atol=1e-12)


def test_parseval_holds() raises:
    """`sum |x|^2 == (1/n) sum |X|^2`. An energy identity catches a wrong
    normalization that a spot check on one bin can miss."""
    var ctx = _cpu()
    var original = _ramp[32]().to_host()
    var real = _ramp[32]()
    var imag = zeros[dtype, 32](ctx)
    var out = fft((real^, imag^))
    var re = out[0].to_host()
    var im = out[1].to_host()

    var time_energy = 0.0
    for i in range(32):
        time_energy += Float64(original[i]) * Float64(original[i])
    var freq_energy = 0.0
    for k in range(32):
        freq_energy += Float64(re[k]) * Float64(re[k]) + Float64(
            im[k]
        ) * Float64(im[k])
    assert_almost_equal(time_energy, freq_energy / 32.0, rtol=1e-12)


def test_rfft_is_the_half_spectrum_of_fft() raises:
    """`rfft` returns `X[0..n/2]` and drops the conjugate mirror, so it must
    agree with the full transform bin for bin on the half it keeps."""
    var ctx = _cpu()
    var real = _ramp[16]()
    var imag = zeros[dtype, 16](ctx)
    var full = fft((real^, imag^))
    var fre = full[0].to_host()
    var fim = full[1].to_host()

    var half = rfft(_ramp[16]())
    assert_equal(half[0].num_elements, 9)
    var hre = half[0].to_host()
    var him = half[1].to_host()
    for k in range(9):
        assert_almost_equal(Float64(hre[k]), Float64(fre[k]), atol=1e-12)
        assert_almost_equal(Float64(him[k]), Float64(fim[k]), atol=1e-12)


def test_fftfreq_matches_numpy() raises:
    """`numpy.fft.fftfreq(8)` -- the second half is negative, which is the
    convention `fftshift` exists to reorder."""
    var got = fftfreq[dtype, 8]().to_host()
    var want: List[Float64] = [
        0.0,
        0.125,
        0.25,
        0.375,
        -0.5,
        -0.375,
        -0.25,
        -0.125,
    ]
    for i in range(8):
        assert_almost_equal(Float64(got[i]), want[i], atol=1e-15)


def test_fftfreq_scales_by_the_sample_spacing() raises:
    """The grid is `k / (n * spacing)`, so halving the spacing doubles every
    frequency."""
    var unit = fftfreq[dtype, 8]().to_host()
    var dense = fftfreq[dtype, 8](0.5).to_host()
    for i in range(8):
        assert_almost_equal(
            Float64(dense[i]), 2.0 * Float64(unit[i]), atol=1e-15
        )


def test_rfftfreq_is_non_negative() raises:
    """`numpy.fft.rfftfreq(8)` -- `n/2 + 1 == 5` bins, none of them
    negative, which is the grid `rfft`'s half spectrum sits on."""
    var got = rfftfreq[dtype, 8]().to_host()
    assert_equal(len(got), 5)
    for i in range(5):
        assert_almost_equal(Float64(got[i]), Float64(i) / 8.0, atol=1e-15)


def test_the_two_tiers_agree() raises:
    """The `Tensor` tier against `numax.fft.array`, which is itself pinned
    to NumPy. Different algorithms -- a `comptime` butterfly over a register
    `Array` against `log2(n) + 1` launches over a bit-reversed device
    buffer -- so agreement here is a real cross-check rather than a
    tautology."""
    var ctx = _cpu()

    var packed = Array[Complex[P], 16](uninitialized=True)
    for i in range(16):
        packed[i] = Complex[P](P(Float64(i + 1)), P(0.0))
    var reference = array_fft[P, 4](packed)

    var real = _ramp[16]()
    var imag = zeros[dtype, 16](ctx)
    var out = fft((real^, imag^))
    var re = out[0].to_host()
    var im = out[1].to_host()

    for k in range(16):
        assert_almost_equal(
            Float64(re[k]), Float64(reference[k].re.v), atol=1e-12
        )
        assert_almost_equal(
            Float64(im[k]), Float64(reference[k].im.v), atol=1e-12
        )


def test_irfft_inverts_rfft() raises:
    """`irfft[n](rfft(x)) == x`, the round trip through the half spectrum
    -- and the half spectrum itself against `numpy.fft.rfft`, so the
    round trip is not two matching mistakes."""
    var samples = _signal()
    var half = rfft(_from[8](samples))
    var re = half[0].to_host()
    var im = half[1].to_host()
    var expected_re = [5.25, 1.005203820042827, 5.5, -5.005203820042826, -0.25]
    var expected_im = [0.0, -1.96966991411009, -1.75, 3.03033008588991, 0.0]
    for k in range(5):
        assert_almost_equal(Float64(re[k]), expected_re[k], atol=1e-12)
        assert_almost_equal(Float64(im[k]), expected_im[k], atol=1e-12)

    var back = irfft(half^).to_host()
    for i in range(8):
        assert_almost_equal(Float64(back[i]), samples[i], atol=1e-12)


def test_irfft_of_a_dc_spike_is_constant() raises:
    """A half spectrum of `[n, 0, 0, 0, 0]` is the constant `1`: the
    mirror kernel has nothing to mirror and the `1/n` is the whole
    story."""
    var ctx = _cpu()
    var re = _from[5]([8.0, 0.0, 0.0, 0.0, 0.0])
    var im = zeros[dtype, 5](ctx)
    var half = (re^, im^)
    var x = irfft(half^).to_host()
    for i in range(8):
        assert_almost_equal(Float64(x[i]), 1.0, atol=1e-12)


def test_fftshift_centres_fftfreq() raises:
    """`fftshift(fftfreq(8))` is the monotone grid `-0.5 .. 0.375`, the
    canonical use, and `ifftshift` puts it back."""
    var centred = fftshift(fftfreq[dtype, 8]()).to_host()
    var expected = [-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375]
    for i in range(8):
        assert_almost_equal(Float64(centred[i]), expected[i], atol=1e-15)

    var restored = ifftshift(fftshift(fftfreq[dtype, 8]())).to_host()
    var grid = fftfreq[dtype, 8]().to_host()
    for i in range(8):
        assert_almost_equal(Float64(restored[i]), Float64(grid[i]), atol=1e-15)


def test_fftshift_and_ifftshift_differ_at_odd_n() raises:
    """NumPy: `fftshift(arange(7)) == [4, 5, 6, 0, 1, 2, 3]` while
    `ifftshift(arange(7)) == [3, 4, 5, 6, 0, 1, 2]`. The two rotations
    coincide only for even `n`, which is why both names exist."""
    var ramp: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var shifted = fftshift(_from[7](ramp)).to_host()
    var unshifted = ifftshift(_from[7](ramp)).to_host()
    var expected_shift = [4.0, 5.0, 6.0, 0.0, 1.0, 2.0, 3.0]
    var expected_unshift = [3.0, 4.0, 5.0, 6.0, 0.0, 1.0, 2.0]
    for i in range(7):
        assert_equal(Float64(shifted[i]), expected_shift[i])
        assert_equal(Float64(unshifted[i]), expected_unshift[i])


def test_fft2_matches_numpy() raises:
    """`numpy.fft.fft2(arange(16).reshape(4, 4) * 0.5 - 3)`, every
    entry."""
    var ctx = _cpu()
    var out = fft2((_matrix[4, 4](_image()), zeros[dtype, 4, 4](ctx)))
    var re = out[0].to_host()
    var im = out[1].to_host()
    var expected_re = [
        12.0,
        -4.0,
        -4.0,
        -4.0,
        -16.0,
        0.0,
        0.0,
        0.0,
        -16.0,
        0.0,
        0.0,
        0.0,
        -16.0,
        0.0,
        0.0,
        0.0,
    ]
    var expected_im = [
        0.0,
        4.0,
        0.0,
        -4.0,
        16.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -16.0,
        0.0,
        0.0,
        0.0,
    ]
    for i in range(16):
        assert_almost_equal(Float64(re[i]), expected_re[i], atol=1e-12)
        assert_almost_equal(Float64(im[i]), expected_im[i], atol=1e-12)


def test_ifft2_inverts_fft2_on_a_rectangle() raises:
    """`ifft2(fft2(x)) == x` on a `2 x 8` image -- rectangular, which the
    `Array` tier cannot do, so the two extents' normalizations are
    checked against each other rather than against a square."""
    var re_values = List[Float64]()
    var im_values = List[Float64]()
    for i in range(16):
        re_values.append(Float64(i % 5))
        im_values.append(Float64(i % 3))
    var back = ifft2(fft2((_matrix[2, 8](re_values), _matrix[2, 8](im_values))))
    var re = back[0].to_host()
    var im = back[1].to_host()
    for i in range(16):
        assert_almost_equal(Float64(re[i]), re_values[i], atol=1e-12)
        assert_almost_equal(Float64(im[i]), im_values[i], atol=1e-12)


def test_fft2_is_rows_then_columns() raises:
    """`fft2` against the 1-D `fft` applied by hand to every row and then
    every column, on a `2 x 4` image. The separability `fft2` is built
    from, checked with the rank-1 engine as the reference."""
    var ctx = _cpu()
    var re_values = List[Float64]()
    var im_values = List[Float64]()
    for i in range(8):
        re_values.append(Float64(i * i % 7))
        im_values.append(Float64(3 - i))

    # Rows.
    var rows_re = List[Float64](length=8, fill=0.0)
    var rows_im = List[Float64](length=8, fill=0.0)
    for r in range(2):
        var row_re = List[Float64]()
        var row_im = List[Float64]()
        for c in range(4):
            row_re.append(re_values[r * 4 + c])
            row_im.append(im_values[r * 4 + c])
        var out = fft((_from[4](row_re), _from[4](row_im)))
        var ore = out[0].to_host()
        var oim = out[1].to_host()
        for c in range(4):
            rows_re[r * 4 + c] = Float64(ore[c])
            rows_im[r * 4 + c] = Float64(oim[c])
    # Columns of the row-transformed image.
    var by_hand_re = List[Float64](length=8, fill=0.0)
    var by_hand_im = List[Float64](length=8, fill=0.0)
    for c in range(4):
        var col_re = List[Float64]()
        var col_im = List[Float64]()
        for r in range(2):
            col_re.append(rows_re[r * 4 + c])
            col_im.append(rows_im[r * 4 + c])
        var out = fft((_from[2](col_re), _from[2](col_im)))
        var ore = out[0].to_host()
        var oim = out[1].to_host()
        for r in range(2):
            by_hand_re[r * 4 + c] = Float64(ore[r])
            by_hand_im[r * 4 + c] = Float64(oim[r])

    var out = fft2((_matrix[2, 4](re_values), _matrix[2, 4](im_values)))
    var re = out[0].to_host()
    var im = out[1].to_host()
    for i in range(8):
        assert_almost_equal(Float64(re[i]), by_hand_re[i], atol=1e-12)
        assert_almost_equal(Float64(im[i]), by_hand_im[i], atol=1e-12)


def test_rfft2_is_the_left_half_of_fft2() raises:
    """`rfft2(x)` is exactly `fft2(x)[:, :cols/2 + 1]` for a real `x`,
    and two rows of it against `numpy.fft.rfft2` directly."""
    var ctx = _cpu()
    var values = List[Float64]()
    for i in range(4):
        for j in range(8):
            values.append(Float64((i * 8 + j) % 5) + 0.25 * Float64(j))

    var full = fft2((_matrix[4, 8](values), zeros[dtype, 4, 8](ctx)))
    var fre = full[0].to_host()
    var fim = full[1].to_host()
    var half = rfft2(_matrix[4, 8](values))
    var hre = half[0].to_host()
    var him = half[1].to_host()
    for i in range(4):
        for k in range(5):
            assert_almost_equal(
                Float64(hre[i * 5 + k]), Float64(fre[i * 8 + k]), atol=1e-12
            )
            assert_almost_equal(
                Float64(him[i * 5 + k]), Float64(fim[i * 8 + k]), atol=1e-12
            )

    var row2_re = [-3.0, -1.4644660940672627, -10.0, -8.535533905932738, -5.0]
    var row1_im = [-2.0, -3.5355339059327373, 5.0, 3.5355339059327373, 0.0]
    for k in range(5):
        assert_almost_equal(Float64(hre[2 * 5 + k]), row2_re[k], atol=1e-12)
        assert_almost_equal(Float64(him[1 * 5 + k]), row1_im[k], atol=1e-12)


def test_fftshift_of_a_2d_spectrum_centres_dc() raises:
    """`fftshift(fft2(constant))` has its one spike at the centre pixel
    `(rows/2, cols/2)`, both axes shifted at once as NumPy's default
    does."""
    var ctx = _cpu()
    var values = List[Float64](length=32, fill=1.5)
    var out = fft2((_matrix[4, 8](values), zeros[dtype, 4, 8](ctx)))
    var re_shifted = fftshift(copy(out[0]))
    var re = re_shifted.to_host()
    for i in range(4):
        for j in range(8):
            var expected = 48.0 if (i == 2 and j == 4) else 0.0
            assert_almost_equal(Float64(re[i * 8 + j]), expected, atol=1e-12)
    _ = out^


def test_the_two_tiers_agree_on_fft2() raises:
    """`fft2` over `Tensor` against `numax.fft.array.fft2` on the one
    shape both can take, a `4 x 4`."""
    var ctx = _cpu()
    var image = _image()
    var packed = Array[Complex[P], 16](uninitialized=True)
    for i in range(16):
        packed[i] = Complex[P](P(image[i]), P(Float64(i % 3)))
    var reference = array_fft2[P, 2](packed)

    var im_values = List[Float64]()
    for i in range(16):
        im_values.append(Float64(i % 3))
    var out = fft2((_matrix[4, 4](image), _matrix[4, 4](im_values)))
    var re = out[0].to_host()
    var im = out[1].to_host()
    for i in range(16):
        assert_almost_equal(
            Float64(re[i]), Float64(reference[i].re.v), atol=1e-12
        )
        assert_almost_equal(
            Float64(im[i]), Float64(reference[i].im.v), atol=1e-12
        )


def _dft_bin(
    re: List[Float64], im: List[Float64], k: Int
) -> Tuple[Float64, Float64]:
    """One bin of the naive `O(n^2)` DFT in host `Float64`, sharing
    nothing with the library -- not its twiddles, not its chirp."""
    var n = len(re)
    var out_re = 0.0
    var out_im = 0.0
    for j in range(n):
        var angle = -6.283185307179586 * Float64(j) * Float64(k) / Float64(n)
        var c = _cos(angle)
        var d = _sin(angle)
        out_re += re[j] * c - im[j] * d
        out_im += re[j] * d + im[j] * c
    return (out_re, out_im)


def _check_fft_against_the_dft[n: Int]() raises where n > 0:
    var re_values = List[Float64]()
    var im_values = List[Float64]()
    for i in range(n):
        re_values.append(_cos(0.7 * Float64(i)) + 0.3 * Float64(i % 5))
        im_values.append(_sin(1.3 * Float64(i)) - 0.1 * Float64(i))
    var out = fft((_from[n](re_values), _from[n](im_values)))
    var re = out[0].to_host()
    var im = out[1].to_host()
    for k in range(n):
        var expected = _dft_bin(re_values, im_values, k)
        assert_almost_equal(Float64(re[k]), expected[0], atol=1e-10)
        assert_almost_equal(Float64(im[k]), expected[1], atol=1e-10)


def test_fft_matches_a_naive_dft_at_every_length_to_13() raises:
    """Lengths 1 through 13 against a from-scratch DFT: the powers of two
    take the radix-2 path and everything else Bluestein, so this is both
    engines against one reference, and the dispatch between them."""
    # Spelled out rather than a `comptime for`: the `where n > 0` on the
    # helper needs evidence the prover cannot extract from a loop variable.
    _check_fft_against_the_dft[1]()
    _check_fft_against_the_dft[2]()
    _check_fft_against_the_dft[3]()
    _check_fft_against_the_dft[4]()
    _check_fft_against_the_dft[5]()
    _check_fft_against_the_dft[6]()
    _check_fft_against_the_dft[7]()
    _check_fft_against_the_dft[8]()
    _check_fft_against_the_dft[9]()
    _check_fft_against_the_dft[10]()
    _check_fft_against_the_dft[11]()
    _check_fft_against_the_dft[12]()
    _check_fft_against_the_dft[13]()


def test_fft_at_a_prime_length_matches_numpy() raises:
    """`numpy.fft.fft([2, 1, -1, 0.5, 0, 3, -2.5])`: a length no radix
    reaches, so this is Bluestein against NumPy's own values."""
    var ctx = _cpu()
    var samples: List[Float64] = [2.0, 1.0, -1.0, 0.5, 0.0, 3.0, -2.5]
    var out = fft((_from[7](samples), zeros[dtype, 7](ctx)))
    var re = out[0].to_host()
    var im = out[1].to_host()
    var expected_re = [
        3.0,
        0.169238995348061,
        0.843588566059,
        4.487172438592938,
        4.487172438592938,
        0.843588566059,
        0.169238995348061,
    ]
    var expected_im = [
        0.0,
        0.946359590530411,
        -4.7568669078726,
        -5.133382972874485,
        5.133382972874485,
        4.7568669078726,
        -0.946359590530411,
    ]
    for k in range(7):
        assert_almost_equal(Float64(re[k]), expected_re[k], atol=1e-12)
        assert_almost_equal(Float64(im[k]), expected_im[k], atol=1e-12)


def test_ifft_inverts_fft_at_an_odd_length() raises:
    """`ifft(fft(x)) == x` at `n = 5`, real and imaginary parts, so the
    inverse chirp and the `1/n` in the Bluestein path are both checked;
    and the forward spectrum against `numpy.fft.fft`."""
    var re_values: List[Float64] = [1.0, -2.0, 0.5, 3.0, 1.5]
    var im_values: List[Float64] = [0.25, 0.0, -1.0, 2.0, -0.5]
    var out = fft((_from[5](re_values), zeros[dtype, 5](_cpu())))
    var re = out[0].to_host()
    var im = out[1].to_host()
    var expected_re = [
        4.0,
        -1.9860679774997898,
        2.48606797749979,
        2.48606797749979,
        -1.9860679774997898,
    ]
    var expected_im = [
        0.0,
        4.79816093776422,
        -0.3203929077142277,
        0.3203929077142277,
        -4.79816093776422,
    ]
    for k in range(5):
        assert_almost_equal(Float64(re[k]), expected_re[k], atol=1e-12)
        assert_almost_equal(Float64(im[k]), expected_im[k], atol=1e-12)

    var back = ifft(fft((_from[5](re_values), _from[5](im_values))))
    var bre = back[0].to_host()
    var bim = back[1].to_host()
    for i in range(5):
        assert_almost_equal(Float64(bre[i]), re_values[i], atol=1e-12)
        assert_almost_equal(Float64(bim[i]), im_values[i], atol=1e-12)


def test_rfft_and_irfft_round_trip_at_an_odd_length() raises:
    """`rfft` of nine samples is five bins, against `numpy.fft.rfft`, and
    `irfft[n=9]` recovers them -- the odd length NumPy's `n=` argument
    exists for, where the default `2 * (keep - 1)` would give eight."""
    var samples: List[Float64] = [
        2.0,
        1.0,
        -1.0,
        0.5,
        0.0,
        3.0,
        -2.5,
        1.25,
        -0.75,
    ]
    var half = rfft(_from[9](samples))
    var re = half[0].to_host()
    var im = half[1].to_host()
    var expected_re = [
        3.5,
        0.4158452928387524,
        5.106622218577189,
        -1.75,
        3.4775324885840577,
    ]
    var expected_im = [
        0.0,
        -0.48107665405028577,
        -0.2841548629949111,
        -0.8660254037844386,
        -7.597306843004573,
    ]
    for k in range(5):
        assert_almost_equal(Float64(re[k]), expected_re[k], atol=1e-12)
        assert_almost_equal(Float64(im[k]), expected_im[k], atol=1e-12)

    var back = irfft[n=9](half^).to_host()
    for i in range(9):
        assert_almost_equal(Float64(back[i]), samples[i], atol=1e-12)


def test_fft2_at_non_power_of_two_extents_matches_numpy() raises:
    """`numpy.fft.fft2(arange(18).reshape(3, 6) * 0.5 - 2)`: both axes
    take the Bluestein path, one of them through the transposed view."""
    var ctx = _cpu()
    var values = List[Float64]()
    for i in range(18):
        values.append(Float64(i) * 0.5 - 2.0)
    var out = fft2((_matrix[3, 6](values), zeros[dtype, 3, 6](ctx)))
    var re = out[0].to_host()
    var im = out[1].to_host()
    var expected_re = [
        40.5,
        -4.5,
        -4.5,
        -4.5,
        -4.5,
        -4.5,
        -27.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -27.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    var expected_im = [
        0.0,
        7.794228634059948,
        2.598076211353316,
        0.0,
        -2.598076211353316,
        -7.794228634059948,
        15.588457268119894,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -15.588457268119894,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    for i in range(18):
        assert_almost_equal(Float64(re[i]), expected_re[i], atol=1e-12)
        assert_almost_equal(Float64(im[i]), expected_im[i], atol=1e-12)

    var back = ifft2(fft2((_matrix[3, 6](values), zeros[dtype, 3, 6](ctx))))
    var bre = back[0].to_host()
    var bim = back[1].to_host()
    for i in range(18):
        assert_almost_equal(Float64(bre[i]), values[i], atol=1e-12)
        assert_almost_equal(Float64(bim[i]), 0.0, atol=1e-12)


def test_bluestein_agrees_with_radix2_at_a_power_of_two() raises:
    """Bluestein is never dispatched to at a power of two, so it is called
    directly here at `n = 16` and pinned against `fft`, which took the
    radix-2 path: two algorithms, one answer, to `1e-12`."""
    var ctx = _cpu()
    var re_values = List[Float64]()
    var im_values = List[Float64]()
    for i in range(16):
        re_values.append(Float64((i * 7) % 11) - 4.0)
        im_values.append(0.5 * Float64(i % 3))

    var src_re = _from[16](re_values)
    var src_im = _from[16](im_values)
    var out_re = zeros[dtype, 16](ctx)
    var out_im = zeros[dtype, 16](ctx)
    _bluestein[dtype, 1, 16, False, False](
        _as_matrix[dtype, 1, 16](src_re),
        _as_matrix[dtype, 1, 16](src_im),
        _as_matrix[dtype, 1, 16](out_re),
        _as_matrix[dtype, 1, 16](out_im),
        ctx,
    )
    var bre = out_re.to_host()
    var bim = out_im.to_host()

    var reference = fft((_from[16](re_values), _from[16](im_values)))
    var rre = reference[0].to_host()
    var rim = reference[1].to_host()
    for k in range(16):
        assert_almost_equal(Float64(bre[k]), Float64(rre[k]), atol=1e-12)
        assert_almost_equal(Float64(bim[k]), Float64(rim[k]), atol=1e-12)


def test_next_fast_len_is_the_next_power_of_two() raises:
    """`1 -> 1`, `7 -> 8`, `8 -> 8`, `9 -> 16`, `1000 -> 1024`: the set
    this engine is fast at, where SciPy's would say `7`, `9` and `1000`."""
    assert_equal(next_fast_len(1), 1)
    assert_equal(next_fast_len(7), 8)
    assert_equal(next_fast_len(8), 8)
    assert_equal(next_fast_len(9), 16)
    assert_equal(next_fast_len(1000), 1024)
    comptime at_compile_time = next_fast_len(1000)
    assert_equal(at_compile_time, 1024)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
