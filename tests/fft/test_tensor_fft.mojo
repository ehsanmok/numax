"""Tests for `numax.fft` over `Tensor`.

The load-bearing check is the last group: the `Tensor` tier is pinned
against `numax.fft.array`, which is separately tested against NumPy's exact
outputs. Two tiers computing the same transform by different routes -- a
register-resident `comptime` butterfly against `log2(n) + 1` device
launches over a bit-reversed buffer -- agreeing to `1e-12` is a much
stronger statement than either matching a table of constants alone.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext
from std.collections import Array

from numax.core.array import Static, zeros
from numax.core.complex import Complex
from numax.core.plain import Plain
from numax.fft import fft, fftfreq, ifft, rfft, rfftfreq
from numax.fft.array import fft as array_fft

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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
