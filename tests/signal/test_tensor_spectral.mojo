"""Tests for `numax.signal`'s spectral estimators, peak finder and IIR
design over `Tensor`, each against `scipy.signal`'s own values on a
16-sample signal at `fs = 4`.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax.core.array import Static
from numax.signal import (
    butter,
    find_peaks,
    freqz,
    hilbert,
    periodogram,
    spectrogram,
    stft,
    welch,
)

comptime dtype = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _from[n: Int](values: List[Float64]) raises -> Static[dtype, n]:
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype](values[i]))
    return Static[dtype, n](_cpu(), out^)


def _x() -> List[Float64]:
    return [
        1.0,
        0.0,
        -1.0,
        2.0,
        0.5,
        3.0,
        -2.0,
        1.0,
        0.75,
        -0.5,
        1.5,
        2.5,
        -1.0,
        0.0,
        1.0,
        -3.0,
    ]


def _assert_close[
    n: Int
](got: Static[dtype, n], want: List[Float64], atol: Float64 = 1e-12) raises:
    var values = got.to_host()
    for i in range(n):
        assert_almost_equal(Float64(values[i]), want[i], atol=atol)


def test_periodogram_matches_scipy() raises:
    """`periodogram(x, fs=4)`: the grid `0 .. 2` in steps of `0.25`, the
    mean removed so DC is zero, the interior doubled; and with a Hann
    window."""
    var x = _from[16](_x())
    var boxcar = periodogram(x, fs=4.0)
    _assert_close(
        boxcar.frequencies, [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    )
    _assert_close(
        boxcar.power,
        [
            0.0,
            0.5767091165021337,
            1.1361454961651343,
            0.2498496724011588,
            0.095703125,
            2.430508034142801,
            2.727135753834866,
            2.1257456769539056,
            0.2822265625,
        ],
    )
    var hann = periodogram(x, fs=4.0, window="hann")
    _assert_close(
        hann.power,
        [
            0.16412782513029878,
            0.05121523857962334,
            0.6060370024910505,
            0.5636535884692869,
            0.285009507327854,
            2.610311966000559,
            2.2326804681914583,
            0.9231122628893013,
            0.9256182327319795,
        ],
    )


def test_welch_matches_scipy() raises:
    """`welch(x, fs=4, nperseg=8)` with the defaults -- Hann, half overlap,
    mean removed -- and with a Hamming window, `noverlap=2` and no
    detrending."""
    var x = _from[16](_x())
    var default = welch[nperseg=8](x, fs=4.0)
    _assert_close(default.frequencies, [0.0, 0.5, 1.0, 1.5, 2.0])
    _assert_close(
        default.power,
        [
            0.15576085775951348,
            0.6526890938988258,
            0.8413670433992527,
            1.4554393133380923,
            0.8813998116337071,
        ],
    )
    var hamming = welch[nperseg=8, noverlap=2](
        x, fs=4.0, window="hamming", detrend=False
    )
    _assert_close(
        hamming.power,
        [
            0.87214283565009,
            1.5042991338542708,
            1.2061171199631842,
            1.1907359937935413,
            1.2462417044955347,
        ],
    )


def test_spectrogram_matches_scipy() raises:
    """`spectrogram(x, fs=4, window="hann", nperseg=8, noverlap=4)`: three
    frames centred at `1, 2, 3` seconds, `power` in `(frequencies,
    times)` orientation."""
    var x = _from[16](_x())
    var result = spectrogram[nperseg=8, noverlap=4](x, fs=4.0)
    _assert_close(result.frequencies, [0.0, 0.5, 1.0, 1.5, 2.0])
    _assert_close(result.times, [1.0, 2.0, 3.0])
    var expected: List[Float64] = [
        0.112949434901121,
        0.22489027181320068,
        0.12944286656421874,
        0.8805653254943946,
        0.34689134739219873,
        0.7306106088098836,
        0.75,
        0.4095177968644246,
        1.3645833333333333,
        1.2142556509887896,
        1.3940461158827733,
        1.7580161731427135,
        2.442809041582063,
        0.044680436264013,
        0.15670995705504462,
    ]
    var power = result.power.to_host()
    for i in range(15):
        assert_almost_equal(Float64(power[i]), expected[i], atol=1e-12)


def test_stft_matches_scipy() raises:
    """`stft(x, fs=4, window="hann", nperseg=8)`: zero boundary, five frames
    at whole seconds, scaled by `sum(w)`, real and imaginary parts."""
    var x = _from[16](_x())
    var result = stft[nperseg=8](x, fs=4.0)
    _assert_close(result.times, [0.0, 1.0, 2.0, 3.0, 4.0])
    var expected_re: List[Float64] = [
        0.19822330470336313,
        0.8535533905932737,
        0.4330582617584079,
        0.4678300858899106,
        -0.5151650429449552,
        -0.1982233047033631,
        -0.8535533905932737,
        -0.12055826175840778,
        -0.21783008588991076,
        0.4526650429449553,
        0.375,
        0.5,
        0.25,
        -0.5625,
        -0.125,
        -0.30177669529663687,
        0.6035533905932737,
        -0.2544417382415922,
        0.7178300858899107,
        -0.4526650429449553,
        0.051776695296636865,
        -1.3535533905932737,
        -0.1830582617584079,
        -0.3428300858899106,
        0.7651650429449552,
    ]
    var expected_im: List[Float64] = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0732233047033631,
        0.05177669529663692,
        0.19822330470336313,
        -0.5044417382415922,
        0.32766504294495524,
        0.07322330470336313,
        -0.1767766952966368,
        0.30177669529663687,
        0.4419417382415922,
        -0.6401650429449552,
        0.1767766952966369,
        0.3017766952966369,
        -0.6767766952966369,
        -0.3794417382415922,
        0.5776650429449552,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    var re = result.real.to_host()
    var im = result.imag.to_host()
    for i in range(25):
        assert_almost_equal(Float64(re[i]), expected_re[i], atol=1e-12)
        assert_almost_equal(Float64(im[i]), expected_im[i], atol=1e-12)


def test_hilbert_matches_scipy() raises:
    """The analytic signal: real part `x` itself, imaginary part SciPy's
    Hilbert transform."""
    var x = _from[16](_x())
    var analytic = hilbert(x)
    _assert_close(analytic[0], _x())
    _assert_close(
        analytic[1],
        [
            -2.3384609839672077,
            1.4525430973798101,
            -2.3882193870031196,
            -0.42267852614520296,
            -1.086459096004841,
            1.2516280076812103,
            1.4411262117472747,
            -1.9152236707514465,
            1.277800812187386,
            -0.7498780544348549,
            -1.1473145189296183,
            1.31100374086998,
            2.1471192677846624,
            -1.0792930506261655,
            2.0944076941854632,
            0.15189845602666974,
        ],
    )


def test_find_peaks_matches_scipy() raises:
    """Unfiltered (a flat top counts once, at its midpoint), then by
    `height`, `distance` and `threshold`."""
    var x = _from[12](
        [0.0, 1.0, 0.5, 2.0, 1.5, 3.0, 0.0, 0.2, 0.1, 0.9, 0.9, 0.3]
    )
    var all_peaks = find_peaks(x)
    assert_equal(len(all_peaks), 5)
    var expected_all: List[Int] = [1, 3, 5, 7, 9]
    for i in range(5):
        assert_equal(all_peaks[i], expected_all[i])
    var tall = find_peaks(x, height=1.0)
    assert_equal(len(tall), 3)
    assert_equal(tall[0], 1)
    assert_equal(tall[1], 3)
    assert_equal(tall[2], 5)
    var spaced = find_peaks(x, distance=3)
    assert_equal(len(spaced), 3)
    assert_equal(spaced[0], 1)
    assert_equal(spaced[1], 5)
    assert_equal(spaced[2], 9)
    var sharp = find_peaks(x, threshold=0.4)
    assert_equal(len(sharp), 3)
    assert_equal(sharp[0], 1)
    assert_equal(sharp[1], 3)
    assert_equal(sharp[2], 5)


def test_butter_matches_scipy() raises:
    """`butter(2, 0.3)` and `butter(4, 0.2, "highpass")`: the `(b, a)` SciPy
    designs, through the same prototype, warp and bilinear transform."""
    var low = butter[dtype, 2](0.3)
    _assert_close(
        low.b, [0.13110643991662593, 0.26221287983325187, 0.13110643991662593]
    )
    _assert_close(low.a, [1.0, -0.7477891782585034, 0.27221493792500717])
    var high = butter[dtype, 4](0.2, "highpass")
    _assert_close(
        high.b,
        [
            0.43284664499029185,
            -1.7313865799611674,
            2.597079869941751,
            -1.7313865799611674,
            0.43284664499029185,
        ],
    )
    _assert_close(
        high.a,
        [
            1.0,
            -2.3695130071820376,
            2.31398841441588,
            -1.0546654058785674,
            0.18737949236818485,
        ],
    )
    var raised = False
    try:
        var band = butter[dtype, 2](0.3, "bandpass")
        _ = band^
    except:
        raised = True
    assert_true(raised)


def test_freqz_matches_scipy() raises:
    """`freqz([.25, .5, .25], [1, -.3], worN=5)`: the grid `k pi / 5` and
    the complex response at each point."""
    var b = _from[3]([0.25, 0.5, 0.25])
    var a = _from[2]([1.0, -0.3])
    var response = freqz[worN=5](b, a)
    _assert_close(
        response.w,
        [
            0.0,
            0.6283185307179586,
            1.2566370614359172,
            1.8849555921538759,
            2.5132741228718345,
        ],
    )
    _assert_close(
        response.real,
        [
            1.4285714285714286,
            0.7615249116918575,
            0.006524171967753254,
            -0.16497452912150362,
            -0.06722166688039091,
        ],
    )
    _assert_close(
        response.imag,
        [
            0.0,
            -0.8793677171730836,
            -0.6881291043722111,
            -0.2576284445145277,
            -0.035627861996001646,
        ],
    )


def test_butter_feeds_freqz_with_unit_gain_at_dc() raises:
    """A lowpass design passes DC exactly and a highpass blocks it: the
    design and the response agree with each other, not only with SciPy."""
    var low = butter[dtype, 3](0.4)
    var response = freqz[worN=4](low.b, low.a)
    var re = response.real.to_host()
    var im = response.imag.to_host()
    assert_almost_equal(Float64(re[0]), 1.0, atol=1e-12)
    assert_almost_equal(Float64(im[0]), 0.0, atol=1e-12)
    var high = butter[dtype, 3](0.4, "highpass")
    var blocked = freqz[worN=4](high.b, high.a).real.to_host()
    assert_almost_equal(Float64(blocked[0]), 0.0, atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
