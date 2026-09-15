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

from std.math import log10 as _log10, sqrt as _sqrt

from numax.core.array import Static
from numax.signal import (
    butter,
    cheby1,
    cheby2,
    ellip,
    find_peaks,
    freqz,
    hilbert,
    iirfilter,
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
    # A single edge with a band `btype` is the overload's own error: the
    # band forms double the order and so live on the pair overload.
    var raised = False
    try:
        var band = butter[dtype, 2](0.3, "bandpass")
        _ = band^
    except:
        raised = True
    assert_true(raised)


def test_butter_band_forms_match_scipy() raises:
    """`lp2bp`/`lp2bs` split every prototype root in two, so a
    `(low, high)` pair is a separate overload returning
    `TransferFunction[dtype, 2 * order]`. Values from
    `scipy.signal.butter(order, (low, high), btype)`."""
    var bp = butter[dtype, 2]((0.2, 0.5), "bandpass")
    _assert_close(
        bp.b,
        [
            0.13110643991662593,
            0.0,
            -0.26221287983325187,
            0.0,
            0.13110643991662593,
        ],
        atol=1e-9,
    )
    _assert_close(
        bp.a,
        [
            1.0,
            -1.4000685161680915,
            1.2722149379250074,
            -0.6584184943899084,
            0.27221493792500717,
        ],
        atol=1e-9,
    )

    var bs = butter[dtype, 2]((0.2, 0.5), "bandstop")
    _assert_close(
        bs.b,
        [
            0.5050010290458776,
            -1.0292435052789999,
            1.5344278177582595,
            -1.029243505279,
            0.5050010290458778,
        ],
        atol=1e-9,
    )
    _assert_close(
        bs.a,
        [
            1.0,
            -1.4000685161680915,
            1.2722149379250074,
            -0.6584184943899084,
            0.27221493792500717,
        ],
        atol=1e-9,
    )

    var wide = butter[dtype, 3]((0.25, 0.6), "bandpass")
    _assert_close(
        wide.b,
        [
            0.07176120384378178,
            0.0,
            -0.21528361153134534,
            0.0,
            0.21528361153134534,
            0.0,
            -0.07176120384378178,
        ],
        atol=1e-9,
    )
    _assert_close(
        wide.a,
        [
            1.0,
            -1.059084699704097,
            1.2633316535876398,
            -0.8192540490036013,
            0.7003182102664679,
            -0.22195927017547845,
            0.09209449209926929,
        ],
        atol=1e-9,
    )

    # And a pair with a single-edge `btype` is the mirror error, rather
    # than a silent lowpass at the first edge.
    var raised = False
    try:
        var wrong = butter[dtype, 2]((0.2, 0.5), "lowpass")
        _ = wrong^
    except:
        raised = True
    assert_true(raised)


def test_cheby1_matches_scipy() raises:
    """`cheby1(order, rp, wn, btype)`: Butterworth's circle squashed onto
    an ellipse, `rp` dB of equiripple in the passband and no finite
    zeros."""
    var low = cheby1[dtype, 3](1.0, 0.3)
    _assert_close(
        low.b,
        [
            0.034384966351816024,
            0.10315489905544807,
            0.10315489905544807,
            0.034384966351816024,
        ],
        atol=1e-9,
    )
    _assert_close(
        low.a,
        [1.0, -1.5804049383264676, 1.2538449813550228, -0.3983603122140273],
        atol=1e-9,
    )

    var high = cheby1[dtype, 4](0.5, 0.25, "highpass")
    _assert_close(
        high.b,
        [
            0.29503202706437404,
            -1.1801281082574961,
            1.7701921623862442,
            -1.1801281082574961,
            0.29503202706437404,
        ],
        atol=1e-9,
    )
    _assert_close(
        high.a,
        [
            1.0,
            -1.7130535304452352,
            1.5349813194610458,
            -0.5928969310617397,
            0.1592885984646984,
        ],
        atol=1e-9,
    )

    var band = cheby1[dtype, 2](1.0, (0.2, 0.5), "bandpass")
    _assert_close(
        band.b,
        [
            0.13822540868747046,
            0.0,
            -0.2764508173749409,
            0.0,
            0.13822540868747046,
        ],
        atol=1e-9,
    )
    _assert_close(
        band.a,
        [
            1.0,
            -1.4131703881533586,
            1.3361878808461982,
            -0.7954919330160503,
            0.39386888746166343,
        ],
        atol=1e-9,
    )


def test_cheby2_matches_scipy() raises:
    """`cheby2(order, rs, wn, btype)`: flat passband, `rs` dB of
    equiripple stopband held down by finite zeros -- so `b` is not a
    scaled `(1 + z)^order` the way type I's is, and an odd order carries
    one fewer zero than pole."""
    var odd = cheby2[dtype, 3](40.0, 0.3)
    _assert_close(
        odd.b,
        [
            0.01461536600299342,
            0.00041761625757361505,
            0.00041761625757361505,
            0.01461536600299342,
        ],
        atol=1e-9,
    )
    _assert_close(
        odd.a,
        [1.0, -2.321702622809876, 1.8559974074997827, -0.5042288201687728],
        atol=1e-9,
    )

    var even = cheby2[dtype, 4](30.0, 0.4)
    _assert_close(
        even.b,
        [
            0.06749365754371305,
            0.04452906442451189,
            0.0989840329798856,
            0.04452906442451189,
            0.06749365754371306,
        ],
        atol=1e-9,
    )
    _assert_close(
        even.a,
        [
            1.0,
            -1.6203535622932628,
            1.3662899249509177,
            -0.5094108782339453,
            0.0865039924926256,
        ],
        atol=1e-9,
    )

    var stop = cheby2[dtype, 2](30.0, (0.2, 0.5), "bandstop")
    _assert_close(
        stop.b,
        [
            0.14257159170153494,
            -0.25719006571628633,
            0.3506656253228662,
            -0.2571900657162864,
            0.14257159170153494,
        ],
        atol=1e-9,
    )
    _assert_close(
        stop.a,
        [
            1.0,
            -0.6197771730141808,
            -0.6525739253304941,
            0.10539704158160795,
            0.28838273405643006,
        ],
        atol=1e-9,
    )


def test_ellip_matches_scipy() raises:
    """`ellip(order, rp, rs, wn, btype)`: the degree equation, Jacobi
    `sn`/`cn`/`dn` and the inverse `sc`, all resting on the
    arithmetic-geometric mean's `K(m)`. Routing them through the tier-1
    `elliptic_k` instead lands these coefficients `1e-8` from SciPy rather
    than `4e-13`, which is what `design.mojo`'s docstring records."""
    var odd = ellip[dtype, 3](1.0, 40.0, 0.3)
    _assert_close(
        odd.b,
        [
            0.05548085891202061,
            0.09185345473381835,
            0.09185345473381833,
            0.0554808589120206,
        ],
        atol=1e-9,
    )
    _assert_close(
        odd.a,
        [1.0, -1.5689413939524117, 1.2627424658881536, -0.3991324446440641],
        atol=1e-9,
    )

    var even = ellip[dtype, 4](0.5, 50.0, 0.25)
    _assert_close(
        even.b,
        [
            0.017054033835956155,
            0.0188110446588136,
            0.030460361036975827,
            0.0188110446588136,
            0.01705403383595615,
        ],
        atol=1e-9,
    )
    _assert_close(
        even.a,
        [
            1.0,
            -2.5464271318846077,
            2.918512308784684,
            -1.6600403786816833,
            0.39620088867898984,
        ],
        atol=1e-9,
    )

    var band = ellip[dtype, 2](1.0, 40.0, (0.2, 0.5), "bandpass")
    _assert_close(
        band.b,
        [
            0.14447571250105187,
            -0.011053673153945743,
            -0.2616252421634211,
            -0.011053673153945566,
            0.1444757125010519,
        ],
        atol=1e-9,
    )
    _assert_close(
        band.a,
        [
            1.0,
            -1.412743118847099,
            1.335877264862293,
            -0.7979915119420344,
            0.3967410190059745,
        ],
        atol=1e-9,
    )


def test_iirfilter_dispatches_on_ftype_and_refuses_an_unknown_one() raises:
    """The named-`ftype` front door only chooses the prototype, so it has
    to give the family's own answer; and a typo raises rather than
    silently designing a different filter."""
    var via_name = iirfilter[dtype, 2](0.3, ftype="butter")
    var direct = butter[dtype, 2](0.3)
    var named_b = via_name.b.to_host()
    var direct_b = direct.b.to_host()
    for i in range(3):
        assert_almost_equal(
            Float64(named_b[i]), Float64(direct_b[i]), atol=1e-15
        )

    var cheb = iirfilter[dtype, 3](0.3, rp=1.0, ftype="cheby1")
    _assert_close(
        cheb.b,
        [
            0.034384966351816024,
            0.10315489905544807,
            0.10315489905544807,
            0.034384966351816024,
        ],
        atol=1e-9,
    )

    var band = iirfilter[dtype, 2](
        (0.2, 0.5), rp=1.0, rs=40.0, btype="bandpass", ftype="ellip"
    )
    _assert_close(
        band.b,
        [
            0.14447571250105187,
            -0.011053673153945743,
            -0.2616252421634211,
            -0.011053673153945566,
            0.1444757125010519,
        ],
        atol=1e-9,
    )

    var raised = False
    try:
        var bad = iirfilter[dtype, 2](0.3, ftype="besel")
        _ = bad^
    except:
        raised = True
    assert_true(raised)


def _response_db[
    nb: Int, na: Int
](
    mut b: Static[dtype, nb], mut a: Static[dtype, na], at: Float64
) raises -> Float64 where (nb > 0 and na > 0):
    """`20 log10 |H(e^{jw})|` at one normalized frequency, read off a
    `freqz` grid fine enough that `at` lands on a sample.

    Takes the two coefficient tensors rather than the `TransferFunction`
    because `freqz`'s `nb > 0 and na > 0` has to be restated on whatever
    generic wrapper calls it, and `order + 1 > 0` is not something the
    prover derives from `order >= 1`."""
    comptime grid = 400
    var response = freqz[worN=grid](b, a)
    var re = response.real.to_host()
    var im = response.imag.to_host()
    var k = Int(at * Float64(grid))
    var mag = _sqrt(
        Float64(re[k]) * Float64(re[k]) + Float64(im[k]) * Float64(im[k])
    )
    return 20.0 * _log10(mag)


def test_the_ripple_and_attenuation_arguments_are_honored() raises:
    """`rp` and `rs` are decibels, and `freqz` is what says whether the
    filter that came back has them: a `cheby1` passband never drops more
    than `rp` below unity, a `cheby2` stopband never rises above `-rs`,
    and an `ellip` does both."""
    var c1 = cheby1[dtype, 5](1.0, 0.3)
    for w in [0.02, 0.1, 0.2, 0.28]:
        var db = _response_db(c1.b, c1.a, w)
        assert_true(db <= 1e-9)
        assert_true(db >= -1.0 - 1e-9)

    var c2 = cheby2[dtype, 5](40.0, 0.3)
    assert_almost_equal(_response_db(c2.b, c2.a, 0.0), 0.0, atol=1e-9)
    for w in [0.35, 0.5, 0.7, 0.9]:
        assert_true(_response_db(c2.b, c2.a, w) <= -40.0 + 1e-9)

    var el = ellip[dtype, 5](1.0, 40.0, 0.3)
    for w in [0.02, 0.1, 0.2, 0.28]:
        var db = _response_db(el.b, el.a, w)
        assert_true(db <= 1e-9)
        assert_true(db >= -1.0 - 1e-9)
    for w in [0.4, 0.6, 0.8]:
        assert_true(_response_db(el.b, el.a, w) <= -40.0 + 1e-9)


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
