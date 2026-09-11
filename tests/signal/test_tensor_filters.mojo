"""Tests for `numax.signal.filters` over `Tensor`.

Every expected value is `scipy.signal`'s own on a 16-sample signal:
`lfilter` (with `a[0] != 1` and with `nb > na`), `lfilter_zi`, `filtfilt`
at the default and an explicit `padlen`, `sosfilt` through one and two
sections of a Butterworth design, `medfilt` at two window sizes, both
`detrend` types, `savgol_filter` in every mode and at two derivative
orders, `resample` up, down, to an odd length and from an odd length, and
`firwin` as lowpass, highpass, bandpass, bandstop, multiband, even-length
and under a different window.
"""

from std.testing import TestSuite, assert_almost_equal, assert_true

from max.gpu.host import DeviceContext

from numax.core.array import Static
from numax.signal import (
    detrend,
    filtfilt,
    firwin,
    lfilter,
    lfilter_zi,
    medfilt,
    resample,
    savgol_filter,
    sosfilt,
)

comptime dtype = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


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


def test_lfilter_matches_scipy() raises:
    """`lfilter([.25, .5, .25], [1, -.3], x)`: a one-pole smoother, the
    recurrence feeding its own output back."""
    var b = _from[3]([0.25, 0.5, 0.25])
    var a = _from[2]([1.0, -0.3])
    var x = _from[16](_x())
    _assert_close(
        lfilter(b, a, x),
        [
            0.25,
            0.575,
            0.1725,
            0.05175000000000002,
            0.890525,
            1.7671575,
            1.6551472500000002,
            0.49654417500000003,
            0.3364632525,
            0.60093897575,
            0.492781692725,
            1.3978345078175,
            1.79435035234525,
            0.6633051057035749,
            0.19899153171107248,
            -0.19030254048667827,
        ],
    )


def test_lfilter_normalizes_by_a0_and_takes_a_longer_numerator() raises:
    """`a = [2]` halves everything (an FIR, `na == 1`), and `b` shorter than
    `a` pads: both against SciPy."""
    var b = _from[4]([0.1, 0.2, 0.3, 0.4])
    var a = _from[1]([2.0])
    var x = _from[16](_x())
    _assert_close(
        lfilter(b, a, x),
        [
            0.05,
            0.1,
            0.09999999999999999,
            0.2,
            0.07500000000000001,
            0.3,
            0.675,
            0.4,
            0.4375000000000001,
            -0.19999999999999998,
            0.3375,
            0.35000000000000003,
            0.325,
            0.5750000000000001,
            0.39999999999999997,
            -0.25,
        ],
    )
    var b1 = _from[1]([0.5])
    var a3 = _from[3]([1.0, -0.5, 0.1])
    _assert_close(
        lfilter(b1, a3, x),
        [
            0.5,
            0.25,
            -0.425,
            0.7625,
            0.67375,
            1.760625,
            -0.1870624999999999,
            0.23040625000000003,
            0.508909375,
            -0.01858593750000001,
            0.68981609375,
            1.596766640625,
            0.22940171093750006,
            -0.04497580859375,
            0.454571924609375,
            -1.2682164568359375,
        ],
    )


def test_lfilter_zi_matches_scipy() raises:
    """`lfilter_zi([.25, .5, .25], [1, -.3]) == [1.1786, .25]`, and for a
    longer numerator than denominator."""
    var b = _from[3]([0.25, 0.5, 0.25])
    var a = _from[2]([1.0, -0.3])
    _assert_close(lfilter_zi(b, a), [1.1785714285714286, 0.25])
    var b4 = _from[4]([0.1, 0.2, 0.3, 0.4])
    var a3 = _from[3]([1.0, -0.5, 0.1])
    _assert_close(
        lfilter_zi(b4, a3), [1.5666666666666669, 0.5333333333333333, 0.4]
    )


def test_filtfilt_matches_scipy() raises:
    """`filtfilt` at the default `padlen = 3 * 3 = 9` and at `padlen = 4`:
    odd extension, `lfilter_zi` starts, both passes."""
    var b = _from[3]([0.25, 0.5, 0.25])
    var a = _from[2]([1.0, -0.3])
    var x = _from[16](_x())
    _assert_close(
        filtfilt(b, a, x),
        [
            2.0408532396127095,
            1.0434545065773362,
            0.9170314676182785,
            1.5592598257690755,
            2.0399458993426958,
            1.6858769418427197,
            0.9604069893525184,
            0.7098517861380973,
            0.8895545002825684,
            1.2721962315552349,
            1.7535918077014523,
            1.6096040030933756,
            0.6571360702044778,
            -0.5761762813504394,
            -2.6464098924444075,
            -6.122446327864241,
        ],
        atol=1e-11,
    )
    _assert_close(
        filtfilt(b, a, x, padlen=4),
        [
            2.0503346330400736,
            1.046298925609769,
            0.9178847966754209,
            1.5595158356442602,
            2.040022739498725,
            1.685900117867774,
            0.9604143554208541,
            0.7098553734946624,
            0.8895601682764194,
            1.2722132379096616,
            1.7536479294620178,
            1.6097909058023356,
            0.6577590282864683,
            -0.5740997696948344,
            -2.6394881915110333,
            -6.099373992795255,
        ],
        atol=1e-11,
    )
    var raised = False
    try:
        _ = filtfilt(b, a, x, padlen=16)
    except:
        raised = True
    assert_true(raised)


def test_sosfilt_matches_scipy() raises:
    """`sosfilt(butter(2, .3, output="sos"), x)` -- one section -- and the
    two-section `butter(4, .3)` cascade, SciPy's coefficients in, SciPy's
    output out."""
    var sos1 = _matrix[1, 6](
        [
            0.13110643991662593,
            0.26221287983325187,
            0.13110643991662593,
            1.0,
            -0.7477891782585034,
            0.27221493792500717,
        ]
    )
    var x = _from[16](_x())
    _assert_close(
        sosfilt(sos1, x),
        [
            0.13110643991662593,
            0.3602528568029034,
            0.23370405635044847,
            0.07669515520207199,
            0.4526066116008969,
            1.104215398750675,
            1.2924930245955837,
            0.6659283705404272,
            0.2444679504641281,
            0.26374771757219917,
            0.29456291092640674,
            0.8040070881234416,
            1.2421287947704418,
            0.7755409511061384,
            0.24181511777017758,
            -0.16139354357244007,
        ],
    )
    var sos2 = _matrix[2, 6](
        [
            0.018563010626897174,
            0.03712602125379435,
            0.018563010626897174,
            1.0,
            -0.6727409111915273,
            0.14453519983312102,
            1.0,
            2.0,
            1.0,
            1.0,
            -0.8976579400366445,
            0.5271869046315972,
        ]
    )
    _assert_close(
        sosfilt(sos2, x),
        [
            0.018563010626897174,
            0.10340337307140438,
            0.23152036771146112,
            0.27779460507980513,
            0.25456289512397906,
            0.3909961288281715,
            0.7774797330067518,
            1.1027825573998704,
            1.0075036562017672,
            0.5872500409131886,
            0.22335967647134142,
            0.18201777390767895,
            0.5195474281639716,
            0.9534515122336964,
            1.0263677773399402,
            0.6240911235618303,
        ],
    )


def test_medfilt_matches_scipy() raises:
    """`medfilt(x, 3)` and `medfilt(x, 5)`: zero padding at both ends, so
    the first and last outputs of this signal are `0`."""
    var x = _from[16](_x())
    _assert_close(
        medfilt(x),
        [
            0.0,
            0.0,
            0.0,
            0.5,
            2.0,
            0.5,
            1.0,
            0.75,
            0.75,
            0.75,
            1.5,
            1.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )
    _assert_close(
        medfilt[kernel_size=5](x),
        [
            0.0,
            0.0,
            0.5,
            0.5,
            0.5,
            1.0,
            0.75,
            0.75,
            0.75,
            1.0,
            0.75,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
    )


def test_detrend_matches_scipy() raises:
    """Linear (default) and constant detrending against SciPy's values; the
    linear residual has zero mean and zero slope."""
    var x = _from[16](_x())
    _assert_close(
        detrend(x),
        [
            0.06433823529411731,
            -0.8588235294117651,
            -1.7819852941176473,
            1.2948529411764702,
            -0.12830882352941209,
            2.4485294117647056,
            -2.474632352941177,
            0.6022058823529409,
            0.42904411764705863,
            -0.7441176470588238,
            1.332720588235294,
            2.4095588235294114,
            -1.0136029411764707,
            0.06323529411764683,
            1.1400735294117645,
            -2.7830882352941178,
        ],
    )
    _assert_close(
        detrend(x, "constant"),
        [
            0.640625,
            -0.359375,
            -1.359375,
            1.640625,
            0.140625,
            2.640625,
            -2.359375,
            0.640625,
            0.390625,
            -0.859375,
            1.140625,
            2.140625,
            -1.359375,
            -0.359375,
            0.640625,
            -3.359375,
        ],
    )


def test_savgol_filter_matches_scipy_in_every_mode() raises:
    """`savgol_filter(x, 5, 2, mode)` for all five modes: the interior is
    one set of coefficients, and only the two edge samples on each side
    differ between modes -- `"interp"`'s come from SciPy's edge
    polynomials."""
    var x = _from[16](_x())
    var interior: List[Float64] = [
        0.07142857142857162,
        0.5428571428571424,
        2.2142857142857135,
        0.6857142857142849,
        0.29285714285714315,
        -0.15714285714285725,
        0.5785714285714283,
        0.2285714285714286,
        1.435714285714285,
        1.428571428571428,
        0.15714285714285736,
        0.04285714285714284,
    ]
    var edges: List[List[Float64]] = [
        [
            0.7285714285714286,
            0.1857142857142855,
            -0.9285714285714295,
            -2.0428571428571445,
        ],
        [
            0.8285714285714281,
            -0.257142857142857,
            -0.20000000000000018,
            -1.8857142857142846,
        ],
        [
            0.6571428571428567,
            -0.17142857142857135,
            -0.5428571428571429,
            -0.7714285714285707,
        ],
        [
            -0.5428571428571429,
            0.08571428571428567,
            -0.5428571428571429,
            -0.7714285714285707,
        ],
        [
            0.5714285714285711,
            -0.17142857142857135,
            -0.4571428571428572,
            -1.1142857142857134,
        ],
    ]
    var modes: List[StaticString] = [
        "interp",
        "nearest",
        "mirror",
        "wrap",
        "constant",
    ]
    for m in range(5):
        var got = savgol_filter[window_length=5, polyorder=2](
            x, mode=modes[m]
        ).to_host()
        for i in range(2, 14):
            assert_almost_equal(Float64(got[i]), interior[i - 2], atol=1e-12)
        assert_almost_equal(Float64(got[0]), edges[m][0], atol=1e-12)
        assert_almost_equal(Float64(got[1]), edges[m][1], atol=1e-12)
        assert_almost_equal(Float64(got[14]), edges[m][2], atol=1e-12)
        assert_almost_equal(Float64(got[15]), edges[m][3], atol=1e-12)


def test_savgol_filter_derivatives_match_scipy() raises:
    """First derivative with `delta = 0.5` and second with `delta = 2`, in
    `"interp"` mode: the `deriv! / delta^deriv` scale on both the interior
    coefficients and the edge polynomials."""
    var x = _from[16](_x())
    _assert_close(
        savgol_filter[window_length=5, polyorder=2, deriv=1](x, delta=0.5),
        [
            -1.514285714285715,
            -0.6571428571428575,
            0.19999999999999957,
            1.4999999999999998,
            -0.1999999999999984,
            -0.8999999999999992,
            -0.2999999999999999,
            -0.8500000000000004,
            1.1000000000000003,
            0.7499999999999996,
            -0.09999999999999926,
            -0.29999999999999855,
            -0.6999999999999997,
            -1.7999999999999994,
            -2.085714285714287,
            -2.3714285714285737,
        ],
    )
    _assert_close(
        savgol_filter[window_length=5, polyorder=2, deriv=2](x, delta=2.0),
        [
            0.10714285714285718,
            0.10714285714285718,
            0.10714285714285703,
            0.08928571428571423,
            -0.42857142857142877,
            0.053571428571428575,
            0.08928571428571411,
            0.1517857142857143,
            -0.10714285714285715,
            0.20535714285714268,
            -0.1964285714285715,
            -0.23214285714285718,
            0.16071428571428556,
            -0.035714285714285705,
            -0.03571428571428583,
            -0.03571428571428583,
        ],
    )
    _assert_close(
        savgol_filter[window_length=7, polyorder=3](x, mode="nearest"),
        [
            0.3333333333333318,
            0.2857142857142872,
            0.07142857142856336,
            1.0476190476190577,
            1.071428571428569,
            1.0238095238095184,
            0.5119047619047729,
            0.14285714285714257,
            -0.20238095238095388,
            1.2619047619047565,
            0.9404761904761972,
            0.7380952380952372,
            1.0714285714285834,
            0.07142857142859696,
            -1.0476190476190221,
            -1.6190476190476082,
        ],
        atol=1e-11,
    )


def test_resample_matches_scipy() raises:
    """Up to 24, down to 10 and to an odd 11 from 16 samples, and to 20 and
    9 from 15: every branch of SciPy's Nyquist-bin handling."""
    var x = _from[16](_x())
    _assert_close(
        resample[num=24](x),
        [
            0.9999999999999998,
            1.5026843387276145,
            -1.5007471181728054,
            -0.9999999999999998,
            1.9110661078357554,
            1.1519336073694553,
            0.5,
            2.8497724361214063,
            1.6735146146141506,
            -1.9999999999999996,
            -0.7371550542034774,
            2.0008966360487275,
            0.7499999999999998,
            -0.7353560095034887,
            0.11751054191426243,
            1.4999999999999993,
            2.571498174804808,
            1.70297832107615,
            -1.0,
            -1.2108507653455327,
            1.2409719616443922,
            1.0000000000000002,
            -2.214159228437086,
            -2.4495585644943327,
        ],
        atol=1e-11,
    )
    _assert_close(
        resample[num=10](x),
        [
            -0.5465739474407585,
            0.06636258695672688,
            0.9525269536368551,
            1.5886396664310742,
            -0.35292697212296553,
            0.49755699447438984,
            0.7870145843627658,
            1.4086391020532547,
            -0.29175295662649914,
            -0.5157360117248432,
        ],
        atol=1e-11,
    )
    _assert_close(
        resample[num=11](x),
        [
            -0.5465739474407585,
            0.31063657085559954,
            0.11737224779464513,
            2.584756125078203,
            -0.7727247829460531,
            0.9775112776104574,
            -0.36087216619062484,
            2.0621991145040273,
            0.31273848420818934,
            0.0664818200573282,
            -0.798399743531014,
        ],
        atol=1e-11,
    )
    var xs = _x()
    var short = List[Float64]()
    for i in range(15):
        short.append(xs[i])
    var x15 = _from[15](short)
    _assert_close(
        resample[num=20](x15),
        [
            1.0,
            0.9309272395655175,
            -1.6998380256282024,
            0.15669184430127833,
            2.0,
            0.32455694435793797,
            2.2533426789679054,
            2.163959296280286,
            -2.0000000000000004,
            -0.3386545151081683,
            2.1134424369107574,
            -0.13097149044180742,
            -0.5000000000000001,
            1.0562642107578621,
            2.306739379339502,
            2.0242905130429167,
            -0.9999999999999997,
            -0.9328155877727664,
            1.359646863743371,
            0.5790848783502768,
        ],
        atol=1e-11,
    )
    _assert_close(
        resample[num=9](x15),
        [
            1.4262041708982405,
            -0.9493503899823117,
            1.853922353700473,
            0.7548148117844631,
            0.13867107411060903,
            0.17530207806271794,
            1.388079317879801,
            0.7228469041984927,
            -0.2604903206524862,
        ],
        atol=1e-11,
    )


def test_firwin_matches_scipy() raises:
    """Lowpass, highpass (`pass_zero=False`), bandpass, bandstop, a
    three-edge multiband, an even-length lowpass and a Hann-windowed
    design, each against `scipy.signal.firwin`."""
    _assert_close(
        firwin[dtype, 7]([0.3]),
        [
            0.003296613269668359,
            0.058973232008108376,
            0.2492098913464448,
            0.37704052675155686,
            0.24920989134644486,
            0.05897323200810844,
            0.003296613269668359,
        ],
    )
    _assert_close(
        firwin[dtype, 7]([0.3], pass_zero=False),
        [
            -0.0026022584415487677,
            -0.04655189379673665,
            -0.19671963024615807,
            0.6944600102180596,
            -0.19671963024615813,
            -0.0465518937967367,
            -0.0026022584415487677,
        ],
    )
    _assert_close(
        firwin[dtype, 7]([0.2, 0.5], pass_zero=False),
        [
            -0.034530147819915404,
            -0.09783581770694454,
            0.2106561322189544,
            0.6255052841316858,
            0.2106561322189545,
            -0.09783581770694463,
            -0.034530147819915404,
        ],
    )
    _assert_close(
        firwin[dtype, 7]([0.2, 0.5]),
        [
            0.026501880312347494,
            0.07508896702824608,
            -0.16167853182223804,
            1.120175368963289,
            -0.16167853182223813,
            0.07508896702824616,
            0.026501880312347494,
        ],
    )
    _assert_close(
        firwin[dtype, 9]([0.2, 0.4, 0.6]),
        [
            0.020850557642848235,
            0.02850267684338193,
            -0.025381263732310703,
            0.21294917533576402,
            0.5261577078206331,
            0.21294917533576402,
            -0.025381263732310703,
            0.02850267684338193,
            0.020850557642848235,
        ],
    )
    _assert_close(
        firwin[dtype, 8]([0.3]),
        [
            -0.001316875432556324,
            0.02637484405478644,
            0.15577480584724548,
            0.3191672255305244,
            0.3191672255305244,
            0.15577480584724548,
            0.02637484405478644,
            -0.001316875432556324,
        ],
    )
    _assert_close(
        firwin[dtype, 7]([0.3], window="hann"),
        [
            0.0,
            0.04966316431031117,
            0.25347606519531835,
            0.39372154098874085,
            0.2534760651953184,
            0.04966316431031124,
            0.0,
        ],
    )
    var raised = False
    try:
        _ = firwin[dtype, 8]([0.3], pass_zero=False)
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
