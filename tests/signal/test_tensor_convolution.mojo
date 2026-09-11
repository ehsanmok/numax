"""Tests for `numax.signal` over `Tensor`: `convolve`, `correlate`,
`fftconvolve` in NumPy's three modes, and the window factories.

Every expected value is NumPy's or SciPy's own: `numpy.convolve` and
`numpy.correlate` on a 7-by-3 pair in `full`, `same` and `valid`,
`scipy.signal.fftconvolve` on the same, and `scipy.signal.windows` /
`get_window` for the symmetric and periodic forms of each window.
"""

from std.testing import TestSuite, assert_almost_equal, assert_true

from max.gpu.host import DeviceContext

from numax.core.array import Static
from numax.signal import (
    bartlett,
    blackman,
    boxcar,
    convolve,
    correlate,
    fftconvolve,
    full,
    get_window,
    hamming,
    hann,
    kaiser,
    same,
    valid,
)

comptime dtype = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _from[n: Int](values: List[Float64]) raises -> Static[dtype, n]:
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype](values[i]))
    return Static[dtype, n](_cpu(), out^)


def _a() -> List[Float64]:
    return [1.0, 2.0, 0.5, -1.0, 3.0, 0.25, -2.0]


def _b() -> List[Float64]:
    return [0.5, -1.0, 2.0]


def _assert_close[
    n: Int
](got: Static[dtype, n], want: List[Float64], atol: Float64 = 1e-12) raises:
    var values = got.to_host()
    for i in range(n):
        assert_almost_equal(Float64(values[i]), want[i], atol=atol)


def test_convolve_matches_numpy_in_every_mode() raises:
    var a = _from[7](_a())
    var b = _from[3](_b())
    _assert_close(
        convolve(a, b), [0.5, 0.0, 0.25, 3.0, 3.5, -4.875, 4.75, 2.5, -4.0]
    )
    _assert_close(
        convolve[mode=same](a, b), [0.0, 0.25, 3.0, 3.5, -4.875, 4.75, 2.5]
    )
    _assert_close(convolve[mode=valid](a, b), [0.25, 3.0, 3.5, -4.875, 4.75])


def test_convolve_is_symmetric_in_its_arguments() raises:
    """`convolve(a, b) == convolve(b, a)`, including `same`, whose length is
    that of the longer input whichever position it is in."""
    var a = _from[7](_a())
    var b = _from[3](_b())
    var ab = convolve(a, b).to_host()
    var ba = convolve(b, a).to_host()
    for i in range(9):
        assert_almost_equal(Float64(ab[i]), Float64(ba[i]), atol=1e-15)
    _assert_close(
        convolve[mode=same](b, a), [0.0, 0.25, 3.0, 3.5, -4.875, 4.75, 2.5]
    )


def test_correlate_matches_numpy_in_every_mode() raises:
    """`numpy.correlate(a, b, mode)`: the zero-lag term of `full` is at
    index `k - 1 = 2`, which is `7.25` here, not index 0."""
    var a = _from[7](_a())
    var b = _from[3](_b())
    _assert_close(
        correlate(a, b), [2.0, 3.0, -0.5, -1.5, 7.25, -3.0, -2.75, 2.125, -1.0]
    )
    _assert_close(
        correlate[mode=same](a, b), [3.0, -0.5, -1.5, 7.25, -3.0, -2.75, 2.125]
    )
    _assert_close(correlate[mode=valid](a, b), [-0.5, -1.5, 7.25, -3.0, -2.75])


def test_correlate_is_convolve_with_the_taps_reversed() raises:
    var a = _from[7](_a())
    var b = _from[3](_b())
    var reversed_b = _from[3]([2.0, -1.0, 0.5])
    var direct = correlate(a, b).to_host()
    var via_convolve = convolve(a, reversed_b).to_host()
    for i in range(9):
        assert_almost_equal(
            Float64(direct[i]), Float64(via_convolve[i]), atol=1e-15
        )


def test_fftconvolve_agrees_with_convolve_in_every_mode() raises:
    """The transform route and the direct sum are one answer to rounding,
    and the transform route matches `scipy.signal.fftconvolve`."""
    var a = _from[7](_a())
    var b = _from[3](_b())
    _assert_close(
        fftconvolve(a, b), [0.5, 0.0, 0.25, 3.0, 3.5, -4.875, 4.75, 2.5, -4.0]
    )
    _assert_close(
        fftconvolve[mode=same](a, b), [0.0, 0.25, 3.0, 3.5, -4.875, 4.75, 2.5]
    )
    _assert_close(fftconvolve[mode=valid](a, b), [0.25, 3.0, 3.5, -4.875, 4.75])


def test_fftconvolve_of_long_inputs_agrees_with_the_direct_sum() raises:
    """`m + k - 1 = 47` pads to `64`: a length where the padding is not
    trivial and every wrapped term would land on a real output if the
    padding were short."""
    var xs = List[Float64]()
    for i in range(30):
        xs.append(Float64((i * 7) % 11) - 4.5)
    var ks = List[Float64]()
    for i in range(18):
        ks.append(0.5 * Float64((i * 5) % 7) - 1.0)
    var a = _from[30](xs)
    var k = _from[18](ks)
    var direct = convolve(a, k).to_host()
    var transformed = fftconvolve(a, k).to_host()
    for i in range(47):
        assert_almost_equal(
            Float64(transformed[i]), Float64(direct[i]), atol=1e-11
        )


def test_windows_match_scipy_symmetric() raises:
    """`scipy.signal.windows.<name>(8)`, `sym=True`: the ends of Hann,
    Blackman and Bartlett are zero and the window is mirror-symmetric."""
    _assert_close(
        hann[dtype, 8](),
        [
            0.0,
            0.18825509907063326,
            0.6112604669781572,
            0.9504844339512095,
            0.9504844339512095,
            0.6112604669781572,
            0.18825509907063326,
            0.0,
        ],
    )
    _assert_close(
        hamming[dtype, 8](),
        [
            0.08000000000000007,
            0.25319469114498266,
            0.6423596296199047,
            0.9544456792351128,
            0.9544456792351128,
            0.6423596296199047,
            0.25319469114498266,
            0.08000000000000007,
        ],
    )
    _assert_close(
        blackman[dtype, 8](),
        [
            0.0,
            0.09045342435412808,
            0.45918295754596367,
            0.9203636180999082,
            0.9203636180999082,
            0.45918295754596367,
            0.09045342435412808,
            0.0,
        ],
        atol=1e-15,
    )
    _assert_close(
        bartlett[dtype, 8](),
        [
            0.0,
            0.2857142857142857,
            0.5714285714285714,
            0.8571428571428571,
            0.8571428571428572,
            0.5714285714285714,
            0.2857142857142858,
            0.0,
        ],
    )
    _assert_close(boxcar[dtype, 4](), [1.0, 1.0, 1.0, 1.0])


def test_windows_match_scipy_periodic() raises:
    """`scipy.signal.get_window(name, 8)`, periodic: the peak is exactly
    `1` at `i = n/2` and the last sample is not the first's mirror."""
    _assert_close(
        hann[dtype, 8](sym=False),
        [
            0.0,
            0.14644660940672627,
            0.5,
            0.8535533905932737,
            1.0,
            0.8535533905932737,
            0.5,
            0.14644660940672627,
        ],
    )
    _assert_close(
        get_window[dtype, 8]("hamming"),
        [
            0.08000000000000007,
            0.21473088065418822,
            0.54,
            0.865269119345812,
            1.0,
            0.865269119345812,
            0.54,
            0.21473088065418822,
        ],
    )
    _assert_close(
        get_window[dtype, 8]("bartlett"),
        [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25],
    )
    _assert_close(
        get_window[dtype, 8]("blackman"),
        [
            0.0,
            0.06644660940672624,
            0.34,
            0.7735533905932738,
            0.9999999999999999,
            0.7735533905932738,
            0.34,
            0.06644660940672624,
        ],
        atol=1e-15,
    )


def test_kaiser_matches_scipy() raises:
    """`kaiser(8, 14)` symmetric and `get_window(("kaiser", 5), 8)`
    periodic -- `I_0` by its series against SciPy's."""
    _assert_close(
        kaiser[dtype, 8](14.0),
        [
            7.726866835270368e-06,
            0.017964073497790785,
            0.2727720681863426,
            0.8708037315729595,
            0.8708037315729595,
            0.2727720681863426,
            0.017964073497790785,
            7.726866835270368e-06,
        ],
        atol=1e-14,
    )
    _assert_close(
        get_window[dtype, 8]("kaiser", beta=5.0),
        [
            0.036710892271286676,
            0.23054433409868888,
            0.5528517696991324,
            0.868017159478478,
            1.0,
            0.868017159478478,
            0.5528517696991324,
            0.23054433409868888,
        ],
        atol=1e-14,
    )


def test_get_window_names_the_factories() raises:
    """`get_window(name, n, fftbins=False)` is the symmetric factory of that
    name; an unknown name raises."""
    var named = get_window[dtype, 8]("hann", fftbins=False).to_host()
    var direct = hann[dtype, 8]().to_host()
    for i in range(8):
        assert_almost_equal(Float64(named[i]), Float64(direct[i]), atol=1e-15)
    var raised = False
    try:
        _ = get_window[dtype, 8]("tukey")
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
