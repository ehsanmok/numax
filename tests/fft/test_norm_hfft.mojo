"""Tests for `norm=` across `numax.fft`'s transforms, and `hfft`/`ihfft`.

`norm` against NumPy's three modes on `fft`/`ifft`, the claim that
`"ortho"` makes the pair unitary (the round trip is exact and energy is
kept), and that every spelling agrees with `"backward"` scaled by hand
on the multi-axis and real forms. `hfft`/`ihfft` against NumPy, at even
and odd output length, and as each other's inverse.
"""

from std.math import sqrt
from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static, copy, zeros
from numax.fft import (
    fft,
    fft2,
    fftn,
    hfft,
    ifft,
    ifftn,
    ihfft,
    irfft,
    rfft,
    rfftn,
)

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _x() raises -> Static[f64, 8]:
    return Static[f64, 8]([1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 0.0, 1.5], _cpu())


def _close(got: List[Float64], want: List[Float64]) raises:
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-12)


def test_fft_norms_match_numpy() raises:
    # numpy: np.fft.fft(x, norm=...).real[:3] and np.fft.ifft likewise
    var b = fft((_x(), zeros[f64, 8](_cpu())))[0].to_host()
    var o = fft[norm="ortho"]((_x(), zeros[f64, 8](_cpu())))[0].to_host()
    var f = fft[norm="forward"]((_x(), zeros[f64, 8](_cpu())))[0].to_host()
    _close(b, [5.0, 1.5355339059327378, 5.0])
    _close(o, [1.7677669529663687, 0.5428932188134524, 1.7677669529663687])
    _close(f, [0.625, 0.19194173824159222, 0.625])
    var ib = ifft((_x(), zeros[f64, 8](_cpu())))[0].to_host()
    var io = ifft[norm="ortho"]((_x(), zeros[f64, 8](_cpu())))[0].to_host()
    var i_f = ifft[norm="forward"]((_x(), zeros[f64, 8](_cpu())))[0].to_host()
    _close(ib, [0.625, 0.19194173824159222, 0.625])
    _close(io, [1.7677669529663687, 0.5428932188134524, 1.7677669529663687])
    _close(i_f, [5.0, 1.5355339059327378, 5.0])


def test_ortho_is_unitary() raises:
    var spectrum = fft[norm="ortho"]((_x(), zeros[f64, 8](_cpu())))
    var re = spectrum[0].to_host()
    var im = spectrum[1].to_host()
    var x = _x().to_host()
    var energy_x = 0.0
    var energy_s = 0.0
    for i in range(8):
        energy_x += Float64(x[i]) ** 2
        energy_s += Float64(re[i]) ** 2 + Float64(im[i]) ** 2
    assert_almost_equal(energy_s, energy_x, atol=1e-12)
    var back = ifft[norm="ortho"](spectrum^)[0].to_host()
    _close(back, x)


def test_norm_on_the_real_and_multi_axis_forms() raises:
    var x = _x().to_host()
    var half = rfft[norm="forward"](_x())[0].to_host()
    var plain = rfft(_x())[0].to_host()
    for i in range(5):
        assert_almost_equal(half[i], plain[i] / 8.0, atol=1e-12)
    _close(irfft[norm="forward"](rfft[norm="forward"](_x())).to_host(), x)
    _close(irfft[norm="ortho"](rfft[norm="ortho"](_x())).to_host(), x)
    var m = Static[f64, 2, 4](
        [1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 0.0, 1.5], _cpu()
    )
    var o2 = fft2[norm="ortho"]((copy(m), zeros[f64, 2, 4](_cpu())))[
        0
    ].to_host()
    var b2 = fft2((copy(m), zeros[f64, 2, 4](_cpu())))[0].to_host()
    var on = fftn[norm="ortho"]((copy(m), zeros[f64, 2, 4](_cpu())))[
        0
    ].to_host()
    for i in range(8):
        assert_almost_equal(o2[i], b2[i] / sqrt(8.0), atol=1e-12)
        assert_almost_equal(on[i], o2[i], atol=1e-12)
    var back = ifftn[norm="forward"](
        fftn[norm="forward"]((copy(m), zeros[f64, 2, 4](_cpu())))
    )[0].to_host()
    _close(back, x)
    var v = Static[f64, 2, 2, 2](
        [1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 0.0, 1.5], _cpu()
    )
    var rv = rfftn[norm="forward"](copy(v))[0].to_host()
    var rb = rfftn(copy(v))[0].to_host()
    for i in range(len(rb)):
        assert_almost_equal(rv[i], rb[i] / 8.0, atol=1e-12)


def test_hfft_and_ihfft_match_numpy() raises:
    # numpy: h = [1, 2-1j, 0.5+0.5j, -1, 0.25j]; np.fft.hfft(h), hfft(h, 7)
    var h_re = Static[f64, 5]([1.0, 2.0, 0.5, -1.0, 0.0], _cpu())
    var h_im = Static[f64, 5]([0.0, -1.0, 0.5, 0.0, 0.25], _cpu())
    var even = hfft((copy(h_re), copy(h_im))).to_host()
    _close(
        even,
        [
            4.0,
            4.82842712474619,
            -2.0,
            -5.656854249492381,
            0.0,
            -0.8284271247461898,
            2.0,
            5.656854249492381,
        ],
    )
    # NumPy truncates `h` to `7 // 2 + 1 = 4` samples for `n = 7`; numax
    # takes exactly those four.
    var h4_re = Static[f64, 4]([1.0, 2.0, 0.5, -1.0], _cpu())
    var h4_im = Static[f64, 4]([0.0, -1.0, 0.5, 0.0], _cpu())
    var odd = hfft[n=7]((h4_re^, h4_im^)).to_host()
    _close(
        odd,
        [
            4.0,
            4.484640956529222,
            -4.421771770926348,
            -3.1849427625414606,
            0.11425515886483151,
            0.3457073560360615,
            5.662111062037694,
        ],
    )
    # numpy: np.fft.ihfft([1, 2, -1, 0.5, 3, -2, 0, 1.5])
    var ih = ihfft(_x())
    _close(
        ih[0].to_host(),
        [0.625, 0.19194173824159216, 0.625, -0.6919417382415922, 0.125],
    )
    _close(
        ih[1].to_host(),
        [0.0, 0.14016504294495535, -0.25, 0.39016504294495535, 0.0],
    )
    _close(hfft(ihfft(_x())).to_host(), _x().to_host())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
