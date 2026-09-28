"""Windows and frequency grids filled on a device context, against the
host.

Given a GPU context, each factory evaluates its formula in one launch at
the tensor's dtype instead of uploading a host table. Each test checks
the result is on the device and matches the host's `Float64` evaluation
at `float32` tolerance, for the symmetric and periodic forms.
"""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.fft import fftfreq, rfftfreq
from numax.signal import (
    bartlett,
    blackman,
    boxcar,
    get_window,
    hamming,
    hann,
    kaiser,
)

comptime f32 = DType.float32
comptime n = 65


def _assert_close(got: List[Scalar[f32]], want: List[Scalar[f32]]) raises:
    for i in range(len(want)):
        assert_almost_equal(
            Float64(got[i]), Float64(want[i]), atol=2e-6, rtol=2e-5
        )


def _check(mut d: Static[f32, n], mut h: Static[f32, n]) raises:
    assert_false(d.on_host())
    _assert_close(d.to_host(), h.to_host())


def test_windows_fill_on_the_device() raises:
    var gpu = Optional(DeviceContext())
    var cpu = Optional(DeviceContext(api="cpu"))
    for sym in [True, False]:
        var d0 = hann[f32, n](sym, gpu)
        var h0 = hann[f32, n](sym, cpu)
        _check(d0, h0)
        var d1 = hamming[f32, n](sym, gpu)
        var h1 = hamming[f32, n](sym, cpu)
        _check(d1, h1)
        var d2 = blackman[f32, n](sym, gpu)
        var h2 = blackman[f32, n](sym, cpu)
        _check(d2, h2)
        var d3 = bartlett[f32, n](sym, gpu)
        var h3 = bartlett[f32, n](sym, cpu)
        _check(d3, h3)
        var d4 = boxcar[f32, n](sym, gpu)
        var h4 = boxcar[f32, n](sym, cpu)
        _check(d4, h4)
        for beta in [0.0, 5.0, 14.0]:
            var d5 = kaiser[f32, n](beta, sym, gpu)
            var h5 = kaiser[f32, n](beta, sym, cpu)
            _check(d5, h5)
    var dg = get_window[f32, n]("hann", ctx=gpu)
    var hg = get_window[f32, n]("hann", ctx=cpu)
    _check(dg, hg)


def test_frequency_grids_fill_on_the_device() raises:
    var gpu = Optional(DeviceContext())
    var cpu = Optional(DeviceContext(api="cpu"))
    var d = fftfreq[f32, n](Float32(0.01), gpu)
    assert_false(d.on_host())
    _assert_close(d.to_host(), fftfreq[f32, n](Float32(0.01), cpu).to_host())
    var e = fftfreq[f32, 64](Float32(0.5), gpu)
    _assert_close(e.to_host(), fftfreq[f32, 64](Float32(0.5), cpu).to_host())
    var r = rfftfreq[f32, n](Float32(0.01), gpu)
    assert_false(r.on_host())
    _assert_close(r.to_host(), rfftfreq[f32, n](Float32(0.01), cpu).to_host())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
