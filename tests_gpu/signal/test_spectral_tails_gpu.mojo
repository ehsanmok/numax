"""`resample`, `hilbert`, `solve_circulant`, `istft` and `savgol_filter` at
`gpu=True`: inputs reach the transforms without a host copy, and the
reshuffle, the division, the overlap-add and the edge reads run on the
device. Against the host."""

from std.math import sin
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
)

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.linalg import solve_circulant
from numax.signal import hilbert, istft, resample, savgol_filter, stft

comptime f32 = DType.float32


def _wave[m: Int](ctx: DeviceContext) raises -> Static[f32, m]:
    var values = List[Scalar[f32]](capacity=m)
    for i in range(m):
        values.append(sin(Float32(i) * 0.3) + 0.25 * sin(Float32(i) * 0.07))
    return Static[f32, m](values^, ctx)


def _close(got: List[Scalar[f32]], want: List[Scalar[f32]]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-4, rtol=1e-4)


def test_resample_down_and_up_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var down = resample[num=96, gpu=True](_wave[128](gpu))
    assert_false(down.on_host())
    _close(down.to_host(), resample[num=96](_wave[128](cpu)).to_host())
    _close(
        resample[num=200, gpu=True](_wave[128](gpu)).to_host(),
        resample[num=200](_wave[128](cpu)).to_host(),
    )


def test_hilbert_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = hilbert[gpu=True](_wave[256](gpu))
    var h = hilbert(_wave[256](cpu))
    _close(d[0].to_host(), h[0].to_host())
    _close(d[1].to_host(), h[1].to_host())


def test_solve_circulant_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var cvals: List[Scalar[f32]] = [4.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0]
    var d = solve_circulant[gpu=True](
        Static[f32, 8](cvals.copy(), gpu), _wave[8](gpu)
    )
    assert_false(d.on_host())
    _close(
        d.to_host(),
        solve_circulant(
            Static[f32, 8](cvals.copy(), cpu), _wave[8](cpu)
        ).to_host(),
    )
    var ones = List[Scalar[f32]](length=8, fill=1.0)
    with assert_raises(contains="singular"):
        _ = solve_circulant[gpu=True](
            Static[f32, 8](ones^, gpu), Static[f32, 8](cvals^, gpu)
        )


def test_istft_round_trips_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var spectra = stft[nperseg=32, gpu=True](_wave[512](gpu), 4.0)
    var back = istft[nperseg=32, noverlap=16, gpu=True](spectra)
    assert_false(back.on_host())
    var host_spectra = stft[nperseg=32](_wave[512](cpu), 4.0)
    _close(
        back.to_host(), istft[nperseg=32, noverlap=16](host_spectra).to_host()
    )


def test_savgol_interp_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    _close(
        savgol_filter[window_length=11, polyorder=3, gpu=True](
            _wave[400](gpu)
        ).to_host(),
        savgol_filter[window_length=11, polyorder=3](_wave[400](cpu)).to_host(),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
