"""`lfilter`, `filtfilt`, `sosfilt` and `decimate` at `gpu=True`: the
block-parallel device recurrence against the host's sequential one, with
SciPy's `butter(4, 0.2)` and `butter(2, 0.2)` coefficients, at a length
that is not a multiple of the block."""

from std.math import sin
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
)

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.signal import decimate, filtfilt, lfilter, sosfilt

comptime f32 = DType.float32
comptime n = 5003


def _signal(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        var t = Float32(i)
        values.append(
            sin(t * 0.05) + 0.3 * sin(t * 1.3) + Float32(i % 7) * 0.01
        )
    return Static[f32, n](values^, ctx)


def _b4(ctx: DeviceContext) raises -> Static[f32, 5]:
    var v: List[Scalar[f32]] = [
        0.004824,
        0.019297,
        0.028946,
        0.019297,
        0.004824,
    ]
    return Static[f32, 5](v^, ctx)


def _a4(ctx: DeviceContext) raises -> Static[f32, 5]:
    var v: List[Scalar[f32]] = [1.0, -2.369513, 2.313988, -1.054665, 0.187379]
    return Static[f32, 5](v^, ctx)


def _close(got: List[Scalar[f32]], want: List[Scalar[f32]]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=2e-4, rtol=1e-3)


def test_lfilter_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = lfilter[gpu=True](_b4(gpu), _a4(gpu), _signal(gpu))
    assert_false(d.on_host())
    _close(d.to_host(), lfilter(_b4(cpu), _a4(cpu), _signal(cpu)).to_host())


def test_filtfilt_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = filtfilt[gpu=True](_b4(gpu), _a4(gpu), _signal(gpu))
    assert_false(d.on_host())
    _close(d.to_host(), filtfilt(_b4(cpu), _a4(cpu), _signal(cpu)).to_host())


def test_sosfilt_and_decimate_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var sections: List[Scalar[f32]] = [
        0.067455,
        0.134911,
        0.067455,
        1.0,
        -1.142980,
        0.412802,
        1.0,
        2.0,
        1.0,
        1.0,
        -1.142980,
        0.412802,
    ]
    var d = sosfilt[gpu=True](
        Static[f32, 2, 6](sections.copy(), gpu), _signal(gpu)
    )
    assert_false(d.on_host())
    _close(
        d.to_host(),
        sosfilt(Static[f32, 2, 6](sections^, cpu), _signal(cpu)).to_host(),
    )
    var q = decimate[q=3, gpu=True](_signal(gpu))
    assert_false(q.on_host())
    _close(q.to_host(), decimate[q=3](_signal(cpu)).to_host())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
