"""`peak_widths` and `argrelmax` at `gpu=True`, against the host, at
`float32`: one thread per peak running the same walk, and one flag per
sample packed by the device compaction, so the indices match exactly and
the widths to `float32` rounding of the interpolation."""

from std.math import cos, sin
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.testing import assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.signal import argrelmax, find_peaks, peak_widths

comptime f32 = DType.float32
comptime n = 2000


def _x(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        var t = Float64(i)
        values.append(
            Float32(sin(0.1 * t) * (1 + 0.001 * t) + 0.3 * cos(0.37 * t))
        )
    return Static[f32, n](values^, ctx)


def test_peak_widths_and_argrelmax_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var dm = argrelmax[gpu=True](_x(gpu))
    assert_false(dm.on_host())
    var got = dm.to_host()
    var want = argrelmax(_x(cpu)).to_host()
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])
    var dw = peak_widths[gpu=True](_x(gpu), find_peaks[gpu=True](_x(gpu)))
    var hw = peak_widths(_x(cpu), find_peaks(_x(cpu)))
    var a = dw.widths.to_host()
    var b = hw.widths.to_host()
    assert_equal(len(a), len(b))
    for i in range(len(b)):
        assert_almost_equal(a[i], b[i], atol=1e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
