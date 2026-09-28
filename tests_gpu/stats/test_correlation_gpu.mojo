"""`pearsonr` and `linregress` at `gpu=True`, against the host.

On the device the means and centered moments are `float32` sums, so the
statistics agree with the host's `Float64` ones to `float32` precision;
the p-values go through the same host `t` tail.
"""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import linregress, pearsonr

comptime f32 = DType.float32
comptime n = 500


def _x(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(i) * 0.01 + 3.0)
    return Static[f32, n](values^, ctx)


def _y(ctx: DeviceContext) raises -> Static[f32, n]:
    """A line plus a deterministic wiggle, so `r` is well inside (-1, 1)
    and the p-value is not zero."""
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(
            Float32(0.02) * Float32(i) - 1.0 + Float32(sin(Float64(i) * 1.7))
        )
    return Static[f32, n](values^, ctx)


def test_pearsonr_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = pearsonr[gpu=True](_x(gpu), _y(gpu))
    var h = pearsonr(_x(cpu), _y(cpu))
    assert_almost_equal(d.statistic, h.statistic, rtol=1e-4)
    assert_almost_equal(d.pvalue, h.pvalue, rtol=1e-2, atol=1e-30)


def test_linregress_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = linregress[gpu=True](_x(gpu), _y(gpu))
    var h = linregress(_x(cpu), _y(cpu))
    assert_almost_equal(d.slope, h.slope, rtol=1e-4)
    assert_almost_equal(d.intercept, h.intercept, rtol=1e-4, atol=1e-4)
    assert_almost_equal(d.rvalue, h.rvalue, rtol=1e-4)
    assert_almost_equal(d.stderr, h.stderr, rtol=1e-3)
    assert_almost_equal(d.intercept_stderr, h.intercept_stderr, rtol=1e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
