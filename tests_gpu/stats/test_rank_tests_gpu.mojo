"""`mannwhitneyu`, `wilcoxon` and `ks_2samp` at `gpu=True`, against the
host.

The device ranks and counts the same values the host does: rank sums
are half-integers and the KS gaps exact integer numerators, so the
statistics match the host to `float32` precision or better, with ties,
with zero differences dropped, and under all three alternatives.
"""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import ks_2samp, mannwhitneyu, wilcoxon

comptime f32 = DType.float32


def _a[n: Int](ctx: DeviceContext) raises -> Static[f32, n]:
    """Rounded to quarters, so ties are common."""
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(Int(sin(Float64(i) * 1.7) * 12.0)) * 0.25)
    return Static[f32, n](ctx, values^)


def _b[n: Int](ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(
            Float32(Int(sin(Float64(i) * 0.9 + 0.3) * 12.0)) * 0.25 + 0.25
        )
    return Static[f32, n](ctx, values^)


def _close(d: Float64, h: Float64) raises:
    assert_almost_equal(d, h, rtol=1e-5, atol=1e-9)


def test_rank_tests_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    for alternative in [
        StaticString("two-sided"),
        StaticString("less"),
        StaticString("greater"),
    ]:
        var d1 = mannwhitneyu[gpu=True](_a[150](gpu), _b[97](gpu), alternative)
        var h1 = mannwhitneyu(_a[150](cpu), _b[97](cpu), alternative)
        _close(d1.statistic, h1.statistic)
        _close(d1.pvalue, h1.pvalue)
        var d2 = wilcoxon[gpu=True](_a[200](gpu), _b[200](gpu), alternative)
        var h2 = wilcoxon(_a[200](cpu), _b[200](cpu), alternative)
        _close(d2.statistic, h2.statistic)
        _close(d2.pvalue, h2.pvalue)
        _close(d2.df, h2.df)
        var d3 = ks_2samp[gpu=True](_a[150](gpu), _b[97](gpu), alternative)
        var h3 = ks_2samp(_a[150](cpu), _b[97](cpu), alternative)
        _close(d3.statistic, h3.statistic)
        _close(d3.pvalue, h3.pvalue)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
