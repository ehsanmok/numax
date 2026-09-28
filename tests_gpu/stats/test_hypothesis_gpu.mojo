"""The `t` tests, `chisquare` and `f_oneway` at `gpu=True`, against the
host.

Their sums run on the device at `float32` and reassociated, so each
statistic agrees with the host's `Float64` one to `float32` precision;
the p-values go through the same host tails.
"""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import chisquare, f_oneway, ttest_1samp, ttest_ind, ttest_rel

comptime f32 = DType.float32
comptime n = 400


def _sample(
    ctx: DeviceContext, shift: Float32, seed: Float64
) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(shift + Float32(sin(Float64(i) * seed)) * 2.0)
    return Static[f32, n](values^, ctx)


def _counts(ctx: DeviceContext, bump: Bool) raises -> Static[f32, 12]:
    var values = List[Scalar[f32]](capacity=12)
    for i in range(12):
        values.append(
            Float32(40 + (i * 7) % 11 + (5 if bump and i == 3 else 0))
        )
    return Static[f32, 12](values^, ctx)


def _close(d: Float64, h: Float64) raises:
    assert_almost_equal(d, h, rtol=1e-3, atol=1e-6)


def test_t_tests_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    for alternative in [
        StaticString("two-sided"),
        StaticString("less"),
        StaticString("greater"),
    ]:
        var d1 = ttest_1samp[gpu=True](_sample(gpu, 0.1, 1.3), 0.0, alternative)
        var h1 = ttest_1samp(_sample(cpu, 0.1, 1.3), 0.0, alternative)
        _close(d1.statistic, h1.statistic)
        _close(d1.pvalue, h1.pvalue)
        for equal_var in [True, False]:
            var d2 = ttest_ind[gpu=True](
                _sample(gpu, 0.0, 1.3),
                _sample(gpu, 0.3, 0.7),
                equal_var,
                alternative,
            )
            var h2 = ttest_ind(
                _sample(cpu, 0.0, 1.3),
                _sample(cpu, 0.3, 0.7),
                equal_var,
                alternative,
            )
            _close(d2.statistic, h2.statistic)
            _close(d2.pvalue, h2.pvalue)
            _close(d2.df, h2.df)
        var d3 = ttest_rel[gpu=True](
            _sample(gpu, 0.0, 1.3), _sample(gpu, 0.2, 0.7), alternative
        )
        var h3 = ttest_rel(
            _sample(cpu, 0.0, 1.3), _sample(cpu, 0.2, 0.7), alternative
        )
        _close(d3.statistic, h3.statistic)
        _close(d3.pvalue, h3.pvalue)


def test_chisquare_and_anova_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d1 = chisquare[gpu=True](_counts(gpu, True))
    var h1 = chisquare(_counts(cpu, True))
    _close(d1.statistic, h1.statistic)
    _close(d1.pvalue, h1.pvalue)
    var d2 = chisquare[gpu=True](_counts(gpu, True), _counts(gpu, True))
    var h2 = chisquare(_counts(cpu, True), _counts(cpu, True))
    _close(d2.statistic, h2.statistic)
    var d3 = f_oneway[gpu=True](
        _sample(gpu, 0.0, 1.3), _sample(gpu, 0.2, 0.7), _sample(gpu, 0.1, 2.1)
    )
    var h3 = f_oneway(
        _sample(cpu, 0.0, 1.3), _sample(cpu, 0.2, 0.7), _sample(cpu, 0.1, 2.1)
    )
    _close(d3.statistic, h3.statistic)
    _close(d3.pvalue, h3.pvalue)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
