"""The order statistics at `gpu=True`, against the host.

The device sorts and gathers the same order statistics the host selects,
and blends them with the same `Float64` weights, so every answer is the
host's exactly: `assert_equal`, for every method NumPy names, at odd and
even counts, with ties, and with NaN propagated or dropped.
"""

from std.math import sin
from std.testing import TestSuite, assert_equal, assert_true
from std.utils.numerics import nan

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import (
    iqr,
    median,
    nanmedian,
    nanpercentile,
    nanquantile,
    percentile,
    quantile,
)

comptime f32 = DType.float32

comptime METHODS: List[StaticString] = [
    "linear",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
]


def _sample[
    n: Int
](ctx: DeviceContext, with_nan: Bool) raises -> Static[f32, n]:
    """Scrambled values with repeats (every seventh value is `2.0`), and
    two NaNs when asked."""
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        if with_nan and (i == 3 or i == n - 2):
            values.append(nan[f32]())
        elif i % 7 == 0:
            values.append(2.0)
        else:
            values.append(Float32(sin(Float64(i) * 12.9898)) * 10.0)
    return Static[f32, n](values^, ctx)


def _check_all[n: Int]() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    comptime for k in range(len(METHODS)):
        comptime method = METHODS[k]
        for q in [0.0, 0.1, 0.25, 0.5, 0.73, 1.0]:
            assert_equal(
                quantile[gpu=True](_sample[n](gpu, False), q, method),
                quantile(_sample[n](cpu, False), q, method),
            )
            assert_equal(
                nanquantile[gpu=True](_sample[n](gpu, True), q, method),
                nanquantile(_sample[n](cpu, True), q, method),
            )


def test_every_method_matches_the_host() raises:
    _check_all[101]()
    _check_all[64]()


def test_median_iqr_and_percentiles_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    comptime n = 1000
    assert_equal(
        median[gpu=True](_sample[n](gpu, False)),
        median(_sample[n](cpu, False)),
    )
    assert_equal(
        nanmedian[gpu=True](_sample[n](gpu, True)),
        nanmedian(_sample[n](cpu, True)),
    )
    assert_equal(
        iqr[gpu=True](_sample[n](gpu, False)), iqr(_sample[n](cpu, False))
    )
    assert_equal(
        iqr[gpu=True](_sample[n](gpu, True), nan_policy="omit"),
        iqr(_sample[n](cpu, True), nan_policy="omit"),
    )
    assert_equal(
        percentile[gpu=True](_sample[n](gpu, False), 37.5),
        percentile(_sample[n](cpu, False), 37.5),
    )
    assert_equal(
        nanpercentile[gpu=True](_sample[n](gpu, True), 90.0),
        nanpercentile(_sample[n](cpu, True), 90.0),
    )


def test_vector_quantiles_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    comptime n = 777
    var qg = Static[f32, 4]([0.05, 0.5, 0.9, 0.99], gpu)
    var qc = Static[f32, 4]([0.05, 0.5, 0.9, 0.99], cpu)
    var d = quantile[gpu=True](_sample[n](gpu, False), qg).to_host()
    var h = quantile(_sample[n](cpu, False), qc).to_host()
    for i in range(4):
        assert_equal(d[i], h[i])
    var pg = Static[f32, 2]([25.0, 75.0], gpu)
    var pc = Static[f32, 2]([25.0, 75.0], cpu)
    var dp = percentile[gpu=True](_sample[n](gpu, False), pg).to_host()
    var hp = percentile(_sample[n](cpu, False), pc).to_host()
    for i in range(2):
        assert_equal(dp[i], hp[i])
    var dn = nanquantile[gpu=True](_sample[n](gpu, True), qg).to_host()
    var hn = nanquantile(_sample[n](cpu, True), qc).to_host()
    for i in range(4):
        assert_equal(dn[i], hn[i])


def test_nan_propagates_on_the_device() raises:
    var gpu = DeviceContext()
    var got = quantile[gpu=True](_sample[50](gpu, True), 0.5)
    assert_true(got != got)
    var m = median[gpu=True](_sample[50](gpu, True))
    assert_true(m != m)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
