"""`skew`, `kurtosis`, `sem`, `gmean`, `hmean`, `entropy` and `describe`
at `gpu=True`, against the host's `Float64` walk.

The device computes at the tensor's dtype (`float32` here, which is what
Metal has) and brings back only scalars, so each answer is compared to a
relative tolerance a float32 accumulation meets.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import describe, entropy, gmean, hmean, kurtosis, sem, skew

comptime f32 = DType.float32
comptime n = 2048


def _positive(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32((i * 7919) % 101) * 0.05 + 0.25)
    return Static[f32, n](values^, ctx)


def _skewed(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        var u = Float32((i * 7919) % 101) / 101.0
        values.append(u * u * u * 4.0 - 0.3)
    return Static[f32, n](values^, ctx)


def test_the_moment_statistics_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    assert_almost_equal(
        skew[gpu=True](_skewed(gpu)), skew(_skewed(cpu)), rtol=1e-3
    )
    assert_almost_equal(
        skew[gpu=True](_skewed(gpu), False),
        skew(_skewed(cpu), False),
        rtol=1e-3,
    )
    assert_almost_equal(
        kurtosis[gpu=True](_skewed(gpu)), kurtosis(_skewed(cpu)), rtol=1e-3
    )
    assert_almost_equal(
        kurtosis[gpu=True](_skewed(gpu), False, False),
        kurtosis(_skewed(cpu), False, False),
        rtol=1e-3,
    )
    assert_almost_equal(
        sem[gpu=True](_skewed(gpu)), sem(_skewed(cpu)), rtol=1e-4
    )
    var d = describe[gpu=True](_skewed(gpu))
    var h = describe(_skewed(cpu))
    assert_equal(d.nobs, h.nobs)
    assert_almost_equal(d.min, h.min, rtol=1e-6)
    assert_almost_equal(d.max, h.max, rtol=1e-6)
    assert_almost_equal(d.mean, h.mean, rtol=1e-4)
    assert_almost_equal(d.variance, h.variance, rtol=1e-4)
    assert_almost_equal(d.skewness, h.skewness, rtol=1e-3)
    assert_almost_equal(d.kurtosis, h.kurtosis, rtol=1e-3)


def test_the_means_and_entropies_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    assert_almost_equal(
        gmean[gpu=True](_positive(gpu)), gmean(_positive(cpu)), rtol=1e-4
    )
    assert_almost_equal(
        hmean[gpu=True](_positive(gpu)), hmean(_positive(cpu)), rtol=1e-4
    )
    assert_almost_equal(
        entropy[gpu=True](_positive(gpu)), entropy(_positive(cpu)), rtol=1e-4
    )
    assert_almost_equal(
        entropy[gpu=True](_positive(gpu), 2.0),
        entropy(_positive(cpu), 2.0),
        rtol=1e-4,
    )
    assert_almost_equal(
        entropy[gpu=True](_positive(gpu), _skewed(gpu) * 0.0 + 1.0),
        entropy(_positive(cpu), _skewed(cpu) * 0.0 + 1.0),
        rtol=1e-4,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
