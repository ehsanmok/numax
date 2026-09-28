"""`bootstrap` and `permutation_test` at `gpu=True`, against the host,
at `float32`. The resample indices are integer draws from the same
Philox words on both targets, so the two gather the same resamples and
their distributions agree to `float32` rounding of the row means; BCa's
bias correction counts ties with the observed statistic, so its interval
is compared more loosely."""

from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.testing import assert_false

from max.gpu.host import DeviceContext

from numax.core.ops import subtract
from numax.core.tensor import Dynamic, Static
from numax.stats import Generator, bootstrap, mean, permutation_test

comptime f32 = DType.float32
comptime n = 40


def _row_mean[gpu: Bool](x: Dynamic[f32, 2]) raises -> Dynamic[f32, 1]:
    return mean[axis=1, gpu=gpu](x)


def _diff_means[
    gpu: Bool
](a: Dynamic[f32, 2], b: Dynamic[f32, 2]) raises -> Dynamic[f32, 1]:
    return subtract[gpu=gpu](mean[axis=1, gpu=gpu](a), mean[axis=1, gpu=gpu](b))


def _x(ctx: DeviceContext, shift: Float32) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32((i * 37) % 41) * 0.1 - 2.0 + shift)
    return Static[f32, n](values^, ctx)


def _close(a: Dynamic[f32, 1], b: Dynamic[f32, 1]) raises:
    var got = a.to_host()
    var want = b.to_host()
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-5)


def test_bootstrap_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var rg = Generator(seed=21)
    var rh = Generator(seed=21)
    var d = bootstrap[statistic=_row_mean, gpu=True](
        _x(gpu, 0.0), rg, n_resamples=2000, method="percentile"
    )
    var h = bootstrap[statistic=_row_mean](
        _x(cpu, 0.0), rh, n_resamples=2000, method="percentile"
    )
    assert_false(d.bootstrap_distribution.on_host())
    _close(d.bootstrap_distribution, h.bootstrap_distribution)
    assert_almost_equal(
        d.confidence_interval.low, h.confidence_interval.low, atol=1e-4
    )
    assert_almost_equal(
        d.confidence_interval.high, h.confidence_interval.high, atol=1e-4
    )
    assert_almost_equal(d.standard_error, h.standard_error, atol=1e-5)
    # BCa's `z0` counts resamples below the observed mean; at `float32`
    # a resample tied with it can round to either side on either target,
    # so the adjusted quantile moves by an order statistic or two.
    var db = bootstrap[statistic=_row_mean, gpu=True](
        _x(gpu, 0.0), rg, n_resamples=2000
    )
    var hb = bootstrap[statistic=_row_mean](_x(cpu, 0.0), rh, n_resamples=2000)
    assert_almost_equal(
        db.confidence_interval.low, hb.confidence_interval.low, atol=1e-2
    )
    assert_almost_equal(
        db.confidence_interval.high, hb.confidence_interval.high, atol=1e-2
    )


def test_permutation_test_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var types: List[StaticString] = ["independent", "samples", "pairings"]
    for k in range(3):
        var rg = Generator(seed=8)
        var rh = Generator(seed=8)
        var d = permutation_test[statistic=_diff_means, gpu=True](
            _x(gpu, 0.0),
            _x(gpu, 0.3),
            rg,
            permutation_type=types[k],
            n_resamples=1500,
        )
        var h = permutation_test[statistic=_diff_means](
            _x(cpu, 0.0),
            _x(cpu, 0.3),
            rh,
            permutation_type=types[k],
            n_resamples=1500,
        )
        assert_false(d.null_distribution.on_host())
        _close(d.null_distribution, h.null_distribution)
        assert_almost_equal(d.statistic, h.statistic, atol=1e-5)
        assert_almost_equal(d.pvalue, h.pvalue, atol=2e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
