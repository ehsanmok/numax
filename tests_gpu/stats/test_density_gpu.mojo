"""`median_abs_deviation`, `ecdf` and `gaussian_kde` at `gpu=True`,
against the host, at `float32`: the medians and the ECDF's distinct
values are exact on both targets, the KDE's sums to `float32` rounding."""

from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.testing import assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static, linspace
from numax.stats import ecdf, gaussian_kde, median_abs_deviation

comptime f32 = DType.float32
comptime n = 600


def _x(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32((i * 37) % 41) * 0.25 - 3.0)
    return Static[f32, n](values^, ctx)


def test_density_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    assert_almost_equal(
        median_abs_deviation[gpu=True](_x(gpu)), median_abs_deviation(_x(cpu))
    )
    var de = ecdf[gpu=True](_x(gpu))
    assert_false(de.probabilities.on_host())
    var he = ecdf(_x(cpu))
    var dq = de.quantiles.to_host()
    var hq = he.quantiles.to_host()
    var dp = de.probabilities.to_host()
    var hp = he.probabilities.to_host()
    assert_equal(len(dq), len(hq))
    for i in range(len(hq)):
        assert_equal(dq[i], hq[i])
        assert_almost_equal(dp[i], hp[i], atol=1e-6)
    var dk = gaussian_kde[f32].create[gpu=True](_x(gpu))
    var hk = gaussian_kde[f32].create(_x(cpu))
    var got = dk.evaluate[gpu=True](
        linspace[64, f32](-4.0, 8.0, ctx=gpu)
    ).to_host()
    var want = hk.evaluate(linspace[64, f32](-4.0, 8.0, ctx=cpu)).to_host()
    for i in range(64):
        assert_almost_equal(got[i], want[i], atol=1e-5, rtol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
