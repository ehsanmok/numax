"""`chi2_contingency` at `gpu=True`, against the host, at `float32`: the
margins, the expected table and the terms on the device, the statistic
to `float32` precision."""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import chi2_contingency

comptime f32 = DType.float32


def _table(ctx: DeviceContext) raises -> Static[f32, 6, 5]:
    var values = List[Scalar[f32]](capacity=30)
    for i in range(30):
        values.append(Float32((i * 37) % 23 + 5))
    return Static[f32, 6, 5](values^, ctx)


def test_chi2_contingency_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = chi2_contingency[gpu=True](_table(gpu))
    assert_false(d.expected_freq.on_host())
    var h = chi2_contingency(_table(cpu))
    assert_almost_equal(d.statistic, h.statistic, rtol=1e-4)
    assert_almost_equal(d.pvalue, h.pvalue, atol=1e-5, rtol=1e-3)
    var de = d.expected_freq.to_host()
    var he = h.expected_freq.to_host()
    for i in range(30):
        assert_almost_equal(de[i], he[i], rtol=1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
