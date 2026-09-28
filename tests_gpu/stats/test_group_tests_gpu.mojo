"""`kruskal`, `levene` and `bartlett` at `gpu=True`, against the host, at
`float32`: the ranking, the centered deviations and the moments on the
device, the statistics to `float32` precision of their sums."""

from std.testing import TestSuite, assert_almost_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import bartlett, kruskal, levene

comptime f32 = DType.float32
comptime n = 400


def _group(
    ctx: DeviceContext, seed: Int, shift: Float32
) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(((i * 37 + seed) % 53)) * 0.1 + shift)
    return Static[f32, n](values^, ctx)


def test_group_tests_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = kruskal[gpu=True](
        _group(gpu, 1, 0.0), _group(gpu, 7, 0.3), _group(gpu, 11, -0.2)
    )
    var h = kruskal(
        _group(cpu, 1, 0.0), _group(cpu, 7, 0.3), _group(cpu, 11, -0.2)
    )
    assert_almost_equal(d.statistic, h.statistic, rtol=1e-4)
    assert_almost_equal(d.pvalue, h.pvalue, atol=1e-5, rtol=1e-3)
    var dl = levene[gpu=True](
        _group(gpu, 1, 0.0), _group(gpu, 7, 0.3), _group(gpu, 11, -0.2)
    )
    var hl = levene(
        _group(cpu, 1, 0.0), _group(cpu, 7, 0.3), _group(cpu, 11, -0.2)
    )
    assert_almost_equal(dl.statistic, hl.statistic, rtol=1e-3, atol=1e-5)
    var db = bartlett[gpu=True](
        _group(gpu, 1, 0.0), _group(gpu, 7, 0.3), _group(gpu, 11, -0.2)
    )
    var hb = bartlett(
        _group(cpu, 1, 0.0), _group(cpu, 7, 0.3), _group(cpu, 11, -0.2)
    )
    assert_almost_equal(db.statistic, hb.statistic, rtol=1e-3, atol=1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
