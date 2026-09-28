"""`rankdata` and `spearmanr` at `gpu=True`, against the host.

Ranks are small integers and half-integers, exact at `float32`, so every
method's ranks must equal the host's exactly on a sample with long runs
of ties; `spearmanr` then agrees to `float32` precision, its moments
being device sums.
"""

from std.math import sin
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
)

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import rankdata, spearmanr

comptime f32 = DType.float32
comptime n = 257


def _tied(ctx: DeviceContext) raises -> Static[f32, n]:
    """Values from a set of eleven, so every value repeats about two dozen
    times, in scrambled order."""
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32((i * 53) % 11) * 0.5 - 2.0)
    return Static[f32, n](values^, ctx)


def _other(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(i % 17) + Float32(sin(Float64(i))))
    return Static[f32, n](values^, ctx)


def _check(method: StaticString) raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = rankdata[gpu=True](_tied(gpu), method)
    assert_false(d.on_host())
    var got = d.to_host()
    var want = rankdata(_tied(cpu), method).to_host()
    for i in range(n):
        assert_equal(got[i], want[i])


def test_rankdata_on_the_device_matches_the_host() raises:
    _check("average")
    _check("min")
    _check("max")
    _check("dense")
    _check("ordinal")


def test_spearmanr_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = spearmanr[gpu=True](_tied(gpu), _other(gpu))
    var h = spearmanr(_tied(cpu), _other(cpu))
    assert_almost_equal(d.statistic, h.statistic, rtol=1e-4)
    assert_almost_equal(d.pvalue, h.pvalue, rtol=1e-2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
