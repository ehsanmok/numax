"""`nanargmax` and `nanargmin` at `gpu=True`, against the host.

An index is exact on both targets, so the comparison is equality: a
sample with scattered NaNs and repeated extremes (the first index must
win on both), and the all-infinite case that takes the second pass.
"""

from std.testing import TestSuite, assert_equal
from std.utils.numerics import inf, nan, neg_inf

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import nanargmax, nanargmin

comptime f32 = DType.float32
comptime n = 1000


def _sample(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        if i % 31 == 0:
            values.append(nan[f32]())
        else:
            values.append(Float32((i * 37) % 101) - 50.0)
    return Static[f32, n](values^, ctx)


def test_nanarg_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    assert_equal(nanargmax[gpu=True](_sample(gpu)), nanargmax(_sample(cpu)))
    assert_equal(nanargmin[gpu=True](_sample(gpu)), nanargmin(_sample(cpu)))


def test_nanarg_all_infinite_on_the_device() raises:
    var gpu = DeviceContext()
    var low = Static[f32, 3]([nan[f32](), neg_inf[f32](), neg_inf[f32]()], gpu)
    var high = Static[f32, 3]([nan[f32](), inf[f32](), inf[f32]()], gpu)
    assert_equal(nanargmax[gpu=True](low), 1)
    assert_equal(nanargmin[gpu=True](high), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
