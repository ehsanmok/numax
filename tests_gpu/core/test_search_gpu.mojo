"""`searchsorted` and `unique` at `gpu=True`, against the host.

Both answer with indices or values the host also produces exactly, so
the comparison is exact: insertion points on either side, with queries
equal to haystack values, below the first and past the last; distinct
values of a sample with many repeats, and with NaNs, each of which is
its own value.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true
from std.utils.numerics import nan

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.core.sorting import searchsorted, unique

comptime f32 = DType.float32


def _haystack(ctx: DeviceContext) raises -> Static[f32, 9]:
    return Static[f32, 9](ctx, [-3.0, -1.0, 0.0, 0.0, 0.0, 2.5, 4.0, 4.0, 9.0])


def _needles(ctx: DeviceContext) raises -> Static[f32, 8]:
    return Static[f32, 8](ctx, [-5.0, -3.0, 0.0, 1.0, 4.0, 8.9, 9.0, 10.0])


def test_searchsorted_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = searchsorted[gpu=True](_haystack(gpu), _needles(gpu))
    assert_false(d.on_host())
    var got = d.to_host()
    var want = searchsorted(_haystack(cpu), _needles(cpu)).to_host()
    for i in range(8):
        assert_equal(got[i], want[i])
    var dr = searchsorted[right=True, gpu=True](
        _haystack(gpu), _needles(gpu)
    ).to_host()
    var hr = searchsorted[right=True](_haystack(cpu), _needles(cpu)).to_host()
    for i in range(8):
        assert_equal(dr[i], hr[i])


def _sample(ctx: DeviceContext, with_nan: Bool) raises -> Static[f32, 300]:
    var values = List[Scalar[f32]](capacity=300)
    for i in range(300):
        if with_nan and i % 97 == 5:
            values.append(nan[f32]())
        else:
            values.append(Float32((i * 37) % 23) - 11.0)
    return Static[f32, 300](ctx, values^)


def test_unique_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = unique[gpu=True](_sample(gpu, False))
    assert_false(d.on_host())
    var got = d.to_host()
    var want = unique(_sample(cpu, False)).to_host()
    assert_equal(len(got), len(want))
    assert_equal(len(got), 23)
    for i in range(len(want)):
        assert_equal(got[i], want[i])
    var dn = unique[gpu=True](_sample(gpu, True)).to_host()
    var hn = unique(_sample(cpu, True)).to_host()
    assert_equal(len(dn), len(hn))
    for i in range(len(hn)):
        if hn[i] != hn[i]:
            assert_true(dn[i] != dn[i])
        else:
            assert_equal(dn[i], hn[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
