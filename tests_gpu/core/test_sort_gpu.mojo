"""`argsort`, `sort`, `partition`, `argpartition` and `take` at `gpu=True`.

`argsort` is numax's device bitonic sort (MAX's GPU sort is wrong past 256
elements on Metal) and returns its indices as a device tensor; the value
sorts are that plus a device gather. It orders `(value, index)` pairs, so
ties and NaN land exactly where the host's stable sort puts them.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_raises

from max.gpu.host import DeviceContext

from numax.core.sorting import argpartition, argsort, partition, sort, take
from numax.core.tensor import Static

comptime f32 = DType.float32
comptime n = 1500


def _shuffled(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32((i * 7919) % n) * 0.5 - 100.0)
    return Static[f32, n](values^, ctx)


def test_argsort_and_sort_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = argsort[gpu=True](_shuffled(gpu))
    assert_false(d.on_host())
    var got = d.to_host()
    var want = argsort(_shuffled(cpu)).to_host()
    for i in range(n):
        assert_equal(got[i], want[i])
    var s = sort[gpu=True](_shuffled(gpu))
    assert_false(s.on_host())
    var sv = s.to_host()
    var hv = sort(_shuffled(cpu)).to_host()
    for i in range(n):
        assert_equal(sv[i], hv[i])


def test_partition_argpartition_and_take_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var p = partition[gpu=True](_shuffled(gpu), 700).to_host()
    var hp = partition(_shuffled(cpu), 700).to_host()
    assert_equal(p[700], hp[700])
    var ip = argpartition[gpu=True](_shuffled(gpu), 10).to_host()
    var ih = argpartition(_shuffled(cpu), 10).to_host()
    assert_equal(ip[10], ih[10])
    var order = argsort[gpu=True](_shuffled(gpu))
    var taken = take[axis=0, gpu=True](_shuffled(gpu), order)
    assert_false(taken.on_host())
    var tv = taken.to_host()
    for i in range(1, n):
        assert_equal(tv[i - 1] < tv[i], True)


def test_take_on_the_device_refuses_an_out_of_range_index() raises:
    var gpu = DeviceContext()
    var bad: List[Scalar[DType.int64]] = [0, 5, Int64(n)]
    with assert_raises(contains="out of range"):
        _ = take[axis=0, gpu=True](
            _shuffled(gpu), Static[DType.int64, 3](bad^, gpu)
        )


def test_ties_and_nan_sort_exactly_as_on_the_host() raises:
    """Heavy ties at a non-power-of-two length, plus NaNs, which NumPy
    sorts last: the device indices equal the host's, one for one."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var nan = Float32.MAX * 2 - Float32.MAX * 2
    var values = List[Scalar[f32]](capacity=777)
    for i in range(777):
        values.append(nan if i % 101 == 50 else Float32(i % 9))
    var d = argsort[gpu=True](Static[f32, 777](values.copy(), gpu)).to_host()
    var h = argsort(Static[f32, 777](values^, cpu)).to_host()
    for i in range(777):
        assert_equal(d[i], h[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
