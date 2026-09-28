"""`nonzero`, `argwhere`, `extract` and `compress` at `gpu=True`: results
whose length the data decides, built on the device by an offsets scan and
one scatter, against the host."""

from std.testing import TestSuite, assert_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.logic import greater
from numax.core.sorting import argwhere, compress, extract, nonzero
from numax.core.tensor import Static

comptime f32 = DType.float32


def _data(ctx: DeviceContext) raises -> Static[f32, 4, 250]:
    var values = List[Scalar[f32]](capacity=1000)
    for i in range(1000):
        values.append(Float32(0) if (i * 7) % 5 < 2 else Float32(i % 11) - 5.0)
    return Static[f32, 4, 250](values^, ctx)


def test_nonzero_and_argwhere_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = nonzero[gpu=True](_data(gpu))
    assert_false(d.on_host())
    var got = d.to_host()
    var want = nonzero(_data(cpu)).to_host()
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])
    var dw = argwhere[gpu=True](_data(gpu)).to_host()
    var hw = argwhere(_data(cpu)).to_host()
    assert_equal(len(dw), len(hw))
    for i in range(len(hw)):
        assert_equal(dw[i], hw[i])


def test_extract_and_compress_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var mask = greater[gpu=True](_data(gpu), Static[f32, 4, 250](gpu))
    var host_mask = greater(_data(cpu), Static[f32, 4, 250](cpu))
    var d = extract[gpu=True](mask, _data(gpu))
    assert_false(d.on_host())
    var got = d.to_host()
    var want = extract(host_mask, _data(cpu)).to_host()
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])
    var cond: List[Scalar[DType.bool]] = [True, False, True, True, False, True]
    var line_values = List[Scalar[f32]](capacity=8)
    for i in range(8):
        line_values.append(Float32(i) * 1.5)
    var dc = compress[gpu=True](
        Static[DType.bool, 6](cond.copy(), gpu),
        Static[f32, 8](line_values.copy(), gpu),
    ).to_host()
    var hc = compress(
        Static[DType.bool, 6](cond^, cpu), Static[f32, 8](line_values^, cpu)
    ).to_host()
    assert_equal(len(dc), 4)
    for i in range(4):
        assert_equal(dc[i], hc[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
