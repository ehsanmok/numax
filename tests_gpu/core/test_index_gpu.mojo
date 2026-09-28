"""`take_along_axis`, `take` with a host index list, and both `put`s at
`gpu=True`, against the host.

Gathers and scatters move values without arithmetic, so the comparison
is exact; an out-of-range index raises on the device as on the host.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
)

from max.gpu.host import DeviceContext

from numax.core.tensor import Dynamic, Static
from numax.core.sorting import put, take, take_along_axis
from layout.tile_layout import row_major
from numax.core.tensor import _dyn_shape, _dyn_shape_from

comptime f32 = DType.float32


def _grid(ctx: DeviceContext) raises -> Static[f32, 3, 5]:
    var values = List[Scalar[f32]](capacity=15)
    for i in range(15):
        values.append(Float32((i * 7) % 15) - 4.0)
    return Static[f32, 3, 5](values^, ctx)


def _picks(
    ctx: DeviceContext, rows: Int, cols: Int, bound: Int
) raises -> Dynamic[DType.int64, 2]:
    var values = List[Scalar[DType.int64]](capacity=rows * cols)
    for i in range(rows * cols):
        values.append(Int64((i * 3 + 1) % bound))
    var shape = List[Int](capacity=2)
    shape.append(rows)
    shape.append(cols)
    return Dynamic[DType.int64, 2](
        row_major(_dyn_shape_from[2](shape)), values^, ctx
    )


def test_take_along_axis_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = take_along_axis[axis=1, gpu=True](_grid(gpu), _picks(gpu, 3, 7, 5))
    assert_false(d.on_host())
    var got = d.to_host()
    var want = take_along_axis[axis=1](
        _grid(cpu), _picks(cpu, 3, 7, 5)
    ).to_host()
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])
    var d0 = take_along_axis[axis=0, gpu=True](
        _grid(gpu), _picks(gpu, 2, 5, 3)
    ).to_host()
    var h0 = take_along_axis[axis=0](_grid(cpu), _picks(cpu, 2, 5, 3)).to_host()
    for i in range(len(h0)):
        assert_equal(d0[i], h0[i])
    with assert_raises():
        _ = take_along_axis[axis=1, gpu=True](_grid(gpu), _picks(gpu, 3, 2, 9))


def test_take_with_a_list_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var indices: List[Int] = [14, 0, 3, 3, 7, 11]
    var d = take[gpu=True](_grid(gpu), indices)
    assert_false(d.on_host())
    var got = d.to_host()
    var want = take(_grid(cpu), indices).to_host()
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def test_put_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var indices: List[Int] = [0, 4, 9, 14]
    var values: List[Scalar[f32]] = [1.5, -2.5, 3.5, 100.0]
    var dg = _grid(gpu)
    var hc = _grid(cpu)
    put[gpu=True](dg, indices, values)
    put(hc, indices, values)
    assert_false(dg.on_host())
    var got = dg.to_host()
    var want = hc.to_host()
    for i in range(15):
        assert_equal(got[i], want[i])
    var tg = Dynamic[DType.int64, 1](row_major(_dyn_shape[1](2)), [2, 6], gpu)
    var tc = Dynamic[DType.int64, 1](row_major(_dyn_shape[1](2)), [2, 6], cpu)
    var one: List[Scalar[f32]] = [9.0]
    put[gpu=True](dg, tg, one)
    put(hc, tc, one)
    got = dg.to_host()
    want = hc.to_host()
    for i in range(15):
        assert_equal(got[i], want[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
