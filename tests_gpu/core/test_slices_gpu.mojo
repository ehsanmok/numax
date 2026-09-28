"""Slices of a device tensor are device views: `a[i:j]` and
`a.block[m, n](r, c)` borrow the device buffer, operators on them run on
the device, and a strided box copies to an owned device tensor."""

from std.testing import TestSuite, assert_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static, copy

comptime f32 = DType.float32


def _grid(ctx: DeviceContext) raises -> Static[f32, 64, 32]:
    var values = List[Scalar[f32]](capacity=64 * 32)
    for i in range(64 * 32):
        values.append(Float32(i % 97))
    return Static[f32, 64, 32](values^, ctx)


def test_device_slices_stay_on_the_device() raises:
    var a = _grid(DeviceContext())
    var h = _grid(DeviceContext(api="cpu"))
    var v = a[8:40]
    assert_false(v.on_host())
    var twice = v * 2.0
    assert_false(twice.on_host())
    var got = twice.to_host()
    var hv = h[8:40]
    var want = (hv * 2.0).to_host()
    assert_equal(len(got), 32 * 32)
    for i in range(len(want)):
        assert_equal(got[i], want[i])
    var box = copy(a.block[4, 4](10, 10))
    assert_false(box.on_host())
    var bg = box.to_host()
    var bh = copy(h.block[4, 4](10, 10)).to_host()
    for i in range(16):
        assert_equal(bg[i], bh[i])


def test_a_static_block_of_a_device_tensor_copies_on_the_device() raises:
    """`tile().tile[m, n]` keeps a compile-time strided layout, which the
    strided gather also serves."""
    from numax.core.tensorlike import TensorView

    var a = _grid(DeviceContext())
    var h = _grid(DeviceContext(api="cpu"))
    var v = TensorView(a.tile().tile[4, 8](2, 1), a.context())
    var hv = TensorView(h.tile().tile[4, 8](2, 1), h.context())
    var got = copy(v)
    assert_false(got.on_host())
    var g = got.to_host()
    var w = copy(hv).to_host()
    for i in range(32):
        assert_equal(g[i], w[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
