"""The order-preserving shape routines on a device tensor: one
device-to-device copy each, no host round trip, and the host's answer.

`reshape`, `reshape_dyn`, `ravel`, `squeeze`, `expand_dims`, `atleast_2d`
and `copy` keep the elements in row-major order, so on a device each is one
`enqueue_copy` into a buffer on the same device. Every test checks the
result stayed on the device and holds the host's values.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_raises

from max.gpu.host import DeviceContext

from numax.core.tensor import (
    Static,
    atleast_2d,
    copy,
    expand_dims,
    ravel,
    reshape,
    reshape_dyn,
    squeeze,
)

comptime f32 = DType.float32


def _grid(ctx: DeviceContext) raises -> Static[f32, 3, 4]:
    var values = List[Scalar[f32]](capacity=12)
    for i in range(12):
        values.append(Float32(i) * 0.5 - 1.0)
    return Static[f32, 3, 4](ctx, values^)


def _assert_same(got: List[Scalar[f32]], want: List[Scalar[f32]]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def test_reshape_and_ravel_stay_on_the_device() raises:
    var gpu = DeviceContext()
    var want = _grid(DeviceContext(api="cpu")).to_host()
    var r = reshape[rows=4, cols=3](ravel(_grid(gpu)))
    assert_false(r.on_host())
    _assert_same(r.to_host(), want)
    var d = reshape_dyn[rank=3](_grid(gpu), 2, 3, 2)
    assert_false(d.on_host())
    assert_equal(d.dim_at(2), 2)
    _assert_same(d.to_host(), want)
    var flat = ravel(_grid(gpu))
    assert_false(flat.on_host())
    _assert_same(flat.to_host(), want)


def test_squeeze_expand_dims_and_atleast_stay_on_the_device() raises:
    var gpu = DeviceContext()
    var want = _grid(DeviceContext(api="cpu")).to_host()
    var e = expand_dims[axis=1](_grid(gpu))
    assert_false(e.on_host())
    assert_equal(e.dim_at(1), 1)
    _assert_same(e.to_host(), want)
    var s = squeeze[axis=1](e)
    assert_false(s.on_host())
    _assert_same(s.to_host(), want)
    var two = atleast_2d(_grid(gpu))
    assert_false(two.on_host())
    _assert_same(two.to_host(), want)


def test_copy_stays_on_the_device_and_does_not_alias() raises:
    var gpu = DeviceContext()
    var a = _grid(gpu)
    var b = copy(a)
    assert_false(b.on_host())
    a[0] = 100.0
    assert_equal(b[0], -1.0)


def test_reshape_dyn_refuses_a_wrong_count_on_the_device() raises:
    var gpu = DeviceContext()
    with assert_raises(contains="does not fit 12"):
        _ = reshape_dyn[rank=2](_grid(gpu), 2, 5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
