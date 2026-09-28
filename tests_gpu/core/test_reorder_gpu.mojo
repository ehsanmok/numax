"""`roll`, `flip`, `repeat`, `tile` and `rot90` at `gpu=True`, against the
host.

Each routine reorders elements along an axis, which the device does with
one gather launch over the result. Every test checks the result stayed on
the device and holds exactly the host's values; a reorder does no
arithmetic, so the comparison is exact.
"""

from std.testing import TestSuite, assert_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static, flip, repeat, roll, rot90, tile

comptime f32 = DType.float32


def _grid(ctx: DeviceContext) raises -> Static[f32, 3, 4]:
    var values = List[Scalar[f32]](capacity=12)
    for i in range(12):
        values.append(Float32(i) - 5.0)
    return Static[f32, 3, 4](values^, ctx)


def _line(ctx: DeviceContext) raises -> Static[f32, 7]:
    var values = List[Scalar[f32]](capacity=7)
    for i in range(7):
        values.append(Float32(i * i))
    return Static[f32, 7](values^, ctx)


def _assert_same(got: List[Scalar[f32]], want: List[Scalar[f32]]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def test_roll_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    for shift in range(-5, 6):
        var d = roll[axis=1, gpu=True](_grid(gpu), shift)
        assert_false(d.on_host())
        _assert_same(d.to_host(), roll[axis=1](_grid(cpu), shift).to_host())
    _assert_same(
        roll[axis=0, gpu=True](_grid(gpu), 1).to_host(),
        roll[axis=0](_grid(cpu), 1).to_host(),
    )


def test_flip_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var d = flip[gpu=True](_line(gpu))
    assert_false(d.on_host())
    _assert_same(d.to_host(), flip(_line(DeviceContext(api="cpu"))).to_host())


def test_repeat_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = repeat[axis=1, gpu=True](_grid(gpu), 3)
    assert_false(d.on_host())
    assert_equal(d.dim_at(1), 12)
    _assert_same(d.to_host(), repeat[axis=1](_grid(cpu), 3).to_host())
    _assert_same(
        repeat[axis=0, gpu=True](_grid(gpu), 2).to_host(),
        repeat[axis=0](_grid(cpu), 2).to_host(),
    )


def test_tile_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = tile[gpu=True](_grid(gpu), 2, 3)
    assert_false(d.on_host())
    assert_equal(d.dim_at(0), 6)
    assert_equal(d.dim_at(1), 12)
    _assert_same(d.to_host(), tile(_grid(cpu), 2, 3).to_host())


def test_rot90_on_the_device_matches_the_host() raises:
    """All four quarter turns of a non-square matrix, so an odd turn's
    swapped extents are exercised."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var one = rot90[k=1, gpu=True](_grid(gpu))
    assert_false(one.on_host())
    _assert_same(one.to_host(), rot90[k=1](_grid(cpu)).to_host())
    _assert_same(
        rot90[k=3, gpu=True](_grid(gpu)).to_host(),
        rot90[k=3](_grid(cpu)).to_host(),
    )
    _assert_same(
        rot90[k=2, gpu=True](_grid(gpu)).to_host(),
        rot90[k=2](_grid(cpu)).to_host(),
    )
    _assert_same(
        rot90[k=0, gpu=True](_grid(gpu)).to_host(),
        rot90[k=0](_grid(cpu)).to_host(),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
