"""`median` and `mode` along an axis, and the whole-tensor `mode`, at
`gpu=True`, against the host.

Both pick values out of each slice -- the median averages at most two --
so the device answers equal the host's exactly, along every axis of a
rank-3 tensor with odd and even slice lengths and many repeated values.
"""

from std.testing import TestSuite, assert_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import median, mode

comptime f32 = DType.float32


def _cube(ctx: DeviceContext) raises -> Static[f32, 4, 5, 6]:
    """Values from a set of nine, so every slice has ties."""
    var values = List[Scalar[f32]](capacity=120)
    for i in range(120):
        values.append(Float32((i * 37 + i // 7) % 9) - 3.0)
    return Static[f32, 4, 5, 6](values^, ctx)


def _same(got: List[Scalar[f32]], want: List[Scalar[f32]]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def test_axis_median_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d0 = median[axis=0, gpu=True](_cube(gpu))
    assert_false(d0.on_host())
    _same(d0.to_host(), median[axis=0](_cube(cpu)).to_host())
    _same(
        median[axis=1, gpu=True](_cube(gpu)).to_host(),
        median[axis=1](_cube(cpu)).to_host(),
    )
    _same(
        median[axis=2, gpu=True](_cube(gpu)).to_host(),
        median[axis=2](_cube(cpu)).to_host(),
    )


def test_mode_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    assert_equal(mode[gpu=True](_cube(gpu)), mode(_cube(cpu)))
    var d0 = mode[axis=0, gpu=True](_cube(gpu))
    assert_false(d0.on_host())
    _same(d0.to_host(), mode[axis=0](_cube(cpu)).to_host())
    _same(
        mode[axis=1, gpu=True](_cube(gpu)).to_host(),
        mode[axis=1](_cube(cpu)).to_host(),
    )
    _same(
        mode[axis=2, gpu=True](_cube(gpu)).to_host(),
        mode[axis=2](_cube(cpu)).to_host(),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
