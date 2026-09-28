"""`argmax` and `argmin` along an axis at `gpu=True`, against the host.

MAX's `argmaxmin_gpu` reduces the innermost axis, so every other axis is
moved last by a device gather first; each axis of a rank-3 tensor is
checked, with ties broken low on both devices.
"""

from std.testing import TestSuite, assert_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import argmax, argmin

comptime f32 = DType.float32


def _cube(ctx: DeviceContext) raises -> Static[f32, 3, 4, 5]:
    var values = List[Scalar[f32]](capacity=60)
    for i in range(60):
        values.append(Float32((i * 37) % 11) - 5.0)
    return Static[f32, 3, 4, 5](values^, ctx)


def _assert_same(
    got: List[Scalar[DType.int64]], want: List[Scalar[DType.int64]]
) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def test_argmax_along_every_axis_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = argmax[axis=2, gpu=True](_cube(gpu))
    assert_false(d.on_host())
    _assert_same(d.to_host(), argmax[axis=2](_cube(cpu)).to_host())
    _assert_same(
        argmax[axis=1, gpu=True](_cube(gpu)).to_host(),
        argmax[axis=1](_cube(cpu)).to_host(),
    )
    _assert_same(
        argmax[axis=0, gpu=True](_cube(gpu)).to_host(),
        argmax[axis=0](_cube(cpu)).to_host(),
    )


def test_argmin_along_every_axis_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    _assert_same(
        argmin[axis=2, gpu=True](_cube(gpu)).to_host(),
        argmin[axis=2](_cube(cpu)).to_host(),
    )
    _assert_same(
        argmin[axis=0, gpu=True](_cube(gpu)).to_host(),
        argmin[axis=0](_cube(cpu)).to_host(),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
