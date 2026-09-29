"""`interpn` at `gpu=True`, against the host, at `float32`, in three
dimensions: one lane per point bisecting each axis and blending the
eight corners, the nearest rule, and the out-of-range flag reduced on
the device."""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal, assert_false
from std.testing import assert_raises

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.interpolate import interpn

comptime f32 = DType.float32
comptime m = 300


def _axes(ctx: DeviceContext) raises -> Static[f32, 21]:
    var values = List[Scalar[f32]](capacity=21)
    for i in range(6):
        values.append(Float32(i) * 0.4)
    for i in range(7):
        values.append(Float32(i * i) * 0.05 - 1.0)
    for i in range(8):
        values.append(Float32(i) * 0.3 + 0.02 * sin(Float32(i)))
    return Static[f32, 21](values^, ctx)


def _values(ctx: DeviceContext) raises -> Static[f32, 6, 7, 8]:
    var values = List[Scalar[f32]](capacity=6 * 7 * 8)
    for i in range(6 * 7 * 8):
        values.append(sin(Float32(i) * 0.37))
    return Static[f32, 6, 7, 8](values^, ctx)


def _xi(ctx: DeviceContext) raises -> Static[f32, m, 3]:
    var values = List[Scalar[f32]](capacity=m * 3)
    for i in range(m):
        values.append(1.0 + 0.95 * sin(Float32(i) * 1.1))
        values.append(-0.2 + 0.75 * sin(Float32(i) * 0.7))
        values.append(1.05 + 1.0 * sin(Float32(i) * 0.3))
    return Static[f32, m, 3](values^, ctx)


def test_interpn_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var methods: List[StaticString] = ["linear", "nearest"]
    for method in methods:
        var d = interpn[gpu=True](_axes(gpu), _values(gpu), _xi(gpu), method)
        assert_false(d.on_host())
        var got = d.to_host()
        var want = interpn(_axes(cpu), _values(cpu), _xi(cpu), method).to_host()
        for i in range(m):
            assert_almost_equal(got[i], want[i], atol=1e-5)
    var far = Static[f32, 1, 3]([5.0, 0.0, 0.0], gpu)
    with assert_raises(contains="out of bounds"):
        _ = interpn[gpu=True](_axes(gpu), _values(gpu), far)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
