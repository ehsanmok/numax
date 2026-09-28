"""`softmax` over a tensor at `gpu=True`: MAX's `nn.softmax` on the device,
against the host, along the last axis (the one MAX's kernel handles)."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
)

from max.gpu.host import DeviceContext

from numax import Static, softmax

comptime f32 = DType.float32


def _grid(ctx: DeviceContext) raises -> Static[f32, 6, 33]:
    var values = List[Scalar[f32]](capacity=6 * 33)
    for i in range(6 * 33):
        values.append(Float32((i * 13) % 29) * 0.25 - 3.0)
    return Static[f32, 6, 33](ctx, values^)


def test_softmax_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = softmax[gpu=True](_grid(gpu))
    assert_false(d.on_host())
    var got = d.to_host()
    var want = softmax(_grid(cpu)).to_host()
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-6, rtol=1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
