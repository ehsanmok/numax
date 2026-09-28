"""`cumsum`, `cumprod` and `cumulative_trapezoid` at `gpu=True`: numax's
device scan (MAX's `nn.cumsum` has no device path), flat and along every
axis, against the host."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
)

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.integrate import cumulative_trapezoid
from numax.stats import cumprod, cumsum

comptime f32 = DType.float32


def _cube(ctx: DeviceContext) raises -> Static[f32, 3, 5, 7]:
    var values = List[Scalar[f32]](capacity=105)
    for i in range(105):
        values.append(Float32((i * 13) % 17) * 0.125 - 1.0)
    return Static[f32, 3, 5, 7](values^, ctx)


def _near_one(ctx: DeviceContext) raises -> Static[f32, 300]:
    var values = List[Scalar[f32]](capacity=300)
    for i in range(300):
        values.append(1.0 + Float32(i % 5 - 2) * 0.001)
    return Static[f32, 300](values^, ctx)


def _assert_close(got: List[Scalar[f32]], want: List[Scalar[f32]]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-4, rtol=1e-4)


def test_flat_scans_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var s = cumsum[gpu=True](_cube(gpu))
    assert_false(s.on_host())
    _assert_close(s.to_host(), cumsum(_cube(cpu)).to_host())
    _assert_close(
        cumprod[gpu=True](_near_one(gpu)).to_host(),
        cumprod(_near_one(cpu)).to_host(),
    )


def test_axis_scans_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    _assert_close(
        cumsum[axis=0, gpu=True](_cube(gpu)).to_host(),
        cumsum[axis=0](_cube(cpu)).to_host(),
    )
    _assert_close(
        cumsum[axis=1, gpu=True](_cube(gpu)).to_host(),
        cumsum[axis=1](_cube(cpu)).to_host(),
    )
    _assert_close(
        cumsum[axis=2, gpu=True](_cube(gpu)).to_host(),
        cumsum[axis=2](_cube(cpu)).to_host(),
    )
    _assert_close(
        cumprod[axis=1, gpu=True](_cube(gpu)).to_host(),
        cumprod[axis=1](_cube(cpu)).to_host(),
    )


def test_cumulative_trapezoid_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var dx = Scalar[f32](0.01)
    _assert_close(
        cumulative_trapezoid[gpu=True](_near_one(gpu), dx).to_host(),
        cumulative_trapezoid(_near_one(cpu), dx).to_host(),
    )
    _assert_close(
        cumulative_trapezoid[initial=True, gpu=True](
            _near_one(gpu), _near_one(gpu)
        ).to_host(),
        cumulative_trapezoid[initial=True](
            _near_one(cpu), _near_one(cpu)
        ).to_host(),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
