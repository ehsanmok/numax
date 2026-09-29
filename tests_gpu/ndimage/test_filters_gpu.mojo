"""`numax.ndimage`'s filters at `gpu=True`, against the host, at
`float32`: the N-d correlation kernel under every boundary mode, the
separable Gaussian and uniform passes in three dimensions, and the rank
filters' in-lane selection."""

from std.math import cos, sin
from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.ndimage import (
    convolve,
    gaussian_filter,
    maximum_filter,
    median_filter,
    uniform_filter,
)

comptime f32 = DType.float32


def _img(ctx: DeviceContext) raises -> Static[f32, 40, 33]:
    var values = List[Scalar[f32]](capacity=40 * 33)
    for i in range(40 * 33):
        values.append(sin(Float32(i) * 0.37) + 0.3 * cos(Float32(i) * 1.9))
    return Static[f32, 40, 33](values^, ctx)


def _vol(ctx: DeviceContext) raises -> Static[f32, 12, 10, 9]:
    var values = List[Scalar[f32]](capacity=12 * 10 * 9)
    for i in range(12 * 10 * 9):
        values.append(sin(Float32(i) * 0.11))
    return Static[f32, 12, 10, 9](values^, ctx)


def _w(ctx: DeviceContext) raises -> Static[f32, 3, 4]:
    return Static[f32, 3, 4](
        [0.1, 0.2, -0.3, 0.05, 1.0, -0.5, 0.25, 0.3, 0.0, 0.7, -0.2, 0.15], ctx
    )


def _close(got: List[Scalar[f32]], want: List[Scalar[f32]]) raises:
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-5, rtol=1e-5)


def test_filters_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var modes: List[StaticString] = [
        "reflect",
        "mirror",
        "nearest",
        "wrap",
        "constant",
    ]
    for mode in modes:
        var d = convolve[gpu=True](_img(gpu), _w(gpu), mode, 0.5)
        assert_false(d.on_host())
        _close(d.to_host(), convolve(_img(cpu), _w(cpu), mode, 0.5).to_host())
    _close(
        gaussian_filter[gpu=True](_vol(gpu), 1.3, 1).to_host(),
        gaussian_filter(_vol(cpu), 1.3, 1).to_host(),
    )
    _close(
        uniform_filter[gpu=True](_vol(gpu), 4, "wrap").to_host(),
        uniform_filter(_vol(cpu), 4, "wrap").to_host(),
    )
    _close(
        median_filter[gpu=True](_img(gpu), 5).to_host(),
        median_filter(_img(cpu), 5).to_host(),
    )
    _close(
        maximum_filter[gpu=True](_vol(gpu), 3, "constant", -2.0).to_host(),
        maximum_filter(_vol(cpu), 3, "constant", -2.0).to_host(),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
