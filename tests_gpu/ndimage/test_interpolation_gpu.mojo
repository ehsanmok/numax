"""`numax.ndimage`'s spline interpolation at `gpu=True`, against the host,
at `float32`: the per-line prefilter recursions, and the evaluation
lanes' coordinate mapping, weights and footprint sums, through `zoom`,
`shift` and `map_coordinates`."""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.ndimage import map_coordinates, shift, zoom

comptime f32 = DType.float32


def _img(ctx: DeviceContext) raises -> Static[f32, 24, 19]:
    var values = List[Scalar[f32]](capacity=24 * 19)
    for i in range(24 * 19):
        values.append(sin(Float32(i) * 0.21) + 0.2 * sin(Float32(i) * 1.7))
    return Static[f32, 24, 19](values^, ctx)


def _coords(ctx: DeviceContext) raises -> Static[f32, 2, 50]:
    var values = List[Scalar[f32]](capacity=100)
    for i in range(50):
        values.append(-1.0 + 26.0 * Float32(i) / 49.0)
    for i in range(50):
        values.append(20.0 - 22.0 * Float32(i) / 49.0)
    return Static[f32, 2, 50](values^, ctx)


def _close(got: List[Scalar[f32]], want: List[Scalar[f32]]) raises:
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-4, rtol=1e-4)


def test_interpolation_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = zoom[gpu=True](_img(gpu), 1.7)
    assert_false(d.on_host())
    _close(d.to_host(), zoom(_img(cpu), 1.7).to_host())
    _close(
        shift[gpu=True](_img(gpu), -2.4, 5, "reflect").to_host(),
        shift(_img(cpu), -2.4, 5, "reflect").to_host(),
    )
    var modes: List[StaticString] = [
        "constant",
        "nearest",
        "mirror",
        "grid-wrap",
    ]
    for mode in modes:
        _close(
            map_coordinates[gpu=True](
                _img(gpu), _coords(gpu), 3, mode, 0.25
            ).to_host(),
            map_coordinates(_img(cpu), _coords(cpu), 3, mode, 0.25).to_host(),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
