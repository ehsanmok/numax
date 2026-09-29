"""`cdist`, `pdist` and `squareform` at `gpu=True`, against the host, at
`float32`, for every metric: one lane per pair on the device, the metric
a uniform branch."""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.spatial import cdist, pdist, squareform

comptime f32 = DType.float32


def _points[
    m: Int
](ctx: DeviceContext, seed: Float32) raises -> Static[f32, m, 5]:
    var values = List[Scalar[f32]](capacity=m * 5)
    for i in range(m * 5):
        values.append(sin(Float32(i) * seed + 0.4))
    return Static[f32, m, 5](values^, ctx)


def _close(got: List[Scalar[f32]], want: List[Scalar[f32]]) raises:
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-5, rtol=1e-4)


def test_distances_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var metrics: List[StaticString] = [
        "euclidean",
        "sqeuclidean",
        "cityblock",
        "chebyshev",
        "minkowski",
        "cosine",
        "correlation",
    ]
    for metric in metrics:
        var d = cdist[gpu=True](
            _points[37](gpu, 1.3), _points[23](gpu, 0.7), metric, 3.0
        )
        assert_false(d.on_host())
        var h = cdist(_points[37](cpu, 1.3), _points[23](cpu, 0.7), metric, 3.0)
        _close(d.to_host(), h.to_host())
    var dp = pdist[gpu=True](_points[37](gpu, 1.3))
    var hp = pdist(_points[37](cpu, 1.3))
    _close(dp.to_host(), hp.to_host())
    _close(squareform[gpu=True](dp).to_host(), squareform(hp).to_host())
    _close(
        squareform[gpu=True](squareform[gpu=True](dp)).to_host(), hp.to_host()
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
