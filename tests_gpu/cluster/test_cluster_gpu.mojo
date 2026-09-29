"""`numax.cluster` at `gpu=True`, against the host, at `float32`: the
fused assignment lanes and the centroid update inside `kmeans` and
`kmeans2`, and `linkage`'s device distance matrix with its per-step
row-minimum, reduction and Lance-Williams update launches, feeding
`fcluster`."""

from std.math import cos, sin
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.testing import assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.cluster import fcluster, kmeans, kmeans2, linkage, vq

comptime f32 = DType.float32
comptime n = 90


def _obs(ctx: DeviceContext) raises -> Static[f32, n, 3]:
    var values = List[Scalar[f32]](capacity=n * 3)
    for i in range(n):
        var c = Float32(i % 3) * 4.0
        values.append(c + sin(Float32(i) * 1.7))
        values.append(-c + cos(Float32(i) * 0.9))
        values.append(0.5 * c + sin(Float32(i) * 0.37))
    return Static[f32, n, 3](values^, ctx)


def _book(ctx: DeviceContext) raises -> Static[f32, 3, 3]:
    return Static[f32, 3, 3](
        [0.5, 0.2, 0.1, 3.5, -3.5, 1.8, 8.4, -7.6, 4.3], ctx
    )


def test_cluster_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var dv = vq[gpu=True](_obs(gpu), _book(gpu))
    assert_false(dv.code.on_host())
    var hv = vq(_obs(cpu), _book(cpu))
    var dc = dv.code.to_host()
    var hc = hv.code.to_host()
    for i in range(n):
        assert_equal(Int(dc[i]), Int(hc[i]))
    var dk = kmeans[gpu=True](_obs(gpu), _book(gpu))
    var hk = kmeans(_obs(cpu), _book(cpu))
    assert_almost_equal(dk.distortion, hk.distortion, atol=1e-4)
    var d2 = kmeans2[gpu=True](_obs(gpu), _book(gpu), 6).centroid.to_host()
    var h2 = kmeans2(_obs(cpu), _book(cpu), 6).centroid.to_host()
    for i in range(9):
        assert_almost_equal(d2[i], h2[i], atol=1e-4)
    var methods: List[StaticString] = ["average", "ward", "single"]
    for method in methods:
        var dz = linkage[gpu=True](_obs(gpu), method)
        var hz = linkage(_obs(cpu), method)
        var dzh = dz.to_host()
        var hzh = hz.to_host()
        for i in range(4 * (n - 1)):
            assert_almost_equal(dzh[i], hzh[i], atol=1e-3, rtol=1e-4)
        var df = fcluster(dz, 3.0, "maxclust").to_host()
        var hf = fcluster(hz, 3.0, "maxclust").to_host()
        for i in range(n):
            assert_equal(Int(df[i]), Int(hf[i]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
