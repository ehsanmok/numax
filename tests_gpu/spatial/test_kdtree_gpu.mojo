"""`KDTree` over device data at `gpu=True`, against the host tree, at
`float32`: the nearest-neighbor and ball queries run one lane per query
point on the device, over the tree's arrays uploaded there."""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.testing import assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.spatial import KDTree

comptime f32 = DType.float32
comptime n = 2000
comptime q = 256


def _data(ctx: DeviceContext) raises -> Static[f32, n, 3]:
    var values = List[Scalar[f32]](capacity=n * 3)
    for i in range(n * 3):
        values.append(sin(Float32(i) * 1.7 + 0.3) * sin(Float32(i) * 0.31))
    return Static[f32, n, 3](values^, ctx)


def _queries(ctx: DeviceContext) raises -> Static[f32, q, 3]:
    var values = List[Scalar[f32]](capacity=q * 3)
    for i in range(q * 3):
        values.append(0.8 * sin(Float32(i) * 2.9 + 1.1))
    return Static[f32, q, 3](values^, ctx)


def test_kdtree_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var dt = KDTree[f32, gpu=True](_data(gpu))
    var ht = KDTree[f32](_data(cpu))
    var d = dt.query[4](_queries(gpu))
    var h = ht.query[4](_queries(cpu))
    assert_false(d.indices.on_host())
    var di = d.indices.to_host()
    var hi = h.indices.to_host()
    var dd = d.distances.to_host()
    var hd = h.distances.to_host()
    for i in range(q * 4):
        assert_equal(Int(di[i]), Int(hi[i]))
        assert_almost_equal(dd[i], hd[i], atol=1e-6)
    var db = dt.query_ball_point(_queries(gpu), 0.2)
    var hb = ht.query_ball_point(_queries(cpu), 0.2)
    var do = db.offsets.to_host()
    var ho = hb.offsets.to_host()
    for j in range(q + 1):
        assert_equal(Int(do[j]), Int(ho[j]))
    var dx = db.indices.to_host()
    var hx = hb.indices.to_host()
    for i in range(Int(ho[q])):
        assert_equal(Int(dx[i]), Int(hx[i]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
