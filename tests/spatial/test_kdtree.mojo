"""Tests for `KDTree` against brute force: `query` at several `k` and
leaf sizes (one point a leaf included) returns the same neighbors and
distances as sorting a `cdist` row, and `query_ball_point` the same sets
as thresholding one, sorted; plus the degenerate cases -- more neighbors
asked than points, and duplicate coordinates."""

from std.math import sin
from max.gpu.host import DeviceContext
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from numax.core.tensor import Static
from numax.spatial import KDTree, cdist

comptime f64 = DType.float64
comptime n = 500
comptime q = 40


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _data() raises -> Static[f64, n, 3]:
    var values = List[Float64](capacity=n * 3)
    for i in range(n * 3):
        values.append(sin(Float64(i) * 1.7 + 0.3) * sin(Float64(i) * 0.31))
    return Static[f64, n, 3](values^, _cpu())


def _queries() raises -> Static[f64, q, 3]:
    var values = List[Float64](capacity=q * 3)
    for i in range(q * 3):
        values.append(0.8 * sin(Float64(i) * 2.9 + 1.1))
    return Static[f64, q, 3](values^, _cpu())


def _check_knn[count: Int](leafsize: Int) raises:
    var tree = KDTree[f64](_data(), leafsize)
    var r = tree.query[count](_queries())
    var d = r.distances.to_host()
    var ids = r.indices.to_host()
    var full = cdist(_queries(), _data()).to_host()
    for j in range(q):
        var row = List[Float64](capacity=n)
        for i in range(n):
            row.append(full[j * n + i])
        # The `count` smallest by selection, for the reference.
        for c in range(count):
            var best = 0
            for i in range(1, n):
                if row[i] < row[best]:
                    best = i
            assert_almost_equal(d[j * count + c], row[best], atol=1e-14)
            assert_equal(Int(ids[j * count + c]), best)
            row[best] = 1e300


def test_query_matches_brute_force() raises:
    _check_knn[1](10)
    _check_knn[5](10)
    _check_knn[7](1)
    _check_knn[3](64)


def test_query_ball_point_matches_brute_force() raises:
    var tree = KDTree[f64](_data())
    var radius = 0.35
    var balls = tree.query_ball_point(_queries(), radius)
    var offsets = balls.offsets.to_host()
    var ids = balls.indices.to_host()
    var full = cdist(_queries(), _data()).to_host()
    for j in range(q):
        var expect = List[Int]()
        for i in range(n):
            if full[j * n + i] <= radius:
                expect.append(i)
        var lo = Int(offsets[j])
        var hi = Int(offsets[j + 1])
        assert_equal(hi - lo, len(expect))
        for t in range(len(expect)):
            assert_equal(Int(ids[lo + t]), expect[t])


def test_degenerate_cases() raises:
    var few = Static[f64, 3, 2]([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], _cpu())
    var tree = KDTree[f64](few, 1)
    var one = Static[f64, 1, 2]([0.1, 0.0], _cpu())
    var r = tree.query[5](one)
    var d = r.distances.to_host()
    var ids = r.indices.to_host()
    assert_almost_equal(d[0], 0.1, atol=1e-15)
    assert_almost_equal(d[1], 0.1, atol=1e-15)
    assert_true(Int(ids[0]) == 0 or Int(ids[0]) == 2)
    assert_equal(Int(ids[3]), -1)
    assert_true(d[4] > 1e300)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
