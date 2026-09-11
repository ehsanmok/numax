"""Tests for `numax.stats.histograms`, each against NumPy's own values on a
16-sample signal: `histogram` with default, fixed-range, explicit-edge,
density and weighted bins, `histogram2d` and `histogramdd` on the same
pairs, `bincount` with `minlength` and weights, and `digitize` on
increasing and decreasing edges with both sides.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax.core.array import Static
from numax.stats import bincount, digitize, histogram, histogram2d, histogramdd

comptime dtype = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _from[n: Int](values: List[Float64]) raises -> Static[dtype, n]:
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype](values[i]))
    return Static[dtype, n](_cpu(), out^)


def _x() -> List[Float64]:
    return [
        1.0,
        0.0,
        -1.0,
        2.0,
        0.5,
        3.0,
        -2.0,
        1.0,
        0.75,
        -0.5,
        1.5,
        2.5,
        -1.0,
        0.0,
        1.0,
        -3.0,
    ]


def _y() -> List[Float64]:
    return [
        0.5,
        0.2,
        -0.8,
        2.5,
        0.0,
        2.0,
        -1.5,
        1.2,
        0.5,
        -1.0,
        1.0,
        3.0,
        -0.5,
        0.5,
        0.8,
        -2.0,
    ]


def _assert_close[
    n: Int
](got: Static[dtype, n], want: List[Float64], atol: Float64 = 1e-12) raises:
    var values = got.to_host()
    for i in range(n):
        assert_almost_equal(Float64(values[i]), want[i], atol=atol)


def test_histogram_matches_numpy() raises:
    """`numpy.histogram(x)` with its ten default bins over the data range,
    the sample at the top edge counted in the last bin."""
    var x = _from[16](_x())
    var h = histogram(x)
    _assert_close(h.counts, [1.0, 1.0, 0.0, 2.0, 1.0, 3.0, 4.0, 1.0, 1.0, 2.0])
    _assert_close(
        h.edges,
        [
            -3.0,
            -2.4,
            -1.8,
            -1.2000000000000002,
            -0.6000000000000001,
            0.0,
            0.5999999999999996,
            1.2000000000000002,
            1.7999999999999998,
            2.3999999999999995,
            3.0,
        ],
        atol=1e-15,
    )


def test_histogram_with_range_edges_density_and_weights() raises:
    """Five bins over a given range, explicit non-uniform edges, `density`
    normalizing to unit area, and weights summed per bin."""
    var x = _from[16](_x())
    var ranged = histogram[bins=5](x, low=-3.0, high=3.0)
    _assert_close(ranged.counts, [2.0, 2.0, 4.0, 5.0, 3.0])
    var edges = _from[5]([-3.0, -1.0, 0.0, 1.0, 3.5])
    var explicit = histogram(x, edges)
    _assert_close(explicit.counts, [2.0, 3.0, 4.0, 7.0])
    var dense = histogram[bins=4](x, density=True)
    _assert_close(
        dense.counts,
        [0.08333333333333333, 0.125, 0.2916666666666667, 0.16666666666666666],
    )
    _assert_close(dense.edges, [-3.0, -1.5, 0.0, 1.5, 3.0])
    var weights = List[Float64]()
    for i in range(16):
        weights.append(Float64(i))
    var weighted = histogram[bins=4](x, _from[16](weights))
    _assert_close(weighted.counts, [21.0, 23.0, 47.0, 29.0])


def test_histogram2d_and_histogramdd_agree_with_numpy() raises:
    """`numpy.histogram2d(x, y, bins=(3, 4))`: counts and both edge sets,
    and `histogramdd` on the same pairs gives the same grid."""
    var x = _from[16](_x())
    var y = _from[16](_y())
    var joint = histogram2d[xbins=3, ybins=4](x, y)
    var expected: List[Float64] = [
        2.0,
        0.0,
        0.0,
        0.0,
        2.0,
        3.0,
        2.0,
        0.0,
        0.0,
        0.0,
        4.0,
        3.0,
    ]
    var counts = joint.counts.to_host()
    for i in range(12):
        assert_almost_equal(Float64(counts[i]), expected[i], atol=1e-15)
    _assert_close(joint.xedges, [-3.0, -1.0, 1.0, 3.0])
    _assert_close(joint.yedges, [-2.0, -0.75, 0.5, 1.75, 3.0])

    var pairs = List[Scalar[dtype]](capacity=32)
    var xs = _x()
    var ys = _y()
    for i in range(16):
        pairs.append(Scalar[dtype](xs[i]))
        pairs.append(Scalar[dtype](ys[i]))
    var points = Static[dtype, 16, 2](_cpu(), pairs^)
    var grid = histogramdd[3, 4](points)
    var dd = grid.counts.to_host()
    for i in range(12):
        assert_almost_equal(Float64(dd[i]), expected[i], atol=1e-15)
    assert_equal(len(grid.edges), 2)
    assert_almost_equal(grid.edges[1][1], -0.75, atol=1e-15)


def test_bincount_matches_numpy() raises:
    var values = List[Scalar[DType.int64]]()
    for v in [0, 1, 1, 3, 2, 1, 7]:
        values.append(Int64(v))
    var xs = Static[DType.int64, 7](_cpu(), values^)
    var counts = bincount(xs).to_host()
    var expected: List[Int] = [1, 3, 1, 1, 0, 0, 0, 1]
    assert_equal(len(counts), 8)
    for i in range(8):
        assert_equal(Int(counts[i]), expected[i])

    var short = List[Scalar[DType.int64]]()
    for v in [0, 1, 1, 3]:
        short.append(Int64(v))
    var small = Static[DType.int64, 4](_cpu(), short^)
    var padded = bincount(small, minlength=6).to_host()
    assert_equal(len(padded), 6)
    assert_equal(Int(padded[1]), 2)
    assert_equal(Int(padded[5]), 0)

    var w = _from[4]([0.5, 1.0, 2.0, 0.25])
    var weighted = bincount(small, w).to_host()
    var expected_w: List[Float64] = [0.5, 3.0, 0.0, 0.25]
    for i in range(4):
        assert_almost_equal(Float64(weighted[i]), expected_w[i], atol=1e-15)


def test_digitize_matches_numpy() raises:
    """Increasing edges with both `right` settings, and decreasing edges."""
    var x = _from[16](_x())
    var bins = _from[4]([-2.0, 0.0, 1.0, 2.0])
    var expected: List[Int] = [3, 2, 1, 4, 2, 4, 1, 3, 2, 1, 3, 4, 1, 2, 3, 0]
    var got = digitize(x, bins).to_host()
    for i in range(16):
        assert_equal(Int(got[i]), expected[i])
    var expected_right: List[Int] = [
        2,
        1,
        1,
        3,
        2,
        4,
        0,
        2,
        2,
        1,
        3,
        4,
        1,
        1,
        2,
        0,
    ]
    var right = digitize(x, bins, right=True).to_host()
    for i in range(16):
        assert_equal(Int(right[i]), expected_right[i])
    var descending = _from[4]([2.0, 1.0, 0.0, -2.0])
    var expected_desc: List[Int] = [
        1,
        2,
        3,
        0,
        2,
        0,
        3,
        1,
        2,
        3,
        1,
        0,
        3,
        2,
        1,
        4,
    ]
    var desc = digitize(x, descending).to_host()
    for i in range(16):
        assert_equal(Int(desc[i]), expected_desc[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
