"""Tests for `kruskal`, `levene` (both centers) and `bartlett` against
SciPy on three groups of six with ties across groups, which exercises
`kruskal`'s tie correction."""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import bartlett, kruskal, levene

comptime f64 = DType.float64


def _g(values: List[Float64]) raises -> Static[f64, 6]:
    var v = List[Scalar[f64]](capacity=6)
    for i in range(6):
        v.append(values[i])
    return Static[f64, 6](v^, DeviceContext(api="cpu"))


def test_group_tests_match_scipy() raises:
    var a = _g([2.9, 3.0, 2.5, 2.6, 3.2, 2.8])
    var b = _g([3.8, 2.7, 4.0, 2.4, 3.1, 3.3])
    var c = _g([2.8, 3.4, 3.7, 2.2, 2.0, 3.0])
    var r = kruskal(a, b, c)
    assert_almost_equal(r.statistic, 1.5118924508790075, atol=1e-12)
    assert_almost_equal(r.pvalue, 0.46956608907395236, atol=1e-12)
    assert_equal(r.df, 2.0)
    var l = levene(a, b, c)
    assert_almost_equal(l.statistic, 2.2489683631361754, atol=1e-12)
    assert_almost_equal(l.pvalue, 0.13988447512447388, atol=1e-12)
    var lmean = levene[center="mean"](a, b, c)
    assert_almost_equal(lmean.statistic, 2.283519553072625, atol=1e-12)
    assert_almost_equal(lmean.pvalue, 0.1362216371913614, atol=1e-12)
    var t = bartlett(a, b, c)
    assert_almost_equal(t.statistic, 3.8574787343013535, atol=1e-12)
    assert_almost_equal(t.pvalue, 0.1453312924535411, atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
