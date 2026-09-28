"""Tests for `skewtest`, `kurtosistest`, `normaltest` and `jarque_bera`
against SciPy on a skewed, heavy-shouldered sample of 40, one-sided
`alternative` included."""

from std.testing import TestSuite, assert_almost_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import jarque_bera, kurtosistest, normaltest, skewtest

comptime f64 = DType.float64


def _x() raises -> Static[f64, 40]:
    return Static[f64, 40](
        [
            0.3,
            1.3884353744753821,
            2.1708994599769205,
            2.026418733297748,
            1.0699763003118101,
            0.09843354462076032,
            -1.1431515448271754,
            -1.2649052252486652,
            -0.4625332757446432,
            0.9336278009686995,
            2.613973197437578,
            3.0763364677540004,
            2.909197816176563,
            1.9381967246987042,
            0.6670417414961466,
            0.04060848005665968,
            -0.3583554583026347,
            0.46372577552593075,
            1.8672460944422735,
            3.2391395243932024,
            4.28121471138974,
            3.791493662285869,
            2.8062367134914084,
            1.5358571656319886,
            0.6248659328369945,
            0.8487479890636849,
            1.3903343551874319,
            2.8008453756136227,
            4.163927240136266,
            4.885532811671814,
            4.973311277072112,
            3.674105302655457,
            2.4048886337571345,
            1.5096252643606398,
            1.4564031085122722,
            2.617284940269751,
            3.73441614505095,
            5.088329336504485,
            5.789289547755676,
            5.554655801190762,
        ],
        DeviceContext(api="cpu"),
    )


def test_normality_tests_match_scipy() raises:
    var s = skewtest(_x())
    assert_almost_equal(s.statistic, 0.5158568769140337, atol=1e-11)
    assert_almost_equal(s.pvalue, 0.6059543744634908, atol=1e-11)
    var sl = skewtest(_x(), "less")
    assert_almost_equal(sl.pvalue, 0.6970228127682546, atol=1e-11)
    var k = kurtosistest(_x())
    assert_almost_equal(k.statistic, -1.0047052143238009, atol=1e-11)
    assert_almost_equal(k.pvalue, 0.3150388166052628, atol=1e-11)
    var n = normaltest(_x())
    assert_almost_equal(n.statistic, 1.2755408851489352, atol=1e-10)
    assert_almost_equal(n.pvalue, 0.5284693643229308, atol=1e-11)
    var j = jarque_bera(_x())
    assert_almost_equal(j.statistic, 1.062560007938165, atol=1e-10)
    assert_almost_equal(j.pvalue, 0.5878520349634584, atol=1e-11)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
