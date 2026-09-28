"""Tests for `fisher_exact`, `binomtest` and `chi2_contingency` against
SciPy: Fisher's test on four tables under all three alternatives (the
two-sided cases on both sides of the mode), the binomial test at six
`(k, n, p)` including both sides of the mean and large `n`, and the
contingency test with and without Yates's correction and on a 3x3."""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import binomtest, chi2_contingency, fisher_exact

comptime f64 = DType.float64


def _near(got: Float64, want: Float64) raises:
    assert_almost_equal(got, want, atol=1e-12, rtol=1e-10)


def _fisher(
    t: List[Float64], alternative: StaticString, odds: Float64, p: Float64
) raises:
    var table = Static[f64, 2, 2](
        [t[0], t[1], t[2], t[3]], DeviceContext(api="cpu")
    )
    var r = fisher_exact(table, alternative)
    _near(r.statistic, odds)
    _near(r.pvalue, p)


def test_fisher_exact_matches_scipy() raises:
    _fisher([8.0, 2.0, 1.0, 5.0], "two-sided", 20.0, 0.034965034965034975)
    _fisher([8.0, 2.0, 1.0, 5.0], "less", 20.0, 0.9991258741258742)
    _fisher([8.0, 2.0, 1.0, 5.0], "greater", 20.0, 0.024475524475524483)
    _fisher(
        [3.0, 9.0, 10.0, 4.0],
        "two-sided",
        0.13333333333333333,
        0.04717997038632386,
    )
    _fisher(
        [3.0, 9.0, 10.0, 4.0], "less", 0.13333333333333333, 0.02358998519316193
    )
    _fisher(
        [3.0, 9.0, 10.0, 4.0],
        "greater",
        0.13333333333333333,
        0.9975837932426975,
    )
    _fisher([20.0, 15.0, 5.0, 30.0], "two-sided", 8.0, 0.00036743402761995746)
    _fisher([20.0, 15.0, 5.0, 30.0], "less", 8.0, 0.9999796078301716)
    _fisher([20.0, 15.0, 5.0, 30.0], "greater", 8.0, 0.0001837170138099787)
    _fisher(
        [1.0, 9.0, 11.0, 3.0],
        "two-sided",
        0.030303030303030304,
        0.0027594561852200836,
    )
    _fisher(
        [1.0, 9.0, 11.0, 3.0],
        "less",
        0.030303030303030304,
        0.0013797280926100418,
    )
    _fisher(
        [1.0, 9.0, 11.0, 3.0],
        "greater",
        0.030303030303030304,
        0.9999663480953022,
    )


def test_binomtest_matches_scipy() raises:
    _near(binomtest(7, 20, 0.3, "two-sided").pvalue, 0.6294979666766769)
    _near(binomtest(7, 20, 0.3, "less").pvalue, 0.7722717974181608)
    _near(binomtest(7, 20, 0.3, "greater").pvalue, 0.3919901877990756)
    _near(binomtest(2, 20, 0.3, "two-sided").pvalue, 0.05262794872972706)
    _near(binomtest(2, 20, 0.3, "less").pvalue, 0.03548313229846864)
    _near(binomtest(2, 20, 0.3, "greater").pvalue, 0.9923627402258)
    _near(binomtest(15, 20, 0.3, "two-sided").pvalue, 4.294002195359168e-05)
    _near(binomtest(15, 20, 0.3, "less").pvalue, 0.9999944497469218)
    _near(binomtest(15, 20, 0.3, "greater").pvalue, 4.294002195359168e-05)
    _near(binomtest(6, 20, 0.3, "two-sided").pvalue, 1.0)
    _near(binomtest(6, 20, 0.3, "less").pvalue, 0.6080098122009244)
    _near(binomtest(6, 20, 0.3, "greater").pvalue, 0.5836291705525192)
    _near(binomtest(500, 1000, 0.47, "two-sided").pvalue, 0.06155280216842457)
    _near(binomtest(500, 1000, 0.47, "less").pvalue, 0.9732697265430892)
    _near(binomtest(500, 1000, 0.47, "greater").pvalue, 0.03088642065088763)
    _near(binomtest(50, 60, 0.9, "two-sided").pvalue, 0.08684258904970954)
    _near(binomtest(50, 60, 0.9, "less").pvalue, 0.07306551008369888)
    _near(binomtest(50, 60, 0.9, "greater").pvalue, 0.9657908769212671)


def test_chi2_contingency_matches_scipy() raises:
    var cpu = DeviceContext(api="cpu")
    var two = chi2_contingency(Static[f64, 2, 2]([10.0, 20.0, 30.0, 25.0], cpu))
    _near(two.statistic, 2.706155303030302)
    _near(two.pvalue, 0.0999616438735348)
    assert_equal(two.dof, 1)
    var raw = chi2_contingency(
        Static[f64, 2, 2]([10.0, 20.0, 30.0, 25.0], cpu), False
    )
    _near(raw.statistic, 3.505892255892255)
    var three = chi2_contingency(
        Static[f64, 3, 3](
            [12.0, 7.0, 9.0, 5.0, 14.0, 11.0, 8.0, 6.0, 20.0], cpu
        )
    )
    _near(three.statistic, 11.741035895839817)
    _near(three.pvalue, 0.019384559113388485)
    assert_equal(three.dof, 4)
    var e = three.expected_freq.to_host()
    var want = [
        7.608695652173913,
        8.217391304347826,
        12.173913043478262,
        8.152173913043478,
        8.804347826086957,
        13.043478260869565,
        9.23913043478261,
        9.978260869565217,
        14.782608695652174,
    ]
    for i in range(9):
        _near(Float64(e[i]), want[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
