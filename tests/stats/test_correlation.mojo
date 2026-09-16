"""Tests for `numax.stats.correlation`, each against NumPy's or SciPy's own
values on two 16-sample series: `cov` and `corrcoef` for a pair and for a
three-variable matrix, `pearsonr`, `spearmanr` and `kendalltau` with their
p-values, `linregress` with both standard errors, `rankdata` under all five
tie methods, and `zscore`.

`cov` and `corrcoef` moved onto MAX's `Welford` plus one GEMM in 0.2, so
they also carry the cases a matrix product can get wrong where a
triangle-mirroring host loop cannot: `ddof` 0, 1 and 2 with `bias`, a
single variable, two observations, and `float32` -- where the GEMM
reassociates the `n` products and the answer is only accurate to the
`float32` epsilon times the sum of squares, not to the last bit. The
`gpu=True`-on-a-CPU-tensor fallback is deliberately not pinned here:
naming `gpu=True` compiles a device kernel, which a GPU-less CI runner
cannot do, so `examples/advanced/unified_tensor_gpu.mojo` proves it
instead.
"""

from std.testing import TestSuite, assert_almost_equal, assert_true

from max.gpu.host import DeviceContext

from numax.core.array import Static
from numax.stats import (
    corrcoef,
    cov,
    kendalltau,
    linregress,
    pearsonr,
    rankdata,
    spearmanr,
    zscore,
)

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


def test_cov_and_corrcoef_match_numpy() raises:
    var x = _from[16](_x())
    var y = _from[16](_y())
    var c = cov(x, y).to_host()
    assert_almost_equal(Float64(c[0]), 2.56640625, atol=1e-13)
    assert_almost_equal(Float64(c[1]), 2.091666666666667, atol=1e-13)
    assert_almost_equal(Float64(c[2]), 2.091666666666667, atol=1e-13)
    assert_almost_equal(Float64(c[3]), 1.9133333333333333, atol=1e-13)
    var biased = cov(x, y, bias=True).to_host()
    assert_almost_equal(Float64(biased[0]), 2.406005859375, atol=1e-13)
    assert_almost_equal(Float64(biased[1]), 1.9609375, atol=1e-13)
    var r = corrcoef(x, y).to_host()
    assert_almost_equal(Float64(r[0]), 1.0, atol=1e-15)
    assert_almost_equal(Float64(r[1]), 0.9439184054429586, atol=1e-13)
    assert_almost_equal(Float64(r[3]), 1.0, atol=1e-15)


def test_cov_of_a_variable_matrix_matches_numpy() raises:
    """Three variables as rows -- `x`, `y`, `x * y` -- against `numpy.cov`
    with its default `rowvar=True`, and the matching `corrcoef`."""
    var xs = _x()
    var ys = _y()
    var rows = List[Scalar[dtype]](capacity=48)
    for i in range(16):
        rows.append(Scalar[dtype](xs[i]))
    for i in range(16):
        rows.append(Scalar[dtype](ys[i]))
    for i in range(16):
        rows.append(Scalar[dtype](xs[i] * ys[i]))
    var m = Static[dtype, 3, 16](_cpu(), rows^)
    var c = cov(m).to_host()
    var expected: List[Float64] = [
        2.56640625,
        2.091666666666667,
        0.9419531250000001,
        2.091666666666667,
        1.9133333333333333,
        1.3105000000000002,
        0.9419531250000001,
        1.3105000000000002,
        6.482351562500001,
    ]
    for i in range(9):
        assert_almost_equal(Float64(c[i]), expected[i], atol=1e-12)
    var r = corrcoef(m).to_host()
    assert_almost_equal(Float64(r[1]), 0.9439184054429586, atol=1e-13)
    assert_almost_equal(Float64(r[4]), 1.0, atol=1e-15)


def _matrix() raises -> Static[dtype, 3, 16]:
    """`x`, `y` and `x * y` as three variables of sixteen observations."""
    var xs = _x()
    var ys = _y()
    var rows = List[Scalar[dtype]](capacity=48)
    for i in range(16):
        rows.append(Scalar[dtype](xs[i]))
    for i in range(16):
        rows.append(Scalar[dtype](ys[i]))
    for i in range(16):
        rows.append(Scalar[dtype](xs[i] * ys[i]))
    return Static[dtype, 3, 16](_cpu(), rows^)


def test_cov_honours_ddof_and_bias() raises:
    """`numpy.cov(m, ddof=k)` at `k = 0` and `k = 2`, and `bias=True` as
    the alias for `ddof=0`. The divisor is the only thing that changes, so
    all three come off one centering and one GEMM."""
    var m = _matrix()
    var unbiased: List[Float64] = [
        2.406005859375,
        1.9609375000000002,
        0.8830810546875001,
        1.9609375000000002,
        1.79375,
        1.2285937500000002,
        0.8830810546875001,
        1.2285937500000002,
        6.077204589843751,
    ]
    var zero = cov(m, ddof=0).to_host()
    var biased = cov(m, bias=True).to_host()
    for i in range(9):
        assert_almost_equal(Float64(zero[i]), unbiased[i], atol=1e-12)
        assert_almost_equal(Float64(biased[i]), unbiased[i], atol=1e-12)
    var two: List[Float64] = [
        2.749720982142857,
        2.241071428571429,
        1.0092354910714285,
        2.241071428571429,
        2.05,
        1.404107142857143,
        1.0092354910714285,
        1.404107142857143,
        6.945376674107143,
    ]
    var by_two = cov(m, ddof=2).to_host()
    for i in range(9):
        assert_almost_equal(Float64(by_two[i]), two[i], atol=1e-12)
    var raised = False
    try:
        _ = cov(m, ddof=16)
    except:
        raised = True
    assert_true(raised)


def test_cov_at_one_variable_and_at_two_observations() raises:
    """The degenerate shapes the GEMM has to survive: a `1 x n` matrix is a
    one-by-one product, and a `rows x 2` one leaves a single degree of
    freedom."""
    var xs = _x()
    var single = List[Scalar[dtype]](capacity=16)
    for i in range(16):
        single.append(Scalar[dtype](xs[i]))
    var one = Static[dtype, 1, 16](_cpu(), single^)
    var c = cov(one).to_host()
    assert_almost_equal(Float64(c[0]), 2.56640625, atol=1e-13)
    var r = corrcoef(one).to_host()
    assert_almost_equal(Float64(r[0]), 1.0, atol=1e-15)

    var pair: List[Scalar[dtype]] = [
        Scalar[dtype](1.0),
        Scalar[dtype](3.0),
        Scalar[dtype](2.0),
        Scalar[dtype](-2.0),
    ]
    var narrow = Static[dtype, 2, 2](_cpu(), pair^)
    var nc = cov(narrow).to_host()
    _assert_pairs(nc, [2.0, -4.0, -4.0, 8.0], 1e-13)
    var nr = corrcoef(narrow).to_host()
    _assert_pairs(nr, [1.0, -1.0, -1.0, 1.0], 1e-14)


def _assert_pairs(
    got: List[Scalar[dtype]], want: List[Float64], atol: Float64
) raises:
    for i in range(len(want)):
        assert_almost_equal(Float64(got[i]), want[i], atol=atol)


def test_cov_at_float32_agrees_with_the_float64_answer() raises:
    """`float32`, where the GEMM's reassociation shows.

    The tolerance is `1e-5` absolute on entries of order one: the products
    summed are `O(n)` terms of order `x^2`, so the error floor is the
    `float32` epsilon (`1.2e-7`) times that sum, not the last bit of the
    result.
    """
    comptime f32 = DType.float32
    var xs = _x()
    var ys = _y()
    var rows = List[Scalar[f32]](capacity=32)
    for i in range(16):
        rows.append(Scalar[f32](xs[i]))
    for i in range(16):
        rows.append(Scalar[f32](ys[i]))
    var m = Static[f32, 2, 16](_cpu(), rows^)
    var c = cov(m).to_host()
    assert_almost_equal(Float64(c[0]), 2.56640625, atol=1e-5)
    assert_almost_equal(Float64(c[1]), 2.091666666666667, atol=1e-5)
    assert_almost_equal(Float64(c[2]), 2.091666666666667, atol=1e-5)
    assert_almost_equal(Float64(c[3]), 1.9133333333333333, atol=1e-5)
    var r = corrcoef(m).to_host()
    assert_almost_equal(Float64(r[0]), 1.0, atol=1e-15)
    assert_almost_equal(Float64(r[1]), 0.9439184054429586, atol=1e-6)


def test_corrcoef_of_a_variable_matrix_matches_numpy() raises:
    """Every entry of `numpy.corrcoef(m)`, not just the two the `cov` test
    spot-checks: the diagonal is exactly one and the matrix is symmetric to
    the GEMM's own reassociation."""
    var m = _matrix()
    var r = corrcoef(m).to_host()
    var want: List[Float64] = [
        1.0,
        0.9439184054429586,
        0.23094060475758754,
        0.9439184054429587,
        1.0,
        0.3721134913212128,
        0.23094060475758757,
        0.3721134913212128,
        1.0,
    ]
    for i in range(9):
        assert_almost_equal(Float64(r[i]), want[i], atol=1e-13)


def test_pearsonr_spearmanr_and_kendalltau_match_scipy() raises:
    """Statistics and two-sided p-values; `kendalltau` against SciPy's
    asymptotic method, which is what SciPy itself uses here since the
    data has ties."""
    var x = _from[16](_x())
    var y = _from[16](_y())
    var p = pearsonr(x, y)
    assert_almost_equal(p.statistic, 0.9439184054429587, atol=1e-13)
    assert_almost_equal(p.pvalue, 4.0310098732200055e-08, atol=1e-18)
    var s = spearmanr(x, y)
    assert_almost_equal(s.statistic, 0.9525936379634137, atol=1e-13)
    assert_almost_equal(s.pvalue, 1.2724117066198675e-08, atol=1e-18)
    var k = kendalltau(x, y)
    assert_almost_equal(k.statistic, 0.8448589801827239, atol=1e-13)
    assert_almost_equal(k.pvalue, 8.429890156632941e-06, atol=1e-15)


def test_linregress_matches_scipy() raises:
    var x = _from[16](_x())
    var y = _from[16](_y())
    var fit = linregress(x, y)
    assert_almost_equal(fit.slope, 0.8150177574835111, atol=1e-13)
    assert_almost_equal(fit.intercept, 0.10710299340436319, atol=1e-13)
    assert_almost_equal(fit.rvalue, 0.9439184054429587, atol=1e-13)
    assert_almost_equal(fit.pvalue, 4.0310098732200624e-08, atol=1e-18)
    assert_almost_equal(fit.stderr, 0.07619347680576888, atol=1e-13)
    assert_almost_equal(fit.intercept_stderr, 0.1213165795638522, atol=1e-13)


def test_rankdata_matches_scipy_under_every_method() raises:
    var x = _from[16](_x())
    _assert_close(
        rankdata(x),
        [
            11.0,
            6.5,
            3.5,
            14.0,
            8.0,
            16.0,
            2.0,
            11.0,
            9.0,
            5.0,
            13.0,
            15.0,
            3.5,
            6.5,
            11.0,
            1.0,
        ],
        atol=1e-15,
    )
    _assert_close(
        rankdata(x, "min"),
        [
            10.0,
            6.0,
            3.0,
            14.0,
            8.0,
            16.0,
            2.0,
            10.0,
            9.0,
            5.0,
            13.0,
            15.0,
            3.0,
            6.0,
            10.0,
            1.0,
        ],
        atol=1e-15,
    )
    _assert_close(
        rankdata(x, "max"),
        [
            12.0,
            7.0,
            4.0,
            14.0,
            8.0,
            16.0,
            2.0,
            12.0,
            9.0,
            5.0,
            13.0,
            15.0,
            4.0,
            7.0,
            12.0,
            1.0,
        ],
        atol=1e-15,
    )
    _assert_close(
        rankdata(x, "dense"),
        [
            8.0,
            5.0,
            3.0,
            10.0,
            6.0,
            12.0,
            2.0,
            8.0,
            7.0,
            4.0,
            9.0,
            11.0,
            3.0,
            5.0,
            8.0,
            1.0,
        ],
        atol=1e-15,
    )
    _assert_close(
        rankdata(x, "ordinal"),
        [
            10.0,
            6.0,
            3.0,
            14.0,
            8.0,
            16.0,
            2.0,
            11.0,
            9.0,
            5.0,
            13.0,
            15.0,
            4.0,
            7.0,
            12.0,
            1.0,
        ],
        atol=1e-15,
    )
    var raised = False
    try:
        _ = rankdata(x, "median")
    except:
        raised = True
    assert_true(raised)


def test_zscore_matches_scipy() raises:
    var x = _from[16](_x())
    _assert_close(
        zscore(x),
        [
            0.4130052215639246,
            -0.23168585599927477,
            -0.8763769335624741,
            1.057696299127124,
            0.09065968278232492,
            1.7023873766903235,
            -1.5210680111256736,
            0.4130052215639246,
            0.2518324521731248,
            -0.5540313947808745,
            0.7353507603455243,
            1.3800418379087236,
            -0.8763769335624741,
            -0.23168585599927477,
            0.4130052215639246,
            -2.165759088688873,
        ],
    )
    var sample = zscore(x, ddof=1).to_host()
    assert_almost_equal(Float64(sample[0]), 0.3998905862534461, atol=1e-13)
    assert_almost_equal(Float64(sample[2]), -0.8485483171719466, atol=1e-13)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
