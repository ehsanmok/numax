"""Tests for the order statistics, NaN-ignoring reductions and the
`ptp`/`average`/`moment` trio over `Tensor`.

Every expected value is NumPy's or SciPy's own on a 16-sample signal:
`numpy.quantile` under all thirteen methods at seven probabilities,
`percentile`, the `nan*` family with two NaNs planted, `scipy.stats.iqr`,
`scipy.stats.moment` at four orders and a given centre, `numpy.ptp` and
`numpy.average` with weights.
"""

from std.testing import TestSuite, assert_almost_equal, assert_true

from max.gpu.host import DeviceContext
from std.utils.numerics import nan as _nan

from numax.core.array import Static
from numax.stats import (
    average,
    iqr,
    moment,
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanpercentile,
    nanprod,
    nanquantile,
    nanstd,
    nansum,
    nanvar,
    percentile,
    ptp,
    quantile,
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


def _with_nans() raises -> Static[dtype, 16]:
    """`_x()` with positions 3 and 9 replaced by NaN."""
    var values = _x()
    var out = List[Scalar[dtype]](capacity=16)
    for i in range(16):
        out.append(
            _nan[dtype]() if (i == 3 or i == 9) else Scalar[dtype](values[i])
        )
    return Static[dtype, 16](_cpu(), out^)


def _probabilities() raises -> Static[dtype, 7]:
    return _from[7]([0.0, 0.1, 0.25, 0.5, 0.66, 0.9, 1.0])


def _assert_close[
    n: Int
](got: Static[dtype, n], want: List[Float64], atol: Float64 = 1e-12) raises:
    var values = got.to_host()
    for i in range(n):
        assert_almost_equal(Float64(values[i]), want[i], atol=atol)


def _check_method(method: StaticString, want: List[Float64]) raises:
    var x = _from[16](_x())
    _assert_close(quantile(x, _probabilities(), method), want)


def test_quantile_matches_numpy_under_every_method() raises:
    """`numpy.quantile(x, q, method=m)` for all thirteen methods at seven
    probabilities, the ends included."""
    _check_method("linear", [-3.0, -1.5, -0.625, 0.625, 1.0, 2.25, 3.0])
    _check_method("lower", [-3.0, -2.0, -1.0, 0.5, 1.0, 2.0, 3.0])
    _check_method("higher", [-3.0, -1.0, -0.5, 0.75, 1.0, 2.5, 3.0])
    _check_method("midpoint", [-3.0, -1.5, -0.75, 0.625, 1.0, 2.25, 3.0])
    _check_method("nearest", [-3.0, -1.0, -0.5, 0.75, 1.0, 2.5, 3.0])
    _check_method("inverted_cdf", [-3.0, -2.0, -1.0, 0.5, 1.0, 2.5, 3.0])
    _check_method("hazen", [-3.0, -1.9, -0.75, 0.625, 1.0, 2.45, 3.0])
    _check_method(
        "weibull", [-3.0, -2.3, -0.875, 0.625, 1.0, 2.6500000000000004, 3.0]
    )
    _check_method(
        "median_unbiased",
        [
            -3.0,
            -2.033333333333333,
            -0.7916666666666665,
            0.625,
            1.0,
            2.5166666666666666,
            3.0,
        ],
    )
    _check_method(
        "normal_unbiased", [-3.0, -2.0, -0.78125, 0.625, 1.0, 2.5, 3.0]
    )
    _check_method(
        "interpolated_inverted_cdf",
        [-3.0, -2.4, -1.0, 0.5, 1.0, 2.2, 3.0],
    )
    _check_method(
        "averaged_inverted_cdf", [-3.0, -2.0, -0.75, 0.625, 1.0, 2.5, 3.0]
    )
    _check_method("closest_observation", [-3.0, -2.0, -1.0, 0.5, 1.0, 2.0, 3.0])


def test_scalar_quantile_and_percentile_agree_with_the_vector_form() raises:
    var x = _from[16](_x())
    assert_almost_equal(Float64(quantile(x, 0.5)), 0.625, atol=1e-15)
    assert_almost_equal(Float64(percentile(x, 30.0)), -0.25, atol=1e-15)
    var many = percentile(x, _from[2]([30.0, 50.0])).to_host()
    assert_almost_equal(Float64(many[0]), -0.25, atol=1e-15)
    assert_almost_equal(Float64(many[1]), 0.625, atol=1e-15)
    var raised = False
    try:
        _ = quantile(x, 0.5, "trimmed")
    except:
        raised = True
    assert_true(raised)


def test_nan_reductions_match_numpy() raises:
    """Two NaNs planted at positions 3 and 9: every `nan*` reduction against
    NumPy's on the same data."""
    var x = _with_nans()
    assert_almost_equal(Float64(nansum(x)), 4.25, atol=1e-13)
    assert_almost_equal(Float64(nanmean(x)), 0.30357142857142855, atol=1e-13)
    assert_almost_equal(Float64(nanmin(x)), -3.0, atol=1e-15)
    assert_almost_equal(Float64(nanmax(x)), 3.0, atol=1e-15)
    assert_almost_equal(Float64(nanvar(x)), 2.501594387755102, atol=1e-12)
    assert_almost_equal(Float64(nanstd(x)), 1.5816429394003888, atol=1e-12)
    assert_almost_equal(Float64(nanprod(x)), 0.0, atol=1e-15)
    assert_almost_equal(Float64(nanmedian(x)), 0.625, atol=1e-15)
    var quartiles = nanquantile(x, _from[2]([0.25, 0.5])).to_host()
    assert_almost_equal(Float64(quartiles[0]), -0.75, atol=1e-15)
    assert_almost_equal(Float64(quartiles[1]), 0.625, atol=1e-15)
    assert_almost_equal(
        Float64(nanpercentile(x, 90.0)), 2.200000000000001, atol=1e-13
    )


def test_nan_reductions_agree_with_the_plain_ones_without_nans() raises:
    """On data with no NaN the `nan*` forms are the plain reductions: the
    mask changes nothing and the convenience agrees with the primitive."""
    var x = _from[16](_x())
    assert_almost_equal(Float64(nansum(x)), 5.75, atol=1e-13)
    assert_almost_equal(Float64(nanmean(x)), 0.359375, atol=1e-13)
    assert_almost_equal(Float64(nanvar(x, ddof=1)), 2.56640625, atol=1e-12)
    assert_almost_equal(Float64(nanmedian(x)), 0.625, atol=1e-15)


def test_all_nan_input_raises_where_numpy_returns_nan() raises:
    var values = List[Scalar[dtype]](length=4, fill=_nan[dtype]())
    var x = Static[dtype, 4](_cpu(), values^)
    assert_almost_equal(Float64(nansum(x)), 0.0, atol=1e-15)
    assert_almost_equal(Float64(nanprod(x)), 1.0, atol=1e-15)
    var raised = False
    try:
        _ = nanmean(x)
    except:
        raised = True
    assert_true(raised)


def test_iqr_matches_scipy() raises:
    var x = _from[16](_x())
    assert_almost_equal(Float64(iqr(x)), 1.75, atol=1e-15)
    assert_almost_equal(Float64(iqr(x, "midpoint")), 2.0, atol=1e-15)
    var with_nans = _with_nans()
    var propagated = Float64(iqr(with_nans))
    assert_true(propagated != propagated)
    assert_almost_equal(
        Float64(iqr(with_nans, nan_policy="omit")), 1.75, atol=1e-15
    )
    var quantile_nan = Float64(quantile(with_nans, 0.5))
    assert_true(quantile_nan != quantile_nan)


def test_ptp_average_and_moment_match_numpy_and_scipy() raises:
    """`numpy.ptp`, `numpy.average` with weights, and `scipy.stats.moment`
    at orders 1-4 about the mean and at order 2 about a given centre."""
    var x = _from[16](_x())
    assert_almost_equal(Float64(ptp(x)), 6.0, atol=1e-15)
    var weights = _from[16](
        [
            1.0,
            2.0,
            0.5,
            1.0,
            3.0,
            1.0,
            0.25,
            2.0,
            1.0,
            1.0,
            0.5,
            2.0,
            1.0,
            1.5,
            1.0,
            0.75,
        ]
    )
    assert_almost_equal(
        Float64(average(x, weights)), 0.6282051282051282, atol=1e-13
    )
    assert_almost_equal(Float64(average(x)), 0.359375, atol=1e-13)
    assert_almost_equal(Float64(moment(x, 1)), 0.0, atol=1e-15)
    assert_almost_equal(Float64(moment(x, 2)), 2.406005859375, atol=1e-12)
    assert_almost_equal(Float64(moment(x, 3)), -1.3640213012695312, atol=1e-12)
    assert_almost_equal(Float64(moment(x, 4)), 15.30258160829544, atol=1e-11)
    assert_almost_equal(
        Float64(moment(x, 2, center=1.0)), 2.81640625, atol=1e-12
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
