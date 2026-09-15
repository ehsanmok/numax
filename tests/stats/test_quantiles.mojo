"""Tests for the order statistics, NaN-ignoring reductions and the
`ptp`/`average`/`moment` trio over `Tensor`.

Every expected value is NumPy's or SciPy's own on a 16-sample signal:
`numpy.quantile` under all thirteen methods at seven probabilities,
`percentile`, the `nan*` family with two NaNs planted, `scipy.stats.iqr`,
`scipy.stats.moment` at four orders and a given centre, `numpy.ptp` and
`numpy.average` with weights.

The 0.2 selection route adds the small-sample and tie cases a quickselect
can get wrong where a sort cannot -- one and two samples under all thirteen
methods, a sample that is all one value, a sample that is mostly ties --
and pins the vector overload on **both** sides of the threshold
`_select_route` sets (`3 m < log2 n`): at `n = 4096` two quantiles select
and eight sort, and both must equal the scalar form, which always selects.
"""

from std.builtin.sort import sort
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


def _check_one(method: StaticString) raises:
    """One sample: every method returns it, at every probability."""
    var x = _from[1]([2.5])
    _assert_close(
        quantile(x, _from[3]([0.0, 0.5, 1.0]), method), [2.5, 2.5, 2.5]
    )


def _check_two(method: StaticString, want: List[Float64]) raises:
    var x = _from[2]([3.0, -1.0])
    var qs = _from[5]([0.0, 0.25, 0.5, 0.75, 1.0])
    _assert_close(quantile(x, qs, method), want)


def test_quantile_at_one_and_two_samples() raises:
    """`n = 1` and `n = 2` against `numpy.quantile`, all thirteen methods.

    The degenerate sizes are where a selection differs from a sort if the
    index bookkeeping is wrong: at `n = 1` the upper neighbour is the lower
    one, and at `n = 2` every method that interpolates reaches both ends.
    """
    _check_one("linear")
    _check_one("lower")
    _check_one("higher")
    _check_one("midpoint")
    _check_one("nearest")
    _check_one("inverted_cdf")
    _check_one("hazen")
    _check_one("weibull")
    _check_one("median_unbiased")
    _check_one("normal_unbiased")
    _check_one("interpolated_inverted_cdf")
    _check_one("averaged_inverted_cdf")
    _check_one("closest_observation")
    _check_two("linear", [-1.0, 0.0, 1.0, 2.0, 3.0])
    _check_two("lower", [-1.0, -1.0, -1.0, -1.0, 3.0])
    _check_two("higher", [-1.0, 3.0, 3.0, 3.0, 3.0])
    _check_two("midpoint", [-1.0, 1.0, 1.0, 1.0, 3.0])
    _check_two("nearest", [-1.0, -1.0, -1.0, 3.0, 3.0])
    _check_two("inverted_cdf", [-1.0, -1.0, -1.0, 3.0, 3.0])
    _check_two("hazen", [-1.0, -1.0, 1.0, 3.0, 3.0])
    _check_two("weibull", [-1.0, -1.0, 1.0, 3.0, 3.0])
    _check_two("median_unbiased", [-1.0, -1.0, 1.0, 3.0, 3.0])
    _check_two("normal_unbiased", [-1.0, -1.0, 1.0, 3.0, 3.0])
    _check_two("interpolated_inverted_cdf", [-1.0, -1.0, -1.0, 1.0, 3.0])
    _check_two("averaged_inverted_cdf", [-1.0, -1.0, 1.0, 3.0, 3.0])
    _check_two("closest_observation", [-1.0, -1.0, -1.0, 3.0, 3.0])
    var two = _from[2]([3.0, -1.0])
    assert_almost_equal(Float64(iqr(two)), 2.0, atol=1e-15)


def test_quantile_with_ties_and_a_constant_sample() raises:
    """Duplicates are what a partition reorders most; the answer must not
    depend on which copy lands where."""
    var dups = _from[6]([1.0, 2.0, 1.0, 5.0, 2.0, 1.0])
    var qs = _from[5]([0.0, 0.2, 0.5, 0.8, 1.0])
    _assert_close(quantile(dups, qs), [1.0, 1.0, 1.5, 2.0, 5.0])
    _assert_close(quantile(dups, qs, "lower"), [1.0, 1.0, 1.0, 2.0, 5.0])
    var flat = _from[5]([7.0, 7.0, 7.0, 7.0, 7.0])
    assert_almost_equal(Float64(quantile(flat, 0.37)), 7.0, atol=1e-15)
    assert_almost_equal(Float64(iqr(flat)), 0.0, atol=1e-15)


def _ramp[n: Int]() -> List[Float64]:
    """A deterministic permutation of `0 .. n - 1` -- `1237` is odd, so it
    is invertible modulo a power of two -- which leaves the sample far from
    sorted while its order statistics stay exactly `0 .. n - 1`."""
    var out = List[Float64](capacity=n)
    for i in range(n):
        out.append(Float64((i * 1237) % n))
    return out^


def test_vector_quantile_agrees_with_the_scalar_form_on_both_routes() raises:
    """`n = 4096`, where `_select_route`'s `3 m < log2 n` puts `m = 2` on
    the selection route and `m = 8` on the sort route. The scalar overload
    always selects, so equality across all three pins the threshold as a
    performance switch and not a semantic one."""
    var x = _from[4096](_ramp[4096]())
    var few = _from[2]([0.25, 0.75])
    var many = _from[8]([0.0, 0.05, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0])
    var few_host = quantile(x, few).to_host()
    var many_host = quantile(x, many, "hazen").to_host()
    var few_q = few.to_host()
    var many_q = many.to_host()
    for i in range(2):
        assert_almost_equal(
            Float64(few_host[i]),
            Float64(quantile(x, Float64(few_q[i]))),
            atol=1e-12,
        )
    for i in range(8):
        assert_almost_equal(
            Float64(many_host[i]),
            Float64(quantile(x, Float64(many_q[i]), "hazen")),
            atol=1e-12,
        )
    # The eight-quantile call sorted; the two-quantile one did not. Both
    # must still see the same sample, so the extremes are the extremes.
    assert_almost_equal(Float64(many_host[0]), 0.0, atol=1e-15)
    assert_almost_equal(Float64(many_host[7]), 4095.0, atol=1e-15)


def test_quantile_of_a_near_constant_sample_stays_linear() raises:
    """A sample of `2^16` equal values with a single different one -- the
    shape a two-way partition cannot shrink.

    `std.builtin.sort.partition` takes 12 s on `2^18` of these (89 s when
    every value is equal) against 1 ms for distinct values, which is why
    `_select_pair` splits three ways. This is a timing regression as much
    as a value one: it runs in milliseconds or it does not finish.
    """
    var n = 1 << 16
    var values = List[Float64](capacity=n)
    for i in range(n):
        values.append(9.0 if i == 40000 else 4.0)
    var x = _from[1 << 16](values)
    assert_almost_equal(Float64(quantile(x, 0.0)), 4.0, atol=1e-15)
    assert_almost_equal(Float64(quantile(x, 0.5)), 4.0, atol=1e-15)
    assert_almost_equal(Float64(quantile(x, 1.0)), 9.0, atol=1e-15)
    assert_almost_equal(Float64(iqr(x)), 0.0, atol=1e-15)
    var flat = List[Float64](length=n, fill=2.0)
    var constant = _from[1 << 16](flat)
    assert_almost_equal(Float64(quantile(constant, 0.5)), 2.0, atol=1e-15)


def test_selection_agrees_with_a_full_sort_on_duplicate_heavy_data() raises:
    """2047 samples over 17 distinct values, read at 21 probabilities,
    against the order statistic taken off a sorted copy here in the test.

    Each call permutes the sample differently -- a selection leaves it
    unsorted -- so this pins that the answer does not depend on the
    permutation the previous call left behind.
    """
    comptime n = 2047
    var raw = List[Float64](capacity=n)
    for i in range(n):
        raw.append(Float64((i * 1237) % 17) - 8.0)
    var x = _from[n](raw)
    var ordered = List[Float64](capacity=n)
    for i in range(n):
        ordered.append(raw[i])
    sort(ordered)
    for step in range(21):
        var q = Float64(step) / 20.0
        var rank = Int(q * Float64(n - 1))
        assert_almost_equal(
            Float64(quantile(x, q, "lower")), ordered[rank], atol=1e-15
        )
    assert_almost_equal(
        Float64(quantile(x, 0.5)), ordered[(n - 1) // 2], atol=1e-15
    )


def test_nan_policies_survive_the_selection_route() raises:
    """A planted NaN propagates through both the scalar and the vector
    overload, and `nanquantile` ignores it on both routes."""
    var x = _with_nans()
    var propagated = quantile(x, _from[3]([0.1, 0.5, 0.9])).to_host()
    for i in range(3):
        assert_true(propagated[i] != propagated[i])
    var scalar = Float64(percentile(x, 50.0))
    assert_true(scalar != scalar)
    var by_percent = percentile(x, _from[2]([25.0, 50.0])).to_host()
    assert_true(by_percent[0] != by_percent[0])
    var ignored = nanquantile(x, _from[3]([0.25, 0.5, 0.75])).to_host()
    assert_almost_equal(Float64(ignored[0]), -0.75, atol=1e-15)
    assert_almost_equal(Float64(ignored[1]), 0.625, atol=1e-15)
    assert_almost_equal(Float64(ignored[2]), 1.0, atol=1e-15)
    var all_nan = List[Scalar[dtype]](length=4, fill=_nan[dtype]())
    var empty = Static[dtype, 4](_cpu(), all_nan^)
    var raised = False
    try:
        _ = nanquantile(empty, 0.5)
    except:
        raised = True
    assert_true(raised)


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
