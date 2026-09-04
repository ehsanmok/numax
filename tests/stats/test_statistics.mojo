"""Tests for `numax.stats`.

The `Plain`-only, `TileTensor`-based functions are checked against
hand-computed expectations on a fixed small array (matching NumPy's
behavior, including its median-of-even-count averaging and
`scipy.stats.mode`'s smallest-among-ties convention). The `FloatLike`-
generic functions are checked the same way at `Plain`, and additionally
demonstrate the real axis-1 win at `Compensated`: `variance` over a large
array of many small additions matches a float64 reference far more closely
than `Plain` does, which is the whole point of writing these against the
trait instead of a concrete `dtype`.
"""

from std.math import sin
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax import Compensated, Plain
from numax.core.array import Tensor, full
from numax.core.numeric import FloatLike
from numax.stats import (
    argmax,
    argmin,
    cumprod,
    cumsum,
    max,
    mean,
    median,
    min,
    mode,
    prod,
    stddev,
    sum,
    variance,
)

comptime dtype = DType.float32


def _fixed_array() raises -> Tensor[dtype, 6]:
    var ctx = DeviceContext(api="cpu")
    var vals = [3.0, 1.0, 9.0, 2.0, 7.0, 2.0]
    var values = List[Scalar[dtype]](capacity=6)
    for i in range(6):
        values.append(Scalar[dtype](vals[i]))
    return Tensor[dtype, 6](ctx, values^)


def test_sum_matches_hand_computed_total() raises:
    var ctx = DeviceContext(api="cpu")
    var xs = _fixed_array()
    assert_almost_equal(sum(xs), Scalar[dtype](24))


def test_prod_matches_hand_computed_product() raises:
    var ctx = DeviceContext(api="cpu")
    var xs = _fixed_array()
    assert_almost_equal(prod(xs), Scalar[dtype](756))


def test_min_returns_the_smallest_element() raises:
    var ctx = DeviceContext(api="cpu")
    var xs = _fixed_array()
    assert_almost_equal(min(xs), Scalar[dtype](1))


def test_max_returns_the_largest_element() raises:
    var ctx = DeviceContext(api="cpu")
    var xs = _fixed_array()
    assert_almost_equal(max(xs), Scalar[dtype](9))


def test_mean_matches_hand_computed_average() raises:
    var ctx = DeviceContext(api="cpu")
    var xs = _fixed_array()
    assert_almost_equal(mean(xs), Scalar[dtype](4))


def test_median_of_an_even_count_averages_the_two_middle_values() raises:
    var ctx = DeviceContext(api="cpu")
    # sorted: [1, 2, 2, 3, 7, 9] -> middle two are 2 and 3
    var xs = _fixed_array()
    assert_almost_equal(median(xs), Scalar[dtype](2.5))


def test_median_of_an_odd_count_returns_the_middle_value() raises:
    var ctx = DeviceContext(api="cpu")
    var xs = full[dtype, 5](ctx, Scalar[dtype](0))
    var v = xs.view()
    var vals = [5.0, 1.0, 3.0, 2.0, 4.0]
    for i in range(5):
        v[i] = Scalar[dtype](vals[i])
    assert_almost_equal(median(xs), Scalar[dtype](3))


def test_mode_returns_the_most_frequent_value() raises:
    var ctx = DeviceContext(api="cpu")
    # sorted: [1, 2, 2, 3, 7, 9] -> 2 appears twice, everything else once
    var xs = _fixed_array()
    assert_almost_equal(mode(xs), Scalar[dtype](2))


def test_mode_returns_the_smallest_among_tied_values() raises:
    var ctx = DeviceContext(api="cpu")
    var xs = full[dtype, 4](ctx, Scalar[dtype](0))
    var v = xs.view()
    var vals = [5.0, 5.0, 1.0, 1.0]
    for i in range(4):
        v[i] = Scalar[dtype](vals[i])
    assert_almost_equal(mode(xs), Scalar[dtype](1))


def test_argmax_returns_the_index_of_the_largest_element() raises:
    var ctx = DeviceContext(api="cpu")
    var xs = _fixed_array()
    assert_equal(argmax(xs), 2)


def test_argmin_returns_the_index_of_the_smallest_element() raises:
    var ctx = DeviceContext(api="cpu")
    var xs = _fixed_array()
    assert_equal(argmin(xs), 1)


def test_cumprod_matches_the_running_product() raises:
    var ctx = DeviceContext(api="cpu")
    var xs = _fixed_array()
    var cp = cumprod(xs)
    var expected = [3.0, 3.0, 27.0, 54.0, 378.0, 756.0]
    for i in range(6):
        assert_almost_equal(cp[i], Scalar[dtype](expected[i]))


def _fixed_list[T: FloatLike]() -> List[T]:
    var lst = List[T](capacity=6)
    var vals = [3.0, 1.0, 9.0, 2.0, 7.0, 2.0]
    for i in range(6):
        lst.append(T.constant(vals[i]))
    return lst^


def test_floatlike_mean_matches_hand_computed_average() raises:
    var ctx = DeviceContext(api="cpu")
    var lst = _fixed_list[Plain[dtype, 1]]()
    assert_almost_equal(mean(lst).v, Scalar[dtype](4))


def test_floatlike_variance_defaults_to_population_variance() raises:
    var ctx = DeviceContext(api="cpu")
    # deviations from mean=4: -1,-3,5,-2,3,-2 -> squares sum to 52, /6
    var lst = _fixed_list[Plain[dtype, 1]]()
    assert_almost_equal(variance(lst).v, Scalar[dtype](52.0 / 6.0))


def test_floatlike_variance_honors_ddof() raises:
    var ctx = DeviceContext(api="cpu")
    var lst = _fixed_list[Plain[dtype, 1]]()
    assert_almost_equal(variance(lst, ddof=1).v, Scalar[dtype](52.0 / 5.0))


def test_floatlike_stddev_is_the_sqrt_of_variance() raises:
    var ctx = DeviceContext(api="cpu")
    var lst = _fixed_list[Plain[dtype, 1]]()
    assert_almost_equal(stddev(lst).v, Scalar[dtype]((52.0 / 6.0) ** 0.5))


def test_floatlike_cumsum_matches_the_running_sum() raises:
    var ctx = DeviceContext(api="cpu")
    var lst = _fixed_list[Plain[dtype, 1]]()
    var cs = cumsum(lst)
    var expected = [3.0, 4.0, 13.0, 15.0, 22.0, 24.0]
    for i in range(6):
        assert_almost_equal(cs[i].v, Scalar[dtype](expected[i]))


def test_compensated_variance_beats_plain_on_a_long_summation() raises:
    var ctx = DeviceContext(api="cpu")
    # Every value is close to 1.0 (so a single float32 stores each one
    # almost exactly), but summing half a million of them drives the
    # running accumulator up to where a single-precision `+=` starts
    # discarding real bits of each new term -- the textbook case
    # `Compensated` exists for. A float64 reference is the ground truth;
    # `Plain`'s variance should drift measurably from it, `Compensated`'s
    # should not.
    comptime n = 300_000
    var plain_list = List[Plain[dtype, 1]](capacity=n)
    var comp_list = List[Compensated[dtype, 1]](capacity=n)
    var f64_sum = Float64(0)
    var f64_sq_sum = Float64(0)
    for i in range(n):
        var x = Float64(1.0) + Float64(0.01) * Float64(sin(Float64(i)))
        plain_list.append(Plain[dtype, 1](Scalar[dtype](x)))
        comp_list.append(
            Compensated[dtype, 1](Scalar[dtype](x), Scalar[dtype](0))
        )
        f64_sum += x
        f64_sq_sum += x * x
    var f64_mean = f64_sum / Float64(n)
    var f64_var = f64_sq_sum / Float64(n) - f64_mean * f64_mean

    var plain_err = abs(Float64(variance(plain_list).v) - f64_var) / f64_var
    var comp_err = abs(Float64(variance(comp_list).value) - f64_var) / f64_var

    assert_true(
        comp_err < 1e-6,
        msg=String("compensated variance drifted too far: ", comp_err),
    )
    assert_true(
        plain_err > 1e-5,
        msg=String(
            (
                "plain variance didn't drift as expected -- test no longer"
                " demonstrates the axis-1 win: "
            ),
            plain_err,
        ),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
