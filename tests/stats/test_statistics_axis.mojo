"""Tests for the axis-wise reductions in `numax.stats`.

Each is checked two ways: against the whole-tensor reduction applied to one
group by hand, and against the identity that folding every axis in turn is
the same as folding the whole tensor. The second is what catches an
`outer`/`length`/`inner` split that is right at one axis and wrong at
another, which a single spot check would not.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.array import Shaped, zeros, zeros_dyn
from numax.stats import max, mean, min, prod, sum

comptime dtype = DType.float64


def _ramp[*dims: Int]() raises -> Shaped[dtype, *dims]:
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, *dims](ctx)
    for i in range(a.size()):
        a[i] = Scalar[dtype](i + 1)
    return a^


def test_sum_axis_drops_the_axis_it_folded() raises:
    var a = _ramp[2, 3, 4]()

    var over_middle = sum[axis=1](a)
    assert_equal(type_of(over_middle).rank, 2)
    assert_equal(over_middle.dim[0](), 2)
    assert_equal(over_middle.dim[1](), 4)

    var over_last = sum[axis=2](a)
    assert_equal(over_last.dim[0](), 2)
    assert_equal(over_last.dim[1](), 3)


def test_sum_axis_sums_the_right_elements() raises:
    var a = _ramp[2, 3]()  # [[1, 2, 3], [4, 5, 6]]

    var down_columns = sum[axis=0](a)
    assert_equal(down_columns.size(), 3)
    for i in range(3):
        assert_almost_equal(down_columns[i], Scalar[dtype](5 + 2 * i))

    var across_rows = sum[axis=1](a)
    assert_equal(across_rows.size(), 2)
    assert_almost_equal(across_rows[0], Scalar[dtype](6.0))
    assert_almost_equal(across_rows[1], Scalar[dtype](15.0))


def test_folding_every_axis_matches_the_whole_tensor_sum() raises:
    var a = _ramp[2, 3, 4]()
    var whole = sum(a)
    for axis in range(3):
        var partial = Scalar[dtype](0)
        if axis == 0:
            var r = sum[axis=0](a)
            for i in range(r.size()):
                partial += r[i]
        elif axis == 1:
            var r = sum[axis=1](a)
            for i in range(r.size()):
                partial += r[i]
        else:
            var r = sum[axis=2](a)
            for i in range(r.size()):
                partial += r[i]
        assert_almost_equal(partial, whole)


def test_min_max_and_prod_along_an_axis() raises:
    var a = _ramp[2, 3]()  # [[1, 2, 3], [4, 5, 6]]

    var row_min = min[axis=1](a)
    assert_almost_equal(row_min[0], Scalar[dtype](1.0))
    assert_almost_equal(row_min[1], Scalar[dtype](4.0))

    var row_max = max[axis=1](a)
    assert_almost_equal(row_max[0], Scalar[dtype](3.0))
    assert_almost_equal(row_max[1], Scalar[dtype](6.0))

    var row_prod = prod[axis=1](a)
    assert_almost_equal(row_prod[0], Scalar[dtype](6.0))
    assert_almost_equal(row_prod[1], Scalar[dtype](120.0))


def test_mean_axis_is_the_sum_divided_by_the_axis_length() raises:
    var a = _ramp[3, 4]()
    var totals = sum[axis=0](a)
    var means = mean[axis=0](a)
    assert_equal(means.size(), totals.size())
    for i in range(means.size()):
        assert_almost_equal(means[i], totals[i] / 3)


def test_an_axis_reduction_accepts_a_run_time_shape() raises:
    # Extents come from the input rather than the type, so a tensor whose
    # shape the compiler cannot see folds the same way.
    var ctx = DeviceContext(api="cpu")
    var static_a = _ramp[2, 3]()
    var dynamic_a = zeros_dyn[dtype, 2](2, 3, ctx=ctx)
    for i in range(6):
        dynamic_a[i] = Scalar[dtype](i + 1)

    var from_static = sum[axis=1](static_a)
    var from_dynamic = sum[axis=1](dynamic_a)
    assert_equal(from_dynamic.size(), from_static.size())
    for i in range(from_static.size()):
        assert_almost_equal(from_dynamic[i], from_static[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
