"""Tests for the axis-wise reductions in `numax.stats`.

Each is checked two ways: against the whole-tensor reduction applied to one
group by hand, and against the identity that folding every axis in turn is
the same as folding the whole tensor. The second is what catches an
`outer`/`length`/`inner` split that is right at one axis and wrong at
another, which a single spot check would not.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.array import Static, zeros, zeros_dyn
from numax.stats import (
    argmax,
    argmin,
    max,
    mean,
    median,
    min,
    mode,
    prod,
    sum,
)

comptime dtype = DType.float64


def _ramp[*dims: Int]() raises -> Static[dtype, *dims]:
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


def _grid[
    rows: Int, cols: Int
](values: List[Float64]) raises -> Static[dtype, rows, cols]:
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, rows, cols](ctx)
    for i in range(rows * cols):
        a[i] = Scalar[dtype](values[i])
    return a^


def test_median_along_an_axis_matches_the_whole_tensor_one_per_slice() raises:
    # Rows [1, 9, 3] and [7, 2, 5]: medians 3 and 5. Odd length, so no
    # averaging; the even case is the next test.
    var a = _grid[2, 3]([1.0, 9.0, 3.0, 7.0, 2.0, 5.0])

    var per_row = median[axis=1](a)
    assert_equal(per_row.size(), 2)
    assert_almost_equal(per_row[0], Scalar[dtype](3.0))
    assert_almost_equal(per_row[1], Scalar[dtype](5.0))

    # Columns [1, 7], [9, 2], [3, 5] average their two middles.
    var per_column = median[axis=0](a)
    assert_equal(per_column.size(), 3)
    assert_almost_equal(per_column[0], Scalar[dtype](4.0))
    assert_almost_equal(per_column[1], Scalar[dtype](5.5))
    assert_almost_equal(per_column[2], Scalar[dtype](4.0))


def test_median_along_an_axis_agrees_with_median_of_one_slice() raises:
    # The claim worth pinning: the axis form and the whole-tensor form are
    # one algorithm, so a slice reduced by each has to agree.
    var a = _grid[2, 4]([1.0, 9.0, 3.0, 4.0, 7.0, 2.0, 5.0, 8.0])
    var row0 = _grid[1, 4]([1.0, 9.0, 3.0, 4.0])

    var per_row = median[axis=1](a)
    assert_almost_equal(per_row[0], median(row0))


def test_mode_along_an_axis_picks_the_smallest_among_ties() raises:
    # Row 0 has 2 twice; row 1 is all distinct, so its mode is the smallest.
    var a = _grid[2, 4]([5.0, 2.0, 2.0, 9.0, 4.0, 1.0, 7.0, 3.0])

    var per_row = mode[axis=1](a)
    assert_almost_equal(per_row[0], Scalar[dtype](2.0))
    assert_almost_equal(per_row[1], Scalar[dtype](1.0))


def test_argmax_along_the_innermost_axis_is_the_position_in_the_slice() raises:
    # numpy.argmax([[1, 9, 3], [7, 2, 5]], axis=1) == [1, 0]. The innermost
    # axis is the one `nn.argmaxmin` handles, so this is the delegated path.
    var a = _grid[2, 3]([1.0, 9.0, 3.0, 7.0, 2.0, 5.0])

    var per_row = argmax[axis=1](a)
    assert_equal(per_row.size(), 2)
    assert_equal(per_row[0], 1)
    assert_equal(per_row[1], 0)


def test_argmax_along_an_outer_axis_takes_the_walked_path() raises:
    # numpy.argmax(..., axis=0) == [1, 0, 1]. `nn.argmaxmin` raises on any
    # axis but the innermost, so this exercises numax's own walk -- and the
    # two paths have to agree on what an index means.
    var a = _grid[2, 3]([1.0, 9.0, 3.0, 7.0, 2.0, 5.0])

    var per_column = argmax[axis=0](a)
    assert_equal(per_column.size(), 3)
    assert_equal(per_column[0], 1)
    assert_equal(per_column[1], 0)
    assert_equal(per_column[2], 1)


def test_argmin_along_both_axes() raises:
    var a = _grid[2, 3]([1.0, 9.0, 3.0, 7.0, 2.0, 5.0])

    var per_row = argmin[axis=1](a)
    assert_equal(per_row[0], 0)
    assert_equal(per_row[1], 1)

    var per_column = argmin[axis=0](a)
    assert_equal(per_column[0], 0)
    assert_equal(per_column[1], 1)
    assert_equal(per_column[2], 0)


def test_argmax_takes_the_first_of_a_tie_on_both_paths() raises:
    # NumPy returns the first maximum. Both the delegated innermost path and
    # numax's own walk have to say so, or the same function means two things
    # depending on which axis it was handed.
    var a = _grid[2, 3]([1.0, 9.0, 9.0, 7.0, 7.0, 5.0])

    var per_row = argmax[axis=1](a)
    assert_equal(per_row[0], 1)
    assert_equal(per_row[1], 0)

    var tied_columns = _grid[2, 2]([4.0, 1.0, 4.0, 2.0])
    var per_column = argmax[axis=0](tied_columns)
    assert_equal(per_column[0], 0)
    assert_equal(per_column[1], 1)


def test_axis_reductions_that_need_a_whole_slice_handle_rank_three() raises:
    # The `outer`/`length`/`inner` split is what a middle axis at rank 3
    # tests and a rank-2 spot check does not.
    var a = _ramp[2, 3, 4]()

    var mid = median[axis=1](a)
    assert_equal(type_of(mid).rank, 2)
    assert_equal(mid.dim[0](), 2)
    assert_equal(mid.dim[1](), 4)
    # Element (0, 0) folds a[0, 0, 0], a[0, 1, 0], a[0, 2, 0] = 1, 5, 9.
    assert_almost_equal(mid[0], Scalar[dtype](5.0))

    var mid_arg = argmax[axis=1](a)
    assert_equal(mid_arg.dim[0](), 2)
    assert_equal(mid_arg.dim[1](), 4)
    assert_equal(mid_arg[0], 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
