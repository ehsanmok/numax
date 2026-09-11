"""Tests for `numax.core.sorting`.

Checked against NumPy's exact outputs, including the conventions that are
easy to get subtly wrong: `searchsorted`'s left-side tie-breaking,
`argsort`'s stability on duplicates, and `unique`'s packed-into-a-full-length
tensor return.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax.core.array import Static, Tensor, asarray, diagflat, ravel, zeros
from numax.core.logic import greater
from numax.core.sorting import (
    all_nonzero,
    any_nonzero,
    argsort,
    count_nonzero,
    extract,
    nonzero,
    searchsorted,
    sort,
    take,
    take_along_axis,
    top_k,
    unique,
    select,
)

comptime dtype = DType.float64


def mk[n: Int](values: List[Float64]) raises -> Static[dtype, n]:
    var ctx = DeviceContext(api="cpu")
    var elements = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        elements.append(Scalar[dtype](values[i]))
    return Static[dtype, n](ctx, elements^)


def mk_mask[n: Int](values: List[Bool]) raises -> Static[DType.bool, n]:
    var ctx = DeviceContext(api="cpu")
    var elements = List[Scalar[DType.bool]](capacity=n)
    for i in range(n):
        elements.append(Scalar[DType.bool](values[i]))
    return Static[DType.bool, n](ctx, elements^)


# ------------------------------------------------------------------
# sort / argsort
# ------------------------------------------------------------------


def test_sort_orders_ascending() raises:
    var ctx = DeviceContext(api="cpu")
    var a = mk[5]([3.0, 1.0, 4.0, 1.0, 5.0])
    var out = sort(a)
    var expected = [1.0, 1.0, 3.0, 4.0, 5.0]
    for i in range(5):
        assert_almost_equal(out[i], Scalar[dtype](expected[i]))


def test_sort_of_an_already_sorted_tensor_is_unchanged() raises:
    var ctx = DeviceContext(api="cpu")
    var a = mk[4]([1.0, 2.0, 3.0, 4.0])
    var out = sort(a)
    for i in range(4):
        assert_almost_equal(out[i], a[i])


def test_sort_handles_negatives_and_zero() raises:
    var ctx = DeviceContext(api="cpu")
    var a = mk[5]([0.0, -2.5, 3.0, -0.5, 1.0])
    var out = sort(a)
    var expected = [-2.5, -0.5, 0.0, 1.0, 3.0]
    for i in range(5):
        assert_almost_equal(out[i], Scalar[dtype](expected[i]))


def test_sort_flattens_a_rank_two_tensor() raises:
    var ctx = DeviceContext(api="cpu")
    # numpy.sort(a, axis=None) flattens; this does the same.
    var a = zeros[dtype, 2, 3](ctx)
    var values = [5.0, 1.0, 4.0, 2.0, 6.0, 3.0]
    for i in range(6):
        a[i] = Scalar[dtype](values[i])
    var out = sort(a)
    assert_equal(out.num_elements, 6)
    for i in range(6):
        assert_almost_equal(out[i], Scalar[dtype](Float64(i + 1)))


def test_argsort_matches_numpy() raises:
    var ctx = DeviceContext(api="cpu")
    # numpy.argsort([3, 1, 4, 1, 5]) == [1, 3, 0, 2, 4]
    var a = mk[5]([3.0, 1.0, 4.0, 1.0, 5.0])
    var order = argsort(a)
    var expected = [1, 3, 0, 2, 4]
    for i in range(5):
        assert_equal(order[i], expected[i])


def test_argsort_is_stable_on_duplicates() raises:
    var ctx = DeviceContext(api="cpu")
    # The two 1.0s are at indices 1 and 3; a stable sort keeps them in
    # that relative order, which is what the expectation above encodes.
    var a = mk[5]([3.0, 1.0, 4.0, 1.0, 5.0])
    var order = argsort(a)
    assert_true(order[0] < order[1])


def test_argsort_indexes_back_into_sorted_order() raises:
    var ctx = DeviceContext(api="cpu")
    var a = mk[6]([2.5, -1.0, 7.0, 0.0, 7.0, 3.5])
    var order = argsort(a)
    var sorted_copy = sort(a)
    for i in range(6):
        assert_almost_equal(a[order[i]], sorted_copy[i])


# ------------------------------------------------------------------
# searchsorted
# ------------------------------------------------------------------


def test_searchsorted_finds_the_insertion_point() raises:
    var ctx = DeviceContext(api="cpu")
    var a = mk[5]([1.0, 1.0, 3.0, 4.0, 5.0])
    assert_equal(searchsorted(a, Scalar[dtype](3.5)), 3)


def test_searchsorted_uses_the_left_side_convention() raises:
    var ctx = DeviceContext(api="cpu")
    # numpy.searchsorted([1, 1, 3, 4, 5], 1) == 0: before its equals.
    var a = mk[5]([1.0, 1.0, 3.0, 4.0, 5.0])
    assert_equal(searchsorted(a, Scalar[dtype](1.0)), 0)


def test_searchsorted_below_everything_is_zero() raises:
    var ctx = DeviceContext(api="cpu")
    var a = mk[3]([1.0, 2.0, 3.0])
    assert_equal(searchsorted(a, Scalar[dtype](-99.0)), 0)


def test_searchsorted_above_everything_is_the_length() raises:
    var ctx = DeviceContext(api="cpu")
    var a = mk[3]([1.0, 2.0, 3.0])
    assert_equal(searchsorted(a, Scalar[dtype](99.0)), 3)


def test_searchsorted_agrees_with_a_linear_scan() raises:
    var ctx = DeviceContext(api="cpu")
    # The binary search's answer must equal "how many elements are
    # strictly less than the value", which is its definition.
    var a = mk[6]([-2.0, 0.0, 0.0, 1.5, 4.0, 9.0])
    var probes = [-3.0, -2.0, 0.0, 1.0, 4.0, 100.0]
    for p in range(6):
        var value = Scalar[dtype](probes[p])
        var linear = 0
        for i in range(6):
            if a[i] < value:
                linear += 1
        assert_equal(searchsorted(a, value), linear)


# ------------------------------------------------------------------
# unique
# ------------------------------------------------------------------


def test_unique_returns_sorted_distinct_values() raises:
    var ctx = DeviceContext(api="cpu")
    var a = mk[5]([3.0, 1.0, 4.0, 1.0, 5.0])
    var result = unique(a)
    assert_equal(result.size(), 4)
    var expected = [1.0, 3.0, 4.0, 5.0]
    for i in range(4):
        assert_almost_equal(result[i], Scalar[dtype](expected[i]))


def test_unique_of_all_identical_values_is_one() raises:
    var ctx = DeviceContext(api="cpu")
    var a = mk[4]([7.0, 7.0, 7.0, 7.0])
    var result = unique(a)
    assert_equal(result.size(), 1)
    assert_almost_equal(result[0], Scalar[dtype](7.0))


def test_unique_of_all_distinct_values_keeps_them_all() raises:
    var ctx = DeviceContext(api="cpu")
    var a = mk[4]([4.0, 1.0, 3.0, 2.0])
    var result = unique(a)
    assert_equal(result.size(), 4)
    for i in range(4):
        assert_almost_equal(result[i], Scalar[dtype](Float64(i + 1)))


# ------------------------------------------------------------------
# counting and searching
# ------------------------------------------------------------------


def test_count_nonzero_counts_nonzeros() raises:
    var ctx = DeviceContext(api="cpu")
    var a = mk[5]([0.0, 1.0, 0.0, 1.0, 1.0])
    assert_equal(count_nonzero(a), 3)


def test_count_nonzero_treats_negative_zero_as_zero() raises:
    var ctx = DeviceContext(api="cpu")
    var a = mk[3]([-0.0, 0.0, 1.0])
    assert_equal(count_nonzero(a), 1)


def test_any_nonzero_and_all_nonzero() raises:
    var ctx = DeviceContext(api="cpu")
    var mixed = mk[3]([0.0, 1.0, 2.0])
    assert_true(any_nonzero(mixed))
    assert_true(not all_nonzero(mixed))

    var full = mk[3]([1.0, 2.0, 3.0])
    assert_true(any_nonzero(full))
    assert_true(all_nonzero(full))

    var empty_valued = mk[3]([0.0, 0.0, 0.0])
    assert_true(not any_nonzero(empty_valued))
    assert_true(not all_nonzero(empty_valued))


def test_nonzero_returns_ascending_flat_indices() raises:
    var ctx = DeviceContext(api="cpu")
    var a = mk[5]([0.0, 1.0, 0.0, 1.0, 1.0])
    var indices = nonzero(a)
    assert_equal(len(indices), 3)
    assert_equal(indices[0], 1)
    assert_equal(indices[1], 3)
    assert_equal(indices[2], 4)


def test_nonzero_length_matches_count_nonzero() raises:
    var ctx = DeviceContext(api="cpu")
    var a = mk[6]([1.0, 0.0, -3.0, 0.0, 0.0, 2.5])
    assert_equal(len(nonzero(a)), count_nonzero(a))


# ------------------------------------------------------------------
# boolean masking
# ------------------------------------------------------------------


def test_extract_selects_where_the_mask_is_nonzero() raises:
    var ctx = DeviceContext(api="cpu")
    var mask = mk_mask[5]([False, True, False, True, True])
    var a = mk[5]([3.0, 1.0, 4.0, 1.0, 5.0])
    var result = extract(mask, a)
    assert_equal(result.size(), 3)
    assert_almost_equal(result[0], Scalar[dtype](1.0))
    assert_almost_equal(result[1], Scalar[dtype](1.0))
    assert_almost_equal(result[2], Scalar[dtype](5.0))


def test_extract_with_an_all_true_mask_returns_everything() raises:
    var ctx = DeviceContext(api="cpu")
    var mask = mk_mask[4]([True, True, True, True])
    var a = mk[4]([9.0, 8.0, 7.0, 6.0])
    var result = extract(mask, a)
    assert_equal(result.size(), 4)
    for i in range(4):
        assert_almost_equal(result[i], a[i])


def test_extract_with_an_all_false_mask_returns_nothing() raises:
    var ctx = DeviceContext(api="cpu")
    var mask = mk_mask[4]([False, False, False, False])
    var a = mk[4]([9.0, 8.0, 7.0, 6.0])
    assert_equal(extract(mask, a).size(), 0)


def test_extract_count_matches_the_masks_nonzero_count() raises:
    var ctx = DeviceContext(api="cpu")
    var mask = mk_mask[6]([True, False, True, False, True, False])
    var a = mk[6]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert_equal(extract(mask, a).size(), count_nonzero(mask))


def test_select_selects_elementwise() raises:
    var ctx = DeviceContext(api="cpu")
    var mask = mk_mask[5]([False, True, False, True, True])
    var x = mk[5]([3.0, 1.0, 4.0, 1.0, 5.0])
    var y = mk[5]([9.0, 9.0, 9.0, 9.0, 9.0])
    var out = select(mask, x, y)
    var expected = [9.0, 1.0, 9.0, 1.0, 5.0]
    for i in range(5):
        assert_almost_equal(out[i], Scalar[dtype](expected[i]))


def test_select_preserves_the_input_shape() raises:
    var ctx = DeviceContext(api="cpu")
    # The one function here that is not flattening, because its output
    # length does not depend on the condition's values.
    var mask_values = List[Scalar[DType.bool]](capacity=4)
    for i in range(4):
        mask_values.append(Scalar[DType.bool](i % 2 == 1))
    var mask = Static[DType.bool, 2, 2](ctx, mask_values^)
    var x = zeros[dtype, 2, 2](ctx)
    var y = zeros[dtype, 2, 2](ctx)
    for i in range(4):
        x[i] = Scalar[dtype](1.0)
        y[i] = Scalar[dtype](2.0)
    var out = select(mask, x, y)
    assert_equal(out.dim[0](), 2)
    assert_equal(out.dim[1](), 2)
    assert_almost_equal(out[0], Scalar[dtype](2.0))
    assert_almost_equal(out[1], Scalar[dtype](1.0))


def test_select_and_extract_agree_on_the_selected_values() raises:
    var ctx = DeviceContext(api="cpu")
    # extract(mask, a) should be the where-selected values, packed.
    var mask = mk_mask[6]([True, False, True, False, False, True])
    var a = mk[6]([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    var extracted = extract(mask, a)
    var indices = nonzero(mask)
    assert_equal(extracted.size(), len(indices))
    for i in range(len(indices)):
        assert_almost_equal(extracted[i], a[indices[i]])


def test_the_masking_family_returns_its_own_length() raises:
    # The reason these are run-time-shaped: the result of a mask is only as
    # long as the data says, and a caller reads that off the tensor rather
    # than carrying a separate count.
    var ctx = DeviceContext(api="cpu")
    var a = mk[6]([3.0, 3.0, 1.0, 0.0, 1.0, 0.0])
    assert_equal(unique(a).size(), 3)
    assert_equal(
        extract(mk_mask[6]([True, True, True, True, True, True]), a).size(), 6
    )
    assert_equal(take(a, nonzero(a)).size(), count_nonzero(a))


def test_take_reads_the_index_lists_the_module_already_returns() raises:
    var ctx = DeviceContext(api="cpu")
    var a = mk[5]([3.0, 1.0, 4.0, 1.0, 5.0])

    # take + argsort is the sorted copy.
    var by_take = take(a, argsort(a))
    var sorted_a = sort(a)
    for i in range(5):
        assert_almost_equal(by_take[i], sorted_a[i])

    # take + nonzero is the nonzero values, which is also extract's answer.
    var b = mk[5]([0.0, 2.0, 0.0, 4.0, 6.0])
    var by_index = take(b, nonzero(b))
    var by_mask = extract(mk_mask[5]([False, True, False, True, True]), b)
    assert_equal(by_index.size(), by_mask.size())
    for i in range(by_index.size()):
        assert_almost_equal(by_index[i], by_mask[i])


def test_take_rejects_an_out_of_range_index() raises:
    var ctx = DeviceContext(api="cpu")
    var a = mk[3]([1.0, 2.0, 3.0])
    var raised = False
    try:
        _ = take(a, [0, 3])
    except:
        raised = True
    assert_true(raised)


# ------------------------------------------------------------------
# Run-time shapes crossing back into the flattening routines
# ------------------------------------------------------------------


def test_sort_takes_what_a_mask_produced() raises:
    # The composition that reaches `sort` with run-time extents: `extract`
    # cannot know its own length until it reads the values.
    var a = mk[5]([3.0, 1.0, 4.0, 1.0, 5.0])
    var kept = extract(mk_mask[5]([True, False, True, False, True]), a)
    var out = sort(kept)

    assert_equal(out.size(), 3)
    assert_almost_equal(out[0], 3.0)
    assert_almost_equal(out[1], 4.0)
    assert_almost_equal(out[2], 5.0)


def test_sort_agrees_across_the_two_shapes() raises:
    var elements = List[Scalar[dtype]](capacity=5)
    for i in range(5):
        elements.append(Scalar[dtype]([3.0, 1.0, 4.0, 1.0, 5.0][i]))

    var fixed = sort(mk[5]([3.0, 1.0, 4.0, 1.0, 5.0]))
    var dynamic = sort(asarray(elements^, DeviceContext(api="cpu")))

    assert_equal(fixed.size(), dynamic.size())
    for i in range(5):
        assert_almost_equal(fixed[i], dynamic[i])


def test_argsort_and_take_compose_at_a_run_time_shape() raises:
    var a = mk[5]([3.0, 1.0, 4.0, 1.0, 5.0])
    var kept = extract(mk_mask[5]([False, True, True, True, False]), a)
    var ordered = take(kept, argsort(kept))

    assert_equal(ordered.size(), 3)
    assert_almost_equal(ordered[0], 1.0)
    assert_almost_equal(ordered[1], 1.0)
    assert_almost_equal(ordered[2], 4.0)


def test_select_runs_at_a_run_time_shape() raises:
    var a = mk[4]([1.0, 2.0, 3.0, 4.0])
    var x = extract(mk_mask[4]([True, True, False, False]), a)
    var y = extract(mk_mask[4]([False, False, True, True]), a)

    # A run-time-shaped mask has to come from a comparison: the zero-fill
    # constructor every other factory routes through is unavailable at
    # `DType.bool`.
    var mask = greater(x, y)

    var picked = select(mask, x, y)
    assert_equal(picked.size(), 2)
    assert_almost_equal(picked[0], 3.0)
    assert_almost_equal(picked[1], 4.0)


def test_ravel_and_diagflat_keep_a_run_time_length() raises:
    var a = mk[4]([1.0, 2.0, 3.0, 4.0])
    var kept = extract(mk_mask[4]([True, False, True, True]), a)

    assert_equal(ravel(kept).size(), 3)

    var square = diagflat(kept)
    assert_equal(square.size(), 9)
    assert_almost_equal(square[0], 1.0)
    assert_almost_equal(square[4], 3.0)
    assert_almost_equal(square[8], 4.0)
    assert_almost_equal(square[1], 0.0)


def test_top_k_returns_the_largest_with_their_indices() raises:
    # torch.topk([3, 1, 4, 1, 5], 3) -> values [5, 4, 3], indices [4, 2, 0]
    var a = mk[5]([3.0, 1.0, 4.0, 1.0, 5.0])
    var got = top_k[k=3](a)
    var values = got[0].to_host()
    var indices = got[1].to_host()
    assert_almost_equal(values[0], 5.0)
    assert_almost_equal(values[1], 4.0)
    assert_almost_equal(values[2], 3.0)
    assert_equal(Int(indices[0]), 4)
    assert_equal(Int(indices[1]), 2)
    assert_equal(Int(indices[2]), 0)


def test_top_k_smallest_inverts_the_order() raises:
    # torch.topk(..., largest=False) -- the two 1.0s, at their own positions
    var a = mk[5]([3.0, 1.0, 4.0, 1.0, 5.0])
    var got = top_k[k=2, largest=False](a)
    var values = got[0].to_host()
    var indices = got[1].to_host()
    assert_almost_equal(values[0], 1.0)
    assert_almost_equal(values[1], 1.0)
    assert_equal(Int(indices[0]), 1)
    assert_equal(Int(indices[1]), 3)


def test_top_k_indices_address_the_original() raises:
    # the claim that makes the indices useful: a[idx[i]] is values[i]
    var a = mk[6]([2.5, 9.0, -1.0, 7.25, 0.0, 9.5])
    var got = top_k[k=4](a)
    var values = got[0].to_host()
    var indices = got[1].to_host()
    var original = a.to_host()
    for i in range(4):
        assert_almost_equal(values[i], original[Int(indices[i])])


def test_top_k_is_row_wise_at_rank_two() raises:
    # torch.topk(a, 2, dim=-1) -- the last axis, not the flattened tensor,
    # which is the one place this module is not flat
    var ctx = DeviceContext(api="cpu")
    var source: List[Float64] = [1.0, 9.0, 3.0, 7.0, 8.0, 2.0, 6.0, 4.0]
    var elements = List[Scalar[dtype]](capacity=8)
    for i in range(8):
        elements.append(Scalar[dtype](source[i]))
    var m = Static[dtype, 2, 4](ctx, elements^)
    var got = top_k[k=2](m)
    var values = got[0].to_host()
    var indices = got[1].to_host()
    # row 0: 9 at column 1, then 7 at column 3
    assert_almost_equal(values[0], 9.0)
    assert_almost_equal(values[1], 7.0)
    assert_equal(Int(indices[0]), 1)
    assert_equal(Int(indices[1]), 3)
    # row 1: 8 at column 0, then 6 at column 2
    assert_almost_equal(values[2], 8.0)
    assert_almost_equal(values[3], 6.0)
    assert_equal(Int(indices[2]), 0)
    assert_equal(Int(indices[3]), 2)


def test_top_k_at_k_equals_n_is_a_full_sort() raises:
    # the boundary the `where` clause allows: k == n agrees with `sort`
    var a = mk[5]([3.0, 1.0, 4.0, 1.0, 5.0])
    var descending = top_k[k=5](a)[0].to_host()
    var ascending = sort(a).to_host()
    for i in range(5):
        assert_almost_equal(descending[i], ascending[4 - i])


def _grid3[
    rows: Int, cols: Int
](values: List[Float64]) raises -> Static[dtype, rows, cols]:
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, rows, cols](ctx)
    for i in range(rows * cols):
        a[i] = Scalar[dtype](values[i])
    return a^


def test_vectorized_searchsorted_matches_the_scalar_one_per_query() raises:
    # The claim worth pinning: the two spellings are one algorithm.
    var bins = mk[4]([1.0, 3.0, 5.0, 7.0])
    var queries = mk[5]([0.0, 3.0, 4.0, 7.0, 9.0])

    var bulk = searchsorted(bins, queries)
    assert_equal(bulk.size(), 5)
    for q in range(5):
        assert_equal(Int(bulk[q]), searchsorted(bins, queries.to_host()[q]))


def test_vectorized_searchsorted_matches_numpy_on_both_sides() raises:
    # numpy.searchsorted([1,3,5,7], [0,3,4,7,9]) == [0, 1, 2, 3, 4]
    # side="right"                              == [0, 2, 2, 4, 4]
    var bins = mk[4]([1.0, 3.0, 5.0, 7.0])
    var queries = mk[5]([0.0, 3.0, 4.0, 7.0, 9.0])

    var left = searchsorted(bins, queries)
    var expected_left = [0, 1, 2, 3, 4]
    for q in range(5):
        assert_equal(Int(left[q]), expected_left[q])

    var right = searchsorted[right=True](bins, queries)
    var expected_right = [0, 2, 2, 4, 4]
    for q in range(5):
        assert_equal(Int(right[q]), expected_right[q])


def test_take_along_an_axis_selects_whole_slices() raises:
    # numpy.take(a, [2, 0], axis=0) on a (3, 2) gives rows 2 and 0.
    var ctx = DeviceContext(api="cpu")
    var a = _grid3[3, 2]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var which = Static[DType.int64, 2](ctx, [2, 0])

    var rows = take[axis=0](a, which)
    assert_equal(rows.dim_at(0), 2)
    assert_equal(rows.dim_at(1), 2)
    var out = rows.to_host()
    assert_equal(out[0], 5.0)
    assert_equal(out[1], 6.0)
    assert_equal(out[2], 1.0)
    assert_equal(out[3], 2.0)


def test_take_along_an_axis_may_duplicate_a_slice() raises:
    # Selecting the same row twice is what makes this a gather rather than
    # a permutation.
    var ctx = DeviceContext(api="cpu")
    var a = _grid3[3, 2]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var which = Static[DType.int64, 3](ctx, [1, 1, 0])

    var rows = take[axis=0](a, which).to_host()
    assert_equal(rows[0], 3.0)
    assert_equal(rows[2], 3.0)
    assert_equal(rows[4], 1.0)


def test_take_along_an_axis_rejects_an_out_of_range_index() raises:
    var ctx = DeviceContext(api="cpu")
    var a = _grid3[3, 2]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var which = Static[DType.int64, 1](ctx, [3])
    var raised = False
    try:
        _ = take[axis=0](a, which)
    except:
        raised = True
    assert_true(raised)


def test_take_along_axis_picks_one_element_per_position() raises:
    # numpy.take_along_axis(a, idx, axis=1) with idx shaped like a.
    var ctx = DeviceContext(api="cpu")
    var a = _grid3[3, 2]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var idx = Static[DType.int64, 3, 2](ctx, [1, 0, 0, 1, 1, 1])

    var picked = take_along_axis[axis=1](a, idx).to_host()
    var expected = [2.0, 1.0, 3.0, 4.0, 6.0, 6.0]
    for i in range(6):
        assert_equal(picked[i], Scalar[dtype](expected[i]))


def test_take_along_axis_sorts_each_row_with_argsort() raises:
    # The reason `take_along_axis` exists: `argsort`'s per-row output had
    # no consumer at rank > 1.
    var ctx = DeviceContext(api="cpu")
    var a = _grid3[2, 3]([3.0, 1.0, 2.0, 9.0, 7.0, 8.0])
    var order = Static[DType.int64, 2, 3](ctx, [1, 2, 0, 1, 2, 0])

    var sorted_rows = take_along_axis[axis=1](a, order).to_host()
    assert_equal(sorted_rows[0], 1.0)
    assert_equal(sorted_rows[1], 2.0)
    assert_equal(sorted_rows[2], 3.0)
    assert_equal(sorted_rows[3], 7.0)
    assert_equal(sorted_rows[5], 9.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
