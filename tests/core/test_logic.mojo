"""Tests for `numax.core.logic`.

Every comparison and predicate is checked elementwise against a
hand-computed expected mask; the reductions are checked on inputs that
force both the short-circuit and the full walk. NaN and both infinities are
exercised directly, since those are the cases the predicates exist for.

The routing tests at the bottom make the claims the `numax.core._drive`
move added. A comparison now runs at the launch width rather than one lane
at a time, so one test pins per-lane answers across a tensor wider than the
SIMD width -- `a == b` on a SIMD vector returns a single `Bool`, which
would splat one lane's answer across all of them. A mask with no true
element is pinned all false, because the destination comes from
`Tensor._uninitialized` and nothing but the launch writes it. A run-time
shaped `Dynamic` operand gives the `Static` answer, and a tensor above the
threading threshold gives the serial path's answer. Asking for a target the
tensor is not on is proven by `examples/advanced/unified_tensor_gpu.mojo`,
not here: `gpu=True` compiles a device kernel, which a GPU-less CI runner
cannot do.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from max.gpu.host import DeviceContext

from numax.core.array import Dynamic, Static, Tensor, zeros_dyn
from numax.core.logic import (
    all,
    allclose,
    any,
    array_equal,
    equal,
    greater,
    greater_equal,
    isclose,
    isfinite,
    isinf,
    isnan,
    isneginf,
    isposinf,
    less,
    less_equal,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    not_equal,
)

comptime dtype = DType.float32


def _tensor[n: Int](values: List[Float64]) raises -> Static[dtype, n]:
    var ctx = DeviceContext(api="cpu")
    var elements = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        elements.append(Scalar[dtype](values[i]))
    return Static[dtype, n](ctx, elements^)


def _bools[n: Int](values: List[Bool]) raises -> Static[DType.bool, n]:
    var ctx = DeviceContext(api="cpu")
    var elements = List[Scalar[DType.bool]](capacity=n)
    for i in range(n):
        elements.append(values[i])
    return Static[DType.bool, n](ctx, elements^)


def test_equal_and_not_equal_are_complements() raises:
    var a = _tensor[4]([1.0, 2.0, 3.0, 4.0])
    var b = _tensor[4]([1.0, 9.0, 3.0, 9.0])
    var eq = equal(a, b)
    var ne = not_equal(a, b)
    var eq_h = eq.to_host()
    var ne_h = ne.to_host()
    for i in range(4):
        assert_equal(eq_h[i], not ne_h[i])
    assert_true(eq_h[0])
    assert_false(eq_h[1])
    assert_true(eq_h[2])
    assert_false(eq_h[3])


def test_ordering_comparisons_match_hand_computed_masks() raises:
    var a = _tensor[3]([1.0, 2.0, 3.0])
    var b = _tensor[3]([2.0, 2.0, 2.0])
    var lt = less(a, b).to_host()
    var le = less_equal(a, b).to_host()
    var gt = greater(a, b).to_host()
    var ge = greater_equal(a, b).to_host()
    assert_true(lt[0] and not lt[1] and not lt[2])
    assert_true(le[0] and le[1] and not le[2])
    assert_true(not gt[0] and not gt[1] and gt[2])
    assert_true(not ge[0] and ge[1] and ge[2])


def test_comparisons_preserve_rank() raises:
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, 2, 3](ctx)
    var b = Static[dtype, 2, 3](ctx)
    var mask = greater(a, b)
    assert_equal(mask.num_elements, 6)
    assert_equal(mask.rank, 2)


def test_isnan_isinf_isfinite_on_the_special_values() raises:
    var inf = Float64(1.0) / Float64(0.0)
    var nan = inf - inf
    var a = _tensor[4]([0.0, nan, inf, -inf])
    var nans = isnan(a).to_host()
    var infs = isinf(a).to_host()
    var finite = isfinite(a).to_host()
    assert_true(not nans[0] and nans[1] and not nans[2] and not nans[3])
    assert_true(not infs[0] and not infs[1] and infs[2] and infs[3])
    assert_true(finite[0] and not finite[1] and not finite[2] and not finite[3])


def test_isposinf_and_isneginf_split_by_sign() raises:
    var inf = Float64(1.0) / Float64(0.0)
    var a = _tensor[3]([inf, -inf, 1.0])
    var pos = isposinf(a).to_host()
    var neg = isneginf(a).to_host()
    assert_true(pos[0] and not pos[1] and not pos[2])
    assert_true(not neg[0] and neg[1] and not neg[2])


def test_logical_ops_match_their_truth_tables() raises:
    var a = _bools[4]([True, True, False, False])
    var b = _bools[4]([True, False, True, False])
    var and_ = logical_and(a, b).to_host()
    var or_ = logical_or(a, b).to_host()
    var xor_ = logical_xor(a, b).to_host()
    assert_true(and_[0] and not and_[1] and not and_[2] and not and_[3])
    assert_true(or_[0] and or_[1] and or_[2] and not or_[3])
    assert_true(not xor_[0] and xor_[1] and xor_[2] and not xor_[3])


def test_logical_not_inverts_every_element() raises:
    var a = _bools[3]([True, False, True])
    var inverted = logical_not(a).to_host()
    assert_true(not inverted[0] and inverted[1] and not inverted[2])


def test_all_and_any() raises:
    var all_set = _bools[3]([True, True, True])
    var some_set = _bools[3]([False, True, False])
    var none_set = _bools[3]([False, False, False])
    assert_true(all(all_set))
    assert_false(all(some_set))
    assert_true(any(some_set))
    assert_false(any(none_set))


def test_isclose_respects_both_tolerances() raises:
    var a = _tensor[3]([1.0, 1.0, 1.0])
    var b = _tensor[3]([1.0, 1.000001, 2.0])
    var close = isclose(a, b).to_host()
    assert_true(close[0])
    assert_true(close[1])
    assert_false(close[2])


def test_allclose_and_array_equal_disagree_on_rounding() raises:
    var a = _tensor[2]([1.0, 2.0])
    # 2.0000001 rounds to exactly 2.0 in float32; 2.000001 does not.
    var b = _tensor[2]([1.0, 2.000001])
    assert_true(allclose(a, b))
    assert_false(array_equal(a, b))


def test_array_equal_is_exact() raises:
    var a = _tensor[3]([1.0, 2.0, 3.0])
    var b = _tensor[3]([1.0, 2.0, 3.0])
    assert_true(array_equal(a, b))


def test_nan_is_never_equal_to_itself() raises:
    var inf = Float64(1.0) / Float64(0.0)
    var nan = inf - inf
    var a = _tensor[2]([nan, 1.0])
    var b = _tensor[2]([nan, 1.0])
    assert_false(array_equal(a, b))


def _matrix[
    rows: Int, cols: Int
](values: List[Float64]) raises -> Static[dtype, rows, cols]:
    var ctx = DeviceContext(api="cpu")
    var elements = List[Scalar[dtype]](capacity=rows * cols)
    for i in range(rows * cols):
        elements.append(Scalar[dtype](values[i]))
    return Static[dtype, rows, cols](ctx, elements^)


def test_greater_broadcasts_a_threshold_row_across_a_matrix() raises:
    # numpy: np.array([[1, 5], [7, 2]]) > np.array([4, 4])
    var a = _matrix[2, 2]([1.0, 5.0, 7.0, 2.0])
    var threshold = _tensor[2]([4.0, 4.0])

    var mask = greater(a, threshold)
    assert_equal(mask.dim_at(0), 2)
    assert_equal(mask.dim_at(1), 2)
    var out = mask.to_host()
    assert_false(out[0])
    assert_true(out[1])
    assert_true(out[2])
    assert_false(out[3])


def test_broadcast_mask_still_composes_with_logical_and() raises:
    # The property that makes truth a boolean tensor rather than 0/1: two
    # broadcast masks combine without anything in between.
    var a = _matrix[2, 2]([1.0, 5.0, 7.0, 2.0])
    var lo = _tensor[2]([0.0, 0.0])
    var hi = _tensor[2]([6.0, 6.0])

    var inside = logical_and(greater(a, lo), less(a, hi)).to_host()
    assert_true(inside[0])
    assert_true(inside[1])
    assert_false(inside[2])
    assert_true(inside[3])


def test_broadcast_comparison_agrees_with_the_same_shape_overload() raises:
    var a = _matrix[2, 2]([1.0, 5.0, 7.0, 2.0])
    var thin = _matrix[1, 2]([4.0, 4.0])
    var wide = _matrix[2, 2]([4.0, 4.0, 4.0, 4.0])

    var broadcast = less_equal(a, thin).to_host()
    var direct = less_equal(a, wide).to_host()
    for i in range(4):
        assert_equal(broadcast[i], direct[i])


def test_comparison_answers_every_lane_across_the_launch_width() raises:
    """Per-lane truth, on a tensor wider than the native SIMD width.

    The drivers call the comparison at the launch width. `a == b` on a SIMD
    vector returns one `Bool` rather than a mask, so a width-generic step
    written with the operator would splat the first lane's answer across
    every lane; the method spellings (`a.eq(b)`, `a.lt(b)`) are what make
    this alternating pattern come out alternating.
    """
    comptime n = 12
    var left = List[Float64](capacity=n)
    var right = List[Float64](capacity=n)
    for i in range(n):
        left.append(Float64(i))
        right.append(Float64(i) if i % 2 == 0 else Float64(i) + 1.0)

    var a = _tensor[n](left)
    var b = _tensor[n](right)
    var eq = equal(a, b).to_host()
    var ne = not_equal(a, b).to_host()
    var lt = less(a, b).to_host()
    var ge = greater_equal(a, b).to_host()
    for i in range(n):
        var same = i % 2 == 0
        assert_equal(eq[i], same)
        assert_equal(ne[i], not same)
        assert_equal(lt[i], not same)
        assert_equal(ge[i], same)


def test_a_mask_with_no_true_element_is_all_false() raises:
    """Garbage detection for `_uninitialized` at `DType.bool`.

    The destination is allocated without being zeroed, so a launch that
    failed to write a lane would leave whatever was in the buffer. Every
    one of these masks is false everywhere, across more lanes than one
    launch width.
    """
    comptime n = 20
    var values = List[Float64](capacity=n)
    for i in range(n):
        values.append(Float64(i) + 1.0)
    var a = _tensor[n](values)
    var b = _tensor[n](values)

    var never_less = less(a, b).to_host()
    var never_nan = isnan(a).to_host()
    var never_inf = isinf(a).to_host()
    var never_posinf = isposinf(a).to_host()
    var never_neginf = isneginf(a).to_host()
    for i in range(n):
        assert_false(never_less[i])
        assert_false(never_nan[i])
        assert_false(never_inf[i])
        assert_false(never_posinf[i])
        assert_false(never_neginf[i])

    var all_true = _bools[n](List[Bool](length=n, fill=True))
    var none = logical_not(all_true).to_host()
    var neither = logical_xor(all_true, all_true).to_host()
    for i in range(n):
        assert_false(none[i])
        assert_false(neither[i])


def test_nan_compares_false_except_through_not_equal() raises:
    """IEEE's rule, per lane: NaN is unordered against everything.

    `NaN == NaN` is false, `NaN != x` is true for every `x` including
    itself, and all four orderings are false -- which is why `isnan` exists
    at all.
    """
    var inf = Float64(1.0) / Float64(0.0)
    var nan = inf - inf
    var a = _tensor[6]([nan, nan, nan, 1.0, nan, nan])
    var b = _tensor[6]([nan, 1.0, -1.0, nan, 0.0, inf])

    var eq = equal(a, b).to_host()
    var ne = not_equal(a, b).to_host()
    var lt = less(a, b).to_host()
    var le = less_equal(a, b).to_host()
    var gt = greater(a, b).to_host()
    var ge = greater_equal(a, b).to_host()
    for i in range(6):
        assert_false(eq[i])
        assert_true(ne[i])
        assert_false(lt[i])
        assert_false(le[i])
        assert_false(gt[i])
        assert_false(ge[i])

    var nans = isnan(a).to_host()
    assert_true(nans[0] and nans[1] and nans[2] and not nans[3])


def test_isclose_honors_tolerances_given_explicitly() raises:
    """Both tolerances, each on its own: `atol` decides near zero and
    `rtol` scales with `abs(b)`, exactly `numpy.isclose`'s formula."""
    var a = _tensor[4]([0.0, 0.0, 100.0, 100.0])
    var b = _tensor[4]([0.5, 2.0, 100.5, 110.0])

    # atol alone, rtol off: 0.5 is inside a 1.0 absolute window, 2.0 is not.
    var absolute = isclose(a, b, rtol=0.0, atol=1.0).to_host()
    assert_true(absolute[0])
    assert_false(absolute[1])

    # rtol alone, atol off: 0.5 out of 100.5 is inside 1%, 10 out of 110 is not.
    var relative = isclose(a, b, rtol=0.01, atol=0.0).to_host()
    assert_true(relative[2])
    assert_false(relative[3])


def test_a_dynamic_operand_gives_the_static_answer() raises:
    """Run-time extents take the same driver, so the mask is the same."""
    var ctx = DeviceContext(api="cpu")
    var left = List[Scalar[dtype]](capacity=6)
    var right = List[Scalar[dtype]](capacity=6)
    for i in range(6):
        left.append(Scalar[dtype](i) - 2.0)
        right.append(Scalar[dtype](1.0))

    var a_fixed = Static[dtype, 2, 3](ctx, left.copy())
    var b_fixed = Static[dtype, 2, 3](ctx, right.copy())
    var a_runtime = zeros_dyn[dtype, 2](2, 3, ctx=ctx)
    var b_runtime = zeros_dyn[dtype, 2](2, 3, ctx=ctx)
    a_runtime.copy_from_host(left)
    b_runtime.copy_from_host(right)

    var from_static = greater(a_fixed, b_fixed).to_host()
    var from_dynamic = greater(a_runtime, b_runtime).to_host()
    assert_equal(len(from_dynamic), 6)
    for i in range(6):
        assert_equal(from_dynamic[i], from_static[i])

    var finite_static = isfinite(a_fixed).to_host()
    var finite_dynamic = isfinite(a_runtime).to_host()
    for i in range(6):
        assert_equal(finite_dynamic[i], finite_static[i])

    var close_static = isclose(a_fixed, b_fixed).to_host()
    var close_dynamic = isclose(a_runtime, b_runtime).to_host()
    for i in range(6):
        assert_equal(close_dynamic[i], close_static[i])


def test_above_the_threading_threshold_the_mask_is_the_same() raises:
    """100003 elements is past `_drive`'s `1 << 16`, so this runs through
    `elementwise[target="cpu"]` rather than the serial loop; a comparison
    stores `SIMD[DType.bool, w]` on that path too."""
    comptime n = 100003
    var ctx = DeviceContext(api="cpu")
    var left = List[Scalar[dtype]](capacity=n)
    var right = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        left.append(Scalar[dtype](i % 7))
        right.append(Scalar[dtype](3))

    var a = Static[dtype, n](ctx, left.copy())
    var b = Static[dtype, n](ctx, right.copy())
    var mask = greater(a, b).to_host()
    for i in range(n):
        assert_equal(mask[i], left[i] > right[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
