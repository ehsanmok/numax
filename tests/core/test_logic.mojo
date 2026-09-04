"""Tests for `numax.core.logic`.

Every comparison and predicate is checked elementwise against a
hand-computed expected mask; the reductions are checked on inputs that
force both the short-circuit and the full walk. NaN and both infinities are
exercised directly, since those are the cases the predicates exist for.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from max.gpu.host import DeviceContext

from numax.core.array import Tensor
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


def _tensor[n: Int](values: List[Float64]) raises -> Tensor[dtype, n]:
    var ctx = DeviceContext(api="cpu")
    var elements = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        elements.append(Scalar[dtype](values[i]))
    return Tensor[dtype, n](ctx, elements^)


def _bools[n: Int](values: List[Bool]) raises -> Tensor[DType.bool, n]:
    var ctx = DeviceContext(api="cpu")
    var elements = List[Scalar[DType.bool]](capacity=n)
    for i in range(n):
        elements.append(values[i])
    return Tensor[DType.bool, n](ctx, elements^)


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
    var a = Tensor[dtype, 2, 3](ctx)
    var b = Tensor[dtype, 2, 3](ctx)
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
