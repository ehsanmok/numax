"""Tests for the set routines in `numax.core.sorting`: `unique_counts`,
`unique_inverse`, `isin`, `intersect1d`, `setdiff1d` and `union1d`.

Each against NumPy on small inputs with repeats, plus the claims the
docstrings make: `unique_counts`' values are `unique`'s and its counts
sum to the size; `unique_inverse` keeps the input's shape and rebuilds it
through `take`; NaN is never a member and is its own value (NumPy's
`equal_nan=False`); `isin`'s `invert` is the complement.
"""

from std.testing import TestSuite, assert_equal, assert_true
from std.utils.numerics import nan

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.core.sorting import (
    intersect1d,
    isin,
    setdiff1d,
    take,
    union1d,
    unique,
    unique_counts,
    unique_inverse,
)

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _x() raises -> Static[f64, 7]:
    return Static[f64, 7]([3.0, 1.0, 2.0, 3.0, 1.0, 5.0, 1.0], _cpu())


def test_unique_counts_matches_numpy() raises:
    # numpy: np.unique_counts([3, 1, 2, 3, 1, 5, 1])
    var r = unique_counts(_x())
    var values = r.values.to_host()
    var counts = r.counts.to_host()
    var want_v = [1.0, 2.0, 3.0, 5.0]
    var want_c = [3, 1, 2, 1]
    assert_equal(len(values), 4)
    assert_equal(len(counts), 4)
    var plain = unique(_x()).to_host()
    for i in range(4):
        assert_equal(Float64(values[i]), want_v[i])
        assert_equal(Int(counts[i]), want_c[i])
        assert_equal(values[i], plain[i])


def test_unique_inverse_matches_numpy_and_rebuilds() raises:
    # numpy: np.unique_inverse([3, 1, 2, 3, 1, 5, 1]).inverse_indices
    var r = unique_inverse(_x())
    var inverse = r.inverse_indices.to_host()
    var want = [2, 0, 1, 2, 0, 3, 0]
    for i in range(7):
        assert_equal(Int(inverse[i]), want[i])
    var rebuilt = take[axis=0](r.values, r.inverse_indices).to_host()
    var x = _x().to_host()
    for i in range(7):
        assert_equal(rebuilt[i], x[i])


def test_unique_inverse_keeps_the_input_shape() raises:
    var m = Static[f64, 2, 3]([4.0, 4.0, 1.0, 9.0, 1.0, 4.0], _cpu())
    var r = unique_inverse(m)
    assert_equal(r.inverse_indices.dim_at(0), 2)
    assert_equal(r.inverse_indices.dim_at(1), 3)
    assert_equal(Int(r.inverse_indices[1, 0]), 2)
    assert_equal(Int(r.inverse_indices[1, 1]), 0)


def test_isin_and_its_invert_match_numpy() raises:
    # numpy: np.isin([3, 1, 2, 3, 1, 5, 1], [1, 5, 7])
    var tests = Static[f64, 3]([1.0, 5.0, 7.0], _cpu())
    var got = isin(_x(), tests).to_host()
    var inv = isin[invert=True](_x(), tests).to_host()
    var want = [False, True, False, False, True, True, True]
    for i in range(7):
        assert_equal(Bool(got[i]), want[i])
        assert_equal(Bool(inv[i]), not want[i])


def test_binary_set_operations_match_numpy() raises:
    # numpy: np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1]) == [1, 3]
    var a = Static[f64, 4]([1.0, 3.0, 4.0, 3.0], _cpu())
    var b = Static[f64, 4]([3.0, 1.0, 2.0, 1.0], _cpu())
    var both = intersect1d(a, b).to_host()
    assert_equal(len(both), 2)
    assert_equal(Float64(both[0]), 1.0)
    assert_equal(Float64(both[1]), 3.0)
    # numpy: np.setdiff1d([5, 2, 1, 2, 4], [2, 3]) == [1, 4, 5]
    var c = Static[f64, 5]([5.0, 2.0, 1.0, 2.0, 4.0], _cpu())
    var d = Static[f64, 2]([2.0, 3.0], _cpu())
    var diff = setdiff1d(c, d).to_host()
    var want_diff = [1.0, 4.0, 5.0]
    assert_equal(len(diff), 3)
    for i in range(3):
        assert_equal(Float64(diff[i]), want_diff[i])
    # numpy: np.union1d([-1, 0, 1], [-2, 0, 2]) == [-2, -1, 0, 1, 2]
    var e = Static[f64, 3]([-1.0, 0.0, 1.0], _cpu())
    var f = Static[f64, 3]([-2.0, 0.0, 2.0], _cpu())
    var either = union1d(e, f).to_host()
    assert_equal(len(either), 5)
    for i in range(5):
        assert_equal(Float64(either[i]), Float64(i - 2))


def test_nan_is_its_own_value_and_never_a_member() raises:
    var x = Static[f64, 3]([nan[f64](), 1.0, nan[f64]()], _cpu())
    var counts = unique_counts(x)
    assert_equal(counts.values.size(), 3)
    var c = counts.counts.to_host()
    for i in range(3):
        assert_equal(Int(c[i]), 1)
    var member = isin(x, x).to_host()
    assert_true(not Bool(member[0]))
    assert_true(Bool(member[1]))
    assert_equal(intersect1d(x, x).size(), 1)
    assert_equal(setdiff1d(x, x).size(), 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
