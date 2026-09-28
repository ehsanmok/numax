"""Tests for `nanargmax` and `nanargmin` in `numax.stats.nanfunctions`.

Against NumPy on a vector and a matrix read flat, the first index winning
a tie, the all-NaN input raising (NumPy raises too), and the one case the
implementation takes a second pass for: every non-NaN element an
infinity, with a NaN ahead of the first one, where a naive fill would
return the NaN's position.
"""

from std.testing import TestSuite, assert_equal, assert_raises
from std.utils.numerics import inf, nan, neg_inf

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import nanargmax, nanargmin

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def test_nanarg_matches_numpy() raises:
    # numpy: np.nanargmax([nan, 3, 1, nan, 3, -2]) == 1,
    #        np.nanargmin(...) == 5
    var a = Static[f64, 6](
        [nan[f64](), 3.0, 1.0, nan[f64](), 3.0, -2.0], _cpu()
    )
    assert_equal(nanargmax(a), 1)
    assert_equal(nanargmin(a), 5)


def test_nanarg_reads_a_matrix_flat() raises:
    # numpy: np.nanargmax([[1, nan], [7, 2]]) == 2
    var m = Static[f64, 2, 2]([1.0, nan[f64](), 7.0, 2.0], _cpu())
    assert_equal(nanargmax(m), 2)
    assert_equal(nanargmin(m), 0)


def test_nanarg_skips_a_nan_ahead_of_all_infinite_values() raises:
    # numpy: np.nanargmax([nan, -inf, -inf]) == 1,
    #        np.nanargmin([nan, inf, inf]) == 1
    var low = Static[f64, 3](
        [nan[f64](), neg_inf[f64](), neg_inf[f64]()], _cpu()
    )
    var high = Static[f64, 3]([nan[f64](), inf[f64](), inf[f64]()], _cpu())
    assert_equal(nanargmax(low), 1)
    assert_equal(nanargmin(high), 1)


def test_nanarg_of_only_nan_raises() raises:
    var a = Static[f64, 3]([nan[f64](), nan[f64](), nan[f64]()], _cpu())
    with assert_raises(contains="every element is NaN"):
        _ = nanargmax(a)
    with assert_raises(contains="every element is NaN"):
        _ = nanargmin(a)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
