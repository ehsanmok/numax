"""`max` and `min` of tensors whose extremes are infinite.

MAX's `ReduceMax` starts from the most negative finite value and
`ReduceMin` from the largest, so a tensor of nothing but `-inf` (or
`+inf`) used to reduce to `-1.797e308` (or `+1.797e308`). NumPy returns
the infinity. These hold the whole-tensor and axis forms to NumPy on
exactly those inputs, and on the input that must *not* be repaired: a
tensor that really contains the finite edge value alongside infinities.
"""

from std.testing import TestSuite, assert_equal
from std.utils.numerics import inf, neg_inf

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import max, min

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def test_whole_tensor_extremes_of_only_infinities() raises:
    var low = Static[f64, 3](
        [neg_inf[f64](), neg_inf[f64](), neg_inf[f64]()], _cpu()
    )
    var high = Static[f64, 3]([inf[f64](), inf[f64](), inf[f64]()], _cpu())
    assert_equal(max(low), neg_inf[f64]())
    assert_equal(min(high), inf[f64]())
    assert_equal(min(low), neg_inf[f64]())
    assert_equal(max(high), inf[f64]())


def test_a_real_edge_value_is_kept() raises:
    var edge = Scalar[f64].MIN_FINITE
    var a = Static[f64, 3]([neg_inf[f64](), edge, neg_inf[f64]()], _cpu())
    assert_equal(max(a), edge)
    var top = Scalar[f64].MAX_FINITE
    var b = Static[f64, 2]([inf[f64](), top], _cpu())
    assert_equal(min(b), top)


def test_axis_extremes_of_infinite_slices() raises:
    # numpy: np.max([[-inf, 1], [-inf, -inf]], axis=1) == [1, -inf]
    #        np.max(..., axis=0) == [-inf, 1]
    var m = Static[f64, 2, 2](
        [neg_inf[f64](), 1.0, neg_inf[f64](), neg_inf[f64]()], _cpu()
    )
    var rows = max[axis=1](m).to_host()
    assert_equal(rows[0], 1.0)
    assert_equal(rows[1], neg_inf[f64]())
    var cols = max[axis=0](m).to_host()
    assert_equal(cols[0], neg_inf[f64]())
    assert_equal(cols[1], 1.0)
    var h = Static[f64, 2, 2]([inf[f64](), inf[f64](), 2.0, inf[f64]()], _cpu())
    var hr = min[axis=1](h).to_host()
    assert_equal(hr[0], inf[f64]())
    assert_equal(hr[1], 2.0)


def test_axis_keeps_a_real_edge_value() raises:
    var edge = Scalar[f64].MIN_FINITE
    var m = Static[f64, 2, 2](
        [neg_inf[f64](), edge, neg_inf[f64](), neg_inf[f64]()], _cpu()
    )
    var rows = max[axis=1](m).to_host()
    assert_equal(rows[0], edge)
    assert_equal(rows[1], neg_inf[f64]())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
