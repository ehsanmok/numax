"""`keepdims=True` on the axis reductions in `numax.stats.statistics`.

Each of `sum`, `prod`, `min`, `max`, `mean`, `median`, `mode`, `argmax`
and `argmin` along either axis of a matrix, and along the middle axis of
a rank-3 tensor: the kept result has the input's rank with the reduced
axis at extent 1, holds exactly the values the default spelling gives,
and broadcasts back against the input -- `x - max(x, keepdims)` is the
spelling the parameter exists for.
"""

from std.testing import TestSuite, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.core.ops import subtract
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

comptime f64 = DType.float64


def _m() raises -> Static[f64, 2, 3]:
    return Static[f64, 2, 3](
        [3.0, 1.0, 2.0, 6.0, 5.0, 5.0], DeviceContext(api="cpu")
    )


def test_kept_axis_1_matches_the_default() raises:
    var m = _m()
    var k = sum[axis=1, keepdims=True](m)
    assert_equal(k.dim_at(0), 2)
    assert_equal(k.dim_at(1), 1)
    var d = sum[axis=1](m).to_host()
    var kh = k.to_host()
    for i in range(2):
        assert_equal(kh[i], d[i])
    var checks = List[Float64]()
    checks.append(Float64(prod[axis=1, keepdims=True](m)[1, 0]))
    checks.append(Float64(min[axis=1, keepdims=True](m)[0, 0]))
    checks.append(Float64(max[axis=1, keepdims=True](m)[1, 0]))
    checks.append(Float64(mean[axis=1, keepdims=True](m)[0, 0]))
    checks.append(Float64(median[axis=1, keepdims=True](m)[1, 0]))
    checks.append(Float64(mode[axis=1, keepdims=True](m)[1, 0]))
    var want = [150.0, 1.0, 6.0, 2.0, 5.0, 5.0]
    for i in range(6):
        assert_equal(checks[i], want[i])
    assert_equal(Int(argmax[axis=1, keepdims=True](m)[1, 0]), 0)
    assert_equal(Int(argmin[axis=1, keepdims=True](m)[0, 0]), 1)


def test_kept_axis_0_has_a_leading_unit_axis() raises:
    var m = _m()
    var k = max[axis=0, keepdims=True](m)
    assert_equal(k.dim_at(0), 1)
    assert_equal(k.dim_at(1), 3)
    assert_equal(k[0, 0], 6.0)
    assert_equal(k[0, 2], 5.0)
    var a = argmin[axis=0, keepdims=True](m)
    assert_equal(a.dim_at(0), 1)
    assert_equal(Int(a[0, 1]), 0)


def test_kept_result_broadcasts_against_the_input() raises:
    # numpy: x - x.max(axis=1, keepdims=True)
    var m = _m()
    var shifted = subtract(m, max[axis=1, keepdims=True](m)).to_host()
    var want = [0.0, -2.0, -1.0, 0.0, -1.0, -1.0]
    for i in range(6):
        assert_equal(Float64(shifted[i]), want[i])


def test_kept_middle_axis_of_rank_three() raises:
    var t = Static[f64, 2, 3, 2](
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
        ],
        DeviceContext(api="cpu"),
    )
    var k = sum[axis=1, keepdims=True](t)
    assert_equal(k.dim_at(0), 2)
    assert_equal(k.dim_at(1), 1)
    assert_equal(k.dim_at(2), 2)
    var h = k.to_host()
    var want = [9.0, 12.0, 27.0, 30.0]
    for i in range(4):
        assert_equal(Float64(h[i]), want[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
