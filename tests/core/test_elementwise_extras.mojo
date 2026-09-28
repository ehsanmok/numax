"""Tests for `logaddexp`, `logaddexp2`, `sinc`, `heaviside` and
`nan_to_num` in `numax.core.elementwise`.

Each is checked against NumPy's value for the same input, at the edges
the implementation makes a claim about: `logaddexp` at equal infinities
and at magnitudes whose exponentials overflow, `sinc` at zero and at the
integers, `heaviside` at `-0.0` and NaN, and `nan_to_num` with and
without its replacement values. The broadcasting overloads are held to
the same-shape ones.
"""

from std.math import isinf
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from std.utils.numerics import inf, max_finite, nan, neg_inf

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.core.elementwise import (
    heaviside,
    logaddexp,
    logaddexp2,
    nan_to_num,
    sinc,
)

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def test_logaddexp_matches_numpy_without_overflow() raises:
    # numpy: np.logaddexp([0, 1, 1000, -1000], [0, 2, 1000, -1001])
    var a = Static[f64, 4]([0.0, 1.0, 1000.0, -1000.0], _cpu())
    var b = Static[f64, 4]([0.0, 2.0, 1000.0, -1001.0], _cpu())
    var got = logaddexp(a, b).to_host()
    var want = [
        0.6931471805599453,
        2.3132616875182228,
        1000.6931471805599,
        -999.6867383124818,
    ]
    for i in range(4):
        assert_almost_equal(Float64(got[i]), want[i], rtol=1e-15)


def test_logaddexp_keeps_equal_infinities_and_nan() raises:
    var a = Static[f64, 4](
        [neg_inf[f64](), inf[f64](), nan[f64](), 1.0], _cpu()
    )
    var b = Static[f64, 4](
        [neg_inf[f64](), inf[f64](), 1.0, neg_inf[f64]()], _cpu()
    )
    var got = logaddexp(a, b).to_host()
    assert_equal(got[0], neg_inf[f64]())
    assert_equal(got[1], inf[f64]())
    assert_true(got[2] != got[2])
    assert_equal(got[3], 1.0)


def test_logaddexp2_matches_numpy() raises:
    # numpy: np.logaddexp2([1, 3, -inf], [2, 3, -inf])
    var a = Static[f64, 3]([1.0, 3.0, neg_inf[f64]()], _cpu())
    var b = Static[f64, 3]([2.0, 3.0, neg_inf[f64]()], _cpu())
    var got = logaddexp2(a, b).to_host()
    assert_almost_equal(Float64(got[0]), 2.584962500721156, rtol=1e-15)
    assert_equal(got[1], 4.0)
    assert_equal(got[2], neg_inf[f64]())


def test_sinc_is_normalized_and_one_at_zero() raises:
    # numpy: np.sinc([0, 0.5, 1, -1.5, 2])
    var x = Static[f64, 5]([0.0, 0.5, 1.0, -1.5, 2.0], _cpu())
    var got = sinc(x).to_host()
    var want = [1.0, 0.6366197723675814, 0.0, -0.2122065907891938, 0.0]
    for i in range(5):
        assert_almost_equal(Float64(got[i]), want[i], atol=1e-15)
    assert_equal(got[0], 1.0)


def test_heaviside_matches_numpy_at_zero_and_nan() raises:
    # numpy: np.heaviside([-1.5, 0, -0.0, 2, nan], 0.5)
    var x = Static[f64, 5]([-1.5, 0.0, -0.0, 2.0, nan[f64]()], _cpu())
    var h0 = Static[f64, 5]([0.5, 0.5, 0.5, 0.5, 0.5], _cpu())
    var got = heaviside(x, h0).to_host()
    assert_equal(got[0], 0.0)
    assert_equal(got[1], 0.5)
    assert_equal(got[2], 0.5)
    assert_equal(got[3], 1.0)
    assert_true(got[4] != got[4])


def test_nan_to_num_defaults_to_the_finite_extremes() raises:
    # numpy: np.nan_to_num([nan, inf, -inf, 1.5])
    var x = Static[f64, 4](
        [nan[f64](), inf[f64](), neg_inf[f64](), 1.5], _cpu()
    )
    var got = nan_to_num(x).to_host()
    assert_equal(got[0], 0.0)
    assert_equal(got[1], max_finite[f64]())
    assert_equal(got[2], -max_finite[f64]())
    assert_equal(got[3], 1.5)
    var given = nan_to_num(x, -1.0, 9.0, -9.0).to_host()
    assert_equal(given[0], -1.0)
    assert_equal(given[1], 9.0)
    assert_equal(given[2], -9.0)
    assert_equal(given[3], 1.5)
    for i in range(4):
        assert_true(not isinf(got[i]))


def test_broadcast_forms_agree_with_same_shape() raises:
    var a = Static[f64, 2, 3]([0.0, 1.0, -2.0, 3.0, 0.0, -0.5], _cpu())
    var row = Static[f64, 3]([0.0, 2.0, 0.25], _cpu())
    var tiled = Static[f64, 2, 3]([0.0, 2.0, 0.25, 0.0, 2.0, 0.25], _cpu())
    var h0 = Static[f64, 1]([0.5], _cpu())
    var h0s = Static[f64, 2, 3]([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], _cpu())
    var same = logaddexp(a, tiled).to_host()
    var wide = logaddexp(a, row).to_host()
    var same2 = logaddexp2(a, tiled).to_host()
    var wide2 = logaddexp2(a, row).to_host()
    var step = heaviside(a, h0s).to_host()
    var step_b = heaviside(a, h0).to_host()
    for i in range(6):
        assert_equal(wide[i], same[i])
        assert_equal(wide2[i], same2[i])
        assert_equal(step_b[i], step[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
