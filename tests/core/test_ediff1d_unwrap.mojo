"""Tests for `ediff1d` and `unwrap` in `numax.core.elementwise`.

`ediff1d` against NumPy at rank 1 and rank 2 (read flat), with and
without the values it can prepend and append, and at the lengths where
there is nothing to difference. `unwrap` against NumPy's documented
examples -- a phase with a jump past `pi`, and integer sequences at a
`period` of 4 -- plus the `discont` rule that a jump smaller than it is
left alone.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.core.elementwise import ediff1d, unwrap

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def test_ediff1d_matches_numpy() raises:
    # numpy: np.ediff1d([1, 2, 4, 7, 0]) and with to_begin=-99, to_end=88
    var a = Static[f64, 5]([1.0, 2.0, 4.0, 7.0, 0.0], _cpu())
    var got = ediff1d(a).to_host()
    var want = [1.0, 2.0, 3.0, -7.0]
    assert_equal(len(got), 4)
    for i in range(4):
        assert_equal(Float64(got[i]), want[i])
    var padded = ediff1d(a, to_end=88.0, to_begin=-99.0).to_host()
    var want_padded = [-99.0, 1.0, 2.0, 3.0, -7.0, 88.0]
    assert_equal(len(padded), 6)
    for i in range(6):
        assert_equal(Float64(padded[i]), want_padded[i])


def test_ediff1d_reads_a_matrix_flat() raises:
    # numpy: np.ediff1d([[1, 2], [4, 7]])
    var m = Static[f64, 2, 2]([1.0, 2.0, 4.0, 7.0], _cpu())
    var got = ediff1d(m).to_host()
    assert_equal(len(got), 3)
    assert_equal(Float64(got[0]), 1.0)
    assert_equal(Float64(got[1]), 2.0)
    assert_equal(Float64(got[2]), 3.0)


def test_ediff1d_of_one_element_is_only_the_padding() raises:
    var one = Static[f64, 1]([5.0], _cpu())
    assert_equal(ediff1d(one).size(), 0)
    var got = ediff1d(one, to_end=88.0).to_host()
    assert_equal(len(got), 1)
    assert_equal(Float64(got[0]), 88.0)


def test_unwrap_removes_a_jump_past_pi() raises:
    # numpy docs: phase = np.linspace(0, np.pi, num=5); phase[3:] += np.pi
    # np.unwrap(phase) == [0, pi/4, pi/2, -pi/4, 0]: the `5 pi / 4` jump
    # is taken as `-3 pi / 4`.
    var pi = 3.141592653589793
    var phase = Static[f64, 5](
        [0.0, pi / 4, pi / 2, 3 * pi / 4 + pi, pi + pi], _cpu()
    )
    var got = unwrap(phase).to_host()
    var want = [0.0, pi / 4, pi / 2, -pi / 4, 0.0]
    for i in range(5):
        assert_almost_equal(Float64(got[i]), want[i], atol=1e-15)


def test_unwrap_at_an_integer_period_matches_numpy() raises:
    # numpy: np.unwrap([0, 1, 2, -1, 0], period=4) and
    #        np.unwrap([2, 3, 4, 5, 2, 3, 4, 5], period=4)
    var a = Static[f64, 5]([0.0, 1.0, 2.0, -1.0, 0.0], _cpu())
    var got = unwrap(a, period=4.0).to_host()
    for i in range(5):
        assert_equal(Float64(got[i]), Float64(i))
    var b = Static[f64, 8]([2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0], _cpu())
    var got_b = unwrap(b, period=4.0).to_host()
    for i in range(8):
        assert_equal(Float64(got_b[i]), Float64(i + 2))


def test_unwrap_leaves_jumps_under_discont() raises:
    # numpy: np.unwrap([0, 3.5]) and np.unwrap([0, 3.5], discont=4)
    var a = Static[f64, 2]([0.0, 3.5], _cpu())
    var wrapped = unwrap(a).to_host()
    assert_almost_equal(
        Float64(wrapped[1]), 3.5 - 2 * 3.141592653589793, atol=1e-15
    )
    var kept = unwrap(a, discont=4.0).to_host()
    assert_equal(Float64(kept[1]), 3.5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
