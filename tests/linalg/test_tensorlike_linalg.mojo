"""`numax.linalg` over the `TensorLike` bound: a `View` of a sub-block is
the same argument as an owned `Tensor`.

The claim the design makes is that one `cholesky` runs on a whole matrix or
on a quadrant of one without a copy. These tests build a 4x4 whose leading
2x2 block is symmetric positive definite, factor that block through a
`View`, and check the factor against the owned copy, then do the same for
`solve`, `det`, `matmul` and `transpose`.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from numax.core.array import Static, arange, reshape, transpose, zeros
from numax.core.tensorlike import View
from numax.linalg import cholesky, det, matmul, solve

comptime f64 = DType.float64


def _big() raises -> Static[f64, 4, 4]:
    # Leading 2x2 block [[4, 2], [2, 3]] is SPD; the rest is filler.
    var a = zeros[f64, 4, 4]()
    a[0, 0] = 4.0
    a[0, 1] = 2.0
    a[1, 0] = 2.0
    a[1, 1] = 3.0
    for i in range(4):
        for j in range(4):
            if i >= 2 or j >= 2:
                a[i, j] = Float64(10 * i + j)
    return a^


def test_cholesky_of_a_view_block_matches_the_owned_block() raises:
    var big = _big()
    var block = View(big.view().tile[2, 2](0, 0), big.context())
    var owned = Static[f64, 2, 2](big.context(), [4.0, 2.0, 2.0, 3.0])
    var from_view = cholesky(block).to_host()
    var from_owned = cholesky(owned).to_host()
    for i in range(4):
        assert_almost_equal(from_view[i], from_owned[i])
    # The parent is untouched: cholesky allocates its result.
    assert_equal(big[0, 1], 2.0)
    assert_equal(big[2, 3], 23.0)


def test_solve_and_det_accept_a_view() raises:
    var big = _big()
    var block = View(big.view().tile[2, 2](0, 0), big.context())
    var owned = Static[f64, 2, 2](big.context(), [4.0, 2.0, 2.0, 3.0])
    var rhs = Static[f64, 2](big.context(), [1.0, 2.0])
    var x_view = solve(block, rhs).to_host()
    var x_owned = solve(owned, rhs).to_host()
    assert_almost_equal(x_view[0], x_owned[0])
    assert_almost_equal(x_view[1], x_owned[1])
    assert_almost_equal(det(block), det(owned))
    assert_almost_equal(det(block), 8.0)


def test_matmul_and_transpose_accept_views() raises:
    var m = reshape[rows=2, cols=3](arange[6, f64]())
    var vm = View(m.view())
    var t_owned = transpose(m)
    var t_view = transpose(vm)
    var p_owned = matmul(m, t_owned).to_host()
    var p_view = matmul(vm, View(t_view.view())).to_host()
    for i in range(4):
        assert_almost_equal(p_owned[i], p_view[i])
    assert_almost_equal(p_view[0], 5.0)
    assert_almost_equal(p_view[3], 50.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
