"""Tests for `ldl` against SciPy's `(lu, d, perm)` on two indefinite
matrices -- one with a `2 x 2` pivot block and interchanges, one whose
diagonal is all zero -- the reconstruction `lu d lu^T = a`, `lu[perm]`
being unit lower triangular, and a positive definite matrix taking only
`1 x 1` pivots."""

from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_almost_equal, assert_equal

from numax.core.tensor import Static, transpose
from numax.linalg import ldl, matmul

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _close(got: List[Float64], want: List[Float64]) raises:
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-14)


def _a() raises -> Static[f64, 4, 4]:
    return Static[f64, 4, 4](
        [
            1.0,
            2.0,
            3.0,
            4.0,
            2.0,
            -1.0,
            0.5,
            2.0,
            3.0,
            0.5,
            0.2,
            -1.0,
            4.0,
            2.0,
            -1.0,
            0.1,
        ],
        _cpu(),
    )


def test_matches_scipy() raises:
    var f = ldl(_a())
    # scipy.linalg.ldl.
    _close(
        f.lu.to_host(),
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.49056603773584906,
            0.37735849056603776,
            -0.3249656121045393,
            1.0,
            -0.27044025157232704,
            0.8176100628930818,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
    )
    _close(
        f.d.to_host(),
        [
            1.0,
            4.0,
            0.0,
            0.0,
            4.0,
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            1.828930817610063,
            0.0,
            0.0,
            0.0,
            0.0,
            -2.9289889958734525,
        ],
    )
    var perm = f.perm.to_host()
    var want: List[Int] = [0, 3, 2, 1]
    for i in range(4):
        assert_equal(Int(perm[i]), want[i])


def test_zero_diagonal_matches_scipy() raises:
    var b = Static[f64, 3, 3](
        [0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0], _cpu()
    )
    var f = ldl(b)
    _close(f.lu.to_host(), [1.0, 0.0, 0.0, 1.5, 0.5, 1.0, 0.0, 1.0, 0.0])
    _close(f.d.to_host(), [0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, -3.0])
    var perm = f.perm.to_host()
    var want: List[Int] = [0, 2, 1]
    for i in range(3):
        assert_equal(Int(perm[i]), want[i])


def test_reconstruction_and_triangularity() raises:
    var f = ldl(_a())
    var back = matmul(matmul(f.lu, f.d), transpose(f.lu)).to_host()
    _close(back, _a().to_host())
    var lu = f.lu.to_host()
    var perm = f.perm.to_host()
    for r in range(4):
        var row = Int(perm[r])
        assert_almost_equal(lu[row * 4 + r], 1.0, atol=0.0)
        for c in range(r + 1, 4):
            assert_almost_equal(lu[row * 4 + c], 0.0, atol=0.0)


def test_definite_takes_only_scalar_pivots() raises:
    var s = Static[f64, 3, 3](
        [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0], _cpu()
    )
    var f = ldl(s)
    var d = f.d.to_host()
    for i in range(3):
        for j in range(3):
            if i != j:
                assert_almost_equal(d[i * 3 + j], 0.0, atol=0.0)
    var back = matmul(matmul(f.lu, f.d), transpose(f.lu)).to_host()
    _close(back, s.to_host())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
