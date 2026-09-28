"""Tests for `lu`'s `(P, L, U)` against SciPy's on a matrix whose first
pivot is zero, the reconstruction `P @ L @ U = A`, the factors' shapes
(`L` unit lower, `U` upper, `P` a permutation), and agreement with
`lu_factor`'s determinant."""

from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_almost_equal

from numax.core.tensor import Static
from numax.linalg import det, lu, matmul

comptime f64 = DType.float64


def _a() raises -> Static[f64, 4, 4]:
    return Static[f64, 4, 4](
        [
            0.0,
            2.0,
            1.0,
            3.0,
            4.0,
            1.0,
            -2.0,
            1.0,
            2.0,
            5.0,
            3.0,
            -1.0,
            1.0,
            -3.0,
            2.0,
            6.0,
        ],
        DeviceContext(api="cpu"),
    )


def _close(got: List[Float64], want: List[Float64]) raises:
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-14)


def test_matches_scipy() raises:
    var f = lu(_a())
    # scipy.linalg.lu.
    _close(
        f.p.to_host(),
        [
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
    )
    _close(
        f.l.to_host(),
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.5,
            1.0,
            0.0,
            0.0,
            0.25,
            -0.7222222222222222,
            1.0,
            0.0,
            0.0,
            0.4444444444444444,
            -0.14432989690721645,
            1.0,
        ],
    )
    _close(
        f.u.to_host(),
        [
            4.0,
            1.0,
            -2.0,
            1.0,
            0.0,
            4.5,
            4.0,
            -1.5,
            0.0,
            0.0,
            5.388888888888889,
            4.666666666666667,
            0.0,
            0.0,
            0.0,
            4.34020618556701,
        ],
    )


def test_reconstructs_and_agrees_with_det() raises:
    var f = lu[block=2](_a())
    var back = matmul(matmul(f.p, f.l), f.u).to_host()
    _close(back, _a().to_host())
    # det(A) = det(P) prod(diag(U)); this `P` is a 4-cycle, an odd
    # permutation, so det(P) = -1.
    var u = f.u.to_host()
    var product = u[0] * u[5] * u[10] * u[15]
    assert_almost_equal(det(_a()), -product, atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
