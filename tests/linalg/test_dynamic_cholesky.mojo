"""Tests for the run-time-shape `cholesky`: bit-identical to the static
overload past two diagonal blocks (the two share `_potrf_blocked`), the
reconstruction `L L^T = a` with exact zeros above the diagonal, and the
non-positive-definite and non-square errors."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

from layout.tile_layout import row_major

from numax.core.tensor import Dynamic, Static, _dyn_shape
from numax.linalg import cholesky

comptime dtype = DType.float64


def _values(n: Int) -> List[Scalar[dtype]]:
    var v = List[Scalar[dtype]](capacity=n * n)
    for i in range(n):
        for j in range(n):
            v.append(
                Scalar[dtype](
                    Float64(n) if i == j else 1.0 / (1.0 + Float64(abs(i - j)))
                )
            )
    return v^


def test_matches_static() raises:
    """`n = 150` crosses two 64-wide blocks and a remainder; every entry
    equals the static overload's exactly."""
    comptime n = 150
    var d = Dynamic[dtype, 2](row_major(_dyn_shape[2](n, n)), _values(n))
    var s = Static[dtype, n, n](_values(n))
    var ld = cholesky(d).to_host()
    var ls = cholesky(s).to_host()
    for e in range(n * n):
        assert_equal(Float64(ld[e]), Float64(ls[e]))


def test_reconstructs() raises:
    """`L L^T` reproduces `a`, and `L` is zero above the diagonal."""
    var n = 7
    var v = _values(n)
    var l = cholesky(
        Dynamic[dtype, 2](row_major(_dyn_shape[2](n, n)), v.copy())
    ).to_host()
    for i in range(n):
        for j in range(n):
            if j > i:
                assert_equal(Float64(l[i * n + j]), 0.0)
            var s = 0.0
            for k in range(n):
                s += Float64(l[i * n + k]) * Float64(l[j * n + k])
            assert_almost_equal(s, Float64(v[i * n + j]), atol=1e-12)


def test_errors() raises:
    """An indefinite matrix names its pivot; a rectangle raises."""
    var indefinite = Dynamic[dtype, 2](
        row_major(_dyn_shape[2](2, 2)), [1.0, 2.0, 2.0, 1.0]
    )
    with assert_raises(contains="not positive definite"):
        _ = cholesky(indefinite)
    var rect = Dynamic[dtype, 2](
        row_major(_dyn_shape[2](2, 3)), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    with assert_raises(contains="square"):
        _ = cholesky(rect)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
