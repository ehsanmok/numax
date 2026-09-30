"""Tests for the run-time-shape spectral routines: each against the static
spelling on the same matrix -- the static overloads are wrappers over the
same run-time-shaped bodies, so the two agree to the bit -- plus the
defining identities at a size the static tests do not instantiate, and the
shape checks."""

from std.math import sin
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

from layout.tile_layout import row_major

from numax.core.tensor import Dynamic, Static, _dyn_shape
from numax.linalg import eigh, eigvalsh, svd, svdvals

comptime dtype = DType.float64


def _entry(i: Int, j: Int) -> Float64:
    """A symmetric entry with a spread spectrum and no structure."""
    var lo = min(i, j)
    var hi = max(i, j)
    var v = sin(Float64(lo * 31 + hi * 17) * 0.37)
    if i == j:
        v += Float64(i % 7) * 0.5
    return v


def _values(n: Int) -> List[Scalar[dtype]]:
    var values = List[Scalar[dtype]](capacity=n * n)
    for i in range(n):
        for j in range(n):
            values.append(Scalar[dtype](_entry(i, j)))
    return values^


def _dynamic(n: Int) raises -> Dynamic[dtype, 2]:
    return Dynamic[dtype, 2](row_major(_dyn_shape[2](n, n)), _values(n))


def _static[n: Int]() raises -> Static[dtype, n, n]:
    return Static[dtype, n, n](_values(n))


def _rect_entry(i: Int, j: Int) -> Float64:
    return sin(Float64(i * 13 + j * 29) * 0.41) + (1.0 if i == j else 0.0)


def _rect_values(m: Int, n: Int) -> List[Scalar[dtype]]:
    var values = List[Scalar[dtype]](capacity=m * n)
    for i in range(m):
        for j in range(n):
            values.append(Scalar[dtype](_rect_entry(i, j)))
    return values^


def _rect_dynamic(m: Int, n: Int) raises -> Dynamic[dtype, 2]:
    return Dynamic[dtype, 2](row_major(_dyn_shape[2](m, n)), _rect_values(m, n))


def _rect_static[m: Int, n: Int]() raises -> Static[dtype, m, n]:
    return Static[dtype, m, n](_rect_values(m, n))


def _same(got: List[Scalar[dtype]], want: List[Scalar[dtype]]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def test_eigvalsh_dynamic_matches_static_bit_for_bit() raises:
    """Both reductions: `sytrd` at `n = 45`, and the two-stage band above
    `n = 256`."""
    _same(eigvalsh(_dynamic(45)).to_host(), eigvalsh(_static[45]()).to_host())
    _same(eigvalsh(_dynamic(260)).to_host(), eigvalsh(_static[260]()).to_host())


def test_eigh_dynamic_matches_static_bit_for_bit() raises:
    var d = eigh(_dynamic(45))
    var s = eigh(_static[45]())
    _same(d.values.to_host(), s.values.to_host())
    _same(d.vectors.to_host(), s.vectors.to_host())
    assert_equal(d.values.dim[0](), 45)
    assert_equal(d.vectors.dim[0](), 45)
    assert_equal(d.vectors.dim[1](), 45)


def test_eigh_dynamic_is_an_eigendecomposition() raises:
    """`A V = V diag(w)`, `V^T V = I` and ascending `w` at two sizes one
    compiled program serves, one of them a single panel and a remainder."""
    for n in [7, 70]:
        var r = eigh(_dynamic(n))
        var w = r.values.to_host()
        var v = r.vectors.to_host()
        for j in range(1, n):
            assert_equal(w[j - 1] <= w[j], True)
        for i in range(n):
            for j in range(n):
                var av = 0.0
                var vtv = 0.0
                for k in range(n):
                    av += _entry(i, k) * Float64(v[k * n + j])
                    vtv += Float64(v[k * n + i]) * Float64(v[k * n + j])
                assert_almost_equal(
                    av, Float64(w[j]) * Float64(v[i * n + j]), atol=1e-10
                )
                assert_almost_equal(vtv, 1.0 if i == j else 0.0, atol=1e-12)
        var only = eigvalsh(_dynamic(n)).to_host()
        for j in range(n):
            assert_almost_equal(Float64(only[j]), Float64(w[j]), atol=1e-11)


def test_svdvals_dynamic_matches_static_bit_for_bit() raises:
    """Both reductions: `gebrd` below 32 columns, the two stages from it."""
    _same(
        svdvals(_rect_dynamic(40, 20)).to_host(),
        svdvals(_rect_static[40, 20]()).to_host(),
    )
    _same(
        svdvals(_rect_dynamic(60, 40)).to_host(),
        svdvals(_rect_static[60, 40]()).to_host(),
    )


def test_svd_dynamic_matches_static_bit_for_bit() raises:
    var d = svd(_rect_dynamic(50, 37))
    var s = svd(_rect_static[50, 37]())
    _same(d.u.to_host(), s.u.to_host())
    _same(d.s.to_host(), s.s.to_host())
    _same(d.v.to_host(), s.v.to_host())
    assert_equal(d.u.dim[0](), 50)
    assert_equal(d.u.dim[1](), 37)
    assert_equal(d.v.dim[0](), 37)


def test_svd_dynamic_reconstructs() raises:
    """`U diag(s) V^T = A`, `U^T U = V^T V = I` and `s` descending at two
    shapes one compiled program serves, one square."""
    for shape in [(9, 4), (33, 33)]:
        var m = shape[0]
        var n = shape[1]
        var r = svd(_rect_dynamic(m, n))
        var u = r.u.to_host()
        var sv = r.s.to_host()
        var v = r.v.to_host()
        for j in range(1, n):
            assert_equal(sv[j - 1] >= sv[j], True)
        for i in range(m):
            for j in range(n):
                var acc = 0.0
                for k in range(n):
                    acc += (
                        Float64(u[i * n + k])
                        * Float64(sv[k])
                        * Float64(v[j * n + k])
                    )
                assert_almost_equal(acc, _rect_entry(i, j), atol=1e-11)
        for i in range(n):
            for j in range(n):
                var utu = 0.0
                var vtv = 0.0
                for k in range(m):
                    utu += Float64(u[k * n + i]) * Float64(u[k * n + j])
                for k in range(n):
                    vtv += Float64(v[k * n + i]) * Float64(v[k * n + j])
                var want = 1.0 if i == j else 0.0
                assert_almost_equal(utu, want, atol=1e-12)
                assert_almost_equal(vtv, want, atol=1e-12)
        var only = svdvals(_rect_dynamic(m, n)).to_host()
        for j in range(n):
            assert_almost_equal(Float64(only[j]), Float64(sv[j]), atol=1e-11)


def test_one_by_one() raises:
    var r = eigh(_dynamic(1))
    assert_almost_equal(Float64(r.values.to_host()[0]), _entry(0, 0))
    assert_almost_equal(Float64(r.vectors.to_host()[0]), 1.0)


def test_dynamic_spectral_shape_checks() raises:
    var rect = Dynamic[dtype, 2](
        row_major(_dyn_shape[2](3, 4)), List[Scalar[dtype]](length=12, fill=0)
    )
    with assert_raises(contains="eigvalsh: the matrix must be square"):
        _ = eigvalsh(rect)
    with assert_raises(contains="eigh: the matrix must be square"):
        _ = eigh(rect)
    var empty = Dynamic[dtype, 2](
        row_major(_dyn_shape[2](0, 0)), List[Scalar[dtype]]()
    )
    with assert_raises(contains="eigh: the matrix is empty"):
        _ = eigh(empty)
    with assert_raises(contains="svd: the matrix must have at least as many"):
        _ = svd(rect)
    with assert_raises(contains="svdvals: the matrix must have at least"):
        _ = svdvals(rect)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
