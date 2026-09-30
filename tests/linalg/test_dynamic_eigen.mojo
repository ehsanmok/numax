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
from numax.linalg import eigh, eigvalsh

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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
