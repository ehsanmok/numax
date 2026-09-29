"""Tests for the run-time-shape `lu_factor`, `solve`, `det` and `slogdet`:
each against the static spelling on the same matrix (one compiled program,
two sizes), a mixed static matrix with a `Dynamic` right-hand side, the
swap parity in `det`'s sign, and the shape checks."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

from numax.core.tensor import Dynamic, Static, asarray
from numax.core.tensor import _dyn_shape
from layout.tile_layout import row_major
from numax.linalg import det, lu_factor, slogdet, solve

comptime dtype = DType.float64


def _entry(i: Int, j: Int, n: Int) -> Float64:
    if i == j:
        return Float64(n) + 1.0
    return Float64((i * 7 + j * 3) % 5) * 0.25 - 0.5


def _dynamic(n: Int) raises -> Dynamic[dtype, 2]:
    var values = List[Scalar[dtype]](capacity=n * n)
    for i in range(n):
        for j in range(n):
            values.append(Scalar[dtype](_entry(i, j, n)))
    return Dynamic[dtype, 2](row_major(_dyn_shape[2](n, n)), values^)


def _static[n: Int]() raises -> Static[dtype, n, n]:
    var values = List[Scalar[dtype]](capacity=n * n)
    for i in range(n):
        for j in range(n):
            values.append(Scalar[dtype](_entry(i, j, n)))
    return Static[dtype, n, n](values^)


def _rhs(n: Int) raises -> Dynamic[dtype, 1]:
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(Scalar[dtype](Float64(i) - 1.5))
    return asarray(values^)


def _check_size[n: Int]() raises:
    var a = _dynamic(n)
    var b = _rhs(n)
    var x = solve(a, b).to_host()
    var xs = solve(_static[n](), Static[dtype, n](_rhs(n).to_host())).to_host()
    for i in range(n):
        assert_almost_equal(Float64(x[i]), Float64(xs[i]), atol=1e-12)
    assert_almost_equal(Float64(det(a)), Float64(det(_static[n]())), rtol=1e-12)
    var pair = slogdet(a)
    var pair_s = slogdet(_static[n]())
    assert_equal(Float64(pair[0]), Float64(pair_s[0]))
    assert_almost_equal(Float64(pair[1]), Float64(pair_s[1]), atol=1e-12)


def test_matches_static_small() raises:
    """At `n = 5` the run-time overloads agree with the static ones."""
    _check_size[5]()


def test_matches_static_past_a_panel() raises:
    """At `n = 70`, past two 32-wide panels, they still agree."""
    _check_size[70]()


def test_factor_reused() raises:
    """One `DynamicLU` solves two right-hand sides, and each solve
    reproduces its `b` through `a`."""
    var n = 9
    var a = _dynamic(n)
    var f = lu_factor(a)
    assert_true(not f.singular())
    var host = a.to_host()
    for shift in range(2):
        var values = List[Scalar[dtype]](capacity=n)
        for i in range(n):
            values.append(Scalar[dtype](Float64(i * (shift + 1)) + 1.0))
        var b = asarray(values.copy())
        var x = f.solve(b).to_host()
        for i in range(n):
            var s = 0.0
            for j in range(n):
                s += Float64(host[i * n + j]) * Float64(x[j])
            assert_almost_equal(s, Float64(values[i]), atol=1e-10)


def test_swap_parity() raises:
    """The antidiagonal permutation of order 3 has one swap: `det = -1`,
    read from the pivots rather than the diagonal alone."""
    var a = Dynamic[dtype, 2](
        row_major(_dyn_shape[2](3, 3)),
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    )
    assert_almost_equal(Float64(det(a)), -1.0, atol=1e-15)


def test_static_matrix_dynamic_rhs() raises:
    """A static `a` with a `Dynamic` `b` takes the run-time overload."""
    var x = solve(_static[4](), _rhs(4)).to_host()
    var xs = solve(_static[4](), Static[dtype, 4](_rhs(4).to_host())).to_host()
    for i in range(4):
        assert_almost_equal(Float64(x[i]), Float64(xs[i]), atol=1e-12)


def test_shape_errors() raises:
    """A non-square matrix and a short right-hand side raise."""
    var rect = Dynamic[dtype, 2](
        row_major(_dyn_shape[2](2, 3)), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    with assert_raises(contains="square"):
        _ = lu_factor(rect)
    with assert_raises(contains="entries"):
        _ = solve(_dynamic(4), _rhs(3))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
