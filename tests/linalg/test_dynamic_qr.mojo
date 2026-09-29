"""Tests for the run-time-shape `qr_factor` and `lstsq`: bit-identical to
the static overloads on a `90 x 37` matrix (three panels and a
remainder), `Q` orthonormal and `Q R = a`, a least-squares solution whose
residual is orthogonal to the columns, reuse across right-hand sides, and
the shape checks."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

from layout.tile_layout import row_major

from numax.core.tensor import Dynamic, Static, _dyn_shape, asarray
from numax.linalg import lstsq, qr_factor

comptime dtype = DType.float64
comptime m = 90
comptime n = 37


def _values() -> List[Scalar[dtype]]:
    var v = List[Scalar[dtype]](capacity=m * n)
    for i in range(m):
        for j in range(n):
            v.append(
                Scalar[dtype](
                    Float64((i * 7 + j * 13) % 11) * 0.1
                    - 0.5
                    + (1.0 if i == j else 0.0)
                )
            )
    return v^


def _rhs(shift: Int) -> List[Scalar[dtype]]:
    var b = List[Scalar[dtype]](capacity=m)
    for i in range(m):
        b.append(Scalar[dtype](Float64((i + shift) % 5) - 2.0))
    return b^


def _dynamic() raises -> Dynamic[dtype, 2]:
    return Dynamic[dtype, 2](row_major(_dyn_shape[2](m, n)), _values())


def test_matches_static() raises:
    """`lstsq`, `Q` and `R` equal the static overloads' exactly."""
    var xd = lstsq(_dynamic(), asarray(_rhs(0))).to_host()
    var xs = lstsq(Static[dtype, m, n](_values()), Static[dtype, m](_rhs(0)))
    var xsh = xs.to_host()
    for i in range(n):
        assert_equal(Float64(xd[i]), Float64(xsh[i]))
    var f = qr_factor(_dynamic())
    var g = qr_factor(Static[dtype, m, n](_values()))
    var qd = f.q().to_host()
    var qs = g.q().to_host()
    for e in range(m * n):
        assert_equal(Float64(qd[e]), Float64(qs[e]))
    var rd = f.r().to_host()
    var rs = g.r().to_host()
    for e in range(n * n):
        assert_equal(Float64(rd[e]), Float64(rs[e]))


def test_factors() raises:
    """`Q^T Q = I`, `R` upper triangular, and `Q R = a`."""
    var f = qr_factor(_dynamic())
    var q = f.q().to_host()
    var r = f.r().to_host()
    var a = _values()
    for i in range(n):
        for j in range(n):
            var s = 0.0
            for k in range(m):
                s += Float64(q[k * n + i]) * Float64(q[k * n + j])
            assert_almost_equal(s, 1.0 if i == j else 0.0, atol=1e-12)
            if j < i:
                assert_equal(Float64(r[i * n + j]), 0.0)
    for i in range(m):
        for j in range(n):
            var s = 0.0
            for k in range(n):
                s += Float64(q[i * n + k]) * Float64(r[k * n + j])
            assert_almost_equal(s, Float64(a[i * n + j]), atol=1e-12)


def test_normal_equations() raises:
    """One factorization, two right-hand sides; each residual is
    orthogonal to every column of `a`."""
    var f = qr_factor(_dynamic())
    var a = _values()
    for shift in range(2):
        var b = _rhs(shift)
        var x = f.solve(asarray(b.copy())).to_host()
        for j in range(n):
            var s = 0.0
            for i in range(m):
                var ax = 0.0
                for k in range(n):
                    ax += Float64(a[i * n + k]) * Float64(x[k])
                s += Float64(a[i * n + j]) * (ax - Float64(b[i]))
            assert_almost_equal(s, 0.0, atol=1e-10)


def test_shape_errors() raises:
    """A wide matrix and a short right-hand side raise."""
    var wide = Dynamic[dtype, 2](
        row_major(_dyn_shape[2](2, 3)), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    with assert_raises(contains="m >= n"):
        _ = qr_factor(wide)
    var f = qr_factor(_dynamic())
    with assert_raises(contains="entries"):
        var short: List[Scalar[dtype]] = [1.0, 2.0, 3.0]
        _ = f.solve(asarray(short^))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
