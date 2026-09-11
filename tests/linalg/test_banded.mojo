"""Tests for `numax.linalg.banded`.

Every solver here is checked against an independent route to the same
answer, because a banded storage scheme is exactly the kind of thing whose
index arithmetic a test written from the same reasoning would share.

- `solve_banded` against the dense `numax.linalg.solve` on the matrix the
  band describes, built with `numax.linalg.toeplitz` or by hand.
- `solve_banded` at `l == u == 1` against
  `numax.linalg.array.tridiagonal_solve`, which is the seam the per-feature
  gate asks about: a new spelling of an existing primitive must agree with
  it.
- `solveh_banded` against `solve_banded` on the same matrix expanded to the
  general form.
- `solve_toeplitz` against the dense solve on the materialized Toeplitz
  matrix.
"""

from std.collections import Array
from std.testing import TestSuite, assert_almost_equal, assert_true

from max.gpu.host import DeviceContext

from numax import Plain
from numax.core.array import Static
from numax.linalg import (
    cho_solve_banded,
    cholesky_banded,
    matvec,
    solve,
    circulant,
    solve_banded,
    solve_circulant,
    solve_toeplitz,
    solveh_banded,
    toeplitz,
)
from numax.linalg.array import tridiagonal_solve

comptime dtype = DType.float64
comptime P = Plain[DType.float64, 1]


def _dense_from_band(
    band: List[Float64], l: Int, u: Int, n: Int, ctx: DeviceContext
) raises -> Static[dtype, 5, 5]:
    """The dense matrix a `(l + u + 1) x n` band describes, for `n == 5`."""
    var entries = List[Scalar[dtype]](length=25, fill=0)
    for i in range(n):
        for j in range(n):
            var offset = u + i - j
            if offset >= 0 and offset < l + u + 1:
                entries[i * n + j] = Scalar[dtype](band[offset * n + j])
    return Static[dtype, 5, 5](ctx, entries^)


# --- solve_banded -----------------------------------------------------------


def test_solve_banded_matches_the_dense_solve() raises:
    """A 5x5 with one sub- and two superdiagonals, solved both ways."""
    var ctx = DeviceContext(api="cpu")
    comptime l = 1
    comptime u = 2
    comptime n = 5

    # Rows of `ab`: two superdiagonals, the diagonal, one subdiagonal.
    var band = List[Float64](length=(l + u + 1) * n, fill=0.0)
    var flat = [
        0.0,
        0.0,
        1.0,
        -2.0,
        3.0,
        0.0,
        2.0,
        -1.0,
        4.0,
        1.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        1.0,
        -1.0,
        2.0,
        3.0,
        0.0,
    ]
    for i in range(len(flat)):
        band[i] = flat[i]

    var entries = List[Scalar[dtype]](capacity=(l + u + 1) * n)
    for i in range((l + u + 1) * n):
        entries.append(Scalar[dtype](band[i]))
    var ab = Static[dtype, l + u + 1, n](ctx, entries^)
    var b = Static[dtype, n](ctx, [1.0, 2.0, 3.0, 4.0, 5.0])
    var got = solve_banded[dtype, l, u, n](ab, b).to_host()

    var dense = _dense_from_band(band, l, u, n, ctx)
    var b2 = Static[dtype, n](ctx, [1.0, 2.0, 3.0, 4.0, 5.0])
    var want = solve[dtype, n](dense, b2).to_host()

    for i in range(n):
        assert_almost_equal(Float64(got[i]), Float64(want[i]), atol=1e-9)


def test_solve_banded_at_bandwidth_one_agrees_with_tridiagonal_solve() raises:
    """The seam. `tridiagonal_solve` is the existing primitive and this is
    a new spelling reaching the same answer, so they must agree."""
    var ctx = DeviceContext(api="cpu")
    comptime n = 5
    var sub = [0.0, 1.0, 1.0, 1.0, 1.0]
    var diag = [4.0, 4.0, 4.0, 4.0, 4.0]
    var sup = [1.0, 1.0, 1.0, 1.0, 0.0]
    var rhs = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Banded form: row 0 the superdiagonal, row 1 the diagonal, row 2 the
    # subdiagonal. `ab[u + i - j, j]`, so the superdiagonal entry at
    # `(j-1, j)` sits at row 0 column j.
    var entries = List[Scalar[dtype]](length=3 * n, fill=0)
    for j in range(n):
        entries[1 * n + j] = Scalar[dtype](diag[j])
    for j in range(1, n):
        entries[0 * n + j] = Scalar[dtype](sup[j - 1])
    for j in range(n - 1):
        entries[2 * n + j] = Scalar[dtype](sub[j + 1])

    var ab = Static[dtype, 3, n](ctx, entries^)
    var b = Static[dtype, n](ctx, [1.0, 2.0, 3.0, 4.0, 5.0])
    var got = solve_banded[dtype, 1, 1, n](ab, b).to_host()

    var sub_a = Array[P, n](fill=P.constant(0.0))
    var diag_a = Array[P, n](fill=P.constant(0.0))
    var sup_a = Array[P, n](fill=P.constant(0.0))
    var rhs_a = Array[P, n](fill=P.constant(0.0))
    for i in range(n):
        sub_a[i] = P.constant(sub[i])
        diag_a[i] = P.constant(diag[i])
        sup_a[i] = P.constant(sup[i])
        rhs_a[i] = P.constant(rhs[i])
    var want = tridiagonal_solve[P, n](sub_a, diag_a, sup_a, rhs_a)

    for i in range(n):
        assert_almost_equal(Float64(got[i]), want[i].v, atol=1e-10)


def test_solve_banded_pivots() raises:
    """A matrix whose first diagonal entry is zero. Without pivoting the
    elimination divides by it; with pivoting the answer is exact."""
    var ctx = DeviceContext(api="cpu")
    comptime n = 3
    # [[0, 1, 0], [2, 1, 1], [0, 1, 3]] -- banded with l = u = 1.
    var entries = List[Scalar[dtype]](length=3 * n, fill=0)
    entries[0 * n + 1] = 1.0
    entries[0 * n + 2] = 1.0
    entries[1 * n + 0] = 0.0
    entries[1 * n + 1] = 1.0
    entries[1 * n + 2] = 3.0
    entries[2 * n + 0] = 2.0
    entries[2 * n + 1] = 1.0

    var ab = Static[dtype, 3, n](ctx, entries^)
    var b = Static[dtype, n](ctx, [1.0, 2.0, 3.0])
    var got = solve_banded[dtype, 1, 1, n](ab, b).to_host()

    # Verify by substitution rather than against another solver.
    var x0 = Float64(got[0])
    var x1 = Float64(got[1])
    var x2 = Float64(got[2])
    assert_almost_equal(x1, 1.0, atol=1e-12)
    assert_almost_equal(2.0 * x0 + x1 + x2, 2.0, atol=1e-12)
    assert_almost_equal(x1 + 3.0 * x2, 3.0, atol=1e-12)


def test_a_singular_band_raises() raises:
    var ctx = DeviceContext(api="cpu")
    var entries = List[Scalar[dtype]](length=9, fill=0)
    var ab = Static[dtype, 3, 3](ctx, entries^)
    var b = Static[dtype, 3](ctx, [1.0, 2.0, 3.0])
    var raised = False
    try:
        _ = solve_banded[dtype, 1, 1, 3](ab, b)
    except e:
        raised = True
        assert_true("singular" in String(e))
    assert_true(raised)


# --- the symmetric routines -------------------------------------------------


def test_solveh_banded_matches_solve_banded() raises:
    """The same symmetric positive definite matrix through both routes."""
    var ctx = DeviceContext(api="cpu")
    comptime n = 5
    comptime u = 1
    # Second-difference operator: 4 on the diagonal, -1 off it.
    var upper = List[Scalar[dtype]](length=(u + 1) * n, fill=0)
    for j in range(1, n):
        upper[0 * n + j] = -1.0
    for j in range(n):
        upper[1 * n + j] = 4.0
    var ab = Static[dtype, u + 1, n](ctx, upper^)
    var b = Static[dtype, n](ctx, [1.0, 2.0, 3.0, 4.0, 5.0])
    var got = solveh_banded[dtype, u, n](ab, b).to_host()

    var general = List[Scalar[dtype]](length=3 * n, fill=0)
    for j in range(1, n):
        general[0 * n + j] = -1.0
    for j in range(n):
        general[1 * n + j] = 4.0
    for j in range(n - 1):
        general[2 * n + j] = -1.0
    var ab2 = Static[dtype, 3, n](ctx, general^)
    var b2 = Static[dtype, n](ctx, [1.0, 2.0, 3.0, 4.0, 5.0])
    var want = solve_banded[dtype, 1, 1, n](ab2, b2).to_host()

    for i in range(n):
        assert_almost_equal(Float64(got[i]), Float64(want[i]), atol=1e-10)


def test_the_lower_and_upper_forms_agree() raises:
    """A symmetric matrix has nothing to distinguish its two triangles, so
    both spellings must give the same solution."""
    var ctx = DeviceContext(api="cpu")
    comptime n = 4
    comptime u = 1
    var upper = List[Scalar[dtype]](length=(u + 1) * n, fill=0)
    for j in range(1, n):
        upper[0 * n + j] = -1.0
    for j in range(n):
        upper[1 * n + j] = 4.0
    var ab_u = Static[dtype, u + 1, n](ctx, upper^)
    var b_u = Static[dtype, n](ctx, [1.0, 2.0, 3.0, 4.0])
    var from_upper = solveh_banded[dtype, u, n, False](ab_u, b_u).to_host()

    var lower = List[Scalar[dtype]](length=(u + 1) * n, fill=0)
    for j in range(n):
        lower[0 * n + j] = 4.0
    for j in range(n - 1):
        lower[1 * n + j] = -1.0
    var ab_l = Static[dtype, u + 1, n](ctx, lower^)
    var b_l = Static[dtype, n](ctx, [1.0, 2.0, 3.0, 4.0])
    var from_lower = solveh_banded[dtype, u, n, True](ab_l, b_l).to_host()

    for i in range(n):
        assert_almost_equal(
            Float64(from_upper[i]), Float64(from_lower[i]), atol=1e-12
        )


def test_cholesky_banded_then_cho_solve_banded_is_solveh_banded() raises:
    var ctx = DeviceContext(api="cpu")
    comptime n = 4
    comptime u = 1
    var upper = List[Scalar[dtype]](length=(u + 1) * n, fill=0)
    for j in range(1, n):
        upper[0 * n + j] = -1.0
    for j in range(n):
        upper[1 * n + j] = 4.0
    var ab = Static[dtype, u + 1, n](ctx, upper^)
    var factor = cholesky_banded[dtype, u, n](ab)
    var b = Static[dtype, n](ctx, [1.0, 2.0, 3.0, 4.0])
    var two_step = cho_solve_banded[dtype, u, n](factor, b).to_host()

    var upper2 = List[Scalar[dtype]](length=(u + 1) * n, fill=0)
    for j in range(1, n):
        upper2[0 * n + j] = -1.0
    for j in range(n):
        upper2[1 * n + j] = 4.0
    var ab2 = Static[dtype, u + 1, n](ctx, upper2^)
    var b2 = Static[dtype, n](ctx, [1.0, 2.0, 3.0, 4.0])
    var one_step = solveh_banded[dtype, u, n](ab2, b2).to_host()

    for i in range(n):
        assert_almost_equal(
            Float64(two_step[i]), Float64(one_step[i]), atol=0.0
        )


def test_an_indefinite_matrix_raises() raises:
    var ctx = DeviceContext(api="cpu")
    var entries = List[Scalar[dtype]](length=2 * 3, fill=0)
    for j in range(3):
        entries[1 * 3 + j] = -1.0
    var ab = Static[dtype, 2, 3](ctx, entries^)
    var raised = False
    try:
        _ = cholesky_banded[dtype, 1, 3](ab)
    except e:
        raised = True
        assert_true("positive definite" in String(e))
    assert_true(raised)


# --- solve_toeplitz ---------------------------------------------------------


def test_solve_toeplitz_matches_the_dense_solve() raises:
    """Levinson against the materialized matrix through the dense solve."""
    var ctx = DeviceContext(api="cpu")
    comptime n = 5
    var c = Static[dtype, n](ctx, [4.0, 1.0, 0.5, 0.25, 0.125])
    var r = Static[dtype, n](ctx, [4.0, 2.0, 1.0, 0.5, 0.25])
    var b = Static[dtype, n](ctx, [1.0, 2.0, 3.0, 4.0, 5.0])
    var got = solve_toeplitz[dtype, n](c, r, b).to_host()

    var c2 = Static[dtype, n](ctx, [4.0, 1.0, 0.5, 0.25, 0.125])
    var r2 = Static[dtype, n](ctx, [4.0, 2.0, 1.0, 0.5, 0.25])
    var dense = toeplitz[dtype, n, n](c2, r2)
    var b2 = Static[dtype, n](ctx, [1.0, 2.0, 3.0, 4.0, 5.0])
    var want = solve[dtype, n](dense, b2).to_host()

    for i in range(n):
        assert_almost_equal(Float64(got[i]), Float64(want[i]), atol=1e-8)


def test_solve_toeplitz_on_a_symmetric_system() raises:
    """The common case: an autocorrelation matrix, where every leading
    minor is nonsingular and Levinson is unconditionally safe."""
    var ctx = DeviceContext(api="cpu")
    comptime n = 4
    var c = Static[dtype, n](ctx, [2.0, 0.5, 0.25, 0.1])
    var r = Static[dtype, n](ctx, [2.0, 0.5, 0.25, 0.1])
    var b = Static[dtype, n](ctx, [1.0, 0.0, -1.0, 2.0])
    var got = solve_toeplitz[dtype, n](c, r, b)

    # Verify by multiplying back through the materialized matrix.
    var c2 = Static[dtype, n](ctx, [2.0, 0.5, 0.25, 0.1])
    var dense = toeplitz[dtype, n](c2)
    var residual = matvec[dtype, n, n](dense, got).to_host()
    var want = [1.0, 0.0, -1.0, 2.0]
    for i in range(n):
        assert_almost_equal(Float64(residual[i]), want[i], atol=1e-9)


def test_a_zero_leading_entry_raises() raises:
    var ctx = DeviceContext(api="cpu")
    var c = Static[dtype, 3](ctx, [0.0, 1.0, 2.0])
    var r = Static[dtype, 3](ctx, [0.0, 1.0, 2.0])
    var b = Static[dtype, 3](ctx, [1.0, 2.0, 3.0])
    var raised = False
    try:
        _ = solve_toeplitz[dtype, 3](c, r, b)
    except e:
        raised = True
        assert_true("singular" in String(e))
    assert_true(raised)


# --- solve_circulant --------------------------------------------------------


def test_solve_circulant_matches_the_dense_solve() raises:
    """Three FFTs and a division against Gaussian elimination on the
    materialized circulant. The two share no code at all."""
    var ctx = DeviceContext(api="cpu")
    comptime n = 4
    var c = Static[dtype, n](ctx, [4.0, 1.0, 2.0, 0.5])
    var b = Static[dtype, n](ctx, [1.0, 2.0, 3.0, 4.0])
    var got = solve_circulant[dtype, n](c, b).to_host()

    var c2 = Static[dtype, n](ctx, [4.0, 1.0, 2.0, 0.5])
    var dense = circulant[dtype, n](c2)
    var b2 = Static[dtype, n](ctx, [1.0, 2.0, 3.0, 4.0])
    var want = solve[dtype, n](dense, b2).to_host()

    for i in range(n):
        assert_almost_equal(Float64(got[i]), Float64(want[i]), atol=1e-10)


def test_solve_circulant_at_a_non_power_of_two_matches_scipy() raises:
    """`scipy.linalg.solve_circulant([4, 1, .5, -.25, .75, 2], [1, 2, 3, -1,
    .5, 4])`: `n = 6` takes `numax.fft`'s Bluestein path, which is the
    length this routine could not accept before."""
    var ctx = DeviceContext(api="cpu")
    comptime n = 6
    var c = Static[dtype, n](ctx, [4.0, 1.0, 0.5, -0.25, 0.75, 2.0])
    var b = Static[dtype, n](ctx, [1.0, 2.0, 3.0, -1.0, 0.5, 4.0])
    var got = solve_circulant[dtype, n](c, b).to_host()
    var want = [
        -0.20338344335456288,
        -0.21549817535214555,
        1.2966862884105805,
        -0.5300349239923758,
        -0.5995528450560177,
        1.4392830993445211,
    ]
    for i in range(n):
        assert_almost_equal(Float64(got[i]), want[i], atol=1e-12)


def test_solve_circulant_residual_is_zero() raises:
    """Multiply the answer back through the circulant it solved."""
    var ctx = DeviceContext(api="cpu")
    comptime n = 8
    var values = [3.0, 0.5, -1.0, 0.25, 2.0, 0.125, -0.5, 1.0]
    var c_entries = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        c_entries.append(Scalar[dtype](values[i]))
    var c = Static[dtype, n](ctx, c_entries^)
    var b = Static[dtype, n](ctx, [1.0, 0.0, -1.0, 2.0, 3.0, 1.0, 0.0, -2.0])
    var x = solve_circulant[dtype, n](c, b)

    var c_entries2 = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        c_entries2.append(Scalar[dtype](values[i]))
    var c2 = Static[dtype, n](ctx, c_entries2^)
    var dense = circulant[dtype, n](c2)
    var residual = matvec[dtype, n, n](dense, x).to_host()

    var want = [1.0, 0.0, -1.0, 2.0, 3.0, 1.0, 0.0, -2.0]
    for i in range(n):
        assert_almost_equal(Float64(residual[i]), want[i], atol=1e-9)


def test_a_singular_circulant_raises() raises:
    """A constant first column makes every eigenvalue but the first zero,
    so `fft(c)` has zeros and the matrix is singular."""
    var ctx = DeviceContext(api="cpu")
    var c = Static[dtype, 4](ctx, [1.0, 1.0, 1.0, 1.0])
    var b = Static[dtype, 4](ctx, [1.0, 2.0, 3.0, 4.0])
    var raised = False
    try:
        _ = solve_circulant[dtype, 4](c, b)
    except e:
        raised = True
        assert_true("singular" in String(e))
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
