"""Tests for `numax.linalg`.

Structured around identities the factorizations must satisfy -- `L@L.T`
reproducing `A`, `A@x` reproducing `b`, `A@A^-1` reproducing the identity --
rather than against precomputed factor entries, since those only check one
matrix while the identity checks the algorithm.
"""

from std.collections import Array
from std.math import log as log_f64
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Compensated, Dual, FloatLike, Plain
from numax.linalg import (
    back_substitution,
    cholesky,
    cholesky_solve,
    det,
    forward_substitution,
    inverse,
    log_det_from_cholesky,
    lu,
    matmul,
    matvec,
    solve,
    tridiagonal_solve,
)

comptime dtype = DType.float64
comptime width = 1
comptime P = Plain[dtype, width]
comptime D = Dual[P]


def pv(x: Float64) -> P:
    return P(SIMD[dtype, width](x))


def s(x: P) -> Float64:
    return Float64(x.v)


def spd3() -> Array[P, 9]:
    """A symmetric positive definite 3x3, used throughout."""
    var a = Array[P, 9](fill=pv(0.0))
    var entries = [4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0]
    for i in range(9):
        a[i] = pv(entries[i])
    return a^


def general3() -> Array[P, 9]:
    """A non-symmetric 3x3 whose leading minors are all nonzero, so the
    unpivoted LU is valid for it."""
    var a = Array[P, 9](fill=pv(0.0))
    var entries = [2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0]
    for i in range(9):
        a[i] = pv(entries[i])
    return a^


def rhs3() -> Array[P, 3]:
    var b = Array[P, 3](fill=pv(0.0))
    b[0] = pv(8.0)
    b[1] = pv(-11.0)
    b[2] = pv(-3.0)
    return b^


def transpose3(m: Array[P, 9]) -> Array[P, 9]:
    var out = Array[P, 9](fill=pv(0.0))
    for i in range(3):
        for j in range(3):
            out[i * 3 + j] = m[j * 3 + i].copy()
    return out^


def test_cholesky_reproduces_the_matrix() raises:
    var a = spd3()
    var lower = cholesky[P, 3](a)
    var product = matmul[P, 3](lower, transpose3(lower))
    for i in range(9):
        assert_almost_equal(s(product[i]), s(a[i]), atol=1e-13)


def test_cholesky_upper_triangle_is_zero() raises:
    var lower = cholesky[P, 3](spd3())
    for i in range(3):
        for j in range(i + 1, 3):
            assert_almost_equal(s(lower[i * 3 + j]), 0.0)


def test_cholesky_solve_matches_the_general_solve() raises:
    var a = spd3()
    var b = rhs3()
    var lower = cholesky[P, 3](a)
    var from_chol = cholesky_solve[P, 3](lower, b)
    var from_lu = solve[P, 3](a, b)
    for i in range(3):
        assert_almost_equal(s(from_chol[i]), s(from_lu[i]), atol=1e-12)


def test_solve_reproduces_the_right_hand_side() raises:
    var a = general3()
    var b = rhs3()
    var x = solve[P, 3](a, b)
    var residual = matvec[P, 3](a, x)
    for i in range(3):
        assert_almost_equal(s(residual[i]), s(b[i]), atol=1e-12)


def test_solve_matches_a_known_answer() raises:
    # The textbook 3x3 whose solution is (2, 3, -1).
    var x = solve[P, 3](general3(), rhs3())
    assert_almost_equal(s(x[0]), 2.0, atol=1e-13)
    assert_almost_equal(s(x[1]), 3.0, atol=1e-13)
    assert_almost_equal(s(x[2]), -1.0, atol=1e-13)


def test_lu_factors_reproduce_the_matrix() raises:
    var a = general3()
    var packed = lu[P, 3](a)

    var lower = Array[P, 9](fill=pv(0.0))
    var upper = Array[P, 9](fill=pv(0.0))
    for i in range(3):
        lower[i * 3 + i] = pv(1.0)
        for j in range(3):
            if j < i:
                lower[i * 3 + j] = packed[i * 3 + j].copy()
            else:
                upper[i * 3 + j] = packed[i * 3 + j].copy()

    var product = matmul[P, 3](lower, upper)
    for i in range(9):
        assert_almost_equal(s(product[i]), s(a[i]), atol=1e-13)


def test_inverse_produces_the_identity() raises:
    var a = general3()
    var inv = inverse[P, 3](a)
    var product = matmul[P, 3](a, inv)
    for i in range(3):
        for j in range(3):
            var expected = 1.0 if i == j else 0.0
            assert_almost_equal(s(product[i * 3 + j]), expected, atol=1e-13)


def test_determinant_matches_the_cofactor_expansion() raises:
    # 4*(30-9) - 2*(12-3) + 1*(6-5) = 84 - 18 + 1 = 67.
    assert_almost_equal(s(det[P, 3](spd3())), 67.0, atol=1e-12)


def test_determinant_is_multiplicative() raises:
    # `B@A` rather than `A@B`: the latter happens to have a zero in its
    # top-left entry, which the unpivoted LU cannot factor -- see
    # `test_unpivoted_lu_fails_on_a_zero_pivot` below, which pins that
    # behaviour down deliberately.
    var a = spd3()
    var b = general3()
    var product = matmul[P, 3](b, a)
    assert_almost_equal(
        s(det[P, 3](product)),
        s(det[P, 3](a)) * s(det[P, 3](b)),
        atol=1e-10,
    )


def test_unpivoted_lu_fails_on_a_zero_pivot() raises:
    # The documented limitation, asserted rather than left implicit: the
    # exchange matrix [[0,1],[1,0]] is perfectly well-conditioned and has
    # determinant -1, but no pivoting means the factorization can't get
    # started, and the floored pivot yields ~0 instead. A caller who needs
    # this case needs a pivoting solver, which this module isn't.
    var a = Array[P, 4](fill=pv(0.0))
    a[1] = pv(1.0)
    a[2] = pv(1.0)

    assert_true(abs(s(det[P, 2](a)) - (-1.0)) > 0.5)
    # Finite rather than NaN, which is the part the pivot floor buys.
    assert_true(s(det[P, 2](a)) == s(det[P, 2](a)))


def test_log_det_from_cholesky_matches_the_determinant() raises:
    var a = spd3()
    var lower = cholesky[P, 3](a)
    assert_almost_equal(
        s(log_det_from_cholesky[P, 3](lower)),
        log_f64(s(det[P, 3](a))),
        atol=1e-13,
    )


def test_substitutions_invert_their_triangular_systems() raises:
    var lower = cholesky[P, 3](spd3())
    var b = rhs3()

    var y = forward_substitution[P, 3](lower, b)
    var recovered = matvec[P, 3](lower, y)
    for i in range(3):
        assert_almost_equal(s(recovered[i]), s(b[i]), atol=1e-12)

    var upper = transpose3(lower)
    var x = back_substitution[P, 3](upper, b)
    var recovered_upper = matvec[P, 3](upper, x)
    for i in range(3):
        assert_almost_equal(s(recovered_upper[i]), s(b[i]), atol=1e-12)


def test_unit_diagonal_substitution_ignores_the_stored_diagonal() raises:
    # The packed `lu` output stores `U`'s diagonal where `L`'s implicit 1
    # would be, so `unit_diagonal=True` must not read it.
    var packed = lu[P, 3](general3())
    var b = rhs3()
    var y = forward_substitution[P, 3, unit_diagonal=True](packed, b)
    # L @ y should be b, with L's diagonal taken as 1.
    for i in range(3):
        var total = 0.0
        for j in range(i):
            total += s(packed[i * 3 + j]) * s(y[j])
        total += s(y[i])
        assert_almost_equal(total, s(b[i]), atol=1e-12)


# ------------------------------------------------------------ tridiagonal


def test_tridiagonal_solve_matches_a_dense_solve() raises:
    comptime n = 5
    var sub = Array[P, n](fill=pv(0.0))
    var diag = Array[P, n](fill=pv(0.0))
    var sup = Array[P, n](fill=pv(0.0))
    var rhs = Array[P, n](fill=pv(0.0))

    for i in range(n):
        sub[i] = pv(-1.0)
        diag[i] = pv(2.5 + Float64(i) * 0.1)
        sup[i] = pv(-1.0)
        rhs[i] = pv(Float64(i) + 1.0)

    var thomas = tridiagonal_solve[P, n](sub, diag, sup, rhs)

    var dense = Array[P, n * n](fill=pv(0.0))
    for i in range(n):
        dense[i * n + i] = diag[i].copy()
        if i > 0:
            dense[i * n + i - 1] = sub[i].copy()
        if i < n - 1:
            dense[i * n + i + 1] = sup[i].copy()
    var reference = solve[P, n](dense, rhs)

    for i in range(n):
        assert_almost_equal(s(thomas[i]), s(reference[i]), atol=1e-12)


def test_tridiagonal_solve_on_the_spline_system() raises:
    # The constant 1-4-1 system a natural cubic spline produces, with a
    # right-hand side chosen so the answer is all ones.
    comptime n = 4
    var sub = Array[P, n](fill=pv(1.0))
    var diag = Array[P, n](fill=pv(4.0))
    var sup = Array[P, n](fill=pv(1.0))
    var rhs = Array[P, n](fill=pv(6.0))
    rhs[0] = pv(5.0)
    rhs[n - 1] = pv(5.0)

    var x = tridiagonal_solve[P, n](sub, diag, sup, rhs)
    for i in range(n):
        assert_almost_equal(s(x[i]), 1.0, atol=1e-13)


# --------------------------------------------------- the differentiable part


def test_determinant_is_differentiable() raises:
    # A(x) = [[x, 1], [1, x]] has det = x^2 - 1, so d/dx = 2x.
    var x = 3.0
    var a = Array[D, 4](fill=D.constant(0.0))
    a[0] = D(pv(x), pv(1.0))
    a[1] = D.constant(1.0)
    a[2] = D.constant(1.0)
    a[3] = D(pv(x), pv(1.0))

    var d = det[D, 2](a)
    assert_almost_equal(s(d.value), x * x - 1.0, atol=1e-13)
    assert_almost_equal(s(d.deriv), 2.0 * x, atol=1e-13)


def test_cholesky_log_det_is_differentiable() raises:
    # A(x) = [[x, 0.5], [0.5, 2]], SPD for x > 0.125.
    # det = 2x - 0.25, so d/dx[ln det] = 2/(2x - 0.25).
    var x = 1.7
    var a = Array[D, 4](fill=D.constant(0.0))
    a[0] = D(pv(x), pv(1.0))
    a[1] = D.constant(0.5)
    a[2] = D.constant(0.5)
    a[3] = D.constant(2.0)

    var lower = cholesky[D, 2](a)
    var log_det = log_det_from_cholesky[D, 2](lower)

    assert_almost_equal(s(log_det.value), log_f64(2.0 * x - 0.25), atol=1e-13)
    assert_almost_equal(s(log_det.deriv), 2.0 / (2.0 * x - 0.25), atol=1e-12)


def test_solve_is_differentiable() raises:
    # A(x) = [[x, 0], [0, 1]] with b = (1, 1) gives x_0 = 1/x, so
    # d(x_0)/dx = -1/x^2. Differentiating *through* a linear solve is what
    # an implicit layer needs.
    var t = 2.5
    var a = Array[D, 4](fill=D.constant(0.0))
    a[0] = D(pv(t), pv(1.0))
    a[3] = D.constant(1.0)
    var b = Array[D, 2](fill=D.constant(1.0))

    var solution = solve[D, 2](a, b)
    assert_almost_equal(s(solution[0].value), 1.0 / t, atol=1e-14)
    assert_almost_equal(s(solution[0].deriv), -1.0 / (t * t), atol=1e-13)


def test_compensated_cholesky_beats_plain_at_the_same_dtype() raises:
    # The 4x4 Hilbert matrix has a condition number around 1.5e4, enough
    # that float32 loses several digits reconstructing it from its own
    # factor -- and enough that `Compensated`'s extra precision shows up
    # without needing a contrived example.
    comptime n = 4
    comptime narrow = DType.float32
    comptime PN = Plain[narrow, 1]
    comptime CN = Compensated[narrow, 1]

    var plain_a = Array[PN, n * n](fill=PN.constant(0.0))
    var comp_a = Array[CN, n * n](fill=CN.constant(0.0))
    for i in range(n):
        for j in range(n):
            var entry = 1.0 / (Float64(i) + Float64(j) + 1.0)
            plain_a[i * n + j] = PN.constant(entry)
            comp_a[i * n + j] = CN.constant(entry)

    var plain_l = cholesky[PN, n](plain_a)
    var comp_l = cholesky[CN, n](comp_a)

    var plain_worst = 0.0
    var comp_worst = 0.0
    for i in range(n):
        for j in range(i + 1):
            var exact = 1.0 / (Float64(i) + Float64(j) + 1.0)
            # Reconstruct A[i,j] = sum_k L[i,k]*L[j,k].
            var plain_sum = 0.0
            var comp_sum = 0.0
            for k in range(j + 1):
                plain_sum += Float64(plain_l[i * n + k].v) * Float64(
                    plain_l[j * n + k].v
                )
                comp_sum += (
                    Float64(comp_l[i * n + k].value)
                    + Float64(comp_l[i * n + k].error)
                ) * (
                    Float64(comp_l[j * n + k].value)
                    + Float64(comp_l[j * n + k].error)
                )
            plain_worst = max(plain_worst, abs(plain_sum - exact))
            comp_worst = max(comp_worst, abs(comp_sum - exact))

    assert_true(comp_worst < plain_worst)


def test_simd_lanes_factor_independent_matrices() raises:
    comptime w = 2
    comptime PW = Plain[dtype, w]
    # Lane 0 gets 2*I, lane 1 gets 8*I, so the Cholesky diagonals should be
    # sqrt(2) and sqrt(8) respectively.
    var a = Array[PW, 4](fill=PW.constant(0.0))
    a[0] = PW(SIMD[dtype, w](2.0, 8.0))
    a[3] = PW(SIMD[dtype, w](2.0, 8.0))

    var lower = cholesky[PW, 2](a)
    assert_almost_equal(Float64(lower[0].v[0]), 1.4142135623730951, atol=1e-14)
    assert_almost_equal(Float64(lower[0].v[1]), 2.8284271247461903, atol=1e-14)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
