"""Tests for `numax.linalg`.

Structured around identities the factorizations must satisfy -- `L@L.T`
reproducing `A`, `A@x` reproducing `b`, `A@A^-1` reproducing the identity --
rather than against precomputed factor entries, since those only check one
matrix while the identity checks the algorithm.
"""

from std.collections import Array
from std.math import log as log_f64
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from numax import Compensated, Dual, FloatLike, Plain
from numax.linalg import (
    back_substitution,
    cholesky,
    cholesky_solve,
    det,
    forward_substitution,
    inverse,
    lstsq,
    slogdet_cholesky,
    lu,
    lu_factor,
    matmul,
    cond,
    dot,
    eigh,
    eigvals,
    matvec,
    inf,
    norm,
    nrm2,
    outer,
    pinv,
    qr,
    solve,
    svd,
    trace,
    tridiagonal_solve,
)

comptime dtype = DType.float64
comptime width = 1
comptime P = Plain[dtype, width]
comptime D = Dual[P]


def pv(x: Float64) -> P:
    return P.constant(x)


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


def test_slogdet_cholesky_matches_the_determinant() raises:
    var a = spd3()
    var lower = cholesky[P, 3](a)
    assert_almost_equal(
        s(slogdet_cholesky[P, 3](lower)),
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
    var log_det = slogdet_cholesky[D, 2](lower)

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
    comptime PN = Plain[narrow]
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


def test_trace_sums_the_diagonal() raises:
    var a = spd3()
    assert_almost_equal(s(trace[P, 3](a)), 4.0 + 5.0 + 6.0)


def test_frobenius_norm_matches_the_entrywise_definition() raises:
    var a = general3()
    var expected = 0.0
    var entries = [2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0]
    for i in range(9):
        expected += entries[i] * entries[i]
    assert_almost_equal(s(norm[P, 3](a)), expected**0.5)


def test_norm_1_is_the_largest_absolute_column_sum() raises:
    # Columns of general3: |2|+|-3|+|-2| = 7, |1|+|-1|+|1| = 3,
    # |-1|+|2|+|2| = 5. The largest is 7.
    var a = general3()
    assert_almost_equal(s(norm[P, 3, 1](a)), 7.0)


def test_norm_inf_is_the_largest_absolute_row_sum() raises:
    # Rows of general3: 4, 6, 5. The largest is 6.
    var a = general3()
    assert_almost_equal(s(norm[P, 3, inf](a)), 6.0)


def test_norm_inf_of_a_symmetric_matrix_equals_its_norm_1() raises:
    var a = spd3()
    assert_almost_equal(s(norm[P, 3, inf](a)), s(norm[P, 3, 1](a)))


def test_qr_reconstructs_the_original_matrix() raises:
    var a = general3()
    var factored = qr[P, 3](a)
    var recon = matmul[P, 3](factored[1].copy(), factored[0].copy())
    for i in range(9):
        assert_almost_equal(s(recon[i]), s(a[i]))


def test_qr_r_is_upper_triangular() raises:
    var a = general3()
    var factored = qr[P, 3](a)
    var r = factored[0].copy()
    for i in range(3):
        for j in range(i):
            assert_almost_equal(s(r[i * 3 + j]), 0.0)


def test_qr_q_is_orthogonal() raises:
    # Q.T @ Q = I is the property that makes QR useful; checking it catches
    # a reflector applied with the wrong sign or scale, which reconstructing
    # A alone would not.
    var a = general3()
    var factored = qr[P, 3](a)
    var q = factored[1].copy()
    var qt = Array[P, 9](fill=pv(0.0))
    for i in range(3):
        for j in range(3):
            qt[i * 3 + j] = q[j * 3 + i].copy()
    var product = matmul[P, 3](qt^, q^)
    for i in range(3):
        for j in range(3):
            var expected = 1.0 if i == j else 0.0
            assert_almost_equal(s(product[i * 3 + j]), expected)


def test_qr_of_a_symmetric_positive_definite_matrix_reconstructs_it() raises:
    var a = spd3()
    var factored = qr[P, 3](a)
    var recon = matmul[P, 3](factored[1].copy(), factored[0].copy())
    for i in range(9):
        assert_almost_equal(s(recon[i]), s(a[i]))


def test_qr_differentiates_through_at_dual() raises:
    # The axis-1 payoff, and the reason this QR exists next to MAX's
    # monomorphic `linalg.qr_factorization`: seed one entry of A with a
    # derivative and it propagates through the factorization with no
    # adjoint rule written anywhere. d(trace(R))/dA[0,0] is checked against
    # a central difference of the same function.
    def trace_r(a00: Float64) -> Float64:
        var a = general3()
        a[0] = pv(a00)
        var factored = qr[P, 3](a)
        var r = factored[0].copy()
        return s(trace[P, 3](r^))

    var a = general3()
    var ad = Array[D, 9](fill=D.constant(0.0))
    for i in range(9):
        ad[i] = D.constant(s(a[i]))
    ad[0] = D(pv(s(a[0])), P.one())

    var factored = qr[D, 3](ad)
    var derivative = Float64(trace[D, 3](factored[0].copy()).deriv.v)

    var h = 1e-6
    var numeric = (trace_r(s(a[0]) + h) - trace_r(s(a[0]) - h)) / (2 * h)
    assert_almost_equal(derivative, numeric, atol=1e-6)


# ------------------------------------------------------------------
# BLAS-1
# ------------------------------------------------------------------


def vec3(a: Float64, b: Float64, c: Float64) -> Array[P, 3]:
    var v = Array[P, 3](fill=pv(0.0))
    v[0] = pv(a)
    v[1] = pv(b)
    v[2] = pv(c)
    return v^


def test_dot_is_the_inner_product() raises:
    var a = vec3(1.0, 2.0, 3.0)
    var b = vec3(4.0, -5.0, 6.0)
    assert_almost_equal(s(dot[P, 3](a, b)), 4.0 - 10.0 + 18.0)


def test_nrm2_is_the_euclidean_length() raises:
    var a = vec3(3.0, 4.0, 0.0)
    assert_almost_equal(s(nrm2[P, 3](a)), 5.0)


def test_nrm2_squared_equals_dot_with_itself() raises:
    var a = vec3(1.5, -2.5, 0.75)
    var norm = s(nrm2[P, 3](a))
    assert_almost_equal(norm * norm, s(dot[P, 3](a, a)))


def test_outer_product_has_rank_one_structure() raises:
    var a = vec3(1.0, 2.0, 3.0)
    var b = vec3(4.0, 5.0, 6.0)
    var m = outer[P, 3](a, b)
    for i in range(3):
        for j in range(3):
            assert_almost_equal(s(m[i * 3 + j]), s(a[i]) * s(b[j]))


def test_compensated_dot_beats_plain_on_a_long_cancelling_sum() raises:
    # The reason a generic `dot` is worth having: the same call at
    # `Compensated` recovers bits `Plain` drops.
    comptime CP = Compensated[dtype, width]
    var big = 1e16
    var a_plain = Array[P, 4](fill=pv(0.0))
    a_plain[0] = pv(big)
    a_plain[1] = pv(1.0)
    a_plain[2] = pv(-big)
    a_plain[3] = pv(1.0)
    var ones_plain = Array[P, 4](fill=pv(1.0))

    var a_comp = Array[CP, 4](fill=CP.constant(0.0))
    a_comp[0] = CP.constant(big)
    a_comp[1] = CP.constant(1.0)
    a_comp[2] = CP.constant(-big)
    a_comp[3] = CP.constant(1.0)
    var ones_comp = Array[CP, 4](fill=CP.constant(1.0))

    var plain_result = s(dot[P, 4](a_plain, ones_plain))
    var comp = dot[CP, 4](a_comp, ones_comp)
    var comp_result = Float64(comp.value) + Float64(comp.error)

    # The true sum is 2. Plain loses the 1s inside the 1e16 cancellation.
    assert_true(abs(comp_result - 2.0) <= abs(plain_result - 2.0))


# ------------------------------------------------------------------
# eigh
# ------------------------------------------------------------------


def test_eigh_eigenvalues_sum_to_the_trace() raises:
    # An invariant of any similarity transform, so it catches a rotation
    # applied inconsistently to rows and columns.
    var a = spd3()
    var values = eigh[P, 3](a)[0].copy()
    var total = s(values[0]) + s(values[1]) + s(values[2])
    assert_almost_equal(total, s(trace[P, 3](a)), atol=1e-12)


def test_eigh_satisfies_the_eigenvalue_equation() raises:
    # A @ v == lambda * v for every returned pair. The actual definition,
    # rather than a check against precomputed values.
    var a = spd3()
    var factored = eigh[P, 3](a)
    var values = factored[0].copy()
    var vectors = factored[1].copy()
    for j in range(3):
        for i in range(3):
            var av = 0.0
            for k in range(3):
                av += s(a[i * 3 + k]) * s(vectors[k * 3 + j])
            assert_almost_equal(
                av, s(values[j]) * s(vectors[i * 3 + j]), atol=1e-12
            )


def test_eigh_eigenvectors_are_orthonormal() raises:
    var a = spd3()
    var vectors = eigh[P, 3](a)[1].copy()
    for p in range(3):
        for q in range(3):
            var inner = 0.0
            for i in range(3):
                inner += s(vectors[i * 3 + p]) * s(vectors[i * 3 + q])
            var expected = 1.0 if p == q else 0.0
            assert_almost_equal(inner, expected, atol=1e-12)


def test_eigh_of_a_diagonal_matrix_returns_its_diagonal() raises:
    var a = Array[P, 9](fill=pv(0.0))
    a[0] = pv(3.0)
    a[4] = pv(-1.0)
    a[8] = pv(7.0)
    var values = eigh[P, 3](a)[0].copy()
    # Unordered, so check the multiset by summing and by extremes.
    var total = s(values[0]) + s(values[1]) + s(values[2])
    assert_almost_equal(total, 9.0, atol=1e-12)
    var biggest = max(max(s(values[0]), s(values[1])), s(values[2]))
    var smallest = min(min(s(values[0]), s(values[1])), s(values[2]))
    assert_almost_equal(biggest, 7.0, atol=1e-12)
    assert_almost_equal(smallest, -1.0, atol=1e-12)


def test_eigh_of_the_identity_is_all_ones() raises:
    var a = Array[P, 9](fill=pv(0.0))
    for i in range(3):
        a[i * 3 + i] = pv(1.0)
    var values = eigh[P, 3](a)[0].copy()
    for i in range(3):
        assert_almost_equal(s(values[i]), 1.0, atol=1e-14)


def test_eigh_positive_definite_matrix_has_positive_eigenvalues() raises:
    var values = eigh[P, 3](spd3())[0].copy()
    for i in range(3):
        assert_true(s(values[i]) > 0.0)


# ------------------------------------------------------------------
# svd
# ------------------------------------------------------------------


def test_svd_reconstructs_the_matrix() raises:
    var a = general3()
    var factored = svd[P, 3](a)
    var u = factored[0].copy()
    var values = factored[1].copy()
    var v = factored[2].copy()
    for i in range(3):
        for j in range(3):
            var total = 0.0
            for k in range(3):
                total += s(u[i * 3 + k]) * s(values[k]) * s(v[j * 3 + k])
            assert_almost_equal(total, s(a[i * 3 + j]), atol=1e-12)


def test_svd_singular_values_are_non_negative() raises:
    var values = svd[P, 3](general3())[1].copy()
    for i in range(3):
        assert_true(s(values[i]) >= 0.0)


def test_svd_left_factor_is_orthonormal() raises:
    var u = svd[P, 3](general3())[0].copy()
    for p in range(3):
        for q in range(3):
            var inner = 0.0
            for i in range(3):
                inner += s(u[i * 3 + p]) * s(u[i * 3 + q])
            var expected = 1.0 if p == q else 0.0
            assert_almost_equal(inner, expected, atol=1e-12)


def test_svd_right_factor_is_orthonormal() raises:
    var v = svd[P, 3](general3())[2].copy()
    for p in range(3):
        for q in range(3):
            var inner = 0.0
            for i in range(3):
                inner += s(v[i * 3 + p]) * s(v[i * 3 + q])
            var expected = 1.0 if p == q else 0.0
            assert_almost_equal(inner, expected, atol=1e-12)


def test_svd_of_a_symmetric_matrix_matches_its_eigenvalue_magnitudes() raises:
    # For a symmetric positive definite matrix the singular values *are*
    # the eigenvalues, which cross-checks two independently written
    # Jacobi loops against each other.
    var a = spd3()
    var eigenvalues = eigh[P, 3](a)[0].copy()
    var singular = svd[P, 3](a)[1].copy()

    var eig_sum = 0.0
    var sv_sum = 0.0
    for i in range(3):
        eig_sum += s(eigenvalues[i])
        sv_sum += s(singular[i])
    assert_almost_equal(eig_sum, sv_sum, atol=1e-10)


def test_svd_of_a_singular_matrix_reports_a_zero_singular_value() raises:
    # Row 2 = 2 * row 0, so the matrix is rank 2.
    var a = Array[P, 9](fill=pv(0.0))
    var entries = [1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 2.0, 4.0, 6.0]
    for i in range(9):
        a[i] = pv(entries[i])
    var values = svd[P, 3](a)[1].copy()
    var smallest = min(min(s(values[0]), s(values[1])), s(values[2]))
    assert_true(smallest < 1e-10)


# ------------------------------------------------------------------
# pinv and cond
# ------------------------------------------------------------------


def test_pinv_of_an_invertible_matrix_is_its_inverse() raises:
    var a = general3()
    var pseudo = pinv[P, 3](a)
    var product = matmul[P, 3](a.copy(), pseudo^)
    for i in range(3):
        for j in range(3):
            var expected = 1.0 if i == j else 0.0
            assert_almost_equal(s(product[i * 3 + j]), expected, atol=1e-10)


def test_pinv_agrees_with_inverse_on_a_well_conditioned_matrix() raises:
    var a = general3()
    var direct = inverse[P, 3](a)
    var pseudo = pinv[P, 3](a)
    for i in range(9):
        assert_almost_equal(s(pseudo[i]), s(direct[i]), atol=1e-10)


def test_pinv_of_a_singular_matrix_stays_finite() raises:
    # `inverse` would divide by a zero pivot here. The pseudoinverse
    # truncates the singular direction instead, which is the whole reason
    # to reach for it.
    var a = Array[P, 9](fill=pv(0.0))
    var entries = [1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 2.0, 4.0, 6.0]
    for i in range(9):
        a[i] = pv(entries[i])
    var pseudo = pinv[P, 3](a)
    for i in range(9):
        assert_true(abs(s(pseudo[i])) < 1e6)


def test_pinv_satisfies_the_moore_penrose_identity() raises:
    # A @ A+ @ A == A holds even when A is singular, which is what makes
    # the pseudoinverse well-defined there.
    var a = Array[P, 9](fill=pv(0.0))
    var entries = [1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 2.0, 4.0, 6.0]
    for i in range(9):
        a[i] = pv(entries[i])
    var pseudo = pinv[P, 3](a)
    var left = matmul[P, 3](a.copy(), pseudo^)
    var recovered = matmul[P, 3](left^, a.copy())
    for i in range(9):
        assert_almost_equal(s(recovered[i]), s(a[i]), atol=1e-9)


def test_cond_of_the_identity_is_one() raises:
    var a = Array[P, 9](fill=pv(0.0))
    for i in range(3):
        a[i * 3 + i] = pv(1.0)
    assert_almost_equal(s(cond[P, 3](a)), 1.0, atol=1e-12)


def test_cond_is_the_ratio_of_extreme_singular_values() raises:
    var a = spd3()
    var values = svd[P, 3](a)[1].copy()
    var biggest = max(max(s(values[0]), s(values[1])), s(values[2]))
    var smallest = min(min(s(values[0]), s(values[1])), s(values[2]))
    assert_almost_equal(s(cond[P, 3](a)), biggest / smallest, atol=1e-10)


def test_cond_of_a_scaled_identity_is_still_one() raises:
    # Scaling every singular value equally cannot change the ratio, which
    # catches an absolute rather than relative comparison in `cond`.
    var a = Array[P, 9](fill=pv(0.0))
    for i in range(3):
        a[i * 3 + i] = pv(1e6)
    assert_almost_equal(s(cond[P, 3](a)), 1.0, atol=1e-10)


def test_eigh_is_differentiable_at_dual() raises:
    # d(trace)/dA[0,0] == 1, and the eigenvalues sum to the trace, so the
    # summed derivative of the eigenvalues must be 1 too -- reached through
    # 12 sweeps of Jacobi rotations with no adjoint rule anywhere.
    var a = Array[D, 9](fill=D.constant(0.0))
    var entries = [4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0]
    for i in range(9):
        a[i] = D.constant(entries[i])
    a[0] = D(pv(4.0), P.one())

    var values = eigh[D, 3](a)[0].copy()
    var derivative_sum = 0.0
    for i in range(3):
        derivative_sum += Float64(values[i].deriv.v)
    assert_almost_equal(derivative_sum, 1.0, atol=1e-9)


def _fit_matrix() -> Array[P, 8]:
    """A 4x2 design matrix for `y = c0 + c1*t` at `t = 0, 1, 2, 3`."""
    var a = Array[P, 8](fill=P.constant(0.0))
    for i in range(4):
        a[i * 2] = P.constant(1.0)
        a[i * 2 + 1] = P.constant(Float64(i))
    return a^


def test_lstsq_recovers_an_exactly_fitting_line() raises:
    # Points on y = 2 + 3t exactly, so the least-squares fit is the line.
    var a = _fit_matrix()
    var b = Array[P, 4](fill=P.constant(0.0))
    for i in range(4):
        b[i] = P.constant(2.0 + 3.0 * Float64(i))

    var x = lstsq[P, 4, 2](a, b)
    assert_almost_equal(Float64(x[0].v), 2.0, atol=1e-10)
    assert_almost_equal(Float64(x[1].v), 3.0, atol=1e-10)


def test_lstsq_residual_is_orthogonal_to_the_columns() raises:
    # The defining property of a least-squares solution: A.T @ (A x - b)
    # is zero, whether or not the fit is exact. These points are not
    # collinear, so the fit is a genuine compromise.
    var a = _fit_matrix()
    var b = Array[P, 4](fill=P.constant(0.0))
    var noisy = [1.0, 4.0, 5.0, 11.0]
    for i in range(4):
        b[i] = P.constant(noisy[i])

    var x = lstsq[P, 4, 2](a, b)
    for j in range(2):
        var projection = 0.0
        for i in range(4):
            var predicted = Float64(a[i * 2].v) * Float64(x[0].v) + Float64(
                a[i * 2 + 1].v
            ) * Float64(x[1].v)
            projection += Float64(a[i * 2 + j].v) * (
                predicted - Float64(b[i].v)
            )
        assert_almost_equal(projection, 0.0, atol=1e-10)


def test_lstsq_at_a_square_system_is_the_solve() raises:
    var a = Array[P, 4](fill=P.constant(0.0))
    a[0] = P.constant(4.0)
    a[1] = P.constant(1.0)
    a[2] = P.constant(1.0)
    a[3] = P.constant(3.0)
    var b = Array[P, 2](fill=P.constant(0.0))
    b[0] = P.constant(9.0)
    b[1] = P.constant(11.0)

    var by_lstsq = lstsq[P, 2, 2](a, b)
    var by_solve = solve[P, 2](a, b)
    for i in range(2):
        assert_almost_equal(
            Float64(by_lstsq[i].v), Float64(by_solve[i].v), atol=1e-10
        )


def test_lstsq_differentiates_through_the_fit() raises:
    # d/db0 of the fitted intercept, for the 4-point evenly spaced design.
    # The hat matrix's first row gives the closed form: the intercept is a
    # fixed linear combination of the observations, so its derivative with
    # respect to b[0] is that combination's first entry, 0.7.
    var a = Array[D, 8](fill=D.constant(0.0))
    for i in range(4):
        a[i * 2] = D.constant(1.0)
        a[i * 2 + 1] = D.constant(Float64(i))
    var b = Array[D, 4](fill=D.constant(0.0))
    for i in range(4):
        b[i] = D.constant(Float64(i) + 1.0)
    b[0] = D(P.constant(1.0), P.one())

    var x = lstsq[D, 4, 2](a, b)
    assert_almost_equal(Float64(x[0].deriv.v), 0.7, atol=1e-10)


def exchange2() -> Array[P, 4]:
    """`[[0, 1], [1, 0]]`: well conditioned, determinant -1, and the
    standard matrix the unpivoted LU cannot start on."""
    var a = Array[P, 4](fill=pv(0.0))
    a[1] = pv(1.0)
    a[2] = pv(1.0)
    return a^


def test_pivoting_factors_the_matrix_the_unpivoted_lu_cannot_start_on() raises:
    var a = exchange2()
    var factored = lu_factor[dtype, 2](a)

    assert_equal(factored.permutation[0], 1)
    assert_equal(factored.permutation[1], 0)
    assert_equal(factored.sign, -1)
    assert_almost_equal(s(factored.det()), -1.0, atol=1e-12)

    # What the tier-1 route returns on the same matrix, pinned so the
    # difference the pivoting buys is visible rather than asserted.
    assert_almost_equal(s(det[P, 2](a)), 0.0, atol=1e-12)


def test_a_pivoted_solve_reproduces_its_right_hand_side() raises:
    var a = exchange2()
    var b = Array[P, 2](fill=pv(0.0))
    b[0] = pv(3.0)
    b[1] = pv(7.0)

    var x = lu_factor[dtype, 2](a).solve(b)

    # A x = b with A the exchange matrix means x is b reversed.
    assert_almost_equal(s(x[0]), 7.0, atol=1e-12)
    assert_almost_equal(s(x[1]), 3.0, atol=1e-12)


def test_one_factorization_serves_several_right_hand_sides() raises:
    var a = general3()
    var factored = lu_factor[dtype, 3](a)

    var first = factored.solve(rhs3())
    var residual = matvec[P, 3](a, first)
    for i in range(3):
        assert_almost_equal(s(residual[i]), s(rhs3()[i]), atol=1e-10)

    var b = Array[P, 3](fill=pv(0.0))
    b[0] = pv(1.0)
    b[1] = pv(2.0)
    b[2] = pv(3.0)
    var second = factored.solve(b)
    var second_residual = matvec[P, 3](a, second)
    for i in range(3):
        assert_almost_equal(s(second_residual[i]), s(b[i]), atol=1e-10)


def test_the_permuted_matrix_is_the_product_of_its_factors() raises:
    var a = general3()
    var factored = lu_factor[dtype, 3](a)

    var lower = Array[P, 9](fill=pv(0.0))
    var upper = Array[P, 9](fill=pv(0.0))
    for i in range(3):
        lower[i * 3 + i] = pv(1.0)
        for j in range(3):
            if j < i:
                lower[i * 3 + j] = factored.factored[i * 3 + j].copy()
            else:
                upper[i * 3 + j] = factored.factored[i * 3 + j].copy()

    var product = matmul[P, 3](lower, upper)
    for i in range(3):
        for j in range(3):
            assert_almost_equal(
                s(product[i * 3 + j]),
                s(a[factored.permutation[i] * 3 + j]),
                atol=1e-10,
            )


def test_pivoting_beats_the_unpivoted_route_on_a_small_pivot() raises:
    # The textbook near-singular-pivot case: solving it without pivoting
    # subtracts a huge multiple of row 0 from row 1 and loses the original
    # entries to rounding.
    var a = Array[P, 4](fill=pv(0.0))
    a[0] = pv(1e-18)
    a[1] = pv(1.0)
    a[2] = pv(1.0)
    a[3] = pv(1.0)
    var b = Array[P, 2](fill=pv(0.0))
    b[0] = pv(1.0)
    b[1] = pv(2.0)

    var pivoted = lu_factor[dtype, 2](a).solve(b)
    var unpivoted = solve[P, 2](a, b)

    # x is approximately (1, 1) here.
    var pivoted_error = abs(s(pivoted[0]) - 1.0) + abs(s(pivoted[1]) - 1.0)
    var unpivoted_error = abs(s(unpivoted[0]) - 1.0) + abs(
        s(unpivoted[1]) - 1.0
    )
    assert_true(pivoted_error < 1e-12)
    assert_true(unpivoted_error > pivoted_error)


def test_eigvals_of_a_triangular_matrix_are_its_diagonal() raises:
    var a = Array[P, 9](fill=pv(0.0))
    var entries = [3.0, 1.0, 2.0, 0.0, 5.0, 4.0, 0.0, 0.0, -2.0]
    for i in range(9):
        a[i] = pv(entries[i])

    var values = eigvals[P, 3](a)
    var found = List[Float64]()
    for i in range(3):
        assert_almost_equal(s(values[i].im), 0.0, atol=1e-8)
        found.append(s(values[i].re))

    for expected in [3.0, 5.0, -2.0]:
        var matched = False
        for got in found:
            if abs(got - expected) < 1e-8:
                matched = True
        assert_true(matched)


def test_eigvals_of_a_quarter_turn_are_the_imaginary_units() raises:
    # [[0, -1], [1, 0]] rotates by 90 degrees, so it has no real eigenvalue
    # and no real matrix to round to -- the case `eigh` cannot express.
    var a = Array[P, 4](fill=pv(0.0))
    a[1] = pv(-1.0)
    a[2] = pv(1.0)

    var values = eigvals[P, 2](a)
    for i in range(2):
        assert_almost_equal(s(values[i].re), 0.0, atol=1e-10)
        assert_almost_equal(abs(s(values[i].im)), 1.0, atol=1e-10)
    assert_almost_equal(s(values[0].im) + s(values[1].im), 0.0, atol=1e-10)


def test_eigvals_sum_to_the_trace_and_multiply_to_the_determinant() raises:
    # The two identities that hold whatever the spectrum turns out to be,
    # which is what makes them the check when convergence is in question.
    var a = general3()
    var values = eigvals[P, 3](a)

    var sum_re = 0.0
    var sum_im = 0.0
    var product_re = 1.0
    var product_im = 0.0
    for i in range(3):
        sum_re += s(values[i].re)
        sum_im += s(values[i].im)
        var re = product_re * s(values[i].re) - product_im * s(values[i].im)
        var im = product_re * s(values[i].im) + product_im * s(values[i].re)
        product_re = re
        product_im = im

    assert_almost_equal(sum_re, s(trace[P, 3](a)), atol=1e-9)
    assert_almost_equal(sum_im, 0.0, atol=1e-9)
    assert_almost_equal(product_re, s(det[P, 3](a)), atol=1e-8)
    assert_almost_equal(product_im, 0.0, atol=1e-9)


def test_eigvals_agrees_with_eigh_on_a_symmetric_matrix() raises:
    var a = spd3()
    var expected = eigh[P, 3](a)[0].copy()
    var values = eigvals[P, 3](a)

    for i in range(3):
        assert_almost_equal(s(values[i].im), 0.0, atol=1e-8)
        var matched = False
        for j in range(3):
            if abs(s(values[i].re) - s(expected[j])) < 1e-7:
                matched = True
        assert_true(matched)


def test_eigvals_recovers_a_mixed_real_and_complex_spectrum() raises:
    # Block diagonal: a 1x1 block at 2, and a rotation-and-scale block whose
    # eigenvalues are 1 +/- 3i.
    var a = Array[P, 9](fill=pv(0.0))
    a[0] = pv(2.0)
    a[4] = pv(1.0)
    a[5] = pv(-3.0)
    a[7] = pv(3.0)
    a[8] = pv(1.0)

    var values = eigvals[P, 3](a)
    var real_count = 0
    var complex_count = 0
    for i in range(3):
        if abs(s(values[i].im)) < 1e-8:
            assert_almost_equal(s(values[i].re), 2.0, atol=1e-8)
            real_count += 1
        else:
            assert_almost_equal(s(values[i].re), 1.0, atol=1e-8)
            assert_almost_equal(abs(s(values[i].im)), 3.0, atol=1e-8)
            complex_count += 1
    assert_equal(real_count, 1)
    assert_equal(complex_count, 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
