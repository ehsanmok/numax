"""Tests for `numax.linalg.eigen`, the spectral factorizations over `Tensor`.

The reductions are checked by the two properties that define them rather
than against a table of numbers: `Q^T A Q` has the promised shape
(tridiagonal, Hessenberg, quasi-triangular), and it is *similar* to `A`,
so `Q T Q^T` reconstructs the matrix it came from. A wrong reflector fails
the first; a reflector applied on one side only fails the second while
passing the first, which is why both are here. The general spectrum is
also checked against `scipy.linalg.eigvals` on a matrix with two complex
pairs.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.array import Static, to_array, transpose, zeros
from numax.core.plain import Plain
from numax.linalg import (
    Eigenvalues,
    cond,
    eigh,
    eigvals,
    eigvalsh,
    gebrd,
    hessenberg,
    lstsq,
    matmul,
    matrix_rank,
    matvec,
    pinv,
    schur,
    svd,
    svdvals,
    sytrd,
)
from numax.linalg.array import eigvalsh as array_eigvalsh
from numax.linalg.array import svdvals as array_svdvals

comptime dtype = DType.float64
comptime P = Plain[dtype, 1]


def _hilbert[n: Int]() raises -> Static[dtype, n, n]:
    """A symmetric matrix with distinct eigenvalues spanning many orders of
    magnitude -- the reduction has to keep every one of them."""
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, n, n](ctx)
    var host = a.to_host()
    for i in range(n):
        for j in range(n):
            host[i * n + j] = Scalar[dtype](1.0 / Float64(i + j + 1))
    a.copy_from_host(host)
    return a^


def _copy_of[n: Int](mut a: Static[dtype, n, n]) raises -> Static[dtype, n, n]:
    var ctx = a.context()
    var out = zeros[dtype, n, n](ctx)
    out.copy_from_host(a.to_host())
    return out^


def test_sytrd_leaves_q_transpose_a_q_tridiagonal() raises:
    comptime n = 5
    var a = _hilbert[n]()
    var original = _copy_of(a)

    var reduced = sytrd(a)
    var q = reduced.q()
    var qt = transpose(q)
    var half = matmul(qt, original)
    var t = matmul(half, q).to_host()

    for i in range(n):
        for j in range(n):
            if j > i + 1 or j + 1 < i:
                assert_almost_equal(
                    t[i * n + j], Scalar[dtype](0.0), atol=1e-12
                )


def test_sytrd_reports_the_band_it_produced() raises:
    comptime n = 5
    var a = _hilbert[n]()
    var original = _copy_of(a)

    var reduced = sytrd(a)
    var q = reduced.q()
    var qt = transpose(q)
    var half = matmul(qt, original)
    var t = matmul(half, q).to_host()

    var d = reduced.d.to_host()
    var e = reduced.e.to_host()
    for i in range(n):
        assert_almost_equal(d[i], t[i * n + i], atol=1e-12)
    for i in range(n - 1):
        assert_almost_equal(e[i], t[(i + 1) * n + i], atol=1e-12)


def test_sytrd_is_a_similarity_so_q_t_q_transpose_reconstructs() raises:
    """`Q^T A Q` being tridiagonal is not enough on its own -- a reflector
    applied on one side only can still produce a band. Reconstructing `A`
    is what says the transformation was orthogonal *and* two-sided."""
    comptime n = 6
    var a = _hilbert[n]()
    var original = _copy_of(a)

    var reduced = sytrd(a)
    var q = reduced.q()
    var d = reduced.d.to_host()
    var e = reduced.e.to_host()

    var ctx = a.context()
    var t = zeros[dtype, n, n](ctx)
    var band = t.to_host()
    for i in range(n):
        band[i * n + i] = d[i]
    for i in range(n - 1):
        band[(i + 1) * n + i] = e[i]
        band[i * n + (i + 1)] = e[i]
    t.copy_from_host(band)

    var qt = transpose(q)
    var half = matmul(q, t)
    var back = matmul(half, qt).to_host()

    var source = original.to_host()
    for i in range(n * n):
        assert_almost_equal(back[i], source[i], atol=1e-12)


def test_sytrd_q_is_orthogonal() raises:
    comptime n = 6
    var a = _hilbert[n]()
    var reduced = sytrd(a)
    var q = reduced.q()
    var qt = transpose(q)
    var identity = matmul(qt, q).to_host()

    for i in range(n):
        for j in range(n):
            var want = Scalar[dtype](1.0) if i == j else Scalar[dtype](0.0)
            assert_almost_equal(identity[i * n + j], want, atol=1e-12)


def test_sytrd_preserves_the_trace() raises:
    """The cheapest similarity invariant, and the one that fails loudly if
    a reflector is scaled wrongly."""
    comptime n = 5
    var a = _hilbert[n]()
    var original = _copy_of(a)

    var reduced = sytrd(a)
    var d = reduced.d.to_host()
    var source = original.to_host()

    var reduced_trace = Scalar[dtype](0)
    var source_trace = Scalar[dtype](0)
    for i in range(n):
        reduced_trace += d[i]
        source_trace += source[i * n + i]
    assert_almost_equal(reduced_trace, source_trace, atol=1e-12)


def test_sytrd_leaves_an_already_tridiagonal_matrix_alone() raises:
    """Every column is already reduced, so every `tau` is zero and the
    identity-reflector branch runs for all of them."""
    comptime n = 5
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, n, n](ctx)
    var host = a.to_host()
    for i in range(n):
        host[i * n + i] = Scalar[dtype](i + 1)
        if i + 1 < n:
            host[(i + 1) * n + i] = Scalar[dtype](0.5)
            host[i * n + (i + 1)] = Scalar[dtype](0.5)
    a.copy_from_host(host)

    var reduced = sytrd(a)
    var d = reduced.d.to_host()
    var e = reduced.e.to_host()
    var taus = reduced.taus.to_host()

    for i in range(n):
        assert_almost_equal(d[i], Scalar[dtype](i + 1))
    for i in range(n - 1):
        assert_almost_equal(e[i], Scalar[dtype](0.5))
    for i in range(n - 2):
        assert_equal(taus[i], Scalar[dtype](0.0))


def test_sytrd_at_the_sizes_with_nothing_to_do() raises:
    """`n < 3` has no reflector to form at all; the loop body never runs
    and `q()` returns the identity."""
    var ctx = DeviceContext(api="cpu")

    var one = zeros[dtype, 1, 1](ctx)
    one[0] = 7
    var r1 = sytrd(one)
    assert_almost_equal(r1.d.to_host()[0], Scalar[dtype](7.0))
    assert_almost_equal(r1.q().to_host()[0], Scalar[dtype](1.0))

    var two = zeros[dtype, 2, 2](ctx)
    var host = two.to_host()
    host[0] = 3
    host[1] = 2
    host[2] = 2
    host[3] = 5
    two.copy_from_host(host)
    var r2 = sytrd(two)
    assert_almost_equal(r2.d.to_host()[0], Scalar[dtype](3.0))
    assert_almost_equal(r2.d.to_host()[1], Scalar[dtype](5.0))
    assert_almost_equal(r2.e.to_host()[0], Scalar[dtype](2.0))


def test_eigvalsh_of_a_diagonal_matrix_is_its_diagonal_ascending() raises:
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, 4, 4](ctx)
    var host = a.to_host()
    host[0] = 3
    host[5] = -1
    host[10] = 7
    host[15] = 2
    a.copy_from_host(host)

    var values = eigvalsh(a).to_host()
    assert_almost_equal(values[0], Scalar[dtype](-1.0), atol=1e-14)
    assert_almost_equal(values[1], Scalar[dtype](2.0), atol=1e-14)
    assert_almost_equal(values[2], Scalar[dtype](3.0), atol=1e-14)
    assert_almost_equal(values[3], Scalar[dtype](7.0), atol=1e-14)


def test_eigvalsh_matches_the_known_spectrum_of_hilbert_four() raises:
    # scipy.linalg.eigvalsh(hilbert(4)), to the digits float64 keeps.
    var a = _hilbert[4]()
    var values = eigvalsh(a).to_host()
    var expected = [
        9.670230402258689e-05,
        6.738273605760965e-03,
        1.691412202214450e-01,
        1.500214280059243e00,
    ]
    for i in range(4):
        assert_almost_equal(
            values[i], Scalar[dtype](expected[i]), rtol=1e-9, atol=1e-13
        )


def test_eigvalsh_agrees_with_the_array_tier_as_a_multiset() raises:
    """The two tiers are different algorithms -- QL after a Householder
    reduction here, cyclic Jacobi there -- so agreement is the check that
    both are right. The `Array` tier returns unsorted, so compare sorted."""
    comptime n = 5
    var a = _hilbert[n]()
    var lifted = to_array[P](a)

    var here = eigvalsh(a).to_host()
    var there = array_eigvalsh[P, n, sweeps=20](lifted)
    var there_sorted = List[Float64](capacity=n)
    for i in range(n):
        there_sorted.append(Float64(there[i].v[0]))
    # Selection sort; five entries.
    for i in range(n):
        for j in range(i + 1, n):
            if there_sorted[j] < there_sorted[i]:
                var tmp = there_sorted[i]
                there_sorted[i] = there_sorted[j]
                there_sorted[j] = tmp
    for i in range(n):
        assert_almost_equal(
            Float64(here[i]), there_sorted[i], rtol=1e-8, atol=1e-13
        )


def test_eigvalsh_sums_to_the_trace_and_multiplies_to_the_determinant() raises:
    # Two similarity invariants on a matrix with a known determinant: the
    # 3x3 [[2,1,0],[1,3,1],[0,1,4]] has trace 9 and det 2*11 - 1*4 = 18.
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, 3, 3](ctx)
    var host = a.to_host()
    var entries = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0]
    for i in range(9):
        host[i] = Scalar[dtype](entries[i])
    a.copy_from_host(host)

    var values = eigvalsh(a).to_host()
    var total = values[0] + values[1] + values[2]
    var product = values[0] * values[1] * values[2]
    assert_almost_equal(total, Scalar[dtype](9.0), atol=1e-12)
    assert_almost_equal(product, Scalar[dtype](18.0), atol=1e-11)
    # And ascending.
    assert_equal(values[0] <= values[1], True)
    assert_equal(values[1] <= values[2], True)


def test_eigvalsh_handles_a_matrix_with_repeated_eigenvalues() raises:
    # A rank-one perturbation of the identity: 1 (twice) and 1 + 3 = 4.
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, 3, 3](ctx)
    var host = a.to_host()
    for i in range(3):
        for j in range(3):
            host[i * 3 + j] = Scalar[dtype](1.0 if i == j else 0.0) + Scalar[
                dtype
            ](1.0)
    a.copy_from_host(host)

    var values = eigvalsh(a).to_host()
    assert_almost_equal(values[0], Scalar[dtype](1.0), atol=1e-12)
    assert_almost_equal(values[1], Scalar[dtype](1.0), atol=1e-12)
    assert_almost_equal(values[2], Scalar[dtype](4.0), atol=1e-12)


def test_eigh_satisfies_the_eigenvalue_equation_for_every_pair() raises:
    # A @ v_j == w_j * v_j: the definition, not a table.
    comptime n = 5
    var a = _hilbert[n]()
    var original = _copy_of(a)

    var result = eigh(a)
    var av = matmul(original, result.vectors).to_host()
    var v = result.vectors.to_host()
    var w = result.values.to_host()
    for j in range(n):
        for i in range(n):
            assert_almost_equal(av[i * n + j], w[j] * v[i * n + j], atol=1e-12)


def test_eigh_eigenvectors_are_orthonormal() raises:
    comptime n = 5
    var a = _hilbert[n]()
    var result = eigh(a)
    var vt = transpose(result.vectors)
    var gram = matmul(vt, result.vectors).to_host()
    for i in range(n):
        for j in range(n):
            var want = Scalar[dtype](1.0) if i == j else Scalar[dtype](0.0)
            assert_almost_equal(gram[i * n + j], want, atol=1e-12)


def test_eigh_reconstructs_a_from_its_factors() raises:
    # A == V diag(w) V^T, which is what "eigendecomposition" means.
    comptime n = 6
    var a = _hilbert[n]()
    var original = _copy_of(a)

    var result = eigh(a)
    var ctx = a.context()
    var scaled = zeros[dtype, n, n](ctx)
    var v = result.vectors.to_host()
    var w = result.values.to_host()
    var host = scaled.to_host()
    for i in range(n):
        for j in range(n):
            host[i * n + j] = v[i * n + j] * w[j]
    scaled.copy_from_host(host)

    var vt = transpose(result.vectors)
    var back = matmul(scaled, vt).to_host()
    var source = original.to_host()
    for i in range(n * n):
        assert_almost_equal(back[i], source[i], atol=1e-12)


def test_eigh_values_are_ascending_and_equal_eigvalsh() raises:
    # The two names must agree exactly: eigvalsh is eigh without the
    # vectors, and a divergence would mean two spectra for one matrix.
    comptime n = 5
    var a = _hilbert[n]()
    var b = _copy_of(a)

    var w = eigh(a).values.to_host()
    var w_only = eigvalsh(b).to_host()
    for i in range(n):
        assert_almost_equal(w[i], w_only[i], atol=1e-13)
    for i in range(n - 1):
        assert_equal(w[i] <= w[i + 1], True)


def test_eigh_of_a_diagonal_matrix_returns_permuted_identity_vectors() raises:
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, 3, 3](ctx)
    var host = a.to_host()
    host[0] = 3
    host[4] = -1
    host[8] = 7
    a.copy_from_host(host)

    var result = eigh(a)
    var w = result.values.to_host()
    assert_almost_equal(w[0], Scalar[dtype](-1.0), atol=1e-14)
    assert_almost_equal(w[1], Scalar[dtype](3.0), atol=1e-14)
    assert_almost_equal(w[2], Scalar[dtype](7.0), atol=1e-14)
    # Column 0 pairs with -1, which lived at index 1: it is +-e_1.
    var v = result.vectors.to_host()
    assert_almost_equal(abs(v[1 * 3 + 0]), Scalar[dtype](1.0), atol=1e-14)
    assert_almost_equal(abs(v[0 * 3 + 1]), Scalar[dtype](1.0), atol=1e-14)
    assert_almost_equal(abs(v[2 * 3 + 2]), Scalar[dtype](1.0), atol=1e-14)


def _tall() raises -> Static[dtype, 5, 3]:
    var ctx = DeviceContext(api="cpu")
    return Static[dtype, 5, 3](
        ctx,
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            10.0,
            2.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    )


def _copy_rect[
    m: Int, n: Int
](mut a: Static[dtype, m, n]) raises -> Static[dtype, m, n]:
    var out = zeros[dtype, m, n](a.context())
    out.copy_from_host(a.to_host())
    return out^


def test_gebrd_leaves_q_transpose_a_p_upper_bidiagonal() raises:
    comptime m = 5
    comptime n = 3
    var a = _tall()
    var original = _copy_rect(a)

    var reduced = gebrd(a)
    var q = reduced.q()
    var p = reduced.p()
    var qt = transpose(q)
    var half = matmul(qt, original)
    var b = matmul(half, p).to_host()

    var d = reduced.d.to_host()
    var e = reduced.e.to_host()
    for i in range(n):
        for j in range(n):
            if i == j:
                assert_almost_equal(b[i * n + j], d[i], atol=1e-12)
            elif j == i + 1:
                assert_almost_equal(b[i * n + j], e[i], atol=1e-12)
            else:
                assert_almost_equal(
                    b[i * n + j], Scalar[dtype](0.0), atol=1e-12
                )


def test_gebrd_q_and_p_are_orthogonal() raises:
    var a = _tall()
    var reduced = gebrd(a)
    var q = reduced.q()
    var p = reduced.p()
    var qt = transpose(q)
    var pt = transpose(p)
    var qtq = matmul(qt, q).to_host()
    var ptp = matmul(pt, p).to_host()
    for i in range(3):
        for j in range(3):
            var want = Scalar[dtype](1.0) if i == j else Scalar[dtype](0.0)
            assert_almost_equal(qtq[i * 3 + j], want, atol=1e-12)
            assert_almost_equal(ptp[i * 3 + j], want, atol=1e-12)


def test_svdvals_matches_scipy_on_a_tall_matrix() raises:
    # scipy.linalg.svdvals of the 5x3 above, descending.
    var a = _tall()
    var s = svdvals(a).to_host()
    var expected = [17.571796832206623, 1.6897901947205591, 0.6136490735587791]
    for i in range(3):
        assert_almost_equal(
            s[i], Scalar[dtype](expected[i]), rtol=1e-10, atol=1e-12
        )


def test_svdvals_matches_scipy_on_hilbert_four() raises:
    var a = _hilbert[4]()
    var s = svdvals(a).to_host()
    var expected = [
        1.5002142800592426,
        0.16914122022145016,
        0.006738273605760801,
        9.670230402258657e-05,
    ]
    for i in range(4):
        assert_almost_equal(
            s[i], Scalar[dtype](expected[i]), rtol=1e-8, atol=1e-13
        )


def test_svdvals_agrees_with_the_array_tier_as_a_multiset() raises:
    # One-sided Jacobi there, Golub-Kahan here: agreement is the check.
    comptime n = 4
    var a = _hilbert[n]()
    var lifted = to_array[P](a)
    var here = svdvals(a).to_host()
    var there = array_svdvals[P, n, sweeps=20](lifted)
    var there_sorted = List[Float64](capacity=n)
    for i in range(n):
        there_sorted.append(Float64(there[i].v[0]))
    for i in range(n):
        for j in range(i + 1, n):
            if there_sorted[j] > there_sorted[i]:
                var tmp = there_sorted[i]
                there_sorted[i] = there_sorted[j]
                there_sorted[j] = tmp
    for i in range(n):
        assert_almost_equal(
            Float64(here[i]), there_sorted[i], rtol=1e-8, atol=1e-12
        )


def test_svd_reconstructs_a_from_its_factors() raises:
    comptime m = 5
    comptime n = 3
    var a = _tall()
    var original = _copy_rect(a)
    var result = svd(a)

    var ctx = a.context()
    var scaled = zeros[dtype, m, n](ctx)
    var u = result.u.to_host()
    var s = result.s.to_host()
    var host = scaled.to_host()
    for i in range(m):
        for j in range(n):
            host[i * n + j] = u[i * n + j] * s[j]
    scaled.copy_from_host(host)

    var vt = transpose(result.v)
    var back = matmul(scaled, vt).to_host()
    var source = original.to_host()
    for i in range(m * n):
        assert_almost_equal(back[i], source[i], atol=1e-11)


def test_svd_factors_are_orthonormal_and_values_descending() raises:
    var a = _tall()
    var result = svd(a)
    var ut = transpose(result.u)
    var vt = transpose(result.v)
    var utu = matmul(ut, result.u).to_host()
    var vtv = matmul(vt, result.v).to_host()
    for i in range(3):
        for j in range(3):
            var want = Scalar[dtype](1.0) if i == j else Scalar[dtype](0.0)
            assert_almost_equal(utu[i * 3 + j], want, atol=1e-11)
            assert_almost_equal(vtv[i * 3 + j], want, atol=1e-11)
    var s = result.s.to_host()
    assert_equal(s[0] >= s[1], True)
    assert_equal(s[1] >= s[2], True)


def test_svd_values_equal_svdvals() raises:
    var a = _tall()
    var b = _copy_rect(a)
    var with_vectors = svd(a).s.to_host()
    var alone = svdvals(b).to_host()
    for i in range(3):
        assert_almost_equal(with_vectors[i], alone[i], atol=1e-13)


def test_svdvals_finds_a_rank_deficient_matrix() raises:
    # Column 1 is twice column 0, so the rank is 2 and the smallest singular
    # value is zero to rounding: scipy.linalg.svdvals gives
    # [12.269416474076381, 1.2088918006435092, 7.4e-16].
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, 4, 3](
        ctx, [1.0, 2.0, 1.0, 2.0, 4.0, 0.0, 3.0, 6.0, 1.0, 4.0, 8.0, 0.0]
    )
    var s = svdvals(a).to_host()
    assert_almost_equal(s[0], Scalar[dtype](12.269416474076381), rtol=1e-10)
    assert_almost_equal(s[1], Scalar[dtype](1.2088918006435092), rtol=1e-10)
    assert_almost_equal(s[2], Scalar[dtype](0.0), atol=1e-12)


def _rank_two() raises -> Static[dtype, 4, 3]:
    var ctx = DeviceContext(api="cpu")
    return Static[dtype, 4, 3](
        ctx, [1.0, 2.0, 1.0, 2.0, 4.0, 0.0, 3.0, 6.0, 1.0, 4.0, 8.0, 0.0]
    )


def test_pinv_inverts_a_full_rank_tall_matrix_from_the_left() raises:
    # pinv(A) is n x m and pinv(A) @ A is the n x n identity at full rank.
    var a = _tall()
    var original = _copy_rect(a)
    var p = pinv(a)
    var product = matmul(p, original).to_host()
    for i in range(3):
        for j in range(3):
            var want = Scalar[dtype](1.0) if i == j else Scalar[dtype](0.0)
            assert_almost_equal(product[i * 3 + j], want, atol=1e-10)


def test_pinv_of_a_square_matrix_is_its_inverse() raises:
    # scipy.linalg.pinv(hilbert(4))[0, 0] = 16, [3, 3] = 2800 -- the
    # inverse Hilbert matrix's exact entries, to cond(H) * eps.
    var a = _hilbert[4]()
    var p = pinv(a).to_host()
    assert_almost_equal(p[0], Scalar[dtype](16.0), rtol=1e-8)
    assert_almost_equal(p[15], Scalar[dtype](2800.0), rtol=1e-8)


def test_pinv_of_a_rank_deficient_matrix_matches_scipy_and_penrose() raises:
    # scipy.linalg.pinv of the rank-2 matrix, rows 0 and 2; and the first
    # Moore-Penrose condition A P A == A, which `inverse` cannot satisfy here.
    var a = _rank_two()
    var original = _copy_rect(a)
    var p = pinv(a)
    var ph = p.to_host()
    var row0 = [
        -0.00909090909090875,
        0.01818181818181808,
        0.00909090909090934,
        0.03636363636363615,
    ]
    var row2 = [
        0.5909090909090909,
        -0.18181818181818196,
        0.4090909090909091,
        -0.36363636363636354,
    ]
    for j in range(4):
        assert_almost_equal(ph[0 * 4 + j], Scalar[dtype](row0[j]), atol=1e-10)
        assert_almost_equal(ph[2 * 4 + j], Scalar[dtype](row2[j]), atol=1e-10)

    var ap = matmul(original, p)
    var apa = matmul(ap, original).to_host()
    var source = original.to_host()
    for i in range(12):
        assert_almost_equal(apa[i], source[i], atol=1e-10)


def test_cond_matches_scipy_on_hilbert_four() raises:
    var a = _hilbert[4]()
    assert_almost_equal(cond(a), Scalar[dtype](15513.738738929003), rtol=1e-8)


def test_cond_of_a_singular_matrix_is_infinite_or_huge() raises:
    # numpy.linalg.cond([[1, 2], [2, 4]]) is about 2.8e16; the smallest
    # singular value is zero to rounding, so the ratio is at the floor of
    # what float64 can express or past it.
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, 2, 2](ctx, [1.0, 2.0, 2.0, 4.0])
    assert_equal(Float64(cond(a)) > 1e15, True)


def test_matrix_rank_counts_independent_directions() raises:
    var full = _hilbert[4]()
    assert_equal(matrix_rank(full), 4)
    var deficient = _rank_two()
    assert_equal(matrix_rank(deficient), 2)
    var tall = _tall()
    assert_equal(matrix_rank(tall), 3)


def test_matrix_rank_honours_an_explicit_tolerance() raises:
    # Hilbert-4's singular values are 1.5, 0.17, 6.7e-3, 9.7e-5: a
    # tolerance of 1e-3 counts three of them.
    var a = _hilbert[4]()
    assert_equal(matrix_rank(a, tol=1e-3), 3)


def test_lstsq_svd_agrees_with_qr_at_full_rank() raises:
    # scipy.linalg.lstsq(A, b) for the tall matrix: the two routes must
    # give the same x, and SciPy's.
    var a = _tall()
    var a2 = _copy_rect(a)
    var ctx = a.context()
    var b = Static[dtype, 5](ctx, [1.0, 2.0, 3.0, 4.0, 5.0])
    var b2 = Static[dtype, 5](ctx, [1.0, 2.0, 3.0, 4.0, 5.0])
    var via_qr = lstsq(a, b).to_host()
    var via_svd = lstsq[method="svd"](a2, b2).to_host()
    var expected = [
        1.915662650602404,
        -0.9638554216867546,
        -0.16867469879517083,
    ]
    for i in range(3):
        assert_almost_equal(via_svd[i], Scalar[dtype](expected[i]), atol=1e-10)
        assert_almost_equal(via_qr[i], via_svd[i], atol=1e-10)


def test_lstsq_svd_returns_the_minimum_norm_solution_when_rank_deficient() raises:
    # scipy.linalg.lstsq on the rank-2 matrix: x = [0.2364, 0.4727, -0.3636],
    # the minimum-norm member of the solution family.
    var a = _rank_two()
    var ctx = a.context()
    var b = Static[dtype, 4](ctx, [1.0, 2.0, 3.0, 5.0])
    var x = lstsq[method="svd"](a, b).to_host()
    var expected = [
        0.23636363636363628,
        0.47272727272727294,
        -0.3636363636363633,
    ]
    for i in range(3):
        assert_almost_equal(x[i], Scalar[dtype](expected[i]), atol=1e-10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


# ------------------------------------------------ Hessenberg, eigvals, schur


def _general[n: Int](values: List[Float64]) raises -> Static[dtype, n, n]:
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, n, n](ctx)
    var host = a.to_host()
    for i in range(n * n):
        host[i] = Scalar[dtype](values[i])
    a.copy_from_host(host)
    return a^


def _matrix_a() raises -> Static[dtype, 4, 4]:
    """Nonsymmetric with four real eigenvalues (SciPy: -2.1975, 1.0844,
    2.2685, 6.8446), trace 8, determinant -37."""
    return _general[4](
        [
            4.0,
            1.0,
            -2.0,
            2.0,
            1.0,
            2.0,
            0.0,
            1.0,
            -2.0,
            0.0,
            3.0,
            -2.0,
            2.0,
            1.0,
            -2.0,
            -1.0,
        ]
    )


def _matrix_b() raises -> Static[dtype, 4, 4]:
    """Two complex pairs, `1 +- 2.449i` and `2 +- 2i`; trace 6, determinant 56.
    """
    return _general[4](
        [
            1.0,
            -3.0,
            0.5,
            0.0,
            2.0,
            1.0,
            0.0,
            1.5,
            0.0,
            0.0,
            2.0,
            -4.0,
            0.0,
            0.0,
            1.0,
            2.0,
        ]
    )


def test_hessenberg_is_zero_below_the_subdiagonal_and_a_similarity() raises:
    comptime n = 4
    var a = _matrix_a()
    var original = _copy_of(a)
    var reduced = hessenberg(a)
    var h = reduced.h.to_host()
    for i in range(n):
        for j in range(n):
            if i > j + 1:
                assert_equal(h[i * n + j], Scalar[dtype](0))
    var q = reduced.q()
    var qt = transpose(q)
    var identity = matmul(qt, q).to_host()
    for i in range(n):
        for j in range(n):
            var want = Scalar[dtype](1.0) if i == j else Scalar[dtype](0.0)
            assert_almost_equal(identity[i * n + j], want, atol=1e-12)
    # Q H Q^T reconstructs A.
    var half = matmul(q, reduced.h)
    var back = matmul(half, qt).to_host()
    var source = original.to_host()
    for i in range(n * n):
        assert_almost_equal(back[i], source[i], atol=1e-12)


def test_hessenberg_of_a_two_by_two_is_itself() raises:
    var a = _general[2]([1.0, 2.0, 3.0, 4.0])
    var reduced = hessenberg(a)
    var h = reduced.h.to_host()
    assert_equal(h[0], 1.0)
    assert_equal(h[3], 4.0)
    var q = reduced.q().to_host()
    assert_equal(q[0], 1.0)
    assert_equal(q[1], 0.0)


def _sorted_pairs(
    mut values: Eigenvalues[dtype, 4]
) raises -> Tuple[List[Float64], List[Float64]]:
    """Eigenvalues sorted by real part, then imaginary part."""
    var re = values.re.to_host()
    var im = values.im.to_host()
    var order = List[Int](capacity=4)
    for i in range(4):
        order.append(i)
    for i in range(1, 4):
        var j = i
        while j > 0:
            var a = order[j]
            var b = order[j - 1]
            var earlier = Float64(re[a]) < Float64(re[b]) - 1e-9 or (
                abs(Float64(re[a]) - Float64(re[b])) <= 1e-9
                and Float64(im[a]) < Float64(im[b])
            )
            if not earlier:
                break
            order[j] = b
            order[j - 1] = a
            j -= 1
    var sre = List[Float64](capacity=4)
    var sim = List[Float64](capacity=4)
    for i in range(4):
        sre.append(Float64(re[order[i]]))
        sim.append(Float64(im[order[i]]))
    return (sre^, sim^)


def test_eigvals_matches_scipy_for_a_real_spectrum() raises:
    var a = _matrix_a()
    var values = eigvals(a)
    var sorted = _sorted_pairs(values)
    var want: List[Float64] = [
        -2.197516977439422,
        1.0843644637732162,
        2.268531406431242,
        6.844621107234967,
    ]
    for i in range(4):
        assert_almost_equal(sorted[0][i], want[i], atol=1e-12)
        assert_equal(sorted[1][i], 0.0)


def test_eigvals_matches_scipy_for_two_complex_pairs() raises:
    var b = _matrix_b()
    var values = eigvals(b)
    var sorted = _sorted_pairs(values)
    var want_re: List[Float64] = [1.0, 1.0, 2.0, 2.0]
    var want_im: List[Float64] = [
        -2.4494897427831783,
        2.4494897427831783,
        -2.0,
        2.0,
    ]
    for i in range(4):
        assert_almost_equal(sorted[0][i], want_re[i], atol=1e-12)
        assert_almost_equal(sorted[1][i], want_im[i], atol=1e-12)
    # Trace and determinant identities on the same spectrum.
    var re = values.re.to_host()
    var im = values.im.to_host()
    var trace = Float64(0)
    for i in range(4):
        trace += Float64(re[i])
    assert_almost_equal(trace, 6.0, atol=1e-12)
    # The pairs are adjacent, positive imaginary part first.
    for i in range(0, 4, 2):
        assert_equal(re[i], re[i + 1])
        assert_equal(im[i], -im[i + 1])
        assert_equal(im[i] > 0, True)


def test_eigvals_of_a_quarter_turn_are_the_imaginary_units() raises:
    var c = _general[2]([0.0, -1.0, 1.0, 0.0])
    var values = eigvals(c)
    var re = values.re.to_host()
    var im = values.im.to_host()
    assert_almost_equal(re[0], 0.0, atol=1e-15)
    assert_almost_equal(re[1], 0.0, atol=1e-15)
    assert_almost_equal(im[0], 1.0, atol=1e-15)
    assert_almost_equal(im[1], -1.0, atol=1e-15)


def test_eigvals_agrees_with_eigvalsh_on_a_symmetric_matrix() raises:
    comptime n = 5
    var a = _hilbert[n]()
    var b = _hilbert[n]()
    var general = eigvals(a)
    var symmetric = eigvalsh(b).to_host()
    var re = general.re.to_host()
    var im = general.im.to_host()
    # Sort the general ones ascending to compare.
    var vals = List[Float64](capacity=n)
    for i in range(n):
        assert_equal(im[i], 0.0)
        vals.append(Float64(re[i]))
    for i in range(1, n):
        var j = i
        while j > 0 and vals[j] < vals[j - 1]:
            var tmp = vals[j]
            vals[j] = vals[j - 1]
            vals[j - 1] = tmp
            j -= 1
    for i in range(n):
        assert_almost_equal(vals[i], Float64(symmetric[i]), atol=1e-12)


def test_schur_reconstructs_and_is_quasi_triangular() raises:
    comptime n = 4
    var b = _matrix_b()
    var original = _copy_of(b)
    var decomposed = schur(b)
    var t = decomposed.t.to_host()
    # Exact zeros below the first subdiagonal.
    for i in range(n):
        for j in range(n):
            if i > j + 1:
                assert_equal(t[i * n + j], Scalar[dtype](0))
    # Every surviving 2x2 diagonal block carries a complex pair.
    var i = 0
    while i < n - 1:
        if t[(i + 1) * n + i] != 0:
            var p = Float64(t[i * n + i] - t[(i + 1) * n + (i + 1)])
            var disc = p * p + 4.0 * Float64(
                t[i * n + (i + 1)] * t[(i + 1) * n + i]
            )
            assert_equal(disc < 0, True)
            i += 2
        else:
            i += 1
    var zt = transpose(decomposed.z)
    var identity = matmul(zt, decomposed.z).to_host()
    for r in range(n):
        for c in range(n):
            var want = Scalar[dtype](1.0) if r == c else Scalar[dtype](0.0)
            assert_almost_equal(identity[r * n + c], want, atol=1e-12)
    var half = matmul(decomposed.z, decomposed.t)
    var back = matmul(half, zt).to_host()
    var source = original.to_host()
    for k in range(n * n):
        assert_almost_equal(back[k], source[k], atol=1e-12)


def test_schur_of_a_real_spectrum_is_triangular_with_the_eigenvalues_on_the_diagonal() raises:
    comptime n = 4
    var a = _matrix_a()
    var original = _copy_of(a)
    var decomposed = schur(a)
    var t = decomposed.t.to_host()
    for i in range(1, n):
        assert_almost_equal(t[i * n + (i - 1)], Scalar[dtype](0), atol=1e-12)
    var diag = List[Float64](capacity=n)
    for i in range(n):
        diag.append(Float64(t[i * n + i]))
    for i in range(1, n):
        var j = i
        while j > 0 and diag[j] < diag[j - 1]:
            var tmp = diag[j]
            diag[j] = diag[j - 1]
            diag[j - 1] = tmp
            j -= 1
    var want: List[Float64] = [
        -2.197516977439422,
        1.0843644637732162,
        2.268531406431242,
        6.844621107234967,
    ]
    for i in range(n):
        assert_almost_equal(diag[i], want[i], atol=1e-12)
    var zt = transpose(decomposed.z)
    var half = matmul(decomposed.z, decomposed.t)
    var back = matmul(half, zt).to_host()
    var source = original.to_host()
    for k in range(n * n):
        assert_almost_equal(back[k], source[k], atol=1e-12)
