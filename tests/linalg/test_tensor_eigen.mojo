"""Tests for `numax.linalg.eigen`, the spectral factorizations over `Tensor`.

The reduction is checked by the two properties that define it rather than
against a table of numbers: `Q^T A Q` is tridiagonal, and it is *similar*
to `A`, so `Q T Q^T` reconstructs the matrix it came from. A wrong
reflector fails the first; a reflector applied on one side only fails the
second while passing the first, which is why both are here.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.array import Static, to_array, transpose, zeros
from numax.core.plain import Plain
from numax.linalg import eigh, eigvalsh, gebrd, matmul, svd, svdvals, sytrd
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
