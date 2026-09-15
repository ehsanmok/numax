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


def _sytrd_q_at[
    n: Int, block: Int
](want: List[Scalar[dtype]]) raises where block >= 1:
    """Form `Q` from a Hilbert `n` reduction at this panel width and pin it
    entry by entry to the reference answer."""
    var a = _hilbert[n]()
    var reduced = sytrd[dtype, n, False, block](a)
    var q = reduced.q().to_host()
    for i in range(n * n):
        assert_almost_equal(q[i], want[i], atol=1e-12)


def test_sytrd_q_matches_the_unblocked_accumulation() raises:
    # `block` decides only how many reflectors ride in one block reflector,
    # so `Q` has to come out the same at every width. `block == 1` is the
    # sharp end -- one reflector per panel, which is the per-reflector walk
    # this replaced -- and `block == n` is the other, a single panel whose
    # `T` is the whole triangular factor.
    comptime six = 6
    var a6 = _hilbert[six]()
    var ref6 = sytrd[dtype, six, False, six](a6)
    var want6 = ref6.q().to_host()
    _sytrd_q_at[six, 1](want6)
    _sytrd_q_at[six, 2](want6)
    _sytrd_q_at[six, 3](want6)
    _sytrd_q_at[six, six](want6)

    comptime five = 5
    var a5 = _hilbert[five]()
    var ref5 = sytrd[dtype, five, False, five](a5)
    var want5 = ref5.q().to_host()
    _sytrd_q_at[five, 1](want5)
    _sytrd_q_at[five, 2](want5)


def test_sytrd_q_at_a_ragged_panel_width() raises:
    # `n = 7` leaves five reflectors, so `block = 3` runs a full panel and
    # a two-wide one -- and the ragged panel is applied *first*, which is
    # where a scratch viewed at two widths would show up.
    comptime n = 7
    comptime block = 3
    var a = _hilbert[n]()
    var original = _copy_of(a)
    var reduced = sytrd[dtype, n, False, block](a)
    var q = reduced.q()
    var qt = transpose(q)

    var identity = matmul(qt, q).to_host()
    for i in range(n):
        for j in range(n):
            var want = Scalar[dtype](1.0) if i == j else Scalar[dtype](0.0)
            assert_almost_equal(identity[i * n + j], want, atol=1e-12)

    var ctx = a.context()
    var t = zeros[dtype, n, n](ctx)
    var band = t.to_host()
    var d = reduced.d.to_host()
    var e = reduced.e.to_host()
    for i in range(n):
        band[i * n + i] = d[i]
    for i in range(n - 1):
        band[(i + 1) * n + i] = e[i]
        band[i * n + (i + 1)] = e[i]
    t.copy_from_host(band)

    var half = matmul(q, t)
    var back = matmul(half, qt).to_host()
    var source = original.to_host()
    for i in range(n * n):
        assert_almost_equal(back[i], source[i], atol=1e-12)


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


def _eigh_agrees[
    n: Int, block: Int
](want_w: List[Scalar[dtype]], want_v: List[Scalar[dtype]]) raises where (
    block >= 1
):
    """Run `eigh` on Hilbert `n` at this `block` and pin it to a reference
    answer, values and vectors alike."""
    var a = _hilbert[n]()
    var got = eigh[dtype, n, False, block](a)
    var w = got.values.to_host()
    var v = got.vectors.to_host()
    for i in range(n):
        assert_almost_equal(w[i], want_w[i], atol=1e-12)
    for i in range(n * n):
        assert_almost_equal(v[i], want_v[i], atol=1e-12)


def test_eigh_blocking_does_not_change_the_answer() raises:
    # `block` decides only how many commuting rotations ride in one GEMM,
    # so every size has to produce the same decomposition. `block == 1` is
    # the sharp end: one rotation per window, so the windows have to come
    # out in exactly the order the sweep emitted them, and a group index
    # that ran the other way would reverse every sweep.
    comptime six = 6
    var a6 = _hilbert[six]()
    var ref6 = eigh[dtype, six, False, six](a6)
    var w6 = ref6.values.to_host()
    var v6 = ref6.vectors.to_host()
    _eigh_agrees[six, 1](w6, v6)
    _eigh_agrees[six, 2](w6, v6)
    _eigh_agrees[six, 3](w6, v6)
    _eigh_agrees[six, six](w6, v6)

    comptime five = 5
    var a5 = _hilbert[five]()
    var ref5 = eigh[dtype, five, False, five](a5)
    var w5 = ref5.values.to_host()
    var v5 = ref5.vectors.to_host()
    _eigh_agrees[five, 1](w5, v5)
    _eigh_agrees[five, 2](w5, v5)


def test_eigh_ragged_window_sizes() raises:
    # `n = 40` at `block = 8` is the shape the edge cases live in: several
    # windows per batch, several batches per run, and windows at both ends
    # of the band narrower than the `2 * block` the formula allows.
    comptime n = 40
    var a = _hilbert[n]()
    var original = _copy_of(a)
    var result = eigh[dtype, n, False, 8](a)

    var av = matmul(original, result.vectors).to_host()
    var v = result.vectors.to_host()
    var w = result.values.to_host()
    var residual = Float64(0)
    for j in range(n):
        for i in range(n):
            var gap = Float64(av[i * n + j] - w[j] * v[i * n + j])
            residual = max(residual, abs(gap))
    assert_equal(residual < 1e-10, True)

    var vt = transpose(result.vectors)
    var gram = matmul(vt, result.vectors).to_host()
    var drift = Float64(0)
    for i in range(n):
        for j in range(n):
            var want = Scalar[dtype](1.0) if i == j else Scalar[dtype](0.0)
            drift = max(drift, abs(Float64(gram[i * n + j] - want)))
    assert_equal(drift < 1e-10, True)


def test_eigh_at_n_one_and_two() raises:
    # The sizes where the sweep never rotates (`n == 1`) and where it
    # rotates on the one index there is (`n == 2`), at both ends of the
    # `block` range.
    var ctx = DeviceContext(api="cpu")
    var a1 = zeros[dtype, 1, 1](ctx)
    a1.copy_from_host([Scalar[dtype](3.5)])
    var r1 = eigh(a1)
    assert_almost_equal(r1.values.to_host()[0], Scalar[dtype](3.5), atol=1e-14)
    assert_almost_equal(
        abs(r1.vectors.to_host()[0]), Scalar[dtype](1.0), atol=1e-14
    )

    var pair: List[Scalar[dtype]] = [
        Scalar[dtype](2.0),
        Scalar[dtype](1.0),
        Scalar[dtype](1.0),
        Scalar[dtype](2.0),
    ]
    var a2 = zeros[dtype, 2, 2](ctx)
    a2.copy_from_host(pair)
    var r2 = eigh[dtype, 2, False, 1](a2)
    var w2 = r2.values.to_host()
    assert_almost_equal(w2[0], Scalar[dtype](1.0), atol=1e-14)
    assert_almost_equal(w2[1], Scalar[dtype](3.0), atol=1e-14)

    var b2 = zeros[dtype, 2, 2](ctx)
    b2.copy_from_host(pair)
    var wide = eigh[dtype, 2, False, 32](b2)
    var w3 = wide.values.to_host()
    var v2 = r2.vectors.to_host()
    var v3 = wide.vectors.to_host()
    for i in range(2):
        assert_almost_equal(w3[i], w2[i], atol=1e-14)
    for i in range(4):
        assert_almost_equal(v3[i], v2[i], atol=1e-14)


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


def _rect[m: Int, n: Int]() raises -> Static[dtype, m, n]:
    """A deterministic `m x n` with no symmetry, so `Q` and `P` are both
    non-trivial at every column."""
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, m, n](ctx)
    var host = a.to_host()
    for i in range(m):
        for j in range(n):
            host[i * n + j] = Scalar[dtype](
                1.0 / Float64(i + j + 1) + 0.25 * Float64((i * 3 + j) % 5)
            )
    a.copy_from_host(host)
    return a^


def _gebrd_qp_at[
    m: Int, n: Int, block: Int
](
    mut a: Static[dtype, m, n],
    want_q: List[Scalar[dtype]],
    want_p: List[Scalar[dtype]],
) raises where (m >= n and n >= 1 and block >= 1):
    """Form `Q` and `P` from a bidiagonal reduction at this panel width and
    pin both entry by entry to the reference answer."""
    var reduced = gebrd[dtype, m, n, False, block](a)
    var q = reduced.q().to_host()
    var p = reduced.p().to_host()
    for i in range(m * n):
        assert_almost_equal(q[i], want_q[i], atol=1e-12)
    for i in range(n * n):
        assert_almost_equal(p[i], want_p[i], atol=1e-12)


def test_bidiagonal_q_and_p_match_the_unblocked_walk() raises:
    # `block` decides only how many reflectors ride in one block reflector.
    # `block == 1` is the per-reflector walk this replaced; `block == n` is
    # one panel. `Q` comes straight off `qr_factor`'s walk because `left` is
    # already in QR's packed form, while `P` goes through a transposing pack
    # and then the row-shifted view `orgtr` uses -- two different adapters,
    # so both are pinned here.
    comptime m = 5
    comptime n = 3
    var a = _tall()
    var reference = gebrd[dtype, m, n, False, n](a)
    var want_q = reference.q().to_host()
    var want_p = reference.p().to_host()
    var a1 = _tall()
    _gebrd_qp_at[m, n, 1](a1, want_q, want_p)
    var a2 = _tall()
    _gebrd_qp_at[m, n, 2](a2, want_q, want_p)
    var a3 = _tall()
    _gebrd_qp_at[m, n, n](a3, want_q, want_p)

    # `n = 7` at `block = 3` leaves a one-wide last panel, which the walk
    # applies *first*: the shape where a scratch viewed at two widths would
    # show up.
    comptime wide = 7
    var b = _rect[wide, wide]()
    var b_ref = gebrd[dtype, wide, wide, False, wide](b)
    var bq = b_ref.q().to_host()
    var bp = b_ref.p().to_host()
    var b1 = _rect[wide, wide]()
    _gebrd_qp_at[wide, wide, 1](b1, bq, bp)
    var b3 = _rect[wide, wide]()
    _gebrd_qp_at[wide, wide, 3](b3, bq, bp)

    # Ragged on a rectangle too: five right reflectors, panels of two.
    comptime tall = 9
    comptime narrow = 5
    var c = _rect[tall, narrow]()
    var c_ref = gebrd[dtype, tall, narrow, False, narrow](c)
    var cq = c_ref.q().to_host()
    var cp = c_ref.p().to_host()
    var c2 = _rect[tall, narrow]()
    _gebrd_qp_at[tall, narrow, 2](c2, cq, cp)


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


def _svd_agrees[
    m: Int, n: Int, block: Int
](
    mut a: Static[dtype, m, n],
    want_s: List[Scalar[dtype]],
    want_u: List[Scalar[dtype]],
    want_v: List[Scalar[dtype]],
) raises where (m >= n and n >= 1 and block >= 1):
    """Run `svd` at this `block` and pin it to a reference answer, values
    and vectors alike.

    `(u_j, v_j)` and `(-u_j, -v_j)` are the same decomposition, so each
    column's sign is fixed from the reference's largest entry before the
    entries are compared.
    """
    var got = svd[dtype, m, n, False, block](a)
    var s = got.s.to_host()
    var u = got.u.to_host()
    var v = got.v.to_host()
    for j in range(n):
        assert_almost_equal(s[j], want_s[j], atol=1e-12)
        var pivot = 0
        for i in range(m):
            if abs(want_u[i * n + j]) > abs(want_u[pivot * n + j]):
                pivot = i
        var sign = Scalar[dtype](1.0)
        if u[pivot * n + j] * want_u[pivot * n + j] < 0:
            sign = Scalar[dtype](-1.0)
        for i in range(m):
            assert_almost_equal(
                sign * u[i * n + j], want_u[i * n + j], atol=1e-12
            )
        for i in range(n):
            assert_almost_equal(
                sign * v[i * n + j], want_v[i * n + j], atol=1e-12
            )


def test_svd_blocking_does_not_change_the_answer() raises:
    # Two knobs share the name: `gebrd`'s panel width, which decides how
    # `Q` and `P` are formed, and the rotation window of the `2n` sweep.
    # Neither may move the answer, so every width is pinned against
    # `block == n`.
    comptime m = 5
    comptime n = 3
    var a = _tall()
    var reference = svd[dtype, m, n, False, n](a)
    var want_s = reference.s.to_host()
    var want_u = reference.u.to_host()
    var want_v = reference.v.to_host()
    var a1 = _tall()
    _svd_agrees[m, n, 1](a1, want_s, want_u, want_v)
    var a2 = _tall()
    _svd_agrees[m, n, 2](a2, want_s, want_u, want_v)
    var a3 = _tall()
    _svd_agrees[m, n, n](a3, want_s, want_u, want_v)

    comptime six = 6
    var b = _hilbert[six]()
    var b_ref = svd[dtype, six, six, False, six](b)
    var bs = b_ref.s.to_host()
    var bu = b_ref.u.to_host()
    var bv = b_ref.v.to_host()
    var b1 = _hilbert[six]()
    _svd_agrees[six, six, 1](b1, bs, bu, bv)
    var b2 = _hilbert[six]()
    _svd_agrees[six, six, 2](b2, bs, bu, bv)
    var b3 = _hilbert[six]()
    _svd_agrees[six, six, six](b3, bs, bu, bv)


def _svd_round_trip[
    m: Int, n: Int
](mut a: Static[dtype, m, n]) raises where m >= n and n >= 1:
    """`U diag(s) V^T == A`, both factors orthonormal, `s` descending and
    equal to `svdvals` -- the whole contract, at whatever size."""
    var original = _copy_rect(a)
    var values_only = _copy_rect(a)
    var result = svd(a)
    var u = result.u.to_host()
    var s = result.s.to_host()

    var ctx = original.context()
    var scaled = zeros[dtype, m, n](ctx)
    var host = scaled.to_host()
    for i in range(m):
        for j in range(n):
            host[i * n + j] = u[i * n + j] * s[j]
    scaled.copy_from_host(host)
    var vt = transpose(result.v)
    var back = matmul(scaled, vt).to_host()
    var source = original.to_host()
    for i in range(m * n):
        assert_almost_equal(back[i], source[i], atol=1e-12)

    var ut = transpose(result.u)
    var utu = matmul(ut, result.u).to_host()
    var vtv = matmul(vt, result.v).to_host()
    for i in range(n):
        for j in range(n):
            var want = Scalar[dtype](1.0) if i == j else Scalar[dtype](0.0)
            assert_almost_equal(utu[i * n + j], want, atol=1e-12)
            assert_almost_equal(vtv[i * n + j], want, atol=1e-12)

    var alone = svdvals(values_only).to_host()
    for j in range(n):
        assert_almost_equal(s[j], alone[j], atol=1e-13)
    for j in range(n - 1):
        assert_equal(s[j] >= s[j + 1], True)


def test_svd_at_n_one_two_three() raises:
    # The de-interleave reads a `2n x 2n` source into an `n x n`
    # destination, and a cross-shape `elementwise` is exactly where small
    # extents have misbehaved before -- so the three smallest are pinned.
    var ctx = DeviceContext(api="cpu")
    var one = Static[dtype, 3, 1](ctx, [2.0, 1.0, 2.0])
    _svd_round_trip[3, 1](one)
    var s1 = svdvals(one).to_host()
    assert_almost_equal(s1[0], Scalar[dtype](3.0), atol=1e-13)

    var two = Static[dtype, 3, 2](ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 7.0])
    _svd_round_trip[3, 2](two)

    var three = _rect[3, 3]()
    _svd_round_trip[3, 3](three)

    var tall_three = _rect[6, 3]()
    _svd_round_trip[6, 3](tall_three)


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


def _hessenberg_q_at[
    block: Int
](mut a: Static[dtype, 4, 4]) raises where block >= 1:
    """`Q^T Q = I` and `Q H Q^T = A` for a 4x4 reduction at this panel
    width -- the two claims that say the panels were applied in the right
    order and on the right rows."""
    comptime n = 4
    var original = _copy_of(a)
    var reduced = hessenberg[dtype, n, False, block](a)
    var q = reduced.q()
    var qt = transpose(q)

    var identity = matmul(qt, q).to_host()
    for i in range(n):
        for j in range(n):
            var want = Scalar[dtype](1.0) if i == j else Scalar[dtype](0.0)
            assert_almost_equal(identity[i * n + j], want, atol=1e-12)

    var half = matmul(q, reduced.h)
    var back = matmul(half, qt).to_host()
    var source = original.to_host()
    for i in range(n * n):
        assert_almost_equal(back[i], source[i], atol=1e-12)


def test_hessenberg_q_is_orthogonal_at_every_block() raises:
    # `orghr` reads the same packed form `orgtr` does, except that a column
    # whose reflection was the identity is left as written zeros rather
    # than the reduced matrix's own entries. Both fixtures, both ends of
    # the `block` range.
    comptime n = 4
    var a1 = _matrix_a()
    _hessenberg_q_at[1](a1)
    var a2 = _matrix_a()
    _hessenberg_q_at[2](a2)
    var a3 = _matrix_a()
    _hessenberg_q_at[n](a3)

    var b1 = _matrix_b()
    _hessenberg_q_at[1](b1)
    var b2 = _matrix_b()
    _hessenberg_q_at[2](b2)
    var b3 = _matrix_b()
    _hessenberg_q_at[n](b3)


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


def _general_hash[n: Int]() raises -> Static[dtype, n, n]:
    """A deterministic nonsymmetric matrix of any size, with a spectrum
    spread enough that the Francis chase runs its full length rather than
    deflating after a step or two."""
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, n, n](ctx)
    var host = a.to_host()
    for i in range(n):
        for j in range(n):
            host[i * n + j] = Scalar[dtype](
                Float64((i * 37 + j * 11) % 17) * 0.125 - 1.0
            )
        host[i * n + i] = Scalar[dtype](Float64(i) * 0.5 + 1.0)
    a.copy_from_host(host)
    return a^


def _assert_quasi_triangular[
    n: Int
](t: List[Scalar[dtype]]) raises where n >= 1:
    """Exact zeros below the first subdiagonal, and every surviving `2 x 2`
    diagonal block carrying a complex pair -- the real Schur form."""
    for i in range(n):
        for j in range(n):
            if i > j + 1:
                assert_equal(t[i * n + j], Scalar[dtype](0))
    var i = 0
    while i < n - 1:
        if t[(i + 1) * n + i] != 0:
            var gap = Float64(t[i * n + i] - t[(i + 1) * n + (i + 1)])
            var disc = gap * gap + 4.0 * Float64(
                t[i * n + (i + 1)] * t[(i + 1) * n + i]
            )
            assert_equal(disc < 0, True)
            i += 2
        else:
            i += 1


def _schur_agrees[
    n: Int, block: Int
](
    mut a: Static[dtype, n, n],
    want_t: List[Scalar[dtype]],
    want_z: List[Scalar[dtype]],
) raises where (block >= 1 and n >= 1):
    """Run `schur` at this `block` and pin both factors to a reference."""
    var got = schur[dtype, n, False, block](a)
    var t = got.t.to_host()
    var z = got.z.to_host()
    for i in range(n * n):
        assert_almost_equal(t[i], want_t[i], atol=1e-12)
        assert_almost_equal(z[i], want_z[i], atol=1e-12)


def test_schur_blocking_does_not_change_the_answer() raises:
    # `block` decides only how many commuting reflectors ride in one GEMM,
    # so every width has to produce the same `T` and the same `Z` -- signs
    # included, since the transformations themselves are unchanged and only
    # their schedule moves. `block == 1` is the sharp end: one entry per
    # window, so the windows have to come out in exactly the order the
    # chase emitted them, and a tag that ran the other way would reverse
    # every sweep.
    comptime n = 4

    var ref_a = _matrix_a()
    var wide_a = schur[dtype, n, False, n](ref_a)
    var ta = wide_a.t.to_host()
    var za = wide_a.z.to_host()
    var a1 = _matrix_a()
    _schur_agrees[n, 1](a1, ta, za)
    var a2 = _matrix_a()
    _schur_agrees[n, 2](a2, ta, za)
    var a3 = _matrix_a()
    _schur_agrees[n, n](a3, ta, za)

    # The second fixture has two complex pairs, so its chase never takes
    # the real `2 x 2` split and the first one never takes anything else.
    var ref_b = _matrix_b()
    var wide_b = schur[dtype, n, False, n](ref_b)
    var tb = wide_b.t.to_host()
    var zb = wide_b.z.to_host()
    var b1 = _matrix_b()
    _schur_agrees[n, 1](b1, tb, zb)
    var b2 = _matrix_b()
    _schur_agrees[n, 2](b2, tb, zb)
    var b3 = _matrix_b()
    _schur_agrees[n, n](b3, tb, zb)


def test_schur_at_n_one_and_two() raises:
    # The sizes where the chase never runs (`n == 1`), where it deflates
    # straight into the real `2 x 2` split, and where the block is a
    # complex pair that survives to the end.
    var ctx = DeviceContext(api="cpu")
    var a1 = zeros[dtype, 1, 1](ctx)
    a1.copy_from_host([Scalar[dtype](3.5)])
    var r1 = schur(a1)
    assert_almost_equal(r1.t.to_host()[0], Scalar[dtype](3.5), atol=1e-14)
    assert_almost_equal(abs(r1.z.to_host()[0]), Scalar[dtype](1.0), atol=1e-14)

    var real_pair = _general[2]([1.0, 2.0, 3.0, 4.0])
    var original = _copy_of(real_pair)
    var r2 = schur[dtype, 2, False, 1](real_pair)
    var t2 = r2.t.to_host()
    assert_almost_equal(t2[2], Scalar[dtype](0), atol=1e-14)
    var zt2 = transpose(r2.z)
    var half2 = matmul(r2.z, r2.t)
    var back2 = matmul(half2, zt2).to_host()
    var src2 = original.to_host()
    for i in range(4):
        assert_almost_equal(back2[i], src2[i], atol=1e-13)

    var wide_pair = _general[2]([1.0, 2.0, 3.0, 4.0])
    var r2w = schur[dtype, 2, False, 32](wide_pair)
    var t2w = r2w.t.to_host()
    var z2w = r2w.z.to_host()
    var z2 = r2.z.to_host()
    for i in range(4):
        assert_almost_equal(t2w[i], t2[i], atol=1e-12)
        assert_almost_equal(z2w[i], z2[i], atol=1e-12)

    var turn = _general[2]([0.0, -1.0, 1.0, 0.0])
    var turn_source = _copy_of(turn)
    var r3 = schur[dtype, 2, False, 1](turn)
    _assert_quasi_triangular[2](r3.t.to_host())
    var zt3 = transpose(r3.z)
    var half3 = matmul(r3.z, r3.t)
    var back3 = matmul(half3, zt3).to_host()
    var src3 = turn_source.to_host()
    for i in range(4):
        assert_almost_equal(back3[i], src3[i], atol=1e-14)


def test_schur_ragged_window_sizes() raises:
    # `n = 40` at `block = 8` is the shape the edge cases live in: several
    # windows per batch, several batches per run, windows narrower than the
    # `2 * block` the tag allows, and split rotations landing between
    # chases as sweeps of their own.
    comptime n = 40
    var a = _general_hash[n]()
    var original = _copy_of(a)
    var decomposed = schur[dtype, n, False, 8](a)
    _assert_quasi_triangular[n](decomposed.t.to_host())

    var zt = transpose(decomposed.z)
    var gram = matmul(zt, decomposed.z).to_host()
    var drift = Float64(0)
    for i in range(n):
        for j in range(n):
            var want = Scalar[dtype](1.0) if i == j else Scalar[dtype](0.0)
            drift = max(drift, abs(Float64(gram[i * n + j] - want)))
    assert_equal(drift < 1e-10, True)

    var half = matmul(decomposed.z, decomposed.t)
    var back = matmul(half, zt).to_host()
    var source = original.to_host()
    var residual = Float64(0)
    for i in range(n * n):
        residual = max(residual, abs(Float64(back[i] - source[i])))
    assert_equal(residual < 1e-9, True)

    # And at this size the windowing is doing real reordering, so the
    # unblocked schedule has to land on the same decomposition.
    var unblocked = _general_hash[n]()
    _schur_agrees[n, 1](
        unblocked, decomposed.t.to_host(), decomposed.z.to_host()
    )
