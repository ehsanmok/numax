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
from numax.linalg import eigvalsh, matmul, sytrd
from numax.linalg.array import eigvalsh as array_eigvalsh

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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
