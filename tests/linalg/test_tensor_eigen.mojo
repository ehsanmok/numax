"""Tests for `numax.linalg.eigen`, the spectral factorizations over `Tensor`.

The reduction is checked by the two properties that define it rather than
against a table of numbers: `Q^T A Q` is tridiagonal, and it is *similar*
to `A`, so `Q T Q^T` reconstructs the matrix it came from. A wrong
reflector fails the first; a reflector applied on one side only fails the
second while passing the first, which is why both are here.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.array import Static, transpose, zeros
from numax.linalg import matmul, sytrd

comptime dtype = DType.float64


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
