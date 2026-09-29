"""Regression test for `_hqr`'s shift schedule: the `float32`
`n = 1024` matrix on which EISPACK's two exceptional shifts (at the tenth
and twentieth sweep, never again) let the Wilkinson shifts cycle until
the iteration budget ran out, at `hessenberg` panel widths 1 and 16.
LAPACK `dlahqr`'s schedule -- an exceptional shift every tenth sweep
without a deflation, alternating ends, and a per-eigenvalue budget --
converges at every width. The matrix is a diagonal plus a period-17
term, so its spectrum is a near-uniform cluster (`findings.mdc` has the
diagnosis). The check is the spectrum's first two power sums against
`trace(A)` and `trace(A^2)`, which the eigenvalues must reproduce
whatever order they deflate in."""

from std.testing import TestSuite, assert_true

from max.gpu.host import DeviceContext

from numax.core.tensor import Static, zeros
from numax.linalg import eigvals

comptime f32 = DType.float32
comptime n = 1024


def _cluster() raises -> Static[f32, n, n]:
    var ctx = DeviceContext(api="cpu")
    var a = zeros[f32, n, n](ctx)
    var host = a.to_host()
    for i in range(n):
        for j in range(n):
            host[i * n + j] = Scalar[f32](
                Float64((i * 37 + j * 11) % 17) * 0.125 - 1.0
            )
        host[i * n + i] = Scalar[f32](Float64(i) * 0.5 + 1.0)
    a.copy_from_host(host)
    return a^


def _check[block: Int]() raises where block >= 1:
    var a = _cluster()
    var host = a.to_host()
    var trace = 0.0
    var trace2 = 0.0
    for i in range(n):
        trace += Float64(host[i * n + i])
        for j in range(n):
            trace2 += Float64(host[i * n + j]) * Float64(host[j * n + i])
    var e = eigvals[block=block](a)
    var re = e.re.to_host()
    var im = e.im.to_host()
    var sum1 = 0.0
    var sum2 = 0.0
    for i in range(n):
        var r = Float64(re[i])
        var m = Float64(im[i])
        sum1 += r
        sum2 += r * r - m * m
    assert_true(abs(sum1 - trace) <= 1e-5 * abs(trace))
    assert_true(abs(sum2 - trace2) <= 1e-4 * abs(trace2))


def test_converges_unblocked() raises:
    """Panel width 1, which raised under the EISPACK schedule."""
    _check[1]()


def test_converges_at_width_16() raises:
    """Panel width 16, which raised after 307,200 sweeps even with the old
    budget raised tenfold."""
    _check[16]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
