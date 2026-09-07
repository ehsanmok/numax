"""Solves against triangular and tridiagonal matrices.
`scipy.linalg`'s `solve_triangular` and the banded solvers.

**Tier 1.** Fixed trip count, no per-lane branching, so all of it launches
inside a GPU thread at any conformer.

MAX has no triangular solve at any size, so there is nothing to delegate
to here and no `Tensor` tier: these are the substitution primitives the
factorizations in `cholesky`, `lu` and `qr` finish with, and they are
`Array`-only.

`tridiagonal_solve` is Thomas, `O(n)` rather than the `O(n^3)` a general
solve costs, which is what makes cubic splines and implicit 1-D PDE steps
tractable. It will not gain a blocked `Tensor` form: Thomas is already
linear and has nothing to hand a GEMM.
"""

from std.collections import Array

from ..core.numeric import FloatLike, guard_nonzero

from .common import _PIVOT_FLOOR, _zeros


def forward_substitution[
    T: FloatLike, n: Int, unit_diagonal: Bool = False
](lower: Array[T, n * n], b: Array[T, n]) -> Array[T, n]:
    """Solve `L @ x = b` for lower-triangular `L`.

    `unit_diagonal=True` treats `L`'s diagonal as an implicit `1` without
    reading it, which is what the packed output of `lu` needs.

    No MAX equivalent at any size: MAX ships no triangular solve (no
    `trsm`, no BLAS-1 at all). This loop is the whole algorithm.
    """
    var x = _zeros[T, n]()
    for i in range(n):
        var total = b[i].copy()
        for j in range(i):
            total = total - (lower[i * n + j] * x[j])
        comptime if unit_diagonal:
            x[i] = total^
        else:
            x[i] = total / guard_nonzero(
                lower[i * n + i], T.constant(_PIVOT_FLOOR)
            )
    return x^


def back_substitution[
    T: FloatLike, n: Int
](upper: Array[T, n * n], b: Array[T, n]) -> Array[T, n]:
    """Solve `U @ x = b` for upper-triangular `U`.

    No MAX equivalent at any size, same as `forward_substitution`.
    """
    var x = _zeros[T, n]()
    for step in range(n):
        var i = n - 1 - step
        var total = b[i].copy()
        for j in range(i + 1, n):
            total = total - (upper[i * n + j] * x[j])
        x[i] = total / guard_nonzero(upper[i * n + i], T.constant(_PIVOT_FLOOR))
    return x^


def tridiagonal_solve[
    T: FloatLike, n: Int
](
    sub: Array[T, n],
    diag: Array[T, n],
    sup: Array[T, n],
    rhs: Array[T, n],
) -> Array[T, n]:
    """Solve a tridiagonal system by the Thomas algorithm.

    `sub[i]` is the entry below the diagonal in row `i` (so `sub[0]` is
    unused) and `sup[i]` is the one above it (so `sup[n-1]` is unused);
    both are passed full-length rather than short by one so the indexing
    matches the row it belongs to.

    No MAX equivalent at any size, and unlike the dense routines here
    there is nothing to gain from one: Thomas is already linear, so
    MAX's `matmul` has nothing to improve on and the blocked treatment
    the dense factorizations get would be pure overhead.

    `O(n)` rather than the `O(n^3)` a general solve would cost, which is
    why cubic splines and implicit one-dimensional PDE steps are tractable
    at all. Also unpivoted -- Thomas is stable without pivoting for
    diagonally dominant or symmetric positive definite systems, which
    covers both of those uses.
    """
    var c_prime = _zeros[T, n]()
    var d_prime = _zeros[T, n]()

    var first = guard_nonzero(diag[0], T.constant(_PIVOT_FLOOR))
    c_prime[0] = sup[0] / first
    d_prime[0] = rhs[0] / first

    for i in range(1, n):
        var denominator = guard_nonzero(
            diag[i] - (sub[i] * c_prime[i - 1]),
            T.constant(_PIVOT_FLOOR),
        )
        c_prime[i] = sup[i] / denominator
        d_prime[i] = (rhs[i] - (sub[i] * d_prime[i - 1])) / denominator

    var x = _zeros[T, n]()
    x[n - 1] = d_prime[n - 1].copy()
    for step in range(1, n):
        var i = n - 1 - step
        x[i] = d_prime[i] - (c_prime[i] * x[i + 1])

    return x^
