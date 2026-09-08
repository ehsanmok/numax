"""Cholesky factorization and its solves over `Array`.
`scipy.linalg`'s `cholesky`/`cho_solve` pair.

**Tier 1** for `cholesky` and the solves: fixed trip count, and it
differentiates. Calling `cholesky` at `Dual` gives the derivative of a
factorization with no adjoint rule written anywhere, which is the concrete
payoff of the conformer layer -- Gaussian process marginal likelihoods,
Kalman updates and multivariate normal densities all bottom out in
`chol(A)` and its log-determinant. It floors the diagonal instead of
raising on an indefinite matrix, because a tier-1 kernel cannot branch on
a value; the blocked `Tensor` overload in `numax.linalg.cholesky` raises
instead.

Neither tier pivots, and neither needs to: a symmetric positive definite
matrix does not require it. That is a theorem, not luck.
"""

from std.collections import Array

from ...core.numeric import FloatLike, guard_nonzero

from ..common import _PIVOT_FLOOR, _zeros
from .triangular import back_substitution, forward_substitution


def cholesky[T: FloatLike, n: Int](a: Array[T, n * n]) -> Array[T, n * n]:
    """The lower-triangular `L` with `L @ L.T = A`, for symmetric positive
    definite `A`.

    Only the lower triangle of `A` is read, so a caller holding just that
    half can leave the rest uninitialized. The returned upper triangle is
    zero.

    Diagonal entries are floored away from zero before the square root, so
    a matrix that isn't quite positive definite produces a finite (wrong)
    answer rather than a NaN that would then spread through every
    subsequent column. There is no error flag -- checking for one would be
    a per-lane branch.

    MAX ships no Cholesky, so past the crossover the answer is the
    `Tensor` overload of this name at the bottom of the file: blocked, with
    its trailing update in MAX's `matmul`. It also raises on a matrix that
    is not positive definite, which this one cannot.
    """
    var out = _zeros[T, n * n]()

    for j in range(n):
        var diagonal = a[j * n + j].copy()
        for k in range(j):
            var ljk = out[j * n + k].copy()
            diagonal = diagonal - (ljk * ljk)
        var ljj = guard_nonzero(diagonal, T.constant(_PIVOT_FLOOR)).sqrt()
        out[j * n + j] = ljj.copy()

        for i in range(j + 1, n):
            var total = a[i * n + j].copy()
            for k in range(j):
                total = total - (out[i * n + k] * out[j * n + k])
            out[i * n + j] = total / ljj

    return out^


def cholesky_solve[
    T: FloatLike, n: Int
](lower: Array[T, n * n], b: Array[T, n]) -> Array[T, n]:
    """Solve `A @ x = b` given `A`'s Cholesky factor `L`.

    Takes the factor rather than `A` because the point of a factorization
    is reusing it: a Gaussian process solves against the same `L` for every
    new right-hand side.

    No MAX equivalent exists to route to at any size (verified: MAX ships
    neither `cholesky` nor a triangular solve) -- this module's own
    register-resident version is the only one available regardless of `n`,
    plain-`dtype` or not.
    """
    var y = forward_substitution[T, n](lower, b)

    # `L.T` transposed on the fly rather than materialized -- the
    # substitution only ever reads `upper[i*n+j]` for `j >= i`, which is
    # `lower[j*n+i]`.
    var transposed = _zeros[T, n * n]()
    for i in range(n):
        for j in range(n):
            transposed[i * n + j] = lower[j * n + i].copy()

    return back_substitution[T, n](transposed, y)


def slogdet_cholesky[T: FloatLike, n: Int](lower: Array[T, n * n]) -> T:
    """`ln(det(A))` from `A`'s Cholesky factor: `2*sum(ln(diag(L)))`.

    Named for `numpy.linalg.slogdet`, which is the same quantity for a
    general matrix and returns it as `(sign, logabsdet)`. A Cholesky factor
    only exists for a positive-definite `A`, so the sign is always `+1` and
    only the logarithm is returned. The general `slogdet` over any square
    matrix belongs with the rest of the `scipy.linalg` depth in v0.2.

    The quantity a Gaussian process log-likelihood actually needs, and the
    reason to compute it this way rather than as `ln(det(A))`: for even a
    moderately large `n` the determinant itself overflows or underflows
    long before its logarithm becomes interesting.

    No MAX equivalent at any size -- it ships neither a Cholesky to take
    the factor from nor a log-determinant to compare against.
    """
    var total = T.constant(0.0)
    for i in range(n):
        total = total + lower[i * n + i].ln()
    return total * T.constant(2.0)
