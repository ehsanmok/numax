"""Cholesky factorization and the solves that reuse it.
`scipy.linalg`'s `cholesky`/`cho_solve` pair.

**Two tiers under one name.** MAX ships no Cholesky at any size -- its only
factorization is `qr_factorization`, on the older `LayoutTensor`, which
numax denies rather than bridges -- so both tiers are numax's.

Over `Tensor`, `cholesky` is right-looking and blocked: each step factors
one `block x block` diagonal panel, solves the panel below it, then
subtracts `L21 @ L21.T` from what remains. That subtraction is the entire
cubic cost and it is a matrix product, so it goes to `blas.matmul` and
inherits MAX's dispatch. Tier 2, host-orchestrated, and it raises on a
matrix that is not positive definite.

Over `Array[T, n*n]`, `cholesky` is tier 1 and differentiates: calling it
at `Dual` gives the derivative of a factorization with no adjoint rule
written anywhere, which is the concrete payoff of the conformer layer --
Gaussian process marginal likelihoods, Kalman updates and multivariate
normal densities all bottom out in `chol(A)` and its log-determinant. It
floors the diagonal instead of raising, because a tier-1 kernel cannot
branch on a value.

Neither tier pivots, and neither needs to: a symmetric positive definite
matrix does not require it. That is a theorem, not luck.
"""

from std.collections import Array
from std.math import sqrt

from ..core.array import Shaped
from ..core.numeric import FloatLike, guard_nonzero

from .blas import matmul
from .common import _PIVOT_FLOOR, _staged, _zeros
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


def cholesky[
    dtype: DType, n: Int, gpu: Bool = False, block: Int = 64
](mut a: Shaped[dtype, n, n]) raises -> Shaped[
    dtype, n, n
] where dtype.is_floating_point():
    """**Tier 2.** The lower-triangular `L` with `L @ L.T == a`, blocked.

    MAX has no Cholesky -- `qr_factorization` is its only factorization,
    and that one is on the older `LayoutTensor` -- so this is numax filling
    a gap rather than calling into one. What it does not do is fill it from
    scratch: this is the right-looking blocked algorithm, so each step
    factors one `block x block` diagonal panel, solves the panel below it,
    and then subtracts `L21 @ L21.T` from everything remaining. That last
    subtraction is the whole cubic cost of a Cholesky, and it is a matrix
    product, so it goes to MAX's `matmul` and inherits its dispatch. The
    panel is `O(n * block^2)` and the GEMM is `O(n^3)`.

    **Ceiling.** The panel factorization and the block staging run on the
    host: the trailing submatrix of a row-major tensor is not contiguous,
    and numax has no strided device sub-view to hand MAX yet, so each step
    copies its panel down and its update back. The flops are MAX's, the
    copies are not, which puts an `O(n^2)` band on what the GPU path can
    win per step. The upgrade is a device-resident tiling that slices
    `TileTensor` in place and calls MAX on the slice; the algorithm above
    does not change when that lands, only `_staged` disappears.

    `block` is a parameter so a caller can tune it or set it to `n` to get
    the unblocked algorithm back. No pivoting, and none is needed: a
    symmetric positive definite matrix does not require it.

    Raises when a diagonal entry comes out non-positive, which is what a
    matrix that is not positive definite looks like from in here. The
    sibling `cholesky` over `Array[T, n*n]` floors the diagonal instead and
    returns something finite, because a tier-1 kernel cannot branch on a
    value; this one is host-side and can afford to tell the truth.
    """
    var ctx = a.context()
    var work = a.to_host()
    var k = 0

    while k < n:
        var nb = min(block, n - k)

        # Panel: the diagonal block, unblocked. Contributions from earlier
        # blocks are already gone, subtracted by their trailing update, so
        # the inner sums start at `k` rather than at zero.
        for j in range(nb):
            var col = k + j
            var diagonal = work[col * n + col]
            for p in range(k, col):
                diagonal -= work[col * n + p] * work[col * n + p]
            if diagonal <= 0:
                raise Error(
                    "cholesky: matrix is not positive definite (pivot ",
                    Float64(diagonal),
                    " at index ",
                    col,
                    ")",
                )
            var root = sqrt(diagonal)
            work[col * n + col] = root
            for i in range(j + 1, nb):
                var row = k + i
                var entry = work[row * n + col]
                for p in range(k, col):
                    entry -= work[row * n + p] * work[col * n + p]
                work[row * n + col] = entry / root

        # Panel: everything below the diagonal block, solved against it.
        for row in range(k + nb, n):
            for j in range(nb):
                var col = k + j
                var entry = work[row * n + col]
                for p in range(k, col):
                    entry -= work[row * n + p] * work[col * n + p]
                work[row * n + col] = entry / work[col * n + col]

        # Trailing update: A22 -= L21 @ L21.T, which is MAX's.
        var m = n - k - nb
        if m > 0:
            var lower = List[Scalar[dtype]](capacity=m * nb)
            var lower_t = List[Scalar[dtype]](capacity=nb * m)
            for i in range(m):
                for j in range(nb):
                    lower.append(work[(k + nb + i) * n + k + j])
            for j in range(nb):
                for i in range(m):
                    lower_t.append(work[(k + nb + i) * n + k + j])

            var left = _staged[dtype](lower^, m, nb, ctx)
            var right = _staged[dtype](lower_t^, nb, m, ctx)
            var update = matmul[dtype, gpu](left, right).to_host()

            for i in range(m):
                for j in range(i + 1):
                    work[(k + nb + i) * n + k + nb + j] -= update[i * m + j]

        k += nb

    for row in range(n):
        for col in range(row + 1, n):
            work[row * n + col] = Scalar[dtype](0)

    return Shaped[dtype, n, n](ctx, work^)


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
