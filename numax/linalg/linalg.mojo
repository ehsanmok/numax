"""Small dense linear algebra, `FloatLike`-generic and compile-time-sized.

MAX already ships `linalg.matmul` and `linalg.qr_factorization` over
`TileTensor`, both fast and both CPU/GPU. Neither is what this module is
for: they are monomorphic in a raw `dtype`, so a `Dual` cannot pass through
them, which makes them exactly as differentiable as a BLAS call. Anything
here that only needs raw speed on large matrices should use MAX's version
instead -- this module exists for the cases where the *type* matters.

A differentiable Cholesky is the concrete payoff. Gaussian process
marginal likelihoods, Kalman filter updates, and multivariate normal
densities all bottom out in `chol(A)` and its log-determinant, and all of
them need gradients with respect to the entries of `A`. Calling `cholesky`
at `Dual` here gives that with no adjoint rule written anywhere, because
the factorization is built from the same arithmetic every other `numax`
kernel is.

## Shape and storage

Matrices are `Array[T, n*n]` in row-major order, with `n` a compile-time
parameter, so an `n x n` matrix lives in registers rather than memory and
every loop bound is known at compile time. That's what keeps these
GPU-launchable and it's also what bounds their usefulness: this is for the
small matrices that appear *inside* a per-element kernel (a 3x3 covariance
per pixel, a 6x6 Jacobian per particle), not for factoring something large.

Each `T` may itself be a SIMD vector, so one call factors one matrix per
lane -- `n` is the matrix dimension, not the vector width.

## Scope: no pivoting

`lu` and `cholesky` do no pivoting, and `det`/`inverse`/`solve` inherit
that. This is the honest limitation of the whole module, in the same
category as `gammainc`'s large-`x` caveat rather than an oversight: partial
pivoting means choosing a row based on the magnitude of a value, which is a
per-lane data-dependent decision, and different SIMD lanes holding
different matrices would want different pivot orders. Nothing in
`FloatLike` can express that.

Consequences worth knowing:

- `cholesky` is unaffected in practice. Symmetric positive definite
  matrices don't need pivoting -- that's a theorem, not luck.
- `lu` fails on a matrix with a zero pivot even when the matrix is
  perfectly well-conditioned (`[[0,1],[1,0]]` is the standard example),
  and loses accuracy on one with a small pivot. Pivots are floored away
  from zero so the result is finite rather than NaN, but finite is not the
  same as correct.

For a general non-symmetric solve where you can't vouch for the pivots,
this module is the wrong tool.

## Use MAX past N (see `docs/parity.md`)

Every function in this module is register-resident and register-bound: an
`n x n` matrix is `n*n` values of `Array[T, n*n]`, so both compile time and
register pressure grow with `n`, and the naive triple-loop `matmul` this
module uses is the right algorithm at small `n` and the wrong one past it.
`bench/bench_matmul.mojo`'s measured crossover on an M3 Pro is `n = 8` for
a single matrix, `n = 16` against the 4-wide batched form, and by `n = 64`
`linalg.matmul` is ~130x faster -- see `matmul`'s own docstring below
and `docs/performance.md`'s "Use MAX past N" table for the numbers.

`linalg.matmul` and `linalg.qr_factorization` (both over
`TileTensor`, both CPU/GPU, both monomorphic in a raw `dtype`) are the only
two dense-linear-algebra primitives MAX itself ships (import root is the
top-level `linalg`, not `max.linalg` -- there is no such package; verified
against
`~/workspace/modular/max/kernels/src/linalg/` -- there is no MAX
`lu`/`solve`/`det`/`trace`/`norm`/`inverse` at all, generic or otherwise).
So for `lu`/`solve`/`det`/`inverse`/`cholesky_solve` at a large, plain-`T`
`n`, there is no direct MAX function to call in this function's place --
the honest recommendation is to build the large-matrix equivalent from
`linalg.qr_factorization` (Householder QR is what LAPACK-style solvers
use for exactly this), not to expect a drop-in replacement. Each function's
own docstring below repeats this where it applies, so the note is visible
at the call site, not only here.
"""

from std.collections import Array

from ..core.numeric import (
    FloatLike,
    ge_indicator,
    guard_nonzero,
    max_of,
    min_of,
)

comptime _PIVOT_FLOOR = 1e-30


def _zeros[T: FloatLike, size: Int]() -> Array[T, size]:
    return Array[T, size](fill=T.constant(0.0))


def matvec[
    T: FloatLike, n: Int
](a: Array[T, n * n], x: Array[T, n]) -> Array[T, n]:
    """`A @ x` for a row-major `n x n` matrix.

    For a large, plain-`dtype` `A` (past the crossover `matmul`'s own
    docstring documents), `max.linalg.matmul` still applies -- a
    matrix-vector product is a matrix-matrix product against an `n x 1`
    `TileTensor`, and MAX has no separate matvec-specific fast path to
    prefer over that.
    """
    var out = _zeros[T, n]()
    for i in range(n):
        var total = T.constant(0.0)
        for j in range(n):
            total = total + a[i * n + j] * x[j]
        out[i] = total^
    return out^


def matmul[
    T: FloatLike, n: Int
](a: Array[T, n * n], b: Array[T, n * n]) -> Array[T, n * n]:
    """`A @ B` for two row-major `n x n` matrices.

    The naive triple loop, which is the right algorithm at these sizes and
    the wrong one past them. Measured against `max.linalg.matmul` on an M3
    Pro (`bench/bench_matmul.mojo`), the crossover is at `n = 8` for a
    single matrix and `n = 16` for the 4-wide batched form; by `n = 64` MAX
    is ~130x faster. So call `max.linalg.matmul` directly for anything
    larger than about 8x8 whose entries are plain `dtype` values.

    What this version has instead: it is generic in `T`, so calling it at
    `Dual` differentiates the product and calling it at `Compensated` runs
    it at extra precision, neither of which a `dtype`-monomorphic kernel
    can do. And since the matrix is an `Array` in registers rather than a
    `TileTensor` in memory, it can be called from inside a single GPU
    thread -- one matrix per SIMD lane, if `T` is itself a vector.
    """
    var out = _zeros[T, n * n]()
    for i in range(n):
        for k in range(n):
            var aik = a[i * n + k].copy()
            for j in range(n):
                out[i * n + j] = out[i * n + j] + aik * b[k * n + j]
    return out^


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

    No MAX equivalent at any size: MAX ships no Cholesky. For a large,
    plain-`dtype` `A` the route is `linalg.qr_factorization`; see this
    module's "Use MAX past N" section.
    """
    var out = _zeros[T, n * n]()

    for j in range(n):
        var diagonal = a[j * n + j].copy()
        for k in range(j):
            var ljk = out[j * n + k].copy()
            diagonal = diagonal + (-(ljk * ljk))
        var ljj = guard_nonzero(diagonal, T.constant(_PIVOT_FLOOR)).sqrt()
        out[j * n + j] = ljj.copy()

        for i in range(j + 1, n):
            var total = a[i * n + j].copy()
            for k in range(j):
                total = total + (-(out[i * n + k] * out[j * n + k]))
            out[i * n + j] = total / ljj

    return out^


def lu[T: FloatLike, n: Int](a: Array[T, n * n]) -> Array[T, n * n]:
    """Doolittle `LU` without pivoting, packed into one matrix.

    The strict lower triangle holds `L` (whose diagonal is an implicit
    `1`), and the upper triangle including the diagonal holds `U`. Packing
    them avoids returning two matrices where the two halves never overlap.

    See this module's docstring for what "without pivoting" costs.

    No MAX equivalent exists to route to for a large, plain-`dtype` `A`
    (verified: MAX ships no `lu` at any size). `max.linalg.qr_factorization`
    is the large-matrix building block LAPACK-style solvers use in `lu`'s
    place; see this module's own "Use MAX past N" section.
    """
    var out = _zeros[T, n * n]()
    for i in range(n * n):
        out[i] = a[i].copy()

    for k in range(n):
        var pivot = guard_nonzero(out[k * n + k], T.constant(_PIVOT_FLOOR))
        for i in range(k + 1, n):
            var factor = out[i * n + k] / pivot
            out[i * n + k] = factor.copy()
            for j in range(k + 1, n):
                out[i * n + j] = out[i * n + j] + (-(factor * out[k * n + j]))

    return out^


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
            total = total + (-(lower[i * n + j] * x[j]))
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
            total = total + (-(upper[i * n + j] * x[j]))
        x[i] = total / guard_nonzero(upper[i * n + i], T.constant(_PIVOT_FLOOR))
    return x^


def solve[
    T: FloatLike, n: Int
](a: Array[T, n * n], b: Array[T, n]) -> Array[T, n]:
    """Solve `A @ x = b` by unpivoted LU followed by two substitutions.

    No MAX equivalent exists to route to for a large, plain-`dtype`
    system (verified: MAX ships no `solve` at any size) -- see this
    module's own "Use MAX past N" section for what to build the
    large-matrix case from instead.
    """
    var factored = lu[T, n](a)
    var y = forward_substitution[T, n, unit_diagonal=True](factored, b)
    return back_substitution[T, n](factored, y)


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


def det[T: FloatLike, n: Int](a: Array[T, n * n]) -> T:
    """The determinant, as the product of the unpivoted LU's diagonal.

    Unpivoted, so the sign is always the product's own -- there is no row
    swap count to correct for.

    No MAX equivalent exists to route to for a large, plain-`dtype` `A`
    (verified: MAX ships no `det` at any size) -- see this module's own
    "Use MAX past N" section for what to build the large-matrix case from
    instead (`prod(diag(R))` up to sign, from `linalg.qr_factorization`).
    """
    var factored = lu[T, n](a)
    var product = T.one()
    for i in range(n):
        product = product * factored[i * n + i]
    return product^


def log_det_from_cholesky[T: FloatLike, n: Int](lower: Array[T, n * n]) -> T:
    """`ln(det(A))` from `A`'s Cholesky factor: `2*sum(ln(diag(L)))`.

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


def inverse[T: FloatLike, n: Int](a: Array[T, n * n]) -> Array[T, n * n]:
    """`A^-1`, by solving against each column of the identity.

    Factors once and substitutes `n` times rather than calling `solve` `n`
    times, which would redo the factorization for every column.

    No MAX equivalent exists to route to for a large, plain-`dtype` `A`
    (verified: MAX ships no `inv` at any size) -- see this module's own
    "Use MAX past N" section. Note that inverting explicitly is rarely the
    right move at any size; solving against a specific right-hand side
    (`solve`/`cholesky_solve`) is both cheaper and better-conditioned than
    forming `A^-1` and multiplying, the same trade-off that holds for
    `numax`'s own register-resident version.
    """
    var factored = lu[T, n](a)
    var out = _zeros[T, n * n]()

    for column in range(n):
        var e = _zeros[T, n]()
        e[column] = T.one()
        var y = forward_substitution[T, n, unit_diagonal=True](factored, e)
        var x = back_substitution[T, n](factored, y)
        for i in range(n):
            out[i * n + column] = x[i].copy()

    return out^


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

    No MAX equivalent at any size, and unlike the dense routines here there
    is no large-`n` MAX primitive to build one from either -- Thomas is
    already linear, so there is nothing for `linalg.matmul` or
    `linalg.qr_factorization` to improve on.

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
            diag[i] + (-(sub[i] * c_prime[i - 1])),
            T.constant(_PIVOT_FLOOR),
        )
        c_prime[i] = sup[i] / denominator
        d_prime[i] = (rhs[i] + (-(sub[i] * d_prime[i - 1]))) / denominator

    var x = _zeros[T, n]()
    x[n - 1] = d_prime[n - 1].copy()
    for step in range(1, n):
        var i = n - 1 - step
        x[i] = d_prime[i] + (-(c_prime[i] * x[i + 1]))

    return x^


def trace[T: FloatLike, n: Int](a: Array[T, n * n]) -> T:
    """The sum of the diagonal entries of `A`.

    No MAX equivalent exists to route to at any size -- MAX ships no
    `trace`, and there is nothing to build one from beyond this loop, which
    is already bandwidth-bound at every `n` this module handles.
    """
    var total = T.constant(0.0)
    for i in range(n):
        total = total + a[i * n + i]
    return total^


def norm_frobenius[T: FloatLike, n: Int](a: Array[T, n * n]) -> T:
    """`sqrt(sum(A[i,j]**2))` -- the Frobenius (entrywise 2-) norm.

    Summed directly rather than in a scaled/squared form, which means a
    matrix whose entries are near the square root of `dtype`'s overflow
    threshold will overflow here. LAPACK's `lange` rescales to avoid that;
    doing the same would need a data-dependent branch on the running
    maximum, which the fixed-iteration invariant rules out. Call this at
    `Compensated` if the summation length is what worries you, or scale `A`
    yourself if its magnitude is.

    No MAX equivalent at any size: MAX ships no norm of any kind.
    """
    var total = T.constant(0.0)
    for i in range(n * n):
        total = total + a[i] * a[i]
    return total.sqrt()


def norm_1[T: FloatLike, n: Int](a: Array[T, n * n]) -> T:
    """The induced 1-norm: the largest absolute column sum.

    The column maximum is taken with `max_of`, not an `if` -- `T` may hold
    a SIMD vector whose lanes disagree about which column is largest, so
    the running maximum has to be arithmetic. Same reason every other
    selection in `numax` is branchless.

    No MAX equivalent at any size.
    """
    var best = T.constant(0.0)
    for j in range(n):
        var column = T.constant(0.0)
        for i in range(n):
            column = column + a[i * n + j].abs()
        best = max_of(best, column)
    return best^


def norm_inf[T: FloatLike, n: Int](a: Array[T, n * n]) -> T:
    """The induced infinity-norm: the largest absolute row sum. `norm_1` of
    the transpose, computed without forming it.

    No MAX equivalent at any size.
    """
    var best = T.constant(0.0)
    for i in range(n):
        var row = T.constant(0.0)
        for j in range(n):
            row = row + a[i * n + j].abs()
        best = max_of(best, row)
    return best^


def qr[
    T: FloatLike, n: Int
](a: Array[T, n * n]) -> Tuple[Array[T, n * n], Array[T, n * n]]:
    """Householder `QR`: returns `(Q, R)` with `Q @ R = A`, `Q` orthogonal
    and `R` upper triangular.

    `Q` is formed explicitly rather than left as a product of reflectors.
    That is the wasteful choice at large `n` -- LAPACK returns the
    reflectors precisely so callers can apply them without materializing
    `Q` -- but at the sizes this module is for, a caller that wanted the
    factored form would be better served by MAX's version anyway (below),
    and an explicit `Q` is what makes `qr` usable as one line.

    The reflector sign is chosen as `-sign(A[k,k]) * ||x||` via `copysign`,
    the standard choice: it makes the subtraction that forms `v` add
    magnitudes rather than cancel them, so the reflector stays
    well-conditioned when `A[k,k]` already dominates its column. Being
    `copysign` rather than a branch, it also works lane-wise on a SIMD `T`.

    Fixed iteration count (`n - 1` reflectors, `n` comptime), so this stays
    launchable inside a GPU thread like everything else here.

    **The one function in this module MAX has a direct counterpart for.**
    `linalg.qr_factorization` (`~/workspace/modular/max/kernels/src/linalg/qr_factorization.mojo`)
    is a LAPACK-style in-place Householder factorization over
    `LayoutTensor`, and is the thing to call for a large, plain-`dtype`
    `A` -- with two caveats worth knowing before switching: it is
    monomorphic in `dtype` (so no `Dual` passes through it, which is the
    whole reason this version exists), and it is a CPU-only scalar-loop
    reference implementation rather than a tuned kernel. It also returns
    the reflectors plus a `sigma` vector, not an explicit `Q`; `apply_q`
    and `form_q` alongside it are how you get `Q`'s action or `Q` itself.
    """
    var r = _zeros[T, n * n]()
    for i in range(n * n):
        r[i] = a[i].copy()

    var q = _zeros[T, n * n]()
    for i in range(n):
        q[i * n + i] = T.one()

    for k in range(n - 1):
        # ||x|| over the sub-column A[k:, k].
        var norm_sq = T.constant(0.0)
        for i in range(k, n):
            norm_sq = norm_sq + r[i * n + k] * r[i * n + k]
        var alpha = norm_sq.sqrt().copysign(-r[k * n + k])

        # v = x - alpha*e1, then vv = v.v. A column already in reflected
        # form gives vv = 0; `guard_nonzero` keeps the division finite
        # rather than producing a NaN that would spread into every later
        # column, the same guard `cholesky` and `lu` use on their pivots.
        var v = _zeros[T, n]()
        for i in range(k, n):
            v[i] = r[i * n + k].copy()
        v[k] = v[k] + (-alpha)

        var vv = T.constant(0.0)
        for i in range(k, n):
            vv = vv + v[i] * v[i]
        var scale = T.constant(2.0) / guard_nonzero(
            vv, T.constant(_PIVOT_FLOOR)
        )

        # R <- (I - scale*v v^T) R, columns k..n-1 only (the rest are zero
        # below the diagonal already).
        for j in range(k, n):
            var vr = T.constant(0.0)
            for i in range(k, n):
                vr = vr + v[i] * r[i * n + j]
            var factor = vr * scale
            for i in range(k, n):
                r[i * n + j] = r[i * n + j] + (-(factor * v[i]))

        # Q <- Q (I - scale*v v^T), accumulating the reflectors' product.
        for i in range(n):
            var qv = T.constant(0.0)
            for j in range(k, n):
                qv = qv + q[i * n + j] * v[j]
            var factor = qv * scale
            for j in range(k, n):
                q[i * n + j] = q[i * n + j] + (-(factor * v[j]))

    # The strict lower triangle holds the reflector residue, not part of R.
    for i in range(n):
        for j in range(i):
            r[i * n + j] = T.constant(0.0)

    return (r^, q^)


def dot[T: FloatLike, n: Int](a: Array[T, n], b: Array[T, n]) -> T:
    """The inner product `sum(a[i] * b[i])` -- BLAS-1 `dot`.

    Summed in order rather than pairwise, so the rounding is the obvious
    one and a caller who cares can recover the lost bits by calling this at
    `Compensated` instead of `Plain`. That is the whole reason a `dot` this
    small is worth writing: MAX ships no BLAS-1 at all, and no BLAS
    anywhere is generic over its scalar type.
    """
    var total = T.constant(0.0)
    for i in range(n):
        total = total + a[i] * b[i]
    return total^


def nrm2[T: FloatLike, n: Int](a: Array[T, n]) -> T:
    """The Euclidean norm `sqrt(sum(a[i]**2))` -- BLAS-1 `nrm2`.

    Not rescaled, so a vector whose entries approach the square root of
    `dtype`'s overflow threshold will overflow here where LAPACK's `nrm2`
    would not. Rescaling needs a running maximum and a data-dependent
    branch, which the fixed-iteration invariant rules out; the same
    trade-off `norm_frobenius` documents.
    """
    return dot[T, n](a, a).sqrt()


def outer[
    T: FloatLike, n: Int
](a: Array[T, n], b: Array[T, n]) -> Array[T, n * n]:
    """The outer product `out[i, j] = a[i] * b[j]`, row-major.

    The rank-1 update every quasi-Newton method and every Householder
    reflector is built from. MAX has `outer_product_acc`, but only on the
    older `LayoutTensor` and only accumulating into an existing matrix.
    """
    var out = _zeros[T, n * n]()
    for i in range(n):
        for j in range(n):
            out[i * n + j] = a[i] * b[j]
    return out^


def _jacobi_rotation[T: FloatLike](numerator: T, denominator: T) -> Tuple[T, T]:
    """The `(cosine, sine)` of the Jacobi rotation that annihilates an
    off-diagonal entry, computed without a branch.

    Given the standard `zeta = denominator / (2 * numerator)`, the
    numerically stable form of the tangent is
    `t = sign(zeta) / (|zeta| + sqrt(1 + zeta**2))` -- chosen over the
    algebraically equivalent `t = -zeta + sqrt(1 + zeta**2)` because the
    latter cancels catastrophically for large `|zeta|`, which is exactly
    the common case near convergence.

    `numerator` is the off-diagonal entry being annihilated. When it is
    already zero the rotation should be the identity, and `guard_nonzero`
    delivers that without an `if`: a floored denominator makes `zeta` huge,
    which makes `t` about `1/(2*zeta)`, which is about zero, which makes
    `(c, s) = (1, 0)`. A branch here would be a per-lane branch, since two
    SIMD lanes can disagree about whether their entry has converged.
    """
    var safe = guard_nonzero(numerator, T.constant(_PIVOT_FLOOR))
    var zeta = denominator / (T.constant(2.0) * safe)
    var magnitude = zeta.abs()
    var tangent = T.one().copysign(zeta) / (
        magnitude + (T.one() + zeta * zeta).sqrt()
    )
    var cosine = T.one() / (T.one() + tangent * tangent).sqrt()
    var sine = tangent * cosine
    return (cosine^, sine^)


def eigh[
    T: FloatLike, n: Int, sweeps: Int = 12
](a: Array[T, n * n]) -> Tuple[Array[T, n], Array[T, n * n]]:
    """Eigenvalues and eigenvectors of a *symmetric* matrix, by cyclic
    Jacobi rotations. Returns `(eigenvalues, eigenvectors)` with the
    eigenvectors as the columns of the second matrix.

    Only the symmetric part is meaningful: the algorithm reads the whole
    matrix but drives it toward diagonal by symmetric similarity
    transforms, so a non-symmetric input gives the eigen-decomposition of
    nothing in particular. There is no symmetry check, because checking
    would be a per-lane branch.

    **Fixed sweeps, so this stays tier 1 and GPU-launchable.** A
    convergence-tested Jacobi would stop when the off-diagonal norm fell
    below a tolerance; this does `sweeps` full passes over all `n*(n-1)/2`
    pairs regardless. Cyclic Jacobi converges quadratically once the
    off-diagonal entries are small, so 12 sweeps is far more than enough
    for the sizes this module handles -- but it is a fixed amount of work,
    not a guarantee, and a pathological matrix can leave residue. Raise
    `sweeps` if a residual check says so.

    **Eigenvalues come out in no particular order.** NumPy's `eigh` sorts
    them ascending; sorting is data-dependent, so it cannot happen inside a
    tier-1 kernel. Sort them yourself afterward if the order matters --
    `numax.stats` has the `Plain`-only machinery for it.

    Differentiable, like everything else here: called at `Dual`, the
    eigenvalues carry their derivatives with respect to whatever the matrix
    entries were seeded from. That is not true of any LAPACK-backed
    `eigh`.
    """
    # `work` is driven toward diagonal; `vectors` accumulates the rotations.
    var work = _zeros[T, n * n]()
    for i in range(n * n):
        work[i] = a[i].copy()
    var vectors = _zeros[T, n * n]()
    for i in range(n):
        vectors[i * n + i] = T.one()

    for _ in range(sweeps):
        for p in range(n - 1):
            for q in range(p + 1, n):
                var apq = work[p * n + q].copy()
                var app = work[p * n + p].copy()
                var aqq = work[q * n + q].copy()
                var rotation = _jacobi_rotation[T](apq, aqq + (-app))
                var c = rotation[0].copy()
                var s = rotation[1].copy()

                # Rows p and q.
                for k in range(n):
                    var akp = work[p * n + k].copy()
                    var akq = work[q * n + k].copy()
                    work[p * n + k] = c * akp + (-(s * akq))
                    work[q * n + k] = s * akp + c * akq
                # Columns p and q, completing the similarity transform.
                for k in range(n):
                    var apk = work[k * n + p].copy()
                    var aqk = work[k * n + q].copy()
                    work[k * n + p] = c * apk + (-(s * aqk))
                    work[k * n + q] = s * apk + c * aqk
                # The same rotation applied to the accumulating basis.
                for k in range(n):
                    var vkp = vectors[k * n + p].copy()
                    var vkq = vectors[k * n + q].copy()
                    vectors[k * n + p] = c * vkp + (-(s * vkq))
                    vectors[k * n + q] = s * vkp + c * vkq

    var values = _zeros[T, n]()
    for i in range(n):
        values[i] = work[i * n + i].copy()
    return (values^, vectors^)


def svd[
    T: FloatLike, n: Int, sweeps: Int = 12
](a: Array[T, n * n]) -> Tuple[Array[T, n * n], Array[T, n], Array[T, n * n]]:
    """The singular value decomposition of a square matrix: returns
    `(U, singular_values, V)` with `A = U @ diag(s) @ V.T`.

    One-sided Jacobi: rotate pairs of *columns* of `A` until they are
    mutually orthogonal, accumulating the rotations into `V`. The column
    norms are then the singular values and the normalized columns are `U`.
    Chosen over forming `A.T @ A` and calling `eigh`, which squares the
    condition number and loses half the significant digits of the small
    singular values -- the classic mistake, and the reason one-sided
    Jacobi exists.

    **Fixed sweeps, tier 1**, on the same terms as `eigh`: `sweeps` full
    passes over all column pairs, no convergence test, GPU-launchable.

    **Singular values are unordered**, unlike `numpy.linalg.svd`'s
    descending convention, for the same reason `eigh`'s eigenvalues are:
    sorting is data-dependent. They are all non-negative.

    Square only. A rectangular SVD needs two size parameters throughout
    and a decision about thin-vs-full factors; nothing in `numax` needs one
    yet, and doing it half-way would be worse than not doing it.
    """
    # Columns of `work` get orthogonalized; `v` accumulates the rotations.
    var work = _zeros[T, n * n]()
    for i in range(n * n):
        work[i] = a[i].copy()
    var v = _zeros[T, n * n]()
    for i in range(n):
        v[i * n + i] = T.one()

    for _ in range(sweeps):
        for p in range(n - 1):
            for q in range(p + 1, n):
                # Gram matrix of the two columns.
                var alpha = T.constant(0.0)
                var beta = T.constant(0.0)
                var gamma = T.constant(0.0)
                for i in range(n):
                    var ip = work[i * n + p].copy()
                    var iq = work[i * n + q].copy()
                    alpha = alpha + ip * ip
                    beta = beta + iq * iq
                    gamma = gamma + ip * iq

                var rotation = _jacobi_rotation[T](gamma, beta + (-alpha))
                var c = rotation[0].copy()
                var s = rotation[1].copy()

                for i in range(n):
                    var ip = work[i * n + p].copy()
                    var iq = work[i * n + q].copy()
                    work[i * n + p] = c * ip + (-(s * iq))
                    work[i * n + q] = s * ip + c * iq
                for i in range(n):
                    var ip = v[i * n + p].copy()
                    var iq = v[i * n + q].copy()
                    v[i * n + p] = c * ip + (-(s * iq))
                    v[i * n + q] = s * ip + c * iq

    var values = _zeros[T, n]()
    var u = _zeros[T, n * n]()
    for j in range(n):
        var norm_sq = T.constant(0.0)
        for i in range(n):
            norm_sq = norm_sq + work[i * n + j] * work[i * n + j]
        var sigma = norm_sq.sqrt()
        values[j] = sigma.copy()
        # A zero singular value leaves its column direction undetermined;
        # the guard makes it come out as zero rather than NaN, which keeps
        # a rank-deficient input usable instead of poisoning `U`.
        var safe = guard_nonzero(sigma, T.constant(_PIVOT_FLOOR))
        for i in range(n):
            u[i * n + j] = work[i * n + j] / safe

    return (u^, values^, v^)


def pinv[
    T: FloatLike, n: Int, sweeps: Int = 12
](a: Array[T, n * n], rcond: Float64 = 1e-12) -> Array[T, n * n]:
    """The Moore-Penrose pseudoinverse, `V @ diag(1/s) @ U.T`, with small
    singular values truncated.

    This is what to reach for instead of `inverse` when the matrix might be
    singular or nearly so: `inverse` solves against the identity through an
    unpivoted LU and will happily return enormous garbage for a
    near-singular input, whereas here a singular direction contributes
    nothing rather than dominating.

    `rcond` is relative to the largest singular value, matching
    `numpy.linalg.pinv`. The truncation is arithmetic, not a branch: each
    reciprocal is multiplied by a `0`/`1` indicator built from
    `ge_indicator`, so no lane decides anything for another lane.
    """
    var factored = svd[T, n, sweeps](a)
    var u = factored[0].copy()
    var values = factored[1].copy()
    var v = factored[2].copy()

    var largest = T.constant(0.0)
    for i in range(n):
        largest = max_of(largest, values[i])
    var threshold = largest * T.constant(rcond)

    # Reciprocal where the value clears the threshold, zero where it does
    # not -- as a multiply by an indicator rather than a branch.
    var inverted = _zeros[T, n]()
    for i in range(n):
        var keep = ge_indicator(values[i], threshold)
        var safe = guard_nonzero(values[i], T.constant(_PIVOT_FLOOR))
        inverted[i] = keep / safe
    var out = _zeros[T, n * n]()
    for i in range(n):
        for j in range(n):
            var total = T.constant(0.0)
            for k in range(n):
                total = total + v[i * n + k] * inverted[k] * u[j * n + k]
            out[i * n + j] = total^
    return out^


def cond[T: FloatLike, n: Int, sweeps: Int = 12](a: Array[T, n * n]) -> T:
    """The 2-norm condition number: the ratio of largest to smallest
    singular value.

    The number that says how much a solve can amplify input error -- a
    `cond` of `1e12` at float64 means about four significant digits survive.
    Worth computing before trusting `solve` or `inverse` on a matrix of
    unknown provenance.

    A singular matrix has a zero smallest singular value and an infinite
    condition number; the floor in the division reports a very large finite
    number instead, since returning an infinity from a branchless kernel
    would need the branch this avoids.
    """
    var values = svd[T, n, sweeps](a)[1].copy()
    var largest = T.constant(0.0)
    var smallest = values[0].copy()
    for i in range(n):
        largest = max_of(largest, values[i])
        smallest = min_of(smallest, values[i])
    return largest / guard_nonzero(smallest, T.constant(_PIVOT_FLOOR))
