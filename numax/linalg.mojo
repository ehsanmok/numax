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

## Use MAX past N (Track F, `parity.mdc`)

Every function in this module is register-resident and register-bound: an
`n x n` matrix is `n*n` values of `Array[T, n*n]`, so both compile time and
register pressure grow with `n`, and the naive triple-loop `matmul` this
module uses is the right algorithm at small `n` and the wrong one past it.
`bench/bench_matmul.mojo`'s measured crossover on an M3 Pro is `n = 8` for
a single matrix, `n = 16` against the 4-wide batched form, and by `n = 64`
`max.linalg.matmul` is ~130x faster -- see `matmul`'s own docstring below
and `docs/performance.md`'s "Use MAX past N" table for the numbers.

`max.linalg.matmul` and `max.linalg.qr_factorization` (both over
`TileTensor`, both CPU/GPU, both monomorphic in a raw `dtype`) are the only
two dense-linear-algebra primitives MAX itself ships (verified against
`~/workspace/modular/max/kernels/src/linalg/` -- there is no MAX
`lu`/`solve`/`det`/`trace`/`norm`/`inverse` at all, generic or otherwise).
So for `lu`/`solve`/`det`/`inverse`/`cholesky_solve` at a large, plain-`T`
`n`, there is no direct MAX function to call in this function's place --
the honest recommendation is to build the large-matrix equivalent from
`max.linalg.qr_factorization` (Householder QR is what LAPACK-style solvers
use for exactly this), not to expect a drop-in replacement. Each function's
own docstring below repeats this where it applies, so the note is visible
at the call site, not only here.
"""

from std.collections import Array

from .numeric import FloatLike, guard_nonzero

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
    """Solve `U @ x = b` for upper-triangular `U`."""
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
    instead (`prod(diag(R))` up to sign, from `max.linalg.qr_factorization`).
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
