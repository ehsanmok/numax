"""Dense linear algebra: a `FloatLike` tier over `Array`, a MAX tier over
`Tensor`.

**The `Array[T, n*n]` half is tier 1**, except `lu_factor`/`lu_solve`/`det_lu`,
which declare tier 2 in their own docstrings. Every tier-1 factorization here
runs a fixed number of passes -- `n - 1` reflectors, `sweeps` Jacobi sweeps
-- and selects branchlessly, so all of it launches inside a GPU thread.
That is also what forecloses pivoting; see "Scope: no pivoting" below.

**The `Tensor` half, at the bottom of the file, goes through MAX.**
`matmul`, `matvec` and `batched_matmul` are MAX kernels outright: numax
allocates a destination, hands over two `TileTensor` views and waits.
`linalg.matmul` alone is a dispatch tree covering Apple
simdgroup, SM100, SM90, Ampere/CDNA, vendor cuBLAS/rocBLAS/hipBLASLt and
AMD RDNA, so numax names no architecture.

`cholesky` is the other kind: MAX has no Cholesky, so numax writes it, but
writes it *blocked* so that the cubic term is a matrix product and goes
back to MAX. That is the pattern the remaining factorizations follow as
they land -- panel on the host, `O(n^3)` in MAX's GEMM -- and each names
its own ceiling.

These entry points are `dtype`-monomorphic, which is exactly why the
`Array` half exists beside them rather than being replaced by them.

The two halves share names deliberately. `matmul(a, b)` picks by argument
type: an `Array[T, n*n]` factors one small matrix per SIMD lane inside a
kernel, a `Tensor` goes to MAX. They live in one module because Mojo wants
a single owner per name -- importing one name from two modules is
deprecated -- so a name that exists at both tiers is defined once, here.

`Tensor` in, `Tensor` out on the MAX half. `TileTensor` appears only as the
`.view()` at a MAX call site; it is the interop type, not something a
caller of this module passes.

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
that. This is the honest limitation of the tier-1 half of the module, in
the same category as `gammainc`'s large-`x` caveat rather than an
oversight: partial
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

For a general non-symmetric solve where you can't vouch for the pivots, the
tier-2 `lu_factor`/`lu_solve`/`det_lu` pivot properly. They give up the GPU
and the generic `T` to do it -- they are `Plain[dtype, 1]` at width 1, since
a SIMD lane holding a different matrix would want a different pivot order
-- which is the whole trade, stated once here and again at each of them.

## Cross the tiers past N (see `docs/parity.md`)

Every `Array` function here is register-resident and register-bound: an
`n x n` matrix is `n*n` values of `Array[T, n*n]`, so both compile time and
register pressure grow with `n`, and the naive triple-loop `matmul` the
`Array` tier uses is the right algorithm at small `n` and the wrong one
past it. `bench/bench_matmul.mojo`'s measured crossover on an M3 Pro is
`n = 8` for a single matrix, `n = 16` against the 4-wide batched form, and
by `n = 64` MAX is ~130x faster -- see `docs/performance.md`'s "Use MAX
past N" table.

Past that crossover the answer is now in this same module: `to_tensor` the
data and call the `Tensor` overload, which is MAX's kernel. `to_array` is
the way back.

What MAX does *not* ship is the rest of `scipy.linalg`. There is no MAX
`lu`, `solve`, `det`, `trace`, `norm`, `inverse`, Cholesky, SVD, eigen or
triangular solve -- verified against `stable` and
`~/workspace/modular-oss/max/kernels/src/linalg/`. MAX's one factorization,
`qr_factorization`, is on the older `LayoutTensor` and is a CPU-only scalar
loop, so numax denies it rather than bridging to it. Those gaps are numax's
to fill, written in MAX's idiom on `TileTensor` so the fill is upstreamable.
"""

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from linalg.bmm import batched_matmul as _max_batched_matmul
from linalg.matmul import matmul as _max_matmul
from max.gpu.host import DeviceContext
from std.collections import Array
from std.math import sqrt

from ..core.array import Dynamic, Shaped, zeros_dyn
from ..core.numeric import (
    FloatLike,
    blend,
    ge_indicator,
    guard_nonzero,
    max_of,
    min_of,
)
from ..core.complex import Complex
from ..core.plain import Plain

comptime _PIVOT_FLOOR = 1e-30


def _zeros[T: FloatLike, size: Int]() -> Array[T, size]:
    return Array[T, size](fill=T.constant(0.0))


def matvec[
    T: FloatLike, n: Int
](a: Array[T, n * n], x: Array[T, n]) -> Array[T, n]:
    """`A @ x` for a row-major `n x n` matrix.

    For a large, plain-`dtype` `A` (past the crossover `matmul`'s own
    docstring documents), MAX's `linalg.matmul` still applies -- a
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
    the wrong one past them. Measured against MAX's `linalg.matmul` on an M3
    Pro (`bench/bench_matmul.mojo`), the crossover is at `n = 8` for a
    single matrix and `n = 16` for the 4-wide batched form; by `n = 64` MAX
    is ~130x faster. So call `linalg.matmul` directly for anything
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
            diagonal = diagonal - (ljk * ljk)
        var ljj = guard_nonzero(diagonal, T.constant(_PIVOT_FLOOR)).sqrt()
        out[j * n + j] = ljj.copy()

        for i in range(j + 1, n):
            var total = a[i * n + j].copy()
            for k in range(j):
                total = total - (out[i * n + k] * out[j * n + k])
            out[i * n + j] = total / ljj

    return out^


def lu[T: FloatLike, n: Int](a: Array[T, n * n]) -> Array[T, n * n]:
    """Doolittle `LU` without pivoting, packed into one matrix.

    The strict lower triangle holds `L` (whose diagonal is an implicit
    `1`), and the upper triangle including the diagonal holds `U`. Packing
    them avoids returning two matrices where the two halves never overlap.

    See this module's docstring for what "without pivoting" costs.

    No MAX equivalent exists to route to for a large, plain-`dtype` `A`
    (verified: MAX ships no `lu` at any size). `linalg.qr_factorization`
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
                out[i * n + j] = out[i * n + j] - (factor * out[k * n + j])

    return out^


@fieldwise_init
struct PivotedLU[dtype: DType, n: Int](Movable where dtype.is_floating_point()):
    """**Tier 2.** An `LU` factorization with partial pivoting, and the
    solves that reuse it. `scipy.linalg`'s `lu_factor`/`lu_solve` pair.

    The answer to this module's "Scope: no pivoting" limitation, and the
    reason it is separate from `lu` rather than an improvement to it:
    choosing a row by the magnitude of a value is a data-dependent branch,
    so this cannot be tier 1 and cannot run inside a GPU thread. It is
    `Plain[dtype, 1]` at width 1 for the same reason -- a SIMD `T` holds
    several matrices whose lanes would want different pivot orders, and
    there is no single order to pick.

    What it buys is correctness where `lu` has none. The exchange matrix
    `[[0, 1], [1, 0]]` is perfectly well conditioned with a determinant of
    `-1`, and unpivoted `lu` cannot start on it at all; this factors it.
    Accuracy on a small-pivot matrix improves for the same reason.

    Rank deficiency is still not detected: a singular matrix factors to a
    zero pivot, which is floored rather than reported, so the result is
    finite and wrong. `cond` on the original matrix is the check.
    """

    var factored: Array[Plain[Self.dtype, 1], Self.n * Self.n]
    """`L` below the diagonal (its own diagonal an implicit `1`) and `U` on
    and above it, packed the way `lu` packs them."""

    var permutation: Array[Int, Self.n]
    """`permutation[i]` is the row of `A` that became row `i`."""

    var sign: Int
    """`+1` or `-1`, by the parity of the row swaps. `det` needs it; a
    determinant read off the diagonal alone would be wrong by that factor.
    """

    def solve(
        self, b: Array[Plain[Self.dtype, 1], Self.n]
    ) -> Array[
        Plain[Self.dtype, 1], Self.n
    ] where Self.dtype.is_floating_point():
        """Solve `A @ x = b`, reusing this factorization.

        Permutes `b` the way the factorization permuted `A`'s rows, then
        runs the same two substitutions the unpivoted `solve` does. Several
        right-hand sides against one matrix is what the split into a
        factorization and a solve is for.
        """
        var permuted = _zeros[Plain[Self.dtype, 1], Self.n]()
        for i in range(Self.n):
            permuted[i] = b[self.permutation[i]].copy()
        var y = forward_substitution[
            Plain[Self.dtype, 1], Self.n, unit_diagonal=True
        ](self.factored, permuted)
        return back_substitution[Plain[Self.dtype, 1], Self.n](self.factored, y)

    def det(self) -> Plain[Self.dtype, 1] where Self.dtype.is_floating_point():
        """The determinant: the product of `U`'s diagonal, signed by the
        permutation.

        The pivoted counterpart of `det`, and it gets the answers `det`
        cannot -- the exchange matrix comes out as `-1` where the unpivoted
        route returns approximately zero.
        """
        var product = Plain[Self.dtype, 1].constant(Float64(self.sign))
        for i in range(Self.n):
            product = product * self.factored[i * Self.n + i]
        return product^


def lu_factor[
    dtype: DType, n: Int
](a: Array[Plain[dtype, 1], n * n]) -> PivotedLU[
    dtype, n
] where dtype.is_floating_point():
    """**Tier 2.** Factor `A` into `L @ U` with partial pivoting.
    `scipy.linalg.lu_factor`.

    See `PivotedLU` for what pivoting costs and what it buys.
    """
    var out = _zeros[Plain[dtype, 1], n * n]()
    for i in range(n * n):
        out[i] = a[i].copy()

    var permutation = Array[Int, n](fill=0)
    for i in range(n):
        permutation[i] = i
    var sign = 1

    for k in range(n):
        var best = k
        var best_magnitude = abs(out[k * n + k].v)
        for i in range(k + 1, n):
            var magnitude = abs(out[i * n + k].v)
            if magnitude > best_magnitude:
                best = i
                best_magnitude = magnitude

        if best != k:
            for j in range(n):
                var swap = out[k * n + j].copy()
                out[k * n + j] = out[best * n + j].copy()
                out[best * n + j] = swap^
            var swap_index = permutation[k]
            permutation[k] = permutation[best]
            permutation[best] = swap_index
            sign = -sign

        var pivot = guard_nonzero(
            out[k * n + k], Plain[dtype, 1].constant(_PIVOT_FLOOR)
        )
        for i in range(k + 1, n):
            var factor = out[i * n + k] / pivot
            out[i * n + k] = factor.copy()
            for j in range(k + 1, n):
                out[i * n + j] = out[i * n + j] - (factor * out[k * n + j])

    return PivotedLU[dtype, n](out^, permutation^, sign)


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


comptime fro = 0
"""`ord` for `norm`: the Frobenius (entrywise 2-) norm. The default."""
comptime inf = -1
"""`ord` for `norm`: the induced infinity-norm."""


def norm[
    T: FloatLike, n: Int, ord: Int = fro
](a: Array[T, n * n]) -> T where ord == fro or ord == 1 or ord == inf:
    """A matrix norm of `A`. `numpy.linalg.norm(A, ord=...)`.

    `ord` picks which one, as a compile-time parameter so the loop is
    chosen at compile time and nothing branches per call:

    | `ord` | Norm |
    |---|---|
    | `fro` (default) | Frobenius: `sqrt(sum(A[i,j]**2))` |
    | `1` | Induced 1-norm: the largest absolute column sum |
    | `inf` | Induced infinity-norm: the largest absolute row sum |

    The Frobenius sum is taken directly rather than in a scaled/squared
    form, which means a matrix whose entries are near the square root of
    `dtype`'s overflow threshold will overflow. LAPACK's `lange` rescales
    to avoid that; doing the same would need a data-dependent branch on the
    running maximum, which the fixed-iteration invariant rules out. Call
    this at `Compensated` if the summation length is what worries you, or
    scale `A` yourself if its magnitude is.

    The induced norms take their column or row maximum with `max_of`, not
    an `if` -- `T` may hold a SIMD vector whose lanes disagree about which
    column is largest, so the running maximum has to be arithmetic. Same
    reason every other selection in `numax` is branchless.

    No MAX equivalent at any size: MAX ships no norm of any kind.
    """
    comptime if ord == fro:
        var total = T.constant(0.0)
        for i in range(n * n):
            total = total + a[i] * a[i]
        return total.sqrt()
    comptime if ord == 1:
        var best = T.constant(0.0)
        for j in range(n):
            var column = T.constant(0.0)
            for i in range(n):
                column = column + a[i * n + j].abs()
            best = max_of(best, column)
        return best^
    comptime if ord == inf:
        var best = T.constant(0.0)
        for i in range(n):
            var row = T.constant(0.0)
            for j in range(n):
                row = row + a[i * n + j].abs()
            best = max_of(best, row)
        return best^
    # Unreachable: the `where` clause above admits no fourth `ord`.
    return T.constant(0.0)


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
        v[k] = v[k] - alpha

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
                r[i * n + j] = r[i * n + j] - (factor * v[i])

        # Q <- Q (I - scale*v v^T), accumulating the reflectors' product.
        for i in range(n):
            var qv = T.constant(0.0)
            for j in range(k, n):
                qv = qv + q[i * n + j] * v[j]
            var factor = qv * scale
            for j in range(k, n):
                q[i * n + j] = q[i * n + j] - (factor * v[j])

    # The strict lower triangle holds the reflector residue, not part of R.
    for i in range(n):
        for j in range(i):
            r[i * n + j] = T.constant(0.0)

    return (r^, q^)


def lstsq[
    T: FloatLike, m: Int, n: Int
](a: Array[T, m * n], b: Array[T, m]) -> Array[T, n] where m >= n:
    """The least-squares solution of the overdetermined `A x = b`: the `x`
    minimizing `||A x - b||`. `numpy.linalg.lstsq`, first return value.

    `A` is `m x n` row-major with `m >= n`, which is the overdetermined
    case -- more equations than unknowns, the shape a fit has. An
    underdetermined system has a solution space rather than a solution and
    wants `pinv` instead.

    Householder QR applied to `A` and to `b` together, then back
    substitution on `R`. `Q` is never formed: each reflector is applied to
    `b` as it is built, which is both cheaper and better conditioned than
    the normal equations `A^T A x = A^T b` -- those square the condition
    number and throw away half the significant digits of an ill-conditioned
    fit.

    Rank-deficient `A` is not detected. The diagonal of `R` is floored away
    from zero so the back substitution stays finite rather than producing
    NaN, but finite is not correct: a rank-deficient fit needs the
    truncation `pinv` does, and detecting the rank means comparing against a
    tolerance, which is a per-lane branch. Reach for `pinv` when the columns
    might be dependent.

    Returns only `x`. The residual is `||A x - b||`, which the caller can
    form from `matvec` and `nrm2`, and the rank and singular values come
    from `svd`.
    """
    var r = _zeros[T, m * n]()
    for i in range(m * n):
        r[i] = a[i].copy()
    var y = _zeros[T, m]()
    for i in range(m):
        y[i] = b[i].copy()

    for k in range(n):
        var norm_sq = T.constant(0.0)
        for i in range(k, m):
            norm_sq = norm_sq + r[i * n + k] * r[i * n + k]
        var alpha = norm_sq.sqrt().copysign(-r[k * n + k])

        var v = _zeros[T, m]()
        for i in range(k, m):
            v[i] = r[i * n + k].copy()
        v[k] = v[k] - alpha

        var vv = T.constant(0.0)
        for i in range(k, m):
            vv = vv + v[i] * v[i]
        var scale = T.constant(2.0) / guard_nonzero(
            vv, T.constant(_PIVOT_FLOOR)
        )

        for j in range(k, n):
            var vr = T.constant(0.0)
            for i in range(k, m):
                vr = vr + v[i] * r[i * n + j]
            var factor = vr * scale
            for i in range(k, m):
                r[i * n + j] = r[i * n + j] - (factor * v[i])

        # The same reflector on `b`, which is what makes forming `Q`
        # unnecessary.
        var vy = T.constant(0.0)
        for i in range(k, m):
            vy = vy + v[i] * y[i]
        var factor_y = vy * scale
        for i in range(k, m):
            y[i] = y[i] - (factor_y * v[i])

    # Back substitution on the leading n x n triangle of R. Rows n..m-1 of
    # `y` are the residual and take no part in it.
    var x = _zeros[T, n]()
    for k in range(n):
        var i = n - 1 - k
        var total = y[i].copy()
        for j in range(i + 1, n):
            total = total - r[i * n + j] * x[j]
        x[i] = total / guard_nonzero(r[i * n + i], T.constant(_PIVOT_FLOOR))
    return x^


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
    trade-off `norm` documents.
    """
    return dot[T, n](a, a).sqrt()


def asum[T: FloatLike, n: Int](a: Array[T, n]) -> T:
    """The sum of magnitudes `sum(|a[i]|)` -- BLAS-1 `asum`, and the vector
    1-norm `norm` gives for a matrix.

    Cannot overflow the way `nrm2` can, which is why a convergence check
    that only needs a magnitude usually wants this one.
    """
    var total = T.constant(0.0)
    for i in range(n):
        total = total + a[i].abs()
    return total^


def axpy[
    T: FloatLike, n: Int
](alpha: T, x: Array[T, n], y: Array[T, n]) -> Array[T, n]:
    """`alpha * x + y` -- BLAS-1 `axpy`.

    Returns a new vector rather than updating `y` in place, since a
    register-resident `Array` has no aliasing to avoid and an expression
    reads better than a mutation at these sizes.
    """
    var out = _zeros[T, n]()
    for i in range(n):
        out[i] = alpha * x[i] + y[i]
    return out^


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


def _hessenberg[T: FloatLike, n: Int](a: Array[T, n * n]) -> Array[T, n * n]:
    """`Q.T @ A @ Q` in upper Hessenberg form, by `n - 2` Householder
    reflections applied from both sides."""
    var h = _zeros[T, n * n]()
    for i in range(n * n):
        h[i] = a[i].copy()

    for k in range(n - 2):
        var norm_sq = T.constant(0.0)
        for i in range(k + 1, n):
            norm_sq = norm_sq + h[i * n + k] * h[i * n + k]
        var alpha = norm_sq.sqrt().copysign(-h[(k + 1) * n + k])

        var v = _zeros[T, n]()
        for i in range(k + 1, n):
            v[i] = h[i * n + k].copy()
        v[k + 1] = v[k + 1] - alpha

        var vv = T.constant(0.0)
        for i in range(k + 1, n):
            vv = vv + v[i] * v[i]
        var scale = T.constant(2.0) / guard_nonzero(
            vv, T.constant(_PIVOT_FLOOR)
        )

        for j in range(n):
            var vh = T.constant(0.0)
            for i in range(k + 1, n):
                vh = vh + v[i] * h[i * n + j]
            var factor = vh * scale
            for i in range(k + 1, n):
                h[i * n + j] = h[i * n + j] - (factor * v[i])

        for i in range(n):
            var hv = T.constant(0.0)
            for j in range(k + 1, n):
                hv = hv + h[i * n + j] * v[j]
            var factor = hv * scale
            for j in range(k + 1, n):
                h[i * n + j] = h[i * n + j] - (factor * v[j])

    return h^


def _wilkinson_shift[T: FloatLike, n: Int](h: Array[T, n * n]) -> T:
    """The eigenvalue of the trailing 2x2 block nearest its bottom-right
    entry, or that entry itself where the block's own eigenvalues are
    complex and no real shift is nearer than another."""
    var p = h[(n - 2) * n + (n - 2)].copy()
    var q = h[(n - 2) * n + (n - 1)].copy()
    var r = h[(n - 1) * n + (n - 2)].copy()
    var s = h[(n - 1) * n + (n - 1)].copy()

    var half = (p - s) * T.constant(0.5)
    var disc = half * half + q * r
    var root = max_of(disc, T.constant(0.0)).sqrt()
    return blend(
        ge_indicator(disc, T.constant(0.0)),
        s + half - root.copysign(half),
        s,
    )


def _qr_sweep[T: FloatLike, n: Int](mut h: Array[T, n * n], shift: T):
    """One shifted QR step in place: factor `H - shift*I` by Givens
    rotations, then re-multiply in the opposite order and undo the shift.

    The rotations are stored rather than accumulated into an explicit `Q`,
    which is the whole reason a QR iteration costs what it does.
    """
    var cosines = _zeros[T, n]()
    var sines = _zeros[T, n]()

    for i in range(n):
        h[i * n + i] = h[i * n + i] - shift

    for k in range(n - 1):
        var x = h[k * n + k].copy()
        var y = h[(k + 1) * n + k].copy()
        var length = guard_nonzero(
            (x * x + y * y).sqrt(), T.constant(_PIVOT_FLOOR)
        )
        var c = x / length
        var s = y / length
        cosines[k] = c.copy()
        sines[k] = s.copy()
        for j in range(n):
            var top = h[k * n + j].copy()
            var bottom = h[(k + 1) * n + j].copy()
            h[k * n + j] = c * top + s * bottom
            h[(k + 1) * n + j] = c * bottom - s * top

    for k in range(n - 1):
        var c = cosines[k].copy()
        var s = sines[k].copy()
        for i in range(n):
            var left = h[i * n + k].copy()
            var right = h[i * n + (k + 1)].copy()
            h[i * n + k] = c * left + s * right
            h[i * n + (k + 1)] = c * right - s * left

    for i in range(n):
        h[i * n + i] = h[i * n + i] + shift


def _block_roots[
    T: FloatLike, n: Int
](h: Array[T, n * n], k: Int) -> Tuple[Complex[T], Complex[T]]:
    """Both eigenvalues of the 2x2 block at `(k, k)`, real pair or conjugate
    pair, without branching on which it is: one of the two square roots is
    always of a negated discriminant and comes out zero."""
    var p = h[k * n + k].copy()
    var q = h[k * n + (k + 1)].copy()
    var r = h[(k + 1) * n + k].copy()
    var s = h[(k + 1) * n + (k + 1)].copy()

    var middle = (p + s) * T.constant(0.5)
    var half = (p - s) * T.constant(0.5)
    var disc = half * half + q * r
    var real_part = max_of(disc, T.constant(0.0)).sqrt()
    var imaginary_part = max_of(-disc, T.constant(0.0)).sqrt()

    return (
        Complex[T](middle + real_part, imaginary_part.copy()),
        Complex[T](middle - real_part, -imaginary_part),
    )


def eigvals[
    T: FloatLike, n: Int, sweeps: Int = 100, tol: Float64 = 1e-8
](a: Array[T, n * n]) -> Array[Complex[T], n]:
    """Eigenvalues of a general square matrix, real or complex.
    `numpy.linalg.eigvals`.

    `eigh` covers the symmetric case with a better algorithm and real
    output; this is for the matrices that have no symmetry to exploit -- a
    Jacobian whose stability is in question, a companion matrix -- and it
    returns `Complex[T]` because a real matrix's eigenvalues genuinely can
    be complex. `[[0, -1], [1, 0]]` is a rotation by a quarter turn and its
    eigenvalues are `+i` and `-i`; there is no real answer to round to.

    Householder reduction to Hessenberg form, then `sweeps` shifted QR
    steps, then the eigenvalues read off the quasi-triangular result.

    **Fixed sweeps, so this stays tier 1 and GPU-launchable**, and that is
    the real limit rather than a formality. LAPACK deflates a converged
    eigenvalue and restarts on what remains, driven by a convergence test
    this cannot run; here every sweep works on the whole matrix and there
    are a fixed number of them. A well separated spectrum converges in far
    fewer than 100; a cluster of nearly equal eigenvalues may not converge
    in any number, and the result then degrades quietly rather than
    reporting failure. `sum(eigvals) == trace` and `prod(eigvals) == det`
    are the two identities to check against if it matters.

    `tol` decides which subdiagonal entries survived as a 2x2 block rather
    than converging to zero, relative to the neighbouring diagonal entries.
    It has to be a parameter and not a machine epsilon because `T` is any
    `FloatLike`, and a `Dual` or an `Interval` has no epsilon to consult.

    **Eigenvalues come out in no particular order**, for the reason `eigh`
    gives: sorting is data-dependent.

    No MAX equivalent exists at any size -- MAX ships no eigensolver.
    """
    var h = _hessenberg[T, n](a)
    comptime if n >= 2:
        for _ in range(sweeps):
            _qr_sweep[T, n](h, _wilkinson_shift[T, n](h))

    var out = Array[Complex[T], n](
        fill=Complex[T](T.constant(0.0), T.constant(0.0))
    )
    for i in range(n):
        var value = Complex[T](h[i * n + i].copy(), T.constant(0.0))

        # A surviving subdiagonal means `i` heads a 2x2 block; a surviving
        # one above means it closes the block that `i - 1` heads. Both
        # candidates are computed for every `i` and blended, since which
        # applies is per-lane data.
        comptime if n >= 2:
            if i + 1 < n:
                var roots = _block_roots[T, n](h, i)
                value = _blend_complex(
                    _subdiagonal_indicator[T, n](h, i, tol), roots[0], value
                )
            if i >= 1:
                var roots = _block_roots[T, n](h, i - 1)
                value = _blend_complex(
                    _subdiagonal_indicator[T, n](h, i - 1, tol),
                    roots[1],
                    value,
                )
        out[i] = value^
    return out^


def _subdiagonal_indicator[
    T: FloatLike, n: Int
](h: Array[T, n * n], k: Int, tol: Float64) -> T:
    """`1` where `h[k+1, k]` is large enough relative to its neighbouring
    diagonal entries to be a 2x2 block rather than a converged zero."""
    var scale = (
        h[k * n + k].abs()
        + h[(k + 1) * n + (k + 1)].abs()
        + T.constant(_PIVOT_FLOOR)
    )
    return ge_indicator(
        h[(k + 1) * n + k].abs() - scale * T.constant(tol), T.constant(0.0)
    )


def _blend_complex[
    T: FloatLike
](indicator: T, if_one: Complex[T], if_zero: Complex[T]) -> Complex[T]:
    """`blend` over a complex value, by blending the parts."""
    return Complex[T](
        blend(indicator, if_one.re, if_zero.re),
        blend(indicator, if_one.im, if_zero.im),
    )


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
                var rotation = _jacobi_rotation[T](apq, aqq - app)
                var c = rotation[0].copy()
                var s = rotation[1].copy()

                # Rows p and q.
                for k in range(n):
                    var akp = work[p * n + k].copy()
                    var akq = work[q * n + k].copy()
                    work[p * n + k] = c * akp - (s * akq)
                    work[q * n + k] = s * akp + c * akq
                # Columns p and q, completing the similarity transform.
                for k in range(n):
                    var apk = work[k * n + p].copy()
                    var aqk = work[k * n + q].copy()
                    work[k * n + p] = c * apk - (s * aqk)
                    work[k * n + q] = s * apk + c * aqk
                # The same rotation applied to the accumulating basis.
                for k in range(n):
                    var vkp = vectors[k * n + p].copy()
                    var vkq = vectors[k * n + q].copy()
                    vectors[k * n + p] = c * vkp - (s * vkq)
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

                var rotation = _jacobi_rotation[T](gamma, beta - alpha)
                var c = rotation[0].copy()
                var s = rotation[1].copy()

                for i in range(n):
                    var ip = work[i * n + p].copy()
                    var iq = work[i * n + q].copy()
                    work[i * n + p] = c * ip - (s * iq)
                    work[i * n + q] = s * ip + c * iq
                for i in range(n):
                    var ip = v[i * n + p].copy()
                    var iq = v[i * n + q].copy()
                    v[i * n + p] = c * ip - (s * iq)
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


# ------------------------------------------------------------------
# The MAX tier: `Tensor` in, `Tensor` out.
#
# Everything below delegates to a MAX kernel over `TileTensor`. numax
# allocates the destination, takes `.view()`s, calls MAX and synchronizes;
# the algorithm, the tiling and the per-architecture choice are all MAX's.
#
# `gpu` is a compile-time parameter rather than a look at `ctx.api()`
# because MAX's `target` is a `StaticString`: deciding it at run time would
# compile the GPU kernels into every CPU-only build. `map` and `reduce` in
# `numax.core.tensor` take the same parameter for the same reason.
#
# These take their operands mutably even though they only read them --
# `view()` hands back a `TileTensor` that can write, and a mutable view
# cannot be built from an immutable binding. `transpose` in
# `numax.core.array` has the same signature for the same reason.
# ------------------------------------------------------------------


def matmul[
    dtype: DType, m: Int, k: Int, n: Int, gpu: Bool = False
](mut a: Shaped[dtype, m, k], mut b: Shaped[dtype, k, n]) raises -> Shaped[
    dtype, m, n
]:
    """The matrix product `a @ b`, on `a`'s own device.

    `linalg.matmul` does the work, and that one call is a whole dispatch
    tree: Apple simdgroup kernels, SM100 `tcgen05`, SM90, Ampere and CDNA
    multistage GEMM with tile shapes chosen by heuristic, GEMV when a
    dimension is 1, vendor cuBLAS/rocBLAS/hipBLASLt, AMD RDNA WMMA, and a
    naive kernel when nothing else fits. numax names no architecture and
    picks no kernel.

    The sibling `matmul` over `Array[T, n*n]` is the one to call inside a
    kernel, or at any conformer other than a raw `dtype`; `to_tensor`
    crosses from there to here and `to_array` back.
    """
    var ctx = a.context()
    var result = Shaped[dtype, m, n](ctx)
    var c = result.view()
    _max_matmul[target="gpu" if gpu else "cpu"](c, a.view(), b.view(), ctx)
    ctx.synchronize()
    return result^


def matmul[
    dtype: DType, gpu: Bool = False
](mut a: Dynamic[dtype, 2], mut b: Dynamic[dtype, 2]) raises -> Dynamic[
    dtype, 2
]:
    """The matrix product `a @ b` at extents known only at run time.

    The run-time-shaped overload of the one above, selected by argument
    type rather than by a `where` clause: `Shaped` and `Dynamic` are
    different layouts, so the two can never be ambiguous. MAX reads the
    extents from the layout either way -- a compile-time shape buys kernel
    specialization, not correctness.

    Raises when `a`'s columns and `b`'s rows disagree, which is the check
    the static overload gets from the type system for free.
    """
    if a.dim[1]() != b.dim[0]():
        raise Error(
            "matmul shape mismatch: a is ",
            a.dim[0](),
            "x",
            a.dim[1](),
            " and b is ",
            b.dim[0](),
            "x",
            b.dim[1](),
        )
    var ctx = a.context()
    var result = zeros_dyn[dtype, 2](a.dim[0](), b.dim[1](), ctx=ctx)
    var c = result.view()
    _max_matmul[target="gpu" if gpu else "cpu"](c, a.view(), b.view(), ctx)
    ctx.synchronize()
    return result^


def matvec[
    dtype: DType, m: Int, k: Int, gpu: Bool = False
](mut a: Shaped[dtype, m, k], mut x: Shaped[dtype, k]) raises -> Shaped[
    dtype, m
]:
    """The matrix-vector product `a @ x`, on `a`'s own device.

    Routed through `linalg.matmul` rather than `linalg.gemv`, with the
    vectors relaid as `k x 1` and `m x 1` over their own pointers. MAX's
    dispatch already sends a matmul with `n == 1` to its GEMV kernels --
    including the split-K and vector variants it picks between by shape --
    so calling `gemv` directly would be a second numax path to the same
    kernels, and one that would have to reproduce the choice.

    The relayout is free: a rank-1 buffer of `k` elements and a `k x 1`
    row-major layout address the same memory in the same order, so this
    hands MAX a different description of bytes it was going to read
    anyway, not a copy.
    """
    var ctx = a.context()
    var result = Shaped[dtype, m](ctx)
    var xv = x.view()
    var yv = result.view()
    var x_col = TileTensor(xv.ptr_at_offset(Coord(0)), row_major(Coord(k, 1)))
    var y_col = TileTensor(yv.ptr_at_offset(Coord(0)), row_major(Coord(m, 1)))
    _max_matmul[target="gpu" if gpu else "cpu"](y_col, a.view(), x_col, ctx)
    ctx.synchronize()
    return result^


def batched_matmul[
    dtype: DType, batch: Int, m: Int, k: Int, n: Int, gpu: Bool = False
](
    mut a: Shaped[dtype, batch, m, k], mut b: Shaped[dtype, batch, k, n]
) raises -> Shaped[dtype, batch, m, n]:
    """`batch` independent matrix products, one per leading index.

    `linalg.bmm.batched_matmul` does the work, in one launch rather than
    `batch` of them. There is no `Array[T, n*n]` counterpart: a batch of
    small matrices is what a `map` over a conformer already expresses, one
    matrix per SIMD lane, so the batched form only earns its own kernel at
    sizes past the crossover.
    """
    var ctx = a.context()
    var result = Shaped[dtype, batch, m, n](ctx)
    var c = result.view()
    _max_batched_matmul[target="gpu" if gpu else "cpu"](
        c, a.view(), b.view(), context=ctx
    )
    ctx.synchronize()
    return result^


def _staged[
    dtype: DType
](
    values: List[Scalar[dtype]],
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises -> Dynamic[dtype, 2]:
    """A `rows x cols` device tensor holding `values`, row-major."""
    var staged = zeros_dyn[dtype, 2](rows, cols, ctx=ctx)
    staged.copy_from_host(values)
    return staged^


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
