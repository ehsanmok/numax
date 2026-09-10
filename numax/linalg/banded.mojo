"""Solvers for banded and Toeplitz systems. `scipy.linalg`'s `solve_banded`
family.

**This module is tier 2, `Plain`-only, and host-side, and that is not a
staging post.** A banded elimination has no `O(n^3)` term: its work is
`O(n * bandwidth^2)`, spread over `n` sequential column steps that each
touch a `bandwidth x bandwidth` corner. There is no GEMM to hand anything
to, which is the same reason `numax.linalg.array.tridiagonal_solve` stays
where it is. So this is an **entry-surface** commit -- it adds names a
SciPy user reaches for and a shape numax could not express, not a faster
path to an answer numax already had.

The dense `numax.linalg.solve` is the alternative and it is not a silly
one: it is blocked, device-resident, and sends its cubic term through
`linalg.matmul`. For a bandwidth that is an appreciable fraction of `n`,
dense wins outright. These are for the narrow band -- a second-difference
operator, a spline's normal equations, an autocorrelation -- where `n` is
large and the bandwidth is single digits, and the dense route would be
`O(n^3)` on a matrix that is almost entirely zero.

## Storage

`ab` is SciPy's diagonal-ordered form, verbatim, so an array transfers
between the two libraries unchanged.

For `solve_banded` with `l` subdiagonals and `u` superdiagonals, `ab` is
`(l + u + 1) x n` and

```
ab[u + i - j, j] == a[i, j]
```

for every `i, j` inside the band. Row `u` is the main diagonal, the rows
above it are the superdiagonals, and the rows below are the subdiagonals.
Entries outside the matrix -- the top-left and bottom-right corners of `ab`
-- are never read, so they may hold anything.

For the symmetric routines, `ab` is `(u + 1) x n` and holds one triangle.
`lower=False` (the default, and SciPy's) means the upper one, with row `u`
the main diagonal; `lower=True` means the lower one, with row `0` the main
diagonal. The two are transposes of each other and the same solver runs
underneath, since a symmetric matrix has nothing to distinguish them by.

## What is not here

`solve_circulant` lives in this module's neighbourhood mathematically but
not in this file: it is an FFT rather than an elimination, so it would give
`numax.linalg` a dependency on `numax.fft` -- an edge worth adding
deliberately rather than as a side effect of a bug fix.

Banded eigenvalues (`eig_banded`, `eigvals_banded`, `eigh_tridiagonal`) are
not here for the reason `numax/linalg/__init__.mojo` gives for the dense
spectral four: the reduction phase is tractable and the iterative phase is
a sequential sweep with data-dependent deflation.
"""

from std.math import sqrt as _sqrt

from ..core.array import Static


def _band_index[l: Int, u: Int, n: Int](row: Int, col: Int) -> Int:
    """Where `a[row, col]` sits in the flattened `(l + u + 1) x n` band."""
    return (u + row - col) * n + col


def solve_banded[
    dtype: DType, l: Int, u: Int, n: Int
](
    mut ab: Static[dtype, l + u + 1, n], mut b: Static[dtype, n]
) raises -> Static[dtype, n] where (
    dtype.is_floating_point() and l >= 0 and u >= 0 and n >= 1
):
    """Solve `a @ x = b` for a general banded `a`. `scipy.linalg.solve_banded`.

    `l` subdiagonals and `u` superdiagonals, `ab` in the diagonal-ordered
    form this module's docstring specifies.

    **Partial pivoting, like SciPy's**, which is what makes this usable on a
    matrix that is not diagonally dominant. Pivoting is also what forces the
    working band to be wider than the input: exchanging row `j` with a row
    up to `l` below it can move an entry up to `l` columns further right, so
    the fill-in occupies `u + l` superdiagonals rather than `u`. The
    workspace is `(2l + u + 1) x n`, which is LAPACK's `gbtrf` layout and
    the reason `ab` is copied rather than factored in place.

    Factors and solves in one pass -- there is no banded `lu_factor` here to
    reuse across right-hand sides, because a second right-hand side is
    `O(n * bandwidth)` against a factorization's `O(n * bandwidth^2)` and
    the split would only pay for many of them. Call this once per `b`.

    A zero pivot -- a singular matrix, or one whose band is misdescribed --
    raises rather than returning infinities.
    """
    var ctx = ab.context()
    var band = ab.to_host()
    var rhs = b.to_host()

    # LAPACK's `gbtrf` workspace: the input band shifted down by `l`, with
    # `l` empty rows above it for the fill-in pivoting creates.
    comptime work_rows = 2 * l + u + 1
    var work = List[Float64](length=work_rows * n, fill=0.0)
    for row in range(l + u + 1):
        for col in range(n):
            work[(l + row) * n + col] = Float64(band[row * n + col])

    var x = List[Float64](capacity=n)
    for i in range(n):
        x.append(Float64(rhs[i]))

    # `a[i, j]` is `work[(l + u + i - j) * n + j]` throughout.
    for j in range(n):
        var last_row = min(j + l, n - 1)

        # Partial pivot: the largest magnitude in column `j`, on or below
        # the diagonal and within the band.
        var pivot = j
        var largest = abs(work[(l + u) * n + j])
        for i in range(j + 1, last_row + 1):
            var candidate = abs(work[(l + u + i - j) * n + j])
            if candidate > largest:
                largest = candidate
                pivot = i

        if largest == 0:
            raise Error(
                "solve_banded: the matrix is singular -- column ",
                j,
                (
                    " has no nonzero pivot within the band. Check that `l` and"
                    " `u` describe the band actually present in `ab`."
                ),
            )

        if pivot != j:
            # Swapping two rows of a banded matrix is a swap per column, at
            # a different offset in each: `a[i, k]` moves with `i - k`.
            var last_col = min(j + u + l, n - 1)
            for k in range(j, last_col + 1):
                var here = (l + u + j - k) * n + k
                var there = (l + u + pivot - k) * n + k
                var keep = work[here]
                work[here] = work[there]
                work[there] = keep
            var keep_rhs = x[j]
            x[j] = x[pivot]
            x[pivot] = keep_rhs

        var diagonal = work[(l + u) * n + j]
        var last_col = min(j + u + l, n - 1)
        for i in range(j + 1, last_row + 1):
            var multiplier = work[(l + u + i - j) * n + j] / diagonal
            if multiplier == 0:
                continue
            work[(l + u + i - j) * n + j] = 0
            for k in range(j + 1, last_col + 1):
                work[(l + u + i - k) * n + k] -= (
                    multiplier * work[(l + u + j - k) * n + k]
                )
            x[i] -= multiplier * x[j]

    # Back substitution over the widened band.
    for step in range(n):
        var i = n - 1 - step
        var last_col = min(i + u + l, n - 1)
        var total = x[i]
        for k in range(i + 1, last_col + 1):
            total -= work[(l + u + i - k) * n + k] * x[k]
        x[i] = total / work[(l + u) * n + i]

    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype](x[i]))
    return Static[dtype, n](ctx, out^)


def cholesky_banded[
    dtype: DType, u: Int, n: Int, lower: Bool = False
](mut ab: Static[dtype, u + 1, n]) raises -> Static[dtype, u + 1, n] where (
    dtype.is_floating_point() and u >= 0 and n >= 1
):
    """The Cholesky factor of a symmetric positive definite banded matrix,
    in the same banded storage. `scipy.linalg.cholesky_banded`.

    `u` superdiagonals (or subdiagonals, when `lower`), so `ab` is
    `(u + 1) x n`. The factor has the same bandwidth as the matrix -- that
    is the property that makes a banded Cholesky worth having, and it is not
    shared by the LU in `solve_banded`, whose pivoting widens the band.

    No pivoting, and none is needed: a symmetric positive definite matrix
    has a Cholesky factorization without it, which is the same reason the
    dense `numax.linalg.cholesky` does not pivot either.

    A non-positive pivot raises. It means the matrix is not positive
    definite -- or, just as often, that `ab` does not hold the triangle
    `lower` says it does.
    """
    var ctx = ab.context()
    var host = ab.to_host()

    # Worked in lower form internally: `band[d * n + j]` is `a[j + d, j]`.
    # The upper input is the transpose of that, entry by entry, since the
    # matrix is symmetric.
    var band = List[Float64](length=(u + 1) * n, fill=0.0)
    for d in range(u + 1):
        for j in range(n):
            if lower:
                band[d * n + j] = Float64(host[d * n + j])
            elif j + d < n:
                # Upper storage: `a[i, k]` at `ab[u + i - k, k]`. With
                # `i = j` and `k = j + d` that is `ab[u - d, j + d]`.
                band[d * n + j] = Float64(host[(u - d) * n + j + d])

    for j in range(n):
        var total = band[j]
        var first = max(0, j - u)
        for k in range(first, j):
            var entry = band[(j - k) * n + k]
            total -= entry * entry
        if total <= 0:
            raise Error(
                (
                    "cholesky_banded: the matrix is not positive definite --"
                    " the pivot at column "
                ),
                j,
                (
                    " is not positive. If the matrix should be, check that"
                    " `ab` holds the triangle `lower` names."
                ),
            )
        var pivot = _sqrt(total)
        band[j] = pivot

        var last = min(j + u, n - 1)
        for i in range(j + 1, last + 1):
            var accumulated = band[(i - j) * n + j]
            var start = max(0, i - u)
            for k in range(start, j):
                accumulated -= band[(i - k) * n + k] * band[(j - k) * n + k]
            band[(i - j) * n + j] = accumulated / pivot

    var out = List[Scalar[dtype]](length=(u + 1) * n, fill=0)
    for d in range(u + 1):
        for j in range(n):
            if lower:
                out[d * n + j] = Scalar[dtype](band[d * n + j])
            elif j + d < n:
                out[(u - d) * n + j + d] = Scalar[dtype](band[d * n + j])
    return Static[dtype, u + 1, n](ctx, out^)


def cho_solve_banded[
    dtype: DType, u: Int, n: Int, lower: Bool = False
](mut cb: Static[dtype, u + 1, n], mut b: Static[dtype, n]) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and u >= 0 and n >= 1):
    """Solve `a @ x = b` given `a`'s banded Cholesky factor.
    `scipy.linalg.cho_solve_banded`.

    `cb` is what `cholesky_banded` returned, with the same `lower`. Two
    triangular sweeps, forward then back, each `O(n * u)`.

    This is the half of `solveh_banded` worth separating: factoring is
    `O(n * u^2)` and solving is `O(n * u)`, so a second right-hand side
    against the same matrix costs a `u`-th of the first. That asymmetry is
    real here in a way it is not for `solve_banded`, whose pivoting makes a
    reusable factorization a different and wider object.
    """
    var ctx = cb.context()
    var host = cb.to_host()
    var rhs = b.to_host()

    var band = List[Float64](length=(u + 1) * n, fill=0.0)
    for d in range(u + 1):
        for j in range(n):
            if lower:
                band[d * n + j] = Float64(host[d * n + j])
            elif j + d < n:
                band[d * n + j] = Float64(host[(u - d) * n + j + d])

    var x = List[Float64](capacity=n)
    for i in range(n):
        x.append(Float64(rhs[i]))

    # Forward: L y = b, with `L[i, j]` at `band[(i - j) * n + j]`.
    for i in range(n):
        var total = x[i]
        var first = max(0, i - u)
        for j in range(first, i):
            total -= band[(i - j) * n + j] * x[j]
        x[i] = total / band[i]

    # Back: L^T x = y.
    for step in range(n):
        var i = n - 1 - step
        var total = x[i]
        var last = min(i + u, n - 1)
        for k in range(i + 1, last + 1):
            total -= band[(k - i) * n + i] * x[k]
        x[i] = total / band[i]

    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype](x[i]))
    return Static[dtype, n](ctx, out^)


def solveh_banded[
    dtype: DType, u: Int, n: Int, lower: Bool = False
](mut ab: Static[dtype, u + 1, n], mut b: Static[dtype, n]) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and u >= 0 and n >= 1):
    """Solve `a @ x = b` for a symmetric positive definite banded `a`.
    `scipy.linalg.solveh_banded`.

    `cholesky_banded` then `cho_solve_banded`, which is what SciPy does and
    is spelled that way here rather than fused: the two halves are each
    useful alone, and a caller with several right-hand sides should be
    factoring once. This is the convenience for the single-solve case.

    Prefer this to `solve_banded` whenever the matrix really is symmetric
    positive definite. It does half the arithmetic, needs no pivoting, and
    -- the part that matters at scale -- keeps the factor's bandwidth equal
    to the matrix's, where `solve_banded`'s pivoting widens it to `u + l`.
    """
    var factor = cholesky_banded[dtype, u, n, lower](ab)
    return cho_solve_banded[dtype, u, n, lower](factor, b)


def solve_toeplitz[
    dtype: DType, n: Int
](
    mut c: Static[dtype, n], mut r: Static[dtype, n], mut b: Static[dtype, n]
) raises -> Static[dtype, n] where (dtype.is_floating_point() and n >= 1):
    """Solve `a @ x = b` where `a` is the Toeplitz matrix with first column
    `c` and first row `r`. `scipy.linalg.solve_toeplitz`.

    `r[0]` is ignored and `c[0]` is the diagonal, matching
    `numax.linalg.special_matrices.toeplitz` and SciPy.

    **Levinson-Durbin**, so the cost is `O(n^2)` against a dense solve's
    `O(n^3)` and the matrix is never materialized -- the whole point, since
    a Toeplitz matrix is `2n - 1` numbers describing `n^2` entries. The
    recursion extends the solution of the leading `k x k` system to
    `k + 1` at each step, using forward and backward vectors it maintains
    alongside.

    **No pivoting, and that is a real limitation rather than an oversight.**
    Levinson's recursion has no room for a row exchange: it is built on the
    system's Toeplitz structure, which a swap destroys. So it needs every
    leading principal minor to be nonsingular, which a symmetric positive
    definite Toeplitz matrix (an autocorrelation, the common case) always
    satisfies and a general one need not. A zero reflection denominator
    raises rather than returning noise; the fallback is
    `numax.linalg.special_matrices.toeplitz` into the dense
    `numax.linalg.solve`, which pivots.
    """
    var ctx = c.context()
    var col = c.to_host()
    var row = r.to_host()
    var rhs = b.to_host()

    var diagonal = Float64(col[0])
    if diagonal == 0:
        raise Error(
            "solve_toeplitz: the leading entry c[0] is zero, so the 1x1"
            " leading minor is singular and the recursion cannot start."
        )

    # `entry(k)` is the matrix value on diagonal `k`: below the main
    # diagonal it comes from `c`, above it from `r`.
    var x = List[Float64](length=n, fill=0.0)
    x[0] = Float64(rhs[0]) / diagonal

    # `n == 1` needs no special case: the recursion below is
    # `range(1, n)`, which is empty, and `x[0]` above is already the whole
    # answer.
    #
    # Forward and backward vectors of the leading system.
    var forward = List[Float64](length=n, fill=0.0)
    var backward = List[Float64](length=n, fill=0.0)
    forward[0] = 1.0 / diagonal
    backward[0] = 1.0 / diagonal

    for k in range(1, n):
        # Reflection coefficients for the two auxiliary vectors.
        # Appending a zero to `forward` and prepending one to `backward`
        # leaves each solving its own system in the *interior* rows of the
        # widened one -- a Toeplitz matrix's trailing `k x k` block is the
        # previous matrix. What each leaves behind is one stray entry, and
        # these are those two.
        #
        # `forward` is padded at the end, so its stray entry is the new
        # bottom row against it: `sum_j t[k - j] * f[j]`, and `t` below the
        # diagonal is `c`. `backward` is padded at the *front*, so its
        # stray entry is the new top row against it, which pairs `b[j]`
        # with `t[-(j + 1)]` -- `r[j + 1]`, indexed forwards. The two are
        # not mirror images, and pairing `backward` with `r[k - j]` by
        # symmetry with the line above is wrong for every non-palindromic
        # right-hand side.
        var forward_error = 0.0
        var backward_error = 0.0
        for j in range(k):
            forward_error += Float64(col[k - j]) * forward[j]
            backward_error += Float64(row[j + 1]) * backward[j]

        var denominator = 1.0 - forward_error * backward_error
        if denominator == 0:
            raise Error(
                "solve_toeplitz: the leading ",
                k + 1,
                "x",
                k + 1,
                (
                    " minor is singular. Levinson's recursion cannot pivot"
                    " around it -- build the matrix with"
                    " `numax.linalg.toeplitz` and use `solve`, which does."
                ),
            )

        # Both new vectors are combinations of the *same* two padded
        # vectors -- the old forward with a zero appended, and the old
        # backward with a zero prepended. Only the coefficients differ.
        var next_forward = List[Float64](length=n, fill=0.0)
        var next_backward = List[Float64](length=n, fill=0.0)
        for j in range(k + 1):
            var padded_forward = forward[j] if j < k else 0.0
            var padded_backward = backward[j - 1] if j >= 1 else 0.0
            next_forward[j] = (
                padded_forward - forward_error * padded_backward
            ) / denominator
            next_backward[j] = (
                padded_backward - backward_error * padded_forward
            ) / denominator
        for j in range(k + 1):
            forward[j] = next_forward[j]
            backward[j] = next_backward[j]

        # Extend the solution itself with the new backward vector.
        var residual = 0.0
        for j in range(k):
            residual += Float64(col[k - j]) * x[j]
        var scale = Float64(rhs[k]) - residual
        for j in range(k + 1):
            x[j] += scale * backward[j]

    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype](x[i]))
    return Static[dtype, n](ctx, out^)
