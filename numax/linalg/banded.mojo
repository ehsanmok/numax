"""Solvers for banded and Toeplitz systems. `scipy.linalg`'s `solve_banded`
family.

**This module is tier 2 and `Plain`-only, and its pivoted eliminations
are host-side.** A banded elimination has no `O(n^3)` term: its work is
`O(n * bandwidth^2)`, spread over `n` sequential column steps that each
touch a `bandwidth x bandwidth` corner. There is no GEMM to hand anything
to, which is the same reason `numax.linalg.array.tridiagonal_solve` stays
where it is. So this is an **entry surface** -- it adds names a SciPy user
reaches for and a shape numax could not express, not a faster path to an
answer numax already had.

The one device path is the tridiagonal case, `solve_banded[l=1, u=1,
gpu=True]` and `solveh_banded[u=1, gpu=True]`: parallel cyclic
reduction, `ceil(log2 n)` launches that each eliminate every equation's
neighbors at once. It does not pivot, so it is for the systems that need
none -- diagonally dominant, as a spline's or a second-difference
operator's is, or symmetric positive definite -- and `numax.interpolate`
builds its splines on it. Every other bandwidth at `gpu=True` runs the host
elimination and says so on `stderr`.

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

`solve_circulant` is the exception to all of the above and is here anyway,
because a circulant *is* a Toeplitz matrix and a caller looking for one will
look here. It is not an elimination at all -- a circulant is diagonalized by
the DFT, so the solve is three transforms and a division, `O(n log n)`
rather than `O(n^2)`. It is the one routine in this module that is not
host-bound, and it is why `numax.linalg` depends on `numax.fft`: a
deliberate cross-subpackage edge, recorded in `CLAUDE.md` and
`docs/architecture.md` alongside the four that came before it.

## What is not here

Banded eigenvalues (`eig_banded`, `eigvals_banded`, `eigh_tridiagonal`) are
not here for the reason `numax/linalg/__init__.mojo` gives for the dense
spectral four: the reduction phase is tractable and the iterative phase is
a sequential sweep with data-dependent deflation.
"""

from std.math import sqrt as _sqrt

from max.gpu.host import DeviceContext

from ..core.tensorlike import TensorLike, TensorView, dim, is_row_major
from algorithm.rowwise_types import RowCoord
from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise

from ..core.rowwise import reduce_all
from ..core._drive import _notice
from ..core.tensor import _canonical, _same_order, Static, copy, zeros
from ..fft.fft import Spectrum, fft, ifft


def _tridiagonal_device[
    dtype: DType, n: Int
](
    var sub: Static[dtype, n],
    var diag: Static[dtype, n],
    var sup: Static[dtype, n],
    var rhs: Static[dtype, n],
    ctx: DeviceContext,
) raises -> Static[dtype, n]:
    """The tridiagonal system `sub[i] x[i-1] + diag[i] x[i] + sup[i] x[i+1]
    = rhs[i]` on the device, by parallel cyclic reduction.

    Each of `ceil(log2(n))` launches eliminates the neighbors `s` places
    away from every equation at once (`s = 1, 2, 4, ...`), between two sets
    of buffers, until every equation stands alone and `x = rhs / diag`.
    `O(n log n)` work, every step parallel. **No pivoting**, so it wants a
    system that needs none -- diagonally dominant, as a spline's is; a
    general tridiagonal system is `solve_banded`'s host path. `sub[0]` and
    `sup[n - 1]` are ignored.
    """
    var sub2 = Static[dtype, n]._uninitialized(ctx)
    var diag2 = Static[dtype, n]._uninitialized(ctx)
    var sup2 = Static[dtype, n]._uninitialized(ctx)
    var rhs2 = Static[dtype, n]._uninitialized(ctx)
    var s = 1
    while s < n:
        var a = sub.tile().as_unsafe_any_origin()
        var b = diag.tile().as_unsafe_any_origin()
        var c = sup.tile().as_unsafe_any_origin()
        var d = rhs.tile().as_unsafe_any_origin()
        var a2 = sub2.tile().as_unsafe_any_origin()
        var b2 = diag2.tile().as_unsafe_any_origin()
        var c2 = sup2.tile().as_unsafe_any_origin()
        var d2 = rhs2.tile().as_unsafe_any_origin()

        @always_inline
        def reduce[
            width: Int, alignment: Int = 1
        ](coord: Coord) {
            var a, var b, var c, var d, var a2, var b2, var c2, var d2, var s
        }:
            var i = coord_to_index_list(coord)[0]
            var ai = a[coord][0] if i > 0 else Scalar[dtype](0)
            var ci = c[coord][0] if i < n - 1 else Scalar[dtype](0)
            var nb = b[coord][0]
            var nd = d[coord][0]
            var na = Scalar[dtype](0)
            var nc = Scalar[dtype](0)
            if i - s >= 0:
                var alpha = -ai / b[Coord(i - s)][0]
                na = alpha * (
                    a[Coord(i - s)][0] if i - s > 0 else Scalar[dtype](0)
                )
                nb += alpha * c[Coord(i - s)][0]
                nd += alpha * d[Coord(i - s)][0]
            if i + s < n:
                var gamma = -ci / b[Coord(i + s)][0]
                nc = gamma * (
                    c[Coord(i + s)][0] if i + s < n - 1 else Scalar[dtype](0)
                )
                nb += gamma * a[Coord(i + s)][0]
                nd += gamma * d[Coord(i + s)][0]
            a2.store[1](coord, na)
            b2.store[1](coord, nb)
            c2.store[1](coord, nc)
            d2.store[1](coord, nd)

        elementwise[simd_width=1, target="gpu"](reduce, Coord(n), ctx)
        swap(sub, sub2)
        swap(diag, diag2)
        swap(sup, sup2)
        swap(rhs, rhs2)
        s *= 2
    var x = Static[dtype, n]._uninitialized(ctx)
    var bv = diag.tile().as_unsafe_any_origin()
    var dv = rhs.tile().as_unsafe_any_origin()
    var xv = x.tile()

    @always_inline
    def finish[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var bv, var dv, var xv}:
        xv.store[1](coord, dv[coord][0] / bv[coord][0])

    elementwise[simd_width=1, target="gpu"](finish, Coord(n), ctx)
    ctx.synchronize()
    return x^


def solve_banded[
    A: TensorLike,
    B: TensorLike,
    l: Int,
    u: Int,
    gpu: Bool = False,
](ab: A, b: B) raises -> Static[A.dtype, dim[A, 1]] where (
    (A.dtype.is_floating_point() and l >= 0 and u >= 0 and dim[A, 1] >= 1)
    and A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and dim[A, 0] == l + u + 1
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 1]
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

    At `gpu=True` with `l == u == 1` and `ab` on a device, the solve stays
    there: cyclic reduction **without pivoting**, so the system must not
    need it (diagonal dominance suffices), and a zero pivot is not
    detected. Any other bandwidth at `gpu=True` takes the host path above
    and prints the `_drive` notice.
    """
    comptime n = dim[A, 1]
    var ctx = ab.context()
    comptime if gpu and l == 1 and u == 1:
        if not ab.on_host():
            # The tridiagonal case on the device, by cyclic reduction
            # without pivoting (`_tridiagonal_device` says what it wants of
            # the system); other bandwidths stay on the host.
            var sub = Static[A.dtype, n]._uninitialized(ctx)
            var diag = Static[A.dtype, n]._uninitialized(ctx)
            var sup = Static[A.dtype, n]._uninitialized(ctx)
            var abv = ab.tile().as_unsafe_any_origin()
            var sv = sub.tile()
            var dv = diag.tile()
            var uv = sup.tile()

            @always_inline
            def split[
                width: Int, alignment: Int = 1
            ](coord: Coord) {var abv, var sv, var dv, var uv}:
                var j = coord_to_index_list(coord)[0]
                dv.store[1](coord, abv[Coord(1, j)][0])
                sv.store[1](
                    coord,
                    abv[Coord(2, j - 1)][0] if j > 0 else Scalar[A.dtype](0),
                )
                uv.store[1](
                    coord,
                    abv[Coord(0, j + 1)][0] if j
                    < n - 1 else Scalar[A.dtype](0),
                )

            elementwise[simd_width=1, target="gpu"](split, Coord(n), ctx)
            return _tridiagonal_device[A.dtype, n](
                sub^,
                diag^,
                sup^,
                _same_order(
                    _canonical[n, dtype=A.dtype](b),
                    Static[A.dtype, n]._static_layout(),
                ),
                ctx,
            )
    elif gpu:
        if not ab.on_host():
            _notice[gpu]("solve_banded")
    var band = ab.to_host()
    var rhs = b.to_host[A.dtype]()

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

    var out = List[Scalar[A.dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[A.dtype](x[i]))
    return Static[A.dtype, n](ctx, out^)


def cholesky_banded[
    T: TensorLike,
    u: Int,
    lower: Bool = False,
](ab: T) raises -> Static[T.dtype, u + 1, dim[T, 1]] where (
    (T.dtype.is_floating_point() and u >= 0 and dim[T, 1] >= 1)
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
    and dim[T, 0] == u + 1
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
    comptime n = dim[T, 1]
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

    var out = List[Scalar[T.dtype]](length=(u + 1) * n, fill=0)
    for d in range(u + 1):
        for j in range(n):
            if lower:
                out[d * n + j] = Scalar[T.dtype](band[d * n + j])
            elif j + d < n:
                out[(u - d) * n + j + d] = Scalar[T.dtype](band[d * n + j])
    return Static[T.dtype, u + 1, n](ctx, out^)


def cho_solve_banded[
    A: TensorLike,
    B: TensorLike,
    u: Int,
    lower: Bool = False,
](cb: A, b: B) raises -> Static[A.dtype, dim[A, 1]] where (
    (A.dtype.is_floating_point() and u >= 0 and dim[A, 1] >= 1)
    and A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and dim[A, 0] == u + 1
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 1]
):
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
    comptime n = dim[A, 1]
    var ctx = cb.context()
    var host = cb.to_host()
    var rhs = b.to_host[A.dtype]()

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

    var out = List[Scalar[A.dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[A.dtype](x[i]))
    return Static[A.dtype, n](ctx, out^)


def solveh_banded[
    A: TensorLike,
    B: TensorLike,
    u: Int,
    lower: Bool = False,
    gpu: Bool = False,
](ab: A, b: B) raises -> Static[A.dtype, dim[A, 1]] where (
    (A.dtype.is_floating_point() and u >= 0 and dim[A, 1] >= 1)
    and A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and dim[A, 0] == u + 1
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 1]
):
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

    At `gpu=True` with `u == 1` and `ab` on a device, the symmetric
    tridiagonal system is solved there by the cyclic reduction
    `solve_banded` uses: no pivoting, which a symmetric positive definite
    matrix does not need, since odd-even elimination is a symmetric
    permutation and keeps every reduced system positive definite. Wider
    bands at `gpu=True` take the host factorization and print the
    `_drive` notice.
    """
    comptime n = dim[A, 1]
    var ctx = ab.context()
    comptime if gpu and u == 1:
        if not ab.on_host():
            var sub = Static[A.dtype, n]._uninitialized(ctx)
            var diag = Static[A.dtype, n]._uninitialized(ctx)
            var sup = Static[A.dtype, n]._uninitialized(ctx)
            var abv = ab.tile().as_unsafe_any_origin()
            var sv = sub.tile()
            var dv = diag.tile()
            var uv = sup.tile()

            @always_inline
            def split[
                width: Int, alignment: Int = 1
            ](coord: Coord) {var abv, var sv, var dv, var uv}:
                var j = coord_to_index_list(coord)[0]
                var zero = Scalar[A.dtype](0)
                comptime if lower:
                    dv.store[1](coord, abv[Coord(0, j)][0])
                    sv.store[1](
                        coord, abv[Coord(1, j - 1)][0] if j > 0 else zero
                    )
                    uv.store[1](
                        coord, abv[Coord(1, j)][0] if j < n - 1 else zero
                    )
                else:
                    dv.store[1](coord, abv[Coord(1, j)][0])
                    sv.store[1](coord, abv[Coord(0, j)][0] if j > 0 else zero)
                    uv.store[1](
                        coord, abv[Coord(0, j + 1)][0] if j < n - 1 else zero
                    )

            elementwise[simd_width=1, target="gpu"](split, Coord(n), ctx)
            return _tridiagonal_device[A.dtype, n](
                sub^,
                diag^,
                sup^,
                _same_order(
                    _canonical[n, dtype=A.dtype](b),
                    Static[A.dtype, n]._static_layout(),
                ),
                ctx,
            )
    elif gpu:
        if not ab.on_host():
            _notice[gpu]("solveh_banded")
    var factor = cholesky_banded[u=u, lower=lower](ab)
    return cho_solve_banded[u=u, lower=lower](factor, b)


def solve_toeplitz[
    A: TensorLike,
    B: TensorLike,
    C: TensorLike,
](c: A, r: B, b: C) raises -> Static[A.dtype, dim[A, 0]] where (
    (A.dtype.is_floating_point() and dim[A, 0] >= 1)
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 0]
    and C.dtype == A.dtype
    and C.LayoutType.rank == 1
    and C.LayoutType.all_dims_known
    and dim[C, 0] == dim[A, 0]
):
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
    comptime n = dim[A, 0]
    var ctx = c.context()
    var col = c.to_host()
    var row = r.to_host[A.dtype]()
    var rhs = b.to_host[A.dtype]()

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
        # Reflection coefficients for the two auxiliary vectors. `forward`
        # is padded at the end, so its stray entry pairs `f[j]` with
        # `t[k - j]`; `backward` is padded at the *front*, so its stray
        # entry pairs `b[j]` with `r[j + 1]`, indexed forwards. The two are
        # not mirror images: pairing `backward` with `r[k - j]` by symmetry
        # is wrong for every non-palindromic right-hand side.
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

    var out = List[Scalar[A.dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[A.dtype](x[i]))
    return Static[A.dtype, n](ctx, out^)


def _circulant_divide[
    n: Int, dtype: DType
](
    var c_spectrum: Spectrum[dtype, n],
    var b_spectrum: Spectrum[dtype, n],
    ctx: DeviceContext,
) raises -> Static[dtype, n] where (dtype.is_floating_point() and n > 0):
    """`solve_circulant`'s spectral division on the device: one launch
    divides `fft(b)` by `fft(c)` in place and writes each `|fft(c)_k|^2`,
    a device `min` over those checks for a zero eigenvalue, and the inverse
    transform's real half is the answer."""
    var cr = c_spectrum[0].tile().as_unsafe_any_origin()
    var ci = c_spectrum[1].tile().as_unsafe_any_origin()
    var br = b_spectrum[0].tile().as_unsafe_any_origin()
    var bi = b_spectrum[1].tile().as_unsafe_any_origin()
    var magnitudes = Static[dtype, n]._uninitialized(ctx)
    var mv = magnitudes.tile()

    @always_inline
    def divide[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var cr, var ci, var br, var bi, var mv}:
        var a = cr[coord][0]
        var b = ci[coord][0]
        var p = br[coord][0]
        var q = bi[coord][0]
        var size = a * a + b * b
        mv.store[1](coord, size)
        br.store[1](coord, (p * a + q * b) / size)
        bi.store[1](coord, (q * a - p * b) / size)

    elementwise[simd_width=1, target="gpu"](divide, Coord(n), ctx)
    var smallest = Static[dtype, 1](ctx)

    @always_inline
    def identity[
        w: Int
    ](tile: SIMD[dtype, w], idx: RowCoord[1]) {} -> SIMD[dtype, w]:
        return tile

    reduce_all[monoid="min", target="gpu"](
        magnitudes.tile(), smallest.tile(), identity, n, Optional(ctx)
    )
    if smallest.to_host()[0] == 0:
        raise Error(
            "solve_circulant: the matrix is singular -- an eigenvalue of the"
            " circulant is zero. Its eigenvalues are exactly the entries of"
            " fft(c)."
        )
    _ = c_spectrum^
    var solved = ifft[dtype, n, True](b_spectrum^)
    return _same_order(solved[0], Static[dtype, n]._static_layout())


def solve_circulant[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](c: A, b: B) raises -> Static[A.dtype, dim[A, 0]] where (
    (A.dtype.is_floating_point() and dim[A, 0] > 0)
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 0]
):
    """Solve `a @ x = b` where `a` is the circulant matrix with first column
    `c`. `scipy.linalg.solve_circulant`.

    Every circulant is diagonalized by the DFT, so this is not an
    elimination at all:

    ```
    x = ifft(fft(b) / fft(c))
    ```

    Three transforms and an elementwise division -- `O(n log n)` against a
    dense solve's `O(n^3)` and `solve_toeplitz`'s `O(n^2)`, and the matrix
    is never built. `numax.linalg.circulant` is the constructor for when it
    is wanted as a matrix.

    Any `n > 0`, as SciPy's. A power of two runs `numax.fft`'s radix-2
    engine and every other length its Bluestein path, which costs three
    power-of-two transforms of `next_fast_len(2n - 1)` per transform here
    -- still `O(n log n)`, at a larger constant. A system that is free to
    be padded is cheaper at `next_fast_len(n)`; one that is not no longer
    has to be.

    A zero in `fft(c)` means the circulant is singular: one of its
    eigenvalues, which are exactly the entries of `fft(c)`, is zero. That
    raises rather than returning infinities. SciPy offers a least-squares
    answer there instead; numax does not, since the pseudoinverse route
    would need a tolerance policy this module has nowhere to state.

    The division is `O(n)` against the transforms' `O(n log n)` and needs
    complex arithmetic `numax.core.ops` does not carry, so it is written
    out over the real/imaginary pair: a host loop, or at `gpu=True` one
    device launch with the singularity check as a device `min` over the
    eigenvalue magnitudes, so nothing crosses to the host.
    """
    comptime n = dim[A, 0]
    var ctx = c.context()

    # `fft` consumes its `Spectrum`, and `c` and `b` are borrowed, so each
    # needs an owned copy. `copy` is the explicit spelling `Tensor` requires
    # -- it is `Movable` and not `Copyable` precisely so that duplicating a
    # buffer is a decision rather than something that happens silently.
    var c_spectrum = fft[A.dtype, n, gpu](
        Spectrum[A.dtype, n](
            _same_order(c, Static[A.dtype, n]._static_layout()),
            zeros[A.dtype, n](ctx),
        )
    )
    var b_spectrum = fft[A.dtype, n, gpu](
        Spectrum[A.dtype, n](
            _same_order(
                _canonical[n, dtype=A.dtype](b),
                Static[A.dtype, n]._static_layout(),
            ),
            zeros[A.dtype, n](ctx),
        )
    )
    comptime if gpu:
        if not c.on_host():
            return _circulant_divide[n](c_spectrum^, b_spectrum^, ctx)

    var c_real = c_spectrum[0].to_host()
    var c_imag = c_spectrum[1].to_host()
    var b_real = b_spectrum[0].to_host()
    var b_imag = b_spectrum[1].to_host()

    var quotient_real = List[Scalar[A.dtype]](capacity=n)
    var quotient_imag = List[Scalar[A.dtype]](capacity=n)
    for k in range(n):
        var cr = Float64(c_real[k])
        var ci = Float64(c_imag[k])
        var magnitude = cr * cr + ci * ci
        if magnitude == 0:
            raise Error(
                "solve_circulant: the matrix is singular -- eigenvalue ",
                k,
                (
                    " of the circulant is zero. Its eigenvalues are exactly"
                    " the entries of fft(c)."
                ),
            )
        var br = Float64(b_real[k])
        var bi = Float64(b_imag[k])
        quotient_real.append(Scalar[A.dtype]((br * cr + bi * ci) / magnitude))
        quotient_imag.append(Scalar[A.dtype]((bi * cr - br * ci) / magnitude))

    var solved = ifft[A.dtype, n, gpu](
        Spectrum[A.dtype, n](
            Static[A.dtype, n](ctx, quotient_real^),
            Static[A.dtype, n](ctx, quotient_imag^),
        )
    )
    # The imaginary part is zero to rounding, `a` and `b` both being real,
    # so the real half is the answer. Copied out of the tuple rather than
    # moved, since a `Tuple` element cannot be transferred out of a
    # temporary.
    var real_part = solved[0].to_host()
    return Static[A.dtype, n](ctx, real_part^)
