"""Householder QR and least squares over `Array`.
`numpy.linalg.qr` and `scipy.linalg.lstsq`.

**Tier 1**: a fixed number of reflectors, no branching, so it
differentiates and runs per SIMD lane inside a GPU kernel. `qr` forms `Q`
explicitly and returns `(R, Q)`, which is the wasteful choice at large `n`
and the right one at the sizes an `Array` holds.

The blocked `Tensor` tier is `numax.linalg.qr`, and it is shaped
differently on purpose: `qr_factor` keeps the reflectors in a `TensorQR`
so `Q` need never be formed. `lstsq` has no `Tensor` overload of its own
because `TensorQR.solve` is it.
"""

from std.collections import Array

from ...core.numeric import FloatLike, guard_nonzero

from ..common import _PIVOT_FLOOR, _zeros


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

    **The one function here MAX has a counterpart for, and it is denied.**
    `linalg.qr_factorization` is a LAPACK-style in-place Householder
    factorization, but it is over the older `LayoutTensor`, which numax
    does not bridge to -- interop is `TileTensor` only, so that this
    library has exactly one owning tensor type and one view type. Three
    other things would argue against it even without that rule: it is
    monomorphic in `dtype` (no `Dual` passes through it, which is the whole
    reason this version exists), it is a CPU-only scalar-loop reference
    rather than a tuned kernel, and it returns reflectors plus a `sigma`
    vector rather than an explicit `Q` (`apply_q`/`form_q` alongside it are
    how a caller gets `Q`'s action or `Q` itself).

    So there is no `Tensor` overload of `qr` yet. It is the next blocked
    factorization to write, in the shape `cholesky` and `lu_factor`
    already have -- panel reflectors, trailing update through MAX's
    `matmul` -- and until it lands, a large QR is genuinely missing rather
    than one import away.
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
