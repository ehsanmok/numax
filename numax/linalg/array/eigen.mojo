"""Eigenvalues, eigenvectors and the SVD.
`scipy.linalg`'s `_decomp`/`_decomp_svd`, `numpy.linalg`'s `eigh`/`svd`.

**Tier 1, `Array`-only.** MAX ships no eigensolver and no SVD at any size,
so there is nothing to delegate to, and no `Tensor` tier yet -- past the
crossover these are genuinely missing rather than one import away.

Fixed trip count is what keeps them tier 1 and therefore GPU-launchable at
any conformer, and it is also what bounds them. `eigh` and `svd` run a
fixed number of Jacobi sweeps rather than iterating to a tolerance;
`eigvals` runs shifted QR on a Hessenberg form for a fixed number of
sweeps and blends the 1x1/2x2 block roots branchlessly, returning
`Complex` because a real matrix's spectrum need not be real.

`eigh` assumes symmetry and does not check it -- checking would be a
per-lane branch. Pass a symmetric matrix or get an answer about its
symmetric part.
"""

from std.collections import Array

from ...core.numeric import (
    FloatLike,
    blend,
    ge_indicator,
    guard_nonzero,
    max_of,
)
from ...core.complex import Complex

from ..common import _PIVOT_FLOOR, _zeros


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
