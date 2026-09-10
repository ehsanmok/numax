"""Functions *of* a matrix over `Array`. `scipy.linalg`'s `_matfuncs`.

**Tier 1**, both of them, and that is the point of this tier existing for
these. `expm` here takes its squaring count as a compile-time parameter
rather than deriving it from a norm, so its work is fixed and nothing
branches on a value -- which means it differentiates at `Dual` and runs
inside a GPU kernel body, one matrix exponential per SIMD lane. The
`Tensor` overload in `numax.linalg.matfuncs` picks the count from
`norm(a)` and is tier 2 for exactly that reason.

The cost of the fixed count is that the caller chooses it, and `expm`'s
docstring says how.

`sqrtm` is here and **symmetric positive definite only**, because that is
the case `eigh` already solves. The general matrix square root needs a
Schur decomposition, which numax does not have --
`numax.linalg.matfuncs` records what that blocks and what unblocks it.
"""

from std.collections import Array

from ...core.numeric import FloatLike

from ..common import _zeros
from .blas import matmul
from .eigen import eigh
from .lu import lu
from .triangular import back_substitution, forward_substitution


comptime _B0 = 64764752532480000.0
comptime _B1 = 32382376266240000.0
comptime _B2 = 7771770303897600.0
comptime _B3 = 1187353796428800.0
comptime _B4 = 129060195264000.0
comptime _B5 = 10559470521600.0
comptime _B6 = 670442572800.0
comptime _B7 = 33522128640.0
comptime _B8 = 1323241920.0
comptime _B9 = 40840800.0
comptime _B10 = 960960.0
comptime _B11 = 16380.0
comptime _B12 = 182.0
comptime _B13 = 1.0
"""The degree-13 Pade coefficients, Higham's. Only their ratios matter;
`numax.linalg.matfuncs` carries the full note."""


def _combine[
    T: FloatLike, n: Int
](a: Array[T, n * n], scale: Float64, b: Array[T, n * n]) -> Array[T, n * n]:
    """`scale * a + b`, entrywise. The one operation the Pade evaluation
    below does over and over, and there is no tensor arithmetic at this
    tier to reach for."""
    var out = _zeros[T, n * n]()
    var factor = T.constant(scale)
    for i in range(n * n):
        out[i] = factor * a[i] + b[i]
    return out^


def expm[
    T: FloatLike, n: Int, squarings: Int = 8
](a: Array[T, n * n]) -> Array[T, n * n] where squarings >= 0:
    """The matrix exponential of `a`, at a **fixed** squaring count.
    `scipy.linalg.expm`.

    Scaling and squaring with a degree-13 Pade approximant, the same
    algorithm as the `Tensor` overload, with one deliberate difference:
    `squarings` is a compile-time parameter rather than a number derived
    from `norm(a)`. That keeps the work fixed and the control flow
    branchless, so this is **tier 1** -- it differentiates at `Dual`, giving
    the derivative of a matrix exponential with no adjoint rule written
    anywhere, and it runs inside a `map[gpu=True]` kernel body, one
    exponential per SIMD lane.

    **Choosing `squarings`.** The approximant is accurate to working
    precision when `||a / 2^squarings||_1` is at most about 5.4. So the
    right value is `ceil(log2(||a||_1 / 5.4))`, and the default of 8 covers
    `||a||_1` up to roughly 1370 -- generous for the small matrices this
    tier is for. Too few is inaccurate; too many costs one matrix product
    each and loses a little accuracy to repeated squaring, so it is not
    free to over-provide. The `Tensor` overload computes it, at the price of
    being tier 2.

    A caller who wants the count chosen automatically wants
    `numax.linalg.expm`. A caller who wants a derivative wants this one.
    """
    comptime scale = 1.0 / Float64(1 << squarings)
    var scaled = _zeros[T, n * n]()
    var factor = T.constant(scale)
    for i in range(n * n):
        scaled[i] = factor * a[i]

    var identity = _zeros[T, n * n]()
    for i in range(n):
        identity[i * n + i] = T.one()

    var a2 = matmul[T, n](scaled, scaled)
    var a4 = matmul[T, n](a2, a2)
    var a6 = matmul[T, n](a4, a2)

    # Odd half: U = A (b13 A^6 + b11 A^4 + b9 A^2) A^6 + (b7 A^6 + b5 A^4 +
    # b3 A^2 + b1 I), with the powers shared between the halves.
    var odd_inner = _combine[T, n](
        a6, _B13, _combine[T, n](a4, _B11, _scaled_copy[T, n](a2, _B9))
    )
    var odd_outer = _combine[T, n](
        a6,
        _B7,
        _combine[T, n](
            a4, _B5, _combine[T, n](a2, _B3, _scaled_copy[T, n](identity, _B1))
        ),
    )
    var odd_part = _combine[T, n](matmul[T, n](a6, odd_inner), 1.0, odd_outer)
    var u = matmul[T, n](scaled, odd_part)

    var even_inner = _combine[T, n](
        a6, _B12, _combine[T, n](a4, _B10, _scaled_copy[T, n](a2, _B8))
    )
    var even_outer = _combine[T, n](
        a6,
        _B6,
        _combine[T, n](
            a4, _B4, _combine[T, n](a2, _B2, _scaled_copy[T, n](identity, _B0))
        ),
    )
    var v = _combine[T, n](matmul[T, n](a6, even_inner), 1.0, even_outer)

    # `(V - U) r = V + U`, one unpivoted factorization and `n` substitution
    # pairs. Unpivoted keeps the trip count fixed, which is what keeps this
    # tier 1; `V - U` is well conditioned after the scaling, which is what
    # makes that safe.
    var denominator = _zeros[T, n * n]()
    var numerator = _zeros[T, n * n]()
    for i in range(n * n):
        denominator[i] = v[i] - u[i]
        numerator[i] = v[i] + u[i]

    var factored = lu[T, n](denominator)
    var result = _zeros[T, n * n]()
    for column in range(n):
        var rhs = _zeros[T, n]()
        for i in range(n):
            rhs[i] = numerator[i * n + column].copy()
        var y = forward_substitution[T, n, unit_diagonal=True](factored, rhs)
        var x = back_substitution[T, n](factored, y)
        for i in range(n):
            result[i * n + column] = x[i].copy()

    comptime for _ in range(squarings):
        result = matmul[T, n](result, result)
    return result^


def _scaled_copy[
    T: FloatLike, n: Int
](a: Array[T, n * n], scale: Float64) -> Array[T, n * n]:
    """`scale * a`, entrywise. The base case of the `_combine` chains."""
    var out = _zeros[T, n * n]()
    var factor = T.constant(scale)
    for i in range(n * n):
        out[i] = factor * a[i]
    return out^


def sqrtm[
    T: FloatLike, n: Int, sweeps: Int = 12
](a: Array[T, n * n]) -> Array[T, n * n]:
    """The **symmetric positive definite** matrix square root:
    `Q diag(sqrt(w)) Q.T` from `eigh`. `scipy.linalg.sqrtm`, restricted.

    The restriction is stated rather than checked, and that is a deliberate
    choice about what this tier can promise. A symmetric matrix with a
    negative eigenvalue has a square root that is complex; a non-symmetric
    one has a square root this route does not compute at all, since `eigh`
    assumes symmetry and reads only one triangle. Neither case raises --
    tier 1 has no channel to raise through, and a `where` clause cannot see
    a run-time matrix. A negative eigenvalue produces a NaN through `sqrt`,
    which is at least loud.

    So: this is `sqrtm` for the case a Cholesky-shaped matrix is in hand,
    and `numax.linalg.matfuncs` records what the general case is waiting
    for (a Schur decomposition) and why that is a research-grade item
    rather than a missing overload.

    Tier 1, inheriting `eigh`'s fixed `sweeps` count. `X @ X == A` to the
    accuracy of the Jacobi diagonalization, which the tests check directly.
    """
    var factored = eigh[T, n, sweeps](a)
    var values = factored[0].copy()
    var vectors = factored[1].copy()

    var roots = _zeros[T, n]()
    for i in range(n):
        roots[i] = values[i].sqrt()

    # `Q diag(sqrt(w)) Q.T`, with `Q`'s columns the eigenvectors.
    var out = _zeros[T, n * n]()
    for i in range(n):
        for j in range(n):
            var total = T.constant(0.0)
            for k in range(n):
                total = (
                    total + vectors[i * n + k] * roots[k] * vectors[j * n + k]
                )
            out[i * n + j] = total^
    return out^
