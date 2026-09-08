"""General solve, inverse and pseudo-inverse over `Array`.
`scipy.linalg`'s `solve`, `inv` and `pinv`.

**Tier 2** for `solve` and `inverse`, which pivot, and for `pinv`, which
goes through `svd`. The `Tensor` tier of the first two lives in
`numax.linalg.basic`; this is the register-resident, `FloatLike`-generic
half, and the half that differentiates.
"""

from std.collections import Array

from ...core.numeric import FloatLike, ge_indicator, guard_nonzero, max_of

from ..common import _PIVOT_FLOOR, _zeros
from .eigen import svd
from .lu import lu
from .triangular import back_substitution, forward_substitution


def solve[
    T: FloatLike, n: Int
](a: Array[T, n * n], b: Array[T, n]) -> Array[T, n]:
    """Solve `A @ x = b` by unpivoted LU followed by two substitutions.

    MAX ships no `solve` at any size. Past the crossover, use the `Tensor`
    overload of this name: blocked LU with partial pivoting, its trailing
    update in MAX's `matmul`, and `lu_factor` exposed separately so one
    factorization can serve several right-hand sides.
    """
    var factored = lu[T, n](a)
    var y = forward_substitution[T, n, unit_diagonal=True](factored, b)
    return back_substitution[T, n](factored, y)


def inverse[T: FloatLike, n: Int](a: Array[T, n * n]) -> Array[T, n * n]:
    """`A^-1`, by solving against each column of the identity.

    Factors once and substitutes `n` times rather than calling `solve` `n`
    times, which would redo the factorization for every column.

    MAX ships no `inv` at any size. Past the crossover, `lu_factor` over
    `Tensor` and call `solve` against each column of the identity. Note
    that inverting explicitly is rarely the
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
