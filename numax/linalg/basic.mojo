"""General solves and inverses: `scipy.linalg._basic`.

**`solve` has two tiers under one name**, resolved by argument type.

Over `Tensor` it factors with the blocked pivoted `lu_factor` -- whose
trailing update is MAX's GEMM -- and substitutes. It pivots, so it solves
systems the `Array` overload cannot start on. Tier 2. Reach for
`lu_factor` directly and reuse the `TensorLU` when there is more than one
right-hand side; this spelling throws the factorization away.

Over `Array[T, n*n]` it is unpivoted LU plus two substitutions, tier 1, and
differentiable at `Dual`.

MAX ships no `solve`, no `inv` and no pseudo-inverse at any size, so
nothing here delegates.

`inverse` factors once and substitutes `n` times rather than calling
`solve` `n` times. Note that inverting explicitly is rarely the right move
at any size: solving against a specific right-hand side is both cheaper and
better conditioned. `pinv` goes through the SVD and is the answer for the
underdetermined and rank-deficient cases `lstsq` refuses.
"""

from std.collections import Array

from ..core.array import Shaped
from ..core.numeric import FloatLike, ge_indicator, guard_nonzero, max_of

from .common import _PIVOT_FLOOR, _zeros
from .eigen import svd
from .lu import lu, lu_factor
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


def solve[
    dtype: DType, n: Int, gpu: Bool = False, block: Int = 64
](mut a: Shaped[dtype, n, n], mut b: Shaped[dtype, n]) raises -> Shaped[
    dtype, n
] where dtype.is_floating_point():
    """**Tier 2.** `x` with `a @ x == b`. `scipy.linalg.solve`.

    Factors with `lu_factor` -- blocked, partially pivoted, trailing update
    in MAX -- and substitutes. Call `lu_factor` directly and reuse the
    `TensorLU` when there is more than one right-hand side; this spelling
    throws the factorization away.

    Unlike the `Array[T, n*n]` sibling, this pivots, so it solves systems
    that one cannot start on. The trade is the GPU and the generic `T`:
    picking a row by magnitude is a branch on data.
    """
    var factorization = lu_factor[dtype, n, gpu, block](a)
    return factorization.solve(b)


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
