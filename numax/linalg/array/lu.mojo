"""LU factorization and the determinant over `Array`.
`scipy.linalg`'s `lu`/`lu_factor` pair.

Two tiers *within* this module. `lu` and `det` are **tier 1**: unpivoted,
fixed trip count, and they differentiate. `lu_factor` and the `PivotedLU`
it returns are **tier 2** and `Plain`-only, because choosing a pivot by
magnitude is a branch on data -- which buys the matrices unpivoted `lu`
cannot factor at the cost of the GPU and of differentiability.

The blocked, device-resident `Tensor` tier is `numax.linalg.lu`, whose
`TensorLU` is this module's `PivotedLU` with its factors in device memory.
"""

from std.collections import Array

from ...core.numeric import FloatLike, guard_nonzero
from ...core.plain import Plain

from ..common import _PIVOT_FLOOR, _zeros
from .triangular import back_substitution, forward_substitution


def lu[T: FloatLike, n: Int](a: Array[T, n * n]) -> Array[T, n * n]:
    """Doolittle `LU` without pivoting, packed into one matrix.

    The strict lower triangle holds `L` (whose diagonal is an implicit
    `1`), and the upper triangle including the diagonal holds `U`. Packing
    them avoids returning two matrices where the two halves never overlap.

    See this module's docstring for what "without pivoting" costs.

    MAX ships no `lu` at any size. Past the crossover, use `lu_factor` over
    `Tensor`: blocked, partially pivoted, trailing update in MAX's
    `matmul`. It pivots, so it also factors matrices this cannot start on.
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


def det[T: FloatLike, n: Int](a: Array[T, n * n]) -> T:
    """The determinant, as the product of the unpivoted LU's diagonal.

    Unpivoted, so the sign is always the product's own -- there is no row
    swap count to correct for.

    MAX ships no `det` at any size. Past the crossover, the `Tensor`
    overload of this name pivots and corrects for the swap parity this one
    has no swaps to correct for.
    """
    var factored = lu[T, n](a)
    var product = T.one()
    for i in range(n):
        product = product * factored[i * n + i]
    return product^
