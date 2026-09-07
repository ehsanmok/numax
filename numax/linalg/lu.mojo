"""LU factorization, pivoted and not, and the determinant.
`scipy.linalg`'s `lu`/`lu_factor`/`lu_solve` and `det`.

MAX ships no LU at any size, so everything here is numax's.

**Three things, because pivoting is a branch on data.** `lu` is tier 1 and
unpivoted, so it runs at any conformer inside a GPU thread and fails on a
matrix with a zero pivot even when that matrix is perfectly well
conditioned -- `[[0, 1], [1, 0]]` is the standard example. `PivotedLU`
(from `lu_factor` over `Array`) pivots properly and gives that up: it is
`Plain[dtype, 1]` at width 1, because a SIMD `T` holds several matrices
whose lanes would want different pivot orders and there is no single order
to pick. `TensorLU` (from `lu_factor` over `Tensor`) pivots too and is
blocked, so its trailing `L21 @ U12` update goes to `blas.matmul`.

Both factorization objects carry the solves that reuse them, which is the
point of returning a factorization rather than a solution: one
factorization, several right-hand sides.

Rank deficiency is not detected at any tier. A singular matrix factors to a
zero pivot; `cond` on the original matrix is the check.
"""

from max.gpu.host import DeviceContext
from std.collections import Array

from ..core.array import Shaped, _context
from ..core.numeric import FloatLike, guard_nonzero
from ..core.plain import Plain

from .blas import matmul
from .common import _PIVOT_FLOOR, _staged, _zeros
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


struct TensorLU[dtype: DType, n: Int](Movable where dtype.is_floating_point()):
    """**Tier 2.** A blocked `LU` factorization of a `Tensor`, with partial
    pivoting, and the solves that reuse it.

    The `Tensor` counterpart of `PivotedLU`, and separate from it for the
    same reason `matmul` has two overloads: this one is `dtype`-monomorphic
    and its matrix has its own storage, that one is `FloatLike`-generic and
    its matrix lives in registers.

    MAX has no LU, so the factorization is numax's -- but blocked, so its
    cubic term is a matrix product and goes to MAX's `matmul`. Each step
    factors a `block`-wide panel with partial pivoting, solves the block
    row to its right, and subtracts `L21 @ U12` from what remains.

    Pivoting is why this is host-side and cannot be tier 1: choosing a row
    by the magnitude of a value is a branch on data. It is also what makes
    it correct where the unpivoted `lu` is not -- the exchange matrix
    `[[0, 1], [1, 0]]` has a zero leading pivot and factors fine here.

    Rank deficiency is still not reported: a singular matrix factors to a
    zero pivot and `solve` divides by it, so the result is not finite
    rather than being flagged. `cond` on the original matrix is the check.
    """

    var factored: List[Scalar[Self.dtype]]
    """`L` below the diagonal (its own diagonal an implicit `1`) and `U` on
    and above it, row-major, packed the way `PivotedLU` packs them.

    A `List` rather than a `Tensor` because every use of it -- the two
    substitutions in `solve`, the diagonal product in `det` -- walks
    elements one at a time in an order fixed by the previous element. There
    is no kernel to hand this to, so it stays where it is addressable, and
    `to_tensor_factored` exists for a caller who wants it on a device
    anyway.
    """

    var permutation: List[Int]
    """Row `i` of the factored matrix is row `permutation[i]` of the
    original."""

    var sign: Int
    """`+1` or `-1`, the parity of the row swaps; `det`'s sign."""

    def __init__(
        out self,
        var factored: List[Scalar[Self.dtype]],
        var permutation: List[Int],
        sign: Int,
    ):
        self.factored = factored^
        self.permutation = permutation^
        self.sign = sign

    def to_tensor_factored(
        self, ctx: Optional[DeviceContext] = None
    ) raises -> Shaped[Self.dtype, Self.n, Self.n]:
        """The packed `L`/`U` as a tensor, for inspection or reuse."""
        return Shaped[Self.dtype, Self.n, Self.n](
            _context(ctx), self.factored.copy()
        )

    def solve(
        self, mut b: Shaped[Self.dtype, Self.n]
    ) raises -> Shaped[Self.dtype, Self.n]:
        """`x` with `A @ x == b`, reusing this factorization.

        `scipy.linalg.lu_solve`. Permute, forward-substitute through `L`
        (unit diagonal, so no division), back-substitute through `U`. Both
        substitutions are inherently sequential -- element `i` needs
        element `i - 1` -- so there is no kernel to delegate to and no MAX
        call here. The cubic work already happened in the factorization,
        which is where MAX was.
        """
        var rhs = b.to_host()
        var x = List[Scalar[Self.dtype]](length=Self.n, fill=0)
        for i in range(Self.n):
            x[i] = rhs[self.permutation[i]]

        for i in range(Self.n):
            var acc = x[i]
            for p in range(i):
                acc -= self.factored[i * Self.n + p] * x[p]
            x[i] = acc

        for step in range(Self.n):
            var i = Self.n - 1 - step
            var acc = x[i]
            for p in range(i + 1, Self.n):
                acc -= self.factored[i * Self.n + p] * x[p]
            x[i] = acc / self.factored[i * Self.n + i]

        return Shaped[Self.dtype, Self.n](b.context(), x^)

    def det(self) raises -> Scalar[Self.dtype]:
        """`det(A)`: the product of `U`'s diagonal, times the swap parity."""
        var product = Scalar[Self.dtype](self.sign)
        for i in range(Self.n):
            product *= self.factored[i * Self.n + i]
        return product


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


def lu_factor[
    dtype: DType, n: Int, gpu: Bool = False, block: Int = 64
](mut a: Shaped[dtype, n, n]) raises -> TensorLU[
    dtype, n
] where dtype.is_floating_point():
    """**Tier 2.** Factor `a` into `P @ L @ U`, blocked.
    `scipy.linalg.lu_factor`.

    Right-looking and blocked: factor a `block`-wide panel with partial
    pivoting, solve the block row to its right against `L11`, then subtract
    `L21 @ U12` from the trailing submatrix. That subtraction is the cubic
    term and it is a matrix product, so it is MAX's `matmul`; the panel and
    the block-row solve are `O(n * block^2)`.

    Row interchanges are applied across the full width of the matrix as
    they are chosen, rather than being recorded and replayed over the
    columns outside the panel afterwards. Same result, one less pass.

    See `TensorLU` for what pivoting costs and what it buys, and
    `cholesky` for the same blocking on a symmetric matrix, where pivoting
    is unnecessary. The `Array[T, n*n]` sibling `lu_factor` is the one to
    call at a conformer other than a raw `dtype`.

    **Ceiling.** As in `cholesky`, the panel runs on the host and each step
    stages its blocks down and its update back, because numax has no
    strided device sub-view to hand MAX. The flops are MAX's, the copies
    are not.
    """
    var ctx = a.context()
    var work = a.to_host()
    var permutation = List[Int](length=n, fill=0)
    for i in range(n):
        permutation[i] = i
    var sign = 1
    var k = 0

    while k < n:
        var nb = min(block, n - k)

        # Panel: unblocked right-looking LU over columns k..k+nb-1, taking
        # the rank-1 update only as far as the panel's own right edge.
        for j in range(nb):
            var col = k + j

            var best = col
            var best_magnitude = abs(Float64(work[col * n + col]))
            for i in range(col + 1, n):
                var magnitude = abs(Float64(work[i * n + col]))
                if magnitude > best_magnitude:
                    best = i
                    best_magnitude = magnitude

            if best != col:
                for c in range(n):
                    var swap = work[col * n + c]
                    work[col * n + c] = work[best * n + c]
                    work[best * n + c] = swap
                var swap_index = permutation[col]
                permutation[col] = permutation[best]
                permutation[best] = swap_index
                sign = -sign

            var pivot = work[col * n + col]
            for i in range(col + 1, n):
                work[i * n + col] /= pivot
                for jj in range(j + 1, nb):
                    work[i * n + k + jj] -= (
                        work[i * n + col] * work[col * n + k + jj]
                    )

        # Block row: U12 = L11^-1 @ A12, with L11 unit-diagonal.
        for i in range(nb):
            var row = k + i
            for col in range(k + nb, n):
                var entry = work[row * n + col]
                for p in range(i):
                    entry -= work[row * n + k + p] * work[(k + p) * n + col]
                work[row * n + col] = entry

        # Trailing update: A22 -= L21 @ U12, which is MAX's.
        var m = n - k - nb
        if m > 0:
            var left_values = List[Scalar[dtype]](capacity=m * nb)
            var right_values = List[Scalar[dtype]](capacity=nb * m)
            for i in range(m):
                for j in range(nb):
                    left_values.append(work[(k + nb + i) * n + k + j])
            for i in range(nb):
                for j in range(m):
                    right_values.append(work[(k + i) * n + k + nb + j])

            var left = _staged[dtype](left_values^, m, nb, ctx)
            var right = _staged[dtype](right_values^, nb, m, ctx)
            var update = matmul[dtype, gpu](left, right).to_host()

            for i in range(m):
                for j in range(m):
                    work[(k + nb + i) * n + k + nb + j] -= update[i * m + j]

        k += nb

    return TensorLU[dtype, n](work^, permutation^, sign)


def det[T: FloatLike, n: Int](a: Array[T, n * n]) -> T:
    """The determinant, as the product of the unpivoted LU's diagonal.

    Unpivoted, so the sign is always the product's own -- there is no row
    swap count to correct for.

    MAX ships no `det` at any size. Past the crossover, `lu_factor` over
    `Tensor` and take `TensorLU.det`, which corrects for the swap parity
    this one has no swaps to correct for.
    """
    var factored = lu[T, n](a)
    var product = T.one()
    for i in range(n):
        product = product * factored[i * n + i]
    return product^
