"""Scalar summaries of a matrix: `scipy.linalg._misc`'s `norm`, plus
`numpy.linalg`'s `trace` and `cond`.

**Tier 1, `Array`-only.** MAX ships no matrix norm of any kind and no
condition number, so nothing here delegates.

`norm` takes `ord` as a compile-time parameter -- `fro` (default), `1` or
`inf`, matching `numpy.linalg.norm` -- because each is a different
reduction and a tier-1 kernel cannot branch on which one at run time.

`cond` is the singular-value ratio from `svd`, so it costs a full
factorization. It is the rank-deficiency check the factorizations
themselves deliberately do not make.
"""

from std.collections import Array

from ..core.numeric import FloatLike, guard_nonzero, max_of, min_of

from .common import _PIVOT_FLOOR
from .eigen import svd


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
