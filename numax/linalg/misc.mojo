"""Scalar summaries of a matrix: `scipy.linalg._misc`'s `norm`, plus
`numpy.linalg`'s `trace` and `cond`.

**`norm` and `trace` have two tiers each**, resolved by argument type.
Over `Array` they are tier-1 loops. Over `Tensor` they are compositions of
MAX's reductions -- `ReduceSum` under the `rowwise` scaffolder for the
folds, `elementwise` for the magnitudes and the diagonal gather -- so they
thread on CPU and tier on GPU, and only the scalar comes back. `cond`
stays `Array`-only because `svd` does.

MAX ships no norm of any kind and no `trace`, so what is delegated is the
reduction underneath, not the operation.

`norm` takes `ord` as a compile-time parameter at both tiers -- `fro`
(default), `1` or `inf`, matching `numpy.linalg.norm` -- because each is a
different reduction. Over `Array` that is forced (a tier-1 kernel cannot
branch on which one at run time); over `Tensor` it keeps the three
launch sequences from being chosen at run time.

`cond` is the singular-value ratio from `svd`, so it costs a full
factorization. It is the rank-deficiency check the factorizations
themselves deliberately do not make.
"""

from layout import Coord, TileTensor, coord_to_index_list
from layout.tile_layout import row_major
from layout.tile_tensor import PointerStorage
from max.algorithm.functional import elementwise
from std.collections import Array
from std.math import sqrt as _sqrt
from std.utils import IndexList

from ..core.array import Shaped
from ..core.numeric import FloatLike, guard_nonzero, max_of, min_of
from ..core.rowwise import max_axis, sum_axis

from .blas import _fused_sum, _target
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


comptime _Flat[dtype: DType] = TileTensor[
    dtype,
    type_of(row_major(Coord(0))),
    MutAnyOrigin,
    Storage=PointerStorage[element_width=1],
]
"""A runtime-shaped rank-1 view over an existing pointer.

What the whole-matrix reductions want: `_fused_sum` folds along one axis,
and a Frobenius norm or a trace is a fold over every entry, so the matrix
is read as the vector it already is in memory. Costs nothing -- same
pointer, different layout."""


def trace[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Shaped[dtype, n, n]) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """**Tier 2.** The sum of the diagonal entries of `A`.
    `numpy.trace`.

    Two MAX launches and no host loop: an `elementwise` gathers the
    diagonal into a vector, then MAX's `ReduceSum` monoid folds it under
    the `rowwise` scaffolder. The gather is the reason it is two rather
    than one -- a diagonal is a strided slice of the flat matrix, and
    masking it inside the fold would cost a lane-index comparison on every
    element instead of touching only the `n` that matter.

    MAX ships no `trace`, so the composition is numax's; the fold is
    MAX's. Reassociated, unlike the `Array` overload's ordered sum.
    """
    var ctx = a.context()
    var diagonal = Shaped[dtype, n](ctx)
    var av = a.view()
    var dv = diagonal.view()

    @always_inline
    def gather[w: Int, alignment: Int = 1](coord: Coord) {var av, var dv}:
        var i = coord_to_index_list(coord)[0]
        dv.store[1](Coord(i), av[Coord(i, i)])

    elementwise[simd_width=1, target=_target[gpu]()](gather, Coord(n), ctx)

    var out = Shaped[dtype, 1](ctx)

    @always_inline
    def identity[
        w: Int
    ](tile: SIMD[dtype, w], idx: IndexList[1]) {} -> SIMD[dtype, w]:
        return tile

    _fused_sum[dtype, n, gpu](dv, out.view(), identity, ctx)
    return out.to_host()[0]


def norm[
    dtype: DType, n: Int, ord: Int = fro, gpu: Bool = False
](mut a: Shaped[dtype, n, n]) raises -> Scalar[
    dtype
] where dtype.is_floating_point() and (ord == fro or ord == 1 or ord == inf):
    """**Tier 2.** A matrix norm of `A`, over `Tensor`.
    `numpy.linalg.norm(A, ord=...)`.

    `ord` is the `Array` overload's, with the same three values, and each
    is a different composition of MAX's reductions:

    | `ord` | Reduction |
    |---|---|
    | `fro` (default) | one `ReduceSum` of squares over the flat matrix |
    | `1` | `sum_axis` down the columns of `abs(A)`, then a maximum |
    | `inf` | `sum_axis` across the rows of `abs(A)`, then a maximum |

    So the Frobenius norm is a single fused launch -- the square goes in
    `rowwise`'s per-tile transform, exactly as in `nrm2` -- and the two
    induced norms are three: an `elementwise` for the magnitudes, a
    `sum_axis`, and a `max_axis` over the `n` sums. Only the final scalar
    crosses to the host.

    Unrescaled like the `Array` overload, so the Frobenius norm of a matrix
    whose entries approach the square root of `dtype`'s overflow threshold
    overflows. The reason differs: there a running maximum would break the
    fixed-iteration invariant, here it would cost a second pass. Scale `A`
    yourself, or take the `1`- or `inf`-norm, which cannot overflow this
    way.

    MAX ships no norm of any kind, so the arrangement is numax's.
    """
    var ctx = a.context()
    var av = a.view()

    comptime if ord == fro:
        var flat: _Flat[dtype] = TileTensor(
            av.ptr_at_offset(Coord(0, 0)), row_major(Coord(n * n))
        )
        var out = Shaped[dtype, 1](ctx)

        @always_inline
        def square[
            w: Int
        ](tile: SIMD[dtype, w], idx: IndexList[1]) {} -> SIMD[dtype, w]:
            return tile * tile

        _fused_sum[dtype, n * n, gpu](flat, out.view(), square, ctx)
        return _sqrt(out.to_host()[0])

    var magnitudes = Shaped[dtype, n, n](ctx)
    var mv = magnitudes.view()

    @always_inline
    def magnitude[w: Int, alignment: Int = 1](coord: Coord) {var av, var mv}:
        mv.store[w](coord, abs(av.load[w](coord)))

    elementwise[simd_width=1, target=_target[gpu]()](
        magnitude, Coord(n, n), ctx
    )

    # Down the columns for the 1-norm, across the rows for the
    # infinity-norm: `sum_axis` collapses the axis it is given. The two
    # calls are written out rather than sharing an `axis` computed from
    # `ord`, because a derived comptime axis has nothing to prove its own
    # bound against.
    var sums = Shaped[dtype, n](ctx)
    comptime if ord == 1:
        sum_axis[axis=0, target=_target[gpu]()](mv, sums.view(), ctx)
    else:
        sum_axis[axis=1, target=_target[gpu]()](mv, sums.view(), ctx)

    var out = Shaped[dtype, 1](ctx)
    max_axis[axis=0, target=_target[gpu]()](sums.view(), out.view(), ctx)
    return out.to_host()[0]


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
