"""Scalar summaries of a matrix: `scipy.linalg._misc`'s `norm`, plus
`numpy.linalg`'s `trace` and `cond`.

**The `Tensor` tier.** `norm` and `trace` here are compositions of MAX's
reductions -- `ReduceSum` under the `rowwise` scaffolder for the folds,
`elementwise` for the magnitudes and the diagonal gather -- so they thread
on CPU and tier on GPU, and only the scalar comes back.
`numax.linalg.array.misc` has the tier-1 loops of the same two names, and
`cond`, which stays `Array`-only because `svd` does.

MAX ships no norm of any kind and no `trace`, so what is delegated is the
reduction underneath, not the operation.

`norm` takes `ord` as a compile-time parameter at both tiers -- `fro`
(default), `1` or `inf`, matching `numpy.linalg.norm` -- because each is a
different reduction. Over `Array` that is forced (a tier-1 kernel cannot
branch on which one at run time); here it keeps the three launch sequences
from being chosen at run time. `fro` and `inf` are defined in this module
and shared by both tiers, so `norm[..., fro]` reads the same either way.
"""

from layout import Coord, TileTensor, coord_to_index_list
from layout.tile_layout import row_major
from layout.tile_tensor import PointerStorage
from max.algorithm.functional import elementwise
from std.math import sqrt as _sqrt
from std.utils import IndexList

from ..core.array import Static
from ..core.rowwise import max_axis, sum_axis

from .blas import _fused_sum, _target


comptime fro = 0
"""`ord` for `norm`: the Frobenius (entrywise 2-) norm. The default."""


comptime inf = -1
"""`ord` for `norm`: the induced infinity-norm."""


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
](mut a: Static[dtype, n, n]) raises -> Scalar[
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
    var diagonal = Static[dtype, n](ctx)
    var av = a.view()
    var dv = diagonal.view()

    @always_inline
    def gather[w: Int, alignment: Int = 1](coord: Coord) {var av, var dv}:
        var i = coord_to_index_list(coord)[0]
        dv.store[1](Coord(i), av[Coord(i, i)])

    elementwise[simd_width=1, target=_target[gpu]()](gather, Coord(n), ctx)

    var out = Static[dtype, 1](ctx)

    @always_inline
    def identity[
        w: Int
    ](tile: SIMD[dtype, w], idx: IndexList[1]) {} -> SIMD[dtype, w]:
        return tile

    _fused_sum[dtype, n, gpu](dv, out.view(), identity, ctx)

    # `view()` erases the origin, so `diagonal` is not kept alive by `dv`
    # and destruction is ASAP. See `numax.linalg.qr`.
    _ = diagonal^

    return out.to_host()[0]


def norm[
    dtype: DType, n: Int, ord: Int = fro, gpu: Bool = False
](mut a: Static[dtype, n, n]) raises -> Scalar[
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
        var out = Static[dtype, 1](ctx)

        @always_inline
        def square[
            w: Int
        ](tile: SIMD[dtype, w], idx: IndexList[1]) {} -> SIMD[dtype, w]:
            return tile * tile

        _fused_sum[dtype, n * n, gpu](flat, out.view(), square, ctx)
        return _sqrt(out.to_host()[0])

    var magnitudes = Static[dtype, n, n](ctx)
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
    var sums = Static[dtype, n](ctx)
    comptime if ord == 1:
        sum_axis[axis=0, target=_target[gpu]()](mv, sums.view(), ctx)
    else:
        sum_axis[axis=1, target=_target[gpu]()](mv, sums.view(), ctx)

    # `view()` erases the origin, so `magnitudes` is not kept alive by
    # `mv` and destruction is ASAP. See `numax.linalg.qr`.
    _ = magnitudes^

    var out = Static[dtype, 1](ctx)
    max_axis[axis=0, target=_target[gpu]()](sums.view(), out.view(), ctx)
    return out.to_host()[0]
