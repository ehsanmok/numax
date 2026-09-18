"""Products and BLAS-1: `numpy.linalg`'s `matmul`/`dot`/`outer` and the
level-1 routines `scipy.linalg.blas` wraps.

**The `Tensor` tier.** `numax.linalg.array.blas` is the same names over
`Array[T, n*n]`, `FloatLike`-generic and tier 1, which is what lets one
call factor a matrix per SIMD lane inside a kernel; past roughly 8x8 this
tier is the faster one and `docs/performance.md` has the crossover.

For the products, this module is a delegation layer and nothing else.
`matmul`, `matvec` and `batched_matmul` call `linalg.matmul` and
`linalg.bmm`, so they inherit MAX's whole dispatch tree -- Apple simdgroup,
SM100, SM90, Ampere/CDNA, GEMV, vendor cuBLAS/rocBLAS/hipBLASLt, AMD RDNA
-- and numax names no architecture. `gpu: Bool` picks MAX's `target`; see
this subpackage's `__init__` docstring for why it is a parameter and not a
run-time test.

MAX ships no BLAS-1 by name, but it ships everything BLAS-1 is made of, so
`dot`/`nrm2`/`asum`/`axpy`/`outer` are built here from MAX primitives
rather than delegated. The reductions (`dot`, `nrm2`, `asum`) call
`numax.core.rowwise.reduce_all`, which drives MAX's `ReduceSum` monoid over
its `rowwise` scaffolder with the multiply, the square or the magnitude
riding the per-tile transform; the maps (`axpy`, `outer`) go through
`max.algorithm.elementwise`. Both give SIMD width, CPU threading and GPU
dispatch without numax naming any of them. The fused sum used to be a
private helper here; it moved to `numax.core.rowwise` when the statistics
reductions wanted the same shape, which is the allowed direction --
`linalg` depends on `core`.

The two tiers together are the answer to a fair criticism of the `Array`
versions: at
`Plain[dtype, w]` each element holds a `w`-wide `SIMD`, so `axpy`'s loop is
a `w`-wide FMA over the batch axis -- but at `w = 1` it is a scalar loop
over `n`, and nothing in it could ever reach an accelerator. The fix was
not to hand-vectorize that loop. It was to add the tier that hands the work
to MAX, and leave the `Array` tier to the thing only it can do: run
`FloatLike`-generically, per SIMD lane, inside a kernel body.
"""

from algorithm.rowwise_types import RowCoord
from layout import Coord, TileTensor, coord_to_index_list
from layout.tile_layout import TensorLayout, row_major
from layout.tile_tensor import DefaultEngine
from linalg.bmm import batched_matmul as _max_batched_matmul
from linalg.matmul import matmul as _max_matmul
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.math import sqrt as _sqrt
from std.sys.info import simd_width_of
from std.utils import IndexList

from ..core.rowwise import reduce_all
from .common import _mut_view, _mut_view_as
from ..core.tensorlike import TensorLike, View, dim, is_row_major
from ..core.array import (
    _canonical_dyn,
    copy,
    Dynamic,
    Static,
    Tensor,
    zeros_dyn,
    _dyn_shape,
    _dyn_shape_from,
)


@always_inline
def _target[gpu: Bool]() -> StaticString:
    """MAX's `target` string for numax's `gpu: Bool` parameter."""
    return "gpu" if gpu else "cpu"


def dot[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Scalar[A.dtype] where (
    A.dtype.is_floating_point()
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 0]
):
    """The inner product `sum(a[i] * b[i])` -- BLAS-1 `dot`, over `Tensor`.

    MAX's `ReduceSum` monoid over its `rowwise` scaffolder, with the
    multiply fused into the per-tile transform, so `b` is never
    materialized as a product vector. Threaded on CPU, warp- or
    block-tiered on GPU. Only the resulting scalar returns to the host.

    Reassociated, unlike the `Array` overload's strict left-to-right sum.
    A caller who needs the rounding pinned, or who needs `Compensated`,
    wants that one.
    """
    comptime n = dim[A, 0]
    var ctx = a.context()
    var out = Static[A.dtype, 1](ctx)
    var rhs = _mut_view_as[A.dtype](b)

    @always_inline
    def times[
        w: Int
    ](tile: SIMD[A.dtype, w], idx: RowCoord[1]) {var rhs} -> SIMD[A.dtype, w]:
        return tile * rhs.load[w](idx.coord)

    reduce_all[monoid="sum", target=_target[gpu]()](
        _mut_view(a), out.view(), times, n, Optional(ctx)
    )
    return out.to_host()[0]


def nrm2[
    T: TensorLike,
    gpu: Bool = False,
](a: T) raises -> Scalar[T.dtype] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
):
    """The Euclidean norm `sqrt(sum(a[i]**2))` -- BLAS-1 `nrm2`, over
    `Tensor`.

    Not rescaled, so a vector whose entries approach the square root of
    `T.dtype`'s overflow threshold overflows here where LAPACK's `nrm2`
    would not -- the same limit the `Array` overload documents, and for a
    different reason: there the fixed-iteration invariant rules out the
    running maximum, here it would cost a second pass over the data.
    """
    comptime n = dim[T, 0]
    var ctx = a.context()
    var out = Static[T.dtype, 1](ctx)

    @always_inline
    def square[
        w: Int
    ](tile: SIMD[T.dtype, w], idx: RowCoord[1]) {} -> SIMD[T.dtype, w]:
        return tile * tile

    reduce_all[monoid="sum", target=_target[gpu]()](
        _mut_view(a), out.view(), square, n, Optional(ctx)
    )
    return _sqrt(out.to_host()[0])


def asum[
    T: TensorLike,
    gpu: Bool = False,
](a: T) raises -> Scalar[T.dtype] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
):
    """The sum of magnitudes `sum(|a[i]|)` -- BLAS-1 `asum`, over `Tensor`.

    Cannot overflow the way `nrm2` can, which is why a convergence check
    that only needs a magnitude usually wants this one.
    """
    comptime n = dim[T, 0]
    var ctx = a.context()
    var out = Static[T.dtype, 1](ctx)

    @always_inline
    def magnitude[
        w: Int
    ](tile: SIMD[T.dtype, w], idx: RowCoord[1]) {} -> SIMD[T.dtype, w]:
        return abs(tile)

    reduce_all[monoid="sum", target=_target[gpu]()](
        _mut_view(a), out.view(), magnitude, n, Optional(ctx)
    )
    return out.to_host()[0]


def axpy[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](alpha: Float64, x: A, y: B) raises -> Static[A.dtype, dim[A, 0]] where (
    A.dtype.is_floating_point()
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 0]
):
    """`alpha * x + y` -- BLAS-1 `axpy`, over `Tensor`.

    One fused `max.algorithm.elementwise` pass: `alpha` rides the body's
    capture list, so no scaled copy of `x` is materialized. That is the
    reason this cannot be `numax.core.tensor.map` -- `map`'s `step` is a
    non-capturing compile-time function, which a run-time scalar cannot
    reach.

    Returns a new vector rather than updating `y` in place, matching the
    `Array` overload, so the two spellings can be checked against each
    other.
    """
    var scale = Scalar[A.dtype](alpha)
    comptime n = dim[A, 0]
    var ctx = x.context()
    # Not zeroed: the `elementwise` pass below writes every element of
    # `out` before anything reads one, so the ordinary constructor's
    # memset would be a full pass over a buffer about to be overwritten,
    # and its synchronize a device round trip for nothing.
    var out = Static[A.dtype, n]._uninitialized(ctx)
    var xv = _mut_view(x)
    var yv = _mut_view_as[A.dtype](y)
    var ov = out.view()

    @always_inline
    def step[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var scale, var xv, var yv, var ov}:
        ov.store[w](coord, scale * xv.load[w](coord) + yv.load[w](coord))

    elementwise[simd_width=simd_width_of[A.dtype](), target=_target[gpu]()](
        step, Coord(n), ctx
    )
    return out^


def outer[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Static[A.dtype, dim[A, 0], dim[B, 0]] where (
    A.dtype.is_floating_point()
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
):
    """The outer product `out[i, j] = a[i] * b[j]` -- over `Tensor`.

    The rank-1 update every quasi-Newton method and every Householder
    reflector is built from. MAX's `outer_product_acc` is the nearest thing
    and is denied twice over: it is on the older `LayoutTensor`, which
    numax does not bridge to, and it only accumulates into an existing
    matrix rather than producing one. So this is an `elementwise` map over
    the `m x n` output, which is what `outer_product_acc` would have been
    used for anyway.

    Unlike the `Array` overload this is not restricted to a square result:
    `a` and `b` may have different lengths, as `numpy.outer` allows.
    """
    comptime m = dim[A, 0]
    comptime n = dim[B, 0]
    var ctx = a.context()
    # Not zeroed, for the reason `axpy` above gives: the `elementwise`
    # pass writes every element of `out` before anything reads one.
    var out = Static[A.dtype, m, n]._uninitialized(ctx)
    var av = _mut_view(a)
    var bv = _mut_view_as[A.dtype](b)
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var av, var bv, var ov}:
        # One row of the output at a time: `a[i]` is uniform across the
        # tile and `b[j:j+w]` is contiguous, so the store is too.
        var i = coord[0]
        var j = coord[1]
        ov.store[w](coord, av[i] * bv.load[w](Coord(j)))

    elementwise[simd_width=simd_width_of[A.dtype](), target=_target[gpu]()](
        step, Coord(m, n), ctx
    )
    return out^


def matvec[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, x: B) raises -> Static[A.dtype, dim[A, 0]] where (
    is_row_major[B]
    and A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 1]
):
    """The matrix-vector product `a @ x`, on `a`'s own device.

    Routed through `linalg.matmul` rather than `linalg.gemv`, with the
    vectors relaid as `k x 1` and `m x 1` over their own pointers. MAX's
    dispatch already sends a matmul with `n == 1` to its GEMV kernels --
    including the split-K and vector variants it picks between by shape --
    so calling `gemv` directly would be a second numax path to the same
    kernels, and one that would have to reproduce the choice.

    The relayout is free: a rank-1 buffer of `k` elements and a `k x 1`
    row-major layout address the same memory in the same order, so this
    hands MAX a different description of bytes it was going to read
    anyway, not a copy.

    **Both extents are grown to a lane multiple first**, when they are not
    ones already. MAX's GEMV over-reads its input in two independent ways
    at `max ==26.5`, and each needs its own padding:

    - **`m`**, the row count. The kernel walks the rows in blocks of
      `simd_width_of[A.dtype]()`, and its final block runs past the last row
      whenever `m` is not a whole number of lanes -- as much as
      `(lanes - 1) * k` elements beyond the matrix.
    - **`k`**, the row length. The reduction over each row is unrolled by
      the same width, so the last load of row `i` reaches into row
      `i + 1` whenever `k` is not a whole number of lanes. That is
      harmless in the interior and reads off the end of the buffer on the
      final row.

    Both are discarded values, so either defect stays invisible until the
    allocation ends on a page boundary and the read faults. `m` alone was
    padded until the `k` case turned up: `matvec[float64, 6, 4]` faulted on
    the Linux CI runner where the same call passed on Apple Silicon, and
    then a padded-`m` call faulted there too -- `simd_width_of[float64]()`
    is 8 under AVX-512 against 2 on NEON, so almost every `k` is misaligned
    on one and almost none on the other.

    Growing the matrix to `m_pad x k_pad` with the phantom row and column
    entries zeroed turns the read the kernel wants into a read numax has
    allocated. The padding cannot change the answer: a phantom column
    contributes `0 * 0` to every dot product, and a phantom row produces an
    output that is trimmed away.

    The growth costs one `m_pad x k_pad` allocation and two `elementwise`
    passes, bounded by `lanes - 1` in each extent, and a call already
    aligned in both skips all of it at compile time -- so no existing
    measurement moves. This is a workaround for the pinned `max ==26.5`,
    not a design choice, and it should go when MAX's GEMV masks its own
    tail.
    """
    comptime m = dim[A, 0]
    comptime k = dim[A, 1]
    comptime lanes = simd_width_of[A.dtype]()
    var ctx = a.context()
    var xv = _mut_view_as[A.dtype](x)
    var x_col = TileTensor(xv.ptr_at_offset(Coord(0)), row_major(Coord(k, 1)))

    comptime if m % lanes == 0 and k % lanes == 0:
        var result = Static[A.dtype, m](ctx)
        var yv = result.view()
        var y_col = TileTensor(
            yv.ptr_at_offset(Coord(0)), row_major(Coord(m, 1))
        )
        _max_matmul[target="gpu" if gpu else "cpu"](
            y_col, _mut_view(a), x_col, ctx
        )
        ctx.synchronize()
        return result^
    else:
        comptime m_pad = ((m + lanes - 1) // lanes) * lanes
        comptime k_pad = ((k + lanes - 1) // lanes) * lanes

        # Not zeroed: `grow` writes every element, phantom entries included.
        var padded = Static[A.dtype, m_pad, k_pad]._uninitialized(ctx)
        var av = _mut_view(a)
        var pv = padded.view()

        # Width 1: this reads its input at an index derived from the
        # *output* coordinate, and a wider tile is not guaranteed to stay
        # inside one row of `a` -- the very over-read this exists to avoid.
        @always_inline
        def grow[w: Int, alignment: Int = 1](coord: Coord) {var av, var pv}:
            var at = coord_to_index_list(coord)
            if at[0] < m and at[1] < k:
                pv.store[1](coord, av[Coord(at[0], at[1])])
            else:
                pv.store[1](coord, Scalar[A.dtype](0))

        elementwise[simd_width=1, target=_target[gpu]()](
            grow, Coord(m_pad, k_pad), ctx
        )

        # `x` grows with it, so the phantom columns pair `0` against `0`.
        var x_wide = Static[A.dtype, k_pad]._uninitialized(ctx)
        var xw = x_wide.view()

        @always_inline
        def grow_x[w: Int, alignment: Int = 1](coord: Coord) {var xv, var xw}:
            var j = coord_to_index_list(coord)[0]
            if j < k:
                xw.store[1](coord, xv[coord])
            else:
                xw.store[1](coord, Scalar[A.dtype](0))

        elementwise[simd_width=1, target=_target[gpu]()](
            grow_x, Coord(k_pad), ctx
        )
        var x_pad_col = TileTensor(
            xw.ptr_at_offset(Coord(0)), row_major(Coord(k_pad, 1))
        )

        var wide = Static[A.dtype, m_pad]._uninitialized(ctx)
        var wv = wide.view()
        var y_col = TileTensor(
            wv.ptr_at_offset(Coord(0)), row_major(Coord(m_pad, 1))
        )
        _max_matmul[target="gpu" if gpu else "cpu"](
            y_col, padded.view(), x_pad_col, ctx
        )
        ctx.synchronize()

        # Not zeroed: `trim` writes every element of `result`.
        var result = Static[A.dtype, m]._uninitialized(ctx)
        var rv = result.view()
        var read = wide.view()

        # Width 1 for the same reason: the source is longer than the
        # destination, so a tile sized to `m` is not a tile of `wide`.
        @always_inline
        def trim[w: Int, alignment: Int = 1](coord: Coord) {var read, var rv}:
            rv.store[1](coord, read[coord])

        elementwise[simd_width=1, target=_target[gpu]()](trim, Coord(m), ctx)
        # `wide`'s last *use* above is `.view()`, and the view is
        # origin-erased, so without this Mojo destroys `wide` before `trim`
        # reads through `read`: the queued free ran at the next
        # `synchronize` and `trim` copied a heap pointer into `result[0]`.
        _ = wide^
        _ = x_wide^
        return result^


def matmul[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Static[A.dtype, dim[A, 0], dim[B, 1]] where (
    A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 2
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 1]
):
    """The matrix product `a @ b`, on `a`'s own device.

    `linalg.matmul` does the work, and that one call is a whole dispatch
    tree: Apple simdgroup kernels, SM100 `tcgen05`, SM90, Ampere and CDNA
    multistage GEMM with tile shapes chosen by heuristic, GEMV when a
    dimension is 1, vendor cuBLAS/rocBLAS/hipBLASLt, AMD RDNA WMMA, and a
    naive kernel when nothing else fits. numax names no architecture and
    picks no kernel.

    The sibling `matmul` over `Array[T, n*n]` is the one to call inside a
    kernel, or at any conformer other than a raw `A.dtype`; `to_tensor`
    crosses from there to here and `to_array` back.
    """
    comptime m = dim[A, 0]
    comptime n = dim[B, 1]
    var ctx = a.context()
    var result = Static[A.dtype, m, n](ctx)
    var c = result.view()
    _max_matmul[target="gpu" if gpu else "cpu"](
        c, _mut_view(a), _mut_view_as[A.dtype](b), ctx
    )
    ctx.synchronize()
    return result^


def inner[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Static[A.dtype, dim[A, 0], dim[B, 0]] where (
    A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 2
    and B.LayoutType.all_dims_known
    and dim[B, 1] == dim[A, 1]
):
    """`a @ b.T`, the matrix of inner products between the rows of `a` and
    the rows of `b`. `numpy.inner` at rank 2.

    Note the shape: `b` is `n x k`, not `k x n`. Both operands are indexed
    by their *rows*, so `out[i, j]` is the inner product of `a`'s row `i`
    with `b`'s row `j` -- which is what makes this the natural spelling for
    a Gram matrix or a pairwise-similarity block, where transposing at the
    call site would be an extra pass over the data.

    And it is exactly that pass this saves: `linalg.matmul` takes
    `transpose_b` as a compile-time parameter and reads `b` transposed in
    place, so no transposed copy is ever materialized. That is the same
    facility `cholesky`'s trailing update uses. There is no `transpose_a`
    to match, which is why `numpy.inner`'s mirror image is not here.

    **Rank 1 is `dot`, deliberately not duplicated here.** `numpy.inner` on
    two vectors is their dot product, and numax already has that name for
    it at both tiers; a second spelling would be one more name meaning
    exactly what an existing one means.
    """
    comptime m = dim[A, 0]
    comptime k = dim[A, 1]
    comptime n = dim[B, 0]
    var ctx = a.context()
    var result = Static[A.dtype, m, n](ctx)
    var c = result.view()
    _max_matmul[transpose_b=True, target=_target[gpu]()](
        c, _mut_view(a), _mut_view_as[A.dtype](b), ctx
    )
    ctx.synchronize()
    return result^


def kron[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Static[
    A.dtype, dim[A, 0] * dim[B, 0], dim[A, 1] * dim[B, 1]
] where (
    A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 2
    and B.LayoutType.all_dims_known
):
    """The Kronecker product `numpy.kron(a, b)`: every entry of `a` scaling
    a whole copy of `b`, tiled into an `(m*p) x (n*q)` result.

    MAX ships nothing of the kind -- searched across `linalg`, `nn`,
    `algorithm` and `layout` at the pin -- so this is an `elementwise` map
    over the output, the same shape `outer` above uses. Each output
    coordinate splits into a block index and an offset within the block,
    which is the whole definition:

    `out[i*p + r, j*q + c] = a[i, j] * b[r, c]`

    The map is over the *output*, so `a` is read `p*q` times and `b` is
    read `m*n` times rather than either being materialized in tiles. That
    is the right trade here: the reads are cached and the alternative is an
    `(m*p) x (n*q)` staging buffer.

    Written width-1. The output's row stride is `n*q` while `b`'s is `q`,
    so consecutive output columns walk `b` contiguously only within a
    block and wrap at every block edge; a wider store would have to special
    -case that boundary for no gain, since the multiplier `a[i, j]` changes
    there too.
    """
    comptime m = dim[A, 0]
    comptime n = dim[A, 1]
    comptime p = dim[B, 0]
    comptime q = dim[B, 1]
    var ctx = a.context()
    var out = Static[A.dtype, m * p, n * q]._uninitialized(ctx)
    var av = _mut_view(a)
    var bv = _mut_view_as[A.dtype](b)
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var av, var bv, var ov}:
        # `Coord`'s elements carry no `//` or `%` and do not convert to
        # `Int` directly, so the split runs on the `IndexList` --
        # `panel.mojo` reaches for the same helper wherever it needs
        # arithmetic on a coordinate.
        var at = coord_to_index_list(coord)
        var row = at[0]
        var col = at[1]
        ov.store[w](
            coord,
            av[Coord(row // p, col // q)] * bv[Coord(row % p, col % q)],
        )

    elementwise[simd_width=1, target=_target[gpu]()](
        step, Coord(m * p, n * q), ctx
    )
    return out^


def matrix_power[
    T: TensorLike,
    power: Int,
    gpu: Bool = False,
](a: T) raises -> Static[T.dtype, dim[T, 0], dim[T, 0]] where (
    power >= 0
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
    and dim[T, 1] == dim[T, 0]
):
    """`a` raised to a non-negative integer `power`.
    `numpy.linalg.matrix_power`.

    Exponentiation by squaring, so `power` costs `O(log(power))` matrix
    products rather than `power - 1` of them, each one `linalg.matmul`.
    `power` is a compile-time parameter, so the squaring chain is decided at
    compile time and the loop below is over a known number of steps.

    `power == 0` is the identity, as NumPy defines it, and does not look at
    `a` at all.

    **Negative powers are a `where` clause rather than an overload.**
    NumPy's `matrix_power` accepts them and means `matrix_power(inv(a),
    -power)`. Spelling that at the call site costs one visible `inverse`
    and keeps this module from depending on `numax.linalg.basic`, which
    depends on `lu`, which depends on this one -- a cycle for a
    convenience. The compile error names the constraint, so a caller who
    wants it is told what to write.
    """
    comptime n = dim[T, 0]
    var ctx = a.context()

    comptime if power == 0:
        var identity = Static[T.dtype, n, n](ctx)
        var host = List[Scalar[T.dtype]](length=n * n, fill=0)
        for i in range(n):
            host[i * n + i] = 1
        identity.copy_from_host(host)
        return identity^

    # `result` accumulates the answer and `base` the repeated squares.
    # `remaining` is consumed a bit at a time, so both are run-time values
    # even though `power` is not -- the trip count is still known.
    var result = Static[T.dtype, n, n](ctx)
    var seeded = False
    var base = Static[T.dtype, n, n](ctx)
    base.copy_from_host(a.to_host())

    var remaining = power
    while remaining > 0:
        if remaining % 2 == 1:
            if seeded:
                result = matmul[gpu=gpu](result, base)
            else:
                result.copy_from_host(base.to_host())
                seeded = True
        remaining = remaining // 2
        if remaining > 0:
            # `matmul(base, base)` is rejected -- both operands are `mut`
            # and Mojo will not pass one binding twice -- so squaring needs
            # a second tensor, and `copy` round-trips through the host.
            # `ponytail:` `floor(log2(power))` host round trips per call; a
            # device-resident `copy` removes them.
            var mirror = copy(base)
            base = matmul[gpu=gpu](base, mirror)

    return result^


def matmul[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[A.dtype, 2] where (
    A.LayoutType.rank == 2
    and not A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 2
    and not B.LayoutType.all_dims_known
):
    """The matrix product `a @ b` at extents known only at run time.

    The run-time-shaped overload of the one above, selected by argument
    type rather than by a `where` clause: `Static` and `Dynamic` are
    different layouts, so the two can never be ambiguous. MAX reads the
    extents from the layout either way -- a compile-time shape buys kernel
    specialization, not correctness.

    Raises when `a`'s columns and `b`'s rows disagree, which is the check
    the static overload gets from the type system for free.
    """
    if a.dim[1]() != b.dim[0]():
        raise Error(
            "matmul shape mismatch: a is ",
            a.dim[0](),
            "x",
            a.dim[1](),
            " and b is ",
            b.dim[0](),
            "x",
            b.dim[1](),
        )
    var ctx = a.context()
    var result = zeros_dyn[A.dtype, 2](a.dim[0](), b.dim[1](), ctx=ctx)
    var c = result.view()
    _max_matmul[target="gpu" if gpu else "cpu"](
        c, _mut_view(a), _mut_view_as[A.dtype](b), ctx
    )
    ctx.synchronize()
    return result^


def batched_matmul[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Static[A.dtype, dim[A, 0], dim[A, 1], dim[B, 2]] where (
    A.LayoutType.rank == 3
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 3
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 0]
    and dim[B, 1] == dim[A, 2]
):
    """`batch` independent matrix products, one per leading index.

    `linalg.bmm.batched_matmul` does the work, in one launch rather than
    `batch` of them. There is no `Array[T, n*n]` counterpart: a batch of
    small matrices is what a `map` over a conformer already expresses, one
    matrix per SIMD lane, so the batched form only earns its own kernel at
    sizes past the crossover.
    """
    comptime batch = dim[A, 0]
    comptime m = dim[A, 1]
    comptime n = dim[B, 2]
    var ctx = a.context()
    var result = Static[A.dtype, batch, m, n](ctx)
    var c = result.view()
    _max_batched_matmul[target="gpu" if gpu else "cpu"](
        c, _mut_view(a), _mut_view_as[A.dtype](b), context=ctx
    )
    ctx.synchronize()
    return result^


# --------------------------------------------------- cross and tensordot


def cross[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Static[A.dtype, 3] where (
    A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and dim[A, 0] == 3
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and dim[B, 0] == 3
):
    """The cross product of two 3-vectors. `numpy.cross(a, b)`.

    One `elementwise` map over the three outputs, `out[i] = a[i+1] b[i+2]
    - a[i+2] b[i+1]` with the indices mod 3 -- MAX ships no cross product,
    and there is nothing to delegate three multiplies to. The `n x 3`
    overload takes rows of vectors.
    """
    var ctx = a.context()
    var out = Static[A.dtype, 3]._uninitialized(ctx)
    var av = _mut_view(a)
    var bv = _mut_view_as[A.dtype](b)
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var av, var bv, var ov}:
        var i = coord_to_index_list(coord)[0]
        var j = (i + 1) % 3
        var k = (i + 2) % 3
        ov.store[1](
            coord, av[Coord(j)] * bv[Coord(k)] - av[Coord(k)] * bv[Coord(j)]
        )

    elementwise[simd_width=1, target=_target[gpu]()](step, Coord(3), ctx)
    return out^


def cross[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Static[A.dtype, dim[A, 0], 3] where (
    A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and dim[A, 1] == 3
    and B.dtype == A.dtype
    and B.LayoutType.rank == 2
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 0]
    and dim[B, 1] == 3
):
    """Row-wise cross products of two `n x 3` tensors, `out[r] = cross(a[r],
    b[r])`. `numpy.cross` on stacks of vectors."""
    comptime n = dim[A, 0]
    var ctx = a.context()
    var out = Static[A.dtype, n, 3]._uninitialized(ctx)
    var av = _mut_view(a)
    var bv = _mut_view_as[A.dtype](b)
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var av, var bv, var ov}:
        var at = coord_to_index_list(coord)
        var r = at[0]
        var i = at[1]
        var j = (i + 1) % 3
        var k = (i + 2) % 3
        ov.store[1](
            coord,
            av[Coord(r, j)] * bv[Coord(r, k)]
            - av[Coord(r, k)] * bv[Coord(r, j)],
        )

    elementwise[simd_width=1, target=_target[gpu]()](step, Coord(n, 3), ctx)
    return out^


def tensordot[
    A: TensorLike,
    B: TensorLike,
    axes: Int = 2,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    A.dtype, A.LayoutType.rank + B.LayoutType.rank - 2 * axes
] where (
    A.dtype == B.dtype
    and axes >= 0
    and axes <= A.LayoutType.rank
    and axes <= B.LayoutType.rank
    and A.LayoutType.rank + B.LayoutType.rank - 2 * axes >= 1
):
    """Contract the last `axes` dimensions of `a` with the first `axes` of
    `b`. `numpy.tensordot(a, b, axes)` in its integer form: `axes=1` is the
    ordinary product of the trailing and leading dimensions, `axes=2` (the
    default, NumPy's) the double contraction, `axes=0` the outer product of
    two tensors.

    **Delegate underneath.** Every tensordot is one matrix product: `a`
    read as `(M, K)` with `M` the product of its leading extents and `K` of
    the contracted ones, `b` read as `(K, N)`, and the result `(M, N)` read
    back at the combined shape. Row-major storage makes all three readings
    free -- the buffer is retyped, never copied -- so the whole operation is
    one `linalg.matmul` at run-time extents. NumPy's tuple form, naming
    arbitrary axes on each side, is `transpose(a, *order)` first and this
    second; the contracted extents are checked at run time and raise on a
    mismatch, which a `Dynamic` result cannot express in its type.

    The result is a `Dynamic` because its rank, not its extents, is what
    the types carry here: the extents of `a` and `b` are behind two
    different layouts and cannot both be spelled in one compile-time pack.
    """
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    comptime ra = ALayout.rank
    comptime rb = BLayout.rank
    comptime rank = ra + rb - 2 * axes
    var m = 1
    for d in range(ra - axes):
        m *= a.dim_at(d)
    var k = 1
    for d in range(axes):
        var extent = a.dim_at(ra - axes + d)
        if b.dim_at(d) != extent:
            raise Error(
                "tensordot: contracted extent ",
                d,
                " is ",
                extent,
                " in a and ",
                b.dim_at(d),
                " in b",
            )
        k *= extent
    var n = 1
    for d in range(rb - axes):
        n *= b.dim_at(axes + d)

    var a2 = _canonical_dyn[2](a, m, k)
    var b2 = _canonical_dyn[2, dtype=A.dtype](b, k, n)

    # Not `matmul`: MAX routes `n == 1` to a GEMV that stores whole SIMD
    # vectors with no masked tail, so a row count off a lane multiple comes
    # back with the tail wrong. No GEMM accelerates a BLAS-2 shape anyway,
    # so this is one `elementwise` over flat rank-1 views.
    if n == 1:
        var ctx = a.context()
        var out = zeros_dyn[A.dtype, 1](m, ctx=ctx)
        var av = a2.view()
        var bv = b2.view()
        var flat_a = TileTensor(
            av.ptr_at_offset(Coord(0, 0)), row_major(Coord(m * k))
        )
        var flat_b = TileTensor(
            bv.ptr_at_offset(Coord(0, 0)), row_major(Coord(k))
        )
        var flat_out = out.view()

        @always_inline
        def contract[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var flat_a, var flat_b, var flat_out, var k}:
            var i = coord_to_index_list(coord)[0]
            var total = Scalar[A.dtype](0)
            for j in range(k):
                total += flat_a[Coord(i * k + j)] * flat_b[Coord(j)]
            flat_out.store[1](coord, total)

        elementwise[simd_width=1, target=_target[gpu]()](
            contract, Coord(m), ctx
        )
        ctx.synchronize()

        var extents_1 = List[Int](capacity=rank)
        for d in range(ra - axes):
            extents_1.append(a.dim_at(d))
        for d in range(rb - axes):
            extents_1.append(b.dim_at(axes + d))
        return Dynamic[A.dtype, rank](
            out.buffer.copy(),
            row_major(_dyn_shape_from[rank](extents_1)),
            out.on_host(),
        )

    var c = matmul(a2, b2)

    var extents = List[Int](capacity=rank)
    for d in range(ra - axes):
        extents.append(a.dim_at(d))
    for d in range(rb - axes):
        extents.append(b.dim_at(axes + d))
    return Dynamic[A.dtype, rank](
        c.buffer.copy(),
        row_major(_dyn_shape_from[rank](extents)),
        c.on_host(),
    )
