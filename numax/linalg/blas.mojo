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
rather than delegated. The reductions (`dot`, `nrm2`, `asum`)
drive MAX's `ReduceSum` monoid over its `rowwise` scaffolder; the maps
(`axpy`, `outer`) go through `max.algorithm.elementwise`. Both give SIMD
width, CPU threading and GPU dispatch without numax naming any of them.

The two tiers together are the answer to a fair criticism of the `Array`
versions: at
`Plain[dtype, w]` each element holds a `w`-wide `SIMD`, so `axpy`'s loop is
a `w`-wide FMA over the batch axis -- but at `w = 1` it is a scalar loop
over `n`, and nothing in it could ever reach an accelerator. The fix was
not to hand-vectorize that loop. It was to add the tier that hands the work
to MAX, and leave the `Array` tier to the thing only it can do: run
`FloatLike`-generically, per SIMD lane, inside a kernel body.
"""

from algorithm import rowwise
from algorithm.reduce_op import ReduceSum
from layout import Coord, TileTensor, coord_to_index_list
from layout.tile_layout import TensorLayout, row_major
from layout.tile_tensor import PointerStorage
from linalg.bmm import batched_matmul as _max_batched_matmul
from linalg.matmul import matmul as _max_matmul
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.math import sqrt as _sqrt
from std.sys.info import simd_width_of
from std.utils import IndexList

from ..core.array import (
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


def _fused_sum[
    dtype: DType,
    n: Int,
    gpu: Bool,
    Contribute: (def[w: Int](SIMD[dtype, w], IndexList[1]) -> SIMD[dtype, w])
    & RegisterPassable
    & ImplicitlyCopyable,
](
    xs: TileTensor[
        dtype,
        _,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
        linear_idx_type=_,
    ],
    dst: TileTensor[
        dtype,
        _,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
        linear_idx_type=_,
    ],
    contribute: Contribute,
    ctx: DeviceContext,
) raises:
    """`sum(contribute(xs[i], i))` over a rank-1 `xs`, into `dst[0]`.

    The shape all three BLAS-1 reductions share. MAX's `rowwise` scaffolder
    already takes a per-tile transform between the load and the fold, which
    is exactly where `dot`'s multiply, `nrm2`'s square and `asum`'s
    magnitude belong -- each reads one tile and none of them needs a second
    pass. So the fold is `ReduceSum` in all three cases and only
    `contribute` differs, which is why this is one function and not three.

    Reassociated, as any monoid reduction is: the `Array` overloads sum
    strictly in order and say so, and these will differ from them in the
    last bits.
    """
    comptime target = "gpu" if gpu else "cpu"
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[dtype, 1], target, 64, dtype
    ]()
    var src = xs
    var out = dst

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut c: rowwise.Context[params]) {
        var src, var out, var contribute
    }:
        @always_inline
        def load[
            width: Int, alignment: Int, coord_rank: Int
        ](idx: IndexList[coord_rank]) {var src} -> SIMD[dtype, width]:
            return src.load[width](Coord(idx))

        var row = rowwise.Row[params, dtype, dtype, 0, 1, is_cached=False](
            row_coords, n, c, load
        )
        var acc = row.reduce[ReduceSum[dtype, params.simd_width]](
            contribute, load
        ).acc

        @always_inline
        def write(oc: IndexList[1]) {var acc, var out}:
            out.store[params.emit_tile_width](
                Coord(0), acc.slice[params.emit_tile_width]()
            )

        row.emit(write)

    rowwise.launch[
        axis=0,
        simd_width=simd_width,
        target=target,
        num_phases=1,
        associative=True,
    ](body, Coord(IndexList[1](n)), Optional(ctx))


def dot[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n], mut b: Static[dtype, n]) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The inner product `sum(a[i] * b[i])` -- BLAS-1 `dot`, over `Tensor`.

    MAX's `ReduceSum` monoid over its `rowwise` scaffolder, with the
    multiply fused into the per-tile transform, so `b` is never
    materialized as a product vector. Threaded on CPU, warp- or
    block-tiered on GPU. Only the resulting scalar returns to the host.

    Reassociated, unlike the `Array` overload's strict left-to-right sum.
    A caller who needs the rounding pinned, or who needs `Compensated`,
    wants that one.
    """
    var ctx = a.context()
    var out = Static[dtype, 1](ctx)
    var rhs = b.view()

    @always_inline
    def times[
        w: Int
    ](tile: SIMD[dtype, w], idx: IndexList[1]) {var rhs} -> SIMD[dtype, w]:
        return tile * rhs.load[w](Coord(idx))

    _fused_sum[dtype, n, gpu](a.view(), out.view(), times, ctx)
    return out.to_host()[0]


def nrm2[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n]) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The Euclidean norm `sqrt(sum(a[i]**2))` -- BLAS-1 `nrm2`, over
    `Tensor`.

    Not rescaled, so a vector whose entries approach the square root of
    `dtype`'s overflow threshold overflows here where LAPACK's `nrm2`
    would not -- the same limit the `Array` overload documents, and for a
    different reason: there the fixed-iteration invariant rules out the
    running maximum, here it would cost a second pass over the data.
    """
    var ctx = a.context()
    var out = Static[dtype, 1](ctx)

    @always_inline
    def square[
        w: Int
    ](tile: SIMD[dtype, w], idx: IndexList[1]) {} -> SIMD[dtype, w]:
        return tile * tile

    _fused_sum[dtype, n, gpu](a.view(), out.view(), square, ctx)
    return _sqrt(out.to_host()[0])


def asum[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n]) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The sum of magnitudes `sum(|a[i]|)` -- BLAS-1 `asum`, over `Tensor`.

    Cannot overflow the way `nrm2` can, which is why a convergence check
    that only needs a magnitude usually wants this one.
    """
    var ctx = a.context()
    var out = Static[dtype, 1](ctx)

    @always_inline
    def magnitude[
        w: Int
    ](tile: SIMD[dtype, w], idx: IndexList[1]) {} -> SIMD[dtype, w]:
        return abs(tile)

    _fused_sum[dtype, n, gpu](a.view(), out.view(), magnitude, ctx)
    return out.to_host()[0]


def axpy[
    dtype: DType, n: Int, gpu: Bool = False
](
    alpha: Scalar[dtype], mut x: Static[dtype, n], mut y: Static[dtype, n]
) raises -> Static[dtype, n] where dtype.is_floating_point():
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
    var ctx = x.context()
    # Not zeroed: the `elementwise` pass below writes every element of
    # `out` before anything reads one, so the ordinary constructor's
    # memset would be a full pass over a buffer about to be overwritten,
    # and its synchronize a device round trip for nothing.
    var out = Static[dtype, n]._uninitialized(ctx)
    var xv = x.view()
    var yv = y.view()
    var ov = out.view()

    @always_inline
    def step[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var alpha, var xv, var yv, var ov}:
        ov.store[w](coord, alpha * xv.load[w](coord) + yv.load[w](coord))

    elementwise[simd_width=simd_width_of[dtype](), target=_target[gpu]()](
        step, Coord(n), ctx
    )
    return out^


def outer[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](mut a: Static[dtype, m], mut b: Static[dtype, n]) raises -> Static[
    dtype, m, n
] where dtype.is_floating_point():
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
    var ctx = a.context()
    # Not zeroed, for the reason `axpy` above gives: the `elementwise`
    # pass writes every element of `out` before anything reads one.
    var out = Static[dtype, m, n]._uninitialized(ctx)
    var av = a.view()
    var bv = b.view()
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var av, var bv, var ov}:
        # One row of the output at a time: `a[i]` is uniform across the
        # tile and `b[j:j+w]` is contiguous, so the store is too.
        var i = coord[0]
        var j = coord[1]
        ov.store[w](coord, av[i] * bv.load[w](Coord(j)))

    elementwise[simd_width=simd_width_of[dtype](), target=_target[gpu]()](
        step, Coord(m, n), ctx
    )
    return out^


def matvec[
    dtype: DType, m: Int, k: Int, gpu: Bool = False
](mut a: Static[dtype, m, k], mut x: Static[dtype, k]) raises -> Static[
    dtype, m
]:
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

    **The row count is grown to a lane multiple first, when it is not one
    already.** MAX's GEMV reduces each row on its own SIMD lane, so it
    walks the rows in blocks of `simd_width_of[dtype]()`, and its final
    block runs past the last row whenever `m` is not a whole number of
    lanes -- reading as much as `(lanes - 1) * k` elements beyond the
    matrix. Those values are discarded, so the defect stays invisible
    until the allocation ends on a page boundary and the read faults,
    which is what `matvec[dtype, 6, 4]` did on the Linux CI runner while
    the same call passed everywhere else. Growing the matrix to
    `m_pad x k` with the phantom rows zeroed turns the read the kernel
    wants into a read numax has allocated, and the extra outputs are zeros
    that get dropped.

    The growth costs one `m_pad x k` allocation and two `elementwise`
    passes, bounded by `lanes - 1` extra rows, and an aligned `m` skips
    all of it at compile time and runs what it ran before -- so no
    existing measurement moves. This is a workaround for the pinned
    `max ==26.5`, not a design choice, and it should go when MAX's GEMV
    masks its own tail.
    """
    comptime lanes = simd_width_of[dtype]()
    var ctx = a.context()
    var xv = x.view()
    var x_col = TileTensor(xv.ptr_at_offset(Coord(0)), row_major(Coord(k, 1)))

    comptime if m % lanes == 0:
        var result = Static[dtype, m](ctx)
        var yv = result.view()
        var y_col = TileTensor(
            yv.ptr_at_offset(Coord(0)), row_major(Coord(m, 1))
        )
        _max_matmul[target="gpu" if gpu else "cpu"](y_col, a.view(), x_col, ctx)
        ctx.synchronize()
        return result^
    else:
        comptime m_pad = ((m + lanes - 1) // lanes) * lanes

        # Not zeroed: `grow` writes every element, phantom rows included.
        var padded = Static[dtype, m_pad, k]._uninitialized(ctx)
        var av = a.view()
        var pv = padded.view()

        # Width 1, for the reason `numax.linalg.special_matrices` gives:
        # this reads its input at an index derived from the *output*
        # coordinate, and a wider tile is not guaranteed to stay inside one
        # row of `a` -- at small `k` it would run off the end of the source
        # while copying the last real row, which is the very fault this
        # function exists to avoid.
        @always_inline
        def grow[w: Int, alignment: Int = 1](coord: Coord) {var av, var pv}:
            var i = coord_to_index_list(coord)[0]
            if i < m:
                pv.store[1](coord, av[coord])
            else:
                pv.store[1](coord, Scalar[dtype](0))

        elementwise[simd_width=1, target=_target[gpu]()](
            grow, Coord(m_pad, k), ctx
        )

        var wide = Static[dtype, m_pad]._uninitialized(ctx)
        var wv = wide.view()
        var y_col = TileTensor(
            wv.ptr_at_offset(Coord(0)), row_major(Coord(m_pad, 1))
        )
        _max_matmul[target="gpu" if gpu else "cpu"](
            y_col, padded.view(), x_col, ctx
        )
        ctx.synchronize()

        # Not zeroed: `trim` writes every element of `result`.
        var result = Static[dtype, m]._uninitialized(ctx)
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
        return result^


def matmul[
    dtype: DType, m: Int, k: Int, n: Int, gpu: Bool = False
](mut a: Static[dtype, m, k], mut b: Static[dtype, k, n]) raises -> Static[
    dtype, m, n
]:
    """The matrix product `a @ b`, on `a`'s own device.

    `linalg.matmul` does the work, and that one call is a whole dispatch
    tree: Apple simdgroup kernels, SM100 `tcgen05`, SM90, Ampere and CDNA
    multistage GEMM with tile shapes chosen by heuristic, GEMV when a
    dimension is 1, vendor cuBLAS/rocBLAS/hipBLASLt, AMD RDNA WMMA, and a
    naive kernel when nothing else fits. numax names no architecture and
    picks no kernel.

    The sibling `matmul` over `Array[T, n*n]` is the one to call inside a
    kernel, or at any conformer other than a raw `dtype`; `to_tensor`
    crosses from there to here and `to_array` back.
    """
    var ctx = a.context()
    var result = Static[dtype, m, n](ctx)
    var c = result.view()
    _max_matmul[target="gpu" if gpu else "cpu"](c, a.view(), b.view(), ctx)
    ctx.synchronize()
    return result^


def inner[
    dtype: DType, m: Int, k: Int, n: Int, gpu: Bool = False
](mut a: Static[dtype, m, k], mut b: Static[dtype, n, k]) raises -> Static[
    dtype, m, n
]:
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
    var ctx = a.context()
    var result = Static[dtype, m, n](ctx)
    var c = result.view()
    _max_matmul[transpose_b=True, target=_target[gpu]()](
        c, a.view(), b.view(), ctx
    )
    ctx.synchronize()
    return result^


def kron[
    dtype: DType, m: Int, n: Int, p: Int, q: Int, gpu: Bool = False
](mut a: Static[dtype, m, n], mut b: Static[dtype, p, q]) raises -> Static[
    dtype, m * p, n * q
]:
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
    var ctx = a.context()
    var out = Static[dtype, m * p, n * q]._uninitialized(ctx)
    var av = a.view()
    var bv = b.view()
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
    dtype: DType, n: Int, power: Int, gpu: Bool = False
](mut a: Static[dtype, n, n]) raises -> Static[dtype, n, n] where power >= 0:
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
    var ctx = a.context()

    comptime if power == 0:
        var identity = Static[dtype, n, n](ctx)
        var host = List[Scalar[dtype]](length=n * n, fill=0)
        for i in range(n):
            host[i * n + i] = 1
        identity.copy_from_host(host)
        return identity^

    # `result` accumulates the answer and `base` the repeated squares.
    # `remaining` is consumed a bit at a time, so both are run-time values
    # even though `power` is not -- the trip count is still known.
    var result = Static[dtype, n, n](ctx)
    var seeded = False
    var base = Static[dtype, n, n](ctx)
    base.copy_from_host(a.to_host())

    var remaining = power
    while remaining > 0:
        if remaining % 2 == 1:
            if seeded:
                result = matmul[dtype, n, n, n, gpu](result, base)
            else:
                result.copy_from_host(base.to_host())
                seeded = True
        remaining = remaining // 2
        if remaining > 0:
            # `matmul(base, base)` is rejected: it takes both operands
            # mutably, and Mojo will not pass one binding through two `mut`
            # arguments. So the squaring step needs a second tensor holding
            # the same values, and `numax.core.array.copy` is the only way
            # to make one -- it round-trips through the host, since MAX
            # exposes no device-to-device copy numax has found.
            #
            # The ceiling: `floor(log2(power))` host round trips per call,
            # which is 1 at `power = 2 or 3` and 2 at `power = 4..7`. It is
            # bounded and small, but on a device tensor it is real. A
            # device-resident `copy` removes it without changing anything
            # here.
            var mirror = copy(base)
            base = matmul[dtype, n, n, n, gpu](base, mirror)

    return result^


def matmul[
    dtype: DType, gpu: Bool = False
](mut a: Dynamic[dtype, 2], mut b: Dynamic[dtype, 2]) raises -> Dynamic[
    dtype, 2
]:
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
    var result = zeros_dyn[dtype, 2](a.dim[0](), b.dim[1](), ctx=ctx)
    var c = result.view()
    _max_matmul[target="gpu" if gpu else "cpu"](c, a.view(), b.view(), ctx)
    ctx.synchronize()
    return result^


def batched_matmul[
    dtype: DType, batch: Int, m: Int, k: Int, n: Int, gpu: Bool = False
](
    mut a: Static[dtype, batch, m, k], mut b: Static[dtype, batch, k, n]
) raises -> Static[dtype, batch, m, n]:
    """`batch` independent matrix products, one per leading index.

    `linalg.bmm.batched_matmul` does the work, in one launch rather than
    `batch` of them. There is no `Array[T, n*n]` counterpart: a batch of
    small matrices is what a `map` over a conformer already expresses, one
    matrix per SIMD lane, so the batched form only earns its own kernel at
    sizes past the crossover.
    """
    var ctx = a.context()
    var result = Static[dtype, batch, m, n](ctx)
    var c = result.view()
    _max_batched_matmul[target="gpu" if gpu else "cpu"](
        c, a.view(), b.view(), context=ctx
    )
    ctx.synchronize()
    return result^


# --------------------------------------------------- cross and tensordot


def cross[
    dtype: DType, gpu: Bool = False
](mut a: Static[dtype, 3], mut b: Static[dtype, 3]) raises -> Static[dtype, 3]:
    """The cross product of two 3-vectors. `numpy.cross(a, b)`.

    One `elementwise` map over the three outputs, `out[i] = a[i+1] b[i+2]
    - a[i+2] b[i+1]` with the indices mod 3 -- MAX ships no cross product,
    and there is nothing to delegate three multiplies to. The `n x 3`
    overload takes rows of vectors.
    """
    var ctx = a.context()
    var out = Static[dtype, 3]._uninitialized(ctx)
    var av = a.view()
    var bv = b.view()
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
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n, 3], mut b: Static[dtype, n, 3]) raises -> Static[
    dtype, n, 3
]:
    """Row-wise cross products of two `n x 3` tensors, `out[r] = cross(a[r],
    b[r])`. `numpy.cross` on stacks of vectors."""
    var ctx = a.context()
    var out = Static[dtype, n, 3]._uninitialized(ctx)
    var av = a.view()
    var bv = b.view()
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
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    axes: Int = 2,
    gpu: Bool = False,
](
    mut a: Tensor[dtype, ALayout], mut b: Tensor[dtype, BLayout]
) raises -> Dynamic[dtype, ALayout.rank + BLayout.rank - 2 * axes] where (
    axes >= 0
    and axes <= ALayout.rank
    and axes <= BLayout.rank
    and ALayout.rank + BLayout.rank - 2 * axes >= 1
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

    var a2 = Dynamic[dtype, 2](
        a.buffer.copy(), row_major(_dyn_shape[2](m, k)), a.host_addressable
    )
    var b2 = Dynamic[dtype, 2](
        b.buffer.copy(), row_major(_dyn_shape[2](k, n)), b.host_addressable
    )

    # A contraction that leaves one column is a matrix-vector product, and
    # that is the one shape not to hand to `matmul`: MAX routes `n == 1`
    # to a GEMV kernel that stores whole SIMD vectors with no masked tail,
    # so a row count that is not a multiple of the lane width comes back
    # with the tail wrong -- uninitialized memory here, and on a machine
    # with wider lanes than the one this was written on, a crash. The
    # arithmetic saved is nothing: `m * k` multiply-adds is a BLAS-2 shape
    # that no GEMM accelerates, so numax writes it as one `elementwise`
    # over the output and reads both operands through flat rank-1 views,
    # which is the indexing that stays correct at every extent.
    if n == 1:
        var ctx = a.context()
        var out = zeros_dyn[dtype, 1](m, ctx=ctx)
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
            var total = Scalar[dtype](0)
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
        return Dynamic[dtype, rank](
            out.buffer.copy(),
            row_major(_dyn_shape_from[rank](extents_1)),
            out.host_addressable,
        )

    var c = matmul[dtype, gpu](a2, b2)

    var extents = List[Int](capacity=rank)
    for d in range(ra - axes):
        extents.append(a.dim_at(d))
    for d in range(rb - axes):
        extents.append(b.dim_at(axes + d))
    return Dynamic[dtype, rank](
        c.buffer.copy(),
        row_major(_dyn_shape_from[rank](extents)),
        c.host_addressable,
    )
