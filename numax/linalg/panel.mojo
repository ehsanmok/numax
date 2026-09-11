"""Kernel-author primitives for blocked factorization: the panel steps.

**Tier 2**, `TileTensor`-only, not re-exported. These are the pieces a
blocked right-looking factorization needs and MAX does not ship: the
unblocked factorization of one diagonal block, the two triangular solves
(BLAS `trsm`, one per side), and the pivoted factorization of one column
panel. `numax.linalg.cholesky` and `numax.linalg.lu` are their only
callers. `linalg.matmul` does the cubic work between panel steps.

Written in MAX's idiom rather than numax's so they stay upstreamable:
`TileTensor` in and out, `target: StaticString = "cpu"`,
`ctx: Optional[DeviceContext]`, and a LAPACK-style `info` tensor instead of
a raised error, because a kernel cannot raise. The caller reads `info` once
at the end of the factorization and raises there -- checking it per block
step would put a device synchronization inside the loop.

**Every one of these takes the whole matrix plus offsets, never a
sub-view.** That is not a style choice. MAX's `matmul` ignores the row
stride of its arguments -- a tile view is accepted, then read as if it were
contiguous, which silently mixes rows -- so a
blocked algorithm cannot pass tile views to it and has to pack instead --
and once packing is on the table, the panel kernels may as well address the
original matrix directly and save the copy. Offsets are `Int` arguments;
the block size is a compile-time bound so nothing is allocated per step.

**How the work is spread.** Two shapes appear, and which one a routine gets
follows from whether its columns are independent:

- The two `trsm`s, `gemv_sub` and the packing are embarrassingly parallel
  over rows or columns, so they go through `max.algorithm.elementwise` --
  one MAX launch, CPU threading and GPU dispatch included, no kernel
  written here.
- `potrf_diag` and `getrf_panel` are sequential over columns, with a
  cross-thread dependency at each one. They are single-thread-block kernels
  with `barrier()` between phases, so the whole panel is one launch rather
  than one launch per column.
- `laswp` and `trsv_diag` are sequential outright and small enough that
  they stay that way: `O(n)` and `O(block^2)` respectively, beside the
  `O(n^2)` of the `gemv_sub`s between them.

The single-block shape is the known ceiling, and it is worth stating
plainly because it is where a device-resident factorization stops being
free. `potrf_diag`'s block is `block x block` and costs `block^3 / 3`
flops, which is negligible at any `n`. `getrf_panel`'s panel is the full
remaining height, so it costs `O(n * block^2)` per step and `O(n^2 *
block / 2)` overall, all on one SM while the rest of the device idles.
Smaller `block` shrinks it; the upgrade is a recursive panel (LAPACK's
`getrf2`), which splits the panel until it fits one block and recovers the
parallelism through GEMM. Nothing above changes when that lands.
"""

from layout import Coord, TileTensor, coord_to_index_list
from layout.tile_layout import TensorLayout, row_major
from linalg.matmul import matmul as _max_matmul
from layout.tile_tensor import PointerStorage
from max.algorithm.functional import elementwise, parallelize
from max.gpu import barrier
from max.gpu.host import DeviceContext
from std.gpu import block_dim, thread_idx
from std.math import sqrt
from std.sys.info import align_of, simd_width_of
from std.utils import IndexList

from .common import _Dense


comptime _PANEL_THREADS = 256
"""Threads in the single block `potrf_diag` and `getrf_panel` launch with.

One block, so this is the whole parallelism those two kernels get. 256 is
eight warps on CUDA and eight SIMD groups on Metal, enough to cover the
memory latency of a `block x block` tile without needing more shared state
than a strided loop.
"""


comptime _PARALLEL_MIN_WORK = 1 << 15
"""Flops below which a row- or column-parallel panel routine stays serial.

`parallelize` pays a thread-dispatch cost per launch, and these routines
are called once per block step, so a factorization of a small matrix issues
many tiny launches. Measured: handing `parallelize` the 4x4 and 5x5 cases
in `tests/linalg/` took the tensor-linalg suite from 2.7 s to 17.0 s, all
of it dispatch. Above the threshold the same call is worth 3.2x on the
Cholesky panel solve at `n = 1024`.

The estimate each caller passes is its own work in flops, not its element
count -- which is the mistake `elementwise`'s own heuristic makes and the
reason these routines cannot use it.
"""


comptime _View[dtype: DType, Lay: TensorLayout] = TileTensor[
    dtype, Lay, MutAnyOrigin, Storage=PointerStorage[element_width=1]
]
"""What every routine here takes: an origin-erased, element-at-a-time view.

Origin-erased because a blocked factorization reads and writes one matrix
through several names at once, and Mojo's exclusivity checker rejects two
live views of one buffer. `numax.core.array`'s `Tensor.view()` already
returns this type, so a caller passes `a.view()` and nothing else.
"""


@always_inline
def _lane[gpu: Bool]() -> Int:
    """This thread's index within the block; `0` on the host."""
    comptime if gpu:
        return Int(thread_idx.x)
    else:
        return 0


@always_inline
def _lanes[gpu: Bool]() -> Int:
    """How many threads share the work; `1` on the host, so the same loop
    body walks the whole range there and strides by the block width on the
    accelerator."""
    comptime if gpu:
        return Int(block_dim.x)
    else:
        return 1


@always_inline
def _sync[gpu: Bool]():
    """A block-wide barrier on the accelerator, nothing on the host."""
    comptime if gpu:
        barrier()


@always_inline
def _at[
    trans: Bool, dtype: DType, Lay: TensorLayout
](a: _View[dtype, Lay], i: Int, j: Int) -> Scalar[dtype]:
    """`a[i, j]`, or `a[j, i]` at `trans=True`.

    The one place the transposed solves differ from the untransposed ones,
    so it lives here rather than being spelled out at each of the four
    read sites."""
    comptime if trans:
        return a[Coord(j, i)]
    else:
        return a[Coord(i, j)]


def potrf_diag[
    dtype: DType,
    ALayout: TensorLayout,
    ILayout: TensorLayout,
    gpu: Bool = False,
](
    a: _View[dtype, ALayout],
    info: _View[DType.int32, ILayout],
    k: Int32,
    nb: Int32,
) where dtype.is_floating_point():
    """Unblocked Cholesky of the `nb x nb` block at `(k, k)`, in place.

    Right-looking within the block: take the square root of the diagonal,
    scale the column below it, then rank-1 update the lower triangle to its
    right. Only the lower triangle is read and written; the upper triangle
    of the block is left exactly as it was.

    Writes the 1-based index of the first non-positive pivot to `info` and
    keeps going, so a caller that hands this a matrix which is not positive
    definite gets a finite answer plus a flag rather than a fault. The
    pivot is floored at the same time, which is what keeps the rest of the
    block finite. Already-set values of `info` are not overwritten, so it
    reports the *first* failure across every block step of a factorization.

    Reads and writes `a` directly instead of staging the block into shared
    memory. At `nb = 64` the block is 16 KB and stays in L1 for the whole
    kernel, and the routine's `nb^3 / 3` flops are negligible beside the
    factorization's `n^3 / 3`, so the staging would buy a fraction of
    something that does not matter. It also lets one body serve both
    targets.

    Launch on the accelerator with `grid_dim=1`,
    `block_dim=_PANEL_THREADS`; the host path runs it single-threaded.
    """
    var t = _lane[gpu]()
    var nt = _lanes[gpu]()
    var k0 = Int(k)
    var n_b = Int(nb)

    for j in range(n_b):
        # One thread takes the square root, because every other thread's
        # next phase divides by it.
        if t == 0:
            var d = a[Coord(k0 + j, k0 + j)]
            if d <= 0:
                if info[Coord(0)] == 0:
                    info.store[1](Coord(0), Int32(k0 + j + 1))
                d = Scalar[dtype](1e-30)
            a.store[1](Coord(k0 + j, k0 + j), sqrt(d))
        _sync[gpu]()

        var root = a[Coord(k0 + j, k0 + j)]
        var i = j + 1 + t
        while i < n_b:
            a.store[1](Coord(k0 + i, k0 + j), a[Coord(k0 + i, k0 + j)] / root)
            i += nt
        _sync[gpu]()

        # A22 -= L21 @ L21.T within the block, lower triangle only. One row
        # per thread, so the reads of column `j` are the only ones repeated.
        i = j + 1 + t
        while i < n_b:
            var lij = a[Coord(k0 + i, k0 + j)]
            for c in range(j + 1, i + 1):
                a.store[1](
                    Coord(k0 + i, k0 + c),
                    a[Coord(k0 + i, k0 + c)] - lij * a[Coord(k0 + c, k0 + j)],
                )
            i += nt
        _sync[gpu]()


def getrf_panel[
    dtype: DType,
    ALayout: TensorLayout,
    PLayout: TensorLayout,
    ILayout: TensorLayout,
    gpu: Bool = False,
](
    a: _View[dtype, ALayout],
    pivots: _View[DType.int32, PLayout],
    info: _View[DType.int32, ILayout],
    k: Int32,
    nb: Int32,
    n: Int32,
) where dtype.is_floating_point():
    """Unblocked LU with partial pivoting of the panel at columns `k..k+nb`.

    The panel is the full remaining height, rows `k..n`, so this both
    factors columns `k..k+nb` and leaves `L21` below them ready for the
    trailing update. Row interchanges are applied across **all** `n`
    columns as they are found, which is why no separate pivot-application
    pass follows: the trailing block is already permuted when the GEMM
    reaches it.

    `pivots[j]` records the row that column `j` interchanged with, in the
    same convention LAPACK's `getrf` uses, so a caller can rebuild the
    permutation without tracking it here. `info` gets the 1-based index of
    the first exactly-zero pivot, as in `potrf_diag`.

    `pivots` needs `n + _PANEL_THREADS` entries, not `n`: the tail is the
    scratch each thread parks its pivot candidate in, since a
    single-block kernel has nowhere else to put `_PANEL_THREADS` values
    that the first thread then reduces.

    Four phases per column, separated by barriers: find the pivot by a
    strided scan reduced through the first thread, swap the two rows,
    scale the column below the diagonal, then rank-1 update the rest of the
    panel. The scan is reduced serially by thread 0 over `_PANEL_THREADS`
    candidates rather than by a shared-memory tree, because the candidates
    are already in registers and a tree would cost more barriers than the
    serial pass costs cycles at this width.

    Launch on the accelerator with `grid_dim=1`,
    `block_dim=_PANEL_THREADS`; the host path runs it single-threaded. See
    the module docstring for why one block, and what replaces it.
    """
    var t = _lane[gpu]()
    var nt = _lanes[gpu]()
    var k0 = Int(k)
    var n_b = Int(nb)
    var rows = Int(n)
    var cols = Int(n)

    # Every thread's best candidate, then thread 0's verdict, both parked in
    # row `0` of `pivots` past the `n` entries it needs for the permutation.
    var scratch = cols

    for j in range(n_b):
        var col = k0 + j

        var best = col
        var best_magnitude = Scalar[dtype](0)
        var i = col + t
        while i < rows:
            var magnitude = abs(a[Coord(i, col)])
            if magnitude > best_magnitude:
                best_magnitude = magnitude
                best = i
            i += nt
        pivots.store[1](Coord(scratch + t), Int32(best))
        _sync[gpu]()

        if t == 0:
            var winner = Int(pivots[Coord(scratch)])
            var winning_magnitude = abs(a[Coord(winner, col)])
            for c in range(1, nt):
                var candidate = Int(pivots[Coord(scratch + c)])
                var magnitude = abs(a[Coord(candidate, col)])
                if magnitude > winning_magnitude:
                    winning_magnitude = magnitude
                    winner = candidate
            pivots.store[1](Coord(col), Int32(winner))
            if winning_magnitude == 0 and info[Coord(0)] == 0:
                info.store[1](Coord(0), Int32(col + 1))
        _sync[gpu]()

        var pivot_row = Int(pivots[Coord(col)])
        if pivot_row != col:
            var c = t
            while c < cols:
                var keep = a[Coord(col, c)]
                a.store[1](Coord(col, c), a[Coord(pivot_row, c)])
                a.store[1](Coord(pivot_row, c), keep)
                c += nt
        _sync[gpu]()

        var pivot = a[Coord(col, col)]
        if pivot == 0:
            pivot = Scalar[dtype](1e-30)

        # Scale and rank-1 update in one pass over the rows below, taking
        # the update only as far as the panel's own right edge: the columns
        # past it are the trailing block's, and the GEMM will do those.
        i = col + 1 + t
        while i < rows:
            var multiplier = a[Coord(i, col)] / pivot
            a.store[1](Coord(i, col), multiplier)
            for c in range(j + 1, n_b):
                a.store[1](
                    Coord(i, k0 + c),
                    a[Coord(i, k0 + c)] - multiplier * a[Coord(col, k0 + c)],
                )
            i += nt
        _sync[gpu]()


comptime _MIN_SPLIT = 4
"""The narrowest panel `getrf2` will split.

Below this the recursion's own GEMM is a couple of columns wide and its
three launches cost more than the rank-one updates they replace. It also
keeps the split clear of a one-column output, which sends `matmul` down the
GEMV path that `numax.linalg.qr`'s `_MIN_GEMM_COLS` records as a segfault
at `float64`.
"""


def getrf2[
    dtype: DType,
    ALayout: TensorLayout,
    PLayout: TensorLayout,
    ILayout: TensorLayout,
    gpu: Bool = False,
](
    a: _View[dtype, ALayout],
    pivots: _View[DType.int32, PLayout],
    info: _View[DType.int32, ILayout],
    left: _Dense[dtype],
    right: _Dense[dtype],
    product: _Dense[dtype],
    k: Int,
    nb: Int,
    n: Int,
    base: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """LAPACK's `getrf2`: the panel at columns `k..k+nb`, split until it
    fits `base`, with the two halves joined by a GEMM.

    **What this is for.** `getrf_panel` below is a single-block kernel whose
    panel spans the full remaining height, so it costs `O(n * block^2)` per
    step on one SM -- and on the host, where `_lanes` is 1, on one thread.
    Profiled at `n = 1024`, `block = 32`, `float32`, it was 57.9% of a whole
    LU. This splits the panel's columns in half, factors the left half, and
    turns the left half's effect on the right half into `linalg.matmul`,
    recursively, until a half is narrow enough that `getrf_panel` is the
    cheaper way to finish it. The recursion's cubic term therefore lands in
    MAX's GEMM, which is threaded, instead of in one scalar loop.

    **The launch count does not fall, and that is not the point.** A
    recursion with cutoff `base` has `n / base` leaves over the whole
    factorization against today's `n / block` panels, so it is launch-neutral
    at `base == block` and worse below it. What it buys is that the serial
    residue stops scaling with `block`: it becomes `O(n^2 * base / 2)` rather
    than `O(n^2 * block / 2)`, which is what lets `block` grow to a size the
    trailing GEMM prefers.

    **Pivoting is numax's convention, not LAPACK's, and it shortens this.**
    `getrf_panel` applies each interchange across all `n` columns as it finds
    it, so a row swapped inside the left half is already swapped everywhere
    -- including in the right half and in the already-written columns of `L`.
    LAPACK needs a pivot-replay pass on each side of the split; this needs
    none, and the body is just recurse-left, solve, GEMM, recurse-right.

    `left`, `right` and `product` are the caller's scratch, reused at every
    level. Their shapes shrink as the recursion descends, so each level
    rebuilds its own view over the same pointer; the buffers must be sized
    for the widest level, which is the caller's `n x block`, `block x n` and
    `n x n`. Stream ordering is what makes the reuse safe -- a deeper level's
    GEMM completes before the shallower one that follows it is enqueued.
    """
    if nb <= base or nb < _MIN_SPLIT:
        comptime if gpu:
            ctx.enqueue_function[
                getrf_panel[
                    dtype,
                    ALayout=ALayout,
                    PLayout=PLayout,
                    ILayout=ILayout,
                    gpu=True,
                ]
            ](
                a,
                pivots,
                info,
                Int32(k),
                Int32(nb),
                Int32(n),
                grid_dim=1,
                block_dim=_PANEL_THREADS,
            )
        else:
            getrf_panel(a, pivots, info, Int32(k), Int32(nb), Int32(n))
        return

    var half = nb // 2

    getrf2[dtype, ALayout, PLayout, ILayout, gpu](
        a, pivots, info, left, right, product, k, half, n, base, ctx
    )

    # `U12 := L11^-1 A12`, over the panel's own right half only. The columns
    # past `k + nb` belong to the caller's trailing block and are its GEMM's.
    trsm_left_lower_unit[target="gpu" if gpu else "cpu"](
        a, k, half, k + nb, ctx
    )

    var rows = n - k - half
    var cols = nb - half
    if rows > 0 and cols > 0:
        var r0 = k + half

        var l21: _Dense[dtype] = TileTensor(
            left.ptr_at_offset(Coord(0, 0)), row_major(Coord(rows, half))
        )
        var u12: _Dense[dtype] = TileTensor(
            right.ptr_at_offset(Coord(0, 0)), row_major(Coord(half, cols))
        )
        pack_block[target="gpu" if gpu else "cpu"](
            a, l21, r0, k, rows, half, ctx
        )
        pack_block[target="gpu" if gpu else "cpu"](
            a, u12, k, r0, half, cols, ctx
        )

        var out: _Dense[dtype] = TileTensor(
            product.ptr_at_offset(Coord(0, 0)), row_major(Coord(rows, cols))
        )

        @parameter
        @always_inline
        @__copy_capture(a, r0)
        def subtract[
            _dtype: DType,
            width: SIMDLength,
            *,
            alignment: Int = align_of[SIMD[_dtype, width]](),
        ](idx: IndexList[2], value: SIMD[_dtype, width]) capturing -> None:
            var at = Coord(r0 + idx[0], r0 + idx[1])
            a.store[width](
                at, a.load[width](at) - rebind[SIMD[dtype, width]](value)
            )

        _max_matmul[
            elementwise_lambda_fn=subtract, target="gpu" if gpu else "cpu"
        ](out, l21, u12, ctx)

    getrf2[dtype, ALayout, PLayout, ILayout, gpu](
        a, pivots, info, left, right, product, k + half, nb - half, n, base, ctx
    )


def sytd2_column[
    dtype: DType,
    ALayout: TensorLayout,
    VLayout: TensorLayout,
    TauLayout: TensorLayout,
    SLayout: TensorLayout,
    gpu: Bool = False,
](
    a: _View[dtype, ALayout],
    vpad: _View[dtype, VLayout],
    tau: _View[dtype, TauLayout],
    scratch: _View[dtype, SLayout],
    k: Int32,
    n: Int32,
) where dtype.is_floating_point():
    """Form column `k`'s Householder reflector for a symmetric
    tridiagonal reduction, and write it out padded. LAPACK's `sytd2`
    inner step, without the update.

    The reflector annihilates `a[k+2.., k]`, leaving `a[k+1, k]` as the
    subdiagonal entry `T` keeps. What it writes back is LAPACK's packed
    form -- `a[k+1, k]` holds that entry, `a[k+2.., k]` holds `v` past its
    implicit leading `1`, and `tau[k]` holds the scale.

    The difference from `geqr2_panel`, and the reason this is its own
    kernel: a QR reflector is applied to the columns to its right, while
    this one has to be applied on *both* sides of the trailing block.
    Neither the update nor the panel's remaining columns happen here --
    the caller does the update with two matrix products, which is what
    keeps the cubic term in MAX's GEMM.

    `vpad` is the reflector as a full-length vector -- zero at and above
    `k`, `1` at `k + 1`, `v` below -- so the caller can multiply the
    *whole* matrix by it and keep the leading block untouched for free,
    rather than staging the trailing block dense on every column.

    `scratch` is `_PANEL_THREADS + 1` entries for the norm reduction, the
    same shape `geqr2_panel` uses.

    Launch on the accelerator with `grid_dim=1`,
    `block_dim=_PANEL_THREADS`; the host path runs it single-threaded.
    """
    var t = _lane[gpu]()
    var nt = _lanes[gpu]()
    var k0 = Int(k)
    var rows = Int(n)
    var first = k0 + 1

    # ||x|| over the column strictly below the subdiagonal.
    var partial = Scalar[dtype](0)
    var i = first + 1 + t
    while i < rows:
        var value = a[Coord(i, k0)]
        partial += value * value
        i += nt
    scratch.store[1](Coord(t), partial)
    _sync[gpu]()

    if t == 0:
        var total = Scalar[dtype](0)
        for c in range(nt):
            total += scratch[Coord(c)]
        scratch.store[1](Coord(nt), total)
    _sync[gpu]()

    var below = scratch[Coord(nt)]
    var alpha = a[Coord(first, k0)]

    # Already tridiagonal in this column: the reflector is the identity,
    # `tau = 0`, and the scaling that would divide by zero never runs.
    if below == 0:
        if t == 0:
            tau.store[1](Coord(k0), Scalar[dtype](0))
        i = t
        while i < rows:
            var one_at = Scalar[dtype](1) if i == first else Scalar[dtype](0)
            vpad.store[1](Coord(i), one_at)
            i += nt
        _sync[gpu]()
        return

    var beta = sqrt(alpha * alpha + below)
    if alpha > 0:
        beta = -beta
    var this_tau = (beta - alpha) / beta
    var scale = Scalar[dtype](1) / (alpha - beta)

    i = first + 1 + t
    while i < rows:
        a.store[1](Coord(i, k0), a[Coord(i, k0)] * scale)
        i += nt
    if t == 0:
        a.store[1](Coord(first, k0), beta)
        tau.store[1](Coord(k0), this_tau)
    _sync[gpu]()

    # `vpad`: zero at and above `k`, the implicit `1` at `k + 1`, `v` below.
    i = t
    while i < rows:
        var value = Scalar[dtype](0)
        if i == first:
            value = Scalar[dtype](1)
        elif i > first:
            value = a[Coord(i, k0)]
        vpad.store[1](Coord(i), value)
        i += nt
    _sync[gpu]()


def sytd2_rank_two[
    dtype: DType,
    VLayout: TensorLayout,
    PLayout: TensorLayout,
    LLayout: TensorLayout,
    gpu: Bool = False,
](
    vpad: _View[dtype, VLayout],
    p: _View[dtype, PLayout],
    left: _View[dtype, LLayout],
    right: _View[dtype, LLayout],
    this_tau: Scalar[dtype],
    kappa: Scalar[dtype],
    k: Int32,
    n: Int32,
) where dtype.is_floating_point():
    """Build the two `n x 2` operands whose product is the symmetric
    rank-2 update `v w^T + w v^T`.

    `w = tau * p - (tau^2 * kappa / 2) * v`, where `p` is the unscaled
    `A v` and `kappa` is `p . v` -- Golub and Van Loan's `w` with the
    scaling deferred so the caller's matrix-vector product carries no
    factor.

    Packing them as `left = [v | w]` and `right = [w | v]` makes
    `left @ right^T` exactly `v w^T + w v^T`, so MAX's GEMM does the
    update in **one** call with `transpose_b=True` rather than two rank-one
    passes. That identity is what lets numax skip the `syrk`/`syr2k` MAX
    does not ship, and it is the same one a blocked `latrd` would use with
    `2 * block` columns instead of two.

    **`w` is forced to zero at and above `k`, and that is load-bearing.**
    `v` already vanishes there, but `p` does not -- it is `A v` over the
    whole matrix, so its leading entries are the rows the reflector does
    not touch. Leaving them in `w` would make the update write column `k`,
    which is where `sytd2_column` just packed the reflector, and the
    factorization would lose the very vectors it needs to form `Q`. With
    both vectors zeroed there the update is confined to the trailing block
    exactly as LAPACK's is, and the caller still multiplies the whole
    matrix and stages nothing.
    """
    var t = _lane[gpu]()
    var nt = _lanes[gpu]()
    var rows = Int(n)
    var shift = this_tau * this_tau * kappa / Scalar[dtype](2)

    var first = Int(k) + 1
    var i = t
    while i < rows:
        var v = vpad[Coord(i)]
        var w = Scalar[dtype](0)
        if i >= first:
            w = this_tau * p[Coord(i)] - shift * v
        left.store[1](Coord(i, 0), v)
        left.store[1](Coord(i, 1), w)
        right.store[1](Coord(i, 0), w)
        right.store[1](Coord(i, 1), v)
        i += nt
    _sync[gpu]()


def gebd2_col[
    dtype: DType,
    ALayout: TensorLayout,
    VLayout: TensorLayout,
    TauLayout: TensorLayout,
    SLayout: TensorLayout,
    gpu: Bool = False,
](
    a: _View[dtype, ALayout],
    left: _View[dtype, VLayout],
    tau: _View[dtype, TauLayout],
    scratch: _View[dtype, SLayout],
    k: Int32,
    m: Int32,
) where dtype.is_floating_point():
    """Form the left Householder reflector of a bidiagonal reduction's
    column `k`, and write it as column `k` of the dense `left`.

    The reflector annihilates `a[k+1.., k]` against `a[k, k]`. Unlike
    `sytd2_column` it does not pack the vector into `a`: it writes the
    full-length vector -- zero above `k`, `1` at `k`, `v` below -- into
    `left[:, k]`, then sets `a[k, k]` to the new diagonal entry and zeroes
    the column below it. The caller applies the reflector to the *other*
    columns with two matrix products, so column `k` is finished here.

    Launch on the accelerator with `grid_dim=1`,
    `block_dim=_PANEL_THREADS`; the host path runs it single-threaded.
    """
    var t = _lane[gpu]()
    var nt = _lanes[gpu]()
    var k0 = Int(k)
    var rows = Int(m)

    var partial = Scalar[dtype](0)
    var i = k0 + 1 + t
    while i < rows:
        var value = a[Coord(i, k0)]
        partial += value * value
        i += nt
    scratch.store[1](Coord(t), partial)
    _sync[gpu]()
    if t == 0:
        var total = Scalar[dtype](0)
        for c in range(nt):
            total += scratch[Coord(c)]
        scratch.store[1](Coord(nt), total)
    _sync[gpu]()

    var below = scratch[Coord(nt)]
    var alpha = a[Coord(k0, k0)]

    if below == 0:
        if t == 0:
            tau.store[1](Coord(k0), Scalar[dtype](0))
        i = t
        while i < rows:
            var one_at = Scalar[dtype](1) if i == k0 else Scalar[dtype](0)
            left.store[1](Coord(i, k0), one_at)
            i += nt
        _sync[gpu]()
        return

    var beta = sqrt(alpha * alpha + below)
    if alpha > 0:
        beta = -beta
    var this_tau = (beta - alpha) / beta
    var scale = Scalar[dtype](1) / (alpha - beta)

    i = t
    while i < rows:
        var value = Scalar[dtype](0)
        if i == k0:
            value = Scalar[dtype](1)
        elif i > k0:
            value = a[Coord(i, k0)] * scale
        left.store[1](Coord(i, k0), value)
        i += nt
    _sync[gpu]()

    i = k0 + 1 + t
    while i < rows:
        a.store[1](Coord(i, k0), Scalar[dtype](0))
        i += nt
    if t == 0:
        a.store[1](Coord(k0, k0), beta)
        tau.store[1](Coord(k0), this_tau)
    _sync[gpu]()


def gebd2_row[
    dtype: DType,
    ALayout: TensorLayout,
    ULayout: TensorLayout,
    TauLayout: TensorLayout,
    SLayout: TensorLayout,
    gpu: Bool = False,
](
    a: _View[dtype, ALayout],
    right: _View[dtype, ULayout],
    tau: _View[dtype, TauLayout],
    scratch: _View[dtype, SLayout],
    k: Int32,
    n: Int32,
) where dtype.is_floating_point():
    """Form the right Householder reflector of a bidiagonal reduction's
    row `k`, and write it as row `k` of the dense `right`.

    The mirror of `gebd2_col` along the row: it annihilates
    `a[k, k+2..]` against `a[k, k+1]`, writes the full-length vector --
    zero through `k`, `1` at `k + 1`, `u` beyond -- into `right[k, :]`,
    then sets `a[k, k+1]` to the new superdiagonal entry and zeroes the row
    beyond it. The caller applies it to the other rows with two products.
    """
    var t = _lane[gpu]()
    var nt = _lanes[gpu]()
    var k0 = Int(k)
    var cols = Int(n)
    var first = k0 + 1

    var partial = Scalar[dtype](0)
    var j = first + 1 + t
    while j < cols:
        var value = a[Coord(k0, j)]
        partial += value * value
        j += nt
    scratch.store[1](Coord(t), partial)
    _sync[gpu]()
    if t == 0:
        var total = Scalar[dtype](0)
        for c in range(nt):
            total += scratch[Coord(c)]
        scratch.store[1](Coord(nt), total)
    _sync[gpu]()

    var beyond = scratch[Coord(nt)]
    var alpha = a[Coord(k0, first)]

    if beyond == 0:
        if t == 0:
            tau.store[1](Coord(k0), Scalar[dtype](0))
        j = t
        while j < cols:
            var one_at = Scalar[dtype](1) if j == first else Scalar[dtype](0)
            right.store[1](Coord(k0, j), one_at)
            j += nt
        _sync[gpu]()
        return

    var beta = sqrt(alpha * alpha + beyond)
    if alpha > 0:
        beta = -beta
    var this_tau = (beta - alpha) / beta
    var scale = Scalar[dtype](1) / (alpha - beta)

    j = t
    while j < cols:
        var value = Scalar[dtype](0)
        if j == first:
            value = Scalar[dtype](1)
        elif j > first:
            value = a[Coord(k0, j)] * scale
        right.store[1](Coord(k0, j), value)
        j += nt
    _sync[gpu]()

    j = first + 1 + t
    while j < cols:
        a.store[1](Coord(k0, j), Scalar[dtype](0))
        j += nt
    if t == 0:
        a.store[1](Coord(k0, first), beta)
        tau.store[1](Coord(k0), this_tau)
    _sync[gpu]()


def laswp[
    dtype: DType,
    XLayout: TensorLayout,
    PLayout: TensorLayout,
    gpu: Bool = False,
](
    x: _View[dtype, XLayout],
    pivots: _View[DType.int32, PLayout],
    n: Int32,
) where dtype.is_floating_point():
    """Apply `getrf_panel`'s recorded row interchanges to a vector.

    LAPACK's `laswp`: walk `j` upward swapping `x[j]` with
    `x[pivots[j]]`, which reproduces on the right-hand side the same
    permutation the factorization applied to the matrix. Order matters --
    the interchanges compose -- so this is sequential and one thread does
    all of it. That is `n` swaps against a factorization's `n^3 / 3`
    flops, so there is nothing to parallelize that would matter.

    Launch on the accelerator with `grid_dim=1`, `block_dim=1`.
    """
    if _lane[gpu]() != 0:
        return
    for j in range(Int(n)):
        var other = Int(pivots[Coord(j)])
        if other != j:
            var keep = x[Coord(j)]
            x.store[1](Coord(j), x[Coord(other)])
            x.store[1](Coord(other), keep)


def trsv_diag[
    dtype: DType,
    ALayout: TensorLayout,
    XLayout: TensorLayout,
    upper: Bool,
    unit: Bool,
    trans: Bool = False,
    gpu: Bool = False,
](
    a: _View[dtype, ALayout],
    x: _View[dtype, XLayout],
    k: Int32,
    nb: Int32,
) where dtype.is_floating_point():
    """Solve the `nb x nb` triangular block at `(k, k)` against
    `x[k:k+nb]`, in place.

    The diagonal step of a blocked triangular solve: `upper` picks back
    substitution over forward, and `unit` says the stored diagonal is not
    the triangle's (which is what `getrf_panel` leaves for `L`).

    `trans=True` reads `a[p, i]` where the untransposed form reads
    `a[i, p]`, so `L^T` is solved against without materializing it. That
    is what a Cholesky solve's second half needs, and transposing an
    `n x n` matrix to get it would cost more than the solve.

    One thread, sequentially, because a substitution's row `i` needs row
    `i - 1` and the inner dot is only `nb` long. At the default `block` the
    whole routine is about 2000 scalar operations, run `n / block` times
    per triangle, against the `O(n^2)` of the updates between them -- so
    parallelizing it would be measuring noise. The updates are where the
    work is, and those are `gemv_sub`.

    Launch on the accelerator with `grid_dim=1`, `block_dim=1`.
    """
    if _lane[gpu]() != 0:
        return
    var k0 = Int(k)
    var n_b = Int(nb)

    for step in range(n_b):
        var i = (n_b - 1 - step) if upper else step
        var total = x[Coord(k0 + i)]
        if upper:
            for p in range(i + 1, n_b):
                total = total - _at[trans](a, k0 + i, k0 + p) * x[Coord(k0 + p)]
        else:
            for p in range(i):
                total = total - _at[trans](a, k0 + i, k0 + p) * x[Coord(k0 + p)]
        comptime if unit:
            x.store[1](Coord(k0 + i), total)
        else:
            x.store[1](Coord(k0 + i), total / a[Coord(k0 + i, k0 + i)])


def gemv_sub[
    dtype: DType,
    ALayout: TensorLayout,
    XLayout: TensorLayout,
    trans: Bool = False,
    target: StaticString = "cpu",
](
    a: _View[dtype, ALayout],
    x: _View[dtype, XLayout],
    row0: Int,
    col0: Int,
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """`x[row0:row0+rows] -= a[row0:, col0:] @ x[col0:col0+cols]`.

    The update between two diagonal steps of a blocked triangular solve,
    and the only `O(n^2)` part of one. Independent per row, so it is one
    `max.algorithm.elementwise`.

    Reads a strided block of `a` directly rather than packing it, which is
    the difference between this and the trailing update of a
    factorization: an elementwise body does its own addressing, so the
    stride is not a problem here the way it is for `matmul`. Writing the
    result into the same vector it reads is safe because the two ranges
    are disjoint by construction -- `x[col0:col0+cols]` is already solved
    and `x[row0:]` is not yet.

    `trans=True` reads the block transposed, matching `trsv_diag`.
    """
    if rows <= 0 or cols <= 0:
        return

    @always_inline
    @parameter
    def update_row(index: Int):
        var row = row0 + index
        var total = x[Coord(row)]
        for j in range(cols):
            total = total - _at[trans](a, row, col0 + j) * x[Coord(col0 + j)]
        x.store[1](Coord(row), total)

    # `parallelize` on the host, for the reason `trsm_right_lower_t` gives:
    # one element per row is below `elementwise`'s count threshold however
    # much work each row does, and this is the `O(n^2)` of every vector
    # triangular solve.
    comptime if target == "cpu":
        if rows * cols >= _PARALLEL_MIN_WORK:
            parallelize[update_row](rows)
        else:
            for index in range(rows):
                update_row(index)
    else:

        @always_inline
        def update[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var a, var x, var row0, var col0, var cols}:
            update_row(coord_to_index_list(coord)[0])

        elementwise[simd_width=1, target=target](update, Coord(rows), ctx)


def geqr2_panel[
    dtype: DType,
    ALayout: TensorLayout,
    TauLayout: TensorLayout,
    SLayout: TensorLayout,
    gpu: Bool = False,
](
    a: _View[dtype, ALayout],
    tau: _View[dtype, TauLayout],
    scratch: _View[dtype, SLayout],
    k: Int32,
    nb: Int32,
    m: Int32,
) where dtype.is_floating_point():
    """Unblocked Householder factorization of the panel at columns
    `k..k+nb`, rows `k..m`. LAPACK's `geqr2`.

    `nb` reflectors, each formed from one column and then applied to the
    columns still inside the panel. What it leaves behind is LAPACK's
    packed form: `a[k+j, k+j]` holds `R`'s diagonal entry, the entries
    below it hold the reflector's `v` with an implicit leading `1`, and
    `tau[k+j]` holds the scale. Nothing above the diagonal is touched.

    The columns to the *right* of the panel are left alone -- they are the
    trailing block's, and `qr`'s block reflector does those with three
    matrix products instead of `nb` rank-one updates. That split is the
    whole reason for blocking a QR.

    `scratch` is `_PANEL_THREADS` entries of workspace for the norm
    reduction, for the same reason `getrf_panel` parks its pivot
    candidates past the end of `pivots`: a single-block kernel has nowhere
    else to put one value per thread.

    Launch on the accelerator with `grid_dim=1`,
    `block_dim=_PANEL_THREADS`; the host path runs it single-threaded.
    """
    var t = _lane[gpu]()
    var nt = _lanes[gpu]()
    var k0 = Int(k)
    var n_b = Int(nb)
    var rows = Int(m)

    for j in range(n_b):
        var col = k0 + j

        # ||x|| over the sub-column strictly below the diagonal, reduced
        # through thread 0 -- the same shape as `getrf_panel`'s pivot scan.
        var partial = Scalar[dtype](0)
        var i = col + 1 + t
        while i < rows:
            var value = a[Coord(i, col)]
            partial += value * value
            i += nt
        scratch.store[1](Coord(t), partial)
        _sync[gpu]()

        if t == 0:
            var total = Scalar[dtype](0)
            for c in range(nt):
                total += scratch[Coord(c)]
            scratch.store[1](Coord(nt), total)
        _sync[gpu]()

        var below = scratch[Coord(nt)]
        var alpha = a[Coord(col, col)]

        # A column already in reflected form has nothing below the
        # diagonal, so its reflector is the identity: `tau = 0`, and the
        # scaling that would divide by zero never runs.
        if below == 0:
            if t == 0:
                tau.store[1](Coord(col), Scalar[dtype](0))
            _sync[gpu]()
            continue

        var beta = sqrt(alpha * alpha + below)
        if alpha > 0:
            beta = -beta
        var this_tau = (beta - alpha) / beta
        var scale = Scalar[dtype](1) / (alpha - beta)

        i = col + 1 + t
        while i < rows:
            a.store[1](Coord(i, col), a[Coord(i, col)] * scale)
            i += nt
        if t == 0:
            a.store[1](Coord(col, col), beta)
            tau.store[1](Coord(col), this_tau)
        _sync[gpu]()

        # `H = I - tau v v^T` on the panel's remaining columns, one column
        # per thread. `v`'s leading entry is the implicit `1`, so the dot
        # picks up `c`'s own diagonal row separately.
        var c = j + 1 + t
        while c < n_b:
            var other = k0 + c
            var dot = a[Coord(col, other)]
            for r in range(col + 1, rows):
                dot += a[Coord(r, col)] * a[Coord(r, other)]
            var factor = this_tau * dot
            a.store[1](Coord(col, other), a[Coord(col, other)] - factor)
            for r in range(col + 1, rows):
                a.store[1](
                    Coord(r, other),
                    a[Coord(r, other)] - factor * a[Coord(r, col)],
                )
            c += nt
        _sync[gpu]()


def larft_panel[
    dtype: DType,
    ALayout: TensorLayout,
    TauLayout: TensorLayout,
    TLayout: TensorLayout,
    gpu: Bool = False,
](
    a: _View[dtype, ALayout],
    tau: _View[dtype, TauLayout],
    t_block: _View[dtype, TLayout],
    k: Int32,
    nb: Int32,
    m: Int32,
) where dtype.is_floating_point():
    """Build the `nb x nb` triangular factor `T` of the panel's block
    reflector, so that `I - V T V^T` is the product of its `nb`
    Householder reflections. LAPACK's `larft`, forward and columnwise.

    This is what turns `nb` rank-one updates into three matrix products:
    with `T` in hand the trailing block's update is
    `C -= V (T^T (V^T C))`, and every factor of that is a GEMM.

    `t_block` is a dense `nb x nb` scratch; only its upper triangle is
    written and the caller packs it from there. `V` is read where `geqr2`
    left it -- the strict lower trapezoid of the panel, unit diagonal
    implicit.

    Column `i` costs `i` dot products of length `m - k - i`, one per
    thread, and then one `i x i` triangular multiply that thread 0 does
    alone: `nb^3 / 3` in total, which is nothing beside the panel it
    describes.

    Launch on the accelerator with `grid_dim=1`,
    `block_dim=_PANEL_THREADS`.
    """
    var lane = _lane[gpu]()
    var nt = _lanes[gpu]()
    var k0 = Int(k)
    var n_b = Int(nb)
    var rows = Int(m)

    if lane == 0:
        t_block.store[1](Coord(0, 0), tau[Coord(k0)])
    _sync[gpu]()

    for i in range(1, n_b):
        var this_tau = tau[Coord(k0 + i)]

        # `w[p] = -tau_i * (V[:, p] . V[:, i])`, parked in `T`'s own column
        # `i` because the triangular multiply below consumes it in place.
        var p = lane
        while p < i:
            var total = a[Coord(k0 + i, k0 + p)]
            for r in range(k0 + i + 1, rows):
                total += a[Coord(r, k0 + p)] * a[Coord(r, k0 + i)]
            t_block.store[1](Coord(p, i), -this_tau * total)
            p += nt
        _sync[gpu]()

        if lane == 0:
            # `T[0:i, i] = T[0:i, 0:i] @ w`, ascending in `p` so the only
            # entry of `w` overwritten before it is read is `w[p]` itself.
            for q in range(i):
                var total = Scalar[dtype](0)
                for r in range(q, i):
                    total += t_block[Coord(q, r)] * t_block[Coord(r, i)]
                t_block.store[1](Coord(q, i), total)
            t_block.store[1](Coord(i, i), this_tau)
        _sync[gpu]()


def pack_reflectors[
    dtype: DType,
    ALayout: TensorLayout,
    DLayout: TensorLayout,
    target: StaticString = "cpu",
](
    a: _View[dtype, ALayout],
    dst: _View[dtype, DLayout],
    k: Int,
    nb: Int,
    rows: Int,
    trans: Bool,
    ctx: DeviceContext,
) raises:
    """Materialize the panel's `V` -- unit lower trapezoidal, `rows x nb`
    -- into the dense `dst`, or its transpose.

    `geqr2_panel` stores `V` implicitly: the diagonal is an unwritten `1`,
    everything above it belongs to `R`, and only the strict lower trapezoid
    is really there. `matmul` cannot be told that, so the operand is built
    once per panel step. `trans` is a run-time argument rather than a
    parameter because both forms are needed in the same step and
    instantiating the kernel twice would only grow the binary.
    """
    if rows <= 0 or nb <= 0:
        return

    @always_inline
    def fill[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var a, var dst, var k, var nb, var trans}:
        var at = coord_to_index_list(coord)
        var i = at[1] if trans else at[0]
        var j = at[0] if trans else at[1]
        var value = Scalar[dtype](0)
        if i == j:
            value = Scalar[dtype](1)
        elif i > j:
            value = a[Coord(k + i, k + j)]
        dst.store[1](coord, value)

    if trans:
        elementwise[simd_width=1, target=target](fill, Coord(nb, rows), ctx)
    else:
        elementwise[simd_width=1, target=target](fill, Coord(rows, nb), ctx)


def trsm_diag[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    upper: Bool,
    unit: Bool,
    trans: Bool = False,
    target: StaticString = "cpu",
](
    a: _View[dtype, ALayout],
    b: _View[dtype, BLayout],
    k: Int,
    nb: Int,
    rhs: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """Solve the `nb x nb` triangular block at `(k, k)` against rows
    `k..k+nb` of `b`, for all `rhs` of its columns, in place.

    `trsv_diag` with a matrix on the right: same substitution, one per
    column of `b`. The columns are independent -- a substitution's
    dependency runs down the rows, not across -- so this is one
    `max.algorithm.elementwise` over `rhs` and needs no barrier, which is
    what makes it the multiple-right-hand-side case worth having rather
    than looping the vector version.

    `b` is the full right-hand side, `n x rhs`, addressed with offsets.
    `upper`, `unit` and `trans` mean what they mean in `trsv_diag`.
    """
    if nb <= 0 or rhs <= 0:
        return

    @always_inline
    def column[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var a, var b, var k, var nb}:
        var c = coord_to_index_list(coord)[0]
        for step in range(nb):
            var i = (nb - 1 - step) if upper else step
            var total = b[Coord(k + i, c)]
            if upper:
                for p in range(i + 1, nb):
                    total = (
                        total - _at[trans](a, k + i, k + p) * b[Coord(k + p, c)]
                    )
            else:
                for p in range(i):
                    total = (
                        total - _at[trans](a, k + i, k + p) * b[Coord(k + p, c)]
                    )
            comptime if unit:
                b.store[1](Coord(k + i, c), total)
            else:
                b.store[1](Coord(k + i, c), total / a[Coord(k + i, k + i)])

    elementwise[simd_width=1, target=target](column, Coord(rhs), ctx)


def laswp_matrix[
    dtype: DType,
    BLayout: TensorLayout,
    PLayout: TensorLayout,
    target: StaticString = "cpu",
](
    b: _View[dtype, BLayout],
    pivots: _View[DType.int32, PLayout],
    n: Int,
    rhs: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """`laswp` with a matrix on the right: apply the recorded row
    interchanges to every column of an `n x rhs` `b`.

    The interchanges compose, so the walk up `j` stays sequential; the
    columns do not, so each one gets its own lane. One
    `max.algorithm.elementwise` over `rhs`, `n` swaps each.
    """
    if rhs <= 0:
        return

    @always_inline
    def column[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var b, var pivots, var n}:
        var c = coord_to_index_list(coord)[0]
        for j in range(n):
            var other = Int(pivots[Coord(j)])
            if other != j:
                var keep = b[Coord(j, c)]
                b.store[1](Coord(j, c), b[Coord(other, c)])
                b.store[1](Coord(other, c), keep)

    elementwise[simd_width=1, target=target](column, Coord(rhs), ctx)


def trsm_right_lower_t[
    dtype: DType, ALayout: TensorLayout, target: StaticString = "cpu"
](
    a: _View[dtype, ALayout],
    k: Int,
    nb: Int,
    rows: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """`X := X @ L^-T` in place, for `X = a[k+nb:rows, k:k+nb]` and `L` the
    lower-triangular block at `(k, k)`.

    BLAS `trsm` at `side=Right, uplo=Lower, transa=T, diag=N`, which is the
    panel solve a right-looking Cholesky does after factoring its diagonal
    block. Every row of `X` solves against the same `L` and none of them
    depend on each other, so this is one `max.algorithm.elementwise` over
    `rows - k - nb` elements whose body forward-substitutes a row -- MAX
    picks the launch, numax writes no kernel, and the same call runs on
    either device.

    Does nothing when the panel is empty, which is the last block step.
    """
    var height = rows - k - nb
    if height <= 0:
        return

    comptime lanes = simd_width_of[dtype]()

    @always_inline
    @parameter
    def solve_row(index: Int):
        var row = k + nb + index
        for j in range(nb):
            # The dot product of this row's finished prefix against row
            # `k + j` of `L`. Both walk `p` along a row, so both are
            # contiguous and the whole thing vectorizes; only the `j` loop
            # around it carries a dependence.
            var acc = SIMD[dtype, lanes](0)
            var p = 0
            while p + lanes <= j:
                acc += a.load[lanes](Coord(row, k + p)) * a.load[lanes](
                    Coord(k + j, k + p)
                )
                p += lanes
            var total = a[Coord(row, k + j)] - acc.reduce_add()
            while p < j:
                total = total - a[Coord(row, k + p)] * a[Coord(k + j, k + p)]
                p += 1
            a.store[1](Coord(row, k + j), total / a[Coord(k + j, k + j)])

    # Not `elementwise`. Its CPU parallelization heuristic reads the element
    # *count*, and this launch is one element per row -- a few hundred at
    # any realistic `n` -- while each of those elements carries `nb^2 / 2`
    # flops. Measured on a domain of exactly this shape, `elementwise` runs
    # at 1.99 GFLOP/s and `parallelize` at 10.59, so the heuristic was
    # leaving eleven cores idle. `parallelize` is CPU-only, so the device
    # keeps `elementwise`, which is the right driver there anyway: a GPU
    # launch wants one thread per row and has no such threshold.
    comptime if target == "cpu":
        if height * nb * nb >= _PARALLEL_MIN_WORK:
            parallelize[solve_row](height)
        else:
            for index in range(height):
                solve_row(index)
    else:

        @always_inline
        def solve[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var a, var k, var nb}:
            solve_row(coord_to_index_list(coord)[0])

        elementwise[simd_width=1, target=target](solve, Coord(height), ctx)


def trsm_left_lower_unit[
    dtype: DType, ALayout: TensorLayout, target: StaticString = "cpu"
](
    a: _View[dtype, ALayout],
    k: Int,
    nb: Int,
    cols: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """`Y := L^-1 @ Y` in place, for `Y = a[k:k+nb, k+nb:cols]` and `L` the
    unit-diagonal lower-triangular block at `(k, k)`.

    BLAS `trsm` at `side=Left, uplo=Lower, transa=N, diag=U`: the `U12`
    block-row solve an LU does once its panel is factored. Unit diagonal
    because that is what `getrf_panel` leaves behind -- `L`'s diagonal is
    implicit and the stored diagonal belongs to `U`.

    Independent per *column* of `Y` rather than per row, so the
    `elementwise` runs over `cols - k - nb` and each body walks down one
    column. Does nothing when the block row is empty.
    """
    var width = cols - k - nb
    if width <= 0:
        return

    @always_inline
    @parameter
    def solve_col(index: Int):
        var col = k + nb + index
        for i in range(nb):
            var total = a[Coord(k + i, col)]
            for p in range(i):
                total = total - a[Coord(k + i, k + p)] * a[Coord(k + p, col)]
            a.store[1](Coord(k + i, col), total)

    # `parallelize` rather than `elementwise` on the host, for the reason
    # `trsm_right_lower_t` above gives: one element per column is far too
    # few for `elementwise`'s count-based threshold, however much work each
    # column carries. The inner `p` loop is not vectorized here the way the
    # other solve's is -- `a[k + p, col]` walks *down* a column, so
    # consecutive `p` are a row apart.
    comptime if target == "cpu":
        if width * nb * nb >= _PARALLEL_MIN_WORK:
            parallelize[solve_col](width)
        else:
            for index in range(width):
                solve_col(index)
    else:

        @always_inline
        def solve[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var a, var k, var nb}:
            solve_col(coord_to_index_list(coord)[0])

        elementwise[simd_width=1, target=target](solve, Coord(width), ctx)


def pack_block[
    dtype: DType,
    ALayout: TensorLayout,
    DLayout: TensorLayout,
    trans: Bool = False,
    target: StaticString = "cpu",
](
    a: _View[dtype, ALayout],
    dst: _View[dtype, DLayout],
    row0: Int,
    col0: Int,
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises:
    """Copy `a[row0:row0+rows, col0:col0+cols]` into the dense `dst`.

    The reason this exists is `matmul`'s stride blindness: MAX accepts a
    strided sub-view and then reads it as if it were contiguous, so an
    operand has to be made dense before it can be multiplied. The
    destination is a separately allocated block, so this is a device-to-
    device copy on the accelerator -- no host round trip -- and one
    `elementwise` launch on either target.

    Only the operands need it. The *result* of the product goes back into
    the strided matrix through `matmul`'s epilogue, which owns its own
    store, so nothing is ever unpacked.

    `trans=True` packs the transpose, `dst[i, j] = a[col0+j, row0+i]`,
    which is how a solve against `A^T` gets a dense operand without
    transposing `A` first. `rows` and `cols` describe `dst` either way.
    """

    @always_inline
    def copy[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var a, var dst, var row0, var col0}:
        var at = coord_to_index_list(coord)
        var i = at[0]
        var j = at[1]
        comptime if trans:
            dst.store[1](coord, a[Coord(col0 + j, row0 + i)])
        else:
            dst.store[w, alignment=alignment](
                coord, a.load[w, alignment=alignment](Coord(row0 + i, col0 + j))
            )

    # The untransposed copy walks `j` contiguously in both source and
    # destination, so it takes the native width; the body was already
    # written for it and only the launch was scalar. The transposed one
    # cannot: consecutive `j` there reads down a column of `a`, a strided
    # gather, so it stays one element at a time.
    comptime if trans:
        elementwise[simd_width=1, target=target](copy, Coord(rows, cols), ctx)
    else:
        elementwise[simd_width=simd_width_of[dtype](), target=target](
            copy, Coord(rows, cols), ctx
        )


def pack_vector[
    dtype: DType,
    ALayout: TensorLayout,
    DLayout: TensorLayout,
    target: StaticString = "cpu",
](
    a: _View[dtype, ALayout],
    dst: _View[dtype, DLayout],
    offset: Int,
    count: Int,
    ctx: DeviceContext,
) raises:
    """`pack_block` for a rank-1 view: `dst[i] = a[offset + i]`.

    A separate name rather than a rank parameter because the body indexes
    with a `Coord` of the tensor's own rank, and a rank-1 view cannot be
    given a two-coordinate index. Used to bring a right-hand side onto the
    device beside a factorization, and to copy one there so a solve does
    not overwrite its caller's vector.
    """

    @always_inline
    def copy[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var a, var dst, var offset}:
        var i = coord_to_index_list(coord)[0]
        dst.store[1](Coord(i), a[Coord(offset + i)])

    elementwise[simd_width=1, target=target](copy, Coord(count), ctx)
