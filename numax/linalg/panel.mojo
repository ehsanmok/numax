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


comptime _LATRD_MIN_WORK = 1 << 17
"""Work below which `latrd_w`'s two parallel phases stay serial.

Four times `_PARALLEL_MIN_WORK`, and it is a different number because the
comparison is different. `gemv_sub` and `latrd_column` are called once per
block *step*; `latrd_w` runs once per *column*, so a dispatch that does not
pay is paid `n` times over a factorization.

Measured on the M3 Pro at `n = 1024`, `float32`, `sytrd` end to end, the
machine not otherwise quiet. One `parallelize` costs tens of microseconds,
and the `w` build is about `n * (4j + 6)` scalar operations, so the
crossover is in `j`: handing the build to the threads from `j = 15` on
(`1 << 16`) took `block = 32` from 153 ms to 168 while `block = 64` went
from 199 to 188 -- the same dispatch, paying at one width and not at the
other. `1 << 17` puts the line between them. It is an absolute work count
rather than a width, so a larger `n` crosses it at a narrower panel, which
is the behaviour wanted.
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


def latrd_column[
    dtype: DType,
    ALayout: TensorLayout,
    LLayout: TensorLayout,
    target: StaticString = "cpu",
](
    a: _View[dtype, ALayout],
    left: _View[dtype, LLayout],
    k0: Int,
    j: Int,
    half: Int,
    n: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """Bring column `k0 + j` up to date with the panel's own reflectors.
    The first half of LAPACK's `latrd` inner step.

    A blocked tridiagonal reduction defers the trailing update to the end
    of the panel, so when column `i = k0 + j` is reached the matrix still
    lacks the rank-two updates of the `j` columns before it. This applies
    exactly those, and only to the one column the reflector is about to be
    formed from:

        A[i:, i] -= V[i:, :j] @ W[i, :j]^T + W[i:, :j] @ V[i, :j]^T

    `left` is the panel's `[V | W]`, `V` in columns `0 .. half-1` and `W`
    in columns `half .. 2*half-1`, which is the operand
    `_subtract_panel` later hands to the GEMM -- so the panel is
    accumulated once and read twice, here per column and there per panel.

    Nothing happens at `j == 0`: the first column of a panel is already
    current, and the pair (`latrd_column`, `sytd2_column`) degenerates to
    `sytd2_column` alone, which is the unblocked step.

    Independent per row, so it is one `max.algorithm.elementwise` on the
    accelerator and `parallelize` above `_PARALLEL_MIN_WORK` on the host
    -- the shape `gemv_sub` uses, and for the same reason: one element per
    row is below `elementwise`'s count threshold however much work each
    row carries.

    The reflector itself is `sytd2_column`, called immediately after this
    on the updated column.
    """
    if j <= 0:
        return
    var i = k0 + j
    var rows = n - i
    if rows <= 0:
        return

    @always_inline
    @parameter
    def update_row(index: Int):
        var row = i + index
        var total = a[Coord(row, i)]
        for c in range(j):
            total = total - left[Coord(row, c)] * left[Coord(i, half + c)]
            total = total - left[Coord(row, half + c)] * left[Coord(i, c)]
        a.store[1](Coord(row, i), total)

    comptime if target == "cpu":
        if rows * j * 4 >= _PARALLEL_MIN_WORK:
            parallelize[update_row](rows)
        else:
            for index in range(rows):
                update_row(index)
    else:

        @always_inline
        def update[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var a, var left, var i, var j, var half}:
            update_row(coord_to_index_list(coord)[0])

        elementwise[simd_width=1, target=target](update, Coord(rows), ctx)


def latrd_w[
    dtype: DType,
    VLayout: TensorLayout,
    PLayout: TensorLayout,
    LLayout: TensorLayout,
    RLayout: TensorLayout,
    TauLayout: TensorLayout,
    target: StaticString = "cpu",
](
    vpad: _View[dtype, VLayout],
    p: _View[dtype, PLayout],
    left: _View[dtype, LLayout],
    right: _View[dtype, LLayout],
    red: _View[dtype, RLayout],
    tau: _View[dtype, TauLayout],
    k0: Int,
    j: Int,
    half: Int,
    n: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """Build column `j` of a `latrd` panel's `W`, and park `v` and `w` in
    the two GEMM operands. The second half of LAPACK's `latrd` inner step.

    `p` is the *raw* product `A v`, where `A` is the matrix as stored --
    still missing the panel's own rank-two updates, exactly as
    `latrd_column`'s column was. Those updates are applied here instead of
    to the matrix, out of the `j` vectors already in the panel:

        p_current = p - V (W^T v) - W (V^T v)
        w         = tau * p_current - (tau^2 / 2) (p_current . v) v

    and `p_current . v = p . v - 2 sum_c (V^T v)_c (W^T v)_c` falls out of
    the same three reductions, so the whole step needs `2j + 1` dot
    products and no second pass. At `j == 0` it is
    `w = tau * p - (tau^2 / 2) (p . v) v`, the unblocked formula, with `p .
    v` computed here rather than by a separate `dot` -- which is what takes
    the per-column device-to-host synchronization out of the reduction.

    **Three launches, one per phase, because the middle one is the only
    part with a cross-thread dependency.** The `2j + 1` reductions are
    independent of each other, so they are one task each; the scalar
    `alpha` needs all of them and is one thread; the `w` build and the two
    operand stores are independent per row. Phases one and three take
    `gemv_sub`'s shape -- `parallelize` above `_PARALLEL_MIN_WORK` on the
    host, one `elementwise` on the accelerator -- and that is what keeps
    the `O(n j)` arithmetic off a single thread. As one single-block
    kernel it ran serially on the host and `sytrd` got monotonically
    *slower* from `block = 8` to `block = 64`, which is the shape a serial
    term makes in a panel-width sweep.

    `red` is `2 * half + 2` entries: `p . v` at `0`, `V^T v` at `1 ..
    j`, `W^T v` at `j + 1 .. 2j`, and `alpha` in the last slot, which is
    fixed rather than at `2j + 1` so the build phase reads one index
    whatever the panel column is.

    `tau` is read from the device, at `tau[k0 + j]`, for the same reason
    the reductions are: a host read of the scale would be a
    synchronization per column. A zero `tau` -- a column that was already
    reduced -- needs no branch, because it scales `w` to zero and the
    panel's update then adds nothing.

    **`w` is forced to zero at and above row `i = k0 + j`, and that is
    load-bearing.** `v` already vanishes there, but `p` does not: it is `A
    v` over the whole matrix, so its leading entries are the rows the
    reflector does not touch. Leaving them in `w` would make the panel's
    update write the columns where the reflectors are packed, and the
    factorization would lose the vectors it needs to form `Q`.

    The two operands are `left = [V | W]` and `right = [W | V]`, each `n x
    2*half` with `V`/`W` in the leading half -- so `left @ right^T` is
    `sum_c (v_c w_c^T + w_c v_c^T)`, the panel's whole symmetric rank-`2j`
    update in one GEMM with `transpose_b=True`. That identity is what lets
    numax skip the `syr2k` MAX does not ship.
    """
    var i = k0 + j
    var first = i + 1
    var cols = 2 * j + 1
    var slot = 2 * half + 1

    # Phase one: the `2j + 1` reductions, one task each.
    @always_inline
    @parameter
    def reduce_one(r: Int):
        var total = Scalar[dtype](0)
        for row in range(first, n):
            var value = Scalar[dtype](0)
            if r == 0:
                value = p[Coord(row)]
            elif r <= j:
                value = left[Coord(row, r - 1)]
            else:
                value = left[Coord(row, half + r - 1 - j)]
            total += value * vpad[Coord(row)]
        red.store[1](Coord(r), total)

    comptime if target == "cpu":
        if (n - first) * cols * 3 >= _LATRD_MIN_WORK:
            parallelize[reduce_one](cols)
        else:
            for r in range(cols):
                reduce_one(r)
    else:

        @always_inline
        def reduce_all[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var p,
            var vpad,
            var left,
            var red,
            var first,
            var j,
            var half,
            var n,
        }:
            reduce_one(coord_to_index_list(coord)[0])

        elementwise[simd_width=1, target=target](reduce_all, Coord(cols), ctx)

    # Phase two: `alpha`, which needs every reduction and is one scalar.
    @always_inline
    @parameter
    def fold_alpha():
        var this_tau = tau[Coord(i)]
        var total = red[Coord(0)]
        for c in range(j):
            total -= (
                Scalar[dtype](2) * red[Coord(1 + c)] * red[Coord(1 + j + c)]
            )
        red.store[1](
            Coord(slot), -this_tau * this_tau * total / Scalar[dtype](2)
        )

    comptime if target == "cpu":
        fold_alpha()
    else:

        @always_inline
        def fold[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var red, var tau, var i, var j, var slot}:
            fold_alpha()

        elementwise[simd_width=1, target=target](fold, Coord(1), ctx)

    # Phase three: the masked axpy and the two operand stores, per row.
    @always_inline
    @parameter
    def build_row(row: Int):
        var this_tau = tau[Coord(i)]
        var alpha = red[Coord(slot)]
        var v = vpad[Coord(row)]
        var w = Scalar[dtype](0)
        if row >= first:
            var acc = p[Coord(row)]
            for c in range(j):
                acc -= left[Coord(row, c)] * red[Coord(1 + j + c)]
                acc -= left[Coord(row, half + c)] * red[Coord(1 + c)]
            w = this_tau * acc + alpha * v
        left.store[1](Coord(row, j), v)
        left.store[1](Coord(row, half + j), w)
        right.store[1](Coord(row, j), w)
        right.store[1](Coord(row, half + j), v)

    comptime if target == "cpu":
        if n * (4 * j + 6) >= _LATRD_MIN_WORK:
            parallelize[build_row](n)
        else:
            for row in range(n):
                build_row(row)
    else:

        @always_inline
        def build[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var p,
            var vpad,
            var left,
            var right,
            var red,
            var tau,
            var i,
            var first,
            var j,
            var half,
            var slot,
        }:
            build_row(coord_to_index_list(coord)[0])

        elementwise[simd_width=1, target=target](build, Coord(n), ctx)


def labrd_column[
    dtype: DType,
    ALayout: TensorLayout,
    LLayout: TensorLayout,
    XLayout: TensorLayout,
    YLayout: TensorLayout,
    RLayout: TensorLayout,
    target: StaticString = "cpu",
](
    a: _View[dtype, ALayout],
    left: _View[dtype, LLayout],
    x: _View[dtype, XLayout],
    y: _View[dtype, YLayout],
    right: _View[dtype, RLayout],
    k0: Int,
    j: Int,
    m: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """Bring column `k0 + j` up to date with the panel's own reflectors.
    The first of LAPACK's `labrd` four steps.

    A blocked bidiagonal reduction defers the trailing update to the end
    of the panel, so when column `i = k0 + j` is reached the matrix still
    lacks the `j` rank-one pairs before it. This applies exactly those,
    and only to the one column the left reflector is about to be formed
    from:

        A[i:, i] -= V[i:, :j] Y[i, :j]^T + X[i:, :j] U[i, :j]^T

    `V` is the left reflectors as columns of `left` and `U` the right
    reflectors as rows of `right`, both in the layout `TensorBidiagonal`
    documents; `Y` and `X` are the corrections `labrd_y` and `labrd_x`
    accumulate. Nothing happens at `j == 0`, where the pair
    (`labrd_column`, `gebd2_col`) degenerates to `gebd2_col` alone, which
    is the unblocked step.

    Independent per row, so it is `gemv_sub`'s shape: `parallelize` above
    `_PARALLEL_MIN_WORK` on the host, one `elementwise` on the
    accelerator.
    """
    if j <= 0:
        return
    var i = k0 + j
    var rows = m - i
    if rows <= 0:
        return

    @always_inline
    @parameter
    def update_row(index: Int):
        var row = i + index
        var total = a[Coord(row, i)]
        for c in range(j):
            total -= left[Coord(row, k0 + c)] * y[Coord(i, c)]
            total -= x[Coord(row, c)] * right[Coord(k0 + c, i)]
        a.store[1](Coord(row, i), total)

    comptime if target == "cpu":
        if rows * j * 4 >= _PARALLEL_MIN_WORK:
            parallelize[update_row](rows)
        else:
            for index in range(rows):
                update_row(index)
    else:

        @always_inline
        def update[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var a, var left, var x, var y, var right, var i, var j, var k0
        }:
            update_row(coord_to_index_list(coord)[0])

        elementwise[simd_width=1, target=target](update, Coord(rows), ctx)


def labrd_row[
    dtype: DType,
    ALayout: TensorLayout,
    LLayout: TensorLayout,
    XLayout: TensorLayout,
    YLayout: TensorLayout,
    RLayout: TensorLayout,
    target: StaticString = "cpu",
](
    a: _View[dtype, ALayout],
    left: _View[dtype, LLayout],
    x: _View[dtype, XLayout],
    y: _View[dtype, YLayout],
    right: _View[dtype, RLayout],
    k0: Int,
    j: Int,
    n: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """Bring row `k0 + j` up to date with the panel's reflectors, the
    left one of this very column included. The third of `labrd`'s steps.

        A[i, i+1:] -= V[i, :j+1] Y[i+1:, :j+1]^T + X[i, :j] U[i+1:, :j]^T

    The `j + 1` rather than `j` is the point: the left reflector formed a
    moment ago is deferred like every other, so the row the right
    reflector is taken from has to carry it. `V[i, j]` is the reflector's
    own unit entry, so the term is `Y[:, j]` unscaled -- which is why this
    step runs at `j == 0` where `labrd_column` does not.

    Independent per column, `gemv_sub`'s shape again.
    """
    var i = k0 + j
    var cols = n - i - 1
    if cols <= 0:
        return

    @always_inline
    @parameter
    def update_col(index: Int):
        var col = i + 1 + index
        var total = a[Coord(i, col)]
        for c in range(j + 1):
            total -= left[Coord(i, k0 + c)] * y[Coord(col, c)]
        for c in range(j):
            total -= x[Coord(i, c)] * right[Coord(k0 + c, col)]
        a.store[1](Coord(i, col), total)

    comptime if target == "cpu":
        if cols * (j + 1) * 4 >= _PARALLEL_MIN_WORK:
            parallelize[update_col](cols)
        else:
            for index in range(cols):
                update_col(index)
    else:

        @always_inline
        def update[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var a, var left, var x, var y, var right, var i, var j, var k0
        }:
            update_col(coord_to_index_list(coord)[0])

        elementwise[simd_width=1, target=target](update, Coord(cols), ctx)


def labrd_y[
    dtype: DType,
    PLayout: TensorLayout,
    LLayout: TensorLayout,
    XLayout: TensorLayout,
    YLayout: TensorLayout,
    RLayout: TensorLayout,
    RedLayout: TensorLayout,
    TauLayout: TensorLayout,
    target: StaticString = "cpu",
](
    t1: _View[dtype, PLayout],
    left: _View[dtype, LLayout],
    x: _View[dtype, XLayout],
    y: _View[dtype, YLayout],
    right: _View[dtype, RLayout],
    red: _View[dtype, RedLayout],
    tau: _View[dtype, TauLayout],
    k0: Int,
    j: Int,
    m: Int,
    n: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """Build column `j` of a `labrd` panel's `Y`, the right-hand
    correction the left reflector leaves behind. `labrd`'s second step.

    `t1` is the *raw* product `v^T A` over the matrix as stored, still
    missing the panel's own deferred pairs. Those are applied here instead
    of to the matrix, out of the `j` vectors already in the panel:

        Y[:, j] = tauq * (t1 - Y (V^T v) - U^T (X^T v))

    masked to zero at and below row `i = k0 + j`, which is what confines
    the deferred update to the trailing block -- the unblocked reduction
    spells the same thing as `_zero_prefix` on `v^T A`.

    Two launches, for the reason `latrd_w` gives: the `2j` reductions are
    independent of each other and the build is independent per row, but
    the build needs every reduction. `red` is the `2 * width + 2` scratch
    they share, `V^T v` in `0 .. j-1` and `X^T v` in `j .. 2j-1`.

    `tauq` is read from the device at `tau[i]`, so the per-column
    `taus_left.to_host()[k]` of the unblocked reduction is gone. A zero
    `tauq` needs no branch: it scales the whole column to zero and the
    panel's update then adds nothing.
    """
    var i = k0 + j

    @always_inline
    @parameter
    def reduce_one(r: Int):
        var total = Scalar[dtype](0)
        if r < j:
            for row in range(i, m):
                total += left[Coord(row, k0 + r)] * left[Coord(row, i)]
        else:
            for row in range(i, m):
                total += x[Coord(row, r - j)] * left[Coord(row, i)]
        red.store[1](Coord(r), total)

    if j > 0:
        comptime if target == "cpu":
            if (m - i) * 2 * j * 3 >= _LATRD_MIN_WORK:
                parallelize[reduce_one](2 * j)
            else:
                for r in range(2 * j):
                    reduce_one(r)
        else:

            @always_inline
            def reduce_all[
                w: Int, alignment: Int = 1
            ](coord: Coord) {
                var left, var x, var red, var i, var j, var k0, var m
            }:
                reduce_one(coord_to_index_list(coord)[0])

            elementwise[simd_width=1, target=target](
                reduce_all, Coord(2 * j), ctx
            )

    @always_inline
    @parameter
    def build_col(col: Int):
        var value = Scalar[dtype](0)
        if col > i:
            var acc = t1[Coord(col)]
            for c in range(j):
                acc -= y[Coord(col, c)] * red[Coord(c)]
                acc -= right[Coord(k0 + c, col)] * red[Coord(j + c)]
            value = tau[Coord(i)] * acc
        y.store[1](Coord(col, j), value)

    comptime if target == "cpu":
        if n * (4 * j + 4) >= _LATRD_MIN_WORK:
            parallelize[build_col](n)
        else:
            for col in range(n):
                build_col(col)
    else:

        @always_inline
        def build[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var t1, var y, var right, var red, var tau, var i, var j, var k0
        }:
            build_col(coord_to_index_list(coord)[0])

        elementwise[simd_width=1, target=target](build, Coord(n), ctx)


def labrd_x[
    dtype: DType,
    PLayout: TensorLayout,
    LLayout: TensorLayout,
    XLayout: TensorLayout,
    YLayout: TensorLayout,
    RLayout: TensorLayout,
    RedLayout: TensorLayout,
    TauLayout: TensorLayout,
    target: StaticString = "cpu",
](
    t2: _View[dtype, PLayout],
    left: _View[dtype, LLayout],
    x: _View[dtype, XLayout],
    y: _View[dtype, YLayout],
    right: _View[dtype, RLayout],
    red: _View[dtype, RedLayout],
    tau: _View[dtype, TauLayout],
    k0: Int,
    j: Int,
    m: Int,
    n: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """Build column `j` of a `labrd` panel's `X`, the left-hand correction
    the right reflector leaves behind. `labrd`'s fourth step.

    `t2` is the raw product `A u` over the matrix as stored; the panel's
    deferred pairs, this column's left reflector included, are applied
    here:

        X[:, j] = taup * (t2 - V (Y^T u) - X (U u))

    masked to zero at and above row `i = k0 + j`, the mirror of `labrd_y`'s
    mask and of the unblocked reduction's `_zero_prefix` on `A u`. The
    `Y^T u` reduction runs over `j + 1` columns rather than `j` for the
    reason `labrd_row` gives: the left reflector of this very column is
    deferred too.

    Two launches and a shared `red`, `Y^T u` in `0 .. j` and `U u` in
    `j+1 .. 2j`. `taup` is read from the device at `tau[i]`.
    """
    var i = k0 + j

    @always_inline
    @parameter
    def reduce_one(r: Int):
        var total = Scalar[dtype](0)
        if r <= j:
            for col in range(i + 1, n):
                total += y[Coord(col, r)] * right[Coord(i, col)]
        else:
            for col in range(i + 1, n):
                total += (
                    right[Coord(k0 + r - j - 1, col)] * right[Coord(i, col)]
                )
        red.store[1](Coord(r), total)

    comptime if target == "cpu":
        if (n - i) * (2 * j + 1) * 3 >= _LATRD_MIN_WORK:
            parallelize[reduce_one](2 * j + 1)
        else:
            for r in range(2 * j + 1):
                reduce_one(r)
    else:

        @always_inline
        def reduce_all[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var y, var right, var red, var i, var j, var k0, var n
        }:
            reduce_one(coord_to_index_list(coord)[0])

        elementwise[simd_width=1, target=target](
            reduce_all, Coord(2 * j + 1), ctx
        )

    @always_inline
    @parameter
    def build_row(row: Int):
        var value = Scalar[dtype](0)
        if row > i:
            var acc = t2[Coord(row)]
            for c in range(j + 1):
                acc -= left[Coord(row, k0 + c)] * red[Coord(c)]
            for c in range(j):
                acc -= x[Coord(row, c)] * red[Coord(j + 1 + c)]
            value = tau[Coord(i)] * acc
        x.store[1](Coord(row, j), value)

    comptime if target == "cpu":
        if m * (4 * j + 6) >= _LATRD_MIN_WORK:
            parallelize[build_row](m)
        else:
            for row in range(m):
                build_row(row)
    else:

        @always_inline
        def build[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var t2, var left, var x, var red, var tau, var i, var j, var k0
        }:
            build_row(coord_to_index_list(coord)[0])

        elementwise[simd_width=1, target=target](build, Coord(m), ctx)


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
    transposed: Bool = False,
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

    `t_block` is a dense `nb x nb` scratch, cleared here and then built in
    its upper triangle -- or, with `transposed`, its lower one: `T^T`
    directly, so the caller that applies `I - V T^T V^T` reads it as it
    is rather than packing a transposed copy in a launch of its own. The
    clearing is what lets one scratch serve panels of different widths.
    `V` is read where `geqr2` left it -- the strict lower trapezoid of the
    panel, unit diagonal implicit.

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

    # Clear the whole block first. Only one triangle is written below, and
    # `t_block` is a view at this step's `nb` over a scratch sized for the
    # widest one: a ragged last panel, or the full panels that follow it
    # when `Q` is formed in reverse, would otherwise read the previous
    # step's entries through a different row stride as the other triangle.
    var e = lane
    while e < n_b * n_b:
        t_block.store[1](Coord(e // n_b, e % n_b), Scalar[dtype](0))
        e += nt
    _sync[gpu]()

    # `T[row, col]` lives at `(row, col)`, or at `(col, row)` when the
    # transpose is what is being built; every access below goes through
    # this one swap.
    @always_inline
    def at(row: Int, col: Int) -> Coord[Int, Int]:
        comptime if transposed:
            return Coord(col, row)
        else:
            return Coord(row, col)

    if lane == 0:
        t_block.store[1](at(0, 0), tau[Coord(k0)])
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
            t_block.store[1](at(p, i), -this_tau * total)
            p += nt
        _sync[gpu]()

        if lane == 0:
            # `T[0:i, i] = T[0:i, 0:i] @ w`, ascending in `p` so the only
            # entry of `w` overwritten before it is read is `w[p]` itself.
            for q in range(i):
                var total = Scalar[dtype](0)
                for r in range(q, i):
                    total += t_block[at(q, r)] * t_block[at(r, i)]
                t_block.store[1](at(q, i), total)
            t_block.store[1](at(i, i), this_tau)
        _sync[gpu]()


def pack_reflectors[
    dtype: DType,
    ALayout: TensorLayout,
    DLayout: TensorLayout,
    TLayout: TensorLayout,
    target: StaticString = "cpu",
](
    a: _View[dtype, ALayout],
    dst: _View[dtype, DLayout],
    dst_t: _View[dtype, TLayout],
    k: Int,
    nb: Int,
    rows: Int,
    ctx: DeviceContext,
) raises:
    """Materialize the panel's `V` -- unit lower trapezoidal, `rows x nb`
    -- into the dense `dst`, and its transpose into `dst_t`, in one
    launch.

    `geqr2_panel` stores `V` implicitly: the diagonal is an unwritten `1`,
    everything above it belongs to `R`, and only the strict lower trapezoid
    is really there. `matmul` cannot be told that, so the operand is built
    once per panel step -- and since it transposes `b` and never `a`, the
    step needs `V` for `V Y` and `V^T` for `V^T C` both. One walk over
    `(rows, nb)` writes each entry to both destinations; the transposed
    store is strided, but it was strided as its own launch too, and a
    launch is what this saves.
    """
    if rows <= 0 or nb <= 0:
        return

    @always_inline
    def fill[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var a, var dst, var dst_t, var k, var nb}:
        var at = coord_to_index_list(coord)
        var i = at[0]
        var j = at[1]
        var value = Scalar[dtype](0)
        if i == j:
            value = Scalar[dtype](1)
        elif i > j:
            value = a[Coord(k + i, k + j)]
        dst.store[1](coord, value)
        dst_t.store[1](Coord(j, i), value)

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
