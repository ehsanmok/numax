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
stride of its arguments (`.cursor/rules/max-feedback.mdc` has the
measurement: a tile view is accepted, then read as if contiguous), so a
blocked algorithm cannot pass tile views to it and has to pack instead --
and once packing is on the table, the panel kernels may as well address the
original matrix directly and save the copy. Offsets are `Int` arguments;
the block size is a compile-time bound so nothing is allocated per step.

**How the work is spread.** Two shapes appear, and which one a routine gets
follows from whether its columns are independent:

- The two `trsm`s and the packing are embarrassingly parallel over rows or
  columns, so they go through `max.algorithm.elementwise` -- one MAX launch,
  CPU threading and GPU dispatch included, no kernel written here.
- `potrf_diag` and `getrf_panel` are sequential over columns, with a
  cross-thread dependency at each one. They are single-thread-block kernels
  with `barrier()` between phases, so the whole panel is one launch rather
  than one launch per column.

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
from layout.tile_layout import TensorLayout
from layout.tile_tensor import PointerStorage
from max.algorithm.functional import elementwise
from max.gpu import barrier
from max.gpu.host import DeviceContext
from std.gpu import block_dim, thread_idx
from std.math import sqrt


comptime _PANEL_THREADS = 256
"""Threads in the single block `potrf_diag` and `getrf_panel` launch with.

One block, so this is the whole parallelism those two kernels get. 256 is
eight warps on CUDA and eight SIMD groups on Metal, enough to cover the
memory latency of a `block x block` tile without needing more shared state
than a strided loop.
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

    @always_inline
    def solve[w: Int, alignment: Int = 1](coord: Coord) {var a, var k, var nb}:
        var row = k + nb + coord_to_index_list(coord)[0]
        for j in range(nb):
            var total = a[Coord(row, k + j)]
            for p in range(j):
                total = total - a[Coord(row, k + p)] * a[Coord(k + j, k + p)]
            a.store[1](Coord(row, k + j), total / a[Coord(k + j, k + j)])

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
    def solve[w: Int, alignment: Int = 1](coord: Coord) {var a, var k, var nb}:
        var col = k + nb + coord_to_index_list(coord)[0]
        for i in range(nb):
            var total = a[Coord(k + i, col)]
            for p in range(i):
                total = total - a[Coord(k + i, k + p)] * a[Coord(k + p, col)]
            a.store[1](Coord(k + i, col), total)

    elementwise[simd_width=1, target=target](solve, Coord(width), ctx)


def pack_block[
    dtype: DType,
    ALayout: TensorLayout,
    DLayout: TensorLayout,
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
    """

    @always_inline
    def copy[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var a, var dst, var row0, var col0}:
        var at = coord_to_index_list(coord)
        var i = at[0]
        var j = at[1]
        dst.store[w](coord, a.load[w](Coord(row0 + i, col0 + j)))

    elementwise[simd_width=1, target=target](copy, Coord(rows, cols), ctx)
