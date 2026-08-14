"""Drive any `FloatLike` kernel across a `TileTensor`, on CPU or GPU.

`ember`'s kernels (`gaussian`, `sigmoid`, `erf`, ...) are written once against
`FloatLike` and get plain SIMD, autodiff, or extra precision for free by the
type they're called with. That composability was, until now, only wired up
one element (or one SIMD lane group) at a time -- every example had to hand-
roll its own walk over memory. `TileTensor` (from MAX's `layout` package) is
the walk: it already knows how to lay out and address a buffer on the CPU or
in GPU global memory, so a `FloatLike` kernel only needs a way to read one
element out of a tensor, and write one element back.

That's what `map`, `map_simd`, and `map_gpu` are. `map`/`map_gpu` take three
small, non-capturing functions -- `kernel` (the `FloatLike`-generic body
itself), `wrap` (raw `SIMD` in, `T` out), and `unwrap` (`T` in, raw `SIMD`
out) -- and thread them through a `TileTensor` walk:

* `map` walks a CPU-backed tensor with an ordinary loop, one `width`-wide
  `SIMD` group per iteration. `width` is a plain type parameter here, so
  it's on the *caller* to actually pass a tensor shaped in `width`-sized
  groups (see `map_simd` below) -- `map` itself has no opinion on where
  `width` came from.
* `map_gpu` is the body of one GPU thread; launch it with `DeviceContext`'s
  `enqueue_function`, and every thread applies it to one element. GPU
  parallelism comes from thread count, not per-thread SIMD registers, so
  `map_gpu` stays one-element-per-thread rather than mirroring `map`'s
  vectorization.
* `map_simd` is the CPU entry point that actually gets you native SIMD
  width for free: give it a single `step` function generic over a width
  parameter (`def[w: Int](SIMD[dtype, w]) thin -> SIMD[dtype, w]`, typically
  `wrap` -> `kernel` -> `unwrap` composed inline), and it walks `xs.num_elements()
  - n % width` elements at the native SIMD width via `TileTensor.vectorize()`,
  then the remaining `n % width` elements one at a time with the same `step`
  instantiated at `width=1`. No overlapping reads, no dropped remainder.

`wrap`/`unwrap` are where a caller picks what "the answer" means for a given
`T`. `Plain.v` is the only field it has. `Dual` has `.value` and `.deriv`;
`Compensated` has `.value` and `.error` -- `unwrap` picks whichever field the
caller actually wants out of that pass. Wanting both means calling `map`
twice, once per field, since a `TileTensor` walk (like the kernel itself)
only ever produces one `SIMD` value per element.

All three are intentionally rank-1 (a flat walk over every element) --
`TileTensor`'s tiling, vectorizing, and partitioning APIs are still there for
a caller to shape *how* the underlying memory is laid out; `map`/`map_simd`/
`map_gpu` just need it flattened to a single index by the time they see it
(see `TileTensor.coalesce()` for multi-dimensional tensors with contiguous
storage).

`reduce`/`reduce_block_gpu` fold a tensor down to one value (sum, max, ...)
instead of producing another tensor the same shape. `reduce` is a plain CPU
loop; `reduce_block_gpu` is one thread block's worth of a shared-memory tree
reduction, writing one partial value per block -- combine the (small)
partials buffer with `reduce` again, CPU-side, for the final scalar. Both
take `add_op`/`max_op` (or any `def(T, T) thin -> T`) as their combining
`op`; `max_op` is derived entirely from operations `FloatLike` already has
(`abs`, `__add__`, `__neg__`, `constant`, `__truediv__`) rather than adding a
`max` method to the trait.

`reduce_rows`/`reduce_rows_gpu` and `broadcast_op_rows`/
`broadcast_op_rows_gpu` are the two 2D building blocks a row-wise op (like
softmax) needs on top of the rank-1 primitives above: `reduce_rows` folds
each row of a 2D `TileTensor` down to one value (one thread per row on GPU --
see their docstrings for why that's the deliberately simple choice here),
and `broadcast_op_rows` combines every element of a 2D tensor with its row's
value (one thread per element on GPU). The purely elementwise part of a
row-wise kernel (e.g. softmax's `exp`) doesn't need either of these -- use
`map`/`map_simd`/`map_gpu` on a `.coalesce()`d view of the same 2D tensor,
same as any other rank-1 walk.
"""

from layout import Coord, TileTensor
from layout.tile_layout import TensorLayout
from layout.tile_tensor import PointerStorage
from max.gpu import AddressSpace, barrier
from std.gpu import block_idx, global_idx, thread_idx
from std.memory import stack_allocation

from .numeric import FloatLike


def add_op[T: FloatLike](a: T, b: T) -> T:
    """Sum two `FloatLike` values -- the default `reduce`/`reduce_rows` op."""
    return a + b


def max_op[T: FloatLike](a: T, b: T) -> T:
    """The larger of two `FloatLike` values, lane-wise.

    `max(a, b) = (a + b + |a - b|) / 2` -- an identity that only needs
    operations `FloatLike` already requires (`abs`, `__add__`, `__neg__`,
    `constant`, `__truediv__`), so this needs no `max` method on the trait
    itself.
    """
    var diff = a + (-b)
    return (a + b + diff.abs()) / T.constant(2.0)


def map[
    dtype: DType,
    T: FloatLike,
    width: Int,
    LayoutType: TensorLayout,
    kernel: def(T) thin -> T,
    wrap: def(SIMD[dtype, width]) thin -> T,
    unwrap: def(T) thin -> SIMD[dtype, width],
](
    xs: TileTensor[
        dtype, LayoutType, ..., Storage=PointerStorage[element_width=width]
    ],
    mut ys: TileTensor[
        mut=True,
        dtype,
        LayoutType,
        ...,
        Storage=PointerStorage[element_width=width],
    ],
):
    """Apply `kernel` to every element of `xs`, writing results into `ys`.

    Walks the tensors on the calling thread -- meant for CPU-backed
    `TileTensor`s (built from a `List`, `Array`, or `HostBuffer`). For a
    GPU-backed tensor, use `map_gpu` as a kernel launched via
    `DeviceContext.enqueue_function`.
    """
    for i in range(xs.num_elements()):
        var x = xs.load[width](Coord(i))
        ys.store[width](Coord(i), unwrap(kernel(wrap(x))))


def map_simd[
    dtype: DType,
    LayoutType: TensorLayout,
    width: Int,
    step: def[w: Int](SIMD[dtype, w]) thin -> SIMD[dtype, w],
](
    xs: TileTensor[
        dtype, LayoutType, ..., Storage=PointerStorage[element_width=1]
    ],
    mut ys: TileTensor[
        mut=True,
        dtype,
        LayoutType,
        ...,
        Storage=PointerStorage[element_width=1],
    ],
):
    """Apply `step` to every element of `xs` at native SIMD width, CPU-side.

    `step` is generic over its own width parameter, so `map_simd` can
    instantiate it twice: once at `width` for every full group of `width`
    elements (via `TileTensor.vectorize()`, which reshapes a slice into
    non-overlapping `width`-wide groups), and once at `width=1` for
    whatever's left over when `xs.num_elements()` isn't a multiple of
    `width`. Pass `simd_width_of[dtype]()` as `width` to get the native
    SIMD width for the target.

    `step` is typically `wrap` -> a `FloatLike` kernel -> `unwrap` composed
    into one function, e.g. for `gaussian` over `Plain`:

    ```mojo
    def gaussian_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
        return gaussian(Plain[dtype, w](x)).v
    ```
    """
    var n = xs.num_elements()
    var vec_n = (n // width) * width
    if vec_n > 0:
        var xs_bulk = xs.slice((0, vec_n)).vectorize[width]()
        var ys_bulk = ys.slice((0, vec_n)).vectorize[width]()
        for i in range(xs_bulk.num_elements()):
            ys_bulk.store[width](
                Coord(i), step[width](xs_bulk.load[width](Coord(i)))
            )
    for i in range(vec_n, n):
        ys.store[1](Coord(i), step[1](xs.load[1](Coord(i))))


def map_gpu[
    dtype: DType,
    T: FloatLike,
    width: Int,
    LayoutType: TensorLayout,
    kernel: def(T) thin -> T,
    wrap: def(SIMD[dtype, width]) thin -> T,
    unwrap: def(T) thin -> SIMD[dtype, width],
](
    xs: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=width],
    ],
    ys: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=width],
    ],
):
    """One GPU thread's worth of `map`: apply `kernel` to a single element.

    Launch with `ctx.enqueue_function[map_gpu[...]](xs, ys, grid_dim=...,
    block_dim=...)`, where `xs`/`ys` are `TileTensor`s built from
    `DeviceBuffer`s. Every thread reads its own element (by
    `global_idx.x`), applies `kernel`, and writes it back -- the same
    `kernel`, `wrap`, and `unwrap` a CPU call to `map` would use.
    """
    var i = global_idx.x
    if i >= xs.num_elements():
        return
    var x = xs.load[width](Coord(i))
    ys.store[width](Coord(i), unwrap(kernel(wrap(x))))


def reduce[
    dtype: DType,
    LayoutType: TensorLayout,
    T: FloatLike,
    wrap: def(SIMD[dtype, 1]) thin -> T,
    unwrap: def(T) thin -> SIMD[dtype, 1],
    op: def(T, T) thin -> T,
](
    xs: TileTensor[
        dtype, LayoutType, ..., Storage=PointerStorage[element_width=1]
    ],
    init: SIMD[dtype, 1],
) -> SIMD[dtype, 1]:
    """Fold every element of `xs` down to one value with `op`, CPU-side.

    `init` is `op`'s identity (`0` for `add_op`, a very negative number for
    `max_op`) -- passed as a raw `SIMD` rather than a `T` so this has the
    same calling convention as `reduce_block_gpu`, whose kernel arguments
    must stay `DevicePassable`.
    """
    var acc = wrap(init)
    for i in range(xs.num_elements()):
        acc = op(acc, wrap(xs.load[1](Coord(i))))
    return unwrap(acc)


def reduce_block_gpu[
    dtype: DType,
    LayoutType: TensorLayout,
    PartialsLayout: TensorLayout,
    T: FloatLike,
    block_size: Int,
    wrap: def(SIMD[dtype, 1]) thin -> T,
    unwrap: def(T) thin -> SIMD[dtype, 1],
    op: def(T, T) thin -> T,
](
    xs: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    partials: TileTensor[
        dtype,
        PartialsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    identity: SIMD[dtype, 1],
):
    """One thread block's worth of `reduce`: a shared-memory tree reduction.

    Launch with `grid_dim = ceildiv(xs.num_elements(), block_size)`,
    `block_dim = block_size`; `partials` needs at least `grid_dim` elements,
    one written per block. `identity` pads out-of-range threads (the last
    block, if `xs.num_elements()` isn't a multiple of `block_size`) so they
    don't affect the result.

    This is deliberately a *two-pass* reduction, not a single kernel: finish
    it off by calling `reduce` again, CPU-side, over the small `partials`
    buffer once it's copied back to the host. A full single-pass reduction
    (a second, smaller kernel combining blocks, or a grid-wide atomic) is
    more machinery than a handful of blocks' worth of partials justifies.
    """
    var shared = stack_allocation[
        block_size, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var tid = thread_idx.x
    var gid = global_idx.x
    shared[unsafe_offset=tid] = (
        xs.load[1](Coord(gid)) if gid < xs.num_elements() else identity
    )
    barrier()

    var stride = block_size // 2
    while stride > 0:
        if tid < stride:
            shared[unsafe_offset=tid] = unwrap(
                op(
                    wrap(shared[unsafe_offset=tid]),
                    wrap(shared[unsafe_offset=tid + stride]),
                )
            )
        barrier()
        stride = stride // 2

    if tid == 0:
        partials.store[1](Coord(block_idx.x), shared[unsafe_offset=0])


def reduce_rows[
    dtype: DType,
    RowsLayout: TensorLayout,
    OutLayout: TensorLayout,
    T: FloatLike,
    wrap: def(SIMD[dtype, 1]) thin -> T,
    unwrap: def(T) thin -> SIMD[dtype, 1],
    op: def(T, T) thin -> T,
](
    xs: TileTensor[
        dtype, RowsLayout, ..., Storage=PointerStorage[element_width=1]
    ],
    mut dst: TileTensor[
        mut=True,
        dtype,
        OutLayout,
        ...,
        Storage=PointerStorage[element_width=1],
    ],
    init: SIMD[dtype, 1],
):
    """Fold each row of a 2D `xs` down to one value in `dst`, CPU-side.

    `dst` needs one element per row of `xs`. This is `reduce`, applied
    independently along `xs`'s first dimension -- the building block a
    row-wise op (row-wise softmax's per-row max and per-row sum) needs on
    top of `reduce`'s single whole-tensor fold.
    """
    var num_rows = xs.dim[0]()
    var num_cols = Int(xs.dim[1]())
    for r in range(num_rows):
        var acc = wrap(init)
        for c in range(num_cols):
            acc = op(acc, wrap(xs.load[1](Coord(r, c))))
        dst.store[1](Coord(r), unwrap(acc))


def reduce_rows_gpu[
    dtype: DType,
    RowsLayout: TensorLayout,
    OutLayout: TensorLayout,
    T: FloatLike,
    wrap: def(SIMD[dtype, 1]) thin -> T,
    unwrap: def(T) thin -> SIMD[dtype, 1],
    op: def(T, T) thin -> T,
](
    xs: TileTensor[
        dtype,
        RowsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    dst: TileTensor[
        dtype,
        OutLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    init: SIMD[dtype, 1],
):
    """One GPU thread's worth of `reduce_rows`: one thread folds one row.

    Launch with `grid_dim = ceildiv(num_rows, block_size)`,
    `block_dim = block_size` (a plain 1D launch, one thread per row). This
    trades away intra-row parallelism for simplicity -- each thread walks
    every column of its row in a loop, rather than the block-level tree
    reduction `reduce_block_gpu` uses for a single long row. That's the
    right trade for softmax-shaped workloads (many rows, each a modest
    width); a wide-row workload would want `reduce_block_gpu`'s approach
    per row instead, which 0.1.0 doesn't build.
    """
    var r = global_idx.x
    if r >= Int(xs.dim[0]()):
        return
    var num_cols = Int(xs.dim[1]())
    var acc = wrap(init)
    for c in range(num_cols):
        acc = op(acc, wrap(xs.load[1](Coord(r, c))))
    dst.store[1](Coord(r), unwrap(acc))


def broadcast_op_rows[
    dtype: DType,
    RowsLayout: TensorLayout,
    ValuesLayout: TensorLayout,
    T: FloatLike,
    wrap: def(SIMD[dtype, 1]) thin -> T,
    unwrap: def(T) thin -> SIMD[dtype, 1],
    op: def(T, T) thin -> T,
](
    xs: TileTensor[
        dtype, RowsLayout, ..., Storage=PointerStorage[element_width=1]
    ],
    row_values: TileTensor[
        dtype, ValuesLayout, ..., Storage=PointerStorage[element_width=1]
    ],
    mut ys: TileTensor[
        mut=True,
        dtype,
        RowsLayout,
        ...,
        Storage=PointerStorage[element_width=1],
    ],
):
    """Combine every element of `xs` with its row's value, CPU-side.

    `ys[r, c] = op(xs[r, c], row_values[r])` -- e.g. `sub_op` to subtract
    each row's max, or a division op to divide by each row's sum, the two
    row-broadcast steps row-wise softmax needs around its elementwise `exp`.
    """
    var num_rows = xs.dim[0]()
    var num_cols = Int(xs.dim[1]())
    for r in range(num_rows):
        var row_val = wrap(row_values.load[1](Coord(r)))
        for c in range(num_cols):
            var x = wrap(xs.load[1](Coord(r, c)))
            ys.store[1](Coord(r, c), unwrap(op(x, row_val)))


def broadcast_op_rows_gpu[
    dtype: DType,
    RowsLayout: TensorLayout,
    ValuesLayout: TensorLayout,
    T: FloatLike,
    wrap: def(SIMD[dtype, 1]) thin -> T,
    unwrap: def(T) thin -> SIMD[dtype, 1],
    op: def(T, T) thin -> T,
](
    xs: TileTensor[
        dtype,
        RowsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    row_values: TileTensor[
        dtype,
        ValuesLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    ys: TileTensor[
        dtype,
        RowsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
):
    """One GPU thread's worth of `broadcast_op_rows`: one thread, one element.

    Launch with `grid_dim = ceildiv(xs.num_elements(), block_size)`,
    `block_dim = block_size` -- a flat 1D launch over every element, same as
    `map_gpu`, with the row/column split recovered from the flat index.
    """
    var idx = global_idx.x
    if idx >= xs.num_elements():
        return
    var num_cols = Int(xs.dim[1]())
    var r = idx // num_cols
    var c = idx % num_cols
    var x = wrap(xs.load[1](Coord(r, c)))
    var row_val = wrap(row_values.load[1](Coord(r)))
    ys.store[1](Coord(r, c), unwrap(op(x, row_val)))
