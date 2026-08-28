"""Row-wise softmax over a 2D `TileTensor`, on CPU and GPU.

Unlike `gaussian`/`sigmoid`/`erf`, `softmax` isn't a single `FloatLike`
kernel `map`ped elementwise -- each output element needs its whole row (the
row's max, for numerical stability, and the row's sum of exponentials), so
it's an orchestration of `numax.tensor`'s reduction and broadcast primitives
instead: `numax.special.softmax` on CPU (via `reduce_rows`/
`broadcast_op_rows`, `gpu=False`), and the same four-step recipe
hand-launched here on GPU (the same `reduce_rows`/`broadcast_op_rows`, with
`gpu=True`), since a GPU orchestration is a sequence of kernel launches
rather than a single function call.
"""

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from max.gpu.host import DeviceContext
from std.math import exp

from numax import Plain, softmax
from numax.tensor import (
    add_combine,
    broadcast_op_rows,
    max_combine,
    reduce_rows,
)

comptime dtype = DType.float32
comptime rows = 8
comptime cols = 16


def sub_exp_combine(a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    return (Plain[dtype, 1](a) + (-Plain[dtype, 1](b))).exp().v


def div_combine(a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    return (Plain[dtype, 1](a) / Plain[dtype, 1](b)).v


def fill_inputs(mut storage: List[Scalar[dtype]]):
    # Row 0 is mild; row 1 carries a large value to exercise the
    # subtract-the-max stabilization (naive softmax would overflow `exp`
    # here); the rest are pseudo-random-looking but deterministic.
    for r in range(rows):
        for c in range(cols):
            var v: Scalar[dtype]
            if r == 0:
                v = Scalar[dtype](c)
            elif r == 1:
                v = Scalar[dtype](50.0) if c == cols - 1 else Scalar[dtype](c)
            else:
                v = Scalar[dtype]((r * 37 + c * 11) % 23) - 11.0
            storage[r * cols + c] = v


def check_rows_sum_to_one(label: String, storage: List[Scalar[dtype]]) raises:
    var max_err = Float64(0)
    for r in range(rows):
        var total = Float64(0)
        for c in range(cols):
            total += Float64(storage[r * cols + c])
        max_err = max(max_err, abs(total - 1.0))
    print(label, "max |row sum - 1| =", max_err)


def main() raises:
    comptime layout2d = row_major[rows, cols]()
    comptime layout1d = row_major[rows]()

    # --- CPU, via `numax.special.softmax` ---
    var xs_storage = List[Scalar[dtype]](length=rows * cols, fill=0)
    fill_inputs(xs_storage)
    var xs = TileTensor(xs_storage, layout2d)

    var tmp_storage = List[Scalar[dtype]](length=rows * cols, fill=0)
    var tmp = TileTensor(tmp_storage, layout2d)
    var ys_storage = List[Scalar[dtype]](length=rows * cols, fill=0)
    var ys = TileTensor(ys_storage, layout2d)
    var row_max_storage = List[Scalar[dtype]](length=rows, fill=0)
    var row_max = TileTensor(row_max_storage, layout1d)
    var row_sum_storage = List[Scalar[dtype]](length=rows, fill=0)
    var row_sum = TileTensor(row_sum_storage, layout1d)

    softmax(xs, tmp, ys, row_max, row_sum)
    check_rows_sum_to_one("CPU  softmax:", ys_storage)
    print("CPU  row 1 (large-value row), last column:", ys[Coord(1, cols - 1)])

    # --- GPU, hand-launched from the same four primitives `softmax` uses ---
    var ctx = DeviceContext()
    print("GPU API:", ctx.api())

    var xs_buf = ctx.enqueue_create_buffer[dtype](rows * cols)
    var tmp_buf = ctx.enqueue_create_buffer[dtype](rows * cols)
    var ys_buf = ctx.enqueue_create_buffer[dtype](rows * cols)
    var row_max_buf = ctx.enqueue_create_buffer[dtype](rows)
    var row_sum_buf = ctx.enqueue_create_buffer[dtype](rows)

    with xs_buf.map_to_host() as h:
        for i in range(rows * cols):
            h[i] = xs_storage[i]

    var xs_gpu = TileTensor(xs_buf, layout2d)
    var tmp_gpu = TileTensor(tmp_buf, layout2d)
    var ys_gpu = TileTensor(ys_buf, layout2d)
    var row_max_gpu = TileTensor(row_max_buf, layout1d)
    var row_sum_gpu = TileTensor(row_sum_buf, layout1d)

    comptime row_block = 32
    comptime row_blocks = (rows + row_block - 1) // row_block
    comptime elem_block = 256
    comptime elem_blocks = (rows * cols + elem_block - 1) // elem_block

    ctx.enqueue_function[
        reduce_rows[
            RowsLayout=type_of(layout2d),
            OutLayout=type_of(layout1d),
            combine=max_combine[dtype],
            gpu=True,
        ]
    ](
        xs_gpu,
        row_max_gpu,
        SIMD[dtype, 1](-1e30),
        grid_dim=row_blocks,
        block_dim=row_block,
    )

    ctx.enqueue_function[
        broadcast_op_rows[
            RowsLayout=type_of(layout2d),
            ValuesLayout=type_of(layout1d),
            combine=sub_exp_combine,
            gpu=True,
        ]
    ](
        xs_gpu,
        row_max_gpu,
        tmp_gpu,
        grid_dim=elem_blocks,
        block_dim=elem_block,
    )

    ctx.enqueue_function[
        reduce_rows[
            RowsLayout=type_of(layout2d),
            OutLayout=type_of(layout1d),
            combine=add_combine[dtype],
            gpu=True,
        ]
    ](
        tmp_gpu,
        row_sum_gpu,
        SIMD[dtype, 1](0),
        grid_dim=row_blocks,
        block_dim=row_block,
    )

    ctx.enqueue_function[
        broadcast_op_rows[
            RowsLayout=type_of(layout2d),
            ValuesLayout=type_of(layout1d),
            combine=div_combine,
            gpu=True,
        ]
    ](
        tmp_gpu,
        row_sum_gpu,
        ys_gpu,
        grid_dim=elem_blocks,
        block_dim=elem_block,
    )
    ctx.synchronize()

    with ys_buf.map_to_host() as ys_h:
        var ys_gpu_storage = List[Scalar[dtype]](length=rows * cols, fill=0)
        for i in range(rows * cols):
            ys_gpu_storage[i] = ys_h[i]
        check_rows_sum_to_one("GPU  softmax:", ys_gpu_storage)
        print(
            "GPU  row 1 (large-value row), last column:",
            ys_h[1 * cols + (cols - 1)],
        )

        var max_diff = Float64(0)
        for i in range(rows * cols):
            max_diff = max(
                max_diff, abs(Float64(ys_h[i]) - Float64(ys_storage[i]))
            )
        print("max |GPU - CPU| softmax difference =", max_diff)
