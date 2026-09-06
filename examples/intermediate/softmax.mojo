"""Row-wise softmax over a 2D `Tensor`, on CPU and GPU.

Unlike `gaussian`/`sigmoid`/`erf`, `softmax` isn't a single `FloatLike`
kernel `map`ped elementwise -- each output element needs its whole row (the
row's max, for numerical stability, and the row's sum of exponentials), so
it's an orchestration of `numax.core.tensor`'s reduction and broadcast primitives
instead: `numax.special.activations.softmax` on CPU (via `reduce_rows`/
`broadcast_op_rows`, `gpu=False`), and the same four-step recipe
hand-launched here on GPU (the same `reduce_rows`/`broadcast_op_rows`, with
`gpu=True`), since a GPU orchestration is a sequence of kernel launches
rather than a single function call.
"""

from max.gpu.host import DeviceContext
from std.math import exp

from numax import Plain, Shaped, softmax
from numax.core.tensor import (
    add_combine,
    broadcast_op_rows,
    max_combine,
    reduce_rows,
)

comptime dtype = DType.float32
comptime rows = 8
comptime cols = 16


def sub_exp_combine(a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    return (Plain[dtype](a) - Plain[dtype](b)).exp().v


def div_combine(a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    return (Plain[dtype](a) / Plain[dtype](b)).v


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
    comptime Rows = Shaped[dtype, rows, cols]
    comptime PerRow = Shaped[dtype, rows]

    # --- CPU, via `numax.special.activations.softmax` ---
    var cpu = DeviceContext(api="cpu")
    var xs_storage = List[Scalar[dtype]](length=rows * cols, fill=0)
    fill_inputs(xs_storage)
    var xs = Rows(cpu, xs_storage.copy())

    var tmp = Rows(cpu)
    var ys = Rows(cpu)
    var row_max = PerRow(cpu)
    var row_sum = PerRow(cpu)

    softmax(xs.view(), tmp.view(), ys.view(), row_max.view(), row_sum.view())
    check_rows_sum_to_one("CPU  softmax:", ys.to_host())
    print("CPU  row 1 (large-value row), last column:", ys[1, cols - 1])

    # --- GPU, hand-launched from the same four primitives `softmax` uses ---
    var ctx = DeviceContext()
    print("GPU API:", ctx.api())

    var xs_gpu = Rows(ctx, xs_storage.copy())
    var tmp_gpu = Rows(ctx)
    var ys_gpu = Rows(ctx)
    var row_max_gpu = PerRow(ctx)
    var row_sum_gpu = PerRow(ctx)

    comptime row_block = 32
    comptime row_blocks = (rows + row_block - 1) // row_block
    comptime elem_block = 256
    comptime elem_blocks = (rows * cols + elem_block - 1) // elem_block

    ctx.enqueue_function[
        reduce_rows[
            RowsLayout=Rows.LayoutType,
            OutLayout=PerRow.LayoutType,
            combine=max_combine[dtype],
            gpu=True,
        ]
    ](
        xs_gpu.view(),
        row_max_gpu.view(),
        SIMD[dtype, 1](-1e30),
        grid_dim=row_blocks,
        block_dim=row_block,
    )

    ctx.enqueue_function[
        broadcast_op_rows[
            RowsLayout=Rows.LayoutType,
            ValuesLayout=PerRow.LayoutType,
            combine=sub_exp_combine,
            gpu=True,
        ]
    ](
        xs_gpu.view(),
        row_max_gpu.view(),
        tmp_gpu.view(),
        grid_dim=elem_blocks,
        block_dim=elem_block,
    )

    ctx.enqueue_function[
        reduce_rows[
            RowsLayout=Rows.LayoutType,
            OutLayout=PerRow.LayoutType,
            combine=add_combine[dtype],
            gpu=True,
        ]
    ](
        tmp_gpu.view(),
        row_sum_gpu.view(),
        SIMD[dtype, 1](0),
        grid_dim=row_blocks,
        block_dim=row_block,
    )

    ctx.enqueue_function[
        broadcast_op_rows[
            RowsLayout=Rows.LayoutType,
            ValuesLayout=PerRow.LayoutType,
            combine=div_combine,
            gpu=True,
        ]
    ](
        tmp_gpu.view(),
        row_sum_gpu.view(),
        ys_gpu.view(),
        grid_dim=elem_blocks,
        block_dim=elem_block,
    )
    ctx.synchronize()

    var ys_gpu_host = ys_gpu.to_host()
    check_rows_sum_to_one("GPU  softmax:", ys_gpu_host)
    print("GPU  row 1 (large-value row), last column:", ys_gpu[1, cols - 1])

    var ys_cpu_host = ys.to_host()
    var max_diff = Float64(0)
    for i in range(rows * cols):
        max_diff = max(
            max_diff, abs(Float64(ys_gpu_host[i]) - Float64(ys_cpu_host[i]))
        )
    print("max |GPU - CPU| softmax difference =", max_diff)
