"""`linalg.matmul` can transpose `b` but not `a`, so an `A^T B` product
needs a materialized transpose.
"""

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu.host import DeviceContext

comptime dtype = DType.float32
comptime n = 4


def main() raises:
    var ctx = DeviceContext(api="cpu")
    var a = ctx.enqueue_create_buffer[dtype](n * n)
    var b = ctx.enqueue_create_buffer[dtype](n * n)
    var c = ctx.enqueue_create_buffer[dtype](n * n)
    ctx.synchronize()

    var at = TileTensor(a.unsafe_ptr(), row_major(Coord(n, n)))
    var bt = TileTensor(b.unsafe_ptr(), row_major(Coord(n, n)))
    var ct = TileTensor(c.unsafe_ptr(), row_major(Coord(n, n)))

    # transpose_b exists ...
    matmul[target="cpu", transpose_b=True](ct, at, bt, ctx)
    # ... and the symmetric one does not.
    matmul[target="cpu", transpose_a=True](ct, at, bt, ctx)
    print("both accepted")
