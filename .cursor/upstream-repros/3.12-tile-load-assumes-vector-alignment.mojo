"""`TileTensor.load[w]` on a GPU target defaults `alignment` to the alignment
of the whole `SIMD[dtype, w]`, so a wide load at an element offset that is
not a multiple of `w` faults on NVIDIA. Metal and the host tolerate it.

Expected on an NVIDIA device: CUDA_ERROR_MISALIGNED_ADDRESS.
Passing `alignment=align_of[Scalar[dtype]]()` to the load makes it work.
"""

from std.sys import align_of

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from max.gpu.host import DeviceContext

comptime dtype = DType.float32


def kernel(
    src: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    var a = TileTensor(src, row_major(Coord(4, 37)))
    # Column 1 of a 37-wide row is 4 bytes in: not 16-byte aligned.
    var v = a.load[4](Coord(0, 1))
    dst.store(v)


def main() raises:
    var ctx = DeviceContext()
    var xs = ctx.enqueue_create_buffer[dtype](4 * 37)
    var ys = ctx.enqueue_create_buffer[dtype](4)
    xs.enqueue_fill(1)
    ctx.enqueue_function[kernel](xs, ys, grid_dim=1, block_dim=1)
    ctx.synchronize()
    print("ok")
