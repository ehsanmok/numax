"""`nn.cumsum` has no `target` parameter and no `DeviceContext`, so a
device-resident tensor has no scan kernel to reach.
"""

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from max.gpu.host import DeviceContext
from nn.cumsum import cumsum

comptime dtype = DType.float32
comptime n = 8


def main() raises:
    var ctx = DeviceContext(api="cpu")
    var xs = ctx.enqueue_create_buffer[dtype](n)
    var ys = ctx.enqueue_create_buffer[dtype](n)
    ctx.synchronize()

    var x = TileTensor(xs.unsafe_ptr(), row_major(Coord(n)))
    var y = TileTensor(ys.unsafe_ptr(), row_major(Coord(n)))

    # Ask for the device path the way every other MAX kernel spells it.
    cumsum[target="gpu"](y, x, ctx)
    print("scanned")
