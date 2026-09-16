"""MAX ships exactly one dense factorization, and it is not callable with
the `TileTensor` every other kernel in `linalg` takes.
"""

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from linalg.qr_factorization import qr_factorization
from max.gpu.host import DeviceContext

comptime dtype = DType.float64
comptime n = 4


def main() raises:
    var ctx = DeviceContext(api="cpu")
    var a_buf = ctx.enqueue_create_buffer[dtype](n * n)
    var tau_buf = ctx.enqueue_create_buffer[dtype](n)
    ctx.synchronize()

    var a = TileTensor(a_buf.unsafe_ptr(), row_major(Coord(n, n)))
    var tau = TileTensor(tau_buf.unsafe_ptr(), row_major(Coord(n)))

    qr_factorization(tau, a)
    print("factored")
