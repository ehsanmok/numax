"""MAX's `linalg.matmul` writes past the end of an `n == 1` destination
whose row count is not a multiple of the SIMD width.

Detected with a sentinel rather than a fault, so it reproduces without
depending on where the allocation happens to sit relative to a page.
"""

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu.host import DeviceContext
from std.sys.info import simd_width_of

comptime dtype = DType.float32
comptime m = 2       # rows of the destination
comptime k = 12      # contraction length
comptime guard = 8  # sentinel elements allocated after the destination
comptime sentinel = Scalar[dtype](-777.0)


def main() raises:
    var ctx = DeviceContext(api="cpu")
    comptime lanes = simd_width_of[dtype]()
    print("dtype float32, simd_width =", lanes, " m =", m, " (m % lanes =", m % lanes, ")")

    # a: m x k of ones.  x: k x 1 of ones.  So each real output is k.
    var a_buf = ctx.enqueue_create_buffer[dtype](m * k)
    var x_buf = ctx.enqueue_create_buffer[dtype](k)
    var y_buf = ctx.enqueue_create_buffer[dtype](m + guard)
    ctx.synchronize()

    var ap = a_buf.unsafe_ptr()
    var xp = x_buf.unsafe_ptr()
    var yp = y_buf.unsafe_ptr()

    for i in range(m * k):
        ap[unsafe_offset=i] = Scalar[dtype](1.0)
    for i in range(k):
        xp[unsafe_offset=i] = Scalar[dtype](1.0)
    # Every destination slot, real and guard, starts at the sentinel.
    for i in range(m + guard):
        yp[unsafe_offset=i] = sentinel

    # Hand MAX a destination that claims only the honest `m` rows.
    var a_t = TileTensor(ap, row_major(Coord(m, k)))
    var x_t = TileTensor(xp, row_major(Coord(k, 1)))
    var y_t = TileTensor(yp, row_major(Coord(m, 1)))

    matmul[target="cpu"](y_t, a_t, x_t, ctx)
    ctx.synchronize()

    print("expected: first", m, "slots =", k, ", the next", guard, "still", sentinel)
    var clobbered = 0
    for i in range(m + guard):
        var tag = String("real  ") if i < m else String("GUARD ")
        var bad = i >= m and yp[unsafe_offset=i] != sentinel
        if bad:
            clobbered += 1
        print("  y[", i, "] ", tag, yp[unsafe_offset=i], " <-- OVERWRITTEN" if bad else "")

    print()
    if clobbered > 0:
        print("RESULT: MAX wrote", clobbered, "element(s) past the end of the destination.")
    else:
        print("RESULT: no overwrite observed at this width.")

    _ = a_buf^
    _ = x_buf^
    _ = y_buf^
