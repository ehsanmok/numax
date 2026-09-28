"""`numax.core.tensor`: NumPy-named creation and manipulation over `TileTensor`.

MAX's `layout` package ships `TileTensor` itself but no NumPy-named factory
functions -- `numax.core.tensor` is the thin gap-filling layer over it. Every
function here is `Plain`-only (no `FloatLike` conformer involvement): this
is the axis-2 (NumPy/SciPy parity) half of `numax`, not the axis-1
(composable-type) half `basic/gaussian.mojo` demonstrates.

`zeros`/`ones`/`full`/`eye`/`linspace`/`logspace` build a new `Tensor`
(an owned buffer plus a compile-time row-major layout -- see
`numax/core/tensor.mojo`'s own docstring for why a bare `TileTensor` can't be
returned from a factory function); `transpose`/`squeeze`/`stack` show the
three manipulation gaps this module fills.
"""


from max.gpu.host import DeviceContext
from numax.core.tensor import (
    arange_n,
    concatenate,
    eye,
    full,
    linspace,
    ones,
    ravel,
    reshape,
    split,
    squeeze,
    stack,
    transpose,
    zeros,
)

comptime dtype = DType.float32


def main() raises:
    var ctx = DeviceContext(api="cpu")
    print("--- creation ---")
    var z = zeros[dtype, 2, 3](ctx)
    print("zeros[2, 3]:", z.num_elements, "elements, all", z[0])

    var o = ones[dtype, 4](ctx)
    print("ones[4]:", o[0], o[1], o[2], o[3])

    var f = full[dtype, 3](7, ctx=ctx)
    print("full[3](7, ctx=ctx):", f[0], f[1], f[2])

    var identity = eye[3, dtype](ctx)
    print(
        "eye[3]: [",
        identity[0, 0],
        identity[0, 1],
        identity[0, 2],
        "] [",
        identity[1, 0],
        identity[1, 1],
        identity[1, 2],
        "] [",
        identity[2, 0],
        identity[2, 1],
        identity[2, 2],
        "]",
    )

    var ls = linspace[5, dtype](0, 1, ctx=ctx)
    print("linspace[5](0, 1, ctx=ctx):", ls[0], ls[1], ls[2], ls[3], ls[4])

    print("--- manipulation ---")
    var m = full[dtype, 2, 3](0, ctx=ctx)
    var counter: Scalar[dtype] = 0
    for r in range(2):
        for c in range(3):
            m[r, c] = counter
            counter += 1
    print(
        "m (2x3): [",
        m[0, 0],
        m[0, 1],
        m[0, 2],
        "] [",
        m[1, 0],
        m[1, 1],
        m[1, 2],
        "]",
    )

    var mt = transpose(m)
    print(
        "transpose(m) (3x2): [",
        mt[0, 0],
        mt[0, 1],
        "] [",
        mt[1, 0],
        mt[1, 1],
        "] [",
        mt[2, 0],
        mt[2, 1],
        "]",
    )

    var row = full[dtype, 1, 4](0, ctx=ctx)
    for i in range(4):
        row[0, i] = Scalar[dtype](i)
    var sq = squeeze(row)
    print("squeeze((1, 4)):", sq[0], sq[1], sq[2], sq[3])

    var a = linspace[3, dtype](0, 2, ctx=ctx)
    var b = linspace[3, dtype](10, 12, ctx=ctx)
    var st = stack(a, b)
    print("stack(a, b) row 0:", st[0, 0], st[0, 1], st[0, 2])
    print("stack(a, b) row 1:", st[1, 0], st[1, 1], st[1, 2])

    # Shape manipulation: arange_n -> reshape -> ravel round-trips, and
    # concatenate/split are inverses of each other.
    var r = arange_n[6, dtype](ctx=ctx)
    print("arange_n(6):", r[0], r[1], r[2], r[3], r[4], r[5])

    var grid = reshape[rows=2, cols=3](r)
    print(
        "reshape(6 -> 2x3): [",
        grid[0, 0],
        grid[0, 1],
        grid[0, 2],
        "] [",
        grid[1, 0],
        grid[1, 1],
        grid[1, 2],
        "]",
    )

    var back = ravel(grid)
    print(
        "ravel(2x3 -> 6):", back[0], back[1], back[2], back[3], back[4], back[5]
    )

    var left = arange_n[3, dtype](ctx=ctx)
    var right = arange_n[2, dtype](100, ctx=ctx)
    var joined = concatenate(left, right)
    print(
        "concatenate([0,1,2], [100,101]):",
        joined[0],
        joined[1],
        joined[2],
        joined[3],
        joined[4],
    )

    var parts = split[at=3](joined)
    print("split(at=3) head:", parts[0][0], parts[0][1], parts[0][2])
    print("split(at=3) tail:", parts[1][0], parts[1][1])
