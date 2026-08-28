"""`numax.array`: NumPy-named creation and manipulation over `TileTensor`.

MAX's `layout` package ships `TileTensor` itself but no NumPy-named factory
functions -- `numax.array` is the thin gap-filling layer over it. Every
function here is `Plain`-only (no `FloatLike` conformer involvement): this
is the axis-2 (NumPy/SciPy parity) half of `numax`, not the axis-1
(composable-type) half `basic/gaussian.mojo` demonstrates.

`zeros`/`ones`/`full`/`eye`/`linspace`/`logspace` build a new `Tensor`
(an owned buffer plus a compile-time row-major layout -- see
`numax/array.mojo`'s own docstring for why a bare `TileTensor` can't be
returned from a factory function); `transpose`/`squeeze`/`stack` show the
three manipulation gaps this module fills.
"""

from numax.array import (
    eye,
    full,
    linspace,
    ones,
    squeeze,
    stack,
    transpose,
    zeros,
)

comptime dtype = DType.float32


def main() raises:
    print("--- creation ---")
    var z = zeros[dtype, 2, 3]()
    print("zeros[2, 3]:", z.num_elements, "elements, all", z[0])

    var o = ones[dtype, 4]()
    print("ones[4]:", o[0], o[1], o[2], o[3])

    var f = full[dtype, 3](Scalar[dtype](7))
    print("full[3](7):", f[0], f[1], f[2])

    var identity = eye[dtype, 3]()
    var iv = identity.view()
    print(
        "eye[3]: [",
        iv[0, 0],
        iv[0, 1],
        iv[0, 2],
        "] [",
        iv[1, 0],
        iv[1, 1],
        iv[1, 2],
        "] [",
        iv[2, 0],
        iv[2, 1],
        iv[2, 2],
        "]",
    )

    var ls = linspace[dtype, 5](Scalar[dtype](0), Scalar[dtype](1))
    print("linspace[5](0, 1):", ls[0], ls[1], ls[2], ls[3], ls[4])

    print("--- manipulation ---")
    var m = full[dtype, 2, 3](Scalar[dtype](0))
    var mv = m.view()
    var counter: Scalar[dtype] = 0
    for r in range(2):
        for c in range(3):
            mv[r, c] = counter
            counter += 1
    print(
        "m (2x3): [",
        mv[0, 0],
        mv[0, 1],
        mv[0, 2],
        "] [",
        mv[1, 0],
        mv[1, 1],
        mv[1, 2],
        "]",
    )

    var mt = transpose(m)
    var mtv = mt.view()
    print(
        "transpose(m) (3x2): [",
        mtv[0, 0],
        mtv[0, 1],
        "] [",
        mtv[1, 0],
        mtv[1, 1],
        "] [",
        mtv[2, 0],
        mtv[2, 1],
        "]",
    )

    var row = full[dtype, 1, 4](Scalar[dtype](0))
    var rv = row.view()
    for i in range(4):
        rv[0, i] = Scalar[dtype](i)
    var sq = squeeze(row)
    print("squeeze((1, 4)):", sq[0], sq[1], sq[2], sq[3])

    var a = linspace[dtype, 3](Scalar[dtype](0), Scalar[dtype](2))
    var b = linspace[dtype, 3](Scalar[dtype](10), Scalar[dtype](12))
    var st = stack(a, b)
    var sv = st.view()
    print("stack(a, b) row 0:", sv[0, 0], sv[0, 1], sv[0, 2])
    print("stack(a, b) row 1:", sv[1, 0], sv[1, 1], sv[1, 2])
