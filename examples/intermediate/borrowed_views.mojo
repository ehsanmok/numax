"""One routine, owned or borrowed: `Tensor`, `TensorView` and the `TensorLike` bound.

Every routine in the `Tensor` tier takes its tensors through `TensorLike`
(`numax.core.tensorlike`), which `Tensor` and `TensorView` both conform to. A
`TensorView` is a `TileTensor` someone else owns plus the device it lives on, so a
sub-block of a matrix is an argument in its own right: `cholesky` on the
leading quadrant of a larger matrix runs on the quadrant in place, with no
copy and no second `cholesky`.

The example builds a 4x4 whose leading 2x2 block is symmetric positive
definite, factors and solves against that block through a `TensorView`, sums a
row of a matrix the same way, writes through a `TensorView` and shows the parent
change, and ends with a routine written once against the bound and called
with both conformers. Nothing here names `TileTensor`: `a.block[2, 2](0, 0)`
is the compile-time-shaped block, `a[1:2]` a row slice, and both come back
as `TensorView`s over `a`'s own storage.
"""

from layout import Coord

from numax.core.tensor import Static, arange_n, reshape, zeros
from numax.core.elementwise import exp
from numax.core.tensorlike import TensorLike, TensorView, dim
from numax.linalg import cholesky, det, solve
from numax.stats import mean, sum

comptime f64 = DType.float64


def largest_row_mean[
    T: TensorLike
](a: T) raises -> Float64 where (
    T.LayoutType.rank == 2 and T.LayoutType.all_dims_known
):
    """The largest row mean of a matrix, written once for any conformer.

    `dim[T, i]` is the compile-time extent, `a.tile()` the tile; on a
    `Tensor` the tile borrows the tensor, on a `TensorView` it is the borrowed
    tile itself.
    """
    var v = a.tile()
    var best = Float64(0)
    for r in range(dim[T, 0]):
        var acc = Float64(0)
        for c in range(dim[T, 1]):
            acc += Float64(v[Coord(r, c)])
        var row_mean = acc / Float64(dim[T, 1])
        if r == 0 or row_mean > best:
            best = row_mean
    return best


def main() raises:
    # A 4x4 whose leading 2x2 block [[4, 2], [2, 3]] is SPD.
    var big = zeros[f64, 4, 4]()
    big[0, 0] = 4.0
    big[0, 1] = 2.0
    big[1, 0] = 2.0
    big[1, 1] = 3.0
    for i in range(4):
        for j in range(4):
            if i >= 2 or j >= 2:
                big[i, j] = Float64(10 * i + j)
    print("--- the parent matrix ---")
    print(big)

    # 1. Factor and solve against the quadrant, through a TensorView: a
    #    block of `big`, on `big`'s device, with no copy.
    var block = big.block[2, 2](0, 0)
    print("--- cholesky of the leading 2x2 block, no copy ---")
    print(cholesky(block))  # [[2, 0], [1, sqrt(2)]]
    var rhs = Static[f64, 2](big.context(), [1.0, 2.0])
    print("solve(block, [1, 2]):", solve(block, rhs))  # [-0.125, 0.75]
    print("det(block):", det(block))  # 8.0

    # 2. The same routines on an owned copy of the block agree exactly.
    var owned = Static[f64, 2, 2](big.context(), [4.0, 2.0, 2.0, 3.0])
    print("det(owned):", det(owned))

    # 3. Reductions and elementwise math take a TensorView too.
    var m = reshape[rows=3, cols=4](arange_n[12, f64]())
    var row1 = m[1:2]
    print("--- stats on one row of a 3x4, through a TensorView ---")
    print("sum(row 1):", sum(row1), " mean(row 1):", mean(row1))
    print("exp(row 1):", exp(TensorView(m.tile())).to_host()[4])

    # 4. Writing through a TensorView lands in the parent.
    var corner = big.block[2, 2](1, 1)
    corner.copy_from_host([-1.0, -1.0, -1.0, -1.0])
    print("--- after writing -1 into the trailing 2x2 through a TensorView ---")
    print(big)

    # 5. One generic routine, both conformers.
    print("--- largest row mean: Tensor", largest_row_mean(m), end="")
    print(", TensorView", largest_row_mean(TensorView(m.tile())))
