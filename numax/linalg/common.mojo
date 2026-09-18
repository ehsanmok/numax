"""Helpers shared across `numax.linalg`, none of them public.

`_zeros` and `_PIVOT_FLOOR` belong to the `Array[T, n*n]` tier: a zeroed
accumulator and the floor that keeps a division finite where a tier-1
kernel cannot branch on the value. `_Dense` and `_device_identity` belong
to the `Tensor` tier: the view type MAX's `matmul` accepts, and the one
launch that seeds a square accumulator with `I` before the factorizations
start multiplying into it.

Nothing here is re-exported. It exists so that the operation modules do not
each carry their own copy.
"""

from layout import Coord, TileTensor, coord_to_index_list
from layout.tile_layout import row_major
from layout.tile_tensor import DefaultEngine
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.collections import Array

from ..core.numeric import FloatLike


comptime _PIVOT_FLOOR = 1e-30


def _zeros[T: FloatLike, size: Int]() -> Array[T, size]:
    return Array[T, size](fill=T.constant(0.0))


comptime _Dense[dtype: DType] = TileTensor[
    dtype,
    type_of(row_major(Coord(0, 0))),
    MutAnyOrigin,
    Engine=DefaultEngine[element_width=1],
]
"""A runtime-shaped, contiguous rank-2 view over an existing pointer.

The type of the operand `matmul` will accept and of the extents-only `c`
it insists on. Runtime-shaped because the trailing block shrinks every
step, contiguous because `matmul` reads its arguments as if they were.
"""


def _device_identity[
    dtype: DType, gpu: Bool = False
](d: _Dense[dtype], rows: Int, cols: Int, ctx: DeviceContext) raises:
    """Write the `rows x cols` identity into the dense view `d`, in one
    launch on whichever target `gpu` names.

    Every element is stored rather than only the diagonal, so this also
    serves a destination that was never zeroed.
    """

    @always_inline
    def seed[w: Int, alignment: Int = 1](coord: Coord) {var d}:
        var at = coord_to_index_list(coord)
        var one = Scalar[dtype](1) if at[0] == at[1] else Scalar[dtype](0)
        d.store[1](coord, one)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        seed, Coord(rows, cols), ctx
    )
