"""Helpers shared across `numax.linalg`, none of them public.

`_zeros` and `_PIVOT_FLOOR` belong to the `Array[T, n*n]` tier: a zeroed
accumulator and the floor that keeps a division finite where a tier-1
kernel cannot branch on the value. `_Dense` belongs to the `Tensor` tier:
it is the view type MAX's `matmul` accepts, and it lives here because both
the factorizations and the triangular solves build one.

Nothing here is re-exported. It exists so that the operation modules do not
each carry their own copy.
"""

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import PointerStorage
from std.collections import Array

from ..core.numeric import FloatLike


comptime _PIVOT_FLOOR = 1e-30


def _zeros[T: FloatLike, size: Int]() -> Array[T, size]:
    return Array[T, size](fill=T.constant(0.0))


comptime _Dense[dtype: DType] = TileTensor[
    dtype,
    type_of(row_major(Coord(0, 0))),
    MutAnyOrigin,
    Storage=PointerStorage[element_width=1],
]
"""A runtime-shaped, contiguous rank-2 view over an existing pointer.

The type of the operand `matmul` will accept and of the extents-only `c`
it insists on. Runtime-shaped because the trailing block shrinks every
step, contiguous because `matmul` reads its arguments as if they were.
"""
