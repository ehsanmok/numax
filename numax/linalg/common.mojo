"""Helpers shared across `numax.linalg`, none of them public.

`_zeros` and `_PIVOT_FLOOR` both belong to the `Array[T, n*n]` tier: a
zeroed accumulator and the floor that keeps a division finite where a
tier-1 kernel cannot branch on the value. The `Tensor` tier has nothing
here -- it stages blocks with `panel.pack_block`, on whichever device the
data already lives on.

Nothing here is re-exported. It exists so that the operation modules do not
each carry their own copy.
"""

from std.collections import Array

from ..core.numeric import FloatLike


comptime _PIVOT_FLOOR = 1e-30


def _zeros[T: FloatLike, size: Int]() -> Array[T, size]:
    return Array[T, size](fill=T.constant(0.0))
