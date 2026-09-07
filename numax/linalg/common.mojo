"""Helpers shared across `numax.linalg`, none of them public.

`_zeros` and `_PIVOT_FLOOR` belong to the `Array[T, n*n]` tier: a zeroed
accumulator and the floor that keeps a division finite where a tier-1
kernel cannot branch on the value. `_staged` belongs to the `Tensor` tier,
where a blocked factorization has to hand MAX a contiguous block.

Nothing here is re-exported. It exists so that the operation modules do not
each carry their own copy.
"""

from max.gpu.host import DeviceContext
from std.collections import Array

from ..core.array import Dynamic, zeros_dyn
from ..core.numeric import FloatLike


comptime _PIVOT_FLOOR = 1e-30


def _zeros[T: FloatLike, size: Int]() -> Array[T, size]:
    return Array[T, size](fill=T.constant(0.0))


def _staged[
    dtype: DType
](
    values: List[Scalar[dtype]],
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises -> Dynamic[dtype, 2]:
    """A `rows x cols` device tensor holding `values`, row-major."""
    var staged = zeros_dyn[dtype, 2](rows, cols, ctx=ctx)
    staged.copy_from_host(values)
    return staged^
