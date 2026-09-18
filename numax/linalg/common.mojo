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

from ..core.tensorlike import TensorLike
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.collections import Array

from ..core.numeric import FloatLike


comptime _PIVOT_FLOOR = 1e-30


def _zeros[T: FloatLike, size: Int]() -> Array[T, size]:
    return Array[T, size](fill=T.constant(0.0))


def _mut_view[
    T: TensorLike
](a: T) -> TileTensor[T.dtype, T.LayoutType, MutAnyOrigin]:
    """`a`'s erased, writable view from a borrow: what the panel kernels'
    `_View` parameters are spelled as, for an operand they only read.

    The old `Tensor.view()` handed this out unconditionally; the tracked
    one follows the binding, and a `TensorLike` argument this tier takes by
    borrow yields a read-only tile. The kernels here copy their inputs into
    scratch or read them under `matmul`, so the mutability added back is
    never exercised. Do not use it for a destination.
    """
    var v = a.view()
    return TileTensor[T.dtype, T.LayoutType, MutAnyOrigin](
        ptr=v.ptr.unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](),
        layout=v.layout,
    )


def _mut_view_as[
    dtype: DType, T: TensorLike
](a: T) -> TileTensor[dtype, T.LayoutType, MutAnyOrigin]:
    """`_mut_view` with the lanes typed `dtype`, for a second operand under
    `where B.dtype == A.dtype`; see `TensorLike.view_as`."""
    var v = a.view()
    return TileTensor[dtype, T.LayoutType, MutAnyOrigin](
        ptr=v.ptr.unsafe_bitcast[Scalar[dtype]]()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutAnyOrigin](),
        layout=v.layout,
    )


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
