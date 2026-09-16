"""Device storage and `TileTensor` are keyed on `DType`, so a two-float
struct cannot be a tensor element -- not even one that is
`TrivialRegisterPassable` and therefore already legal as a kernel argument.

Uncomment either attempt to see the rejection. Both are compile errors,
and there is no third spelling: the `Storage`/`Engine` extension point
that `TileTensor` provides is itself parameterized on `dtype: DType`.
"""

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from max.gpu.host import DeviceContext


@fieldwise_init
struct Dual(Copyable, Movable):
    """A value and its derivative. Two floats, no pointer, no allocation."""

    var value: Float32
    var deriv: Float32


def main() raises:
    var ctx = DeviceContext(api="cpu")

    # --- Attempt 1: allocate a device buffer of `Dual` ---------------
    # `enqueue_create_buffer` takes a `DType` parameter, and `Dual` is a
    # struct, so this is a kind error rather than a missing overload.
    #
    # var buf = ctx.enqueue_create_buffer[Dual](16)

    # --- Attempt 2: name a `TileTensor` over `Dual` -----------------
    # `TileTensor`'s first value parameter is `dtype: DType`.
    #
    # var t = TileTensor[Dual, type_of(row_major(Coord(4, 4)))]

    # --- What is actually available: two parallel tensors ------------
    # The structure-of-arrays encoding, done by hand at the call site.
    # This is what numax's `to_tensor` emits for a `Dual`, and what
    # MAX's own `ArgMax` does with `acc_values`/`acc_indices`.
    var values = ctx.enqueue_create_buffer[DType.float32](16)
    var derivs = ctx.enqueue_create_buffer[DType.float32](16)
    ctx.synchronize()

    var v = TileTensor(values.unsafe_ptr(), row_major(Coord(4, 4)))
    var d = TileTensor(derivs.unsafe_ptr(), row_major(Coord(4, 4)))
    print("two tensors for one logical element:", v.ptr_at_offset(Coord(0, 0)) != d.ptr_at_offset(Coord(0, 0)))

    _ = values^
    _ = derivs^
