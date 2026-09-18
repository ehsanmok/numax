"""`TensorLike`: one bound over an owned `Tensor` and a borrowed `View`.

**This module is tier-agnostic.** It defines no kernel; it names what a
kernel may be handed. `Tensor` (`numax.core.array`) owns its storage and
`View` borrows a `TileTensor` someone else owns, and every public routine in
the `Tensor` tier takes either through this one trait, so a factorization
runs on a whole tensor or on a sub-block of one without a copy:

```mojo
var a = zeros[DType.float64, 8, 8]()
var l = cholesky(a)                                   # owned
var block = View(a.view().tile[4, 4](0, 0), a.context())
var lb = cholesky(block)                              # borrowed, no copy
```

**Why a trait and not one type.** The Rust shape this borrows from is
`Cow`: one name for "owned or borrowed", and a kernel written once. Mojo
cannot spell that as a single `TileTensor` with an owning engine, and the
reasons are structural rather than missing work. MAX's `TensorEngine`
contract is borrowed-only (its `StorageType` must be trivially destructible
and the trait has no allocate or free hook); `TileTensor` is unconditionally
`ImplicitlyCopyable` and `TrivialRegisterPassable`, which a reference-counted
`DeviceBuffer` cannot satisfy; Mojo excludes `TrivialRegisterPassable` from
conditional conformance, so MAX could not make it depend on the engine even
if it wanted to; and `TileTensor` is MAX's type, with no way to add a
conformance from outside. What Mojo offers instead is a trait with
associated `comptime` members (traits themselves take no parameters, and
the compiler says so at parse time), which is also how MAX writes its own
`DenseTensor`. So this trait's member names match that one: `dtype`,
`LayoutType`, `Engine`.

**What the trait promises.** `view(ref self)` hands back a `TileTensor`
whose origin is the borrow of `self`: bind the tensor `mut` and the tile is
writable, bind it immutably and the tile is read-only, and either way the
tile keeps `self` alive. MAX's implicit origin cast then turns that tile into
the `MutAnyOrigin` spelling `numax.core.tensor` and every MAX kernel take,
at the call site and with parameter inference intact. `context()` names the
device the storage lives on, so a kernel can allocate its result beside its
input without being told twice.

**`View` holds a mutable tile on purpose.** Its origin is `MutOrigin`, and a
read-only use is expressed by binding the `View` immutably, the same way it
is for a `Tensor`. Wrapping an immutable tile is refused at compile time
rather than quietly producing a writable view over read-only storage.

A `View` built without a context is a host view, matching every factory in
`numax.core.array`: the `DeviceContext` comes last, optional, and its
absence means the CPU. Pass the owning tensor's `context()` for a view
over device memory.
"""

from layout import TileTensor
from layout.tile_layout import TensorLayout
from layout.tile_tensor import DefaultEngine, TensorEngine
from max.gpu.host import DeviceContext


trait TensorLike:
    """Something a `Tensor`-tier kernel can be handed: an owned `Tensor` or a
    borrowed `View`.

    Conformers name their element type and layout as associated members, so
    a generic routine spells `T.dtype`, `T.LayoutType.static_shape[0]` and
    `T.rank` where it used to spell explicit parameters. `Engine` is
    `DefaultEngine` unless a conformer says otherwise; nothing in `numax`
    does yet.
    """

    comptime dtype: DType
    """The element type."""

    comptime LayoutType: TensorLayout
    """The layout, which carries the shape one extent at a time, each either
    compile-time or run-time."""

    comptime Engine: TensorEngine = DefaultEngine[element_width=1]
    """The `TileTensor` storage engine the view is built on."""

    comptime rank: Int = Self.LayoutType.rank
    """The number of axes."""

    def view(
        ref self,
    ) -> TileTensor[
        Self.dtype, Self.LayoutType, origin_of(self), Engine=Self.Engine
    ]:
        """A `TileTensor` over the storage, with the mutability of this
        borrow of `self` and its lifetime.

        The type `numax.core.tensor`'s walks and every MAX kernel accept,
        once MAX's implicit origin cast has erased the origin at the call
        site. A view outliving the borrow is a compile error rather than a
        dangling pointer.
        """
        ...

    def context(self) raises -> DeviceContext:
        """The device the storage lives on."""
        ...


comptime dim[T: TensorLike, i: Int] = T.LayoutType.static_shape[i]
"""The compile-time extent of axis `i` of a `TensorLike`, for signatures.

`Static[T.dtype, dim[T, 0], dim[T, 0]]` is how a routine generic over `T`
names the square matrix its argument is. Only meaningful where
`T.LayoutType.all_dims_known`; a run-time extent reads as MAX's unknown
sentinel. The spelling in a return type has to match the one the body
builds: a `where` clause can check that `dim[T, 0] == dim[T, 1]`, but the
type checker does not rewrite one into the other.
"""

comptime is_row_major[T: TensorLike] = TileTensor[
    T.dtype, T.LayoutType, MutAnyOrigin
].is_row_major
"""Whether `T`'s strides are the contiguous row-major ones, at compile time.

The `where` clause every routine that flattens its argument must carry: a
`View` over `a.view().tile[2, 2](0, 0)` of a 4x4 has stride 4 on its first
axis, and a walk that treats it as eight contiguous elements reads the
wrong ones. `numax.core.tensor`'s walks already require this; the trait
makes it one spelling for the public tier too. For a run-time layout the
check is on the stride *types*, which every layout `numax` builds
satisfies, and the flattening helpers verify the values at run time.
"""


struct View[
    dtype_: DType,
    LayoutType_: TensorLayout,
    origin: MutOrigin,
](ImplicitlyCopyable, TensorLike, Writable):
    """A borrowed tensor: a `TileTensor` plus the device it lives on.

    The `TensorLike` conformer for storage `numax` does not own -- a
    sub-block of a `Tensor`, a `List`'s span, a tile another library handed
    over. It is what lets `cholesky(View(a.view().tile[4, 4](0, 0),
    a.context()))` factor one quadrant of `a` in place, with no copy and no
    second `cholesky`.

    Copying a `View` copies a pointer and a layout, so it is
    `ImplicitlyCopyable` like the tile it wraps. It owns nothing: dropping
    it frees nothing, and it is valid exactly as long as the storage behind
    `origin` is.
    """

    comptime dtype = Self.dtype_
    comptime LayoutType = Self.LayoutType_
    comptime rank = Self.LayoutType.rank
    comptime TileType = TileTensor[Self.dtype, Self.LayoutType, Self.origin]

    var tile: Self.TileType
    """The borrowed storage."""
    var ctx: DeviceContext
    """The device `tile` lives on."""

    def __init__(
        out self,
        tile: Self.TileType,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Borrow `tile`, on `ctx` or the host when none is given.

        The tile must be mutable: `a.view()` on a `mut`-bound or `var`
        tensor is one, `a.view().tile[...](...)` is one, a tile over an
        immutable borrow is not and is refused where it is written.
        """
        self.tile = tile
        self.ctx = ctx.value() if ctx else DeviceContext(api="cpu")

    def view(
        ref self,
    ) -> TileTensor[Self.dtype, Self.LayoutType, origin_of(self)]:
        """The wrapped tile, at the mutability of this borrow of `self`.

        An immutable `View` yields a read-only tile. The origin narrows
        from the storage's own to this borrow's, which is a shortening and
        so sound; nothing here widens a lifetime or adds mutability.
        """
        return TileTensor[Self.dtype, Self.LayoutType, origin_of(self)](
            ptr=self.tile.ptr.unsafe_mut_cast[
                origin_of(self).mut
            ]().unsafe_origin_cast[origin_of(self)](),
            layout=self.tile.layout,
        )

    def context(self) raises -> DeviceContext:
        """The device the storage lives on."""
        return self.ctx

    def size(self) -> Int:
        """The element count, read from the layout."""
        return self.tile.layout.size()

    def dim[i: Int](self) -> Int:
        """The extent of axis `i`."""
        return Int(self.tile.layout.shape[i]().value())

    def write_to(self, mut writer: Some[Writer]):
        """`print(v)`: the tile's own printer."""
        writer.write(self.tile)
