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

**`View` tracks its owner immutably and writes through the binding.** It
accepts any mutable `TileTensor` and stores it at the *immutable* form of
that tile's origin. Tracked, so the owner stays alive for as long as the
`View` is used (an untracked spelling was tried and dangled the moment the
owner's last mention passed). Immutable, so two `View`s over one tensor, or
the same one passed twice to `add`, are immutable aliases the exclusivity
checker accepts, where a tracked mutable origin made `add(v, v)` an error a
`Tensor` never raises. Mutability then comes from how the `View` itself is
bound: `view(ref self)` on a `var` or `mut` `View` re-adds it, on an
immutable binding it does not, the same rule `Tensor` follows. Wrapping an
immutable tile is refused at compile time rather than quietly producing a
writable view over read-only storage.

A `View` built without a context is a host view, matching every factory in
`numax.core.array`: the `DeviceContext` comes last, optional, and its
absence means the CPU. Pass the owning tensor's `context()` for a view
over device memory.
"""

from layout import TileTensor
from layout.tile_layout import TensorLayout
from layout.tile_tensor import DefaultEngine
from max.gpu.host import DeviceContext


trait TensorLike:
    """Something a `Tensor`-tier kernel can be handed: an owned `Tensor` or a
    borrowed `View`.

    Conformers name their element type and layout as associated members, so
    a generic routine spells `T.dtype`, `T.LayoutType.static_shape[0]` and
    `T.rank` where it used to spell explicit parameters.
    """

    comptime dtype: DType
    """The element type."""

    comptime LayoutType: TensorLayout
    """The layout, which carries the shape one extent at a time, each either
    compile-time or run-time."""

    comptime Engine = DefaultEngine[element_width=1]
    """The `TileTensor` storage engine the view is built on: one constant
    for every conformer rather than an associated member each may choose,
    so `view()`'s element width is `1` in a generic body instead of the
    symbolic `T.Engine.element_size` a body could not store through. MAX's
    `DenseTensor` leaves it open; nothing in numax needs that yet."""

    comptime rank: Int = Self.LayoutType.rank
    """The number of axes."""

    comptime num_elements: Int = Self.LayoutType.static_product
    """The compile-time element count; meaningful only where
    `LayoutType.all_dims_known`. `size()` is the run-time count."""

    def view(
        ref self,
    ) -> TileTensor[Self.dtype, Self.LayoutType, origin_of(self)]:
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

    def on_host(self) -> Bool:
        """Whether the storage can be read through a plain host pointer:
        true on a CPU context, false on a discrete GPU. What a `gpu`-targeted
        launch checks its operands against."""
        ...

    def to_host[dtype: DType = Self.dtype](self) raises -> List[Scalar[dtype]]:
        """A host copy of every element, in row-major order of the logical
        shape. The bulk read path on either device.

        `dtype` is `Self.dtype` unless named. A routine over two
        conformers `A` and `B` under `where A.dtype == B.dtype` still sees
        `b`'s elements typed `Scalar[B.dtype]`, since the checker does not
        rewrite one into the other from the clause; `b.to_host[A.dtype]()`
        is how it reads them at the type the body computes in. The cast is
        the identity at equal dtypes.
        """
        ...

    def copy_from_host(mut self, values: List[Scalar[Self.dtype]]) raises:
        """Overwrite every element from a host list, row-major over the
        logical shape. The bulk write path on either device."""
        ...

    def view_as[
        dtype: DType
    ](ref self) -> TileTensor[dtype, Self.LayoutType, origin_of(self)]:
        """`view()` with its lanes typed `dtype`, a same-width bitcast.

        For a routine over two conformers `A` and `B` under `where A.dtype
        == B.dtype`: the checker types `b.view()`'s lanes `Scalar[B.dtype]`
        and will not rewrite that into `Scalar[A.dtype]` from the clause,
        so the body reads `b.view_as[A.dtype]()` instead. The bitcast is the
        identity at equal dtypes, which the clause guarantees.
        """
        var v = self.view()
        return TileTensor[dtype, Self.LayoutType, origin_of(self)](
            ptr=v.ptr.unsafe_bitcast[Scalar[dtype]](), layout=v.layout
        )

    def size(self) -> Int:
        """The run-time element count, read from the layout."""
        return self.view().layout.size()

    def dim[i: Int](self) -> Int:
        """The extent of axis `i`."""
        return Int(self.view().layout.shape[i]().value())

    def dim_at(self, axis: Int) -> Int:
        """The extent of `axis`, chosen at run time; `0` outside the rank."""
        var extent = 0
        comptime for i in range(Self.rank):
            if i == axis:
                extent = self.dim[i]()
        return extent

    def stride_at(self, axis: Int) -> Int:
        """The stride of `axis` in elements, chosen at run time; `0` outside
        the rank."""
        var stride = 0
        comptime for i in range(Self.rank):
            if i == axis:
                stride = Int(self.view().layout.stride[i]().value())
        return stride


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


comptime ViewOver[
    dtype: DType, LayoutType: TensorLayout, src: MutOrigin
] = View[dtype, LayoutType, ImmOrigin(src)]
"""The `View` type built over a mutable tile at origin `src`, for a
signature that returns one: `ViewOver[T.dtype, L, origin_of(a)]`."""


struct View[
    dtype_: DType,
    LayoutType_: TensorLayout,
    origin: ImmOrigin,
](ImplicitlyCopyable, TensorLike, Writable):
    """A borrowed tensor: a `TileTensor` plus the device it lives on.

    The `TensorLike` conformer for storage `numax` does not own -- a
    sub-block of a `Tensor`, a `List`'s span, a tile another library handed
    over. It is what lets `cholesky(View(a.view().tile[4, 4](0, 0),
    a.context()))` factor one quadrant of `a` in place, with no copy and no
    second `cholesky`.

    Copying a `View` copies a pointer and a layout, so it is
    `ImplicitlyCopyable` like the tile it wraps. It owns nothing: dropping
    it frees nothing, and `origin` keeps the storage it was built over
    alive for as long as the `View` is used.
    """

    comptime dtype = Self.dtype_
    comptime LayoutType = Self.LayoutType_
    comptime rank = Self.LayoutType.rank
    comptime TileType = TileTensor[Self.dtype, Self.LayoutType, Self.origin]
    comptime Over[src: MutOrigin] = ViewOver[Self.dtype_, Self.LayoutType_, src]
    """The `View` type built over a mutable tile at origin `src`."""

    var tile: Self.TileType
    """The borrowed storage."""
    var ctx: DeviceContext
    """The device `tile` lives on."""
    var host_addressable: Bool
    """`ctx.api() == "cpu"`, recorded once, as `Tensor` records it."""

    def __init__[
        src: MutOrigin
    ](
        out self: Self.Over[src],
        tile: TileTensor[Self.dtype, Self.LayoutType, src],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Borrow `tile`, on `ctx` or the host when none is given.

        The tile must be mutable: `a.view()` on a `mut`-bound or `var`
        tensor is one, `a.view().tile[...](...)` is one, a tile over an
        immutable borrow is not and is refused where it is written. The
        `View` records the immutable form of its origin; the module
        docstring says why.
        """
        self.tile = tile.as_immut()
        self.ctx = ctx.value() if ctx else DeviceContext(api="cpu")
        self.host_addressable = self.ctx.api() == "cpu"

    def view(
        ref self,
    ) -> TileTensor[Self.dtype, Self.LayoutType, origin_of(self)]:
        """The wrapped tile, at the mutability of this borrow of `self`.

        An immutable `View` yields a read-only tile; a `var` or `mut` one
        yields a writable tile over storage that was mutable when the
        `View` was built. The lifetime it names is this borrow's, which the
        tracked `origin` keeps inside the owner's.
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

    def on_host(self) -> Bool:
        """Whether the tile can be read through a plain host pointer."""
        return self.host_addressable

    def to_host[dtype: DType = Self.dtype](self) raises -> List[Scalar[dtype]]:
        """A host copy of every element, row-major over the logical shape.

        On the host the elements are read through the layout, so a strided
        block comes back in its own row-major order rather than the
        parent's. Over device memory the tile must be contiguous: one
        `enqueue_copy` of `size()` elements from its pointer, which is what
        a `DeviceBuffer` has and a bare tile does not; a strided device
        block raises rather than copying the wrong elements.
        """
        var n = self.size()
        var out = List[Scalar[dtype]](capacity=n)
        if self.host_addressable:
            for i in range(n):
                out.append(
                    self.tile.ptr[unsafe_offset=self._offset(i)].cast[dtype]()
                )
            return out^
        if not is_row_major[Self]:
            raise Error(
                "View.to_host: a strided view over device memory cannot be"
                " copied as one block"
            )
        var host = self.ctx.enqueue_create_host_buffer[Self.dtype](n)
        self.ctx.enqueue_copy(host, self.tile.ptr.as_imm())
        self.ctx.synchronize()
        for i in range(n):
            out.append(host[i].cast[dtype]())
        return out^

    def copy_from_host(mut self, values: List[Scalar[Self.dtype]]) raises:
        """Overwrite every element from `values`, row-major over the logical
        shape; the write counterpart of `to_host`, with the same contiguity
        rule over device memory."""
        var n = self.size()
        var ptr = self.tile.ptr.unsafe_mut_cast[True]()
        if self.host_addressable:
            for i in range(n):
                ptr[unsafe_offset=self._offset(i)] = values[i]
            return
        if not is_row_major[Self]:
            raise Error(
                "View.copy_from_host: a strided view over device memory"
                " cannot be written as one block"
            )
        var host = self.ctx.enqueue_create_host_buffer[Self.dtype](n)
        for i in range(n):
            host[i] = values[i]
        self.ctx.enqueue_copy(ptr, host)
        self.ctx.synchronize()

    def _offset(self, i: Int) -> Int:
        """The memory offset of the `i`-th element in row-major order of
        the logical shape, through the layout's strides. Not
        `layout.idx2crd`, which inverts the *memory* index and so is the
        wrong decomposition for a strided block."""
        var rem = i
        var off = 0
        comptime for k in range(Self.rank):
            comptime d = Self.rank - 1 - k
            var extent = self.dim[d]()
            off += (rem % extent) * Int(self.tile.layout.stride[d]().value())
            rem //= extent
        return off

    def write_to(self, mut writer: Some[Writer]):
        """`print(v)`: the tile's own printer."""
        writer.write(self.tile)
