"""`TensorLike`: one bound over an owned `Tensor` and a borrowed `TensorView`.

**This module is tier-agnostic.** It defines no kernel; it names what a
kernel may be handed. `Tensor` (`numax.core.tensor`) owns its storage and
`TensorView` borrows a `TileTensor` someone else owns, and every public routine in
the `Tensor` tier takes either through this one trait, so a factorization
runs on a whole tensor or on a sub-block of one without a copy:

```mojo
var a = zeros[DType.float64, 8, 8]()
var l = cholesky(a)                                   # owned
var block = TensorView(a.tile().tile[4, 4](0, 0), a.context())
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

**What the trait promises.** `tile(ref self)` hands back a `TileTensor`
whose origin is the borrow of `self`: bind the tensor `mut` and the tile is
writable, bind it immutably and the tile is read-only, and either way the
tile keeps `self` alive. MAX's implicit origin cast then turns that tile into
the `MutAnyOrigin` spelling `numax.core.functional` and every MAX kernel take,
at the call site and with parameter inference intact. `context()` names the
device the storage lives on, so a kernel can allocate its result beside its
input without being told twice.

**`TensorView` tracks its owner immutably and writes through the binding.** It
accepts any mutable `TileTensor` and stores it at the *immutable* form of
that tile's origin. Tracked, so the owner stays alive for as long as the
`TensorView` is used (an untracked spelling was tried and dangled the moment the
owner's last mention passed). Immutable, so two `TensorView`s over one tensor, or
the same one passed twice to `add`, are immutable aliases the exclusivity
checker accepts, where a tracked mutable origin made `add(v, v)` an error a
`Tensor` never raises. Mutability then comes from how the `TensorView` itself is
bound: `tile(ref self)` on a `var` or `mut` `TensorView` re-adds it, on an
immutable binding it does not, the same rule `Tensor` follows. Wrapping an
immutable tile is refused at compile time rather than quietly producing a
writable view over read-only storage.

A `TensorView` built without a context is a host view, matching every factory in
`numax.core.tensor`: the `DeviceContext` comes last, optional, and its
absence means the CPU. Pass the owning tensor's `context()` for a view
over device memory.
"""

from layout import TileTensor
from layout.tile_layout import TensorLayout
from layout.tile_tensor import DefaultEngine
from max.gpu.host import DeviceContext
from std.sys import has_accelerator

from .tensor import Dynamic, Tensor
from ._drive import _BroadcastRank, _dtype_mismatch, _shapes_broadcast


trait TensorLike:
    """Something a `Tensor`-tier kernel can be handed: an owned `Tensor` or a
    borrowed `TensorView`.

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
    so `tile()`'s element width is `1` in a generic body instead of the
    symbolic `T.Engine.element_size` a body could not store through. MAX's
    `DenseTensor` leaves it open; nothing in numax needs that yet."""

    comptime rank: Int = Self.LayoutType.rank
    """The number of axes."""

    comptime num_elements: Int = Self.LayoutType.static_product
    """The compile-time element count; meaningful only where
    `LayoutType.all_dims_known`. `size()` is the run-time count."""

    def tile(
        ref self,
    ) -> TileTensor[Self.dtype, Self.LayoutType, origin_of(self)]:
        """A `TileTensor` over the storage, with the mutability of this
        borrow of `self` and its lifetime.

        The type `numax.core.functional`'s walks and every MAX kernel accept,
        once MAX's implicit origin cast has erased the origin at the call
        site. A view outliving the borrow is a compile error rather than a
        dangling pointer.

        Returns:
            A `TileTensor` over the storage at `origin_of(self)`.
        """
        ...

    def context(self) raises -> DeviceContext:
        """The device the storage lives on.

        Returns:
            The `DeviceContext` the storage was allocated on.

        Raises:
            Raises if the conformer cannot produce its context.
        """
        ...

    def on_host(self) -> Bool:
        """Whether the storage can be read through a plain host pointer:
        true on a CPU context, false on a discrete GPU. What a `gpu`-targeted
        launch checks its operands against.

        Returns:
            `True` on a CPU context, `False` on a GPU one.
        """
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

        Parameters:
            dtype: The element type of the returned list; defaults to
                `Self.dtype`, and other values cast each element.

        Returns:
            A `List` of `size()` elements, row-major over the logical shape.

        Raises:
            Raises if the device-to-host copy fails.
        """
        ...

    def copy_from_host(mut self, values: List[Scalar[Self.dtype]]) raises:
        """Overwrite every element from a host list, row-major over the
        logical shape. The bulk write path on either device.

        Args:
            values: The new elements, row-major over the logical shape.

        Raises:
            Raises if the host-to-device copy fails or `values` has the wrong
            length.
        """
        ...

    def tile_as[
        dtype: DType
    ](ref self) -> TileTensor[dtype, Self.LayoutType, origin_of(self)]:
        """`tile()` with its lanes typed `dtype`, a same-width bitcast.

        For a routine over two conformers `A` and `B` under `where A.dtype
        == B.dtype`: the checker types `b.tile()`'s lanes `Scalar[B.dtype]`
        and will not rewrite that into `Scalar[A.dtype]` from the clause,
        so the body reads `b.tile_as[A.dtype]()` instead. The bitcast is the
        identity at equal dtypes, which the clause guarantees.

        Parameters:
            dtype: The lane type to reinterpret the storage as, which must have
                the width of `Self.dtype`.

        Returns:
            `tile()` bitcast to `dtype` lanes, same layout and origin.
        """
        var v = self.tile()
        return TileTensor[dtype, Self.LayoutType, origin_of(self)](
            ptr=v.ptr.unsafe_bitcast[Scalar[dtype]](), layout=v.layout
        )

    def size(self) -> Int:
        """The run-time element count, read from the layout.

        Returns:
            The product of the run-time extents.
        """
        return self.tile().layout.size()

    def dim[i: Int](self) -> Int:
        """The extent of axis `i`.

        Parameters:
            i: The axis, in `[0, rank)`.

        Returns:
            The run-time extent of axis `i`.
        """
        return Int(self.tile().layout.shape[i]().value())

    def dim_at(self, axis: Int) -> Int:
        """The extent of `axis`, chosen at run time; `0` outside the rank.

        Args:
            axis: The axis, read at run time.

        Returns:
            The extent of `axis`, or `0` when `axis` is outside `[0, rank)`.
        """
        var extent = 0
        comptime for i in range(Self.rank):
            if i == axis:
                extent = self.dim[i]()
        return extent

    def stride_at(self, axis: Int) -> Int:
        """The stride of `axis` in elements, chosen at run time; `0` outside
        the rank.

        Args:
            axis: The axis, read at run time.

        Returns:
            The stride of `axis` in elements, or `0` when `axis` is outside `[0,
            rank)`.
        """
        var stride = 0
        comptime for i in range(Self.rank):
            if i == axis:
                stride = Int(self.tile().layout.stride[i]().value())
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
`TensorView` over `a.tile().tile[2, 2](0, 0)` of a 4x4 has stride 4 on its first
axis, and a walk that treats it as eight contiguous elements reads the
wrong ones. `numax.core.functional`'s walks already require this; the trait
makes it one spelling for the public tier too. For a run-time layout the
check is on the stride *types*, which every layout `numax` builds
satisfies, and the flattening helpers verify the values at run time.
"""


comptime TensorViewOver[
    dtype: DType, LayoutType: TensorLayout, src: MutOrigin
] = TensorView[dtype, LayoutType, ImmOrigin(src)]
"""The `TensorView` type built over a mutable tile at origin `src`, for a
signature that returns one: `TensorViewOver[T.dtype, L, origin_of(a)]`."""


struct TensorView[
    dtype_: DType,
    LayoutType_: TensorLayout,
    origin: ImmOrigin,
](ImplicitlyCopyable, TensorLike, Writable):
    """A borrowed tensor: a `TileTensor` plus the device it lives on.

    The `TensorLike` conformer for storage `numax` does not own -- a
    sub-block of a `Tensor`, a `List`'s span, a tile another library handed
    over. It is what lets `cholesky(TensorView(a.tile().tile[4, 4](0, 0),
    a.context()))` factor one quadrant of `a` in place, with no copy and no
    second `cholesky`.

    Copying a `TensorView` copies a pointer and a layout, so it is
    `ImplicitlyCopyable` like the tile it wraps. It owns nothing: dropping
    it frees nothing, and `origin` keeps the storage it was built over
    alive for as long as the `TensorView` is used.
    """

    comptime dtype = Self.dtype_
    comptime LayoutType = Self.LayoutType_
    comptime rank = Self.LayoutType.rank
    comptime TileType = TileTensor[Self.dtype, Self.LayoutType, Self.origin]
    comptime Over[src: MutOrigin] = TensorViewOver[
        Self.dtype_, Self.LayoutType_, src
    ]
    """The `TensorView` type built over a mutable tile at origin `src`."""

    var _tile: Self.TileType
    """The borrowed storage."""
    var _ctx: DeviceContext
    """The device the tile lives on."""
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

        The tile must be mutable: `a.tile()` on a `mut`-bound or `var`
        tensor is one, `a.tile().tile[...](...)` is one, a tile over an
        immutable borrow is not and is refused where it is written. The
        `TensorView` records the immutable form of its origin; the module
        docstring says why.

        Parameters:
            src: The mutable origin of `tile`, inferred; the view records its
                immutable form.

        Args:
            tile: The mutable `TileTensor` to borrow.
            ctx: The device the tile lives on; `None` (the default) means a host
                view on a fresh CPU context.

        Raises:
            Raises if building the default CPU `DeviceContext` fails.
        """
        self._tile = tile.as_immut()
        self._ctx = ctx.value() if ctx else DeviceContext(api="cpu")
        self.host_addressable = self._ctx.api() == "cpu"

    def tile(
        ref self,
    ) -> TileTensor[Self.dtype, Self.LayoutType, origin_of(self)]:
        """The wrapped tile, at the mutability of this borrow of `self`.

        An immutable `TensorView` yields a read-only tile; a `var` or `mut` one
        yields a writable tile over storage that was mutable when the
        `TensorView` was built. The lifetime it names is this borrow's, which the
        tracked `origin` keeps inside the owner's.

        Returns:
            The wrapped `TileTensor` at `origin_of(self)`.
        """
        return TileTensor[Self.dtype, Self.LayoutType, origin_of(self)](
            ptr=self._tile.ptr.unsafe_mut_cast[
                origin_of(self).mut
            ]().unsafe_origin_cast[origin_of(self)](),
            layout=self._tile.layout,
        )

    def context(self) raises -> DeviceContext:
        """The device the storage lives on.

        Returns:
            The `DeviceContext` the view was built with.

        Raises:
            Declared by `TensorLike`; this conformer never raises.
        """
        return self._ctx

    def on_host(self) -> Bool:
        """Whether the tile can be read through a plain host pointer.

        Returns:
            `True` when the view's context is a CPU one.
        """
        return self.host_addressable

    def to_host[dtype: DType = Self.dtype](self) raises -> List[Scalar[dtype]]:
        """A host copy of every element, row-major over the logical shape.

        On the host the elements are read through the layout, so a strided
        block comes back in its own row-major order rather than the
        parent's. Over device memory the tile must be contiguous: one
        `enqueue_copy` of `size()` elements from its pointer, which is what
        a `DeviceBuffer` has and a bare tile does not; a strided device
        block raises rather than copying the wrong elements.

        Parameters:
            dtype: The element type of the returned list; defaults to
                `Self.dtype`, and other values cast each element.

        Returns:
            A `List` of `size()` elements, row-major over the logical shape.

        Raises:
            Raises if the view is strided over device memory, or if the
            device-to-host copy fails.
        """
        var n = self.size()
        var out = List[Scalar[dtype]](capacity=n)
        if self.host_addressable:
            for i in range(n):
                out.append(
                    self._tile.ptr[unsafe_offset=self._offset(i)].cast[dtype]()
                )
            return out^
        if not is_row_major[Self]:
            raise Error(
                "TensorView.to_host: a strided view over device memory cannot"
                " be copied as one block"
            )
        var host = self._ctx.enqueue_create_host_buffer[Self.dtype](n)
        self._ctx.enqueue_copy(host, self._tile.ptr.as_imm())
        self._ctx.synchronize()
        for i in range(n):
            out.append(host[i].cast[dtype]())
        return out^

    def copy_from_host(mut self, values: List[Scalar[Self.dtype]]) raises:
        """Overwrite every element from `values`, row-major over the logical
        shape; the write counterpart of `to_host`, with the same contiguity
        rule over device memory. A list of the wrong length raises.

        Args:
            values: The new elements, `size()` of them, row-major over the
                logical shape.

        Raises:
            Raises if `len(values)` is not `size()`, if the view is strided over
            device memory, or if the host-to-device copy fails.
        """
        var n = self.size()
        if len(values) != n:
            raise Error(
                "TensorView.copy_from_host: ",
                len(values),
                " values for ",
                n,
                " elements",
            )
        var ptr = self._tile.ptr.unsafe_mut_cast[True]()
        if self.host_addressable:
            for i in range(n):
                ptr[unsafe_offset=self._offset(i)] = values[i]
            return
        if not is_row_major[Self]:
            raise Error(
                "TensorView.copy_from_host: a strided view over device memory"
                " cannot be written as one block"
            )
        var host = self._ctx.enqueue_create_host_buffer[Self.dtype](n)
        for i in range(n):
            host[i] = values[i]
        self._ctx.enqueue_copy(ptr, host)
        self._ctx.synchronize()

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
            off += (rem % extent) * Int(self._tile.layout.stride[d]().value())
            rem //= extent
        return off

    # Mixed operands: any other `TensorLike` of the same dtype -- a view
    # of other storage, a `Static` beside a `Dynamic`, a broadcastable
    # shape -- through the broadcasting free functions, into a
    # run-time-shaped result. A same-typed operand takes the overloads
    # above, which keep the static shape.

    def __add__[
        B: TensorLike
    ](self, other: B) raises -> Dynamic[
        Self.dtype, _BroadcastRank[Self.LayoutType, B.LayoutType]
    ] where (is_row_major[Self] and is_row_major[B]):
        """`a + b` against any `TensorLike` `b` of the same dtype, at two
        broadcastable shapes. Forwards to `numax.core.ops.add`'s
        broadcasting overload and follows `a` to its device.

        Parameters:
            B: The type of `other`, any `TensorLike` of the same dtype,
                row-major.

        Args:
            other: The right operand, at a shape that broadcasts against this
                view's.

        Returns:
            A new run-time-shaped `Dynamic` tensor at the broadcast shape
            holding the elementwise sum, on this view's device.

        Raises:
            Raises if the shapes do not broadcast, if the device launch fails,
            or under the `"raise"` fallback policy when the call falls back to
            the host.
        """
        comptime assert B.dtype == Self.dtype, _dtype_mismatch["+"]
        _shapes_broadcast["+", Self.LayoutType, B.LayoutType]()
        from .ops import add as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __sub__[
        B: TensorLike
    ](self, other: B) raises -> Dynamic[
        Self.dtype, _BroadcastRank[Self.LayoutType, B.LayoutType]
    ] where (is_row_major[Self] and is_row_major[B]):
        """`a - b` against any `TensorLike` `b` of the same dtype, at two
        broadcastable shapes. Forwards to `numax.core.ops.subtract`'s
        broadcasting overload and follows `a` to its device.

        Parameters:
            B: The type of `other`, any `TensorLike` of the same dtype,
                row-major.

        Args:
            other: The right operand, at a shape that broadcasts against this
                view's.

        Returns:
            A new run-time-shaped `Dynamic` tensor at the broadcast shape
            holding the elementwise difference, on this view's device.

        Raises:
            Raises if the shapes do not broadcast, if the device launch fails,
            or under the `"raise"` fallback policy when the call falls back to
            the host.
        """
        comptime assert B.dtype == Self.dtype, _dtype_mismatch["-"]
        _shapes_broadcast["-", Self.LayoutType, B.LayoutType]()
        from .ops import subtract as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __mul__[
        B: TensorLike
    ](self, other: B) raises -> Dynamic[
        Self.dtype, _BroadcastRank[Self.LayoutType, B.LayoutType]
    ] where (is_row_major[Self] and is_row_major[B]):
        """`a * b` against any `TensorLike` `b` of the same dtype, at two
        broadcastable shapes. Forwards to `numax.core.ops.multiply`'s
        broadcasting overload and follows `a` to its device.

        Parameters:
            B: The type of `other`, any `TensorLike` of the same dtype,
                row-major.

        Args:
            other: The right operand, at a shape that broadcasts against this
                view's.

        Returns:
            A new run-time-shaped `Dynamic` tensor at the broadcast shape
            holding the elementwise product, on this view's device.

        Raises:
            Raises if the shapes do not broadcast, if the device launch fails,
            or under the `"raise"` fallback policy when the call falls back to
            the host.
        """
        comptime assert B.dtype == Self.dtype, _dtype_mismatch["*"]
        _shapes_broadcast["*", Self.LayoutType, B.LayoutType]()
        from .ops import multiply as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __truediv__[
        B: TensorLike
    ](self, other: B) raises -> Dynamic[
        Self.dtype, _BroadcastRank[Self.LayoutType, B.LayoutType]
    ] where (is_row_major[Self] and is_row_major[B]):
        """`a / b` against any `TensorLike` `b` of the same dtype, at two
        broadcastable shapes. Forwards to `numax.core.ops.divide`'s
        broadcasting overload and follows `a` to its device.

        Parameters:
            B: The type of `other`, any `TensorLike` of the same dtype,
                row-major.

        Args:
            other: The right operand, at a shape that broadcasts against this
                view's.

        Returns:
            A new run-time-shaped `Dynamic` tensor at the broadcast shape
            holding the elementwise quotient, on this view's device.

        Raises:
            Raises if the shapes do not broadcast, if the device launch fails,
            or under the `"raise"` fallback policy when the call falls back to
            the host.
        """
        comptime assert B.dtype == Self.dtype, _dtype_mismatch["/"]
        _shapes_broadcast["/", Self.LayoutType, B.LayoutType]()
        from .ops import divide as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __lt__[
        B: TensorLike
    ](self, other: B) raises -> Dynamic[
        DType.bool, _BroadcastRank[Self.LayoutType, B.LayoutType]
    ] where (is_row_major[Self] and is_row_major[B]):
        """`a < b` against any `TensorLike` `b` of the same dtype, at two
        broadcastable shapes, as a `bool` tensor.

        Parameters:
            B: The type of `other`, any `TensorLike` of the same dtype,
                row-major.

        Args:
            other: The right operand, at a shape that broadcasts against this
                view's.

        Returns:
            A new run-time-shaped `bool` tensor at the broadcast shape, on this
            view's device.

        Raises:
            Raises if the shapes do not broadcast, if the device launch fails,
            or under the `"raise"` fallback policy when the call falls back to
            the host.
        """
        comptime assert B.dtype == Self.dtype, _dtype_mismatch["<"]
        _shapes_broadcast["<", Self.LayoutType, B.LayoutType]()
        from .logic import less as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __le__[
        B: TensorLike
    ](self, other: B) raises -> Dynamic[
        DType.bool, _BroadcastRank[Self.LayoutType, B.LayoutType]
    ] where (is_row_major[Self] and is_row_major[B]):
        """`a <= b` against any `TensorLike` `b` of the same dtype, at two
        broadcastable shapes, as a `bool` tensor.

        Parameters:
            B: The type of `other`, any `TensorLike` of the same dtype,
                row-major.

        Args:
            other: The right operand, at a shape that broadcasts against this
                view's.

        Returns:
            A new run-time-shaped `bool` tensor at the broadcast shape, on this
            view's device.

        Raises:
            Raises if the shapes do not broadcast, if the device launch fails,
            or under the `"raise"` fallback policy when the call falls back to
            the host.
        """
        comptime assert B.dtype == Self.dtype, _dtype_mismatch["<="]
        _shapes_broadcast["<=", Self.LayoutType, B.LayoutType]()
        from .logic import less_equal as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __gt__[
        B: TensorLike
    ](self, other: B) raises -> Dynamic[
        DType.bool, _BroadcastRank[Self.LayoutType, B.LayoutType]
    ] where (is_row_major[Self] and is_row_major[B]):
        """`a > b` against any `TensorLike` `b` of the same dtype, at two
        broadcastable shapes, as a `bool` tensor.

        Parameters:
            B: The type of `other`, any `TensorLike` of the same dtype,
                row-major.

        Args:
            other: The right operand, at a shape that broadcasts against this
                view's.

        Returns:
            A new run-time-shaped `bool` tensor at the broadcast shape, on this
            view's device.

        Raises:
            Raises if the shapes do not broadcast, if the device launch fails,
            or under the `"raise"` fallback policy when the call falls back to
            the host.
        """
        comptime assert B.dtype == Self.dtype, _dtype_mismatch[">"]
        _shapes_broadcast[">", Self.LayoutType, B.LayoutType]()
        from .logic import greater as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __ge__[
        B: TensorLike
    ](self, other: B) raises -> Dynamic[
        DType.bool, _BroadcastRank[Self.LayoutType, B.LayoutType]
    ] where (is_row_major[Self] and is_row_major[B]):
        """`a >= b` against any `TensorLike` `b` of the same dtype, at two
        broadcastable shapes, as a `bool` tensor.

        Parameters:
            B: The type of `other`, any `TensorLike` of the same dtype,
                row-major.

        Args:
            other: The right operand, at a shape that broadcasts against this
                view's.

        Returns:
            A new run-time-shaped `bool` tensor at the broadcast shape, on this
            view's device.

        Raises:
            Raises if the shapes do not broadcast, if the device launch fails,
            or under the `"raise"` fallback policy when the call falls back to
            the host.
        """
        comptime assert B.dtype == Self.dtype, _dtype_mismatch[">="]
        _shapes_broadcast[">=", Self.LayoutType, B.LayoutType]()
        from .logic import greater_equal as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __eq__[
        B: TensorLike
    ](self, other: B) raises -> Dynamic[
        DType.bool, _BroadcastRank[Self.LayoutType, B.LayoutType]
    ] where (is_row_major[Self] and is_row_major[B]):
        """`a == b` against any `TensorLike` `b` of the same dtype, at two
        broadcastable shapes, as a `bool` tensor.

        Parameters:
            B: The type of `other`, any `TensorLike` of the same dtype,
                row-major.

        Args:
            other: The right operand, at a shape that broadcasts against this
                view's.

        Returns:
            A new run-time-shaped `bool` tensor at the broadcast shape, on this
            view's device.

        Raises:
            Raises if the shapes do not broadcast, if the device launch fails,
            or under the `"raise"` fallback policy when the call falls back to
            the host.
        """
        comptime assert B.dtype == Self.dtype, _dtype_mismatch["=="]
        _shapes_broadcast["==", Self.LayoutType, B.LayoutType]()
        from .logic import equal as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __ne__[
        B: TensorLike
    ](self, other: B) raises -> Dynamic[
        DType.bool, _BroadcastRank[Self.LayoutType, B.LayoutType]
    ] where (is_row_major[Self] and is_row_major[B]):
        """`a != b` against any `TensorLike` `b` of the same dtype, at two
        broadcastable shapes, as a `bool` tensor.

        Parameters:
            B: The type of `other`, any `TensorLike` of the same dtype,
                row-major.

        Args:
            other: The right operand, at a shape that broadcasts against this
                view's.

        Returns:
            A new run-time-shaped `bool` tensor at the broadcast shape, on this
            view's device.

        Raises:
            Raises if the shapes do not broadcast, if the device launch fails,
            or under the `"raise"` fallback policy when the call falls back to
            the host.
        """
        comptime assert B.dtype == Self.dtype, _dtype_mismatch["!="]
        _shapes_broadcast["!=", Self.LayoutType, B.LayoutType]()
        from .logic import not_equal as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def write_to(self, mut writer: Some[Writer]):
        """`print(v)`, in the same format as `print(a)` for a `Tensor`:
        both go through `numax.core.tensor._format_tensor`.

        Args:
            writer: The destination the formatted tensor is written to.
        """
        from .tensor import _format_tensor

        try:
            writer.write(_format_tensor(self, 4, 1000, 3))
        except e:
            writer.write("<unreadable: ", String(e), ">")

    # The operators, as `Tensor` has them: each returns a new owned tensor
    # (a view owns nothing to write into), forwards to `numax.core.ops` or
    # `numax.core.logic`, and follows the view to its device. The imports
    # are local because `ops` and `logic` import this module.

    def __add__(
        self, other: Self
    ) raises -> Tensor[Self.dtype, Self.LayoutType] where is_row_major[Self]:
        """`a + b` against a view of the same type, into a new tensor. Forwards to
        `numax.core.ops.add` and follows the view, as `Tensor.__add__`
        describes.

        Args:
            other: A view of the same type, and so the same shape.

        Returns:
            A new `Tensor` of this view's shape and dtype holding the
            elementwise sum, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .ops import add as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __add__(
        self, other: Scalar[Self.dtype]
    ) raises -> Tensor[Self.dtype, Self.LayoutType] where is_row_major[Self]:
        """`a + b` against a scalar, into a new tensor. Forwards to
        `numax.core.ops.add` and follows the view, as `Tensor.__add__`
        describes.

        Args:
            other: The scalar applied to every element.

        Returns:
            A new `Tensor` of this view's shape and dtype holding the
            elementwise sum, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .ops import add as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __sub__(
        self, other: Self
    ) raises -> Tensor[Self.dtype, Self.LayoutType] where is_row_major[Self]:
        """`a - b` against a view of the same type, into a new tensor. Forwards to
        `numax.core.ops.subtract` and follows the view, as `Tensor.__add__`
        describes.

        Args:
            other: A view of the same type, and so the same shape.

        Returns:
            A new `Tensor` of this view's shape and dtype holding the
            elementwise difference, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .ops import subtract as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __sub__(
        self, other: Scalar[Self.dtype]
    ) raises -> Tensor[Self.dtype, Self.LayoutType] where is_row_major[Self]:
        """`a - b` against a scalar, into a new tensor. Forwards to
        `numax.core.ops.subtract` and follows the view, as `Tensor.__add__`
        describes.

        Args:
            other: The scalar applied to every element.

        Returns:
            A new `Tensor` of this view's shape and dtype holding the
            elementwise difference, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .ops import subtract as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __mul__(
        self, other: Self
    ) raises -> Tensor[Self.dtype, Self.LayoutType] where is_row_major[Self]:
        """`a * b` against a view of the same type, into a new tensor. Forwards to
        `numax.core.ops.multiply` and follows the view, as `Tensor.__add__`
        describes.

        Args:
            other: A view of the same type, and so the same shape.

        Returns:
            A new `Tensor` of this view's shape and dtype holding the
            elementwise product, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .ops import multiply as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __mul__(
        self, other: Scalar[Self.dtype]
    ) raises -> Tensor[Self.dtype, Self.LayoutType] where is_row_major[Self]:
        """`a * b` against a scalar, into a new tensor. Forwards to
        `numax.core.ops.multiply` and follows the view, as `Tensor.__add__`
        describes.

        Args:
            other: The scalar applied to every element.

        Returns:
            A new `Tensor` of this view's shape and dtype holding the
            elementwise product, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .ops import multiply as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __truediv__(
        self, other: Self
    ) raises -> Tensor[Self.dtype, Self.LayoutType] where is_row_major[Self]:
        """`a / b` against a view of the same type, into a new tensor. Forwards to
        `numax.core.ops.divide` and follows the view, as `Tensor.__add__`
        describes.

        Args:
            other: A view of the same type, and so the same shape.

        Returns:
            A new `Tensor` of this view's shape and dtype holding the
            elementwise quotient, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .ops import divide as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __truediv__(
        self, other: Scalar[Self.dtype]
    ) raises -> Tensor[Self.dtype, Self.LayoutType] where is_row_major[Self]:
        """`a / b` against a scalar, into a new tensor. Forwards to
        `numax.core.ops.divide` and follows the view, as `Tensor.__add__`
        describes.

        Args:
            other: The scalar applied to every element.

        Returns:
            A new `Tensor` of this view's shape and dtype holding the
            elementwise quotient, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .ops import divide as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __radd__(
        self, other: Scalar[Self.dtype]
    ) raises -> Tensor[Self.dtype, Self.LayoutType] where is_row_major[Self]:
        """`b + a` with a scalar `b` on the left.

        Args:
            other: The scalar on the left of the operator.

        Returns:
            A new `Tensor` of this view's shape and dtype holding the
            elementwise sum `other + a`, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        return self.__add__(other)

    def __rmul__(
        self, other: Scalar[Self.dtype]
    ) raises -> Tensor[Self.dtype, Self.LayoutType] where is_row_major[Self]:
        """`b * a` with a scalar `b` on the left.

        Args:
            other: The scalar on the left of the operator.

        Returns:
            A new `Tensor` of this view's shape and dtype holding the
            elementwise product `other * a`, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        return self.__mul__(other)

    def __rsub__(
        self, other: Scalar[Self.dtype]
    ) raises -> Tensor[Self.dtype, Self.LayoutType] where is_row_major[Self]:
        """`b - a` with a scalar `b` on the left, one launch.

        Args:
            other: The scalar on the left of the operator.

        Returns:
            A new `Tensor` of this view's shape and dtype holding the
            elementwise difference `other - a`, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .ops import _reflected

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _reflected[kind="sub", gpu=True](self, other)
        return _reflected[kind="sub"](self, other)

    def __rtruediv__(
        self, other: Scalar[Self.dtype]
    ) raises -> Tensor[Self.dtype, Self.LayoutType] where is_row_major[Self]:
        """`b / a` with a scalar `b` on the left, one launch.

        Args:
            other: The scalar on the left of the operator.

        Returns:
            A new `Tensor` of this view's shape and dtype holding the
            elementwise quotient `other / a`, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .ops import _reflected

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _reflected[kind="div", gpu=True](self, other)
        return _reflected[kind="div"](self, other)

    def __neg__(
        self,
    ) raises -> Tensor[Self.dtype, Self.LayoutType] where is_row_major[Self]:
        """`-a`, into a new tensor.

        Returns:
            A new `Tensor` of this view's shape and dtype holding `-a`, on this
            view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .ops import negative as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self)
        return _op(self)

    def __pow__(
        self, other: Self
    ) raises -> Tensor[Self.dtype, Self.LayoutType] where (
        is_row_major[Self] and Self.dtype.is_floating_point()
    ):
        """`a ** b`, into a new tensor.

        Args:
            other: A view of the same type holding the exponents.

        Returns:
            A new `Tensor` of this view's shape and dtype holding the
            elementwise power `a ** other`, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .ops import power as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __pow__(
        self, other: Scalar[Self.dtype]
    ) raises -> Tensor[Self.dtype, Self.LayoutType] where (
        is_row_major[Self] and Self.dtype.is_floating_point()
    ):
        """`a ** b`, into a new tensor.

        Args:
            other: The exponent applied to every element.

        Returns:
            A new `Tensor` of this view's shape and dtype holding the
            elementwise power `a ** other`, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .ops import power as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __lt__(
        self, other: Self
    ) raises -> Tensor[DType.bool, Self.LayoutType] where is_row_major[Self]:
        """`a < b`, elementwise, as a new `bool` tensor.

        Args:
            other: A view of the same type, and so the same shape.

        Returns:
            A new `bool` `Tensor` of this view's shape, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .logic import less as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __lt__(
        self, other: Scalar[Self.dtype]
    ) raises -> Tensor[DType.bool, Self.LayoutType] where is_row_major[Self]:
        """`a < b`, elementwise, as a new `bool` tensor.

        Args:
            other: The scalar applied to every element.

        Returns:
            A new `bool` `Tensor` of this view's shape, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .logic import less as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __le__(
        self, other: Self
    ) raises -> Tensor[DType.bool, Self.LayoutType] where is_row_major[Self]:
        """`a <= b`, elementwise, as a new `bool` tensor.

        Args:
            other: A view of the same type, and so the same shape.

        Returns:
            A new `bool` `Tensor` of this view's shape, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .logic import less_equal as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __le__(
        self, other: Scalar[Self.dtype]
    ) raises -> Tensor[DType.bool, Self.LayoutType] where is_row_major[Self]:
        """`a <= b`, elementwise, as a new `bool` tensor.

        Args:
            other: The scalar applied to every element.

        Returns:
            A new `bool` `Tensor` of this view's shape, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .logic import less_equal as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __gt__(
        self, other: Self
    ) raises -> Tensor[DType.bool, Self.LayoutType] where is_row_major[Self]:
        """`a > b`, elementwise, as a new `bool` tensor.

        Args:
            other: A view of the same type, and so the same shape.

        Returns:
            A new `bool` `Tensor` of this view's shape, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .logic import greater as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __gt__(
        self, other: Scalar[Self.dtype]
    ) raises -> Tensor[DType.bool, Self.LayoutType] where is_row_major[Self]:
        """`a > b`, elementwise, as a new `bool` tensor.

        Args:
            other: The scalar applied to every element.

        Returns:
            A new `bool` `Tensor` of this view's shape, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .logic import greater as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __ge__(
        self, other: Self
    ) raises -> Tensor[DType.bool, Self.LayoutType] where is_row_major[Self]:
        """`a >= b`, elementwise, as a new `bool` tensor.

        Args:
            other: A view of the same type, and so the same shape.

        Returns:
            A new `bool` `Tensor` of this view's shape, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .logic import greater_equal as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __ge__(
        self, other: Scalar[Self.dtype]
    ) raises -> Tensor[DType.bool, Self.LayoutType] where is_row_major[Self]:
        """`a >= b`, elementwise, as a new `bool` tensor.

        Args:
            other: The scalar applied to every element.

        Returns:
            A new `bool` `Tensor` of this view's shape, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .logic import greater_equal as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __eq__(
        self, other: Self
    ) raises -> Tensor[DType.bool, Self.LayoutType] where is_row_major[Self]:
        """`a == b`, elementwise, as a new `bool` tensor.

        Args:
            other: A view of the same type, and so the same shape.

        Returns:
            A new `bool` `Tensor` of this view's shape, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .logic import equal as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __eq__(
        self, other: Scalar[Self.dtype]
    ) raises -> Tensor[DType.bool, Self.LayoutType] where is_row_major[Self]:
        """`a == b`, elementwise, as a new `bool` tensor.

        Args:
            other: The scalar applied to every element.

        Returns:
            A new `bool` `Tensor` of this view's shape, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .logic import equal as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __ne__(
        self, other: Self
    ) raises -> Tensor[DType.bool, Self.LayoutType] where is_row_major[Self]:
        """`a != b`, elementwise, as a new `bool` tensor.

        Args:
            other: A view of the same type, and so the same shape.

        Returns:
            A new `bool` `Tensor` of this view's shape, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .logic import not_equal as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)

    def __ne__(
        self, other: Scalar[Self.dtype]
    ) raises -> Tensor[DType.bool, Self.LayoutType] where is_row_major[Self]:
        """`a != b`, elementwise, as a new `bool` tensor.

        Args:
            other: The scalar applied to every element.

        Returns:
            A new `bool` `Tensor` of this view's shape, on this view's device.

        Raises:
            Raises if the device launch fails, or under the `"raise"` fallback
            policy when the call falls back to the host.
        """
        from .logic import not_equal as _op

        comptime if has_accelerator() and Self.dtype != DType.float64:
            if not self.host_addressable:
                return _op[gpu=True](self, other)
        return _op(self, other)
