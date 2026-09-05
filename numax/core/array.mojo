"""Gap-only NumPy-named creation and manipulation surface over `TileTensor`.

`docs/parity.md` picks array creation/manipulation as a genuine
`numax` gap: MAX's `layout` package ships `TileTensor` itself (slicing,
tiling, reshaping, coalescing) but no NumPy-named *factory* functions --
`TileTensor.zeros`/`.ones`/`.full`/`.arange` all fail to resolve (verified
directly against `~/workspace/modular/max/kernels/src/layout/tile_tensor.mojo`),
and `max.algorithm.functional` ships only `elementwise`. This module adds
**only** those factory/manipulation names, comptime-shape, as a thin layer
over `TileTensor` -- not a competing array type. Any array-level work in
numax builds as a thin layer over `TileTensor`, because that is what every
MAX kernel already takes.

**One tensor type, CPU and GPU.** `Shaped[dtype, *dims]` is the only owning
tensor in `numax`, and it works on either kind of device because its storage
is a MAX `DeviceBuffer` obtained from a `DeviceContext`: pass
`DeviceContext(api="cpu")` and the buffer is host memory, pass
`DeviceContext()` and it is device memory. Nothing else about the type
changes between the two, and `.view()` hands back the same
`TileTensor[dtype, LayoutType, MutAnyOrigin]` either way -- which is exactly
what `numax.core.tensor.map`/`reduce` and every MAX kernel already take, on both
paths (`map[gpu=False]` walks the host view, `map[gpu=True]` launches on the
device view through `enqueue_function`).

**Why a `Tensor` wrapper, not a bare `TileTensor`.** `TileTensor` is a
*view*: a pointer plus a layout, not the memory itself, in every storage
policy MAX ships (`PointerStorage`, `DevicePointerStorage`,
`StaticOffsetStorage` -- all non-owning). Confirmed directly: a function
that builds a local buffer, wraps it in a `TileTensor`, and returns the
`TileTensor` alone produces a dangling pointer the instant the function
returns; a stress test that allocated 2000 more buffers between the call and
first read corrupted 100% of the "returned" tensor's elements. `Tensor` owns
the `DeviceBuffer` alongside a compile-time row-major layout, so the value
`zeros[dtype, 4, 4](ctx)` returns can safely outlive the call that built it.
The view is valid only as long as the owning `Tensor` is.

The seam runs one way cheaply and the other way at a cost, and that
asymmetry is inherent rather than an omission: `view()` hands out a pointer
and a layout for free, while `from_view` has to *copy*, because a
`TileTensor` owns nothing that could be adopted. There is deliberately no
constructor taking a `TileTensor`, `DeviceBuffer` or raw pointer.

**Host access goes through `to_host`/`copy_from_host`, never a raw
pointer.** `DeviceBuffer.unsafe_ptr()` is not a safe way to read a tensor's
elements on the host: on CUDA it does not raise, it returns a *device*
pointer, and the host read segfaults the process (verified on an A10G --
a crash, not a catchable error). `map_to_host` is the one accessor correct on
both devices (on a CPU context it maps the same memory; on a GPU it stages a
transfer and flushes writes back on scope exit), so the two methods below
wrap it and there is deliberately no `__getitem__`/`__setitem__` on `Tensor`
to be reached for by accident.

**Comptime shape only.** `row_major[*dims: Int]()` (compile-time variadic)
is what satisfies `numax.core.tensor`'s `where all_dims_known and is_row_major`
clause; the runtime-shape sibling `row_major(Coord)` produces
`all_dims_known=False` and fails that same clause. So every function here
takes a compile-time `*dims: Int` shape, matching `numax.core.tensor`'s existing
contract exactly. Dynamic-shape creation is out of scope; `numax.core.tensor`'s
runtime-shape overloads take a `TileTensor` the caller has laid out.

**Two parameter shapes, and why.** The fixed-arity factories put the count
first with `dtype` defaulted -- `linspace[5](0, 1)`, `eye[3]()`, and
`linspace[5, f32](0, 1)` when the dtype is not `float64`. The variadic ones
cannot: `*dims` has to come last, so `zeros`/`ones`/`full`/`empty` keep
`zeros[f32, 2, 3]()`. That is the whole of the inconsistency, and it is a
language constraint rather than a choice. The fixed-arity group also takes
its endpoints as `Float64` rather than `Scalar[dtype]`: a `Scalar[dtype]`
argument lets the *literals* infer `dtype`, and inference outranks a
default, so `linspace[5](0, 1)` would quietly hand back an integer tensor.

**Which functions need a `DeviceContext`.** The root factories (`zeros`,
`ones`, `full`, `empty`, `eye`, `linspace`, `logspace`, `arange`) take one,
because they allocate with no input tensor to inherit a device from.
Everything derived from an existing tensor (`*_like`, `transpose`,
`squeeze`, `stack`, `reshape`, `ravel`, `concatenate`, `split`) allocates on
the *input's* device via `DeviceBuffer.context()`, so those signatures are
unchanged and a derived tensor can never silently land on the wrong device.

**What MAX's own `nn` versions are, and why these are written here.**
`nn` does ship `arange`, `reshape`, `concat`, `split`, `slice`, `tile`,
`broadcast`, `cumsum`, `argsort` and `argmax`/`argmin` over `TileTensor`,
but they are graph-operator kernels, not array functions: `nn.arange`
returns one SIMD vector for a given index rather than filling a tensor,
`nn.concat` wants a pre-sized output tensor plus a `DeviceContext`, and
`nn.reshape` returns a *dynamically*-laid-out `TileTensor` that fails
`numax.core.tensor`'s `all_dims_known` clause. So the four below are written
against this module's own comptime-shaped `Tensor` instead. The ones still
not wrapped -- `slice`, `tile`, `broadcast`, `cumsum`, `argsort` -- are
either genuinely better reached through `nn`/`numax.stats` or wait on
the runtime-shape array; `numax.stats.argmax`/`argmin` already route
into `nn.argmaxmin`.

**Manipulation scope, stated plainly.** `TileTensor.transpose()` already
gives a zero-copy *view* with every axis reversed (not merely the last two);
`transpose` here is a genuinely different thing -- an owned-copy 2D matrix
transpose, useful when the result needs to outlive the source or be handed
to something that wants its own storage. `squeeze` covers the two concrete
directions a matrix collapses to a vector (`(1, n) -> (n,)` and
`(n, 1) -> (n,)`); a fully general N-dimensional squeeze would need to
build a new variadic shape parameter pack from an arbitrary subset of an
existing one, which Mojo's parameter-pack machinery doesn't expose a public
way to do. `stack` covers exactly two same-shaped tensors along a new
leading axis (`axis=0`); `axis=1` stacking is not provided -- both are
real, documented scope limits, not oversights.

Every manipulation here runs its element walk on the host, through
`to_host`/`copy_from_host`. On a CPU context that is the memory itself and
costs nothing; on a GPU context it stages a round trip, which is the wrong
shape for a large device-resident tensor.
# ponytail: host-staged manipulation walks, correct on both devices but a
# round trip on GPU -- give `transpose`/`stack`/`concatenate` device kernels
# (or route them into `nn.concat`/`linalg.transpose`) if a profile ever shows
# one of them on a hot device path.
"""

from std.collections import Array

from max.gpu.host import DeviceBuffer, DeviceContext
from layout import Coord, TileTensor
from layout.coord import DynamicCoord
from layout.tile_layout import row_major, TensorLayout
from linalg.transpose import transpose as _max_transpose

from .numeric import FloatLike
from .plain import Plain
from .ops import (
    add as _add,
    divide as _divide,
    multiply as _multiply,
    negative as _negative,
    subtract as _subtract,
)


def _product[*dims: Int]() -> Int:
    """The element count of a compile-time shape -- the product of `dims`."""
    var p = 1
    comptime for i in range(dims.__len__()):
        p *= dims[i]
    return p


comptime _LayoutOf[*dims: Int] = type_of(row_major[*dims]())
"""The row-major layout type of a compile-time shape."""

comptime _DynLayoutOf[rank: Int] = type_of(
    row_major(DynamicCoord[DType.int64, rank]())
)
"""The row-major layout type of a rank-`rank` shape carried at run time.

Static and dynamic are two spellings rather than one with a sentinel extent
because a sentinel makes staticness depend on a parameter's *value*: under a
generic `*dims` the compiler cannot decide whether `dims[i] >= 0`, so every
generic wrapper -- `zeros_like`, and any caller's own -- would have to
restate a `where` clause to say so. Keeping the two apart makes the answer
structural, and a shape that is static in some axes and dynamic in others is
still expressible by naming its layout type directly.
"""


def _dyn_shape[rank: Int](*extents: Int) -> DynamicCoord[DType.int64, rank]:
    """A rank-`rank` shape coordinate holding `extents`."""
    var shape = DynamicCoord[DType.int64, rank]()
    comptime for i in range(rank):
        shape[i] = rebind[DynamicCoord[DType.int64, rank].element_types[i]](
            Scalar[DType.int64](extents[i])
        )
    return shape


def _dyn_shape_from[
    rank: Int
](extents: List[Int]) -> DynamicCoord[DType.int64, rank]:
    """`_dyn_shape` for extents held in a list rather than a pack, which is
    what a rank-generic computation produces."""
    var shape = DynamicCoord[DType.int64, rank]()
    comptime for i in range(rank):
        shape[i] = rebind[DynamicCoord[DType.int64, rank].element_types[i]](
            Scalar[DType.int64](extents[i])
        )
    return shape


struct Tensor[dtype: DType, LayoutType: TensorLayout](Movable, Writable):
    """An owned `DeviceBuffer` paired with a row-major layout.

    The one owning tensor type in `numax`, on CPU and GPU alike -- see this
    module's docstring for why the wrapper exists and how the device is
    chosen. The shape lives in `LayoutType`, one dimension at a time: an
    extent compiled in as a `ComptimeInt` is known to every `where` clause
    and every kernel launch, one carried as a `Scalar` is read at run time.
    `Shaped[dtype, 4, 4]` names the fully-static case and
    `Dynamic[dtype, 2]` a matrix whose extents arrive at run time, both of
    them this same type.

    `num_elements` is the compile-time element count and is only meaningful
    when `LayoutType.all_dims_known`; `size()` is the run-time count and is
    correct either way.

    Conforms to `Writable`, so `print(a)` works; `a.format(precision=8)`
    is the same output with the precision and truncation under the
    caller's control.
    """

    comptime rank = Self.LayoutType.rank
    comptime num_elements = Self.LayoutType.static_product

    var buffer: DeviceBuffer[Self.dtype]
    var layout: Self.LayoutType
    var host_addressable: Bool
    """Whether this tensor's storage can be read through a plain host
    pointer -- true on a CPU context, false on a discrete GPU.

    Recorded once at construction rather than asked per access, because
    `DeviceContext.api()` builds a `String` every call. A GPU whose memory
    *is* host-addressable (Apple's unified memory) is treated as if it were
    not: the mapping path is correct there too, just slower, and this way
    the fast path is only ever taken where it is unconditionally safe.
    """

    @staticmethod
    def _static_layout() raises -> Self.LayoutType:
        """The layout of a fully compile-time shape, which needs nothing at
        run time to build.

        Raises on a run-time-shaped `LayoutType`, where the extents are not
        in the type to read: those tensors are built from a layout, which is
        what `zeros_dyn` and its siblings pass. The check is a comptime
        constant folded away in either case, and is a raise rather than a
        `where` clause because a constraint on `all_dims_known` cannot be
        discharged from inside a function generic over the shape -- every
        wrapper up the call chain would have to restate it.
        """
        if not Self.LayoutType.all_dims_known:
            raise Error(
                "shape is not compile-time; build this tensor from a layout"
            )
        return rebind[Self.LayoutType](
            row_major(Coord[*Self.LayoutType._shape_types]())
        )

    def __init__(out self, ctx: DeviceContext) raises:
        """A zero-filled tensor on `ctx`'s device, at a compile-time shape.
        See the layout-taking form below."""
        self = Self(ctx, Self._static_layout())

    def __init__(
        out self, ctx: DeviceContext, var values: List[Scalar[Self.dtype]]
    ) raises:
        """A tensor on `ctx`'s device holding `values`, row-major, at a
        compile-time shape."""
        self = Self(ctx, Self._static_layout(), values^)

    def __init__(out self, ctx: DeviceContext, layout: Self.LayoutType) raises:
        """A zero-filled tensor on `ctx`'s device with the given layout.

        Zeroing is `ctx.enqueue_memset`, which runs on the device rather
        than staging a host write, so this is the cheap constructor on both
        paths.

        Note the `ctx` here comes **first and required**, while every factory
        below takes it **last and optional** -- the constructor has no other
        way to learn the device, the factories do. The two spellings sit
        within a couple of lines of each other in ordinary code
        (`var xs = linspace[n](0, 1, ctx=cpu)` beside `var ys = T(cpu)`), and
        writing the factory the constructor's way is a compile error, not a
        silent one.

        Not available at `DType.bool`: `enqueue_memset` on a `bool` buffer
        fails to compile under the pinned toolchain ("failed to run the
        pass manager"), and neither does a `comptime if` around it, since
        both branches are still type-checked and `Scalar[bool](0)` is
        rejected on its own. Build a boolean mask from a `List` through the
        constructor below, or get one from a `numax.core.logic` comparison,
        which is where masks come from in practice.

        The `synchronize` is not optional: `enqueue_memset` only *queues*
        the fill, and `__getitem__`'s host-pointer fast path does not order
        itself against queued work the way a host mapping does. Without it,
        a read immediately after construction can see unwritten memory
        (caught by `tests/core/test_array.mojo`'s zero-content check).
        """
        self.layout = layout
        self.buffer = ctx.enqueue_create_buffer[Self.dtype](layout.size())
        self.host_addressable = ctx.api() == "cpu"
        ctx.enqueue_memset(self.buffer, Scalar[Self.dtype](0))
        ctx.synchronize()

    def __init__(
        out self,
        ctx: DeviceContext,
        layout: Self.LayoutType,
        var values: List[Scalar[Self.dtype]],
    ) raises:
        """A tensor on `ctx`'s device holding `values`, row-major.

        `values` must have exactly `layout.size()` entries; this is the
        escape hatch the host-filled factory functions below funnel through.
        """
        self.layout = layout
        self.buffer = ctx.enqueue_create_buffer[Self.dtype](layout.size())
        self.host_addressable = ctx.api() == "cpu"
        with self.buffer.map_to_host() as host:
            for i in range(layout.size()):
                host[i] = values[i]

    def context(self) raises -> DeviceContext:
        """The device this tensor's storage lives on.

        What the derived manipulations use to allocate their result next to
        their input rather than on a device the caller has to name again.
        """
        return self.buffer.context()

    def size(self) -> Int:
        """The element count, read from the layout.

        Correct whether or not the shape is fully compile-time; the
        `num_elements` alias is the comptime answer and is only meaningful
        when `LayoutType.all_dims_known`.
        """
        return self.layout.size()

    def dim[i: Int](self) -> Int:
        """The extent of axis `i`."""
        return Int(self.layout.shape[i]().value())

    def dim_at(self, axis: Int) -> Int:
        """The extent of `axis`, chosen at run time.

        `dim` takes its axis as a parameter, which is what a caller writing
        `a.dim[0]()` wants and what a loop over a computed axis cannot use.
        This walks the compile-time axis list and picks one, so an axis
        outside the rank returns `0` rather than failing to compile.
        """
        var extent = 0
        comptime for i in range(Self.rank):
            if i == axis:
                extent = self.dim[i]()
        return extent

    def stride_at(self, axis: Int) -> Int:
        """The stride of `axis` in elements, chosen at run time.

        The distance in memory between neighbours along `axis`. Row-major
        for every tensor this module builds, but read from the layout
        rather than assumed, so it stays correct for a layout that is not.
        """
        var stride = 0
        comptime for i in range(Self.rank):
            if i == axis:
                stride = Int(self.layout.stride[i]().value())
        return stride

    def view(mut self) -> TileTensor[Self.dtype, Self.LayoutType, MutAnyOrigin]:
        """A `TileTensor` view over this tensor's storage.

        The type every `numax.core.tensor` entry point and every MAX kernel
        takes, on CPU and GPU alike. Valid only as long as `self` is alive
        -- pass `self` (or a `mut` reference to it) around, not just the
        value returned here, if the view needs to outlive this call.
        """
        var v: TileTensor[
            Self.dtype, Self.LayoutType, MutAnyOrigin
        ] = TileTensor(self.buffer, self.layout)
        return v

    @staticmethod
    def from_view(
        v: TileTensor[Self.dtype, Self.LayoutType, MutAnyOrigin],
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """A new owning tensor holding a **copy** of `v`'s elements.

        The way back up from the view layer, and the name says copy because
        that is the only thing it can be: a `TileTensor` is a pointer plus a
        layout in every storage policy MAX ships, so there is no buffer to
        adopt and no way to take ownership of one. `view()` down is free;
        this direction costs an allocation and an element walk.

        Reach for it when a kernel wrote into a view over borrowed storage
        (a `List`, another tensor's slice) and the result has to outlive it.

        **Host-resident views only.** All this has to work with is a pointer
        and a layout -- there is no `DeviceBuffer` behind a `TileTensor` to
        `map_to_host`, so the element walk is a plain host read and a view
        over device memory would segfault it on CUDA, exactly as this
        module's docstring describes for `unsafe_ptr()`. When the source is
        already an owning device-resident `Tensor`, `to_host()` plus a
        factory is the route; `from_view` is for the borrowed-storage case,
        which is a host case by construction.
        """
        var device = _context(ctx)
        var src = v.ptr_at_offset(Coord(0))
        var n = v.layout.size()
        var values = List[Scalar[Self.dtype]](capacity=n)
        for i in range(n):
            values.append(src[unsafe_offset=i])
        return Self(device, v.layout, values^)

    def to_host(self) raises -> List[Scalar[Self.dtype]]:
        """A host copy of every element, row-major.

        The bulk read path, and the only one that costs a single mapping
        regardless of device -- see this module's docstring for why
        `DeviceBuffer.unsafe_ptr()` alone is not a safe substitute.
        """
        var n = self.size()
        var out = List[Scalar[Self.dtype]](capacity=n)
        with self.buffer.map_to_host() as host:
            for i in range(n):
                out.append(host[i])
        return out^

    def __getitem__(self, i: Int) raises -> Scalar[Self.dtype]:
        """Flat (row-major) element read.

        On a CPU context this is a direct pointer read. On a GPU it stages
        a host mapping *per access*, which costs about 0.8 ms an element
        (measured) -- take one `to_host()` copy and index that instead when
        reading more than a handful of elements off a device tensor.
        """
        if self.host_addressable:
            return self.buffer.unsafe_ptr()[unsafe_offset=i]
        with self.buffer.map_to_host() as host:
            return host[i]

    def __getitem__(self, r: Int, c: Int) raises -> Scalar[Self.dtype]:
        """`a[r, c]` on a rank-2 tensor, row-major.

        The same read as the flat `a[r * cols + c]`, spelled the way the
        shape is. Rank-2 only -- axis `1` is a compile-time error on
        a rank-1 tensor, so a wrong-rank call fails where it is written.
        `.view()[i, j, k]` is the general form, and the same per-access
        mapping cost applies on a GPU.
        """
        return self[r * self.dim[1]() + c]

    def __setitem__(mut self, r: Int, c: Int, value: Scalar[Self.dtype]) raises:
        """`a[r, c] = v` on a rank-2 tensor. See `__getitem__` above."""
        self[r * self.dim[1]() + c] = value

    def __setitem__(mut self, i: Int, value: Scalar[Self.dtype]) raises:
        """Flat (row-major) element assignment.

        Same split as `__getitem__`: a direct pointer write on a CPU
        context, a per-access host mapping on a GPU (flushed to the device
        when this call's mapping scope exits). `copy_from_host` is the bulk
        path.
        """
        if self.host_addressable:
            self.buffer.unsafe_ptr()[unsafe_offset=i] = value
            return
        with self.buffer.map_to_host() as host:
            host[i] = value

    def format(
        self,
        *,
        precision: Int = 4,
        threshold: Int = 1000,
        edge_items: Int = 3,
    ) raises -> String:
        """`a`'s contents as a string, NumPy-`arrayprint`-shaped.

        `print(a)` is this at the defaults; call it directly when the
        defaults are wrong -- `print(a.format(precision=8))`.

        Every element is printed with `precision` digits after the decimal
        point. Past `threshold` elements only the first and last
        `edge_items` are shown with a `...` between them, the same
        truncation NumPy's default printer applies. NumPy's line-wrapping
        (`linewidth`) is not reproduced: this is one flat, comma-separated
        line.
        """
        return _format_tensor(self, precision, threshold, edge_items)

    def write_to(self, mut writer: Some[Writer]):
        """`print(a)`, at `format`'s default settings.

        `format` above is the same output with `precision`, `threshold`
        and `edge_items` under the caller's control; both go through
        `_format_tensor` below, so the two cannot disagree.

        `Writable.write_to` cannot raise, but reading the elements can --
        it is a device mapping. A failed read prints as `<unreadable>`
        rather than taking the process down inside a `print`; a caller who
        needs the error takes `to_host()` itself, which does raise.
        """
        try:
            writer.write(_format_tensor(self, 4, 1000, 3))
        except e:
            writer.write("<unreadable: ", String(e), ">")

    def __add__(self, other: Self) raises -> Self:
        """`a + b`, elementwise. Forwards to `numax.core.ops.add`."""
        return _add(self, other)

    def __add__(self, other: Scalar[Self.dtype]) raises -> Self:
        """`a + b` with a scalar `b`."""
        return _add(self, other)

    def __sub__(self, other: Self) raises -> Self:
        """`a - b`, elementwise. Forwards to `numax.core.ops.subtract`."""
        return _subtract(self, other)

    def __sub__(self, other: Scalar[Self.dtype]) raises -> Self:
        """`a - b` with a scalar `b`."""
        return _subtract(self, other)

    def __mul__(self, other: Self) raises -> Self:
        """`a * b`, elementwise. Forwards to `numax.core.ops.multiply`."""
        return _multiply(self, other)

    def __mul__(self, other: Scalar[Self.dtype]) raises -> Self:
        """`a * b` with a scalar `b`."""
        return _multiply(self, other)

    def __truediv__(self, other: Self) raises -> Self:
        """`a / b`, elementwise. Forwards to `numax.core.ops.divide`."""
        return _divide(self, other)

    def __truediv__(self, other: Scalar[Self.dtype]) raises -> Self:
        """`a / b` with a scalar `b`."""
        return _divide(self, other)

    def __neg__(self) raises -> Self:
        """`-a`, elementwise. Forwards to `numax.core.ops.negative`."""
        return _negative(self)

    def copy_from_host(mut self, values: List[Scalar[Self.dtype]]) raises:
        """Overwrite every element from a host buffer, row-major.

        `values` must have exactly `size()` entries. On a GPU
        context the write is flushed to the device when the mapping scope
        exits, which is inside this call.
        """
        var n = self.size()
        with self.buffer.map_to_host() as host:
            for i in range(n):
                host[i] = values[i]


comptime Shaped[dtype: DType, *dims: Int] = Tensor[dtype, _LayoutOf[*dims]]
"""`Tensor` at a compile-time shape: `Shaped[f32, 2, 3]` is a 2x3.

The struct's own name cannot double as this alias, so a signature that
names a shape spells `Shaped`; the factories take the same `*dims`.
"""

comptime Dynamic[dtype: DType, rank: Int] = Tensor[dtype, _DynLayoutOf[rank]]
"""`Tensor` at a rank that is compile-time and extents that are not:
`Dynamic[f32, 2]` is a matrix whose shape is a constructor argument.

`zeros_dyn` and its siblings build one. The name differs from `Shaped`
rather than overloading it because a parameter list of extents and a
parameter list holding only a rank cannot be told apart at a call site.
"""


def _context(ctx: Optional[DeviceContext]) raises -> DeviceContext:
    """The device a factory should allocate on: the caller's, or the host.

    Every factory below takes `ctx` last and optional, so `zeros[f32, 2,
    3]()` lands on the CPU and `zeros[f32, 2, 3](gpu)` lands on `gpu` --
    unlike `Tensor`'s own constructor, which takes it first and required. It
    cannot be an ordinary default argument because `DeviceContext(api="cpu")`
    raises, and a raising expression is not admissible in that position.
    """
    return ctx.value() if ctx else DeviceContext(api="cpu")


def zeros[
    dtype: DType, *dims: Int
](ctx: Optional[DeviceContext] = None) raises -> Shaped[dtype, *dims]:
    """A new tensor of the given compile-time shape on `ctx`'s device,
    filled with `0`."""
    return Shaped[dtype, *dims](_context(ctx))


def zeros_dyn[
    dtype: DType, rank: Int
](*extents: Int, ctx: Optional[DeviceContext] = None) raises -> Dynamic[
    dtype, rank
]:
    """A new zero-filled tensor of rank `rank` whose extents are `extents`.

    The run-time-shaped sibling of `zeros`: `zeros_dyn[f32, 2](rows, cols)`
    where `zeros[f32, 4, 3]()` would have compiled the shape in.
    """
    return Dynamic[dtype, rank](
        _context(ctx), row_major(_dyn_shape[rank](*extents))
    )


def asarray[
    dtype: DType
](
    var values: List[Scalar[dtype]], ctx: Optional[DeviceContext] = None
) raises -> Dynamic[dtype, 1]:
    """A rank-1 tensor holding `values`, as long as `values` is.

    The constructor for data whose length is a run-time fact.
    `Shaped[dtype, n](ctx, values)` needs `n` in the type, so anything
    producing a count instead of a constant -- a boolean mask, `unique`, a
    file read -- had no way to hand back a right-sized tensor. This does,
    and it is what those functions return.
    """
    var n = len(values)
    var result = Dynamic[dtype, 1](_context(ctx), row_major(_dyn_shape[1](n)))
    result.copy_from_host(values)
    return result^


def ones[
    dtype: DType, *dims: Int
](ctx: Optional[DeviceContext] = None) raises -> Shaped[dtype, *dims]:
    """A new tensor of the given compile-time shape on `ctx`'s device,
    filled with `1`."""
    return full[dtype, *dims](1, ctx=ctx)


def ones_dyn[
    dtype: DType, rank: Int
](*extents: Int, ctx: Optional[DeviceContext] = None) raises -> Dynamic[
    dtype, rank
]:
    """A new one-filled tensor sized by `extents`; see `zeros_dyn`."""
    return _filled(zeros_dyn[dtype, rank](*extents, ctx=ctx), 1)


def _filled[
    dtype: DType, LayoutType: TensorLayout
](
    var result: Tensor[dtype, LayoutType], fill_value: Scalar[dtype]
) raises -> Tensor[dtype, LayoutType]:
    """`result` overwritten with `fill_value` on its own device.

    The fill is `enqueue_memset`, so it runs on the device rather than
    staging a host write -- the same reason `zeros` is cheap on both paths.
    """
    var device = result.context()
    device.enqueue_memset(result.buffer, fill_value)
    device.synchronize()
    return result^


def full[
    dtype: DType, *dims: Int
](
    fill_value: Scalar[dtype], ctx: Optional[DeviceContext] = None
) raises -> Shaped[dtype, *dims]:
    """A new tensor of the given compile-time shape on `ctx`'s device,
    filled with `fill_value`."""
    return _filled(zeros[dtype, *dims](ctx), fill_value)


def full_dyn[
    dtype: DType, rank: Int
](
    fill_value: Scalar[dtype],
    *extents: Int,
    ctx: Optional[DeviceContext] = None,
) raises -> Dynamic[dtype, rank]:
    """A new `fill_value`-filled tensor sized by `extents`; see `zeros_dyn`."""
    return _filled(zeros_dyn[dtype, rank](*extents, ctx=ctx), fill_value)


def empty[
    dtype: DType, *dims: Int
](ctx: Optional[DeviceContext] = None) raises -> Shaped[dtype, *dims]:
    """A new tensor of the given compile-time shape, its contents unspecified.

    Unlike NumPy's `empty`, this zero-initializes rather than truly leaving
    the memory uninitialized -- `numax` favors a memory-safe default over
    the small allocation-time saving, matching every other factory function
    here. Callers that write every element before reading (the usual reason
    to reach for `empty` at all) pay nothing extra in practice.
    """
    return Shaped[dtype, *dims](_context(ctx))


def empty_dyn[
    dtype: DType, rank: Int
](*extents: Int, ctx: Optional[DeviceContext] = None) raises -> Dynamic[
    dtype, rank
]:
    """A new tensor sized by `extents`; see `empty` for why this
    zero-initializes rather than leaving memory uninitialized."""
    return zeros_dyn[dtype, rank](*extents, ctx=ctx)


def eye[
    n: Int, dtype: DType = DType.float64
](ctx: Optional[DeviceContext] = None) raises -> Shaped[dtype, n, n]:
    """The `n`x`n` identity matrix."""
    var values = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        values[i * n + i] = 1
    return Shaped[dtype, n, n](_context(ctx), values^)


def linspace[
    num: Int, dtype: DType = DType.float64
](
    start: Float64,
    stop: Float64,
    ctx: Optional[DeviceContext] = None,
) raises -> Shaped[dtype, num]:
    """`num` evenly spaced values from `start` to `stop`, inclusive of both.

    Matches `numpy.linspace`'s default `endpoint=True`. `num == 1` returns
    `[start]`, avoiding a division by zero in the step computation.

    The count comes first and `dtype` defaults to `float64`, so the common
    call is `linspace[5](0, 1)` and the dtype is named only when it is not
    the default: `linspace[5, f32](0, 1)`. The endpoints are `Float64`
    rather than `Scalar[dtype]` deliberately -- taking them at the tensor's
    own dtype would let `linspace[5](0, 1)` *infer* an integer dtype from
    the literals and hand back an integer tensor, since inference outranks
    a default. Every factory below is shaped the same way for the same
    reason.
    """
    var lo = Scalar[dtype](start)
    var values = List[Scalar[dtype]](capacity=num)
    comptime if num == 1:
        values.append(lo)
    else:
        var step = (Scalar[dtype](stop) - lo) / Scalar[dtype](num - 1)
        for i in range(num):
            values.append(lo + Scalar[dtype](i) * step)
    return Shaped[dtype, num](_context(ctx), values^)


def logspace[
    num: Int, dtype: DType = DType.float64
](
    start: Float64,
    stop: Float64,
    base: Float64 = 10,
    ctx: Optional[DeviceContext] = None,
) raises -> Shaped[dtype, num]:
    """`num` values evenly spaced on a log scale: `base**x` for `x` in
    `linspace(start, stop, num)`. Matches `numpy.logspace`'s defaults.

    Count first, `dtype` defaulted -- see `linspace` for why the arguments
    are `Float64`."""
    var lo = Scalar[dtype](start)
    var b = Scalar[dtype](base)
    var values = List[Scalar[dtype]](capacity=num)
    comptime if num == 1:
        values.append(b**lo)
    else:
        var step = (Scalar[dtype](stop) - lo) / Scalar[dtype](num - 1)
        for i in range(num):
            values.append(b ** (lo + Scalar[dtype](i) * step))
    return Shaped[dtype, num](_context(ctx), values^)


def arange[
    num: Int, dtype: DType = DType.float64
](
    start: Float64 = 0,
    step: Float64 = 1,
    ctx: Optional[DeviceContext] = None,
) raises -> Shaped[dtype, num]:
    """`num` values starting at `start`, spaced by `step`.

    `numpy.arange` takes a `stop` and derives the count from it, which makes
    the output extent depend on runtime values; this module's shapes are
    comptime, so the count is the parameter and `stop` is implied
    (`start + num*step`). `numax.core.array.linspace` is the one to reach for
    when the endpoints are what matter.
    """
    var first = Scalar[dtype](start)
    var by = Scalar[dtype](step)
    var values = List[Scalar[dtype]](capacity=num)
    for i in range(num):
        values.append(first + Scalar[dtype](i) * by)
    return Shaped[dtype, num](_context(ctx), values^)


def zeros_like[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Tensor[dtype, LayoutType]:
    """A new zero-filled tensor with `a`'s dtype, shape and device."""
    return Tensor[dtype, LayoutType](a.context(), a.layout)


def ones_like[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Tensor[dtype, LayoutType]:
    """A new one-filled tensor with `a`'s dtype, shape and device."""
    return full_like(a, 1)


def full_like[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], fill_value: Scalar[dtype]) raises -> Tensor[
    dtype, LayoutType
]:
    """A new `fill_value`-filled tensor with `a`'s dtype, shape and device."""
    return _filled(zeros_like(a), fill_value)


def empty_like[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Tensor[dtype, LayoutType]:
    """A new tensor with `a`'s dtype, shape and device; see `empty`'s own
    docstring for why this zero-initializes rather than leaving memory
    uninitialized.
    """
    return zeros_like(a)


def transpose[
    dtype: DType, rows: Int, cols: Int
](mut a: Shaped[dtype, rows, cols]) raises -> Shaped[dtype, cols, rows]:
    """An owned-copy transpose of a 2D tensor, on `a`'s own device.

    The permutation itself is `linalg.transpose` -- MAX's own kernel, which
    dispatches to SIMD-shuffle tile kernels and runs on either device. numax
    only allocates the destination and names the axis permutation; an
    earlier version here walked the elements one at a time on the host.

    Takes `a` mutably even though it only reads it: `view()` hands back a
    `TileTensor` that can write, and a mutable view cannot be built from an
    immutable binding -- the compiler enforces that, so an immutable
    `transpose` would have to copy the source first. Callers hold their
    tensors in `var` bindings anyway, so `transpose(m)` reads the same
    either way.

    Distinct from `TileTensor.transpose()`, which returns a zero-copy view
    with every axis reversed over the *same* memory -- this allocates a new
    buffer, for when the result needs its own storage (e.g. to outlive the
    source, or to feed something that wants a plain `Tensor`).
    """
    var ctx = a.context()
    var result = Shaped[dtype, cols, rows](ctx)
    var src = a.view()
    var dst = result.view()
    var perms = List[Int]()
    perms.append(1)
    perms.append(0)
    _max_transpose(dst, src, perms.unsafe_ptr(), ctx)
    ctx.synchronize()
    return result^


def squeeze[
    dtype: DType, n: Int
](a: Shaped[dtype, 1, n]) raises -> Shaped[dtype, n]:
    """Drop a size-1 leading axis: `(1, n) -> (n,)`."""
    return Shaped[dtype, n](a.context(), a.to_host())


def squeeze[
    dtype: DType, n: Int
](a: Shaped[dtype, n, 1]) raises -> Shaped[dtype, n]:
    """Drop a size-1 trailing axis: `(n, 1) -> (n,)`."""
    return Shaped[dtype, n](a.context(), a.to_host())


def stack[
    dtype: DType, n: Int
](a: Shaped[dtype, n], b: Shaped[dtype, n]) raises -> Shaped[dtype, 2, n]:
    """Stack two same-shaped rank-1 tensors along a new leading axis
    (`axis=0`): `ys[0, :] = a`, `ys[1, :] = b`.

    `axis=1` stacking is not provided -- see this module's own docstring --
    and neither is a variadic `stack(a, b, c)`. A variadic pack's length is
    a runtime property, but the result's leading extent is part of its
    type, so the count would have to be passed a second time as a
    parameter (`stack[dtype, n, 3](a, b, c)`) and checked against the pack
    at runtime. Two spellings of the same number is worse than two
    arguments.
    """
    var a_values = a.to_host()
    var b_values = b.to_host()
    var values = List[Scalar[dtype]](capacity=2 * n)
    for i in range(n):
        values.append(a_values[i])
    for i in range(n):
        values.append(b_values[i])
    return Shaped[dtype, 2, n](a.context(), values^)


def reshape[
    dtype: DType, n: Int, rows: Int, cols: Int
](a: Shaped[dtype, n]) raises -> Shaped[dtype, rows, cols] where (
    rows * cols == n
):
    """A rank-2 copy of a rank-1 tensor, in row-major order.

    The `where` clause is the element-count check: a shape that does not
    multiply out to `n` fails to compile rather than producing a tensor
    whose storage is the wrong length.

    Concrete ranks rather than a general `*new_dims` pack, for two reasons.
    A general form would need to pair two variadic packs (`*dims` in, and
    `*new_dims` out) in one signature, which Mojo's pack machinery does not
    expose; and the element-count constraint has to be *provable* to
    `where`, which admits builtin arithmetic like `rows * cols` but not a
    call to an ordinary `def` such as `_product`. `squeeze` above takes the
    same shape for the same kind of reason. `ravel` is the inverse, and the
    two compose into any reshape this module can express.
    """
    return Shaped[dtype, rows, cols](a.context(), a.to_host())


def reshape[
    dtype: DType, n: Int, d0: Int, d1: Int, d2: Int
](a: Shaped[dtype, n]) raises -> Shaped[dtype, d0, d1, d2] where (
    d0 * d1 * d2 == n
):
    """A rank-3 copy of a rank-1 tensor, in row-major order. See the rank-2
    overload above for why the ranks are spelled out."""
    return Shaped[dtype, d0, d1, d2](a.context(), a.to_host())


def reshape_dyn[
    dtype: DType, LayoutType: TensorLayout, rank: Int
](a: Tensor[dtype, LayoutType], *extents: Int) raises -> Dynamic[dtype, rank]:
    """A copy of `a` at the given shape, in row-major order, at any rank.

    What the comptime `reshape` overloads above cannot do: the target shape
    is a run-time value, so it can be computed -- `reshape_dyn[rank=2](a, n
    // cols, cols)` -- rather than spelled as a literal. It also takes any
    input rank, where those take rank 1.

    The element count is checked here and raises on a mismatch, which is
    the price of a shape the compiler cannot see. Prefer `reshape` when the
    shape is a constant: there the same check is a `where` clause and the
    mismatch is a compile error.
    """
    var wanted = 1
    for i in range(rank):
        wanted *= extents[i]
    if wanted != a.size():
        raise Error(
            "reshape_dyn: a shape of ",
            wanted,
            " elements does not fit ",
            a.size(),
        )
    var result = Dynamic[dtype, rank](
        a.context(), row_major(_dyn_shape[rank](*extents))
    )
    result.copy_from_host(a.to_host())
    return result^


def slice[
    dtype: DType, LayoutType: TensorLayout
](
    a: Tensor[dtype, LayoutType], starts: List[Int], stops: List[Int]
) raises -> Dynamic[dtype, LayoutType.rank]:
    """The sub-box `a[starts[0]:stops[0], starts[1]:stops[1], ...]`.

    Basic slicing, at any rank, with bounds read at run time -- so the
    result's extents are run-time values too, which is why it comes back
    `Dynamic`. Half-open, like NumPy and like Mojo's own ranges.

    This copies into compact storage rather than returning a view. A view
    is expressible -- `TileTensor.slice` produces one, and
    `numax.core.tensor.map_strided` walks it -- but it would borrow from a
    tensor this module does not own, and every other result here is owned.
    Reach for the view directly when the copy is the expensive part.

    Bounds are checked, and a `stop` below its `start` raises rather than
    quietly producing an empty axis, since that is far more often a bug
    than an intent.
    """
    comptime rank = LayoutType.rank
    if len(starts) != rank or len(stops) != rank:
        raise Error("slice: expected ", rank, " bounds per side")

    var extents = List[Int](capacity=rank)
    var count = 1
    for d in range(rank):
        if starts[d] < 0 or stops[d] > a.dim_at(d) or stops[d] < starts[d]:
            raise Error(
                "slice: axis ",
                d,
                " bounds [",
                starts[d],
                ", ",
                stops[d],
                ") do not fit an extent of ",
                a.dim_at(d),
            )
        extents.append(stops[d] - starts[d])
        count *= extents[d]

    var values = a.to_host()
    var out = List[Scalar[dtype]](capacity=count)
    for flat in range(count):
        # Walk the result in row-major order, converting each coordinate
        # back into a flat index of the source.
        var rem = flat
        var src = 0
        for k in range(rank):
            var d = rank - 1 - k
            var c = rem % extents[d]
            rem //= extents[d]
            src += (starts[d] + c) * a.stride_at(d)
        out.append(values[src])

    var result = Dynamic[dtype, rank](
        a.context(), row_major(_dyn_shape_from[rank](extents))
    )
    result.copy_from_host(out)
    return result^


def concatenate_dyn[
    dtype: DType, ALayout: TensorLayout, BLayout: TensorLayout
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    dtype, 1
]:
    """Join two tensors end to end as one flat tensor.

    The run-time-shaped `concatenate`: the two inputs need not have the
    same layout type, and the result's length is their combined size rather
    than a sum the compiler has to see. Reach for `concatenate` when both
    lengths are constants and the result's should be too.
    """
    var values = a.to_host()
    var b_values = b.to_host()
    for i in range(len(b_values)):
        values.append(b_values[i])
    return asarray(values^, a.context())


def split_dyn[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], at: Int) raises -> Tuple[
    Dynamic[dtype, 1], Dynamic[dtype, 1]
]:
    """Cut a tensor in two at a run-time index: elements `[0, at)` and
    `[at, size())`, both flat. The inverse of `concatenate_dyn`.

    `split` needs `at` as a parameter because both output lengths are part
    of their types; here they are not, so the cut point is an ordinary
    argument and can be computed.
    """
    var n = a.size()
    if at < 0 or at > n:
        raise Error(
            "split_dyn: cannot cut at ", at, " in a tensor of ", n, " elements"
        )
    var values = a.to_host()
    var head = List[Scalar[dtype]](capacity=at)
    for i in range(at):
        head.append(values[i])
    var tail = List[Scalar[dtype]](capacity=n - at)
    for i in range(at, n):
        tail.append(values[i])
    var ctx = a.context()
    return (asarray(head^, ctx), asarray(tail^, ctx))


def ravel[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Shaped[
    dtype, LayoutType.static_product
]:
    """A rank-1 copy in row-major order -- the inverse of `reshape`.

    `Tensor` owns its storage and Mojo will not let a field be moved out of
    a value that still has to be destroyed, so this copies rather than
    retypes in place. Every manipulation in this module copies for the same
    reason; `TileTensor.reshape()` is the zero-copy view when the result
    does not need to outlive its source.
    """
    return Shaped[dtype, LayoutType.static_product](a.context(), a.to_host())


def concatenate[
    dtype: DType, n: Int, m: Int
](a: Shaped[dtype, n], b: Shaped[dtype, m]) raises -> Shaped[dtype, n + m]:
    """Join two rank-1 tensors end to end: `numpy.concatenate` at `axis=0`.

    Rank-1 only, for the same reason `stack` takes exactly two rank-1
    inputs: an axis-`k` concatenation of arbitrary-rank tensors needs a new
    parameter pack built from an existing one with a single extent changed.
    A real scope limit, stated rather than hidden.

    The element copy stays here rather than routing to `nn.concat`, and the
    reason is specific: `nn.concat` takes its inputs as a `StaticTuple`, so
    every input must share one layout *type* **and** one origin. `[n]` and
    `[m]` are different comptime layout types, and runtime-shaped views over
    the two buffers still carry different origins (`origin_of(a.buffer)` vs
    `origin_of(b.buffer)`), which only an unsafe origin cast erases. Two
    buffers, one memcpy each, is not worth that.
    """
    var a_values = a.to_host()
    var b_values = b.to_host()
    var values = List[Scalar[dtype]](capacity=n + m)
    for i in range(n):
        values.append(a_values[i])
    for i in range(m):
        values.append(b_values[i])
    return Shaped[dtype, n + m](a.context(), values^)


def split[
    dtype: DType, n: Int, at: Int
](a: Shaped[dtype, n]) raises -> Tuple[
    Shaped[dtype, at], Shaped[dtype, n - at]
] where (at >= 0 and at <= n):
    """Cut a rank-1 tensor in two at comptime index `at`: elements
    `[0, at)` and `[at, n)`. The inverse of `concatenate`.

    `numpy.split` takes a section count or a list of indices and returns a
    variable-length list; both make the *number* of outputs a runtime value,
    which a comptime-shaped tensor cannot express. One index, two outputs,
    is the part that survives that constraint.
    """
    var ctx = a.context()
    var values = a.to_host()
    var head = List[Scalar[dtype]](capacity=at)
    for i in range(at):
        head.append(values[i])
    var tail = List[Scalar[dtype]](capacity=n - at)
    for i in range(at, n):
        tail.append(values[i])
    return (
        Shaped[dtype, at](ctx, head^),
        Shaped[dtype, n - at](ctx, tail^),
    )


def geomspace[
    num: Int, dtype: DType = DType.float64
](
    start: Float64,
    stop: Float64,
    ctx: Optional[DeviceContext] = None,
) raises -> Shaped[dtype, num] where dtype.is_floating_point():
    """`num` values spaced evenly on a geometric progression, endpoints
    included. `numpy.geomspace`.

    `start` and `stop` must share a sign and neither may be zero -- a
    geometric progression through zero does not exist. Unchecked, like
    NumPy's own, because the check costs a branch the caller is better
    placed to make.
    """
    var values = List[Scalar[dtype]](capacity=num)
    comptime if num == 1:
        values.append(Scalar[dtype](start))
    else:
        var ratio = (stop / start) ** (Float64(1) / Float64(num - 1))
        var current = Scalar[dtype](start)
        for _ in range(num):
            values.append(current)
            current = current * Scalar[dtype](ratio)
    return Shaped[dtype, num](_context(ctx), values^)


def identity[
    n: Int, dtype: DType = DType.float64
](ctx: Optional[DeviceContext] = None) raises -> Shaped[dtype, n, n]:
    """The `n`x`n` identity matrix. `numpy.identity`.

    Same result as `eye`; both names exist in NumPy and a caller reaching
    for one should not have to discover the other.
    """
    return eye[n, dtype](ctx)


def diag[
    dtype: DType, n: Int
](a: Shaped[dtype, n]) raises -> Shaped[dtype, n, n]:
    """A square matrix with `a` on its main diagonal. `numpy.diag`.

    The vector-to-matrix direction only; `diagonal` is the inverse.
    """
    var values = List[Scalar[dtype]](length=n * n, fill=0)
    var source = a.to_host()
    for i in range(n):
        values[i * n + i] = source[i]
    return Shaped[dtype, n, n](a.context(), values^)


def diagonal[
    dtype: DType, n: Int
](a: Shaped[dtype, n, n]) raises -> Shaped[dtype, n]:
    """The main diagonal of a square matrix. `numpy.diagonal`."""
    var source = a.to_host()
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(source[i * n + i])
    return Shaped[dtype, n](a.context(), values^)


def diagflat[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Shaped[
    dtype, LayoutType.static_product, LayoutType.static_product
]:
    """`a` flattened onto the diagonal of a square matrix.
    `numpy.diagflat`."""
    comptime n = LayoutType.static_product
    var source = a.to_host()
    var values = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        values[i * n + i] = source[i]
    return Shaped[dtype, n, n](a.context(), values^)


def tri[
    dtype: DType, n: Int
](ctx: Optional[DeviceContext] = None) raises -> Shaped[dtype, n, n]:
    """An `n`x`n` matrix of ones at and below the diagonal. `numpy.tri`."""
    var values = List[Scalar[dtype]](length=n * n, fill=0)
    for r in range(n):
        for c in range(r + 1):
            values[r * n + c] = 1
    return Shaped[dtype, n, n](_context(ctx), values^)


def tril[
    dtype: DType, n: Int
](a: Shaped[dtype, n, n]) raises -> Shaped[dtype, n, n]:
    """`a` with everything above the diagonal zeroed. `numpy.tril`."""
    var source = a.to_host()
    var values = List[Scalar[dtype]](length=n * n, fill=0)
    for r in range(n):
        for c in range(r + 1):
            values[r * n + c] = source[r * n + c]
    return Shaped[dtype, n, n](a.context(), values^)


def triu[
    dtype: DType, n: Int
](a: Shaped[dtype, n, n]) raises -> Shaped[dtype, n, n]:
    """`a` with everything below the diagonal zeroed. `numpy.triu`."""
    var source = a.to_host()
    var values = List[Scalar[dtype]](length=n * n, fill=0)
    for r in range(n):
        for c in range(r, n):
            values[r * n + c] = source[r * n + c]
    return Shaped[dtype, n, n](a.context(), values^)


def vander[
    dtype: DType, n: Int, cols: Int
](a: Shaped[dtype, n]) raises -> Shaped[
    dtype, n, cols
] where dtype.is_floating_point():
    """The Vandermonde matrix of `a`: `out[i, j] = a[i] ** (cols - 1 - j)`.
    `numpy.vander` with its default `increasing=False`."""
    var source = a.to_host()
    var values = List[Scalar[dtype]](length=n * cols, fill=0)
    for r in range(n):
        var power = Scalar[dtype](1)
        for j in range(cols):
            values[r * cols + (cols - 1 - j)] = power
            power = power * source[r]
    return Shaped[dtype, n, cols](a.context(), values^)


def meshgrid[
    dtype: DType, n: Int, m: Int
](x: Shaped[dtype, n], y: Shaped[dtype, m]) raises -> Tuple[
    Shaped[dtype, m, n], Shaped[dtype, m, n]
]:
    """Coordinate matrices from two coordinate vectors. `numpy.meshgrid`
    with its default `indexing="xy"`, so both outputs are `(m, n)`."""
    var xs = x.to_host()
    var ys = y.to_host()
    var xx = List[Scalar[dtype]](length=m * n, fill=0)
    var yy = List[Scalar[dtype]](length=m * n, fill=0)
    for r in range(m):
        for c in range(n):
            xx[r * n + c] = xs[c]
            yy[r * n + c] = ys[r]
    var ctx = x.context()
    return (
        Shaped[dtype, m, n](ctx, xx^),
        Shaped[dtype, m, n](ctx, yy^),
    )


def flip[dtype: DType, n: Int](a: Shaped[dtype, n]) raises -> Shaped[dtype, n]:
    """A rank-1 tensor reversed. `numpy.flip` at `axis=0`."""
    var source = a.to_host()
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(source[n - 1 - i])
    return Shaped[dtype, n](a.context(), values^)


def copy[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Tensor[dtype, LayoutType]:
    """An independent copy of `a`, on `a`'s device. `numpy.copy`.

    `Tensor` is `Movable` and not `Copyable` on purpose -- a tensor is a
    buffer, and copying one should be a decision rather than something
    that happens because a value was passed by value. This is that
    decision, spelled out.
    """
    return Tensor[dtype, LayoutType](a.context(), a.layout, a.to_host())


def vstack[
    dtype: DType, rows_a: Int, rows_b: Int, cols: Int
](
    a: Shaped[dtype, rows_a, cols], b: Shaped[dtype, rows_b, cols]
) raises -> Shaped[dtype, rows_a + rows_b, cols]:
    """Two matrices joined along their rows. `numpy.vstack`.

    Row-major storage makes this the concatenating direction: the two
    buffers go back to back with no interleaving.
    """
    var a_values = a.to_host()
    var b_values = b.to_host()
    var values = List[Scalar[dtype]](capacity=(rows_a + rows_b) * cols)
    for i in range(rows_a * cols):
        values.append(a_values[i])
    for i in range(rows_b * cols):
        values.append(b_values[i])
    return Shaped[dtype, rows_a + rows_b, cols](a.context(), values^)


def hstack[
    dtype: DType, rows: Int, cols_a: Int, cols_b: Int
](
    a: Shaped[dtype, rows, cols_a], b: Shaped[dtype, rows, cols_b]
) raises -> Shaped[dtype, rows, cols_a + cols_b]:
    """Two matrices joined along their columns. `numpy.hstack`."""
    var a_values = a.to_host()
    var b_values = b.to_host()
    var values = List[Scalar[dtype]](length=rows * (cols_a + cols_b), fill=0)
    for r in range(rows):
        for c in range(cols_a):
            values[r * (cols_a + cols_b) + c] = a_values[r * cols_a + c]
        for c in range(cols_b):
            values[r * (cols_a + cols_b) + cols_a + c] = b_values[
                r * cols_b + c
            ]
    return Shaped[dtype, rows, cols_a + cols_b](a.context(), values^)


def _format_tensor[
    dtype: DType, LayoutType: TensorLayout
](
    a: Tensor[dtype, LayoutType],
    precision: Int,
    threshold: Int,
    edge_items: Int,
) raises -> String:
    var n = a.size()
    var values = a.to_host()
    var out = String("[")
    if n <= threshold:
        for i in range(n):
            out += _format_one(values[i], precision)
            if i != n - 1:
                out += ", "
    else:
        for i in range(edge_items):
            out += _format_one(values[i], precision)
            out += ", "
        out += "..."
        for i in range(n - edge_items, n):
            out += ", "
            out += _format_one(values[i], precision)
    out += "]"
    return out^


def _round_to[dtype: DType](x: Scalar[dtype], precision: Int) -> Scalar[dtype]:
    """Round `x` to `precision` decimal digits, half-away-from-zero.

    An approximation, not `Python`'s exact `%.4f` truncation -- the
    rounded value is still a `dtype` float, and its default `String`
    conversion (shortest round-trippable representation) occasionally
    shows one digit more or fewer than `precision` when the rounded value
    itself isn't exactly representable in binary. Good enough for a
    convenience printer; not a claim of exact fixed-point formatting.
    """
    var scale = Scalar[dtype](10.0) ** Scalar[dtype](precision)
    var sign = Scalar[dtype](1) if x >= 0 else Scalar[dtype](-1)
    var shifted = x * scale + sign * Scalar[dtype](0.5)
    return Scalar[dtype](Int64(shifted)) / scale


def _format_one[dtype: DType](x: Scalar[dtype], precision: Int) -> String:
    """One element, rounded for display.

    Rounding is only meaningful for a float; an integer or boolean tensor
    prints its elements as they are. Without the split, `print` on a
    `Shaped[DType.int32, ...]` was a *compile* error rather than a missing
    feature, because `_round_to`'s `10.0 ** precision` requires a
    floating-point SIMD.
    """
    comptime if dtype.is_floating_point():
        return String(_round_to(x, precision))
    else:
        return String(x)


# --------------------------------------------------------------------------
# The seam between the two array types
# --------------------------------------------------------------------------
#
# `Tensor` is the array of `numax.core`, `numax.stats` and `numax.io`: heap
# storage on a device, a shape in its type. `Array[T, n]` is the array of
# `numax.linalg`, `numax.signal`, `numax.interpolate` and `numax.optimize`:
# comptime-sized, register-resident, and generic over the `FloatLike`
# conformer -- which is what makes `cholesky` differentiable at `Dual` and
# launchable inside a GPU thread.
#
# Both are right for their half of the library, and the pair below is how a
# program crosses between them: load a matrix with `numax.io.numpy.load`,
# `to_array` it, factor it, `to_tensor` the result, save it.


def to_array[
    T: FloatLike, dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Array[T, LayoutType.static_product]:
    """`a`'s elements as an `Array` of `T`, row-major.

    The lift into the conformer layer: `numax.linalg`'s matrices and
    vectors, `numax.signal`'s sequences and `numax.interpolate`'s nodes are
    all `Array[T, n]`. `T` is whatever the caller wants the result computed
    at -- `Plain` for a plain answer, `Dual` for a derivative, `Compensated`
    for precision -- and each element arrives through `T.constant`, so a
    `Dual` lifted this way carries a zero derivative until the caller seeds
    one.

    One definition covers every rank, since an `Array` is flat and only the
    element count matters: a rank-1 `n` and a square `n*n` both land where
    `numax.linalg` expects them, and a rank-3 tensor lifts too.

    The shape has to be compile-time -- an `Array`'s length is a parameter,
    so there is nothing to read a run-time extent into. Name the shape with
    `static_view` first.
    """
    if not LayoutType.all_dims_known:
        raise Error("shape is not compile-time; name it with static_view")
    comptime n = LayoutType.static_product
    var values = a.to_host()
    var out = Array[T, n](fill=T.constant(0.0))
    for i in range(n):
        out[i] = T.constant(Float64(values[i]))
    return out^


def to_tensor[
    dtype: DType, *dims: Int
](
    a: Array[Plain[dtype], _LayoutOf[*dims].static_product],
    ctx: Optional[DeviceContext] = None,
) raises -> Shaped[dtype, *dims]:
    """A `Tensor` of the named shape holding `a`'s elements, row-major.

    The way back down from the conformer layer, and `Plain`-only on
    purpose: `FloatLike` can build any conformer from a `Float64`
    (`T.constant`) but offers no way to read one back out, and there is no
    single right answer for what a `Dual` or an `Interval` would even mean
    as a tensor element. Take `.value` or `.lo`/`.hi` first, then lower.

    The shape is named rather than inferred because an `Array` is flat: an
    `Array[Plain[dtype], 4]` is as good a 2x2 as it is a rank-1 of four, and
    only the caller knows which was meant.
    """
    comptime n = _LayoutOf[*dims].static_product
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(a[i].v)
    return Shaped[dtype, *dims](_context(ctx), values^)
