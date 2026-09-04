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

**One tensor type, CPU and GPU.** `Tensor[dtype, *dims]` is the only owning
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

from max.gpu.host import DeviceBuffer, DeviceContext
from layout import Coord, TileTensor
from layout.tile_layout import row_major, TensorLayout
from linalg.transpose import transpose as _max_transpose

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


struct Tensor[dtype: DType, *dims: Int](Movable, Writable):
    """An owned `DeviceBuffer` paired with a compile-time row-major layout.

    The one owning tensor type in `numax`, on CPU and GPU alike -- see this
    module's docstring for why the wrapper exists and how the device is
    chosen. `dims` is the shape, exactly as passed to the factory function
    that built this (`Tensor[dtype, 4, 4]` for a 4x4 matrix); `rank` and
    `num_elements` are derived from it at compile time.

    Conforms to `Writable`, so `print(a)` works; `numax.io.print_tensor` is
    the same output with the precision and truncation under the caller's
    control.
    """

    comptime layout = row_major[*Self.dims]()
    comptime LayoutType = type_of(Self.layout)
    comptime rank = Self.dims.__len__()
    comptime num_elements = _product[*Self.dims]()

    var buffer: DeviceBuffer[Self.dtype]
    var host_addressable: Bool
    """Whether this tensor's storage can be read through a plain host
    pointer -- true on a CPU context, false on a discrete GPU.

    Recorded once at construction rather than asked per access, because
    `DeviceContext.api()` builds a `String` every call. A GPU whose memory
    *is* host-addressable (Apple's unified memory) is treated as if it were
    not: the mapping path is correct there too, just slower, and this way
    the fast path is only ever taken where it is unconditionally safe.
    """

    def __init__(out self, ctx: DeviceContext) raises:
        """A zero-filled tensor on `ctx`'s device.

        Zeroing is `ctx.enqueue_memset`, which runs on the device rather
        than staging a host write, so this is the cheap constructor on both
        paths.

        The `synchronize` is not optional: `enqueue_memset` only *queues*
        the fill, and `__getitem__`'s host-pointer fast path does not order
        itself against queued work the way a host mapping does. Without it,
        a read immediately after construction can see unwritten memory
        (caught by `tests/core/test_array.mojo`'s zero-content check).
        """
        self.buffer = ctx.enqueue_create_buffer[Self.dtype](Self.num_elements)
        self.host_addressable = ctx.api() == "cpu"
        ctx.enqueue_memset(self.buffer, Scalar[Self.dtype](0))
        ctx.synchronize()

    def __init__(
        out self, ctx: DeviceContext, var values: List[Scalar[Self.dtype]]
    ) raises:
        """A tensor on `ctx`'s device holding `values`, row-major.

        `values` must have exactly `Self.num_elements` entries; this is the
        escape hatch the host-filled factory functions below funnel through.
        """
        self.buffer = ctx.enqueue_create_buffer[Self.dtype](Self.num_elements)
        self.host_addressable = ctx.api() == "cpu"
        with self.buffer.map_to_host() as host:
            for i in range(Self.num_elements):
                host[i] = values[i]

    def context(self) raises -> DeviceContext:
        """The device this tensor's storage lives on.

        What the derived manipulations use to allocate their result next to
        their input rather than on a device the caller has to name again.
        """
        return self.buffer.context()

    @staticmethod
    def dim[i: Int]() -> Int:
        """The compile-time extent of axis `i`."""
        return Self.dims[i]

    def view(mut self) -> TileTensor[Self.dtype, Self.LayoutType, MutAnyOrigin]:
        """A `TileTensor` view over this tensor's storage.

        The type every `numax.core.tensor` entry point and every MAX kernel
        takes, on CPU and GPU alike. Valid only as long as `self` is alive
        -- pass `self` (or a `mut` reference to it) around, not just the
        value returned here, if the view needs to outlive this call.
        """
        var v: TileTensor[
            Self.dtype, Self.LayoutType, MutAnyOrigin
        ] = TileTensor(self.buffer, Self.layout)
        return v

    def to_host(self) raises -> List[Scalar[Self.dtype]]:
        """A host copy of every element, row-major.

        The bulk read path, and the only one that costs a single mapping
        regardless of device -- see this module's docstring for why
        `DeviceBuffer.unsafe_ptr()` alone is not a safe substitute.
        """
        var out = List[Scalar[Self.dtype]](capacity=Self.num_elements)
        with self.buffer.map_to_host() as host:
            for i in range(Self.num_elements):
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

    def write_to(self, mut writer: Some[Writer]):
        """`print(a)`, at `print_tensor`'s default settings.

        `numax.io.print_tensor` is the same output with `precision`,
        `threshold` and `edge_items` under the caller's control; both go
        through `_format_tensor` below, so the two cannot disagree.

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

        `values` must have exactly `Self.num_elements` entries. On a GPU
        context the write is flushed to the device when the mapping scope
        exits, which is inside this call.
        """
        with self.buffer.map_to_host() as host:
            for i in range(Self.num_elements):
                host[i] = values[i]


def zeros[
    dtype: DType, *dims: Int
](ctx: DeviceContext) raises -> Tensor[dtype, *dims]:
    """A new tensor of the given compile-time shape on `ctx`'s device,
    filled with `0`."""
    return Tensor[dtype, *dims](ctx)


def ones[
    dtype: DType, *dims: Int
](ctx: DeviceContext) raises -> Tensor[dtype, *dims]:
    """A new tensor of the given compile-time shape on `ctx`'s device,
    filled with `1`."""
    return full[dtype, *dims](ctx, 1)


def full[
    dtype: DType, *dims: Int
](ctx: DeviceContext, fill_value: Scalar[dtype]) raises -> Tensor[dtype, *dims]:
    """A new tensor of the given compile-time shape on `ctx`'s device,
    filled with `fill_value`.

    The fill is `enqueue_memset`, so it runs on the device rather than
    staging a host write -- the same reason `zeros` is cheap on both paths.
    """
    var result = Tensor[dtype, *dims](ctx)
    ctx.enqueue_memset(result.buffer, fill_value)
    ctx.synchronize()
    return result^


def empty[
    dtype: DType, *dims: Int
](ctx: DeviceContext) raises -> Tensor[dtype, *dims]:
    """A new tensor of the given compile-time shape, its contents unspecified.

    Unlike NumPy's `empty`, this zero-initializes rather than truly leaving
    the memory uninitialized -- `numax` favors a memory-safe default over
    the small allocation-time saving, matching every other factory function
    here. Callers that write every element before reading (the usual reason
    to reach for `empty` at all) pay nothing extra in practice.
    """
    return Tensor[dtype, *dims](ctx)


def eye[dtype: DType, n: Int](ctx: DeviceContext) raises -> Tensor[dtype, n, n]:
    """The `n`x`n` identity matrix."""
    var values = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        values[i * n + i] = 1
    return Tensor[dtype, n, n](ctx, values^)


def linspace[
    dtype: DType, num: Int
](
    ctx: DeviceContext, start: Scalar[dtype], stop: Scalar[dtype]
) raises -> Tensor[dtype, num]:
    """`num` evenly spaced values from `start` to `stop`, inclusive of both.

    Matches `numpy.linspace`'s default `endpoint=True`. `num == 1` returns
    `[start]`, avoiding a division by zero in the step computation.
    """
    var values = List[Scalar[dtype]](capacity=num)
    comptime if num == 1:
        values.append(start)
    else:
        var step = (stop - start) / Scalar[dtype](num - 1)
        for i in range(num):
            values.append(start + Scalar[dtype](i) * step)
    return Tensor[dtype, num](ctx, values^)


def logspace[
    dtype: DType, num: Int
](
    ctx: DeviceContext,
    start: Scalar[dtype],
    stop: Scalar[dtype],
    base: Scalar[dtype] = 10,
) raises -> Tensor[dtype, num]:
    """`num` values evenly spaced on a log scale: `base**x` for `x` in
    `linspace(start, stop, num)`. Matches `numpy.logspace`'s defaults."""
    var values = List[Scalar[dtype]](capacity=num)
    comptime if num == 1:
        values.append(base**start)
    else:
        var step = (stop - start) / Scalar[dtype](num - 1)
        for i in range(num):
            values.append(base ** (start + Scalar[dtype](i) * step))
    return Tensor[dtype, num](ctx, values^)


def arange[
    dtype: DType, num: Int
](
    ctx: DeviceContext, start: Scalar[dtype] = 0, step: Scalar[dtype] = 1
) raises -> Tensor[dtype, num]:
    """`num` values starting at `start`, spaced by `step`.

    `numpy.arange` takes a `stop` and derives the count from it, which makes
    the output extent depend on runtime values; this module's shapes are
    comptime, so the count is the parameter and `stop` is implied
    (`start + num*step`). `numax.core.array.linspace` is the one to reach for
    when the endpoints are what matter.
    """
    var values = List[Scalar[dtype]](capacity=num)
    for i in range(num):
        values.append(start + Scalar[dtype](i) * step)
    return Tensor[dtype, num](ctx, values^)


def zeros_like[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[dtype, *dims]:
    """A new zero-filled tensor with `a`'s dtype, shape and device."""
    return zeros[dtype, *dims](a.context())


def ones_like[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[dtype, *dims]:
    """A new one-filled tensor with `a`'s dtype, shape and device."""
    return ones[dtype, *dims](a.context())


def full_like[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], fill_value: Scalar[dtype]) raises -> Tensor[
    dtype, *dims
]:
    """A new `fill_value`-filled tensor with `a`'s dtype, shape and device."""
    return full[dtype, *dims](a.context(), fill_value)


def empty_like[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[dtype, *dims]:
    """A new tensor with `a`'s dtype, shape and device; see `empty`'s own
    docstring for why this zero-initializes rather than leaving memory
    uninitialized.
    """
    return empty[dtype, *dims](a.context())


def transpose[
    dtype: DType, rows: Int, cols: Int
](mut a: Tensor[dtype, rows, cols]) raises -> Tensor[dtype, cols, rows]:
    """An owned-copy transpose of a 2D tensor, on `a`'s own device.

    The permutation itself is `linalg.transpose` -- MAX's own kernel, which
    dispatches to SIMD-shuffle tile kernels and runs on either device. numax
    only allocates the destination and names the axis permutation; an
    earlier version here walked the elements one at a time on the host.

    Distinct from `TileTensor.transpose()`, which returns a zero-copy view
    with every axis reversed over the *same* memory -- this allocates a new
    buffer, for when the result needs its own storage (e.g. to outlive the
    source, or to feed something that wants a plain `Tensor`).
    """
    var ctx = a.context()
    var result = Tensor[dtype, cols, rows](ctx)
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
](a: Tensor[dtype, 1, n]) raises -> Tensor[dtype, n]:
    """Drop a size-1 leading axis: `(1, n) -> (n,)`."""
    return Tensor[dtype, n](a.context(), a.to_host())


def squeeze[
    dtype: DType, n: Int
](a: Tensor[dtype, n, 1]) raises -> Tensor[dtype, n]:
    """Drop a size-1 trailing axis: `(n, 1) -> (n,)`."""
    return Tensor[dtype, n](a.context(), a.to_host())


def stack[
    dtype: DType, n: Int
](a: Tensor[dtype, n], b: Tensor[dtype, n]) raises -> Tensor[dtype, 2, n]:
    """Stack two same-shaped rank-1 tensors along a new leading axis
    (`axis=0`): `ys[0, :] = a`, `ys[1, :] = b`.

    `axis=1` stacking is not provided -- see this module's own docstring.
    """
    var a_values = a.to_host()
    var b_values = b.to_host()
    var values = List[Scalar[dtype]](capacity=2 * n)
    for i in range(n):
        values.append(a_values[i])
    for i in range(n):
        values.append(b_values[i])
    return Tensor[dtype, 2, n](a.context(), values^)


def reshape[
    dtype: DType, n: Int, rows: Int, cols: Int
](a: Tensor[dtype, n]) raises -> Tensor[dtype, rows, cols] where (
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
    return Tensor[dtype, rows, cols](a.context(), a.to_host())


def reshape[
    dtype: DType, n: Int, d0: Int, d1: Int, d2: Int
](a: Tensor[dtype, n]) raises -> Tensor[dtype, d0, d1, d2] where (
    d0 * d1 * d2 == n
):
    """A rank-3 copy of a rank-1 tensor, in row-major order. See the rank-2
    overload above for why the ranks are spelled out."""
    return Tensor[dtype, d0, d1, d2](a.context(), a.to_host())


def ravel[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[dtype, _product[*dims]()]:
    """A rank-1 copy in row-major order -- the inverse of `reshape`.

    `Tensor` owns its storage and Mojo will not let a field be moved out of
    a value that still has to be destroyed, so this copies rather than
    retypes in place. Every manipulation in this module copies for the same
    reason; `TileTensor.reshape()` is the zero-copy view when the result
    does not need to outlive its source.
    """
    return Tensor[dtype, _product[*dims]()](a.context(), a.to_host())


def concatenate[
    dtype: DType, n: Int, m: Int
](a: Tensor[dtype, n], b: Tensor[dtype, m]) raises -> Tensor[dtype, n + m]:
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
    return Tensor[dtype, n + m](a.context(), values^)


def split[
    dtype: DType, n: Int, at: Int
](a: Tensor[dtype, n]) raises -> Tuple[
    Tensor[dtype, at], Tensor[dtype, n - at]
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
        Tensor[dtype, at](ctx, head^),
        Tensor[dtype, n - at](ctx, tail^),
    )


def geomspace[
    dtype: DType, num: Int
](
    ctx: DeviceContext, start: Scalar[dtype], stop: Scalar[dtype]
) raises -> Tensor[dtype, num] where dtype.is_floating_point():
    """`num` values spaced evenly on a geometric progression, endpoints
    included. `numpy.geomspace`.

    `start` and `stop` must share a sign and neither may be zero -- a
    geometric progression through zero does not exist. Unchecked, like
    NumPy's own, because the check costs a branch the caller is better
    placed to make.
    """
    var values = List[Scalar[dtype]](capacity=num)
    comptime if num == 1:
        values.append(start)
    else:
        var ratio = (stop / start) ** (Float64(1) / Float64(num - 1))
        var current = start
        for i in range(num):
            values.append(current)
            current = current * Scalar[dtype](ratio)
    return Tensor[dtype, num](ctx, values^)


def identity[
    dtype: DType, n: Int
](ctx: DeviceContext) raises -> Tensor[dtype, n, n]:
    """The `n`x`n` identity matrix. `numpy.identity`.

    Same result as `eye`; both names exist in NumPy and a caller reaching
    for one should not have to discover the other.
    """
    return eye[dtype, n](ctx)


def diag[
    dtype: DType, n: Int
](a: Tensor[dtype, n]) raises -> Tensor[dtype, n, n]:
    """A square matrix with `a` on its main diagonal. `numpy.diag`.

    The vector-to-matrix direction only; `diagonal` is the inverse.
    """
    var values = List[Scalar[dtype]](length=n * n, fill=0)
    var source = a.to_host()
    for i in range(n):
        values[i * n + i] = source[i]
    return Tensor[dtype, n, n](a.context(), values^)


def diagonal[
    dtype: DType, n: Int
](a: Tensor[dtype, n, n]) raises -> Tensor[dtype, n]:
    """The main diagonal of a square matrix. `numpy.diagonal`."""
    var source = a.to_host()
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(source[i * n + i])
    return Tensor[dtype, n](a.context(), values^)


def diagflat[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    dtype, _product[*dims](), _product[*dims]()
]:
    """`a` flattened onto the diagonal of a square matrix.
    `numpy.diagflat`."""
    comptime n = _product[*dims]()
    var source = a.to_host()
    var values = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        values[i * n + i] = source[i]
    return Tensor[dtype, n, n](a.context(), values^)


def tri[dtype: DType, n: Int](ctx: DeviceContext) raises -> Tensor[dtype, n, n]:
    """An `n`x`n` matrix of ones at and below the diagonal. `numpy.tri`."""
    var values = List[Scalar[dtype]](length=n * n, fill=0)
    for r in range(n):
        for c in range(r + 1):
            values[r * n + c] = 1
    return Tensor[dtype, n, n](ctx, values^)


def tril[
    dtype: DType, n: Int
](a: Tensor[dtype, n, n]) raises -> Tensor[dtype, n, n]:
    """`a` with everything above the diagonal zeroed. `numpy.tril`."""
    var source = a.to_host()
    var values = List[Scalar[dtype]](length=n * n, fill=0)
    for r in range(n):
        for c in range(r + 1):
            values[r * n + c] = source[r * n + c]
    return Tensor[dtype, n, n](a.context(), values^)


def triu[
    dtype: DType, n: Int
](a: Tensor[dtype, n, n]) raises -> Tensor[dtype, n, n]:
    """`a` with everything below the diagonal zeroed. `numpy.triu`."""
    var source = a.to_host()
    var values = List[Scalar[dtype]](length=n * n, fill=0)
    for r in range(n):
        for c in range(r, n):
            values[r * n + c] = source[r * n + c]
    return Tensor[dtype, n, n](a.context(), values^)


def vander[
    dtype: DType, n: Int, cols: Int
](a: Tensor[dtype, n]) raises -> Tensor[
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
    return Tensor[dtype, n, cols](a.context(), values^)


def meshgrid[
    dtype: DType, n: Int, m: Int
](x: Tensor[dtype, n], y: Tensor[dtype, m]) raises -> Tuple[
    Tensor[dtype, m, n], Tensor[dtype, m, n]
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
        Tensor[dtype, m, n](ctx, xx^),
        Tensor[dtype, m, n](ctx, yy^),
    )


def flip[dtype: DType, n: Int](a: Tensor[dtype, n]) raises -> Tensor[dtype, n]:
    """A rank-1 tensor reversed. `numpy.flip` at `axis=0`."""
    var source = a.to_host()
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(source[n - 1 - i])
    return Tensor[dtype, n](a.context(), values^)


def copy[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[dtype, *dims]:
    """An independent copy of `a`, on `a`'s device. `numpy.copy`.

    `Tensor` is `Movable` and not `Copyable` on purpose -- a tensor is a
    buffer, and copying one should be a decision rather than something
    that happens because a value was passed by value. This is that
    decision, spelled out.
    """
    return Tensor[dtype, *dims](a.context(), a.to_host())


def vstack[
    dtype: DType, rows_a: Int, rows_b: Int, cols: Int
](
    a: Tensor[dtype, rows_a, cols], b: Tensor[dtype, rows_b, cols]
) raises -> Tensor[dtype, rows_a + rows_b, cols]:
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
    return Tensor[dtype, rows_a + rows_b, cols](a.context(), values^)


def hstack[
    dtype: DType, rows: Int, cols_a: Int, cols_b: Int
](
    a: Tensor[dtype, rows, cols_a], b: Tensor[dtype, rows, cols_b]
) raises -> Tensor[dtype, rows, cols_a + cols_b]:
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
    return Tensor[dtype, rows, cols_a + cols_b](a.context(), values^)


def _format_tensor[
    dtype: DType, *dims: Int
](
    a: Tensor[dtype, *dims], precision: Int, threshold: Int, edge_items: Int
) raises -> String:
    comptime n = Tensor[dtype, *dims].num_elements
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
    return String(_round_to(x, precision))
