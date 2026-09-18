"""One launch policy for the NumPy-named surface over `Tensor`.

The surface modules -- `numax.core.elementwise`, `numax.core.ops` and
`numax.core.logic` -- name the operations. This module owns the single
question they all have to answer: given a tensor, a per-element `op` and a
`gpu: Bool`, where does the work run and how is it driven. It is private
because no *caller* of numax should have to know the answer.

Four of its pieces are shared past `numax.core`, and deliberately:
`numax.stats.statistics` takes `_check_device`, `_notice`, `_flat` and
`_dense` so that its monoid reductions answer a host/device mismatch the
same way the elementwise surface does -- one policy, written once, rather
than a second one that drifts. What does not leave is the drivers.

The drivers are `unary`, `unary_to`, `binary`, `binary_to`,
`binary_scalar`, `broadcast_binary` and `broadcast_binary_to`. Each builds
its destination with `Tensor._uninitialized` -- sound because the launch
that follows writes every element -- flattens both operands to rank 1 over
their own buffer pointers, and hands one capturing body to `_launch`.

`_launch` is the whole policy:

- `gpu=True` runs `max.algorithm.elementwise[target="gpu"]`, one element per
  thread. `elementwise` computes its grid from a run-time `Coord`, so a
  run-time-shaped `Dynamic` launches exactly as a `Static` does; that is
  why there is one signature per driver and no static/runtime doubling.
- `gpu=False` below `_THREADED_FROM` elements is a serial SIMD loop. Thread
  dispatch loses on small inputs (`numax/core/tensor.mojo` measures the
  crossover near a quarter of a million elements), and the NumPy-named
  surface is mostly called on small tensors.
- `gpu=False` at or above it is `elementwise[target="cpu"]`, which is
  `map_threaded`'s walk: every core, at the native SIMD width.

`elementwise` rather than `enqueue_function` for the reason
`.cursor/rules/findings.mdc` records: a kernel carrying a layout `where`
clause cannot be named inside `enqueue_function` from generic code.

**The device gate.** A driver -- or any routine that borrows
`_check_device`/`_notice` -- asked for a target the tensor's memory does
not live on runs the old host walk -- `to_host`, a scalar loop, rebuild --
and prints one line to `stderr` naming the spelling that would not have.
That is a deliberate fallback rather than a raise: `exp(a)` on a GPU tensor
keeps working and keeps returning the right answer, which is what it did
before this module existed. The fallback is proven by
`examples/advanced/unified_tensor_gpu.mojo` and deliberately has no test:
naming `gpu=True` compiles a device kernel, which stops with `Unknown GPU
architecture detected` on the GPU-less runners CI uses.

No driver calls `ctx.synchronize()`. The launches are stream-ordered and
`Tensor.to_host` maps, which orders against them.
"""

from layout import Coord, TileTensor, coord_to_index_list
from layout.tile_layout import row_major, TensorLayout
from layout.tile_tensor import DefaultEngine
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.io import FileDescriptor
from std.sys.info import simd_width_of
from std.utils import IndexList

from .tensorlike import TensorLike, is_row_major
from .array import (
    Dynamic,
    Tensor,
    _dyn_shape_from,
    _extents_of,
    _stretch_strides,
    _strides_of,
    broadcast_shapes,
)


comptime _THREADED_FROM = 1 << 16
"""Element count at which the host path switches from a serial SIMD loop to
`elementwise[target="cpu"]`.

Provisional: thread dispatch is measured to lose below roughly a quarter of
a million elements in `numax/core/tensor.mojo`, and this is the conservative
power of two below that. Retuned against `bench/bench_core_surface.mojo`.
"""


@always_inline
def _target[gpu: Bool]() -> StaticString:
    """`elementwise`'s target string for numax's `gpu: Bool` parameter."""
    return "gpu" if gpu else "cpu"


@always_inline
def _width[dtype: DType, gpu: Bool]() -> Int:
    """Native SIMD width on the host, one element per thread on the device
    -- the same choice `numax.core.tensor.map` documents measuring."""
    comptime if gpu:
        return 1
    else:
        return simd_width_of[dtype]()


comptime _FlatLayout = type_of(row_major(Coord(0)))
"""The rank-1 run-time layout every operand is flattened to."""

comptime _FlatIn[dtype: DType] = TileTensor[
    dtype,
    _FlatLayout,
    ImmutAnyOrigin,
    Engine=DefaultEngine[element_width=1],
]
"""A read-only flat view. Immutable because a driver takes its input
borrowed, and `DeviceBuffer`'s tile constructor borrows the buffer's own
origin -- an immutable `self` cannot produce a `MutAnyOrigin` view."""

comptime _FlatOut[dtype: DType] = TileTensor[
    dtype, _FlatLayout, MutAnyOrigin, Engine=DefaultEngine[element_width=1]
]
"""A writable flat view, for a destination the driver owns."""


def _require_contiguous[T: TensorLike](a: T) raises:
    """Raise unless `a`'s strides are the row-major ones, at run time.

    `is_row_major[T]` settles this at compile time for a static layout. For
    a run-time layout it only checks the stride *types*, which every layout
    `numax` builds satisfies, so a `View` over a strided run-time tile
    would slip through and be flattened wrong; this is the check that
    stops it, O(rank) per launch.
    """
    comptime if T.LayoutType.all_dims_known:
        return
    var expected = 1
    comptime for k in range(T.rank):
        comptime d = T.rank - 1 - k
        if a.dim_at(d) > 1 and a.stride_at(d) != expected:
            raise Error(
                "a strided view cannot be flattened: axis ",
                d,
                " has stride ",
                a.stride_at(d),
                ", expected ",
                expected,
            )
        expected *= a.dim_at(d)


def _flat[
    T: TensorLike, *, dtype: DType = T.dtype
](a: T) raises -> _FlatIn[dtype] where is_row_major[T]:
    """`a`'s elements as one rank-1 run-time view over the same storage.

    Valid for `Static` and `Dynamic` alike, which is what lets every driver
    have one signature: the flattening is a layout built over the view's
    pointer rather than `coalesce()`, which would need every extent at
    compile time. Immutable, because `a` is borrowed here.

    `dtype` is `T.dtype` unless a caller names it, which the two-operand
    drivers do: with `a: A` and `b: B` under `where A.dtype == B.dtype`,
    the checker still types `b`'s lanes as `Scalar[B.dtype]`, so the second
    view is built at `A.dtype` through a same-width pointer bitcast.
    """
    _require_contiguous(a)
    var v = a.view()
    return _FlatIn[dtype](
        ptr=v.ptr.unsafe_bitcast[Scalar[dtype]]().unsafe_origin_cast[
            ImmutAnyOrigin
        ](),
        layout=row_major(Coord(a.size())),
    )


def _flat_unchecked[T: TensorLike](a: T) raises -> _FlatIn[T.dtype]:
    """`_flat` without the compile-time clause, for a caller generic over a
    layout it cannot prove row-major but built row-major itself -- a
    `Tensor` from a numax factory at a symbolic rank. The run-time check
    still refuses a strided view; what is lost is only the compile-time
    refusal."""
    _require_contiguous(a)
    var v = a.view()
    return _FlatIn[T.dtype](
        ptr=v.ptr.unsafe_origin_cast[ImmutAnyOrigin](),
        layout=row_major(Coord(a.size())),
    )


def _flat_out[T: TensorLike](mut a: T) raises -> _FlatOut[T.dtype]:
    """`_flat` for a destination, which the driver owns and may write.

    No `is_row_major` clause: every destination is a tensor a driver just
    allocated row-major, at a rank that is sometimes a symbolic
    `_BroadcastRank` the prover cannot tabulate. The run-time check keeps
    the guarantee.
    """
    _require_contiguous(a)
    var v = a.view()
    return _FlatOut[T.dtype](
        ptr=v.ptr.unsafe_origin_cast[MutAnyOrigin](),
        layout=row_major(Coord(a.size())),
    )


comptime _DenseIn[dtype: DType, LayoutType: TensorLayout] = TileTensor[
    dtype, LayoutType, ImmutAnyOrigin, Engine=DefaultEngine[element_width=1]
]
"""A read-only view at the tensor's own rank and layout."""


def _dense[T: TensorLike](a: T) raises -> _DenseIn[T.dtype, T.LayoutType]:
    """`a`'s own layout as a read-only, origin-erased view.

    The tracked `view()` already lets a borrowed argument be viewed; this
    is the spelling for a kernel whose parameter is fixed at
    `ImmutAnyOrigin`, at the tensor's rank rather than `_flat`'s rank 1 --
    what the axis reductions want, since they need the extents. Strides
    are kept, so a strided `View` reads correctly here.
    """
    var v = a.view()
    return _DenseIn[T.dtype, T.LayoutType](
        ptr=v.ptr.unsafe_origin_cast[ImmutAnyOrigin](), layout=v.layout
    )


def _launch[
    FuncType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    //,
    gpu: Bool,
    lanes: Int,
](body: FuncType, n: Int, ctx: DeviceContext) raises:
    """Run `body` over `n` elements on the target `gpu` names.

    The three-way policy the module docstring describes. `body` captures its
    operands **by value** (`{var xs, var ys}`): an `imm` capture list
    compiles and silently writes nothing on the device, recorded in
    `findings.mdc`.
    """
    comptime if gpu:
        elementwise[simd_width=lanes, target=_target[gpu]()](
            body, Coord(n), ctx
        )
    else:
        if n >= _THREADED_FROM:
            elementwise[simd_width=lanes, target=_target[gpu]()](
                body, Coord(n), ctx
            )
        else:
            var vec_n = (n // lanes) * lanes
            var i = 0
            while i < vec_n:
                body[lanes](Coord(i))
                i += lanes
            for j in range(vec_n, n):
                body[1](Coord(j))


def _check_device[T: TensorLike, gpu: Bool](a: T) -> Bool:
    """Whether `a`'s storage is where a `gpu`-targeted launch needs it.

    `host_addressable` is true exactly on a CPU context, so the launch and
    the residency agree when the two disagree as booleans.
    """
    return gpu != a.on_host()


def _notice[gpu: Bool](name: StaticString):
    """One line on `stderr` saying the call fell back to the host walk.

    Not a raise: the answer is still correct, only slow, and a NumPy-named
    routine that stops working because its tensor moved to a device would be
    worse than one that says so.
    """
    comptime if gpu:
        print(
            "numax: ",
            name,
            (
                " ran on the host because gpu=True was asked of a tensor on a"
                " CPU context; build the tensor on a GPU context"
            ),
            sep="",
            file=FileDescriptor(2),
        )
    else:
        print(
            "numax: ",
            name,
            " ran on the host because the tensor lives on a GPU context; call ",
            name,
            "[gpu=True]",
            sep="",
            file=FileDescriptor(2),
        )


# The retained host walks. These are what every routine in the surface
# modules did before the drivers existed, kept verbatim so the fallback is
# the old behavior rather than a second implementation of it.


def _host_walk_unary[
    T: TensorLike,
    op: def[w: Int](SIMD[T.dtype, w]) thin -> SIMD[T.dtype, w],
](a: T) raises -> Tensor[T.dtype, T.LayoutType]:
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    var n = a.size()
    var values = a.to_host()
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = op[1](values[i])
    return Tensor[dtype, LayoutType](a.context(), a.view().layout, out^)


def _host_walk_unary_to[
    T: TensorLike,
    out_dtype: DType,
    op: def[w: Int](SIMD[T.dtype, w]) thin -> SIMD[out_dtype, w],
](a: T) raises -> Tensor[out_dtype, T.LayoutType]:
    comptime LayoutType = T.LayoutType
    var n = a.size()
    var values = a.to_host()
    var out = List[Scalar[out_dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = op[1](values[i])
    return Tensor[out_dtype, LayoutType](a.context(), a.view().layout, out^)


def _host_walk_binary[
    T: TensorLike,
    op: def[w: Int](SIMD[T.dtype, w], SIMD[T.dtype, w]) thin -> SIMD[
        T.dtype, w
    ],
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType]:
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    var n = a.size()
    var a_values = a.to_host()
    var b_values = b.to_host()
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = op[1](a_values[i], b_values[i])
    return Tensor[dtype, LayoutType](a.context(), a.view().layout, out^)


def _host_walk_binary_to[
    T: TensorLike,
    out_dtype: DType,
    op: def[w: Int](SIMD[T.dtype, w], SIMD[T.dtype, w]) thin -> SIMD[
        out_dtype, w
    ],
](a: T, b: T) raises -> Tensor[out_dtype, T.LayoutType]:
    comptime LayoutType = T.LayoutType
    var n = a.size()
    var a_values = a.to_host()
    var b_values = b.to_host()
    var out = List[Scalar[out_dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = op[1](a_values[i], b_values[i])
    return Tensor[out_dtype, LayoutType](a.context(), a.view().layout, out^)


def _host_walk_binary_scalar[
    T: TensorLike,
    op: def[w: Int](SIMD[T.dtype, w], SIMD[T.dtype, w]) thin -> SIMD[
        T.dtype, w
    ],
](a: T, s: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType]:
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    var n = a.size()
    var values = a.to_host()
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = op[1](values[i], s)
    return Tensor[dtype, LayoutType](a.context(), a.view().layout, out^)


comptime _BroadcastRank[
    ALayout: TensorLayout, BLayout: TensorLayout
] = ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
"""NumPy's broadcast rank: the wider of the two operands'."""


def _broadcast_plan[
    A: TensorLike, B: TensorLike
](a: A, b: B) raises -> Tuple[
    IndexList[_BroadcastRank[A.LayoutType, B.LayoutType]],
    IndexList[_BroadcastRank[A.LayoutType, B.LayoutType]],
    IndexList[_BroadcastRank[A.LayoutType, B.LayoutType]],
] where (A.dtype == B.dtype):
    """The result extents and each operand's strides against them.

    A stretched axis gets stride 0, so reading through these visits the same
    element for every position along it rather than materializing a copy.
    `IndexList` rather than `List` because a kernel body captures this.
    """
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    comptime rank = _BroadcastRank[ALayout, BLayout]
    var a_extents = _extents_of(a)
    var b_extents = _extents_of(b)
    var extents = broadcast_shapes(a_extents, b_extents)
    var a_strides = _stretch_strides(a_extents, _strides_of(a), rank)
    var b_strides = _stretch_strides(b_extents, _strides_of(b), rank)

    var ext = IndexList[rank]()
    var a_str = IndexList[rank]()
    var b_str = IndexList[rank]()
    for d in range(rank):
        ext[d] = extents[d]
        a_str[d] = a_strides[d]
        b_str[d] = b_strides[d]
    return (ext, a_str, b_str)


def _broadcast_layout[
    rank: Int
](ext: IndexList[rank]) raises -> type_of(
    row_major(_dyn_shape_from[rank](List[Int]()))
):
    """The run-time row-major layout of a broadcast result."""
    var extents = List[Int](capacity=rank)
    for d in range(rank):
        extents.append(ext[d])
    return row_major(_dyn_shape_from[rank](extents))


def _host_walk_broadcast[
    A: TensorLike,
    B: TensorLike,
    op: def[w: Int](SIMD[A.dtype, w], SIMD[A.dtype, w]) thin -> SIMD[
        A.dtype, w
    ],
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype):
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    comptime rank = _BroadcastRank[ALayout, BLayout]
    var ext: IndexList[rank]
    var a_str: IndexList[rank]
    var b_str: IndexList[rank]
    ext, a_str, b_str = _broadcast_plan(a, b)

    var count = 1
    for d in range(rank):
        count *= ext[d]

    var a_values = a.to_host()
    var b_values = b.to_host()
    var out = List[Scalar[dtype]](length=count, fill=0)
    for flat in range(count):
        var rem = flat
        var ai = 0
        var bi = 0
        for k in range(rank):
            var d = rank - 1 - k
            var c = rem % ext[d]
            rem //= ext[d]
            ai += c * a_str[d]
            bi += c * b_str[d]
        out[flat] = op[1](
            a_values[ai],
            rebind[Scalar[type_of(a_values[ai]).dtype]](b_values[bi]),
        )

    var result = Dynamic[dtype, rank]._uninitialized(
        a.context(), _broadcast_layout(ext)
    )
    result.copy_from_host(out)
    return result^


def _host_walk_broadcast_to[
    A: TensorLike,
    B: TensorLike,
    out_dtype: DType,
    op: def[w: Int](SIMD[A.dtype, w], SIMD[A.dtype, w]) thin -> SIMD[
        out_dtype, w
    ],
](a: A, b: B) raises -> Dynamic[
    out_dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype):
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    comptime rank = _BroadcastRank[ALayout, BLayout]
    var ext: IndexList[rank]
    var a_str: IndexList[rank]
    var b_str: IndexList[rank]
    ext, a_str, b_str = _broadcast_plan(a, b)

    var count = 1
    for d in range(rank):
        count *= ext[d]

    var a_values = a.to_host()
    var b_values = b.to_host()
    var out = List[Scalar[out_dtype]](length=count, fill=0)
    for flat in range(count):
        var rem = flat
        var ai = 0
        var bi = 0
        for k in range(rank):
            var d = rank - 1 - k
            var c = rem % ext[d]
            rem //= ext[d]
            ai += c * a_str[d]
            bi += c * b_str[d]
        out[flat] = op[1](
            a_values[ai],
            rebind[Scalar[type_of(a_values[ai]).dtype]](b_values[bi]),
        )

    # `_uninitialized` rather than the zero-filling constructor: the copy
    # below writes every element, and at `DType.bool` the zero fill is the
    # compiler crash `findings.mdc` records.
    var result = Dynamic[out_dtype, rank]._uninitialized(
        a.context(), _broadcast_layout(ext)
    )
    result.copy_from_host(out)
    return result^


# The drivers.


def unary[
    T: TensorLike,
    op: def[w: Int](SIMD[T.dtype, w]) thin -> SIMD[T.dtype, w],
    gpu: Bool,
    name: StaticString,
](a: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """`op` over every element of `a`, into a tensor of the same shape."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    if not _check_device[gpu=gpu](a):
        _notice[gpu](name)
        return _host_walk_unary[T, op](a)

    var ctx = a.context()
    var out = Tensor[dtype, LayoutType]._uninitialized(ctx, a.view().layout)
    var xs = _flat(a)
    var ys = _flat_out(out)

    @always_inline
    def body[width: Int, alignment: Int = 1](coord: Coord) {var xs, var ys}:
        ys.store[width](coord, op[width](xs.load[width](coord)))

    _launch[gpu=gpu, lanes=_width[dtype, gpu]()](body, a.size(), ctx)
    return out^


def unary_to[
    T: TensorLike,
    out_dtype: DType,
    op: def[w: Int](SIMD[T.dtype, w]) thin -> SIMD[out_dtype, w],
    gpu: Bool,
    name: StaticString,
](a: T) raises -> Tensor[out_dtype, T.LayoutType] where is_row_major[T]:
    """`unary` where `op` changes dtype -- `astype`, and the predicates."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    comptime in_dtype = T.dtype
    if not _check_device[gpu=gpu](a):
        _notice[gpu](name)
        return _host_walk_unary_to[T, out_dtype, op](a)

    var ctx = a.context()
    var out = Tensor[out_dtype, LayoutType]._uninitialized(ctx, a.view().layout)
    var xs = _flat(a)
    var ys = _flat_out(out)

    @always_inline
    def body[width: Int, alignment: Int = 1](coord: Coord) {var xs, var ys}:
        ys.store[width](coord, op[width](xs.load[width](coord)))

    _launch[gpu=gpu, lanes=_width[in_dtype, gpu]()](body, a.size(), ctx)
    return out^


def binary[
    T: TensorLike,
    op: def[w: Int](SIMD[T.dtype, w], SIMD[T.dtype, w]) thin -> SIMD[
        T.dtype, w
    ],
    gpu: Bool,
    name: StaticString,
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where is_row_major[T]:
    """`op` over `a` and `b` at one shape."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    if not _check_device[gpu=gpu](a) or not _check_device[gpu=gpu](b):
        _notice[gpu](name)
        return _host_walk_binary[T, op](a, b)

    var ctx = a.context()
    var out = Tensor[dtype, LayoutType]._uninitialized(ctx, a.view().layout)
    var xs = _flat(a)
    var zs = _flat(b)
    var ys = _flat_out(out)

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var zs, var ys}:
        ys.store[width](
            coord, op[width](xs.load[width](coord), zs.load[width](coord))
        )

    _launch[gpu=gpu, lanes=_width[dtype, gpu]()](body, a.size(), ctx)
    return out^


def binary_to[
    T: TensorLike,
    out_dtype: DType,
    op: def[w: Int](SIMD[T.dtype, w], SIMD[T.dtype, w]) thin -> SIMD[
        out_dtype, w
    ],
    gpu: Bool,
    name: StaticString,
](a: T, b: T) raises -> Tensor[out_dtype, T.LayoutType] where is_row_major[T]:
    """`binary` where `op` changes dtype -- the comparisons."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    comptime in_dtype = T.dtype
    if not _check_device[gpu=gpu](a) or not _check_device[gpu=gpu](b):
        _notice[gpu](name)
        return _host_walk_binary_to[T, out_dtype, op](a, b)

    var ctx = a.context()
    var out = Tensor[out_dtype, LayoutType]._uninitialized(ctx, a.view().layout)
    var xs = _flat(a)
    var zs = _flat(b)
    var ys = _flat_out(out)

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var zs, var ys}:
        ys.store[width](
            coord, op[width](xs.load[width](coord), zs.load[width](coord))
        )

    _launch[gpu=gpu, lanes=_width[in_dtype, gpu]()](body, a.size(), ctx)
    return out^


def binary_scalar[
    T: TensorLike,
    op: def[w: Int](SIMD[T.dtype, w], SIMD[T.dtype, w]) thin -> SIMD[
        T.dtype, w
    ],
    gpu: Bool,
    name: StaticString,
](a: T, s: Scalar[T.dtype]) raises -> Tensor[
    T.dtype, T.LayoutType
] where is_row_major[T]:
    """`op` over `a` against one scalar, captured by the body."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    if not _check_device[gpu=gpu](a):
        _notice[gpu](name)
        return _host_walk_binary_scalar[T, op](a, s)

    var ctx = a.context()
    var out = Tensor[dtype, LayoutType]._uninitialized(ctx, a.view().layout)
    var xs = _flat(a)
    var ys = _flat_out(out)

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var ys, var s}:
        ys.store[width](
            coord, op[width](xs.load[width](coord), SIMD[dtype, width](s))
        )

    _launch[gpu=gpu, lanes=_width[dtype, gpu]()](body, a.size(), ctx)
    return out^


def broadcast_binary[
    A: TensorLike,
    B: TensorLike,
    op: def[w: Int](SIMD[A.dtype, w], SIMD[A.dtype, w]) thin -> SIMD[
        A.dtype, w
    ],
    gpu: Bool,
    name: StaticString,
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """`op` over two shapes NumPy would broadcast.

    One element per thread (`simd_width=1`): a stretched axis has stride 0,
    so neighbouring result elements are not neighbouring operand elements
    and a vector load would read the wrong thing. The body rebuilds each
    operand's flat offset from the result's extents and the stretched
    strides, in `Int` arithmetic only -- Metal has no `double`, and a
    run-time `Int` widened to `Float64` is the recorded way to compile a
    kernel that only runs on the host.
    """
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    comptime rank = _BroadcastRank[ALayout, BLayout]
    if not _check_device[gpu=gpu](a) or not _check_device[gpu=gpu](b):
        _notice[gpu](name)
        return _host_walk_broadcast[A, B, op](a, b)

    var ext: IndexList[rank]
    var a_str: IndexList[rank]
    var b_str: IndexList[rank]
    ext, a_str, b_str = _broadcast_plan(a, b)

    var count = 1
    for d in range(rank):
        count *= ext[d]

    var ctx = a.context()
    var out = Dynamic[dtype, rank]._uninitialized(ctx, _broadcast_layout(ext))
    var xs = _flat(a)
    var zs = _flat[dtype=dtype](b)
    var ys = _flat_out(out)

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var zs, var ys, var ext, var a_str, var b_str}:
        var rem = coord_to_index_list(coord)[0]
        var ai = 0
        var bi = 0
        comptime for k in range(rank):
            var d = rank - 1 - k
            var c = rem % ext[d]
            rem //= ext[d]
            ai += c * a_str[d]
            bi += c * b_str[d]
        ys.store[width](
            coord,
            op[width](
                SIMD[dtype, width](xs.load[1](Coord(ai))),
                SIMD[dtype, width](zs.load[1](Coord(bi))),
            ),
        )

    _launch[gpu=gpu, lanes=1](body, count, ctx)
    return out^


def broadcast_binary_to[
    A: TensorLike,
    B: TensorLike,
    out_dtype: DType,
    op: def[w: Int](SIMD[A.dtype, w], SIMD[A.dtype, w]) thin -> SIMD[
        out_dtype, w
    ],
    gpu: Bool,
    name: StaticString,
](a: A, b: B) raises -> Dynamic[
    out_dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """`broadcast_binary` where `op` changes dtype -- the comparisons."""
    comptime dtype = A.dtype
    comptime in_dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    comptime rank = _BroadcastRank[ALayout, BLayout]
    if not _check_device[gpu=gpu](a) or not _check_device[gpu=gpu](b):
        _notice[gpu](name)
        return _host_walk_broadcast_to[A, B, out_dtype, op](a, b)

    var ext: IndexList[rank]
    var a_str: IndexList[rank]
    var b_str: IndexList[rank]
    ext, a_str, b_str = _broadcast_plan(a, b)

    var count = 1
    for d in range(rank):
        count *= ext[d]

    var ctx = a.context()
    var out = Dynamic[out_dtype, rank]._uninitialized(
        ctx, _broadcast_layout(ext)
    )
    var xs = _flat(a)
    var zs = _flat[dtype=in_dtype](b)
    var ys = _flat_out(out)

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var zs, var ys, var ext, var a_str, var b_str}:
        var rem = coord_to_index_list(coord)[0]
        var ai = 0
        var bi = 0
        comptime for k in range(rank):
            var d = rank - 1 - k
            var c = rem % ext[d]
            rem //= ext[d]
            ai += c * a_str[d]
            bi += c * b_str[d]
        ys.store[width](
            coord,
            op[width](
                SIMD[in_dtype, width](xs.load[1](Coord(ai))),
                SIMD[in_dtype, width](zs.load[1](Coord(bi))),
            ),
        )

    _launch[gpu=gpu, lanes=1](body, count, ctx)
    return out^
