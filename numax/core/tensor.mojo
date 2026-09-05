"""Drive any `FloatLike` kernel across a `TileTensor`, on CPU or GPU.

`numax`'s kernels (`gaussian`, `sigmoid`, `erf`, ...) are written once against
`FloatLike` and get plain SIMD, autodiff, or extra precision for free by the
type they're called with. That composability was, until now, only wired up
one element (or one SIMD lane group) at a time -- every example had to hand-
roll its own walk over memory. `TileTensor` (from MAX's `layout` package) is
the walk: it already knows how to lay out and address a buffer on the CPU or
in GPU global memory, so a `FloatLike` kernel only needs a way to read one
element out of a tensor, and write one element back.

Every function below takes exactly *one* higher-order parameter -- `step`
for the elementwise family (`map`), `combine` for the folding family
(`reduce`/`reduce_rows`/`broadcast_op_rows`) -- rather than the
`kernel`/`wrap`/`unwrap` trio an earlier version of this module used.
Composing a `FloatLike` kernel with the raw `SIMD` in/out conversion is now
the caller's job, in one small function, the same way `gaussian_step` below
does it; `numax.core.tensor` itself only ever sees raw `SIMD`.

**`map`, `reduce_rows`, and `broadcast_op_rows` run on either CPU or GPU,
picked with a `gpu: Bool` compile-time parameter, same function either way.**
`reduce` doesn't get this treatment -- see its own docstring and
`reduce_block_gpu` for why a full reduction and a GPU block's *partial*
reduction aren't the same operation behind a flag, only two steps of one.

Two things make the `gpu` parameter possible at all:

- `comptime if gpu: ... else: ...` picks one of two genuinely different
  algorithms at compile time, with the untaken branch fully eliminated --
  GPU parallelism comes from thread count (`width` elements per thread
  starting at `global_idx.x * width`, one element with the default
  `width=1`), CPU parallelism (such as it is here) comes from SIMD
  registers (`TileTensor.vectorize()` over `width`-wide groups plus a scalar
  tail) -- so there's no shared loop body to factor out, just a shared
  *signature* and *call site*. `width` means "per thread" on one path and
  "per SIMD register" on the other; raising it above 1 on the GPU path
  measured neutral-to-slightly-worse on both Metal and CUDA, since that
  kernel is already bandwidth-bound (see `map`'s own docstring).
- `step`/`combine` is a `thin` (non-capturing) function, taken as a
  compile-time parameter, for *both* branches -- not just the `gpu=True`
  one. This is stricter than it needs to be for `gpu=False` alone (an
  earlier version of this module let CPU-only `step`/`combine` be a genuine
  capturing closure, since there's no host/device boundary for a capture to
  cross on that path), but it's what makes one shared signature launchable
  through `DeviceContext.enqueue_function` when `gpu=True` (which is
  backend-agnostic: the same `map[gpu=True]` launches on Metal and CUDA
  with no per-backend branch anywhere in this module):
  `enqueue_function` needs the *entire* kernel type resolved before it can
  match runtime arguments against it, so `step`/`combine` has to already be
  bound as part of the function's identity (a compile-time parameter, in
  square brackets) rather than inferred from a later runtime argument --
  confirmed directly: making `step` a runtime argument with its type
  inferred (which does accept capturing closures) type-checks fine standing
  alone, but `enqueue_function` then can't infer that argument's type early
  enough and rejects the call outright, independent of whether the closure
  captures anything. No real call site in `numax` passes a capturing
  closure here today, so this costs nothing in practice.

* `map[gpu=False]` (the default) walks a CPU-backed tensor at native SIMD
  width: give it a `width` (usually `simd_width_of[dtype]()`) and a `step`
  generic over its own width parameter (`def[w: Int](SIMD[dtype, w]) thin ->
  SIMD[dtype, w]`), and it walks `xs.num_elements() - n % width` elements at
  `width` via `TileTensor.vectorize()`, then whatever's left over one at a
  time with the same `step` instantiated at `width=1`. No overlapping
  reads, no dropped remainder.
* `map[gpu=True]` is the body of one GPU thread instead: launch it with
  `ctx.enqueue_function[map[gpu=True, step=...]](xs, ys, grid_dim=...,
  block_dim=...)`, and every thread applies `step` to `width` consecutive
  elements -- one element with the default `width=1`, which is the setting
  that measured fastest on both Metal and CUDA. Raise it only after
  measuring on the target: coarsening changes the required `grid_dim`
  (see `map`'s docstring) and bought nothing on either.

`reduce`/`reduce_block_gpu` fold a tensor down to one value (sum, max, ...)
instead of producing another tensor the same shape. `reduce` is a plain CPU
loop; `reduce_block_gpu` is one thread block's worth of a shared-memory tree
reduction, writing one partial value per block -- combine the (small)
partials buffer with `reduce` again, CPU-side, for the final scalar.
`add_combine`/`max_combine` are the two ready-made `combine` functions most
callers want, pre-composed for `Plain[dtype]` from `add_op`/`max_op` (both
`FloatLike`-generic, in case a future caller wants to fold `Dual` or
`Compensated` values directly instead).

`reduce_rows`/`broadcast_op_rows` are the two 2D building blocks a row-wise
op (like softmax) needs on top of the rank-1 primitives above:
`reduce_rows` folds each row of a 2D `TileTensor` down to one value (one
thread per row on GPU -- see its docstring for why that's the deliberately
simple choice here), and `broadcast_op_rows` combines every element of a 2D
tensor with its row's value (one thread per element on GPU). The purely
elementwise part of a row-wise kernel (e.g. softmax's `exp`) doesn't need
either of these -- pass the same 2D tensor straight to `map`, which
coalesces it to a flat walk internally.

`map` has a second overload taking *two* input tensors and one output,
with a two-argument `step` -- `out[i] = step(lhs[i], rhs[i])`. Overloading
on arity rather than introducing a `zip_map` name works on both paths,
including through `enqueue_function` (confirmed on Metal and CUDA in
`examples/advanced/gaussian_gpu.mojo`), which has to resolve the overload
before it sees a single runtime argument. There is deliberately no
three-input or variadic form: past two tensors the thing to compose is the
*kernel*, not the walk. A `step` computing `exp(-(a*a))*b + c` in one pass beats three
`map` calls that each re-traverse memory, and composing inside `step` is
what makes a `FloatLike` kernel a fused kernel by construction. Two inputs
is simply the point where the operation cannot be expressed inside a
single `step` at all, because it needs a second buffer to read from.

Every function here accepts a `TileTensor` of *any* rank, and how it gets
from that rank down to a walk is what separates the three groups below.

* **Statically shaped and row-major.** `TileTensor.coalesce()` flattens it,
  which is the only form that can be launched on a GPU (the thread count
  has to come from the type) and the only one that vectorizes. Everything
  with a `gpu` parameter is in this group, and a `where` clause
  (`all_dims_known and is_row_major`, exactly `coalesce()`'s own
  requirement) keeps anything else out at compile time.
* **Row-major with run-time extents.** `map`, `map_to`, `zip_to`,
  `map_threaded`, `reduce`, `reduce_axis` and `broadcast_op_axis` each have
  a second overload under the *same name*, selected by a `where` clause
  that is the exact negation, so the two can never be ambiguous. They
  flatten by constructing a rank-1 layout over the same pointer rather than
  by `coalesce()` -- the same addresses in the same order, without needing
  to prove the shape first. No `gpu` parameter, for the reason above.
  `reduce_rows`/`broadcast_op_rows` need no second overload at all: they
  index by coordinate and never flatten, so they already accept either.
* **Not row-major at all** -- a transposed view, or a slice with gaps in
  it. `map_strided` and `reduce_strided` address each element through its
  own strides, so they assume nothing about contiguity, and they are
  generic over `Storage` because a sliced view does not carry the storage
  type the tensor it came from did. Scalar, and one integer division per
  axis per element; compact the view with a copy first if it is walked more
  than once.

numax gains no second tensor type from any of this. The argument is a
`TileTensor` throughout, which is what every MAX kernel already takes, and
the distinction lives in the layout -- which is where MAX put it.
"""

from layout import Coord, TileTensor
from layout.tile_layout import TensorLayout, row_major
from layout.tile_tensor import PointerStorage, TensorStorage
from max.algorithm.functional import elementwise
from max.gpu import AddressSpace, barrier
from max.gpu.host import DeviceContext
from std.gpu import block_idx, global_idx, thread_idx
from std.memory import stack_allocation

from .numeric import FloatLike
from .plain import Plain


def add_op[T: FloatLike](a: T, b: T) -> T:
    """Sum two `FloatLike` values -- the default `reduce`/`reduce_rows` op."""
    return a + b


def max_op[T: FloatLike](a: T, b: T) -> T:
    """The larger of two `FloatLike` values, lane-wise.

    `max(a, b) = (a + b + |a - b|) / 2` -- an identity that only needs
    operations `FloatLike` already requires (`abs`, `__add__`, `__neg__`,
    `constant`, `__truediv__`), so this needs no `max` method on the trait
    itself.
    """
    var diff = a - b
    return (a + b + diff.abs()) / T.constant(2.0)


def add_combine[
    dtype: DType
](a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """`add_op`, pre-composed for `Plain[dtype]` raw `SIMD` in and out --
    the `combine` most `reduce`/`reduce_rows` callers want directly."""
    return add_op(Plain[dtype](a), Plain[dtype](b)).v


def max_combine[
    dtype: DType
](a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """`max_op`, pre-composed for `Plain[dtype]` raw `SIMD` in and out."""
    return max_op(Plain[dtype](a), Plain[dtype](b)).v


def map[
    dtype: DType,
    LayoutType: TensorLayout,
    step: def[w: Int](SIMD[dtype, w]) thin -> SIMD[dtype, w],
    width: Int = 1,
    gpu: Bool = False,
](
    xs: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    ys: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
) where (
    TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
):
    """Apply `step` to every element of `xs`, writing into `ys`.

    `xs`/`ys` can be any rank -- `map` calls `.coalesce()` on both
    internally and walks the result, so a multi-dimensional tensor doesn't
    need flattening at the call site. This is why the `where` clause above
    exists: `coalesce()` itself only works for contiguous, statically-shaped
    storage (`all_dims_known and is_row_major`), so `map` requires the same
    up front instead of failing inside `coalesce()` with a less direct
    error. A rank-1 tensor already satisfies this trivially, so existing
    callers are unaffected.

    `gpu=False` (the default): walks `xs` at native SIMD width, CPU-side.
    `step` is generic over its own width parameter, so `map` can instantiate
    it twice: once at `width` for every full group of `width` elements (via
    `TileTensor.vectorize()`, which reshapes a slice into non-overlapping
    `width`-wide groups), and once at `width=1` for whatever's left over
    when `xs.num_elements()` isn't a multiple of `width`. Pass
    `simd_width_of[dtype]()` as `width` to get the native SIMD width for the
    target.

    `gpu=True`: this is one GPU thread's worth of work instead -- launch
    with `ctx.enqueue_function[map[gpu=True, step=...]](xs, ys,
    grid_dim=..., block_dim=...)`, where `xs`/`ys` are `TileTensor`s built
    from `DeviceBuffer`s. Each thread takes `width` *consecutive* elements
    starting at `global_idx.x * width`, so with the default `width=1` every
    thread handles exactly one element and the launch is
    `grid_dim = ceildiv(n, block_dim)` as before. For `width > 1` (thread
    coarsening) the thread count drops accordingly and the launch becomes
    `grid_dim = ceildiv(ceildiv(n, width), block_dim)` -- getting that wrong
    silently leaves the tail of the tensor unwritten, since nothing checks
    that the grid covers `n`.

    Coarsening is worth knowing about and not worth turning on: measured
    across `width` in `{1, 2, 4, 8}` against `block_dim` in
    `{128, 256, 512, 1024}` at 67M elements
    (`bench/bench_gpu_roofline.mojo`), `width=1` was the fastest or tied
    every time, and `width=8` was consistently a few percent slower. That
    held on both backends measured -- Metal and CUDA -- which is worth more
    than either result alone, since it says the conclusion is about the
    access pattern rather than about one vendor's memory controller. The
    reason is that this kernel is already at ~78-82% of the device's memory
    bandwidth with scalar per-thread accesses: neighbouring threads in a
    warp read neighbouring addresses, which the hardware coalesces into
    wide transactions anyway, so widening each thread's own access has
    nothing left to recover. The parameter is honored rather than ignored
    because it is genuinely a per-target question -- it was silently
    accepted and discarded here before, which was worse than either
    answer -- and because CUDA validation is still outstanding.

    The scalar tail (`for i in range(base, n)`) is an ordinary `if`/loop on
    a *thread* index, which is not a breach of this library's
    fixed-iteration rule: that rule is about SIMD lanes inside a single
    `FloatLike` value, which cannot branch independently of one another,
    whereas separate GPU threads taking different paths is what threads do.
    With `width=1` the tail loop is unreachable dead code that collapses to
    the old bounds check.

    `step` is typically `wrap` -> a `FloatLike` kernel -> `unwrap` composed
    into one function, e.g. for `gaussian` over `Plain`:

    ```mojo
    def gaussian_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
        return gaussian(Plain[dtype, w](x)).v

    map[width = simd_width_of[dtype](), step=gaussian_step](xs, ys)
    ```
    """
    var xs_flat = xs.coalesce()
    var ys_flat = ys.coalesce()
    comptime if gpu:
        var n = xs_flat.num_elements()
        var base = Int(global_idx.x) * width
        if base + width <= n:
            ys_flat.store[width](
                Coord(base), step[width](xs_flat.load[width](Coord(base)))
            )
        else:
            for i in range(base, n):
                ys_flat.store[1](Coord(i), step[1](xs_flat.load[1](Coord(i))))
    else:
        var n = xs_flat.num_elements()
        var vec_n = (n // width) * width
        if vec_n > 0:
            var xs_bulk = xs_flat.slice((0, vec_n)).vectorize[width]()
            var ys_bulk = ys_flat.slice((0, vec_n)).vectorize[width]()
            for i in range(xs_bulk.num_elements()):
                ys_bulk.store[width](
                    Coord(i), step[width](xs_bulk.load[width](Coord(i)))
                )
        for i in range(vec_n, n):
            ys_flat.store[1](Coord(i), step[1](xs_flat.load[1](Coord(i))))


def map_to[
    in_dtype: DType,
    out_dtype: DType,
    LayoutType: TensorLayout,
    step: def[w: Int](SIMD[in_dtype, w]) thin -> SIMD[out_dtype, w],
    width: Int = 1,
    gpu: Bool = False,
](
    xs: TileTensor[
        in_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    ys: TileTensor[
        out_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
) where (
    TileTensor[
        in_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ].all_dims_known
    and TileTensor[
        in_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ].is_row_major
):
    """`map`, but `step` may return a different dtype than it takes.

    The same walk and the same `gpu` parameter as `map`; the only
    difference is that input and output tensors carry different `dtype`s
    over one shared layout. This is what a predicate needs -- `xs > 0`
    reads `float32` and writes `bool` -- and `map` cannot express it,
    because its `step` is `SIMD[dtype, w] -> SIMD[dtype, w]`.

    `numax.core.logic` and `numax.core.array.astype` are built on this.
    """
    var xs_flat = xs.coalesce()
    var ys_flat = ys.coalesce()
    comptime if gpu:
        var n = xs_flat.num_elements()
        var base = Int(global_idx.x) * width
        if base + width <= n:
            ys_flat.store[width](
                Coord(base), step[width](xs_flat.load[width](Coord(base)))
            )
        else:
            for i in range(base, n):
                ys_flat.store[1](Coord(i), step[1](xs_flat.load[1](Coord(i))))
    else:
        var n = xs_flat.num_elements()
        var vec_n = (n // width) * width
        if vec_n > 0:
            var xs_bulk = xs_flat.slice((0, vec_n)).vectorize[width]()
            var ys_bulk = ys_flat.slice((0, vec_n)).vectorize[width]()
            for i in range(xs_bulk.num_elements()):
                ys_bulk.store[width](
                    Coord(i), step[width](xs_bulk.load[width](Coord(i)))
                )
        for i in range(vec_n, n):
            ys_flat.store[1](Coord(i), step[1](xs_flat.load[1](Coord(i))))


def zip_to[
    in_dtype: DType,
    out_dtype: DType,
    LayoutType: TensorLayout,
    step: def[w: Int](SIMD[in_dtype, w], SIMD[in_dtype, w]) thin -> SIMD[
        out_dtype, w
    ],
    width: Int = 1,
    gpu: Bool = False,
](
    lhs: TileTensor[
        in_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    rhs: TileTensor[
        in_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    ys: TileTensor[
        out_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
) where (
    TileTensor[
        in_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ].all_dims_known
    and TileTensor[
        in_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ].is_row_major
):
    """The two-input `map_to`: `out[i] = step(lhs[i], rhs[i])`.

    What an elementwise comparison needs -- two `float32` inputs, one
    `bool` output. Same shape for all three, same reasoning as `map`'s own
    two-input overload for why there is no three-input form.
    """
    var lhs_flat = lhs.coalesce()
    var rhs_flat = rhs.coalesce()
    var ys_flat = ys.coalesce()
    comptime if gpu:
        var n = lhs_flat.num_elements()
        var base = Int(global_idx.x) * width
        if base + width <= n:
            ys_flat.store[width](
                Coord(base),
                step[width](
                    lhs_flat.load[width](Coord(base)),
                    rhs_flat.load[width](Coord(base)),
                ),
            )
        else:
            for i in range(base, n):
                ys_flat.store[1](
                    Coord(i),
                    step[1](
                        lhs_flat.load[1](Coord(i)), rhs_flat.load[1](Coord(i))
                    ),
                )
    else:
        var n = lhs_flat.num_elements()
        var vec_n = (n // width) * width
        if vec_n > 0:
            var lhs_bulk = lhs_flat.slice((0, vec_n)).vectorize[width]()
            var rhs_bulk = rhs_flat.slice((0, vec_n)).vectorize[width]()
            var ys_bulk = ys_flat.slice((0, vec_n)).vectorize[width]()
            for i in range(lhs_bulk.num_elements()):
                ys_bulk.store[width](
                    Coord(i),
                    step[width](
                        lhs_bulk.load[width](Coord(i)),
                        rhs_bulk.load[width](Coord(i)),
                    ),
                )
        for i in range(vec_n, n):
            ys_flat.store[1](
                Coord(i),
                step[1](lhs_flat.load[1](Coord(i)), rhs_flat.load[1](Coord(i))),
            )


def map_threaded[
    dtype: DType,
    LayoutType: TensorLayout,
    step: def[w: Int](SIMD[dtype, w]) thin -> SIMD[dtype, w],
    width: Int = 1,
](
    xs: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    ys: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    ctx: DeviceContext,
) raises where (
    TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
):
    """`map`, but distributed across CPU threads by `max.algorithm.elementwise`.

    Same `step`, same result, same coalescing rules as `map[gpu=False]` --
    the only difference is that the walk is threaded instead of serial. On
    an M3 Pro that is worth about 4-5x above a million elements and nothing
    at all below about 250K, where thread dispatch costs more than the work
    (`bench/bench_elementwise.mojo` has the sweep).

    `ctx` must be a CPU context (`DeviceContext(api="cpu")`) -- a GPU one
    aborts at runtime inside MAX's own dispatch. It is cheap to build
    (~3us) but not free, so hoist it out of a loop rather than constructing
    one per call.

    This is a separate function rather than another flag on `map` for the
    reason `reduce_block_gpu` is separate: `map`'s signature is shared with
    the GPU kernel body that `enqueue_function` launches, and a
    `DeviceContext` argument cannot cross that boundary. Keeping them apart
    means the requirement shows up in this signature instead of failing
    somewhere less obvious.

    One measured behavioral difference, which is MAX's and not `numax`'s:
    the threaded path flushes denormals to zero, so results below the
    smallest normal value of `dtype` (about 1.2e-38 for `float32`) can come
    back as exactly `0` where the serial path returns the denormal.
    Everything at or above that threshold agrees bit for bit.
    """
    var xs_flat = xs.coalesce()
    var ys_flat = ys.coalesce()
    var n = xs_flat.num_elements()

    def body[
        w: Int, alignment: Int = 1
    ](coord: Coord) {imm xs_flat, imm ys_flat}:
        ys_flat.store[w](coord, step[w](xs_flat.load[w](coord)))

    elementwise[simd_width=width, target="cpu"](body, Coord(n), ctx)


def map[
    dtype: DType,
    LayoutType: TensorLayout,
    step: def[w: Int](SIMD[dtype, w], SIMD[dtype, w]) thin -> SIMD[dtype, w],
    width: Int = 1,
    gpu: Bool = False,
](
    lhs: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    rhs: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    out_tensor: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
) where (
    TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
):
    """Apply a two-argument `step` to `lhs` and `rhs`, writing `out_tensor`.

    The binary counterpart of the `map` above -- `out[i] = step(lhs[i],
    rhs[i])` -- and the primitive that makes combining two tensors possible
    at all (`a + b`, `a * b`, `atan2(a, b)`, a fused multiply-add against a
    second buffer). Same `gpu` parameter, same coalescing, same `where`
    clause, same reason `step` is a `thin` compile-time parameter, and the
    same meaning for `width` on each path -- including the `grid_dim =
    ceildiv(ceildiv(n, width), block_dim)` requirement when coarsening the
    GPU path above `width=1`. See the unary `map`'s docstring for why
    `width=1` is the right default there.

    There's deliberately no three-input or variadic version. Past two
    tensors, the thing to compose is the *kernel*, not the walk: a `step`
    that computes `exp(-(a*a)) * b + c` in one pass beats three `map` calls
    that each re-traverse memory, and writing it that way is what makes a
    `FloatLike` kernel a fused kernel by construction. Two inputs is the
    point where a caller genuinely cannot express the operation inside a
    single `step` without a second buffer to read from.

    ```mojo
    map[width = simd_width_of[dtype](), step = add_step[dtype, _]](
        xs, ys, zs
    )
    ```
    """
    var lhs_flat = lhs.coalesce()
    var rhs_flat = rhs.coalesce()
    var out_flat = out_tensor.coalesce()
    comptime if gpu:
        var n = lhs_flat.num_elements()
        var base = Int(global_idx.x) * width
        if base + width <= n:
            out_flat.store[width](
                Coord(base),
                step[width](
                    lhs_flat.load[width](Coord(base)),
                    rhs_flat.load[width](Coord(base)),
                ),
            )
        else:
            for i in range(base, n):
                out_flat.store[1](
                    Coord(i),
                    step[1](
                        lhs_flat.load[1](Coord(i)), rhs_flat.load[1](Coord(i))
                    ),
                )
    else:
        var n = lhs_flat.num_elements()
        var vec_n = (n // width) * width
        if vec_n > 0:
            var lhs_bulk = lhs_flat.slice((0, vec_n)).vectorize[width]()
            var rhs_bulk = rhs_flat.slice((0, vec_n)).vectorize[width]()
            var out_bulk = out_flat.slice((0, vec_n)).vectorize[width]()
            for i in range(lhs_bulk.num_elements()):
                out_bulk.store[width](
                    Coord(i),
                    step[width](
                        lhs_bulk.load[width](Coord(i)),
                        rhs_bulk.load[width](Coord(i)),
                    ),
                )
        for i in range(vec_n, n):
            out_flat.store[1](
                Coord(i),
                step[1](lhs_flat.load[1](Coord(i)), rhs_flat.load[1](Coord(i))),
            )


def add_step[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    """`a + b`, for the binary `map`, as `step=add_step[dtype, _]`.

    `dtype` comes first in the parameter list on purpose: binding it while
    leaving `w` unbound (that's what the `_` is for -- Mojo won't infer an
    unmentioned trailing parameter here) yields a function still generic
    over its width, which is the shape `map`'s `step` requires, since `map`
    instantiates `step` twice, at the vector width and again at `1` for the
    tail. The elementwise counterpart of what `add_combine` does for
    `reduce`.
    """
    return add_op(Plain[dtype, w](a), Plain[dtype, w](b)).v


def mul_step[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    """`a * b`, for the binary `map`, as `step=mul_step[dtype, _]`."""
    return (Plain[dtype, w](a) * Plain[dtype, w](b)).v


def reduce[
    dtype: DType,
    LayoutType: TensorLayout,
    O: Origin,
    combine: def(SIMD[dtype, 1], SIMD[dtype, 1]) thin -> SIMD[dtype, 1],
](
    xs: TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ],
    init: SIMD[dtype, 1],
) -> SIMD[dtype, 1] where (
    TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].is_row_major
):
    """Fold every element of `xs` down to one value with `combine`, CPU-side.

    `init` is `combine`'s identity (`0` for `add_combine`, a very negative
    number for `max_combine`).

    `xs` can be any rank -- like `map`, `reduce` calls `.coalesce()`
    internally before walking it, under the same `where`-enforced
    requirement (`all_dims_known and is_row_major`).

    No `gpu` parameter here, unlike `map`/`reduce_rows`/`broadcast_op_rows`
    -- `reduce_block_gpu`, below, is not this same operation with a
    different backend. It produces one *partial* value per thread block,
    not the final scalar `reduce` returns, so folding it into `reduce`
    behind a flag would make the same name mean two different contracts
    depending on a parameter's value. Call `reduce_block_gpu`, then finish
    with a CPU-side call to `reduce` over its (small) `partials` output.
    """
    var xs_flat = xs.coalesce()
    var acc = init
    for i in range(xs_flat.num_elements()):
        acc = combine(acc, xs_flat.load[1](Coord(i)))
    return acc


def reduce_block_gpu[
    dtype: DType,
    LayoutType: TensorLayout,
    PartialsLayout: TensorLayout,
    block_size: Int,
    combine: def(SIMD[dtype, 1], SIMD[dtype, 1]) thin -> SIMD[dtype, 1],
](
    xs: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    partials: TileTensor[
        dtype,
        PartialsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    identity: SIMD[dtype, 1],
):
    """One thread block's worth of `reduce`: a shared-memory tree reduction.

    Launch with `grid_dim = ceildiv(xs.num_elements(), block_size)`,
    `block_dim = block_size`; `partials` needs at least `grid_dim` elements,
    one written per block. `identity` pads out-of-range threads (the last
    block, if `xs.num_elements()` isn't a multiple of `block_size`) so they
    don't affect the result.

    This is deliberately a *two-pass* reduction, not a single kernel: finish
    it off by calling `reduce` again, CPU-side, over the small `partials`
    buffer once it's copied back to the host. A full single-pass reduction
    (a second, smaller kernel combining blocks, or a grid-wide atomic) is
    more machinery than a handful of blocks' worth of partials justifies.
    """
    var shared = stack_allocation[
        block_size, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var tid = thread_idx.x
    var gid = global_idx.x
    shared[unsafe_offset=tid] = (
        xs.load[1](Coord(gid)) if gid < xs.num_elements() else identity
    )
    barrier()

    var stride = block_size // 2
    while stride > 0:
        if tid < stride:
            shared[unsafe_offset=tid] = combine(
                shared[unsafe_offset=tid], shared[unsafe_offset=tid + stride]
            )
        barrier()
        stride = stride // 2

    if tid == 0:
        partials.store[1](Coord(block_idx.x), shared[unsafe_offset=0])


def reduce_rows[
    dtype: DType,
    RowsLayout: TensorLayout,
    OutLayout: TensorLayout,
    combine: def(SIMD[dtype, 1], SIMD[dtype, 1]) thin -> SIMD[dtype, 1],
    gpu: Bool = False,
](
    xs: TileTensor[
        dtype,
        RowsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    dst: TileTensor[
        dtype,
        OutLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    init: SIMD[dtype, 1],
):
    """Fold each row of a 2D `xs` down to one value in `dst`.

    `dst` needs one element per row of `xs`. This is `reduce`, applied
    independently along `xs`'s first dimension -- the building block a
    row-wise op (row-wise softmax's per-row max and per-row sum) needs on
    top of `reduce`'s single whole-tensor fold.

    `gpu=False` (the default) is a CPU double loop (rows, then each row's
    columns). `gpu=True` is one GPU thread's worth: one thread per row,
    launched with `grid_dim = ceildiv(num_rows, block_size)`, `block_dim =
    block_size` (a plain 1D launch). That trades away intra-row parallelism
    for simplicity -- each thread walks every column of its row in a loop,
    rather than the block-level tree reduction `reduce_block_gpu` uses for
    a single long row. That's the right trade for softmax-shaped workloads
    (many rows, each a modest width); a wide-row workload would want
    `reduce_block_gpu`'s approach per row instead, which 0.1.0 doesn't
    build.
    """
    comptime if gpu:
        var r = global_idx.x
        if r >= Int(xs.dim[0]()):
            return
        var num_cols = Int(xs.dim[1]())
        var acc = init
        for c in range(num_cols):
            acc = combine(acc, xs.load[1](Coord(r, c)))
        dst.store[1](Coord(r), acc)
    else:
        var num_rows = xs.dim[0]()
        var num_cols = Int(xs.dim[1]())
        for r in range(num_rows):
            var acc = init
            for c in range(num_cols):
                acc = combine(acc, xs.load[1](Coord(r, c)))
            dst.store[1](Coord(r), acc)


def reduce_axis[
    dtype: DType,
    XsLayout: TensorLayout,
    OutLayout: TensorLayout,
    combine: def(SIMD[dtype, 1], SIMD[dtype, 1]) thin -> SIMD[dtype, 1],
    axis: Int,
    gpu: Bool = False,
](
    xs: TileTensor[
        dtype,
        XsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    dst: TileTensor[
        dtype,
        OutLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    init: SIMD[dtype, 1],
) where (
    TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and TileTensor[
        dtype, OutLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, OutLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and axis >= 0
    and axis
    < TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].rank
):
    """Fold `xs` along one axis, writing the surviving axes into `dst`.

    `reduce_rows` generalized: it is this function at `axis=1` on a rank-2
    tensor, and `reduce` is roughly this on a rank-1 one. For a rank-3
    `(2, 3, 4)` tensor, `axis=1` produces a `(2, 4)` result -- `dst` needs
    that many elements, in any shape that coalesces to them (the `(2, 4)`
    tensor or a flat 8-element one both work, since both are indexed
    through their coalesced view).

    A row-major tensor splits around any axis into three counts: `outer`
    (the dimensions before `axis`, multiplied together), `length` (the axis
    itself), and `inner` (the dimensions after it). Element `(o, k, i)` then
    sits at flat index `(o*length + k)*inner + i`, so one flat walk handles
    every rank and axis without a per-rank special case. All three counts
    come from the static layout, so no loop bound here depends on data.

    `gpu=True` puts one thread on each surviving `(outer, inner)` position,
    each walking the reduced axis serially -- the same trade `reduce_rows`
    makes, for the same reason. A long reduced axis would want
    `reduce_block_gpu`'s tree instead.
    """
    comptime rank = type_of(xs).rank

    var length = Int(xs.dim[axis]())
    var outer = 1
    comptime for d in range(axis):
        outer *= Int(xs.dim[d]())
    var inner = 1
    comptime for d in range(axis + 1, rank):
        inner *= Int(xs.dim[d]())

    var xs_flat = xs.coalesce()
    var dst_flat = dst.coalesce()

    comptime if gpu:
        var pos = global_idx.x
        if pos >= outer * inner:
            return
        var o = pos // inner
        var i = pos % inner
        var acc = init
        for k in range(length):
            acc = combine(
                acc, xs_flat.load[1](Coord((o * length + k) * inner + i))
            )
        dst_flat.store[1](Coord(pos), acc)
    else:
        for o in range(outer):
            for i in range(inner):
                var acc = init
                for k in range(length):
                    acc = combine(
                        acc,
                        xs_flat.load[1](Coord((o * length + k) * inner + i)),
                    )
                dst_flat.store[1](Coord(o * inner + i), acc)


def broadcast_op_axis[
    dtype: DType,
    XsLayout: TensorLayout,
    ValuesLayout: TensorLayout,
    combine: def(SIMD[dtype, 1], SIMD[dtype, 1]) thin -> SIMD[dtype, 1],
    axis: Int,
    gpu: Bool = False,
](
    xs: TileTensor[
        dtype,
        XsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    values: TileTensor[
        dtype,
        ValuesLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    ys: TileTensor[
        dtype,
        XsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
) where (
    TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and TileTensor[
        dtype,
        ValuesLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ].all_dims_known
    and TileTensor[
        dtype,
        ValuesLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ].is_row_major
    and axis >= 0
    and axis
    < TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].rank
):
    """Combine `xs` with a `values` tensor that is missing one axis.

    `ys[o, k, i] = combine(xs[o, k, i], values[o, i])` -- the inverse shape
    of `reduce_axis`, so the two compose: reduce along an axis, then
    broadcast the result back along it. `broadcast_op_rows` is this at
    `axis=1` on a rank-2 tensor.

    This is broadcasting only in the direction that pairs with a reduction,
    which is the direction every use in this library needs (softmax, per-
    axis normalization, centering). Full NumPy-style broadcasting between
    two arbitrary shapes -- size-1 dimensions stretched independently on
    either side -- is not implemented.
    """
    comptime rank = type_of(xs).rank

    var length = Int(xs.dim[axis]())
    var outer = 1
    comptime for d in range(axis):
        outer *= Int(xs.dim[d]())
    var inner = 1
    comptime for d in range(axis + 1, rank):
        inner *= Int(xs.dim[d]())

    var xs_flat = xs.coalesce()
    var ys_flat = ys.coalesce()
    var values_flat = values.coalesce()

    comptime if gpu:
        var idx = global_idx.x
        if idx >= xs_flat.num_elements():
            return
        var i = idx % inner
        var o = idx // (length * inner)
        var value = values_flat.load[1](Coord(o * inner + i))
        ys_flat.store[1](
            Coord(idx), combine(xs_flat.load[1](Coord(idx)), value)
        )
    else:
        for o in range(outer):
            for i in range(inner):
                var value = values_flat.load[1](Coord(o * inner + i))
                for k in range(length):
                    var idx = (o * length + k) * inner + i
                    ys_flat.store[1](
                        Coord(idx), combine(xs_flat.load[1](Coord(idx)), value)
                    )


def broadcast_op_rows[
    dtype: DType,
    RowsLayout: TensorLayout,
    ValuesLayout: TensorLayout,
    combine: def(SIMD[dtype, 1], SIMD[dtype, 1]) thin -> SIMD[dtype, 1],
    gpu: Bool = False,
](
    xs: TileTensor[
        dtype,
        RowsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    row_values: TileTensor[
        dtype,
        ValuesLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    ys: TileTensor[
        dtype,
        RowsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
):
    """Combine every element of `xs` with its row's value.

    `ys[r, c] = combine(xs[r, c], row_values[r])` -- e.g. a fused
    subtract-then-`exp` to shift each row by its max, or a division to
    divide by each row's sum, the two row-broadcast steps row-wise softmax
    needs around its elementwise `exp`.

    `gpu=False` (the default) is a CPU double loop. `gpu=True` is one GPU
    thread's worth: one thread per element, launched with `grid_dim =
    ceildiv(xs.num_elements(), block_size)`, `block_dim = block_size` -- a
    flat 1D launch over every element, same as `map[gpu=True]`, with the
    row/column split recovered from the flat index.
    """
    comptime if gpu:
        var idx = global_idx.x
        if idx >= xs.num_elements():
            return
        var num_cols = Int(xs.dim[1]())
        var r = idx // num_cols
        var c = idx % num_cols
        var row_val = row_values.load[1](Coord(r))
        ys.store[1](Coord(r, c), combine(xs.load[1](Coord(r, c)), row_val))
    else:
        var num_rows = xs.dim[0]()
        var num_cols = Int(xs.dim[1]())
        for r in range(num_rows):
            var row_val = row_values.load[1](Coord(r))
            for c in range(num_cols):
                ys.store[1](
                    Coord(r, c), combine(xs.load[1](Coord(r, c)), row_val)
                )


# ------------------------------------------------------------------
# Runtime-shape overloads
#
# Everything above requires `all_dims_known`, because it walks a tensor by
# calling `TileTensor.coalesce()` and `coalesce()` itself is constrained to
# statically-shaped storage. That is what a GPU launch needs, and it stays
# exactly as it is.
#
# But a NumPy-shaped caller cannot always supply it: `reshape` to a computed
# shape, a boolean mask, `unique`, or anything whose output extent depends on
# input *values* produces a tensor whose dims are runtime integers.
# `row_major(Coord(3, 4))` builds precisely that -- `is_row_major=True`,
# `all_dims_known=False` -- and `coalesce()` rejects it.
#
# The overloads below take that case. They are the *same* functions under the
# *same* names, selected by a `where` clause that is the exact negation of the
# static one, so the two can never be ambiguous and no existing call site
# changes behavior. numax gains no new tensor type from this: the argument is
# still a `TileTensor`, which is what every MAX kernel takes.
#
# The one thing they do differently is flatten by construction rather than by
# `coalesce()`: a row-major tensor's elements are already contiguous, so a
# rank-1 `row_major(Coord(n))` layout over the same pointer addresses exactly
# the same memory in the same order. That is all `coalesce()` does for a
# row-major input; it just insists on proving the shape at compile time first.
# ------------------------------------------------------------------


def map[
    dtype: DType,
    LayoutType: TensorLayout,
    step: def[w: Int](SIMD[dtype, w]) thin -> SIMD[dtype, w],
    width: Int = 1,
](
    xs: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    ys: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
) where (
    not TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
):
    """`map` for a runtime-shaped tensor -- same contract as the static
    overload above, minus the GPU path.

    There is no `gpu` parameter here on purpose. A kernel launched through
    `enqueue_function` needs its entire type resolved before the launch, and
    a runtime extent is not part of the type; the thread count would also
    have to come from a value the host reads out of the tensor rather than
    from the type. Reach for the static overload when the shape is known,
    which is the case at every GPU call site in this library.

    Same `width` meaning as the static CPU path (elements per SIMD
    register), same non-overlapping bulk-then-tail walk, so a `width` that
    does not divide `num_elements()` is handled rather than rounded away.
    """
    var n = xs.num_elements()
    var xs_flat = TileTensor(xs.ptr_at_offset(Coord(0)), row_major(Coord(n)))
    var ys_flat = TileTensor(ys.ptr_at_offset(Coord(0)), row_major(Coord(n)))
    var vec_n = (n // width) * width
    var i = 0
    while i < vec_n:
        ys_flat.store[width](
            Coord(i), step[width](xs_flat.load[width](Coord(i)))
        )
        i += width
    for j in range(vec_n, n):
        ys_flat.store[1](Coord(j), step[1](xs_flat.load[1](Coord(j))))


def map[
    dtype: DType,
    LayoutType: TensorLayout,
    step: def[w: Int](SIMD[dtype, w], SIMD[dtype, w]) thin -> SIMD[dtype, w],
    width: Int = 1,
](
    lhs: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    rhs: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    out_tensor: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
) where (
    not TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
):
    """The binary `map` for runtime-shaped tensors: `out[i] = step(lhs[i],
    rhs[i])`. Same reasoning as the unary runtime-shape overload above, and
    the same deliberate absence of a three-input form as the static one."""
    var n = lhs.num_elements()
    var lhs_flat = TileTensor(lhs.ptr_at_offset(Coord(0)), row_major(Coord(n)))
    var rhs_flat = TileTensor(rhs.ptr_at_offset(Coord(0)), row_major(Coord(n)))
    var out_flat = TileTensor(
        out_tensor.ptr_at_offset(Coord(0)), row_major(Coord(n))
    )
    var vec_n = (n // width) * width
    var i = 0
    while i < vec_n:
        out_flat.store[width](
            Coord(i),
            step[width](
                lhs_flat.load[width](Coord(i)), rhs_flat.load[width](Coord(i))
            ),
        )
        i += width
    for j in range(vec_n, n):
        out_flat.store[1](
            Coord(j),
            step[1](lhs_flat.load[1](Coord(j)), rhs_flat.load[1](Coord(j))),
        )


def reduce[
    dtype: DType,
    LayoutType: TensorLayout,
    O: Origin,
    combine: def(SIMD[dtype, 1], SIMD[dtype, 1]) thin -> SIMD[dtype, 1],
](
    xs: TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ],
    init: SIMD[dtype, 1],
) -> SIMD[dtype, 1] where (
    not TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].is_row_major
):
    """`reduce` for a runtime-shaped tensor. Scalar and left-to-right, like
    the static overload, so the two agree bit-for-bit on the same values --
    which is what makes the runtime-shape path testable against the static
    one rather than only against itself."""
    var n = xs.num_elements()
    var xs_flat = TileTensor(xs.ptr_at_offset(Coord(0)), row_major(Coord(n)))
    var total = init
    for i in range(n):
        total = combine(total, xs_flat.load[1](Coord(i)))
    return total


def map_to[
    in_dtype: DType,
    out_dtype: DType,
    LayoutType: TensorLayout,
    step: def[w: Int](SIMD[in_dtype, w]) thin -> SIMD[out_dtype, w],
    width: Int = 1,
](
    xs: TileTensor[
        in_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    ys: TileTensor[
        out_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
) where (
    not TileTensor[
        in_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ].all_dims_known
    and TileTensor[
        in_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ].is_row_major
):
    """`map_to` for a runtime-shaped tensor -- the dtype-changing walk a
    predicate needs, at a shape the compiler cannot see. No `gpu`
    parameter, for the reason the runtime `map` gives."""
    var n = xs.num_elements()
    var xs_flat = TileTensor(xs.ptr_at_offset(Coord(0)), row_major(Coord(n)))
    var ys_flat = TileTensor(ys.ptr_at_offset(Coord(0)), row_major(Coord(n)))
    var vec_n = (n // width) * width
    var i = 0
    while i < vec_n:
        ys_flat.store[width](
            Coord(i), step[width](xs_flat.load[width](Coord(i)))
        )
        i += width
    for j in range(vec_n, n):
        ys_flat.store[1](Coord(j), step[1](xs_flat.load[1](Coord(j))))


def zip_to[
    in_dtype: DType,
    out_dtype: DType,
    LayoutType: TensorLayout,
    step: def[w: Int](SIMD[in_dtype, w], SIMD[in_dtype, w]) thin -> SIMD[
        out_dtype, w
    ],
    width: Int = 1,
](
    lhs: TileTensor[
        in_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    rhs: TileTensor[
        in_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    ys: TileTensor[
        out_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
) where (
    not TileTensor[
        in_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ].all_dims_known
    and TileTensor[
        in_dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ].is_row_major
):
    """`zip_to` for runtime-shaped tensors: two inputs, one output, and a
    `step` free to change dtype -- what an elementwise comparison needs."""
    var n = lhs.num_elements()
    var lhs_flat = TileTensor(lhs.ptr_at_offset(Coord(0)), row_major(Coord(n)))
    var rhs_flat = TileTensor(rhs.ptr_at_offset(Coord(0)), row_major(Coord(n)))
    var ys_flat = TileTensor(ys.ptr_at_offset(Coord(0)), row_major(Coord(n)))
    var vec_n = (n // width) * width
    var i = 0
    while i < vec_n:
        ys_flat.store[width](
            Coord(i),
            step[width](
                lhs_flat.load[width](Coord(i)), rhs_flat.load[width](Coord(i))
            ),
        )
        i += width
    for j in range(vec_n, n):
        ys_flat.store[1](
            Coord(j),
            step[1](lhs_flat.load[1](Coord(j)), rhs_flat.load[1](Coord(j))),
        )


def map_threaded[
    dtype: DType,
    LayoutType: TensorLayout,
    step: def[w: Int](SIMD[dtype, w]) thin -> SIMD[dtype, w],
    width: Int = 1,
](
    xs: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    ys: TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    ctx: DeviceContext,
) raises where (
    not TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
):
    """`map_threaded` for a runtime-shaped tensor.

    The thread count comes from the element count either way, and that is a
    run-time value on both paths -- so unlike the GPU launch, threading has
    nothing to lose here. Same CPU-context requirement and same
    denormal-flushing caveat as the static overload.
    """
    var n = xs.num_elements()
    var xs_flat = TileTensor(xs.ptr_at_offset(Coord(0)), row_major(Coord(n)))
    var ys_flat = TileTensor(ys.ptr_at_offset(Coord(0)), row_major(Coord(n)))

    def body[
        w: Int, alignment: Int = 1
    ](coord: Coord) {imm xs_flat, imm ys_flat}:
        ys_flat.store[w](coord, step[w](xs_flat.load[w](coord)))

    elementwise[simd_width=width, target="cpu"](body, Coord(n), ctx)


def reduce_axis[
    dtype: DType,
    XsLayout: TensorLayout,
    OutLayout: TensorLayout,
    combine: def(SIMD[dtype, 1], SIMD[dtype, 1]) thin -> SIMD[dtype, 1],
    axis: Int,
](
    xs: TileTensor[
        dtype,
        XsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    dst: TileTensor[
        dtype,
        OutLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    init: SIMD[dtype, 1],
) where (
    not TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and TileTensor[
        dtype, OutLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and axis >= 0
    and axis
    < TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].rank
):
    """`reduce_axis` for a runtime-shaped tensor.

    The `outer`/`length`/`inner` split the static overload describes is
    arithmetic over extents, not over types, so it carries across unchanged
    -- `rank` and `axis` are still compile-time, and only the extents they
    multiply are read at run time.
    """
    comptime rank = type_of(xs).rank

    var length = Int(xs.dim[axis]())
    var outer = 1
    comptime for d in range(axis):
        outer *= Int(xs.dim[d]())
    var inner = 1
    comptime for d in range(axis + 1, rank):
        inner *= Int(xs.dim[d]())

    var n = xs.num_elements()
    var xs_flat = TileTensor(xs.ptr_at_offset(Coord(0)), row_major(Coord(n)))
    var dst_flat = TileTensor(
        dst.ptr_at_offset(Coord(0)), row_major(Coord(outer * inner))
    )

    for o in range(outer):
        for i in range(inner):
            var acc = init
            for k in range(length):
                acc = combine(
                    acc, xs_flat.load[1](Coord((o * length + k) * inner + i))
                )
            dst_flat.store[1](Coord(o * inner + i), acc)


def broadcast_op_axis[
    dtype: DType,
    XsLayout: TensorLayout,
    ValuesLayout: TensorLayout,
    combine: def(SIMD[dtype, 1], SIMD[dtype, 1]) thin -> SIMD[dtype, 1],
    axis: Int,
](
    xs: TileTensor[
        dtype,
        XsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    values: TileTensor[
        dtype,
        ValuesLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    ys: TileTensor[
        dtype,
        XsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
) where (
    not TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and TileTensor[
        dtype,
        ValuesLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ].is_row_major
    and axis >= 0
    and axis
    < TileTensor[
        dtype, XsLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ].rank
):
    """`broadcast_op_axis` for a runtime-shaped tensor -- the inverse of the
    runtime `reduce_axis`, so the two compose the same way their static
    counterparts do."""
    comptime rank = type_of(xs).rank

    var length = Int(xs.dim[axis]())
    var outer = 1
    comptime for d in range(axis):
        outer *= Int(xs.dim[d]())
    var inner = 1
    comptime for d in range(axis + 1, rank):
        inner *= Int(xs.dim[d]())

    var n = xs.num_elements()
    var xs_flat = TileTensor(xs.ptr_at_offset(Coord(0)), row_major(Coord(n)))
    var ys_flat = TileTensor(ys.ptr_at_offset(Coord(0)), row_major(Coord(n)))
    var values_flat = TileTensor(
        values.ptr_at_offset(Coord(0)), row_major(Coord(outer * inner))
    )

    for o in range(outer):
        for i in range(inner):
            var value = values_flat.load[1](Coord(o * inner + i))
            for k in range(length):
                var idx = (o * length + k) * inner + i
                ys_flat.store[1](
                    Coord(idx), combine(xs_flat.load[1](Coord(idx)), value)
                )


# ------------------------------------------------------------------
# Strided views
#
# Both families above walk memory linearly, which is only meaningful when
# the elements are contiguous in the order the shape implies. A transposed
# view or a sliced one is neither: `a.T` reorders the strides without moving
# a byte, and `a[1:3]` leaves gaps between rows. `is_row_major` is what
# separates the two cases, and it is false for exactly these.
#
# The two functions below take that case. They address each element through
# its own coordinates -- `offset = sum_d coord[d] * stride[d]`, with the
# coordinates recovered from a linear counter by dividing through the
# extents -- so no assumption about contiguity is made anywhere. Rank is
# compile-time, so the per-axis arithmetic unrolls; extents and strides may
# be either.
#
# They are also generic over `Storage`, which the linear walkers are not: a
# view produced by `slice` carries a different storage type than the tensor
# it came from, so pinning `PointerStorage` would reject the very inputs
# these exist for. And input and output layouts are independent, since the
# useful direction is reading a strided view into compact storage.
#
# The cost is one integer division per axis per element, against a pointer
# bump. Compact the view with a copy first if the same data is walked more
# than once.
# ------------------------------------------------------------------


def map_strided[
    dtype: DType,
    XsLayout: TensorLayout,
    XsStorage: TensorStorage,
    YsLayout: TensorLayout,
    YsStorage: TensorStorage,
    step: def[w: Int](SIMD[dtype, w]) thin -> SIMD[dtype, w],
](
    xs: TileTensor[dtype, XsLayout, MutAnyOrigin, Storage=XsStorage],
    ys: TileTensor[dtype, YsLayout, MutAnyOrigin, Storage=YsStorage],
) where (
    TileTensor[dtype, XsLayout, MutAnyOrigin, Storage=XsStorage].rank
    == TileTensor[dtype, YsLayout, MutAnyOrigin, Storage=YsStorage].rank
):
    """`ys[c] = step(xs[c])` at every coordinate `c`, whatever the strides.

    The general elementwise walk: `map` when both tensors are row-major,
    this when either is not. `xs` and `ys` need matching extents but not
    matching strides, which is what makes "read a transposed view into a
    compact buffer" a single call.

    Scalar only. A strided view has no contiguous run to fill a SIMD
    register from in general, and the one case that does -- a slice whose
    innermost axis is intact -- is not worth a second code path when a
    `copy` into row-major storage hands the whole vectorized family back.
    """
    comptime rank = type_of(xs).rank
    var n = xs.num_elements()
    var xs_ptr = xs.ptr_at_offset(Coord(0))
    var ys_ptr = ys.ptr_at_offset(Coord(0))
    for flat in range(n):
        var rem = flat
        var xs_off = 0
        var ys_off = 0
        comptime for k in range(rank):
            comptime d = rank - 1 - k
            var extent = Int(xs.dim[d]())
            var c = rem % extent
            rem //= extent
            xs_off += c * Int(xs.layout.stride[d]().value())
            ys_off += c * Int(ys.layout.stride[d]().value())
        ys_ptr[unsafe_offset=ys_off] = step[1](xs_ptr[unsafe_offset=xs_off])


def reduce_strided[
    dtype: DType,
    XsLayout: TensorLayout,
    XsStorage: TensorStorage,
    combine: def(SIMD[dtype, 1], SIMD[dtype, 1]) thin -> SIMD[dtype, 1],
](
    xs: TileTensor[dtype, XsLayout, MutAnyOrigin, Storage=XsStorage],
    init: SIMD[dtype, 1],
) -> SIMD[dtype, 1]:
    """Fold every element of a strided view down to one value.

    Visits coordinates in row-major order, so this agrees element for
    element -- and therefore bit for bit -- with `reduce` over a compacted
    copy of the same view, for a `combine` that is not associative as well
    as one that is.
    """
    comptime rank = type_of(xs).rank
    var n = xs.num_elements()
    var xs_ptr = xs.ptr_at_offset(Coord(0))
    var total = init
    for flat in range(n):
        var rem = flat
        var xs_off = 0
        comptime for k in range(rank):
            comptime d = rank - 1 - k
            var extent = Int(xs.dim[d]())
            var c = rem % extent
            rem //= extent
            xs_off += c * Int(xs.layout.stride[d]().value())
        total = combine(total, xs_ptr[unsafe_offset=xs_off])
    return total
