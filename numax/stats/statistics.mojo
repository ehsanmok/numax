"""NumPy-named statistics, composed from `numax.core.tensor` and `FloatLike`.

**Tier 2, with three exceptions.** `mean`, `variance` and `stddev` fold
through MAX's `Welford` monoid over its `rowwise` scaffolder
(`numax.core.rowwise`), so they are threaded on CPU, tiered on GPU under
`gpu=True`, and never download the input -- only the resulting scalars come
back. Every other reduction here still walks a host copy;
`numax.core.tensor`'s `reduce` and `reduce_axis` are the GPU-launchable
primitives underneath those.

`docs/parity.md` picks statistics as a genuine `numax` gap with a
selective axis-1 lift: MAX ships no NumPy-named `mean`/`var`/`std`/`median`/
`mode` entry point, but it does ship `argmax`/`argmin` (`nn.argmaxmin`),
which this module routes to directly rather than re-implementing (the
MAX-first check firing first, same shape as `numax.linalg.matmul`'s `Tensor`
overload being MAX's `linalg.matmul`).

What MAX *does* ship, and an earlier revision of this docstring denied, is
the fold underneath: `algorithm.reduce_op`'s `Welford` is an online
mean/variance monoid, and `algorithm.rowwise` drives it on CPU and GPU from
one body. The denial cited `max.algorithm.functional`, a `max.*` path that
does export only `elementwise` -- but the kernel package is the **top-level
`algorithm`** root, the same distinction that puts matmul in `linalg` and not
`max.linalg`. So the entry points here are numax's to name, while the
reduction they sit on is MAX's to provide -- which is now what happens:
`mean`, `variance`, `stddev` and the axis-wise `mean`/`variance_axis` are
`Welford` calls, and numax writes only the shape bookkeeping around them.

Two genuinely different shapes live in this one file, because they answer
two different questions:

- **`Plain`-only, `TileTensor`-based** (`sum`, `prod`, `min`, `max`, `mean`,
  `variance`, `stddev`, `variance_axis`, `median`, `mode`, `argmax`,
  `argmin`, `cumprod`): "what NumPy-shaped
  statistic can I compute over a buffer of raw `dtype` values". These
  compose from MAX's reductions and `numax.core.tensor.reduce` the same way
  `numax.core.array`'s creation
  routines compose from `TileTensor` -- a thin, `Plain`-only layer, axis 2
  only. `median`/`mode` sort first via the standard library's `List.sort()`
  -- the fixed-iteration invariant restricts what *numax* writes inside a
  `FloatLike`-generic kernel, not what a `Plain`-only orchestration calls
  out to.
- **`FloatLike`-generic, `List[T]`-based** (`mean`, `variance`, `stddev`,
  `cumsum`): "does calling this at `Compensated` instead of `Plain` recover
  precision a long summation would otherwise lose". These take a
  `List[T]`, not a `TileTensor`, on purpose -- `TileTensor` only ever holds
  raw `dtype` SIMD lanes, and `Compensated`/`Decimal`/`Dual` values have no
  such flat representation to be laid out in one, so an "array of `T`" here
  can only mean a `List` of individually-boxed values. This is the same
  precision motivation `numax.core.compensated`'s own tests already measure
  (`tests/core/test_compensated.mojo`), applied to a running sum instead of a
  single kernel evaluation.

`var` cannot be the public name for the variance function -- `var` is a
reserved keyword that introduces a variable declaration in Mojo, and
`def var(...)` is rejected by the parser outright. Likewise `std` cannot
name the standard-deviation function -- `std` is Mojo's own standard
library package name, always in scope, and `def std(...)` is rejected as
an "invalid redefinition of 'std'". The `FloatLike`-generic variance and
standard deviation are named `variance` and `stddev` here for exactly
these two reasons.

## Axis-wise reductions

`sum`, `prod`, `min`, `max` and `mean` each have a second overload folding
one axis instead of the whole tensor: `sum(a)` is `numpy.sum(a)` and
`sum[axis=k](a)` is `numpy.sum(a, axis=k)`. One name carries both because
the axis form's extra parameter is what tells them apart -- a call naming
no axis cannot reach the folding overload, and a call naming one cannot
reach the whole-tensor overload.

The axis is a compile-time parameter because the result's *rank* depends on it
(`rank - 1`, the reduced axis dropped, matching NumPy's default
`keepdims=False`), and rank is compile-time throughout this library. The
extents are not: the result comes back run-time-shaped, since they are read
from the input rather than named.

`numax.core.tensor.reduce_axis` is the same fold one layer down, over a
`TileTensor` a caller allocated the output for, and it launches on a GPU
where these do not.

**Explicitly out of scope**, matching this module's own gap-only mandate:
sorting, which now lives in `numax.core.sorting` as a tier-2 module (`sort`,
`argsort`, `searchsorted`, `unique`, plus the counting and masking family).
`List.sort()` is still what `median`/`mode` reach for internally; what
changed is that a NumPy caller now has a `numax` name for it too, on the
tier-2 terms `docs/architecture.md` sets out.
"""

from std.math import sqrt as _sqrt

from layout import Coord, TileTensor
from layout.tile_layout import row_major, TensorLayout
from layout.tile_tensor import PointerStorage
from nn.argmaxmin import argmax as _nn_argmax, argmin as _nn_argmin
from nn.cumsum import cumsum as _nn_cumsum

from ..core.array import Dynamic, Static, Tensor, _dyn_shape_from
from ..core.numeric import FloatLike
from ..core.rowwise import mean_variance_axis


@always_inline
def _target[gpu: Bool]() -> StaticString:
    """`rowwise`'s target string for numax's `gpu: Bool` parameter."""
    return "gpu" if gpu else "cpu"


def _fold_axis[
    dtype: DType,
    LayoutType: TensorLayout,
    axis: Int,
    combine: def(SIMD[dtype, 1], SIMD[dtype, 1]) thin -> SIMD[dtype, 1],
](xs: Tensor[dtype, LayoutType], init: Scalar[dtype]) raises -> Dynamic[
    dtype, LayoutType.rank - 1
] where (axis >= 0 and axis < LayoutType.rank and LayoutType.rank > 1):
    """Fold `xs` along `axis` with `combine`, dropping that axis.

    A row-major tensor splits around any axis into `outer` (the extents
    before it, multiplied), `length` (the axis), and `inner` (the extents
    after it), so element `(o, k, i)` is at flat index `(o*length +
    k)*inner + i` and one flat walk covers every rank and axis.
    """
    comptime rank = LayoutType.rank
    var length = xs.dim_at(axis)
    var outer = 1
    for d in range(axis):
        outer *= xs.dim_at(d)
    var inner = 1
    for d in range(axis + 1, rank):
        inner *= xs.dim_at(d)

    var values = xs.to_host()
    var out = List[Scalar[dtype]](capacity=outer * inner)
    for o in range(outer):
        for i in range(inner):
            var acc = init
            for k in range(length):
                acc = combine(acc, values[(o * length + k) * inner + i])
            out.append(acc)

    var extents = List[Int](capacity=rank - 1)
    for d in range(rank):
        if d != axis:
            extents.append(xs.dim_at(d))
    var result = Dynamic[dtype, rank - 1](
        xs.context(), row_major(_dyn_shape_from[rank - 1](extents))
    )
    result.copy_from_host(out)
    return result^


def _add[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return a + b


def _mul[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return a * b


def _smaller[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return a if a < b else b


def _larger[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return a if a > b else b


def sum[
    dtype: DType, LayoutType: TensorLayout, axis: Int
](xs: Tensor[dtype, LayoutType]) raises -> Dynamic[
    dtype, LayoutType.rank - 1
] where (
    dtype.is_floating_point()
    and axis >= 0
    and axis < LayoutType.rank
    and LayoutType.rank > 1
):
    """`xs` summed along `axis`. `numpy.sum(a, axis=k)`."""
    return _fold_axis[axis=axis, combine=_add[dtype]](xs, 0)


def prod[
    dtype: DType, LayoutType: TensorLayout, axis: Int
](xs: Tensor[dtype, LayoutType]) raises -> Dynamic[
    dtype, LayoutType.rank - 1
] where (
    dtype.is_floating_point()
    and axis >= 0
    and axis < LayoutType.rank
    and LayoutType.rank > 1
):
    """`xs` multiplied along `axis`. `numpy.prod(a, axis=k)`."""
    return _fold_axis[axis=axis, combine=_mul[dtype]](xs, 1)


def min[
    dtype: DType, LayoutType: TensorLayout, axis: Int
](xs: Tensor[dtype, LayoutType]) raises -> Dynamic[
    dtype, LayoutType.rank - 1
] where (
    dtype.is_floating_point()
    and axis >= 0
    and axis < LayoutType.rank
    and LayoutType.rank > 1
):
    """The smallest element along `axis`. `numpy.min(a, axis=k)`.

    Seeded with positive infinity, so an axis of length zero yields
    infinity rather than reading an element that is not there.
    """
    return _fold_axis[axis=axis, combine=_smaller[dtype]](
        xs, Scalar[dtype].MAX_FINITE
    )


def max[
    dtype: DType, LayoutType: TensorLayout, axis: Int
](xs: Tensor[dtype, LayoutType]) raises -> Dynamic[
    dtype, LayoutType.rank - 1
] where (
    dtype.is_floating_point()
    and axis >= 0
    and axis < LayoutType.rank
    and LayoutType.rank > 1
):
    """The largest element along `axis`. `numpy.max(a, axis=k)`. Seeded the
    mirror of `min`'s."""
    return _fold_axis[axis=axis, combine=_larger[dtype]](
        xs, Scalar[dtype].MIN_FINITE
    )


def mean[
    dtype: DType, LayoutType: TensorLayout, axis: Int, gpu: Bool = False
](mut xs: Tensor[dtype, LayoutType]) raises -> Dynamic[
    dtype, LayoutType.rank - 1
] where (
    dtype.is_floating_point()
    and axis >= 0
    and axis < LayoutType.rank
    and LayoutType.rank > 1
    and TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ].is_row_major
):
    """The arithmetic mean along `axis`. `numpy.mean(a, axis=k)`.

    MAX's `Welford` monoid, as the whole-tensor `mean` above. The variance
    the monoid also produces is discarded here; `variance_axis` returns
    both from the one traversal for a caller that wants them together.
    """
    var means = _welford_dst[axis=axis](xs)
    var variances = _welford_dst[axis=axis](xs)
    _welford_axis[axis=axis, gpu=gpu](xs, means, variances, 0)
    return means^


def variance_axis[
    dtype: DType, LayoutType: TensorLayout, axis: Int, gpu: Bool = False
](mut xs: Tensor[dtype, LayoutType], ddof: Int = 0) raises -> Tuple[
    Dynamic[dtype, LayoutType.rank - 1], Dynamic[dtype, LayoutType.rank - 1]
] where (
    dtype.is_floating_point()
    and axis >= 0
    and axis < LayoutType.rank
    and LayoutType.rank > 1
    and TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ].is_row_major
):
    """`(mean, variance)` along `axis`, from one traversal.

    `numpy.mean(a, axis=k)` and `numpy.var(a, axis=k, ddof=ddof)` at once.
    Named for the variance rather than the mean because `mean[axis=k]`
    already exists and a caller reaching for both is reaching for this one;
    MAX's `Welford` computes both regardless, so returning only one would
    mean reducing twice to get the other.
    """
    var means = _welford_dst[axis=axis](xs)
    var variances = _welford_dst[axis=axis](xs)
    _welford_axis[axis=axis, gpu=gpu](xs, means, variances, ddof)
    return (means^, variances^)


def _welford_dst[
    dtype: DType, LayoutType: TensorLayout, axis: Int
](mut xs: Tensor[dtype, LayoutType]) raises -> Dynamic[
    dtype, LayoutType.rank - 1
] where (axis >= 0 and axis < LayoutType.rank and LayoutType.rank > 1):
    """A destination shaped like `xs` with `axis` dropped.

    Run-time-shaped because the surviving extents are read from the input
    rather than named, which is what `Dynamic` is for.
    """
    comptime rank = LayoutType.rank
    var extents = List[Int](capacity=rank - 1)
    for d in range(rank):
        if d != axis:
            extents.append(xs.dim_at(d))
    return Dynamic[dtype, rank - 1](
        xs.context(), row_major(_dyn_shape_from[rank - 1](extents))
    )


def _welford_axis[
    dtype: DType, LayoutType: TensorLayout, axis: Int, gpu: Bool
](
    mut xs: Tensor[dtype, LayoutType],
    mut means: Dynamic[dtype, LayoutType.rank - 1],
    mut variances: Dynamic[dtype, LayoutType.rank - 1],
    ddof: Int,
) raises where (
    dtype.is_floating_point()
    and axis >= 0
    and axis < LayoutType.rank
    and LayoutType.rank > 1
    and TileTensor[
        dtype,
        LayoutType,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ].is_row_major
):
    """Fill `means` and `variances` along `axis` from one traversal.

    Both destinations are arguments rather than return values so that
    `mean` can discard one without the caller-visible tuple a returned
    pair would need -- a `Dynamic` is not copyable, so moving one out of a
    tuple is more ceremony than passing two in.
    """
    mean_variance_axis[dtype, _, _, axis=axis, target=_target[gpu]()](
        xs.view(), means.view(), variances.view(), ddof, Optional(xs.context())
    )


def sum[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """The sum of every element of `xs`."""
    var n = xs.size()
    var values = xs.to_host()
    var acc = Scalar[dtype](0)
    for i in range(n):
        acc += values[i]
    return acc


def prod[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """The product of every element of `xs`."""
    var n = xs.size()
    var values = xs.to_host()
    var acc = Scalar[dtype](1)
    for i in range(n):
        acc *= values[i]
    return acc


def min[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """The smallest element of `xs`. `xs` must have at least one element."""
    var n = xs.size()
    var values = xs.to_host()
    var best = values[0]
    for i in range(1, n):
        if values[i] < best:
            best = values[i]
    return best


def max[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """The largest element of `xs`. `xs` must have at least one element."""
    var n = xs.size()
    var values = xs.to_host()
    var best = values[0]
    for i in range(1, n):
        if values[i] > best:
            best = values[i]
    return best


def _welford[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool
](mut xs: Tensor[dtype, LayoutType], ddof: Int) raises -> Tuple[
    Scalar[dtype], Scalar[dtype]
] where dtype.is_floating_point():
    """`(mean, variance)` of every element of `xs`, in one traversal.

    The whole tensor is one row: a rank-1 view over its buffer reduced
    along axis 0. Only the two resulting scalars come back to the host, so
    a device-resident `xs` stays where it is instead of being downloaded.
    """
    var ctx = xs.context()
    var n = xs.size()
    var flat = TileTensor(
        xs.view().ptr_at_offset(Coord(0)), row_major(Coord(n))
    )
    var mean_out = Static[dtype, 1](ctx)
    var var_out = Static[dtype, 1](ctx)
    mean_variance_axis[dtype, _, _, axis=0, target=_target[gpu]()](
        flat, mean_out.view(), var_out.view(), ddof, Optional(ctx)
    )
    return (mean_out.to_host()[0], var_out.to_host()[0])


def mean[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](mut xs: Tensor[dtype, LayoutType]) raises -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """The arithmetic mean of `xs`.

    MAX's `Welford` monoid over its `rowwise` scaffolder, so this is
    threaded on CPU, tiered on GPU, and never downloads `xs` -- only the
    resulting scalar crosses back. `gpu=True` requires `xs` to have been
    allocated on an accelerator context.

    `Plain`-only: a mean is a single scalar with no derivative to
    propagate through the division by a plain `Int` count, so there is no
    axis-1 win here the way there is for `variance`/`stddev`/`cumsum`.
    """
    var pair = _welford[gpu=gpu](xs, 0)
    return pair[0]


def _median_of[
    dtype: DType
](var values: List[Scalar[dtype]]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The median of `values`, sorting them in place.

    Shared by the whole-tensor `median` and the axis one, so the even-count
    convention is decided once.
    """
    var n = len(values)
    sort(values)
    if n % 2 == 1:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / Scalar[dtype](2)


def _mode_of[
    dtype: DType
](var values: List[Scalar[dtype]]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    """The most frequent value in `values`, smallest among ties, sorting
    them in place. Shared by the whole-tensor `mode` and the axis one."""
    var n = len(values)
    sort(values)
    var best_value = values[0]
    var best_count = 1
    var run_value = values[0]
    var run_count = 1
    for i in range(1, n):
        if values[i] == run_value:
            run_count += 1
        else:
            run_value = values[i]
            run_count = 1
        if run_count > best_count:
            best_count = run_count
            best_value = run_value
    return best_value


def _axis_split[
    dtype: DType, LayoutType: TensorLayout, axis: Int
](xs: Tensor[dtype, LayoutType]) raises -> Tuple[Int, Int, Int]:
    """`xs` split around `axis` into `(outer, length, inner)`.

    The same decomposition `_fold_axis` walks -- element `(o, k, i)` sits at
    flat index `(o * length + k) * inner + i` -- for the reductions that
    need a whole slice at a time rather than a running accumulator.
    """
    var length = xs.dim_at(axis)
    var outer = 1
    for d in range(axis):
        outer *= xs.dim_at(d)
    var inner = 1
    for d in range(axis + 1, LayoutType.rank):
        inner *= xs.dim_at(d)
    return (outer, length, inner)


def _axis_extents[
    dtype: DType, LayoutType: TensorLayout, axis: Int
](xs: Tensor[dtype, LayoutType]) raises -> List[Int]:
    """`xs`'s extents with `axis` dropped -- the shape an axis reduction
    returns, matching NumPy's default of no `keepdims`."""
    var out = List[Int](capacity=LayoutType.rank - 1)
    for d in range(LayoutType.rank):
        if d != axis:
            out.append(xs.dim_at(d))
    return out^


def median[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """The median of `xs` -- the average of the two middle elements when
    `xs` has an even count, matching NumPy's default."""
    return _median_of(xs.to_host())


def median[
    dtype: DType, LayoutType: TensorLayout, axis: Int
](xs: Tensor[dtype, LayoutType]) raises -> Dynamic[
    dtype, LayoutType.rank - 1
] where (
    dtype.is_floating_point()
    and axis >= 0
    and axis < LayoutType.rank
    and LayoutType.rank > 1
):
    """`xs` reduced to its median along `axis`. `numpy.median(a, axis=k)`.

    A median is not a fold -- it needs the whole slice at once -- so this
    gathers each slice rather than running `_fold_axis`. The even-count
    convention is `_median_of`'s, the same one the whole-tensor overload
    uses.
    """
    var split = _axis_split[axis=axis](xs)
    var outer = split[0]
    var length = split[1]
    var inner = split[2]
    var values = xs.to_host()
    var out = List[Scalar[dtype]](capacity=outer * inner)
    for o in range(outer):
        for i in range(inner):
            var slice_ = List[Scalar[dtype]](capacity=length)
            for k in range(length):
                slice_.append(values[(o * length + k) * inner + i])
            out.append(_median_of(slice_^))
    return Dynamic[dtype, LayoutType.rank - 1](
        xs.context(),
        row_major(
            _dyn_shape_from[LayoutType.rank - 1](_axis_extents[axis=axis](xs))
        ),
        out^,
    )


def mode[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """The most frequent value in `xs`; the smallest among ties, matching
    `scipy.stats.mode`'s convention."""
    return _mode_of(xs.to_host())


def mode[
    dtype: DType, LayoutType: TensorLayout, axis: Int
](xs: Tensor[dtype, LayoutType]) raises -> Dynamic[
    dtype, LayoutType.rank - 1
] where (
    dtype.is_floating_point()
    and axis >= 0
    and axis < LayoutType.rank
    and LayoutType.rank > 1
):
    """`xs` reduced to its most frequent value along `axis`.
    `scipy.stats.mode(a, axis=k)`, smallest among ties."""
    var split = _axis_split[axis=axis](xs)
    var outer = split[0]
    var length = split[1]
    var inner = split[2]
    var values = xs.to_host()
    var out = List[Scalar[dtype]](capacity=outer * inner)
    for o in range(outer):
        for i in range(inner):
            var slice_ = List[Scalar[dtype]](capacity=length)
            for k in range(length):
                slice_.append(values[(o * length + k) * inner + i])
            out.append(_mode_of(slice_^))
    return Dynamic[dtype, LayoutType.rank - 1](
        xs.context(),
        row_major(
            _dyn_shape_from[LayoutType.rank - 1](_axis_extents[axis=axis](xs))
        ),
        out^,
    )


def argmax[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> Int where dtype.is_floating_point():
    """The flat index of the largest element of `xs`, via `nn.argmaxmin`
    (MAX-first: `numax` writes no comparison logic of its own here)."""
    var n = xs.size()
    var values = xs.to_host()
    var flat = TileTensor(values, row_major(Coord(n)))
    var out_storage = List[Scalar[DType.int64]](length=1, fill=0)
    var out = TileTensor(out_storage, row_major[1]())
    _nn_argmax(flat, 0, out)
    return Int(out[0])


def argmin[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> Int where dtype.is_floating_point():
    """The flat index of the smallest element of `xs`, via `nn.argmaxmin`."""
    var n = xs.size()
    var values = xs.to_host()
    var flat = TileTensor(values, row_major(Coord(n)))
    var out_storage = List[Scalar[DType.int64]](length=1, fill=0)
    var out = TileTensor(out_storage, row_major[1]())
    _nn_argmin(flat, 0, out)
    return Int(out[0])


def _argn_axis[
    dtype: DType, LayoutType: TensorLayout, axis: Int, largest: Bool
](xs: Tensor[dtype, LayoutType]) raises -> Dynamic[
    DType.int64, LayoutType.rank - 1
] where (
    dtype.is_floating_point()
    and axis >= 0
    and axis < LayoutType.rank
    and LayoutType.rank > 1
):
    """Indices of the largest or smallest element of `xs` along `axis`.

    **`nn.argmaxmin` handles only the innermost axis**: `_argn` raises
    "axis other than innermost not supported yet" for anything else, and
    wants an output of the input's own rank with that axis at extent 1.
    So the delegation fires for `axis == rank - 1`, where MAX
    `parallelize`s the outer rows, and numax walks the other axes itself.
    Both return the *first* extremum among ties, which is NumPy's rule and
    (verified) also MAX's.

    The returned extents drop `axis`, matching NumPy's default of no
    `keepdims`, so the innermost case reshapes MAX's `(..., 1)` output.
    """
    comptime rank = LayoutType.rank
    var out_extents = _axis_extents[axis=axis](xs)
    var count = 1
    for d in range(rank - 1):
        count *= out_extents[d]

    var values = xs.to_host()
    var out = List[Scalar[DType.int64]](length=count, fill=0)

    comptime if axis == rank - 1:
        var in_extents = List[Int](capacity=rank)
        var keep_extents = List[Int](capacity=rank)
        for d in range(rank):
            in_extents.append(xs.dim_at(d))
            keep_extents.append(1 if d == axis else xs.dim_at(d))
        var inp = TileTensor(
            values, row_major(_dyn_shape_from[rank](in_extents))
        )
        var keep = TileTensor(
            out, row_major(_dyn_shape_from[rank](keep_extents))
        )
        comptime if largest:
            _nn_argmax(inp, axis, keep)
        else:
            _nn_argmin(inp, axis, keep)
    else:
        var split = _axis_split[axis=axis](xs)
        var outer = split[0]
        var length = split[1]
        var inner = split[2]
        for o in range(outer):
            for i in range(inner):
                var best = values[o * length * inner + i]
                var best_k = 0
                for k in range(1, length):
                    var v = values[(o * length + k) * inner + i]
                    var better = v > best if largest else v < best
                    if better:
                        best = v
                        best_k = k
                out[o * inner + i] = Scalar[DType.int64](best_k)

    return Dynamic[DType.int64, rank - 1](
        xs.context(),
        row_major(_dyn_shape_from[rank - 1](out_extents)),
        out^,
    )


def argmax[
    dtype: DType, LayoutType: TensorLayout, axis: Int
](xs: Tensor[dtype, LayoutType]) raises -> Dynamic[
    DType.int64, LayoutType.rank - 1
] where (
    dtype.is_floating_point()
    and axis >= 0
    and axis < LayoutType.rank
    and LayoutType.rank > 1
):
    """Indices of the largest element along `axis`. `numpy.argmax(a, axis=k)`.

    Indices are positions *along `axis`*, not flat ones -- which is why
    this returns a tensor where the whole-tensor overload returns a single
    flat `Int`.
    """
    return _argn_axis[axis=axis, largest=True](xs)


def argmin[
    dtype: DType, LayoutType: TensorLayout, axis: Int
](xs: Tensor[dtype, LayoutType]) raises -> Dynamic[
    DType.int64, LayoutType.rank - 1
] where (
    dtype.is_floating_point()
    and axis >= 0
    and axis < LayoutType.rank
    and LayoutType.rank > 1
):
    """Indices of the smallest element along `axis`. `numpy.argmin(a, axis=k)`.
    """
    return _argn_axis[axis=axis, largest=False](xs)


def _scan_axis[
    dtype: DType, LayoutType: TensorLayout, axis: Int, multiply: Bool
](xs: Tensor[dtype, LayoutType]) raises -> List[Scalar[dtype]]:
    """A running sum or product of `xs` along `axis`, in row-major order.

    The sum half delegates to `nn.cumsum`, which takes `TileTensor` in and
    out, takes its axis as a compile-time parameter, handles any axis (not
    only the innermost, unlike `nn.argmaxmin`) and accumulates in
    `float64` for a `float32` input -- so it is both MAX-first and more
    accurate than the obvious loop. There is no `nn.cumprod`, so the
    product half is numax's own walk over `_axis_split`'s decomposition.

    `ponytail:` both halves are host-side. `nn.cumsum` has no `target` and
    no `DeviceContext` -- the graph operator takes a context and drops it,
    so `mo.cumsum` has no GPU kernel either -- and a device scan is a
    blocked Blelloch pass rather than a flag on this one, so it is a
    separate commit rather than a parameter here.
    """
    var count = xs.size()
    var values = xs.to_host()
    var out = List[Scalar[dtype]](length=count, fill=0)

    comptime if multiply:
        var split = _axis_split[axis=axis](xs)
        var outer = split[0]
        var length = split[1]
        var inner = split[2]
        for o in range(outer):
            for i in range(inner):
                var acc = Scalar[dtype](1)
                for k in range(length):
                    var at = (o * length + k) * inner + i
                    acc = acc * values[at]
                    out[at] = acc
    else:
        comptime rank = LayoutType.rank
        var extents = List[Int](capacity=rank)
        for d in range(rank):
            extents.append(xs.dim_at(d))
        var shape = row_major(_dyn_shape_from[rank](extents))
        _nn_cumsum[exclusive=False, reverse=False, axis=axis](
            TileTensor(out, shape), TileTensor(values, shape)
        )

    return out^


def cumprod[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> Static[
    dtype, LayoutType.static_product
] where LayoutType.all_dims_known:
    """The running product of every element of `xs`, flattened row-major.
    `numpy.cumprod(a)`.

    Rank-1 in gives rank-1 out at the same length, so this is the same call
    it always was; at higher rank it flattens, which is what
    `numpy.cumprod` with no `axis` does.
    """
    comptime n = LayoutType.static_product
    var values = xs.to_host()
    var storage = List[Scalar[dtype]](capacity=n)
    var acc = Scalar[dtype](1)
    for i in range(n):
        acc = acc * values[i]
        storage.append(acc)
    return Static[dtype, n](xs.context(), storage^)


def cumprod[
    dtype: DType, LayoutType: TensorLayout, axis: Int
](xs: Tensor[dtype, LayoutType]) raises -> Tensor[dtype, LayoutType] where (
    axis >= 0 and axis < LayoutType.rank
):
    """The running product along `axis`. `numpy.cumprod(a, axis=k)`.

    Keeps `xs`'s shape rather than dropping the axis -- a scan is not a
    reduction.
    """
    return Tensor[dtype, LayoutType](
        xs.context(),
        xs.layout,
        _scan_axis[axis=axis, multiply=True](xs)^,
    )


def variance[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](mut xs: Tensor[dtype, LayoutType], ddof: Int = 0) raises -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """The variance of every element of `xs`, `ddof` subtracted from the
    divisor (`ddof=1` for the sample variance).

    One traversal through MAX's `Welford` monoid, which carries
    `{count, mean, M2}` and combines partials by Chan's formula, rather
    than the two host passes this used to make. Both the threading and the
    accuracy come from that: Welford never forms `E[x^2]`, so a variance
    small relative to the mean survives, where the one-pass
    subtract-of-squares shortcut cancels away to noise.

    The `List[T]` form below is the `FloatLike`-generic one -- call that at
    `Compensated` when the summation length is what threatens the result.
    """
    var pair = _welford[gpu=gpu](xs, ddof)
    return pair[1]


def stddev[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](mut xs: Tensor[dtype, LayoutType], ddof: Int = 0) raises -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """The standard deviation of `xs`: `variance(xs, ddof)` square-rooted.

    Named `stddev`, not NumPy's `std`, for the reason the `List[T]` form
    below documents: `std` is Mojo's standard library package and cannot be
    defined as a function name at all.
    """
    return _sqrt(variance[gpu=gpu](xs, ddof))


def cumsum[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> Static[
    dtype, LayoutType.static_product
] where LayoutType.all_dims_known:
    """The running sum of every element of `xs`, flattened row-major.
    `numpy.cumsum(a)`.

    The counterpart of `cumprod`; the `List[T]` form below is the
    `FloatLike`-generic one. Rank-1 in gives rank-1 out at the same length,
    and at higher rank it flattens, which is what `numpy.cumsum` with no
    `axis` does.
    """
    comptime n = LayoutType.static_product
    var values = xs.to_host()
    var storage = List[Scalar[dtype]](length=n, fill=0)
    _nn_cumsum[exclusive=False, reverse=False, axis=0](
        TileTensor(storage, row_major(Coord(n))),
        TileTensor(values, row_major(Coord(n))),
    )
    return Static[dtype, n](xs.context(), storage^)


def cumsum[
    dtype: DType, LayoutType: TensorLayout, axis: Int
](xs: Tensor[dtype, LayoutType]) raises -> Tensor[dtype, LayoutType] where (
    axis >= 0 and axis < LayoutType.rank
):
    """The running sum along `axis`. `numpy.cumsum(a, axis=k)`.

    Keeps `xs`'s shape rather than dropping the axis -- a scan is not a
    reduction. Routed to `nn.cumsum`.
    """
    return Tensor[dtype, LayoutType](
        xs.context(),
        xs.layout,
        _scan_axis[axis=axis, multiply=False](xs)^,
    )


def mean[T: FloatLike](xs: List[T]) -> T:
    """The arithmetic mean of `xs`, over any `FloatLike` conformer.

    Also the helper `variance` composes with below -- calling this at
    `Compensated` keeps the running sum in extra precision before the
    final division, the same axis-1 win `variance`/`std`/`cumsum` document.
    """
    var acc = T.constant(0.0)
    for x in xs:
        acc = acc + x
    return acc / T.constant(Float64(len(xs)))


def variance[T: FloatLike](xs: List[T], ddof: Int = 0) -> T:
    """The variance of `xs`, over any `FloatLike` conformer.

    Named `variance`, not NumPy's `var` -- `var` is a Mojo keyword that
    introduces a variable declaration, so `def var(...)` does not parse.

    `ddof` (delta degrees of freedom) divides by `len(xs) - ddof`; NumPy's
    default `ddof=0` is the population variance.

    Calling this at `Compensated` instead of `Plain` is the real axis-1
    win this module was built to demonstrate: summing many squared
    deviations accumulates rounding error in `Plain`'s ordinary `float32`
    the same way any long summation does (`numax.core.compensated`'s own tests
    already measure this for a single running sum), and `Compensated`
    recovers it here for free -- this kernel was written once, against
    `FloatLike`, with no `Compensated`-specific code path.
    """
    var m = mean(xs)
    var acc = T.constant(0.0)
    for x in xs:
        var d = x - m
        acc = acc + d * d
    return acc / T.constant(Float64(len(xs) - ddof))


def stddev[T: FloatLike](xs: List[T], ddof: Int = 0) -> T:
    """The standard deviation of `xs`: `variance(xs, ddof).sqrt()`.

    Named `stddev`, not NumPy's `std` -- `std` is the name of Mojo's own
    standard library package (`from std.math import ...` etc.), always in
    scope, and a top-level `def std(...)` collides with it outright
    ("invalid redefinition of 'std'"), the same class of keyword/name
    collision `variance` above was renamed to avoid.
    """
    return variance(xs, ddof).sqrt()


def cumsum[T: FloatLike](xs: List[T]) -> List[T]:
    """The running sum of `xs`: `ys[i] = xs[0] + ... + xs[i]`.

    `FloatLike`-generic for the same reason `variance`/`std` are: a long
    running sum is exactly where `Compensated`'s extra precision earns its
    keep over `Plain`, and this kernel gets that for free by being written
    against the trait rather than a concrete `dtype`.
    """
    var result = List[T](capacity=len(xs))
    var acc = T.constant(0.0)
    for x in xs:
        acc = acc + x
        result.append(acc.copy())
    return result^
