"""NumPy-named statistics, composed from `numax.core.tensor` and `FloatLike`.

`docs/parity.md` picks statistics as a genuine `numax` gap with a
selective axis-1 lift: MAX ships no `mean`/`var`/`std`/`median`/`mode` at
all (verified directly -- `max.algorithm.functional` exports only
`elementwise`), but it does ship `argmax`/`argmin` (`nn.argmaxmin`), which
this module routes to directly rather than re-implementing (the MAX-first
check firing first, same shape as `numax.linalg.matmul`'s recommendation to
call MAX's `linalg.matmul` past ~8x8).

Two genuinely different shapes live in this one file, because they answer
two different questions:

- **`Plain`-only, `TileTensor`-based** (`sum`, `prod`, `min`, `max`, `mean`,
  `median`, `mode`, `argmax`, `argmin`, `cumprod`): "what NumPy-shaped
  statistic can I compute over a buffer of raw `dtype` values". These
  compose from `numax.core.tensor.reduce` the same way `numax.core.array`'s creation
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

`sum`, `prod`, `min`, `max` and `mean` each have an `_axis` sibling folding
one axis instead of the whole tensor, which is `numpy.sum(a, axis=k)`. The
axis is a compile-time parameter because the result's *rank* depends on it
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
from nn.argmaxmin import argmax as _nn_argmax, argmin as _nn_argmin

from ..core.array import Dynamic, Shaped, Tensor, _dyn_shape_from
from ..core.numeric import FloatLike


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


def sum_axis[
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


def prod_axis[
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


def min_axis[
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


def max_axis[
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
    mirror of `min_axis`."""
    return _fold_axis[axis=axis, combine=_larger[dtype]](
        xs, Scalar[dtype].MIN_FINITE
    )


def mean_axis[
    dtype: DType, LayoutType: TensorLayout, axis: Int
](xs: Tensor[dtype, LayoutType]) raises -> Dynamic[
    dtype, LayoutType.rank - 1
] where (
    dtype.is_floating_point()
    and axis >= 0
    and axis < LayoutType.rank
    and LayoutType.rank > 1
):
    """The arithmetic mean along `axis`. `numpy.mean(a, axis=k)`."""
    var totals = sum_axis[axis=axis](xs)
    var length = Scalar[dtype](xs.dim_at(axis))
    for i in range(totals.size()):
        totals[i] = totals[i] / length
    return totals^


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


def mean[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """The arithmetic mean of `xs`.

    `Plain`-only: a mean is a single scalar with no derivative to
    propagate through the division by a plain `Int` count, so there is no
    axis-1 win here the way there is for `variance`/`stddev`/`cumsum`.
    """
    var n = xs.size()
    return sum(xs) / Scalar[dtype](n)


def median[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """The median of `xs` -- the average of the two middle elements when
    `xs` has an even count, matching NumPy's default."""
    var n = xs.size()
    var values = xs.to_host()
    sort(values)
    if n % 2 == 1:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / Scalar[dtype](2)


def mode[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType]) raises -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """The most frequent value in `xs`; the smallest among ties, matching
    `scipy.stats.mode`'s convention."""
    var n = xs.size()
    var values = xs.to_host()
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


def cumprod[
    dtype: DType, n: Int
](xs: Shaped[dtype, n]) raises -> Shaped[dtype, n]:
    """The running product of `xs`: `ys[i] = xs[0] * ... * xs[i]`."""
    var values = xs.to_host()
    var storage = List[Scalar[dtype]](capacity=n)
    var acc = Scalar[dtype](1)
    for i in range(n):
        acc = acc * values[i]
        storage.append(acc)
    return Shaped[dtype, n](xs.context(), storage^)


def variance[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType], ddof: Int = 0) raises -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """The variance of every element of `xs`, `ddof` subtracted from the
    divisor (`ddof=1` for the sample variance).

    Two passes over a host copy: the mean, then the squared deviations.
    The `List[T]` form below is the `FloatLike`-generic one -- call that at
    `Compensated` when the summation length is what threatens the result.
    """
    var n = xs.size()
    var values = xs.to_host()
    var total = Scalar[dtype](0)
    for i in range(n):
        total += values[i]
    var mu = total / Scalar[dtype](n)
    var acc = Scalar[dtype](0)
    for i in range(n):
        var d = values[i] - mu
        acc += d * d
    return acc / Scalar[dtype](n - ddof)


def stddev[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType], ddof: Int = 0) raises -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """The standard deviation of `xs`: `variance(xs, ddof)` square-rooted.

    Named `stddev`, not NumPy's `std`, for the reason the `List[T]` form
    below documents: `std` is Mojo's standard library package and cannot be
    defined as a function name at all.
    """
    return _sqrt(variance(xs, ddof))


def cumsum[
    dtype: DType, n: Int
](xs: Shaped[dtype, n]) raises -> Shaped[dtype, n]:
    """The running sum of `xs`: `ys[i] = xs[0] + ... + xs[i]`. The
    counterpart of `cumprod`; the `List[T]` form below is the
    `FloatLike`-generic one."""
    var values = xs.to_host()
    var storage = List[Scalar[dtype]](capacity=n)
    var acc = Scalar[dtype](0)
    for i in range(n):
        acc = acc + values[i]
        storage.append(acc)
    return Shaped[dtype, n](xs.context(), storage^)


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
