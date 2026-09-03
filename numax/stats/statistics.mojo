"""NumPy-named statistics, composed from `numax.core.tensor` and `FloatLike`.

`docs/parity.md` picks statistics as a genuine `numax` gap with a
selective axis-1 lift: MAX ships no `mean`/`var`/`std`/`median`/`mode` at
all (verified directly -- `max.algorithm.functional` exports only
`elementwise`), but it does ship `argmax`/`argmin` (`nn.argmaxmin`), which
this module routes to directly rather than re-implementing (the MAX-first
check firing first, same shape as `numax.linalg.matmul`'s recommendation to
call `max.linalg.matmul` past ~8x8).

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

**Explicitly out of scope**, matching this module's own gap-only mandate:
axis-keyword reductions (`numax.core.tensor.reduce_axis`/`broadcast_op_axis`
already provide the primitive directly for a caller that needs one), and
sorting, which now lives in `numax.core.sorting` as a tier-2 module (`sort`,
`argsort`, `searchsorted`, `unique`, plus the counting and masking family).
`List.sort()` is still what `median`/`mode` reach for internally; what
changed is that a NumPy caller now has a `numax` name for it too, on the
tier-2 terms `docs/architecture.md` sets out.
"""

from layout import Coord, TileTensor
from layout.tile_layout import TensorLayout
from layout.tile_tensor import PointerStorage
from nn.argmaxmin import argmax as _nn_argmax, argmin as _nn_argmin
from layout.tile_layout import row_major

from ..core.array import Tensor
from ..core.numeric import FloatLike
from ..core.tensor import add_combine, max_combine, reduce


def mul_combine[
    dtype: DType
](a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """`a * b` -- the `combine` for `prod`."""
    return a * b


def min_combine[
    dtype: DType
](a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    """The smaller of `a`, `b` -- the `combine` for `min`."""
    return a if a < b else b


def sum[
    dtype: DType, LayoutType: TensorLayout, O: Origin
](
    xs: TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ]
) -> SIMD[dtype, 1] where (
    TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and dtype.is_floating_point()
):
    """The sum of every element of `xs`."""
    return reduce[combine=add_combine[dtype]](xs, SIMD[dtype, 1](0))


def prod[
    dtype: DType, LayoutType: TensorLayout, O: Origin
](
    xs: TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ]
) -> SIMD[dtype, 1] where (
    TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and dtype.is_floating_point()
):
    """The product of every element of `xs`."""
    return reduce[combine=mul_combine[dtype]](xs, SIMD[dtype, 1](1))


def min[
    dtype: DType, LayoutType: TensorLayout, O: Origin
](
    xs: TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ]
) -> SIMD[dtype, 1] where (
    TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and dtype.is_floating_point()
):
    """The smallest element of `xs`. `xs` must have at least one element."""
    var flat = xs.coalesce()
    var init = flat.load[1](Coord(0))
    return reduce[combine=min_combine[dtype]](xs, init)


def max[
    dtype: DType, LayoutType: TensorLayout, O: Origin
](
    xs: TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ]
) -> SIMD[dtype, 1] where (
    TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and dtype.is_floating_point()
):
    """The largest element of `xs`. `xs` must have at least one element."""
    var flat = xs.coalesce()
    var init = flat.load[1](Coord(0))
    return reduce[combine=max_combine[dtype]](xs, init)


def mean[
    dtype: DType, LayoutType: TensorLayout, O: Origin
](
    xs: TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ]
) -> SIMD[dtype, 1] where (
    TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and dtype.is_floating_point()
):
    """The arithmetic mean of `xs`.

    `Plain`-only: a mean is a single scalar with no derivative to
    propagate through the division by a plain `Int` count, so there is no
    axis-1 win here the way there is for `variance`/`std`/`cumsum` below.
    """
    var flat = xs.coalesce()
    var n = flat.num_elements()
    return sum(xs) / Scalar[dtype](n)


def median[
    dtype: DType, LayoutType: TensorLayout, O: Origin
](
    xs: TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ]
) -> SIMD[dtype, 1] where (
    TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and dtype.is_floating_point()
):
    """The median of `xs` -- the average of the two middle elements when
    `xs` has an even count, matching NumPy's default."""
    var flat = xs.coalesce()
    var n = flat.num_elements()
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(flat.load[1](Coord(i)))
    sort(values)
    if n % 2 == 1:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / Scalar[dtype](2)


def mode[
    dtype: DType, LayoutType: TensorLayout, O: Origin
](
    xs: TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ]
) -> SIMD[dtype, 1] where (
    TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and dtype.is_floating_point()
):
    """The most frequent value in `xs`; the smallest among ties, matching
    `scipy.stats.mode`'s convention."""
    var flat = xs.coalesce()
    var n = flat.num_elements()
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(flat.load[1](Coord(i)))
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
    dtype: DType, LayoutType: TensorLayout, O: Origin
](
    xs: TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ]
) raises -> Int where (
    TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and dtype.is_floating_point()
):
    """The flat index of the largest element of `xs`, via `nn.argmaxmin`
    (MAX-first: `numax` writes no comparison logic of its own here)."""
    var flat = xs.coalesce()
    var out_storage = List[Scalar[DType.int64]](length=1, fill=0)
    var out = TileTensor(out_storage, row_major[1]())
    _nn_argmax(flat, 0, out)
    return Int(out[0])


def argmin[
    dtype: DType, LayoutType: TensorLayout, O: Origin
](
    xs: TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ]
) raises -> Int where (
    TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].all_dims_known
    and TileTensor[
        dtype, LayoutType, O, Storage=PointerStorage[element_width=1]
    ].is_row_major
    and dtype.is_floating_point()
):
    """The flat index of the smallest element of `xs`, via `nn.argmaxmin`."""
    var flat = xs.coalesce()
    var out_storage = List[Scalar[DType.int64]](length=1, fill=0)
    var out = TileTensor(out_storage, row_major[1]())
    _nn_argmin(flat, 0, out)
    return Int(out[0])


def cumprod[
    dtype: DType, n: Int
](xs: Tensor[dtype, n]) raises -> Tensor[dtype, n]:
    """The running product of `xs`: `ys[i] = xs[0] * ... * xs[i]`."""
    var values = xs.to_host()
    var storage = List[Scalar[dtype]](capacity=n)
    var acc = Scalar[dtype](1)
    for i in range(n):
        acc = acc * values[i]
        storage.append(acc)
    return Tensor[dtype, n](xs.context(), storage^)


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
        var d = x + (-m)
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
