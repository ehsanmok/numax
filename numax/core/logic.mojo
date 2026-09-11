"""Elementwise comparison, predicates and boolean reduction over `Tensor`.

**This module is tier 2.** Every routine walks a host copy of the elements,
the same shape as `numax.core.sorting`; the reductions additionally short-circuit,
which is data-dependent control flow.

The elementwise half is *mathematically* tier 1 -- fixed work per element, no
branching -- and `numax.core.tensor.map_to`/`zip_to` are the tier-1, GPU-launchable
primitives that express it. They are not called from here because a `where`
clause proving a view is contiguous and statically shaped cannot be forwarded
through a function that is generic over `*dims`; a caller holding a concrete
shape should reach for `map_to`/`zip_to` directly when the walk needs to run
on a device.

`Plain`-only. A comparison returns truth, not a number, so there is nothing
for a `Dual` derivative or a `Compensated` error term to carry.

Truth is a `Static[DType.bool, *dims]`, not a same-dtype tensor of 0/1. That
is what makes `greater(a, b)` compose with `logical_and`, and it is the type
`numax.core.sorting.extract`/`select` take, so `select(a > b, x, y)` needs
nothing in between. A tensor of values becomes a mask with
`numax.core.ops.astype[DType.bool]`, which is where nonzero-means-true lives.

Each comparison and each `logical_*` has a second overload taking two
shapes NumPy would broadcast, returning a `Dynamic[DType.bool]` mask --
still truth, so it still composes with `logical_and` and still feeds
`select`.

Names: `where` is a Mojo keyword, so the selection stays
`numax.core.sorting.select`. `all` and `any` shadow the builtins of those
names in an importing file, which is the price `numpy`'s own spelling
charges in Python too.
"""

from std.math import (
    isfinite as _std_isfinite,
    isinf as _std_isinf,
    isnan as _std_isnan,
)

from layout.tile_layout import TensorLayout, row_major
from .array import (
    Dynamic,
    Static,
    Tensor,
    _dyn_shape_from,
    _extents_of,
    _product,
    _stretch_strides,
    _strides_of,
    broadcast_shapes,
)


def _eq_step[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    return a == b


def _ne_step[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    return a != b


def _lt_step[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    return a < b


def _le_step[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    return a <= b


def _gt_step[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    return a > b


def _ge_step[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    return a >= b


def _isnan_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[DType.bool, w] where dtype.is_floating_point():
    return _std_isnan(x)


def _isinf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[DType.bool, w] where dtype.is_floating_point():
    return _std_isinf(x)


def _isfinite_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[DType.bool, w] where dtype.is_floating_point():
    return _std_isfinite(x)


def _isposinf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[DType.bool, w] where dtype.is_floating_point():
    return _std_isinf(x) & (x > 0)


def _isneginf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[DType.bool, w] where dtype.is_floating_point():
    return _std_isinf(x) & (x < 0)


def _and_step[
    w: Int
](a: SIMD[DType.bool, w], b: SIMD[DType.bool, w]) -> SIMD[DType.bool, w]:
    return a & b


def _or_step[
    w: Int
](a: SIMD[DType.bool, w], b: SIMD[DType.bool, w]) -> SIMD[DType.bool, w]:
    return a | b


def _xor_step[
    w: Int
](a: SIMD[DType.bool, w], b: SIMD[DType.bool, w]) -> SIMD[DType.bool, w]:
    return a ^ b


def _not_step[w: Int](x: SIMD[DType.bool, w]) -> SIMD[DType.bool, w]:
    return ~x


def _nonzero_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    return x != 0


def _compare[
    dtype: DType,
    LayoutType: TensorLayout,
    step: def[w: Int](SIMD[dtype, w], SIMD[dtype, w]) thin -> SIMD[
        DType.bool, w
    ],
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
]:
    var n = a.size()
    var a_values = a.to_host()
    var b_values = b.to_host()
    var out = List[Scalar[DType.bool]](length=n, fill=False)
    for i in range(n):
        out[i] = step[1](a_values[i], b_values[i])[0]
    return Tensor[DType.bool, LayoutType](a.context(), a.layout, out^)


def _compare_broadcast[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    step: def[w: Int](SIMD[dtype, w], SIMD[dtype, w]) thin -> SIMD[
        DType.bool, w
    ],
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    DType.bool, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`_compare` over two shapes NumPy would broadcast.

    The truth it returns is a `Dynamic[DType.bool]` rather than a `Static`,
    since the broadcast extents are run-time values -- but it is still a
    boolean tensor, so it still composes with `logical_and` and still feeds
    `numax.core.sorting.select`.
    """
    comptime rank = ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
    var a_extents = _extents_of(a)
    var b_extents = _extents_of(b)
    var extents = broadcast_shapes(a_extents, b_extents)
    var a_strides = _stretch_strides(a_extents, _strides_of(a), rank)
    var b_strides = _stretch_strides(b_extents, _strides_of(b), rank)

    var count = 1
    for d in range(rank):
        count *= extents[d]

    var a_values = a.to_host()
    var b_values = b.to_host()
    var out = List[Scalar[DType.bool]](length=count, fill=False)
    for flat in range(count):
        var rem = flat
        var ai = 0
        var bi = 0
        for k in range(rank):
            var d = rank - 1 - k
            var c = rem % extents[d]
            rem //= extents[d]
            ai += c * a_strides[d]
            bi += c * b_strides[d]
        out[flat] = step[1](a_values[ai], b_values[bi])[0]

    return Dynamic[DType.bool, rank](
        a.context(), row_major(_dyn_shape_from[rank](extents)), out^
    )


def _predicate[
    dtype: DType,
    LayoutType: TensorLayout,
    step: def[w: Int](SIMD[dtype, w]) thin -> SIMD[DType.bool, w],
](a: Tensor[dtype, LayoutType]) raises -> Tensor[DType.bool, LayoutType]:
    var n = a.size()
    var values = a.to_host()
    var out = List[Scalar[DType.bool]](length=n, fill=False)
    for i in range(n):
        out[i] = step[1](values[i])[0]
    return Tensor[DType.bool, LayoutType](a.context(), a.layout, out^)


def equal[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
]:
    """`a == b`, elementwise. `numpy.equal`."""
    return _compare[dtype, LayoutType, step=_eq_step[dtype, _]](a, b)


def not_equal[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
]:
    """`a != b`, elementwise. `numpy.not_equal`."""
    return _compare[dtype, LayoutType, step=_ne_step[dtype, _]](a, b)


def less[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
]:
    """`a < b`, elementwise. `numpy.less`."""
    return _compare[dtype, LayoutType, step=_lt_step[dtype, _]](a, b)


def less_equal[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
]:
    """`a <= b`, elementwise. `numpy.less_equal`."""
    return _compare[dtype, LayoutType, step=_le_step[dtype, _]](a, b)


def greater[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
]:
    """`a > b`, elementwise. `numpy.greater`."""
    return _compare[dtype, LayoutType, step=_gt_step[dtype, _]](a, b)


def greater_equal[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
]:
    """`a >= b`, elementwise. `numpy.greater_equal`."""
    return _compare[dtype, LayoutType, step=_ge_step[dtype, _]](a, b)


def isnan[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
] where dtype.is_floating_point():
    """Which elements are NaN. `numpy.isnan`."""
    return _predicate[dtype, LayoutType, step=_isnan_step[dtype, _]](a)


def isinf[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
] where dtype.is_floating_point():
    """Which elements are an infinity of either sign. `numpy.isinf`."""
    return _predicate[dtype, LayoutType, step=_isinf_step[dtype, _]](a)


def isfinite[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
] where dtype.is_floating_point():
    """Which elements are neither NaN nor infinite. `numpy.isfinite`."""
    return _predicate[dtype, LayoutType, step=_isfinite_step[dtype, _]](a)


def isposinf[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
] where dtype.is_floating_point():
    """Which elements are `+inf`. `numpy.isposinf`."""
    return _predicate[dtype, LayoutType, step=_isposinf_step[dtype, _]](a)


def isneginf[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
] where dtype.is_floating_point():
    """Which elements are `-inf`. `numpy.isneginf`."""
    return _predicate[dtype, LayoutType, step=_isneginf_step[dtype, _]](a)


def logical_and[
    LayoutType: TensorLayout
](
    a: Tensor[DType.bool, LayoutType], b: Tensor[DType.bool, LayoutType]
) raises -> Tensor[DType.bool, LayoutType]:
    """`a and b`, elementwise. `numpy.logical_and`."""
    return _compare[DType.bool, LayoutType, step=_and_step](a, b)


def logical_or[
    LayoutType: TensorLayout
](
    a: Tensor[DType.bool, LayoutType], b: Tensor[DType.bool, LayoutType]
) raises -> Tensor[DType.bool, LayoutType]:
    """`a or b`, elementwise. `numpy.logical_or`."""
    return _compare[DType.bool, LayoutType, step=_or_step](a, b)


def logical_xor[
    LayoutType: TensorLayout
](
    a: Tensor[DType.bool, LayoutType], b: Tensor[DType.bool, LayoutType]
) raises -> Tensor[DType.bool, LayoutType]:
    """`a xor b`, elementwise. `numpy.logical_xor`."""
    return _compare[DType.bool, LayoutType, step=_xor_step](a, b)


def logical_not[
    LayoutType: TensorLayout
](a: Tensor[DType.bool, LayoutType]) raises -> Tensor[DType.bool, LayoutType]:
    """`not a`, elementwise. `numpy.logical_not`."""
    return _predicate[DType.bool, LayoutType, step=_not_step](a)


def all[
    LayoutType: TensorLayout
](a: Tensor[DType.bool, LayoutType]) raises -> Bool:
    """Whether every element is true. `numpy.all`.

    Short-circuits on the first false, which is why this is tier 2. This
    name hides Mojo's builtin `all` in any file that imports it -- the
    price `from numpy import all` charges in Python too, and worth paying
    for the name a NumPy caller actually reaches for.
    """
    var values = a.to_host()
    for i in range(a.size()):
        if not values[i]:
            return False
    return True


def any[
    LayoutType: TensorLayout
](a: Tensor[DType.bool, LayoutType]) raises -> Bool:
    """Whether any element is true. `numpy.any`. Short-circuits, tier 2.
    Hides the builtin `any` in an importing file, like `all` above."""
    var values = a.to_host()
    for i in range(a.size()):
        if values[i]:
            return True
    return False


def isclose[
    dtype: DType, LayoutType: TensorLayout
](
    a: Tensor[dtype, LayoutType],
    b: Tensor[dtype, LayoutType],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
) raises -> Tensor[DType.bool, LayoutType] where dtype.is_floating_point():
    """Which elements are within `atol + rtol * abs(b)`. `numpy.isclose`.

    Tolerances are runtime values, so this is a host walk rather than a
    `zip_to` -- a `thin` step cannot close over them.
    """
    var n = a.size()
    var a_values = a.to_host()
    var b_values = b.to_host()
    var out = List[Scalar[DType.bool]](length=n, fill=False)
    for i in range(n):
        var diff = abs(a_values[i] - b_values[i])
        out[i] = diff <= atol + rtol * abs(b_values[i])
    return Tensor[DType.bool, LayoutType](a.context(), a.layout, out^)


def allclose[
    dtype: DType, LayoutType: TensorLayout
](
    a: Tensor[dtype, LayoutType],
    b: Tensor[dtype, LayoutType],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
) raises -> Bool where dtype.is_floating_point():
    """Whether every element is within tolerance. `numpy.allclose`."""
    var close = isclose[dtype, LayoutType](a, b, rtol, atol)
    return all[LayoutType](close)


def array_equal[
    dtype: DType, LayoutType: TensorLayout
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Bool:
    """Whether every element is exactly equal. `numpy.array_equal`.

    Exact, so NaN compares unequal to itself and two tensors of NaN are not
    equal -- matching NumPy.
    """
    var same = equal[dtype, LayoutType](a, b)
    return all[LayoutType](same)


# The broadcasting forms, matching `numax.core.ops` and
# `numax.core.elementwise`: the same comparison at two shapes NumPy would
# broadcast, returning a `Dynamic` mask because the broadcast extents are
# run-time values.


def equal[
    dtype: DType, ALayout: TensorLayout, BLayout: TensorLayout
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    DType.bool, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`a == b` at two broadcastable shapes. `numpy.equal`."""
    return _compare_broadcast[dtype, ALayout, BLayout, step=_eq_step[dtype, _]](
        a, b
    )


def not_equal[
    dtype: DType, ALayout: TensorLayout, BLayout: TensorLayout
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    DType.bool, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`a != b` at two broadcastable shapes. `numpy.not_equal`."""
    return _compare_broadcast[dtype, ALayout, BLayout, step=_ne_step[dtype, _]](
        a, b
    )


def less[
    dtype: DType, ALayout: TensorLayout, BLayout: TensorLayout
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    DType.bool, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`a < b` at two broadcastable shapes. `numpy.less`."""
    return _compare_broadcast[dtype, ALayout, BLayout, step=_lt_step[dtype, _]](
        a, b
    )


def less_equal[
    dtype: DType, ALayout: TensorLayout, BLayout: TensorLayout
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    DType.bool, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`a <= b` at two broadcastable shapes. `numpy.less_equal`."""
    return _compare_broadcast[dtype, ALayout, BLayout, step=_le_step[dtype, _]](
        a, b
    )


def greater[
    dtype: DType, ALayout: TensorLayout, BLayout: TensorLayout
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    DType.bool, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`a > b` at two broadcastable shapes. `numpy.greater`."""
    return _compare_broadcast[dtype, ALayout, BLayout, step=_gt_step[dtype, _]](
        a, b
    )


def greater_equal[
    dtype: DType, ALayout: TensorLayout, BLayout: TensorLayout
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    DType.bool, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`a >= b` at two broadcastable shapes. `numpy.greater_equal`."""
    return _compare_broadcast[dtype, ALayout, BLayout, step=_ge_step[dtype, _]](
        a, b
    )


def logical_and[
    ALayout: TensorLayout, BLayout: TensorLayout
](
    a: Tensor[DType.bool, ALayout], b: Tensor[DType.bool, BLayout]
) raises -> Dynamic[
    DType.bool, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`a and b` at two broadcastable shapes. `numpy.logical_and`."""
    return _compare_broadcast[DType.bool, ALayout, BLayout, step=_and_step](
        a, b
    )


def logical_or[
    ALayout: TensorLayout, BLayout: TensorLayout
](
    a: Tensor[DType.bool, ALayout], b: Tensor[DType.bool, BLayout]
) raises -> Dynamic[
    DType.bool, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`a or b` at two broadcastable shapes. `numpy.logical_or`."""
    return _compare_broadcast[DType.bool, ALayout, BLayout, step=_or_step](a, b)


def logical_xor[
    ALayout: TensorLayout, BLayout: TensorLayout
](
    a: Tensor[DType.bool, ALayout], b: Tensor[DType.bool, BLayout]
) raises -> Dynamic[
    DType.bool, ALayout.rank if ALayout.rank > BLayout.rank else BLayout.rank
]:
    """`a xor b` at two broadcastable shapes. `numpy.logical_xor`."""
    return _compare_broadcast[DType.bool, ALayout, BLayout, step=_xor_step](
        a, b
    )
