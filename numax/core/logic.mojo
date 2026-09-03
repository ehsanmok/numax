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

Truth is a `Tensor[DType.bool, *dims]`, not a same-dtype tensor of 0/1. That
is what makes `greater(a, b)` compose with `logical_and` and read as a mask.
`numax.core.sorting.extract`/`select` predate this module and take a same-dtype
condition where nonzero means true; `to_mask` converts between the two.

Names: `all`/`any` are Mojo builtins and `where` is a keyword, so the
reductions here are `all_true`/`any_true` and the selection stays
`numax.core.sorting.select`. Same collision rule that produced `variance`,
`stddev` and `any_nonzero`.
"""

from std.math import (
    isfinite as _std_isfinite,
    isinf as _std_isinf,
    isnan as _std_isnan,
)

from .array import Tensor, _product


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
    *dims: Int,
    step: def[w: Int](SIMD[dtype, w], SIMD[dtype, w]) thin -> SIMD[
        DType.bool, w
    ],
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    DType.bool, *dims
]:
    comptime n = _product[*dims]()
    var a_values = a.to_host()
    var b_values = b.to_host()
    var out = List[Scalar[DType.bool]](length=n, fill=False)
    for i in range(n):
        out[i] = step[1](a_values[i], b_values[i])[0]
    return Tensor[DType.bool, *dims](a.context(), out^)


def _predicate[
    dtype: DType,
    *dims: Int,
    step: def[w: Int](SIMD[dtype, w]) thin -> SIMD[DType.bool, w],
](a: Tensor[dtype, *dims]) raises -> Tensor[DType.bool, *dims]:
    comptime n = _product[*dims]()
    var values = a.to_host()
    var out = List[Scalar[DType.bool]](length=n, fill=False)
    for i in range(n):
        out[i] = step[1](values[i])[0]
    return Tensor[DType.bool, *dims](a.context(), out^)


def equal[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    DType.bool, *dims
]:
    """`a == b`, elementwise. `numpy.equal`."""
    return _compare[dtype, *dims, step=_eq_step[dtype, _]](a, b)


def not_equal[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    DType.bool, *dims
]:
    """`a != b`, elementwise. `numpy.not_equal`."""
    return _compare[dtype, *dims, step=_ne_step[dtype, _]](a, b)


def less[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    DType.bool, *dims
]:
    """`a < b`, elementwise. `numpy.less`."""
    return _compare[dtype, *dims, step=_lt_step[dtype, _]](a, b)


def less_equal[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    DType.bool, *dims
]:
    """`a <= b`, elementwise. `numpy.less_equal`."""
    return _compare[dtype, *dims, step=_le_step[dtype, _]](a, b)


def greater[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    DType.bool, *dims
]:
    """`a > b`, elementwise. `numpy.greater`."""
    return _compare[dtype, *dims, step=_gt_step[dtype, _]](a, b)


def greater_equal[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Tensor[
    DType.bool, *dims
]:
    """`a >= b`, elementwise. `numpy.greater_equal`."""
    return _compare[dtype, *dims, step=_ge_step[dtype, _]](a, b)


def isnan[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    DType.bool, *dims
] where dtype.is_floating_point():
    """Which elements are NaN. `numpy.isnan`."""
    return _predicate[dtype, *dims, step=_isnan_step[dtype, _]](a)


def isinf[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    DType.bool, *dims
] where dtype.is_floating_point():
    """Which elements are an infinity of either sign. `numpy.isinf`."""
    return _predicate[dtype, *dims, step=_isinf_step[dtype, _]](a)


def isfinite[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    DType.bool, *dims
] where dtype.is_floating_point():
    """Which elements are neither NaN nor infinite. `numpy.isfinite`."""
    return _predicate[dtype, *dims, step=_isfinite_step[dtype, _]](a)


def isposinf[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    DType.bool, *dims
] where dtype.is_floating_point():
    """Which elements are `+inf`. `numpy.isposinf`."""
    return _predicate[dtype, *dims, step=_isposinf_step[dtype, _]](a)


def isneginf[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[
    DType.bool, *dims
] where dtype.is_floating_point():
    """Which elements are `-inf`. `numpy.isneginf`."""
    return _predicate[dtype, *dims, step=_isneginf_step[dtype, _]](a)


def logical_and[
    *dims: Int
](a: Tensor[DType.bool, *dims], b: Tensor[DType.bool, *dims]) raises -> Tensor[
    DType.bool, *dims
]:
    """`a and b`, elementwise. `numpy.logical_and`."""
    return _compare[DType.bool, *dims, step=_and_step](a, b)


def logical_or[
    *dims: Int
](a: Tensor[DType.bool, *dims], b: Tensor[DType.bool, *dims]) raises -> Tensor[
    DType.bool, *dims
]:
    """`a or b`, elementwise. `numpy.logical_or`."""
    return _compare[DType.bool, *dims, step=_or_step](a, b)


def logical_xor[
    *dims: Int
](a: Tensor[DType.bool, *dims], b: Tensor[DType.bool, *dims]) raises -> Tensor[
    DType.bool, *dims
]:
    """`a xor b`, elementwise. `numpy.logical_xor`."""
    return _compare[DType.bool, *dims, step=_xor_step](a, b)


def logical_not[
    *dims: Int
](a: Tensor[DType.bool, *dims]) raises -> Tensor[DType.bool, *dims]:
    """`not a`, elementwise. `numpy.logical_not`."""
    return _predicate[DType.bool, *dims, step=_not_step](a)


def to_mask[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) raises -> Tensor[DType.bool, *dims]:
    """Nonzero-means-true, as a boolean tensor.

    The bridge to `numax.core.sorting.extract`/`select`, which take a same-dtype
    condition rather than a mask because they predate this module.
    """
    return _predicate[dtype, *dims, step=_nonzero_step[dtype, _]](a)


def all_true[*dims: Int](a: Tensor[DType.bool, *dims]) raises -> Bool:
    """Whether every element is true. `numpy.all`, named around the builtin.

    Short-circuits on the first false, which is why this is tier 2.
    """
    var values = a.to_host()
    for i in range(_product[*dims]()):
        if not values[i]:
            return False
    return True


def any_true[*dims: Int](a: Tensor[DType.bool, *dims]) raises -> Bool:
    """Whether any element is true. `numpy.any`. Short-circuits, tier 2."""
    var values = a.to_host()
    for i in range(_product[*dims]()):
        if values[i]:
            return True
    return False


def isclose[
    dtype: DType, *dims: Int
](
    a: Tensor[dtype, *dims],
    b: Tensor[dtype, *dims],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
) raises -> Tensor[DType.bool, *dims] where dtype.is_floating_point():
    """Which elements are within `atol + rtol * abs(b)`. `numpy.isclose`.

    Tolerances are runtime values, so this is a host walk rather than a
    `zip_to` -- a `thin` step cannot close over them.
    """
    comptime n = _product[*dims]()
    var a_values = a.to_host()
    var b_values = b.to_host()
    var out = List[Scalar[DType.bool]](length=n, fill=False)
    for i in range(n):
        var diff = abs(a_values[i] - b_values[i])
        out[i] = diff <= atol + rtol * abs(b_values[i])
    return Tensor[DType.bool, *dims](a.context(), out^)


def allclose[
    dtype: DType, *dims: Int
](
    a: Tensor[dtype, *dims],
    b: Tensor[dtype, *dims],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
) raises -> Bool where dtype.is_floating_point():
    """Whether every element is within tolerance. `numpy.allclose`."""
    var close = isclose[dtype, *dims](a, b, rtol, atol)
    return all_true[*dims](close)


def array_equal[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], b: Tensor[dtype, *dims]) raises -> Bool:
    """Whether every element is exactly equal. `numpy.array_equal`.

    Exact, so NaN compares unequal to itself and two tensors of NaN are not
    equal -- matching NumPy.
    """
    var same = equal[dtype, *dims](a, b)
    return all_true[*dims](same)
