"""Elementwise comparison, predicates and boolean reduction over `Tensor`.

**Tier 2 in shape, both targets in fact**, on the same terms as
`numax.core.ops`, `numax.core.elementwise` and `numax.core.rowwise`. These
are `Plain`-only -- a comparison returns truth, not a number, so there is
nothing for a `Dual` derivative or a `Compensated` error term to carry --
but they are not host-only. Every comparison and every predicate runs
through `numax.core._drive`, which launches one capturing body on the target
its `gpu: Bool` parameter names: `max.algorithm.elementwise` on the device,
the same threaded walk on a large host tensor, a serial SIMD loop on a small
one.

`gpu: Bool = False` is the last compile-time parameter of every elementwise
routine here, so `greater(a, b)` is the host spelling and
`greater[gpu=True](a, b)` the device one. Asking for a target the tensor's
memory does not live on is not an error: the call falls back to a host walk
and prints one line on `stderr` naming the spelling that would not have.

`all` and `any` stay host reads. They return a `Bool`, so the answer has to
cross back over the launch boundary whatever the mask cost, and both
short-circuit on the first element that decides the question -- data
dependent control flow, which is what tier 2 means in the strict sense.
`allclose` and `array_equal` are a device comparison followed by that host
read, so they take `gpu` and pass it down.

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

from layout import Coord
from layout.tile_layout import TensorLayout

from .array import Dynamic, Tensor
from ._drive import (
    _BroadcastRank,
    _check_device,
    _flat,
    _flat_out,
    _launch,
    _notice,
    _width,
    binary,
    binary_to,
    broadcast_binary_to,
    unary_to,
)


# The per-element operations. Each is generic in the SIMD width, because the
# drivers call them at the launch width rather than one lane at a time, and
# each comparison is spelled as a method (`a.lt(b)`) rather than an operator.
# `a < b` on a SIMD vector is `Strict inequality is only defined for Scalars`
# and `a == b` at a width above one returns a single `Bool` -- which would
# splat one lane's answer across the whole vector. The methods are the
# ordered comparisons, which is what IEEE asks for everywhere except
# not-equal; see `_ne_op`.


def _eq_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    return a.eq(b)


def _ne_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    # `~a.eq(b)` rather than `a.ne(b)`: `SIMD.ne` is the *ordered*
    # comparison, so it answers false for a NaN against anything, including
    # another NaN. NumPy's `not_equal` and Mojo's own scalar `!=` are
    # unordered -- true whenever the two are not equal, NaN included -- and
    # the complement of ordered equality is exactly that.
    return ~a.eq(b)


def _lt_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    return a.lt(b)


def _le_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    return a.le(b)


def _gt_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    return a.gt(b)


def _ge_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    return a.ge(b)


def _isnan_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[DType.bool, w] where dtype.is_floating_point():
    return _std_isnan(x)


def _isinf_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[DType.bool, w] where dtype.is_floating_point():
    return _std_isinf(x)


def _isfinite_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[DType.bool, w] where dtype.is_floating_point():
    return _std_isfinite(x)


def _isposinf_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[DType.bool, w] where dtype.is_floating_point():
    return _std_isinf(x) & x.gt(0)


def _isneginf_op[
    dtype: DType, w: Int
](x: SIMD[dtype, w]) -> SIMD[DType.bool, w] where dtype.is_floating_point():
    return _std_isinf(x) & x.lt(0)


def _and_op[
    w: Int
](a: SIMD[DType.bool, w], b: SIMD[DType.bool, w]) -> SIMD[DType.bool, w]:
    return a & b


def _or_op[
    w: Int
](a: SIMD[DType.bool, w], b: SIMD[DType.bool, w]) -> SIMD[DType.bool, w]:
    return a | b


def _xor_op[
    w: Int
](a: SIMD[DType.bool, w], b: SIMD[DType.bool, w]) -> SIMD[DType.bool, w]:
    return a ^ b


def _not_op[w: Int](x: SIMD[DType.bool, w]) -> SIMD[DType.bool, w]:
    return ~x


def equal[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
]:
    """`a == b`, elementwise. `numpy.equal`."""
    return binary_to[
        dtype,
        DType.bool,
        LayoutType,
        op=_eq_op[dtype, _],
        gpu=gpu,
        name="equal",
    ](a, b)


def not_equal[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
]:
    """`a != b`, elementwise. `numpy.not_equal`."""
    return binary_to[
        dtype,
        DType.bool,
        LayoutType,
        op=_ne_op[dtype, _],
        gpu=gpu,
        name="not_equal",
    ](a, b)


def less[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
]:
    """`a < b`, elementwise. `numpy.less`."""
    return binary_to[
        dtype, DType.bool, LayoutType, op=_lt_op[dtype, _], gpu=gpu, name="less"
    ](a, b)


def less_equal[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
]:
    """`a <= b`, elementwise. `numpy.less_equal`."""
    return binary_to[
        dtype,
        DType.bool,
        LayoutType,
        op=_le_op[dtype, _],
        gpu=gpu,
        name="less_equal",
    ](a, b)


def greater[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
]:
    """`a > b`, elementwise. `numpy.greater`."""
    return binary_to[
        dtype,
        DType.bool,
        LayoutType,
        op=_gt_op[dtype, _],
        gpu=gpu,
        name="greater",
    ](a, b)


def greater_equal[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
]:
    """`a >= b`, elementwise. `numpy.greater_equal`."""
    return binary_to[
        dtype,
        DType.bool,
        LayoutType,
        op=_ge_op[dtype, _],
        gpu=gpu,
        name="greater_equal",
    ](a, b)


def isnan[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
] where dtype.is_floating_point():
    """Which elements are NaN. `numpy.isnan`."""
    return unary_to[
        dtype,
        DType.bool,
        LayoutType,
        op=_isnan_op[dtype, _],
        gpu=gpu,
        name="isnan",
    ](a)


def isinf[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
] where dtype.is_floating_point():
    """Which elements are an infinity of either sign. `numpy.isinf`."""
    return unary_to[
        dtype,
        DType.bool,
        LayoutType,
        op=_isinf_op[dtype, _],
        gpu=gpu,
        name="isinf",
    ](a)


def isfinite[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
] where dtype.is_floating_point():
    """Which elements are neither NaN nor infinite. `numpy.isfinite`."""
    return unary_to[
        dtype,
        DType.bool,
        LayoutType,
        op=_isfinite_op[dtype, _],
        gpu=gpu,
        name="isfinite",
    ](a)


def isposinf[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
] where dtype.is_floating_point():
    """Which elements are `+inf`. `numpy.isposinf`."""
    return unary_to[
        dtype,
        DType.bool,
        LayoutType,
        op=_isposinf_op[dtype, _],
        gpu=gpu,
        name="isposinf",
    ](a)


def isneginf[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType]) raises -> Tensor[
    DType.bool, LayoutType
] where dtype.is_floating_point():
    """Which elements are `-inf`. `numpy.isneginf`."""
    return unary_to[
        dtype,
        DType.bool,
        LayoutType,
        op=_isneginf_op[dtype, _],
        gpu=gpu,
        name="isneginf",
    ](a)


def logical_and[
    LayoutType: TensorLayout, gpu: Bool = False
](
    a: Tensor[DType.bool, LayoutType], b: Tensor[DType.bool, LayoutType]
) raises -> Tensor[DType.bool, LayoutType]:
    """`a and b`, elementwise. `numpy.logical_and`."""
    return binary[
        DType.bool, LayoutType, op=_and_op, gpu=gpu, name="logical_and"
    ](a, b)


def logical_or[
    LayoutType: TensorLayout, gpu: Bool = False
](
    a: Tensor[DType.bool, LayoutType], b: Tensor[DType.bool, LayoutType]
) raises -> Tensor[DType.bool, LayoutType]:
    """`a or b`, elementwise. `numpy.logical_or`."""
    return binary[
        DType.bool, LayoutType, op=_or_op, gpu=gpu, name="logical_or"
    ](a, b)


def logical_xor[
    LayoutType: TensorLayout, gpu: Bool = False
](
    a: Tensor[DType.bool, LayoutType], b: Tensor[DType.bool, LayoutType]
) raises -> Tensor[DType.bool, LayoutType]:
    """`a xor b`, elementwise. `numpy.logical_xor`."""
    return binary[
        DType.bool, LayoutType, op=_xor_op, gpu=gpu, name="logical_xor"
    ](a, b)


def logical_not[
    LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[DType.bool, LayoutType]) raises -> Tensor[DType.bool, LayoutType]:
    """`not a`, elementwise. `numpy.logical_not`."""
    return unary_to[
        DType.bool,
        DType.bool,
        LayoutType,
        op=_not_op,
        gpu=gpu,
        name="logical_not",
    ](a)


def all[
    LayoutType: TensorLayout
](a: Tensor[DType.bool, LayoutType]) raises -> Bool:
    """Whether every element is true. `numpy.all`.

    A host read, and no `gpu` parameter: the answer is one `Bool`, so it has
    to cross back over the launch boundary whatever produced the mask, and
    this short-circuits on the first false -- data-dependent control flow,
    tier 2 in the strict sense rather than in shape only. This name hides
    Mojo's builtin `all` in any file that imports it -- the price
    `from numpy import all` charges in Python too, and worth paying for the
    name a NumPy caller actually reaches for.
    """
    var values = a.to_host()
    for i in range(a.size()):
        if not values[i]:
            return False
    return True


def any[
    LayoutType: TensorLayout
](a: Tensor[DType.bool, LayoutType]) raises -> Bool:
    """Whether any element is true. `numpy.any`. A host read that
    short-circuits, tier 2 on the same terms as `all` above, and hiding the
    builtin `any` in an importing file the same way."""
    var values = a.to_host()
    for i in range(a.size()):
        if values[i]:
            return True
    return False


def isclose[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](
    a: Tensor[dtype, LayoutType],
    b: Tensor[dtype, LayoutType],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
) raises -> Tensor[DType.bool, LayoutType] where dtype.is_floating_point():
    """Which elements are within `atol + rtol * abs(b)`. `numpy.isclose`.

    Two run-time tolerances rather than a `thin` step's parameters, so this
    carries its own body instead of going through `_drive.binary_to`; both
    are captured by value at `Scalar[dtype]` and splatted to the launch
    width, which keeps a `Float64` out of a kernel Metal has no `double` in.
    """
    if not _check_device[gpu=gpu](a) or not _check_device[gpu=gpu](b):
        _notice[gpu]("isclose")
        var n = a.size()
        var a_values = a.to_host()
        var b_values = b.to_host()
        var walked = List[Scalar[DType.bool]](length=n, fill=False)
        for i in range(n):
            var diff = abs(a_values[i] - b_values[i])
            walked[i] = diff <= atol + rtol * abs(b_values[i])
        return Tensor[DType.bool, LayoutType](a.context(), a.layout, walked^)

    var ctx = a.context()
    var out = Tensor[DType.bool, LayoutType]._uninitialized(ctx, a.layout)
    var xs = _flat(a)
    var zs = _flat(b)
    var ys = _flat_out(out)

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var zs, var ys, var rtol, var atol}:
        var av = xs.load[width](coord)
        var bv = zs.load[width](coord)
        ys.store[width](
            coord,
            abs(av - bv).le(
                SIMD[dtype, width](atol) + SIMD[dtype, width](rtol) * abs(bv)
            ),
        )

    _launch[gpu=gpu, lanes=_width[dtype, gpu]()](body, a.size(), ctx)
    return out^


def allclose[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](
    a: Tensor[dtype, LayoutType],
    b: Tensor[dtype, LayoutType],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
) raises -> Bool where dtype.is_floating_point():
    """Whether every element is within tolerance. `numpy.allclose`.

    The comparison runs where `gpu` says; the fold back to one `Bool` is
    `all`'s host read.
    """
    var close = isclose[dtype, LayoutType, gpu=gpu](a, b, rtol, atol)
    return all[LayoutType](close)


def array_equal[
    dtype: DType, LayoutType: TensorLayout, gpu: Bool = False
](a: Tensor[dtype, LayoutType], b: Tensor[dtype, LayoutType]) raises -> Bool:
    """Whether every element is exactly equal. `numpy.array_equal`.

    Exact, so NaN compares unequal to itself and two tensors of NaN are not
    equal -- matching NumPy. The comparison runs where `gpu` says and the
    fold back to one `Bool` is `all`'s host read.
    """
    var same = equal[dtype, LayoutType, gpu=gpu](a, b)
    return all[LayoutType](same)


# The broadcasting forms, matching `numax.core.ops` and
# `numax.core.elementwise`: the same comparison at two shapes NumPy would
# broadcast, returning a `Dynamic` mask because the broadcast extents are
# run-time values.


def equal[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    DType.bool, _BroadcastRank[ALayout, BLayout]
]:
    """`a == b` at two broadcastable shapes. `numpy.equal`."""
    return broadcast_binary_to[
        dtype,
        DType.bool,
        ALayout,
        BLayout,
        op=_eq_op[dtype, _],
        gpu=gpu,
        name="equal",
    ](a, b)


def not_equal[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    DType.bool, _BroadcastRank[ALayout, BLayout]
]:
    """`a != b` at two broadcastable shapes. `numpy.not_equal`."""
    return broadcast_binary_to[
        dtype,
        DType.bool,
        ALayout,
        BLayout,
        op=_ne_op[dtype, _],
        gpu=gpu,
        name="not_equal",
    ](a, b)


def less[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    DType.bool, _BroadcastRank[ALayout, BLayout]
]:
    """`a < b` at two broadcastable shapes. `numpy.less`."""
    return broadcast_binary_to[
        dtype,
        DType.bool,
        ALayout,
        BLayout,
        op=_lt_op[dtype, _],
        gpu=gpu,
        name="less",
    ](a, b)


def less_equal[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    DType.bool, _BroadcastRank[ALayout, BLayout]
]:
    """`a <= b` at two broadcastable shapes. `numpy.less_equal`."""
    return broadcast_binary_to[
        dtype,
        DType.bool,
        ALayout,
        BLayout,
        op=_le_op[dtype, _],
        gpu=gpu,
        name="less_equal",
    ](a, b)


def greater[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    DType.bool, _BroadcastRank[ALayout, BLayout]
]:
    """`a > b` at two broadcastable shapes. `numpy.greater`."""
    return broadcast_binary_to[
        dtype,
        DType.bool,
        ALayout,
        BLayout,
        op=_gt_op[dtype, _],
        gpu=gpu,
        name="greater",
    ](a, b)


def greater_equal[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](a: Tensor[dtype, ALayout], b: Tensor[dtype, BLayout]) raises -> Dynamic[
    DType.bool, _BroadcastRank[ALayout, BLayout]
]:
    """`a >= b` at two broadcastable shapes. `numpy.greater_equal`."""
    return broadcast_binary_to[
        dtype,
        DType.bool,
        ALayout,
        BLayout,
        op=_ge_op[dtype, _],
        gpu=gpu,
        name="greater_equal",
    ](a, b)


def logical_and[
    ALayout: TensorLayout, BLayout: TensorLayout, gpu: Bool = False
](
    a: Tensor[DType.bool, ALayout], b: Tensor[DType.bool, BLayout]
) raises -> Dynamic[DType.bool, _BroadcastRank[ALayout, BLayout]]:
    """`a and b` at two broadcastable shapes. `numpy.logical_and`."""
    return broadcast_binary_to[
        DType.bool,
        DType.bool,
        ALayout,
        BLayout,
        op=_and_op,
        gpu=gpu,
        name="logical_and",
    ](a, b)


def logical_or[
    ALayout: TensorLayout, BLayout: TensorLayout, gpu: Bool = False
](
    a: Tensor[DType.bool, ALayout], b: Tensor[DType.bool, BLayout]
) raises -> Dynamic[DType.bool, _BroadcastRank[ALayout, BLayout]]:
    """`a or b` at two broadcastable shapes. `numpy.logical_or`."""
    return broadcast_binary_to[
        DType.bool,
        DType.bool,
        ALayout,
        BLayout,
        op=_or_op,
        gpu=gpu,
        name="logical_or",
    ](a, b)


def logical_xor[
    ALayout: TensorLayout, BLayout: TensorLayout, gpu: Bool = False
](
    a: Tensor[DType.bool, ALayout], b: Tensor[DType.bool, BLayout]
) raises -> Dynamic[DType.bool, _BroadcastRank[ALayout, BLayout]]:
    """`a xor b` at two broadcastable shapes. `numpy.logical_xor`."""
    return broadcast_binary_to[
        DType.bool,
        DType.bool,
        ALayout,
        BLayout,
        op=_xor_op,
        gpu=gpu,
        name="logical_xor",
    ](a, b)
