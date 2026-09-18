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

from .tensorlike import TensorLike, is_row_major
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
    broadcast_binary,
    broadcast_binary_to,
    unary_to,
)


# Comparisons are spelled as methods (`a.lt(b)`), not operators: `a < b` on
# a SIMD vector is "Strict inequality is only defined for Scalars", and
# `a == b` above width one returns one `Bool`, splatting a lane's answer
# across the vector. The methods are the ordered comparisons; see `_ne_op`.


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


# Spelled over a `dtype` parameter rather than `DType.bool` so they type
# against a `T: TensorLike` whose `dtype` the `where` clause pins to bool:
# the checker does not rewrite `T.dtype` into `DType.bool` from the clause.
def _and_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return a & b


def _or_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return a | b


def _xor_op[
    dtype: DType, w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return a ^ b


def _not_op[dtype: DType, w: Int](x: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    return (~x).cast[DType.bool]()


def equal[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[DType.bool, T.LayoutType] where is_row_major[T]:
    """`a == b`, elementwise. `numpy.equal`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_to[
        T,
        DType.bool,
        op=_eq_op[dtype, _],
        gpu=gpu,
        name="equal",
    ](a, b)


def not_equal[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[DType.bool, T.LayoutType] where is_row_major[T]:
    """`a != b`, elementwise. `numpy.not_equal`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_to[
        T,
        DType.bool,
        op=_ne_op[dtype, _],
        gpu=gpu,
        name="not_equal",
    ](a, b)


def less[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[DType.bool, T.LayoutType] where is_row_major[T]:
    """`a < b`, elementwise. `numpy.less`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_to[T, DType.bool, op=_lt_op[dtype, _], gpu=gpu, name="less"](
        a, b
    )


def less_equal[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[DType.bool, T.LayoutType] where is_row_major[T]:
    """`a <= b`, elementwise. `numpy.less_equal`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_to[
        T,
        DType.bool,
        op=_le_op[dtype, _],
        gpu=gpu,
        name="less_equal",
    ](a, b)


def greater[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[DType.bool, T.LayoutType] where is_row_major[T]:
    """`a > b`, elementwise. `numpy.greater`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_to[
        T,
        DType.bool,
        op=_gt_op[dtype, _],
        gpu=gpu,
        name="greater",
    ](a, b)


def greater_equal[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[DType.bool, T.LayoutType] where is_row_major[T]:
    """`a >= b`, elementwise. `numpy.greater_equal`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return binary_to[
        T,
        DType.bool,
        op=_ge_op[dtype, _],
        gpu=gpu,
        name="greater_equal",
    ](a, b)


def isnan[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[DType.bool, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Which elements are NaN. `numpy.isnan`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary_to[
        T,
        DType.bool,
        op=_isnan_op[dtype, _],
        gpu=gpu,
        name="isnan",
    ](a)


def isinf[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[DType.bool, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Which elements are an infinity of either sign. `numpy.isinf`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary_to[
        T,
        DType.bool,
        op=_isinf_op[dtype, _],
        gpu=gpu,
        name="isinf",
    ](a)


def isfinite[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[DType.bool, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Which elements are neither NaN nor infinite. `numpy.isfinite`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary_to[
        T,
        DType.bool,
        op=_isfinite_op[dtype, _],
        gpu=gpu,
        name="isfinite",
    ](a)


def isposinf[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[DType.bool, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Which elements are `+inf`. `numpy.isposinf`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary_to[
        T,
        DType.bool,
        op=_isposinf_op[dtype, _],
        gpu=gpu,
        name="isposinf",
    ](a)


def isneginf[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[DType.bool, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Which elements are `-inf`. `numpy.isneginf`."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    return unary_to[
        T,
        DType.bool,
        op=_isneginf_op[dtype, _],
        gpu=gpu,
        name="isneginf",
    ](a)


def logical_and[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    T.dtype == DType.bool and is_row_major[T]
):
    """`a and b`, elementwise. `numpy.logical_and`."""
    comptime LayoutType = T.LayoutType
    return binary[T, op=_and_op[T.dtype, _], gpu=gpu, name="logical_and"](a, b)


def logical_or[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    T.dtype == DType.bool and is_row_major[T]
):
    """`a or b`, elementwise. `numpy.logical_or`."""
    comptime LayoutType = T.LayoutType
    return binary[T, op=_or_op[T.dtype, _], gpu=gpu, name="logical_or"](a, b)


def logical_xor[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Tensor[T.dtype, T.LayoutType] where (
    T.dtype == DType.bool and is_row_major[T]
):
    """`a xor b`, elementwise. `numpy.logical_xor`."""
    comptime LayoutType = T.LayoutType
    return binary[T, op=_xor_op[T.dtype, _], gpu=gpu, name="logical_xor"](a, b)


def logical_not[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> Tensor[DType.bool, T.LayoutType] where (
    T.dtype == DType.bool and is_row_major[T]
):
    """`not a`, elementwise. `numpy.logical_not`."""
    comptime LayoutType = T.LayoutType
    return unary_to[
        T, DType.bool, op=_not_op[T.dtype, _], gpu=gpu, name="logical_not"
    ](a)


def all[T: TensorLike](a: T) raises -> Bool where T.dtype == DType.bool:
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


def any[T: TensorLike](a: T) raises -> Bool where T.dtype == DType.bool:
    """Whether any element is true. `numpy.any`. A host read that
    short-circuits, tier 2 on the same terms as `all` above, and hiding the
    builtin `any` in an importing file the same way."""
    var values = a.to_host()
    for i in range(a.size()):
        if values[i]:
            return True
    return False


def isclose[
    T: TensorLike, gpu: Bool = False
](
    a: T,
    b: T,
    rtol: Scalar[T.dtype] = 1e-5,
    atol: Scalar[T.dtype] = 1e-8,
) raises -> Tensor[DType.bool, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Which elements are within `atol + rtol * abs(b)`. `numpy.isclose`.

    Two run-time tolerances rather than a `thin` step's parameters, so this
    carries its own body instead of going through `_drive.binary_to`; both
    are captured by value at `Scalar[dtype]` and splatted to the launch
    width, which keeps a `Float64` out of a kernel Metal has no `double` in.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    if not _check_device[gpu=gpu](a) or not _check_device[gpu=gpu](b):
        _notice[gpu]("isclose")
        var n = a.size()
        var a_values = a.to_host()
        var b_values = b.to_host()
        var walked = List[Scalar[DType.bool]](length=n, fill=False)
        for i in range(n):
            var diff = abs(a_values[i] - b_values[i])
            walked[i] = diff <= atol + rtol * abs(b_values[i])
        return Tensor[DType.bool, LayoutType](
            a.context(), a.view().layout, walked^
        )

    var ctx = a.context()
    var out = Tensor[DType.bool, LayoutType]._uninitialized(
        ctx, a.view().layout
    )
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
    T: TensorLike, gpu: Bool = False
](
    a: T,
    b: T,
    rtol: Scalar[T.dtype] = 1e-5,
    atol: Scalar[T.dtype] = 1e-8,
) raises -> Bool where (T.dtype.is_floating_point() and is_row_major[T]):
    """Whether every element is within tolerance. `numpy.allclose`.

    The comparison runs where `gpu` says; the fold back to one `Bool` is
    `all`'s host read.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    var close = isclose[T, gpu=gpu](a, b, rtol, atol)
    return all(close)


def array_equal[
    T: TensorLike, gpu: Bool = False
](a: T, b: T) raises -> Bool where is_row_major[T]:
    """Whether every element is exactly equal. `numpy.array_equal`.

    Exact, so NaN compares unequal to itself and two tensors of NaN are not
    equal -- matching NumPy. The comparison runs where `gpu` says and the
    fold back to one `Bool` is `all`'s host read.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    var same = equal[T, gpu=gpu](a, b)
    return all(same)


# The broadcasting forms, matching `numax.core.ops` and
# `numax.core.elementwise`: the same comparison at two shapes NumPy would
# broadcast, returning a `Dynamic` mask because the broadcast extents are
# run-time values.


def equal[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    DType.bool, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """`a == b` at two broadcastable shapes. `numpy.equal`."""
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary_to[
        A,
        B,
        DType.bool,
        op=_eq_op[dtype, _],
        gpu=gpu,
        name="equal",
    ](a, b)


def not_equal[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    DType.bool, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """`a != b` at two broadcastable shapes. `numpy.not_equal`."""
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary_to[
        A,
        B,
        DType.bool,
        op=_ne_op[dtype, _],
        gpu=gpu,
        name="not_equal",
    ](a, b)


def less[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    DType.bool, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """`a < b` at two broadcastable shapes. `numpy.less`."""
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary_to[
        A,
        B,
        DType.bool,
        op=_lt_op[dtype, _],
        gpu=gpu,
        name="less",
    ](a, b)


def less_equal[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    DType.bool, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """`a <= b` at two broadcastable shapes. `numpy.less_equal`."""
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary_to[
        A,
        B,
        DType.bool,
        op=_le_op[dtype, _],
        gpu=gpu,
        name="less_equal",
    ](a, b)


def greater[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    DType.bool, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """`a > b` at two broadcastable shapes. `numpy.greater`."""
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary_to[
        A,
        B,
        DType.bool,
        op=_gt_op[dtype, _],
        gpu=gpu,
        name="greater",
    ](a, b)


def greater_equal[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](a: A, b: B) raises -> Dynamic[
    DType.bool, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (A.dtype == B.dtype and is_row_major[A] and is_row_major[B]):
    """`a >= b` at two broadcastable shapes. `numpy.greater_equal`."""
    comptime dtype = A.dtype
    comptime ALayout = A.LayoutType
    comptime BLayout = B.LayoutType
    return broadcast_binary_to[
        A,
        B,
        DType.bool,
        op=_ge_op[dtype, _],
        gpu=gpu,
        name="greater_equal",
    ](a, b)


def logical_and[
    A: TensorLike, B: TensorLike, gpu: Bool = False
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == DType.bool
    and B.dtype == DType.bool
    and is_row_major[A]
    and is_row_major[B]
):
    """`a and b` at two broadcastable shapes. `numpy.logical_and`."""
    return broadcast_binary[
        A, B, op=_and_op[A.dtype, _], gpu=gpu, name="logical_and"
    ](a, b)


def logical_or[
    A: TensorLike, B: TensorLike, gpu: Bool = False
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == DType.bool
    and B.dtype == DType.bool
    and is_row_major[A]
    and is_row_major[B]
):
    """`a or b` at two broadcastable shapes. `numpy.logical_or`."""
    return broadcast_binary[
        A, B, op=_or_op[A.dtype, _], gpu=gpu, name="logical_or"
    ](a, b)


def logical_xor[
    A: TensorLike, B: TensorLike, gpu: Bool = False
](a: A, b: B) raises -> Dynamic[
    A.dtype, _BroadcastRank[A.LayoutType, B.LayoutType]
] where (
    A.dtype == DType.bool
    and B.dtype == DType.bool
    and is_row_major[A]
    and is_row_major[B]
):
    """`a xor b` at two broadcastable shapes. `numpy.logical_xor`."""
    return broadcast_binary[
        A, B, op=_xor_op[A.dtype, _], gpu=gpu, name="logical_xor"
    ](a, b)
