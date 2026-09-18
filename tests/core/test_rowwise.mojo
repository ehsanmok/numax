"""`numax.core.rowwise` against `numax.core.tensor`'s `reduce`/`reduce_axis`.

The claim these assert is agreement: the MAX-delegated reduction and the
hand-written `combine` fold compute the same thing over the same tensor, at
every rank and axis, so a caller can move between them. Where they cannot
agree bit for bit -- a reassociated floating-point sum or product -- the
tolerance is stated rather than assumed away; `max`, `min` and the two
`arg` reductions are exact and asserted exactly.

CPU only, because the tests aggregate runs on GPU-less CI. The `target="gpu"`
path of the same functions is exercised by hand on real hardware.
"""

from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.utils import IndexList

from algorithm.rowwise_types import RowCoord
from numax.core.array import Static, zeros, zeros_dyn
from numax.core._drive import _flat
from numax.core.rowwise import (
    argmax_all,
    argmin_all,
    max_axis,
    min_axis,
    prod_axis,
    reduce_all,
    sum_axis,
)
from numax.core.tensor import reduce, reduce_axis

comptime dtype = DType.float64


def _add[dtype: DType](a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    return a + b


def _larger[
    dtype: DType
](a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    return a if a > b else b


def _mul[dtype: DType](a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    return a * b


def _smaller[
    dtype: DType
](a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    return a if a < b else b


# `reduce_all`'s per-tile transform is spelled as a nested closure at every
# call site: a capture list is only legal on a nested `def`, and the
# identity transform is what a caller wants when the point is a plain
# reduction rather than `dot`'s fused multiply.


def _ramp(n: Int) -> List[Scalar[dtype]]:
    """Values with both signs and no symmetry, so a dropped lane shows up."""
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(Scalar[dtype](i % 7) - 3)
    return values^


def test_sum_axis_matches_reduce_axis_along_rows() raises:
    comptime rows = 4
    comptime cols = 6
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, rows, cols](ctx, _ramp(rows * cols))

    var want = zeros[dtype, rows](ctx)
    reduce_axis[dtype, _, _, combine=_add[dtype], axis=1](
        a.view(), want.view(), 0
    )
    var got = zeros[dtype, rows](ctx)
    sum_axis[dtype, _, _, axis=1](a.view(), got.view())

    var wh = want.to_host()
    var gh = got.to_host()
    for i in range(rows):
        assert_almost_equal(Float64(gh[i]), Float64(wh[i]))


def test_sum_axis_matches_reduce_axis_down_columns() raises:
    """The non-inner axis, where MAX puts one output column per SIMD lane.

    Reading only lane 0 of the monoid passes the row test above and fails
    this one, which is the whole reason it is here.
    """
    comptime rows = 4
    comptime cols = 6
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, rows, cols](ctx, _ramp(rows * cols))

    var want = zeros[dtype, cols](ctx)
    reduce_axis[dtype, _, _, combine=_add[dtype], axis=0](
        a.view(), want.view(), 0
    )
    var got = zeros[dtype, cols](ctx)
    sum_axis[dtype, _, _, axis=0](a.view(), got.view())

    var wh = want.to_host()
    var gh = got.to_host()
    for i in range(cols):
        assert_almost_equal(Float64(gh[i]), Float64(wh[i]))


def test_max_axis_agrees_exactly() raises:
    """A maximum is order-independent, so this one is exact, not almost."""
    comptime rows = 4
    comptime cols = 6
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, rows, cols](ctx, _ramp(rows * cols))

    var want = zeros[dtype, rows](ctx)
    reduce_axis[dtype, _, _, combine=_larger[dtype], axis=1](
        a.view(), want.view(), -1e30
    )
    var got = zeros[dtype, rows](ctx)
    max_axis[dtype, _, _, axis=1](a.view(), got.view())

    var wh = want.to_host()
    var gh = got.to_host()
    for i in range(rows):
        assert_equal(Float64(gh[i]), Float64(wh[i]))


def test_sum_axis_reduces_the_middle_axis_of_a_rank_3_tensor() raises:
    """`axis=1` of `(2, 3, 4)` leaves `(2, 4)`, flattened row-major.

    Both the surviving-axis collapse and the strided walk are wrong in
    different ways if the flat output index is computed per-rank instead of
    from the dropped axis.
    """
    comptime d0 = 2
    comptime d1 = 3
    comptime d2 = 4
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, d0, d1, d2](ctx, _ramp(d0 * d1 * d2))

    var want = zeros[dtype, d0 * d2](ctx)
    reduce_axis[dtype, _, _, combine=_add[dtype], axis=1](
        a.view(), want.view(), 0
    )
    var got = zeros[dtype, d0 * d2](ctx)
    sum_axis[dtype, _, _, axis=1](a.view(), got.view())

    var wh = want.to_host()
    var gh = got.to_host()
    for i in range(d0 * d2):
        assert_almost_equal(Float64(gh[i]), Float64(wh[i]))


def test_sum_axis_folds_a_rank_1_tensor_to_one_value() raises:
    """The whole-tensor fold, which is the rank-1 case of the axis one."""
    comptime n = 16
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, n](ctx, _ramp(n))

    var got = zeros[dtype, 1](ctx)
    sum_axis[dtype, _, _, axis=0](a.view(), got.view())

    var want = Float64(0)
    var values = _ramp(n)
    for i in range(n):
        want += Float64(values[i])
    assert_almost_equal(Float64(got.to_host()[0]), want)


def test_reduce_all_matches_reduce_for_every_monoid() raises:
    """Each of the four monoids against `numax.core.tensor.reduce`.

    `reduce` folds strictly left to right with an explicit `combine`; these
    fold through MAX's monoid. Sum and product get a tolerance because the
    monoid reassociates, min and max are asserted exactly because they
    cannot.
    """
    comptime n = 40
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, n](ctx, _ramp(n))
    var out = zeros[dtype, 1](ctx)

    @always_inline
    def _identity[
        w: Int
    ](tile: SIMD[dtype, w], idx: RowCoord[1]) {} -> SIMD[dtype, w]:
        return tile

    reduce_all[monoid="sum"](_flat(a), out.view(), _identity, n, Optional(ctx))
    assert_almost_equal(
        Float64(out.to_host()[0]),
        Float64(reduce[dtype, _, _, combine=_add[dtype]](a.view(), 0)),
    )

    reduce_all[monoid="prod"](_flat(a), out.view(), _identity, n, Optional(ctx))
    assert_almost_equal(
        Float64(out.to_host()[0]),
        Float64(reduce[dtype, _, _, combine=_mul[dtype]](a.view(), 1)),
    )

    reduce_all[monoid="max"](_flat(a), out.view(), _identity, n, Optional(ctx))
    assert_equal(
        Float64(out.to_host()[0]),
        Float64(reduce[dtype, _, _, combine=_larger[dtype]](a.view(), -1e30)),
    )

    reduce_all[monoid="min"](_flat(a), out.view(), _identity, n, Optional(ctx))
    assert_equal(
        Float64(out.to_host()[0]),
        Float64(reduce[dtype, _, _, combine=_smaller[dtype]](a.view(), 1e30)),
    )


def test_reduce_all_applies_the_per_tile_transform() raises:
    """The transform is what makes `dot` one pass: `sum(x*x)` without ever
    materializing the squared vector."""
    comptime n = 33
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, n](ctx, _ramp(n))
    var out = zeros[dtype, 1](ctx)

    @always_inline
    def square[
        w: Int
    ](tile: SIMD[dtype, w], idx: RowCoord[1]) {} -> SIMD[dtype, w]:
        return tile * tile

    reduce_all[monoid="sum"](_flat(a), out.view(), square, n, Optional(ctx))

    var values = _ramp(n)
    var want = Float64(0)
    for i in range(n):
        want += Float64(values[i]) * Float64(values[i])
    assert_almost_equal(Float64(out.to_host()[0]), want)


def test_reduce_all_agrees_between_a_static_and_a_dynamic_input() raises:
    """One signature serves both shapes: `_flat` builds the rank-1 view over
    the buffer rather than `coalesce()`, which would need every extent at
    compile time."""
    comptime rows = 5
    comptime cols = 7
    var ctx = DeviceContext(api="cpu")
    var values = _ramp(rows * cols)

    var stat = Static[dtype, rows, cols](ctx, values.copy())
    var dyn = zeros_dyn[dtype, 2](rows, cols, ctx=ctx)
    dyn.copy_from_host(values)

    var from_static = zeros[dtype, 1](ctx)
    var from_dynamic = zeros[dtype, 1](ctx)

    @always_inline
    def _identity[
        w: Int
    ](tile: SIMD[dtype, w], idx: RowCoord[1]) {} -> SIMD[dtype, w]:
        return tile

    reduce_all[monoid="sum"](
        _flat(stat), from_static.view(), _identity, rows * cols, Optional(ctx)
    )
    reduce_all[monoid="sum"](
        _flat(dyn), from_dynamic.view(), _identity, rows * cols, Optional(ctx)
    )
    assert_equal(
        Float64(from_static.to_host()[0]), Float64(from_dynamic.to_host()[0])
    )


def test_prod_axis_matches_reduce_axis() raises:
    comptime rows = 4
    comptime cols = 6
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, rows, cols](ctx, _ramp(rows * cols))

    var want = zeros[dtype, rows](ctx)
    reduce_axis[dtype, _, _, combine=_mul[dtype], axis=1](
        a.view(), want.view(), 1
    )
    var got = zeros[dtype, rows](ctx)
    prod_axis[dtype, _, _, axis=1](a.view(), got.view())

    var wh = want.to_host()
    var gh = got.to_host()
    for i in range(rows):
        assert_almost_equal(Float64(gh[i]), Float64(wh[i]))


def test_min_axis_agrees_exactly() raises:
    """`max_axis`'s mirror, down the non-inner axis so each output column is
    its own lane."""
    comptime rows = 4
    comptime cols = 6
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, rows, cols](ctx, _ramp(rows * cols))

    var want = zeros[dtype, cols](ctx)
    reduce_axis[dtype, _, _, combine=_smaller[dtype], axis=0](
        a.view(), want.view(), 1e30
    )
    var got = zeros[dtype, cols](ctx)
    min_axis[dtype, _, _, axis=0](a.view(), got.view())

    var wh = want.to_host()
    var gh = got.to_host()
    for i in range(cols):
        assert_equal(Float64(gh[i]), Float64(wh[i]))


def _with_a_tie(n: Int) -> List[Scalar[dtype]]:
    """A ramp whose extremes each occur twice, so the tie-break is pinned
    rather than incidental."""
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(Scalar[dtype](i % 5))
    return values^


def test_argmax_all_and_argmin_all_take_the_first_of_a_tie() raises:
    """The whole point of the index monoid over a host scan: MAX breaks ties
    to the lower index, which is NumPy's rule, and this asserts it against a
    host scan written to do the same."""
    comptime n = 37
    var ctx = DeviceContext(api="cpu")
    var values = _with_a_tie(n)
    var a = Static[dtype, n](ctx, values.copy())

    var best = values[0]
    var best_at = 0
    var worst = values[0]
    var worst_at = 0
    for i in range(1, n):
        if values[i] > best:
            best = values[i]
            best_at = i
        if values[i] < worst:
            worst = values[i]
            worst_at = i

    assert_equal(argmax_all[dtype](_flat(a), ctx), best_at)
    assert_equal(argmin_all[dtype](_flat(a), ctx), worst_at)
    # Pinned literally too: with `i % 5` the first 4 is at index 4 and the
    # first 0 at index 0, and every later repeat must lose.
    assert_equal(argmax_all[dtype](_flat(a), ctx), 4)
    assert_equal(argmin_all[dtype](_flat(a), ctx), 0)


def test_argmax_all_agrees_between_a_static_and_a_dynamic_input() raises:
    comptime rows = 3
    comptime cols = 11
    var ctx = DeviceContext(api="cpu")
    var values = _with_a_tie(rows * cols)

    var stat = Static[dtype, rows, cols](ctx, values.copy())
    var dyn = zeros_dyn[dtype, 2](rows, cols, ctx=ctx)
    dyn.copy_from_host(values)

    assert_equal(
        argmax_all[dtype](_flat(stat), ctx), argmax_all[dtype](_flat(dyn), ctx)
    )
    assert_equal(
        argmin_all[dtype](_flat(stat), ctx), argmin_all[dtype](_flat(dyn), ctx)
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
