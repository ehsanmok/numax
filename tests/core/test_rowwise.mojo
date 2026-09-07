"""`numax.core.rowwise` against `numax.core.tensor.reduce_axis`.

The claim these assert is agreement: the MAX-delegated reduction and the
hand-written `combine` fold compute the same thing over the same tensor, at
every rank and axis, so a caller can move between them. Where they cannot
agree bit for bit -- a reassociated floating-point sum -- the tolerance is
stated rather than assumed away.

CPU only, because the tests aggregate runs on GPU-less CI. The `target="gpu"`
path of the same functions is exercised by hand on real hardware; see
`.cursor/rules/max-feedback.mdc`.
"""

from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_almost_equal, assert_equal

from numax.core.array import Shaped, zeros
from numax.core.rowwise import max_axis, sum_axis
from numax.core.tensor import reduce_axis

comptime dtype = DType.float64


def _add[dtype: DType](a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    return a + b


def _larger[
    dtype: DType
](a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    return a if a > b else b


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
    var a = Shaped[dtype, rows, cols](ctx, _ramp(rows * cols))

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
    var a = Shaped[dtype, rows, cols](ctx, _ramp(rows * cols))

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
    var a = Shaped[dtype, rows, cols](ctx, _ramp(rows * cols))

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
    var a = Shaped[dtype, d0, d1, d2](ctx, _ramp(d0 * d1 * d2))

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
    var a = Shaped[dtype, n](ctx, _ramp(n))

    var got = zeros[dtype, 1](ctx)
    sum_axis[dtype, _, _, axis=0](a.view(), got.view())

    var want = Float64(0)
    var values = _ramp(n)
    for i in range(n):
        want += Float64(values[i])
    assert_almost_equal(Float64(got.to_host()[0]), want)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
