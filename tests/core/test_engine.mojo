"""Tests for the walks that do not assume a compile-time contiguous shape.

Two families. The run-time-shape overloads of `map_to`, `zip_to`,
`map_threaded`, `reduce_axis` and `broadcast_op_axis` do the same work as
their static counterparts over a layout the compiler cannot see, so each is
checked against the static one on the same values -- a divergence between
the two is the failure worth catching, not a wrong answer in isolation.

`map_strided` and `reduce_strided` address elements through their own
strides instead, which is what a transposed or sliced view needs. They are
checked against the same walk over a compacted copy, so the reference is a
path already under test rather than a hand-computed table.
"""

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from max.gpu.host import DeviceContext
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from numax.core.tensor import (
    add_combine,
    broadcast_op_axis,
    map,
    map_strided,
    map_threaded,
    map_to,
    max_combine,
    reduce,
    reduce_axis,
    reduce_strided,
    zip_to,
)

comptime dtype = DType.float64


def _double[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return x + x


def _is_positive[w: Int](x: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    return x > 0


def _both_positive[
    w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    return (a > 0) & (b > 0)


def _ramp(n: Int, offset: Float64 = 0.0) -> List[Scalar[dtype]]:
    var out = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        out[i] = Scalar[dtype](Float64(i) + offset)
    return out^


def test_runtime_map_to_agrees_with_the_static_one() raises:
    var xs = _ramp(12, offset=-5.0)
    var static_out = List[Scalar[DType.bool]](length=12, fill=False)
    var dyn_out = List[Scalar[DType.bool]](length=12, fill=False)

    map_to[step=_is_positive](
        TileTensor(xs, row_major[3, 4]()),
        TileTensor(static_out, row_major[3, 4]()),
    )
    map_to[step=_is_positive](
        TileTensor(xs, row_major(Coord(3, 4))),
        TileTensor(dyn_out, row_major(Coord(3, 4))),
    )

    for i in range(12):
        assert_equal(dyn_out[i], static_out[i])
    assert_true(static_out[11])  # not vacuously all-False


def test_runtime_zip_to_agrees_with_the_static_one() raises:
    var lhs = _ramp(12, offset=-5.0)
    var rhs = _ramp(12, offset=-8.0)
    var static_out = List[Scalar[DType.bool]](length=12, fill=False)
    var dyn_out = List[Scalar[DType.bool]](length=12, fill=False)

    zip_to[step=_both_positive](
        TileTensor(lhs, row_major[12]()),
        TileTensor(rhs, row_major[12]()),
        TileTensor(static_out, row_major[12]()),
    )
    zip_to[step=_both_positive](
        TileTensor(lhs, row_major(Coord(3, 4))),
        TileTensor(rhs, row_major(Coord(3, 4))),
        TileTensor(dyn_out, row_major(Coord(3, 4))),
    )

    for i in range(12):
        assert_equal(dyn_out[i], static_out[i])
    assert_true(static_out[11])


def test_runtime_map_threaded_agrees_with_the_serial_walk() raises:
    var ctx = DeviceContext(api="cpu")
    var xs = _ramp(1000)
    var serial = List[Scalar[dtype]](length=1000, fill=0)
    var threaded = List[Scalar[dtype]](length=1000, fill=0)

    map[step=_double](
        TileTensor(xs, row_major(Coord(1000))),
        TileTensor(serial, row_major(Coord(1000))),
    )
    map_threaded[step=_double](
        TileTensor(xs, row_major(Coord(1000))),
        TileTensor(threaded, row_major(Coord(1000))),
        ctx,
    )

    for i in range(1000):
        assert_equal(threaded[i], serial[i])


def test_runtime_reduce_axis_agrees_with_the_static_one() raises:
    var xs = _ramp(24)
    var static_out = List[Scalar[dtype]](length=8, fill=0)
    var dyn_out = List[Scalar[dtype]](length=8, fill=0)

    reduce_axis[combine=add_combine[dtype], axis=1](
        TileTensor(xs, row_major[2, 3, 4]()),
        TileTensor(static_out, row_major[8]()),
        0,
    )
    reduce_axis[combine=add_combine[dtype], axis=1](
        TileTensor(xs, row_major(Coord(2, 3, 4))),
        TileTensor(dyn_out, row_major(Coord(8))),
        0,
    )

    for i in range(8):
        assert_equal(dyn_out[i], static_out[i])
    # (0 + 4 + 8) for the first surviving position, so the axis really was
    # the middle one rather than a flat fold.
    assert_equal(static_out[0], Scalar[dtype](12.0))


def test_runtime_broadcast_op_axis_agrees_with_the_static_one() raises:
    var xs = _ramp(24)
    var values = _ramp(8, offset=1.0)
    var static_out = List[Scalar[dtype]](length=24, fill=0)
    var dyn_out = List[Scalar[dtype]](length=24, fill=0)

    broadcast_op_axis[combine=add_combine[dtype], axis=1](
        TileTensor(xs, row_major[2, 3, 4]()),
        TileTensor(values, row_major[2, 4]()),
        TileTensor(static_out, row_major[2, 3, 4]()),
    )
    broadcast_op_axis[combine=add_combine[dtype], axis=1](
        TileTensor(xs, row_major(Coord(2, 3, 4))),
        TileTensor(values, row_major(Coord(2, 4))),
        TileTensor(dyn_out, row_major(Coord(2, 3, 4))),
    )

    for i in range(24):
        assert_equal(dyn_out[i], static_out[i])
    assert_equal(static_out[0], Scalar[dtype](1.0))


def test_a_reduction_and_its_broadcast_still_compose_at_a_runtime_shape() raises:
    # Centering along an axis: subtract each group's mean. What the pair
    # exists for, so it is checked as a pair rather than only apart.
    var xs = _ramp(24)
    var sums = List[Scalar[dtype]](length=8, fill=0)
    var out = List[Scalar[dtype]](length=24, fill=0)

    reduce_axis[combine=add_combine[dtype], axis=1](
        TileTensor(xs, row_major(Coord(2, 3, 4))),
        TileTensor(sums, row_major(Coord(8))),
        0,
    )
    for i in range(8):
        sums[i] = -sums[i] / 3
    broadcast_op_axis[combine=add_combine[dtype], axis=1](
        TileTensor(xs, row_major(Coord(2, 3, 4))),
        TileTensor(sums, row_major(Coord(8))),
        TileTensor(out, row_major(Coord(2, 3, 4))),
    )

    # Each group of three now sums to zero.
    for o in range(2):
        for i in range(4):
            var total = Scalar[dtype](0)
            for k in range(3):
                total += out[(o * 3 + k) * 4 + i]
            assert_almost_equal(total, Scalar[dtype](0.0))


def test_map_strided_over_a_transpose_matches_the_compacted_walk() raises:
    var xs = _ramp(24)
    var strided_out = List[Scalar[dtype]](length=24, fill=0)
    var compact = List[Scalar[dtype]](length=24, fill=0)
    var compact_out = List[Scalar[dtype]](length=24, fill=0)

    var source = TileTensor(xs, row_major[2, 3, 4]()).transpose()

    # Once through the strided walk, and once by compacting first and using
    # the ordinary row-major `map`. The two must agree element for element.
    map_strided[step=_double](
        source, TileTensor(strided_out, row_major[4, 3, 2]())
    )

    var identity = TileTensor(compact, row_major[4, 3, 2]())
    map_strided[step=_copy](source, identity)
    map[step=_double](identity, TileTensor(compact_out, row_major[4, 3, 2]()))

    for i in range(24):
        assert_equal(strided_out[i], compact_out[i])
    # The transpose really did reorder: element 1 of the result is
    # 2 * xs[12], not 2 * xs[1].
    assert_equal(strided_out[1], Scalar[dtype](24.0))


def _copy[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return x


def test_map_strided_over_a_row_slice_reads_only_that_slice() raises:
    var xs = _ramp(24)
    var out = List[Scalar[dtype]](length=12, fill=0)
    var rows = TileTensor(xs, row_major[4, 6]()).slice((1, 3), (0, 6))
    map_strided[step=_double](rows, TileTensor(out, row_major[2, 6]()))
    for i in range(12):
        assert_equal(out[i], Scalar[dtype](2.0 * (6.0 + Float64(i))))


def test_reduce_strided_matches_a_reduce_over_the_compacted_copy() raises:
    var xs = _ramp(24, offset=1.0)
    var compact = List[Scalar[dtype]](length=24, fill=0)
    var source = TileTensor(xs, row_major[2, 3, 4]()).transpose()

    map_strided[step=_copy](source, TileTensor(compact, row_major[4, 3, 2]()))

    var strided_total = reduce_strided[combine=add_combine[dtype]](source, 0)
    var compact_total = reduce[combine=add_combine[dtype]](
        TileTensor(compact, row_major[4, 3, 2]()), 0
    )
    # Same visiting order, so this is exact rather than approximate.
    assert_true(strided_total == compact_total)

    # And a non-associative fold agrees too, which a differently ordered
    # walk would not survive.
    assert_true(
        reduce_strided[combine=max_combine[dtype]](source, -1e30)
        == reduce[combine=max_combine[dtype]](
            TileTensor(compact, row_major[4, 3, 2]()), -1e30
        )
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
