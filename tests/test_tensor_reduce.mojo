"""Tests for `ember.tensor.reduce`/`reduce_rows`/`broadcast_op_rows` (CPU).

GPU coverage for the same primitives (`reduce_block_gpu`, `reduce_rows_gpu`,
`broadcast_op_rows_gpu`) lives in `examples/softmax.mojo`, which exercises
them end to end rather than in isolation here.
"""

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from std.testing import TestSuite, assert_almost_equal

from ember import Plain
from ember.tensor import add_op, broadcast_op_rows, max_op, reduce, reduce_rows

comptime dtype = DType.float32


def wrap_plain(x: SIMD[dtype, 1]) -> Plain[dtype, 1]:
    return Plain[dtype, 1](x)


def unwrap_plain(p: Plain[dtype, 1]) -> SIMD[dtype, 1]:
    return p.v


def sub_op(a: Plain[dtype, 1], b: Plain[dtype, 1]) -> Plain[dtype, 1]:
    return a + (-b)


def div_op(a: Plain[dtype, 1], b: Plain[dtype, 1]) -> Plain[dtype, 1]:
    return a / b


def test_reduce_sums_a_1d_tensor() raises:
    comptime n = 10
    comptime layout = row_major[n]()
    var xs_storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        xs_storage.append(Scalar[dtype](i + 1))
    var xs = TileTensor(xs_storage, layout)

    var total = reduce[
        wrap=wrap_plain, unwrap=unwrap_plain, op=add_op[Plain[dtype, 1]]
    ](xs, SIMD[dtype, 1](0))

    assert_almost_equal(total, SIMD[dtype, 1](55))


def test_reduce_finds_the_max_of_a_1d_tensor() raises:
    comptime n = 6
    comptime layout = row_major[n]()
    var xs_storage: List[Scalar[dtype]] = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
    var xs = TileTensor(xs_storage, layout)

    var m = reduce[
        wrap=wrap_plain, unwrap=unwrap_plain, op=max_op[Plain[dtype, 1]]
    ](xs, SIMD[dtype, 1](-1e30))

    assert_almost_equal(m, SIMD[dtype, 1](9))


def test_reduce_rows_finds_each_row_max() raises:
    comptime rows = 2
    comptime cols = 4
    comptime layout2d = row_major[rows, cols]()
    comptime layout1d = row_major[rows]()

    var xs_storage: List[Scalar[dtype]] = [
        0.0,
        1.0,
        2.0,
        3.0,
        10.0,
        11.0,
        12.0,
        13.0,
    ]
    var xs = TileTensor(xs_storage, layout2d)
    var row_max_storage = List[Scalar[dtype]](length=rows, fill=0)
    var row_max = TileTensor(row_max_storage, layout1d)

    reduce_rows[
        wrap=wrap_plain, unwrap=unwrap_plain, op=max_op[Plain[dtype, 1]]
    ](xs, row_max, SIMD[dtype, 1](-1e30))

    assert_almost_equal(row_max[0], SIMD[dtype, 1](3))
    assert_almost_equal(row_max[1], SIMD[dtype, 1](13))


def test_broadcast_op_rows_subtracts_row_max() raises:
    comptime rows = 2
    comptime cols = 4
    comptime layout2d = row_major[rows, cols]()
    comptime layout1d = row_major[rows]()

    var xs_storage: List[Scalar[dtype]] = [
        0.0,
        1.0,
        2.0,
        3.0,
        10.0,
        11.0,
        12.0,
        13.0,
    ]
    var xs = TileTensor(xs_storage, layout2d)
    var row_max_storage: List[Scalar[dtype]] = [3.0, 13.0]
    var row_max = TileTensor(row_max_storage, layout1d)
    var shifted_storage = List[Scalar[dtype]](length=rows * cols, fill=0)
    var shifted = TileTensor(shifted_storage, layout2d)

    broadcast_op_rows[wrap=wrap_plain, unwrap=unwrap_plain, op=sub_op](
        xs, row_max, shifted
    )

    for r in range(rows):
        for c in range(cols):
            assert_almost_equal(
                shifted[Coord(r, c)], xs[Coord(r, c)] - row_max[r]
            )


def test_broadcast_op_rows_divides_by_row_sum() raises:
    comptime rows = 2
    comptime cols = 3
    comptime layout2d = row_major[rows, cols]()
    comptime layout1d = row_major[rows]()

    var xs_storage: List[Scalar[dtype]] = [1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
    var xs = TileTensor(xs_storage, layout2d)
    var row_sum_storage = List[Scalar[dtype]](length=rows, fill=0)
    var row_sum = TileTensor(row_sum_storage, layout1d)
    reduce_rows[
        wrap=wrap_plain, unwrap=unwrap_plain, op=add_op[Plain[dtype, 1]]
    ](xs, row_sum, SIMD[dtype, 1](0))

    var normalized_storage = List[Scalar[dtype]](length=rows * cols, fill=0)
    var normalized = TileTensor(normalized_storage, layout2d)
    broadcast_op_rows[wrap=wrap_plain, unwrap=unwrap_plain, op=div_op](
        xs, row_sum, normalized
    )

    for r in range(rows):
        var row_total = SIMD[dtype, 1](0)
        for c in range(cols):
            row_total += normalized[Coord(r, c)]
        assert_almost_equal(row_total, SIMD[dtype, 1](1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
