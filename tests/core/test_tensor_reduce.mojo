"""Tests for `numax.core.tensor.reduce`/`reduce_rows`/`broadcast_op_rows` (CPU).

GPU coverage for `reduce_block_gpu` and for `reduce_rows`/`broadcast_op_rows`
with `gpu=True` lives in `examples/softmax.mojo`, which exercises them end to
end rather than in isolation here.
"""

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from std.testing import TestSuite, assert_almost_equal

from numax.core.tensor import (
    add_combine,
    broadcast_op_axis,
    broadcast_op_rows,
    max_combine,
    reduce,
    reduce_axis,
    reduce_rows,
)

comptime dtype = DType.float32


def sub_combine(a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    return a - b


def div_combine(a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    return a / b


def test_reduce_sums_a_1d_tensor() raises:
    comptime n = 10
    comptime layout = row_major[n]()
    var xs_storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        xs_storage.append(Scalar[dtype](i + 1))
    var xs = TileTensor(xs_storage, layout)

    var total = reduce[combine=add_combine[dtype]](xs, SIMD[dtype, 1](0))

    assert_almost_equal(total, SIMD[dtype, 1](55))


def test_reduce_finds_the_max_of_a_1d_tensor() raises:
    comptime n = 6
    comptime layout = row_major[n]()
    var xs_storage: List[Scalar[dtype]] = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
    var xs = TileTensor(xs_storage, layout)

    var m = reduce[combine=max_combine[dtype]](xs, SIMD[dtype, 1](-1e30))

    assert_almost_equal(m, SIMD[dtype, 1](9))


def test_reduce_sums_a_multidimensional_tensor_via_coalesce() raises:
    # `reduce` accepts any rank directly, the same way `map` does -- 2*2*3
    # is a genuine rank-3 tensor, not a rank-1 tensor merely labeled
    # otherwise.
    comptime layout = row_major[2, 2, 3]()
    var xs_storage = List[Scalar[dtype]](capacity=12)
    for i in range(12):
        xs_storage.append(Scalar[dtype](i + 1))
    var xs = TileTensor(xs_storage, layout)

    var total = reduce[combine=add_combine[dtype]](xs, SIMD[dtype, 1](0))

    assert_almost_equal(total, SIMD[dtype, 1](78))


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

    reduce_rows[combine=max_combine[dtype]](xs, row_max, SIMD[dtype, 1](-1e30))

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

    broadcast_op_rows[combine=sub_combine](xs, row_max, shifted)

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
    reduce_rows[combine=add_combine[dtype]](xs, row_sum, SIMD[dtype, 1](0))

    var normalized_storage = List[Scalar[dtype]](length=rows * cols, fill=0)
    var normalized = TileTensor(normalized_storage, layout2d)
    broadcast_op_rows[combine=div_combine](xs, row_sum, normalized)

    for r in range(rows):
        var row_total = SIMD[dtype, 1](0)
        for c in range(cols):
            row_total += normalized[Coord(r, c)]
        assert_almost_equal(row_total, SIMD[dtype, 1](1))


def _ramp[n: Int]() -> List[Scalar[dtype]]:
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype](i))
    return out^


def test_reduce_axis_folds_each_axis_of_a_rank_three_tensor() raises:
    # xs is 0..23 shaped (2, 3, 4), so every expected sum below is an
    # arithmetic series that can be checked by hand.
    comptime layout = row_major[2, 3, 4]()
    var storage = _ramp[24]()
    var xs = TileTensor(storage, layout)

    # axis=0: pairs (i, i+12).
    var out0 = List[Scalar[dtype]](length=12, fill=0)
    reduce_axis[combine=add_combine[dtype], axis=0](
        xs, TileTensor(out0, row_major[3, 4]()), SIMD[dtype, 1](0)
    )
    for i in range(12):
        assert_almost_equal(out0[i], Scalar[dtype](2 * i + 12))

    # axis=1: strides of 4 within each 12-element block.
    var out1 = List[Scalar[dtype]](length=8, fill=0)
    reduce_axis[combine=add_combine[dtype], axis=1](
        xs, TileTensor(out1, row_major[2, 4]()), SIMD[dtype, 1](0)
    )
    for o in range(2):
        for i in range(4):
            var want = Scalar[dtype](0)
            for k in range(3):
                want += Scalar[dtype]((o * 3 + k) * 4 + i)
            assert_almost_equal(out1[o * 4 + i], want)

    # axis=2: contiguous runs of 4.
    var out2 = List[Scalar[dtype]](length=6, fill=0)
    reduce_axis[combine=add_combine[dtype], axis=2](
        xs, TileTensor(out2, row_major[2, 3]()), SIMD[dtype, 1](0)
    )
    for r in range(6):
        assert_almost_equal(out2[r], Scalar[dtype](4 * (4 * r) + 6))


def test_reduce_axis_reproduces_reduce_rows() raises:
    # reduce_rows is reduce_axis at axis=1 on a rank-2 tensor. Checking one
    # against the other is worth more than checking both against the same
    # hand-computed numbers.
    comptime layout = row_major[3, 5]()
    var storage = List[Scalar[dtype]](capacity=15)
    for i in range(15):
        storage.append(Scalar[dtype]((i * 7) % 11) - 4.0)
    var xs = TileTensor(storage, layout)

    var rows_out = List[Scalar[dtype]](length=3, fill=0)
    var axis_out = List[Scalar[dtype]](length=3, fill=0)
    reduce_rows[combine=max_combine[dtype]](
        xs, TileTensor(rows_out, row_major[3]()), SIMD[dtype, 1](-1e30)
    )
    reduce_axis[combine=max_combine[dtype], axis=1](
        xs, TileTensor(axis_out, row_major[3]()), SIMD[dtype, 1](-1e30)
    )
    for r in range(3):
        assert_almost_equal(rows_out[r], axis_out[r])


def test_broadcast_op_axis_inverts_reduce_axis() raises:
    # Subtracting each axis-1 mean should leave every axis-1 sum at zero,
    # which checks the two functions agree about which elements share a
    # broadcast value -- a transposed convention would fail this.
    comptime layout = row_major[2, 3, 4]()
    var storage = _ramp[24]()
    var xs = TileTensor(storage, layout)

    var sums = List[Scalar[dtype]](length=8, fill=0)
    reduce_axis[combine=add_combine[dtype], axis=1](
        xs, TileTensor(sums, row_major[2, 4]()), SIMD[dtype, 1](0)
    )
    var means = List[Scalar[dtype]](length=8, fill=0)
    for i in range(8):
        means[i] = sums[i] / 3.0

    var centered = List[Scalar[dtype]](length=24, fill=0)
    broadcast_op_axis[combine=sub_combine, axis=1](
        xs,
        TileTensor(means, row_major[2, 4]()),
        TileTensor(centered, layout),
    )

    var residual = List[Scalar[dtype]](length=8, fill=0)
    reduce_axis[combine=add_combine[dtype], axis=1](
        TileTensor(centered, layout),
        TileTensor(residual, row_major[2, 4]()),
        SIMD[dtype, 1](0),
    )
    for i in range(8):
        assert_almost_equal(residual[i], SIMD[dtype, 1](0), atol=1e-5)


def test_broadcast_op_axis_reproduces_broadcast_op_rows() raises:
    comptime layout = row_major[3, 5]()
    var storage = List[Scalar[dtype]](capacity=15)
    for i in range(15):
        storage.append(Scalar[dtype](i) * 0.5)
    var values = List[Scalar[dtype]](length=3, fill=0)
    for r in range(3):
        values[r] = Scalar[dtype](r + 1)

    var rows_out = List[Scalar[dtype]](length=15, fill=0)
    var axis_out = List[Scalar[dtype]](length=15, fill=0)
    broadcast_op_rows[combine=div_combine](
        TileTensor(storage, layout),
        TileTensor(values, row_major[3]()),
        TileTensor(rows_out, layout),
    )
    broadcast_op_axis[combine=div_combine, axis=1](
        TileTensor(storage, layout),
        TileTensor(values, row_major[3]()),
        TileTensor(axis_out, layout),
    )
    for i in range(15):
        assert_almost_equal(rows_out[i], axis_out[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
