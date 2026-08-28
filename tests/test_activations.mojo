"""Tests for `numax.special`'s `relu`/`leaky_relu`/`gelu`/`softmax`."""

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from std.math import exp
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Dual, Plain, gelu, leaky_relu, relu, softmax

comptime dtype = DType.float64
comptime width = 1
comptime D = Dual[Plain[dtype, width]]


def pv(x: Float64) -> Plain[dtype, width]:
    return Plain[dtype, width](SIMD[dtype, width](x))


def test_relu_matches_max_with_zero() raises:
    for x_raw in [-2.0, -0.5, 0.0, 0.5, 2.0]:
        var x = Plain[dtype, width](SIMD[dtype, width](x_raw))
        assert_almost_equal(relu(x).v, SIMD[dtype, width](max(x_raw, 0.0)))


def test_relu_derivative_is_zero_or_one() raises:
    var neg = D(pv(-1), pv(1))
    assert_almost_equal(relu(neg).deriv.v, SIMD[dtype, width](0))

    var pos = D(pv(1), pv(1))
    assert_almost_equal(relu(pos).deriv.v, SIMD[dtype, width](1))


def test_leaky_relu_matches_closed_form() raises:
    comptime alpha = 0.1
    for x_raw in [-2.0, -0.5, 0.5, 2.0]:
        var x = Plain[dtype, width](SIMD[dtype, width](x_raw))
        var expected = x_raw if x_raw >= 0 else alpha * x_raw
        assert_almost_equal(
            leaky_relu(x, alpha).v, SIMD[dtype, width](expected)
        )


def test_leaky_relu_derivative_is_one_or_alpha() raises:
    comptime alpha = 0.1
    var neg = D(pv(-1), pv(1))
    assert_almost_equal(
        leaky_relu(neg, alpha).deriv.v, SIMD[dtype, width](alpha)
    )

    var pos = D(pv(1), pv(1))
    assert_almost_equal(leaky_relu(pos, alpha).deriv.v, SIMD[dtype, width](1))


def test_gelu_at_zero_is_zero() raises:
    var x = Plain[dtype, width](SIMD[dtype, width](0))
    assert_almost_equal(gelu(x).v, SIMD[dtype, width](0))


def test_gelu_matches_known_value() raises:
    # Reference from PyTorch's tanh-approximation GELU at x=1.
    var x = Plain[dtype, width](SIMD[dtype, width](1.0))
    assert_almost_equal(
        gelu(x).v, SIMD[dtype, width](0.8411919906082768), atol=1e-6
    )


def test_softmax_rows_sum_to_one_and_match_reference() raises:
    comptime rows = 2
    comptime cols = 4
    comptime layout2d = row_major[rows, cols]()
    comptime layout1d = row_major[rows]()

    var xs_storage: List[Scalar[dtype]] = [
        1.0,
        2.0,
        3.0,
        4.0,
        -1.0,
        0.0,
        1.0,
        1000.0,
    ]
    var xs = TileTensor(xs_storage, layout2d)
    var tmp_storage = List[Scalar[dtype]](length=rows * cols, fill=0)
    var tmp = TileTensor(tmp_storage, layout2d)
    var ys_storage = List[Scalar[dtype]](length=rows * cols, fill=0)
    var ys = TileTensor(ys_storage, layout2d)
    var row_max_storage = List[Scalar[dtype]](length=rows, fill=0)
    var row_max = TileTensor(row_max_storage, layout1d)
    var row_sum_storage = List[Scalar[dtype]](length=rows, fill=0)
    var row_sum = TileTensor(row_sum_storage, layout1d)

    softmax(xs, tmp, ys, row_max, row_sum)

    for r in range(rows):
        var total = SIMD[dtype, 1](0)
        for c in range(cols):
            total += ys[Coord(r, c)]
        assert_almost_equal(total, SIMD[dtype, 1](1))

    # Row 0, a mild input, checked against a direct `Float64` reference.
    var row0: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    var row0_max = 4.0
    var row0_exp = List[Float64]()
    var row0_denom = 0.0
    for v in row0:
        var e = exp(v - row0_max)
        row0_exp.append(e)
        row0_denom += e
    for c in range(cols):
        assert_almost_equal(
            ys[Coord(0, c)], SIMD[dtype, 1](row0_exp[c] / row0_denom)
        )

    # Row 1 has a very large value (1000) -- softmax's shift-by-max is what
    # keeps `exp` from overflowing here; the largest input should get
    # essentially all the probability mass.
    assert_true(Float64(ys[Coord(1, 3)]) > 0.999)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
