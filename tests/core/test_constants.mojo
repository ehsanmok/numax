"""Tests for `numax.core.constants`, `numax.stats`'s integer/boolean draws, and
`numax.core.ops.invert`.

The constants are checked against their float64 literals and against a
`FloatLike` round trip; the draws are checked for range, shape and
reproducibility under a fixed seed rather than for a distribution, which
`tests/stats/test_random.mojo` already covers for the continuous families.
"""

from std.math import cos, sin
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax import Plain
from numax.core.array import Tensor
from numax.core.constants import e, e_at, pi, pi_at
from numax.core.logic import all, any
from numax.core.ops import invert
from numax.stats import randbool, randint, seed

comptime dtype = DType.float64
comptime P = Plain[DType.float64, 1]


def test_pi_and_e_match_their_literals() raises:
    assert_almost_equal(pi, 3.141592653589793)
    assert_almost_equal(e, 2.718281828459045)


def test_pi_closes_the_circle() raises:
    assert_almost_equal(sin(pi), 0.0, atol=1e-15)
    assert_almost_equal(cos(pi), -1.0)


def test_constants_round_trip_through_floatlike() raises:
    assert_almost_equal(Float64(pi_at[P]().v), pi)
    assert_almost_equal(Float64(e_at[P]().v), e)


def test_randint_stays_in_range_and_shape() raises:
    var ctx = DeviceContext(api="cpu")
    var draws = randint[DType.int64, 64](ctx, -5, 5)
    assert_equal(draws.num_elements, 64)
    var values = draws.to_host()
    for i in range(64):
        assert_true(values[i] >= -5)
        assert_true(values[i] < 5)


def test_randint_is_reproducible_under_a_fixed_seed() raises:
    var ctx = DeviceContext(api="cpu")
    seed(2026)
    var first = randint[DType.int64, 16](ctx, 0, 100).to_host()
    seed(2026)
    var second = randint[DType.int64, 16](ctx, 0, 100).to_host()
    for i in range(16):
        assert_equal(first[i], second[i])


def test_randbool_honours_the_extremes() raises:
    var ctx = DeviceContext(api="cpu")
    var never = randbool[32](ctx, 0.0)
    var always = randbool[32](ctx, 1.0)
    assert_true(not any(never))
    assert_true(all(always))


def test_randbool_preserves_rank() raises:
    var ctx = DeviceContext(api="cpu")
    var draws = randbool[2, 3](ctx)
    assert_equal(draws.num_elements, 6)
    assert_equal(draws.rank, 2)


def test_invert_is_bitwise_not() raises:
    var ctx = DeviceContext(api="cpu")
    var values = List[Scalar[DType.int32]](capacity=3)
    values.append(0)
    values.append(1)
    values.append(-1)
    var a = Tensor[DType.int32, 3](ctx, values^)
    var inverted = invert(a).to_host()
    assert_equal(Int(inverted[0]), -1)
    assert_equal(Int(inverted[1]), -2)
    assert_equal(Int(inverted[2]), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
