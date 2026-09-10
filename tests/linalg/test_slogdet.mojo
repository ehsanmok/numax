"""Tests for `slogdet` at both `numax.linalg` tiers.

The claim that justifies the name existing beside `det` is the one worth
testing directly: a determinant is a product of `n` diagonal entries and
leaves `float64`'s range for perfectly ordinary matrices, while a sum of
logarithms does not. `test_slogdet_survives_where_det_overflows` builds such
a matrix and checks that `det` returns infinity where `slogdet` returns a
finite, correct answer.

The rest pins agreement: `exp(logabsdet) * sign` must reproduce `det` on
matrices where `det` is representable, at both tiers, and the two tiers must
agree with each other.
"""

from std.collections import Array
from std.math import exp as exp_f64, log as log_f64
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax import Plain
from numax.core.array import Static
from numax.linalg import det, lu_factor, slogdet
from numax.linalg.array import slogdet as array_slogdet

comptime dtype = DType.float64
comptime P = Plain[DType.float64, 1]


def _m(ctx: DeviceContext) raises -> Static[dtype, 3, 3]:
    """Determinant is -30: a well-conditioned, sign-negative example."""
    return Static[dtype, 3, 3](
        ctx, [2.0, -1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 2.0, -2.0]
    )


def _singular(ctx: DeviceContext) raises -> Static[dtype, 3, 3]:
    """Row 2 is row 0 doubled, so the determinant is exactly zero."""
    return Static[dtype, 3, 3](
        ctx, [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 7.0, 8.0, 10.0]
    )


def _as_array(values: Array[Float64, 9]) -> Array[P, 9]:
    var out = Array[P, 9](fill=P.constant(0.0))
    for i in range(9):
        out[i] = P.constant(values[i])
    return out^


def test_slogdet_reproduces_det() raises:
    var ctx = DeviceContext(api="cpu")
    var a = _m(ctx)
    var pair = slogdet[dtype, 3](a)

    var b = _m(ctx)
    var reference = Float64(det[dtype, 3](b))

    var rebuilt = Float64(pair[0]) * exp_f64(Float64(pair[1]))
    assert_almost_equal(rebuilt, reference, atol=1e-9)


def test_slogdet_reports_the_sign_separately() raises:
    var ctx = DeviceContext(api="cpu")
    var a = _m(ctx)
    var pair = slogdet[dtype, 3](a)

    var b = _m(ctx)
    var reference = Float64(det[dtype, 3](b))
    assert_true(reference < 0)
    assert_almost_equal(Float64(pair[0]), -1.0, atol=0.0)
    assert_almost_equal(Float64(pair[1]), log_f64(abs(reference)), atol=1e-9)


def test_a_singular_matrix_gives_sign_zero_and_minus_infinity() raises:
    """NumPy's convention, and the case a `log` would otherwise be handed
    a zero for."""
    var ctx = DeviceContext(api="cpu")
    var a = _singular(ctx)
    var pair = slogdet[dtype, 3](a)

    assert_equal(Float64(pair[0]), 0.0)
    assert_true(Float64(pair[1]) < -1e300)


def test_slogdet_survives_where_det_overflows() raises:
    """The reason this name exists. A 200x200 diagonal matrix of 10s has a
    determinant of 1e200 -- representable -- but scaled to 40 it is 1e280
    and the product overflows on the way to an answer `slogdet` reports
    exactly."""
    var ctx = DeviceContext(api="cpu")
    comptime n = 200
    var entries = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        entries[i * n + i] = 40.0
    var a = Static[dtype, n, n](ctx, entries^)

    var pair = slogdet[dtype, n](a)
    var want = Float64(n) * log_f64(40.0)
    assert_almost_equal(Float64(pair[0]), 1.0, atol=0.0)
    assert_almost_equal(Float64(pair[1]), want, atol=1e-8)

    var entries_again = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        entries_again[i * n + i] = 40.0
    var b = Static[dtype, n, n](ctx, entries_again^)
    var overflowed = Float64(det[dtype, n](b))
    assert_true(
        overflowed > 1e308 or overflowed != overflowed,
        "det must overflow here, or this test is not testing anything",
    )


def test_the_two_tiers_agree() raises:
    var ctx = DeviceContext(api="cpu")
    var a = _m(ctx)
    var want = slogdet[dtype, 3](a)

    var values = [2.0, -1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 2.0, -2.0]
    var got = array_slogdet[DType.float64, 3](_as_array(values))

    assert_almost_equal(got[0].v, Float64(want[0]), atol=0.0)
    assert_almost_equal(got[1].v, Float64(want[1]), atol=1e-9)


def test_the_array_tier_reports_a_singular_matrix_too() raises:
    var values = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 7.0, 8.0, 10.0]
    var got = array_slogdet[DType.float64, 3](_as_array(values))
    assert_equal(got[0].v, 0.0)
    assert_true(got[1].v < -1e300)


def test_the_factor_can_be_reused_for_both() raises:
    """`slogdet` free-standing factors; holding the `TensorLU` gives both
    numbers from one factorization."""
    var ctx = DeviceContext(api="cpu")
    var a = _m(ctx)
    var factored = lu_factor[dtype, 3](a)
    var pair = factored.slogdet()
    var determinant = Float64(factored.det())

    var rebuilt = Float64(pair[0]) * exp_f64(Float64(pair[1]))
    assert_almost_equal(rebuilt, determinant, atol=1e-9)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
