"""Tests for `numax.interpolate` over `Tensor`: `interp` and `horner`.

The expected values are NumPy's own -- `numpy.interp` on a non-uniform
grid with queries before, on, between and past the knots, and
`numpy.polynomial.polynomial.polyval` -- so the bisection, the clamps and
the `left`/`right` overrides are each checked against the definition.
"""

from std.testing import TestSuite, assert_almost_equal

from max.gpu.host import DeviceContext

from numax.core.array import Static
from numax.interpolate import horner, interp

comptime dtype = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _from[n: Int](values: List[Float64]) raises -> Static[dtype, n]:
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype](values[i]))
    return Static[dtype, n](_cpu(), out^)


def _knots() -> List[Float64]:
    """Five non-uniform knots, so a uniform-grid shortcut would show."""
    return [0.0, 1.0, 2.5, 4.0, 7.0]


def _samples() -> List[Float64]:
    return [1.0, 3.0, 2.0, 5.0, -1.0]


def _queries() -> List[Float64]:
    """Before the first knot, on knots, strictly between them, and past
    the last one."""
    return [-1.0, 0.0, 0.5, 1.0, 1.75, 3.0, 4.0, 6.5, 7.0, 8.0]


def _assert_close[
    n: Int
](got: Static[dtype, n], want: List[Float64], atol: Float64 = 1e-12) raises:
    var values = got.to_host()
    for i in range(n):
        assert_almost_equal(Float64(values[i]), want[i], atol=atol)


def test_interp_matches_numpy() raises:
    """`numpy.interp(x, xp, fp)`: clamped to `fp[0]`/`fp[-1]` outside the
    knots, exact on a knot, linear between."""
    var x = _from[10](_queries())
    var xp = _from[5](_knots())
    var fp = _from[5](_samples())
    _assert_close(
        interp(x, xp, fp),
        [1.0, 1.0, 2.0, 3.0, 2.5, 3.0, 5.0, 0.0, -1.0, -1.0],
    )


def test_interp_takes_left_and_right() raises:
    """`left`/`right` replace the clamps strictly outside the knots; a
    point *on* the first knot is inside and keeps `fp[0]`."""
    var x = _from[10](_queries())
    var xp = _from[5](_knots())
    var fp = _from[5](_samples())
    _assert_close(
        interp(x, xp, fp, left=Scalar[dtype](-9.0), right=Scalar[dtype](9.0)),
        [-9.0, 1.0, 2.0, 3.0, 2.5, 3.0, 5.0, 0.0, -1.0, 9.0],
    )


def test_interp_reuses_the_grid() raises:
    """The grid is borrowed, not consumed: two queries against one `xp`,
    `fp`, and the second sees the same table as the first."""
    var xp = _from[5](_knots())
    var fp = _from[5](_samples())
    var first = _from[2]([0.5, 3.0])
    var second = _from[3]([1.75, 6.5, 8.0])
    _assert_close(interp(first, xp, fp), [2.0, 3.0])
    _assert_close(interp(second, xp, fp), [2.5, 0.0, -1.0])


def test_interp_on_a_single_knot_is_constant() raises:
    """`n = 1`: every query is outside or on the one knot, so the answer is
    `fp[0]` everywhere -- the bisection never runs and the clamps do all
    the work, as in NumPy."""
    var x = _from[3]([-1.0, 2.0, 5.0])
    var xp = _from[1]([2.0])
    var fp = _from[1]([7.0])
    _assert_close(interp(x, xp, fp), [7.0, 7.0, 7.0])


def test_horner_matches_polyval() raises:
    """`numpy.polynomial.polynomial.polyval([0.1, 0.5, -0.3, 2], [1, -2,
    0.5, 0.25])` -- ascending coefficients, as the `polynomial` package
    orders them."""
    var c = _from[4]([1.0, -2.0, 0.5, 0.25])
    var x = _from[4]([0.1, 0.5, -0.3, 2.0])
    _assert_close(horner(c, x), [0.80525, 0.15625, 1.63825, 1.0], atol=1e-14)


def test_horner_with_one_coefficient_is_constant() raises:
    var c = _from[1]([3.5])
    var x = _from[3]([-2.0, 0.0, 9.0])
    _assert_close(horner(c, x), [3.5, 3.5, 3.5])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
