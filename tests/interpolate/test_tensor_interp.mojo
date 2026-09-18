"""Tests for `numax.interpolate` over `Tensor`: `interp`, `horner` and
the legacy `numpy.poly*` family.

The expected values are NumPy's own -- `numpy.interp` on a non-uniform
grid with queries before, on, between and past the knots, and
`numpy.polynomial.polynomial.polyval` -- so the bisection, the clamps and
the `left`/`right` overrides are each checked against the definition.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.array import Static
from numax.core.tensorlike import View
from numax.interpolate import (
    horner,
    interp,
    polyder,
    polyfit,
    polyint,
    polyval,
    roots,
)

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


# ------------------------------------------------------------------
# The legacy descending-coefficient family
# ------------------------------------------------------------------


def test_polyval_is_horner_with_the_order_reversed() raises:
    """numpy: np.polyval([2, -3, 1], [0, 1, 2, -1]) == [1, 0, 3, 6]."""
    var p = _from[3]([2.0, -3.0, 1.0])
    var x = _from[4]([0.0, 1.0, 2.0, -1.0])
    var got = polyval(p, x).to_host()
    assert_almost_equal(got[0], 1.0)
    assert_almost_equal(got[1], 0.0)
    assert_almost_equal(got[2], 3.0)
    assert_almost_equal(got[3], 6.0)

    # The same polynomial ascending, through `horner`, agrees.
    var ascending = _from[3]([1.0, -3.0, 2.0])
    var viahorner = horner(ascending, x).to_host()
    for i in range(4):
        assert_almost_equal(got[i], viahorner[i])


def test_polyder_and_polyint_are_inverse_up_to_the_constant() raises:
    """numpy: np.polyder([2, -3, 1]) == [4, -3]."""
    var p = _from[3]([2.0, -3.0, 1.0])
    var d = polyder(p)
    assert_equal(d.num_elements, 2)
    var dh = d.to_host()
    assert_almost_equal(dh[0], 4.0)
    assert_almost_equal(dh[1], -3.0)

    # numpy: np.polyint([2, -3, 1]) == [0.666..., -1.5, 1, 0]
    var i = polyint(p)
    assert_equal(i.num_elements, 4)
    var ih = i.to_host()
    assert_almost_equal(ih[0], 2.0 / 3.0)
    assert_almost_equal(ih[1], -1.5)
    assert_almost_equal(ih[2], 1.0)
    assert_almost_equal(ih[3], 0.0)

    # polyder(polyint(p)) is p.
    var round_trip = polyder(polyint(p)).to_host()
    var source = p.to_host()
    for k in range(3):
        assert_almost_equal(round_trip[k], source[k])


def test_polyint_places_the_constant_in_the_last_slot() raises:
    var p = _from[2]([2.0, 0.0])
    var i = polyint(p, Scalar[dtype](5.0)).to_host()
    assert_almost_equal(i[0], 1.0)
    assert_almost_equal(i[1], 0.0)
    assert_almost_equal(i[2], 5.0)


def test_roots_finds_a_real_pair() raises:
    """x^2 - 3x + 2 has roots 2 and 1; numpy.roots agrees."""
    var p = _from[3]([1.0, -3.0, 2.0])
    var r = roots(p)
    var re = r.re.to_host()
    var im = r.im.to_host()
    assert_equal(len(re), 2)
    # Order is the deflation order, so compare as a set.
    var lo = re[0] if re[0] < re[1] else re[1]
    var hi = re[0] if re[0] >= re[1] else re[1]
    assert_almost_equal(lo, 1.0)
    assert_almost_equal(hi, 2.0)
    assert_almost_equal(im[0], 0.0)
    assert_almost_equal(im[1], 0.0)


def test_roots_finds_a_complex_pair() raises:
    """x^2 + 1 has roots +i and -i."""
    var p = _from[3]([1.0, 0.0, 1.0])
    var r = roots(p)
    var re = r.re.to_host()
    var im = r.im.to_host()
    assert_almost_equal(re[0], 0.0)
    assert_almost_equal(re[1], 0.0)
    # Conjugate pair, positive imaginary part first.
    assert_almost_equal(im[0], 1.0)
    assert_almost_equal(im[1], -1.0)


def test_roots_agrees_with_polyval_at_the_roots_it_finds() raises:
    var p = _from[4]([1.0, -6.0, 11.0, -6.0])
    var r = roots(p)
    var re = r.re.to_host()
    var at = _from[3]([Float64(re[0]), Float64(re[1]), Float64(re[2])])
    var residual = polyval(p, at).to_host()
    for i in range(3):
        assert_almost_equal(residual[i], 0.0, atol=1e-9)


def test_polyfit_recovers_a_polynomial_it_was_sampled_from() raises:
    """A quadratic sampled exactly is fit exactly, and the coefficients
    come back descending so they feed straight into polyval."""
    var xs = _from[5]([-2.0, -1.0, 0.0, 1.0, 2.0])
    # y = 3x^2 - 2x + 1
    var ys = _from[5]([17.0, 6.0, 1.0, 2.0, 9.0])
    var fit = polyfit[deg=2](xs, ys)
    assert_equal(fit.num_elements, 3)
    var c = fit.to_host()
    assert_almost_equal(c[0], 3.0, atol=1e-9)
    assert_almost_equal(c[1], -2.0, atol=1e-9)
    assert_almost_equal(c[2], 1.0, atol=1e-9)

    var back = polyval(fit, xs).to_host()
    var want = ys.to_host()
    for i in range(5):
        assert_almost_equal(back[i], want[i], atol=1e-9)


def test_polyfit_at_degree_one_is_a_straight_line() raises:
    var xs = _from[4]([0.0, 1.0, 2.0, 3.0])
    var ys = _from[4]([1.0, 3.0, 5.0, 7.0])
    var fit = polyfit[deg=1](xs, ys).to_host()
    assert_almost_equal(fit[0], 2.0, atol=1e-9)
    assert_almost_equal(fit[1], 1.0, atol=1e-9)


def test_interp_accepts_views() raises:
    var x = _from[10](_queries())
    var xp = _from[5](_knots())
    var fp = _from[5](_samples())
    var want = interp(x, xp, fp).to_host()
    var got = interp(View(x.view()), View(xp.view()), View(fp.view())).to_host()
    for i in range(10):
        assert_almost_equal(Float64(got[i]), Float64(want[i]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
