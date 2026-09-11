"""Tests for `numax.interpolate.spline`: the cubic splines over `Tensor`.

Every expected value is `scipy.interpolate`'s own on the same five
non-uniform knots -- values, first, second and third derivatives, knot
slopes and definite integrals -- for each boundary condition of
`CubicSpline`, for `PchipInterpolator`, `Akima1DInterpolator` and a
`CubicHermiteSpline` with prescribed slopes. A uniform-grid shortcut, a
wrong boundary row or a wrong local-coefficient formula would each show.
"""

from std.testing import TestSuite, assert_almost_equal, assert_true

from max.gpu.host import DeviceContext

from numax.core.array import Static
from numax.interpolate import (
    Akima1DInterpolator,
    CubicHermiteSpline,
    CubicSpline,
    PchipInterpolator,
)
from numax.interpolate.array import cubic_spline_eval as eval_a
from numax.interpolate.array import cubic_spline_moments as moments_a
from numax import Plain
from std.collections import Array

comptime dtype = DType.float64
comptime P = Plain[dtype]


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _from[n: Int](values: List[Float64]) raises -> Static[dtype, n]:
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype](values[i]))
    return Static[dtype, n](_cpu(), out^)


def _knots() -> List[Float64]:
    return [0.0, 1.0, 2.5, 4.0, 7.0]


def _samples() -> List[Float64]:
    return [1.0, 3.0, 2.0, 5.0, -1.0]


def _queries() -> List[Float64]:
    """Before, on, between and past the knots."""
    return [-0.5, 0.0, 0.7, 1.0, 2.0, 3.3, 4.0, 5.5, 7.0, 7.5]


def _assert_close[
    n: Int
](got: Static[dtype, n], want: List[Float64], atol: Float64 = 1e-11) raises:
    var values = got.to_host()
    for i in range(n):
        assert_almost_equal(Float64(values[i]), want[i], atol=atol)


def test_not_a_knot_spline_matches_scipy() raises:
    """`scipy.interpolate.CubicSpline(x, y)` -- the default boundary
    condition -- values and first three derivatives, extrapolated past
    both ends with the end cubics."""
    var x = _from[5](_knots())
    var y = _from[5](_samples())
    var spline = CubicSpline(x, y)
    var q = _from[10](_queries())
    _assert_close(
        spline(q),
        [
            -2.381818181818181,
            1.0,
            2.889745454545454,
            3.0,
            2.1636363636363636,
            3.188051178451177,
            5.0,
            6.963636363636366,
            -0.9999999999999964,
            -7.397306397306398,
        ],
    )
    _assert_close(
        spline[nu=1](q),
        [
            8.878787878787875,
            4.824242424242423,
            0.9195151515151517,
            -0.12121212121212088,
            -0.8484848484848486,
            2.319434343434343,
            2.642424242424243,
            -1.0121212121212104,
            -10.593939393939396,
            -15.10505050505051,
        ],
    )
    _assert_close(
        spline[nu=2](q),
        [
            -9.163636363636357,
            -7.054545454545451,
            -4.101818181818182,
            -2.836363636363637,
            1.3818181818181818,
            1.3834343434343441,
            -0.46060606060605824,
            -4.412121212121213,
            -8.363636363636369,
            -9.680808080808086,
        ],
    )
    # Not-a-knot: the first two and last two pieces share one cubic, so
    # the third derivative is constant across each pair.
    _assert_close(
        spline[nu=3](q),
        [
            4.218181818181813,
            4.218181818181813,
            4.218181818181813,
            4.218181818181819,
            4.218181818181819,
            -2.634343434343433,
            -2.6343434343434367,
            -2.6343434343434367,
            -2.6343434343434367,
            -2.6343434343434367,
        ],
        atol=1e-9,
    )
    _assert_close(spline[nu=4](q), List[Float64](length=10, fill=0.0))


def test_natural_spline_matches_scipy() raises:
    """`bc_type="natural"`: zero second derivative at both ends, checked
    directly at the knots, and SciPy's values between them."""
    var x = _from[5](_knots())
    var y = _from[5](_samples())
    var spline = CubicSpline(x, y, "natural")
    var q = _from[10](_queries())
    _assert_close(
        spline(q),
        [
            -0.28773584905660354,
            1.0,
            2.673924528301886,
            3.0,
            2.1949685534591197,
            3.4527016072676444,
            5.0,
            3.9386792452830193,
            -0.9999999999999991,
            -2.8377009084556235,
        ],
    )
    var knots = _from[5](_knots())
    _assert_close(
        spline[nu=1](knots),
        [
            2.7672955974842766,
            0.46540880503144655,
            0.5220125786163521,
            1.4465408805031448,
            -3.7232704402515724,
        ],
    )
    _assert_close(
        spline[nu=2](knots),
        [0.0, -4.60377358490566, 4.679245283018868, -3.4465408805031448, 0.0],
        atol=1e-10,
    )


def test_clamped_spline_matches_scipy() raises:
    """`bc_type="clamped"`: zero slope at both ends."""
    var x = _from[5](_knots())
    var y = _from[5](_samples())
    var spline = CubicSpline(x, y, "clamped")
    var q = _from[10](_queries())
    _assert_close(
        spline(q),
        [
            2.509615384615385,
            1.0,
            2.3757692307692304,
            3.0,
            2.301994301994302,
            3.5617094017094013,
            5.0,
            2.3173076923076934,
            -0.9999999999999964,
            -0.362179487179489,
        ],
    )
    var ends = _from[2]([0.0, 7.0])
    _assert_close(spline[nu=1](ends), [0.0, 0.0], atol=1e-12)


def test_spline_integrates_exactly() raises:
    """`CubicSpline.integrate(a, b)` against SciPy's, over the whole range
    and over a sub-range that starts and ends mid-interval, for both
    boundary conditions -- and reversed limits negate."""
    var x = _from[5](_knots())
    var y = _from[5](_samples())
    var spline = CubicSpline(x, y)
    assert_almost_equal(
        Float64(spline.integrate(0.0, 7.0)), 26.821212121212127, atol=1e-11
    )
    assert_almost_equal(
        Float64(spline.integrate(0.5, 6.2)), 23.932843569023568, atol=1e-11
    )
    assert_almost_equal(
        Float64(spline.integrate(6.2, 0.5)), -23.932843569023568, atol=1e-11
    )
    var natural = CubicSpline(x, y, "natural")
    assert_almost_equal(
        Float64(natural.integrate(0.0, 7.0)), 20.885220125786166, atol=1e-11
    )
    assert_almost_equal(
        Float64(natural.integrate(0.5, 6.2)), 19.679457617051014, atol=1e-11
    )


def test_two_and_three_knots_take_scipys_special_cases() raises:
    """`n = 2` not-a-knot is the line through the points; `n = 3` is the
    parabola through them, and natural at `n = 3` is a different curve --
    each SciPy's own numbers."""
    var x2 = _from[2]([0.0, 2.0])
    var y2 = _from[2]([1.0, 5.0])
    var line = CubicSpline(x2, y2)
    var q2 = _from[3]([0.5, 1.0, 3.0])
    _assert_close(line(q2), [2.0, 3.0, 7.0])

    var x3 = _from[3]([0.0, 1.0, 3.0])
    var y3 = _from[3]([1.0, 2.0, 0.0])
    var parabola = CubicSpline(x3, y3)
    var q3 = _from[3]([0.5, 2.0, 4.0])
    _assert_close(
        parabola(q3),
        [1.6666666666666667, 1.6666666666666663, -2.9999999999999996],
    )
    var natural = CubicSpline(x3, y3, "natural")
    _assert_close(
        natural(q3),
        [1.6249999999999998, 1.5000000000000002, -1.4999999999999991],
    )


def test_unknown_boundary_condition_raises() raises:
    var x = _from[5](_knots())
    var y = _from[5](_samples())
    var raised = False
    try:
        var spline = CubicSpline(x, y, "periodic")
        _ = spline^
    except:
        raised = True
    assert_true(raised)


def test_extrapolate_false_gives_nan_outside_the_knots() raises:
    """With `extrapolate=False`, a query past either end is NaN and an
    integral that leaves the knots is NaN; everything inside is
    unchanged."""
    var x = _from[5](_knots())
    var y = _from[5](_samples())
    var spline = CubicSpline(x, y, extrapolate=False)
    var q = _from[4]([-0.5, 0.0, 3.3, 7.5])
    var values = spline(q).to_host()
    assert_true(Float64(values[0]) != Float64(values[0]))
    assert_almost_equal(Float64(values[1]), 1.0, atol=1e-12)
    assert_almost_equal(Float64(values[2]), 3.188051178451177, atol=1e-11)
    assert_true(Float64(values[3]) != Float64(values[3]))
    var outside = Float64(spline.integrate(-1.0, 2.0))
    assert_true(outside != outside)
    assert_almost_equal(
        Float64(spline.integrate(0.0, 7.0)), 26.821212121212127, atol=1e-11
    )


def test_hermite_spline_takes_prescribed_slopes() raises:
    """`scipy.interpolate.CubicHermiteSpline(x, y, dydx)`: values from the
    given slopes, the slopes reproduced at the knots, and the integral."""
    var x = _from[5](_knots())
    var y = _from[5](_samples())
    var d = _from[5]([1.0, -0.5, 2.0, 0.0, -3.0])
    var spline = CubicHermiteSpline(x, y, d)
    var q = _from[10](_queries())
    _assert_close(
        spline(q),
        [
            2.0625,
            1.0,
            2.7044999999999995,
            3.0,
            1.7592592592592593,
            3.9982222222222212,
            5.0,
            3.125,
            -1.0,
            -2.4861111111111116,
        ],
    )
    var knots = _from[5](_knots())
    _assert_close(spline[nu=1](knots), [1.0, -0.5, 2.0, 0.0, -3.0])
    assert_almost_equal(
        Float64(spline.integrate(0.0, 7.0)), 19.28125, atol=1e-12
    )


def test_pchip_matches_scipy() raises:
    """`scipy.interpolate.PchipInterpolator`: the knot slopes are zero
    wherever the secants change sign -- every interior knot of this data
    -- and the end slopes are the guarded three-point estimates."""
    var x = _from[5](_knots())
    var y = _from[5](_samples())
    var pchip = PchipInterpolator(x, y)
    var q = _from[10](_queries())
    _assert_close(
        pchip(q),
        [
            -0.45000000000000023,
            1.0,
            2.7611999999999997,
            3.0,
            2.2592592592592595,
            3.649777777777777,
            5.0,
            3.75,
            -0.9999999999999996,
            -3.6203703703703702,
        ],
    )
    _assert_close(
        pchip[nu=1](q),
        [
            2.500000000000001,
            3.066666666666667,
            1.508,
            0.0,
            -0.8888888888888888,
            2.9866666666666672,
            0.0,
            -1.8333333333333333,
            -4.666666666666667,
            -5.833333333333334,
        ],
    )
    var knots = _from[5](_knots())
    _assert_close(
        pchip[nu=1](knots),
        [3.066666666666667, 0.0, 0.0, 0.0, -4.666666666666667],
    )


def test_akima_matches_scipy_and_is_nan_outside() raises:
    """`scipy.interpolate.Akima1DInterpolator`: values, first derivative
    and knot slopes inside the knots, NaN outside by default, and the end
    cubics with `extrapolate=True`."""
    var x = _from[5](_knots())
    var y = _from[5](_samples())
    var akima = Akima1DInterpolator(x, y)
    var inside = _from[8]([0.0, 0.7, 1.0, 2.0, 3.3, 4.0, 5.5, 7.0])
    _assert_close(
        akima(inside),
        [
            1.0,
            2.68,
            3.0,
            2.2444444444444445,
            3.639822222222221,
            5.0,
            3.65,
            -0.9999999999999996,
        ],
    )
    _assert_close(
        akima[nu=1](inside),
        [
            3.3333333333333335,
            1.4666666666666668,
            0.6666666666666667,
            -1.1111111111111112,
            2.7893333333333343,
            0.3999999999999999,
            -2.1,
            -3.9999999999999996,
        ],
    )
    var knots = _from[5](_knots())
    _assert_close(
        akima[nu=1](knots),
        [3.3333333333333335, 0.6666666666666667, 0.4, 0.3999999999999999, -4.0],
    )
    var edges = _from[2]([-0.5, 7.5])
    var outside = akima(edges).to_host()
    assert_true(Float64(outside[0]) != Float64(outside[0]))
    assert_true(Float64(outside[1]) != Float64(outside[1]))

    var extended = Akima1DInterpolator(x, y, extrapolate=True)
    var past = extended(edges).to_host()
    assert_true(Float64(past[0]) == Float64(past[0]))
    assert_true(Float64(past[1]) == Float64(past[1]))


def test_array_tier_non_uniform_moments_agree_with_the_tensor_natural_spline() raises:
    """`numax.interpolate.array`'s non-uniform `cubic_spline_moments` are
    SciPy's natural second derivatives at the knots, and its
    `cubic_spline_eval` agrees with the `Tensor` natural spline strictly
    inside the knots -- two tiers, two formulations (moments against
    slopes), one curve."""
    var knots = _knots()
    var samples = _samples()
    var xa = Array[P, 5](uninitialized=True)
    var ya = Array[P, 5](uninitialized=True)
    for i in range(5):
        xa[i] = P(knots[i])
        ya[i] = P(samples[i])
    var moments = moments_a[P, 5](xa, ya)
    var expected = [
        0.0,
        -4.60377358490566,
        4.679245283018868,
        -3.4465408805031448,
        0.0,
    ]
    for i in range(5):
        assert_almost_equal(Float64(moments[i].v), expected[i], atol=1e-11)

    var x = _from[5](knots)
    var y = _from[5](samples)
    var spline = CubicSpline(x, y, "natural")
    var inside: List[Float64] = [0.0, 0.7, 1.0, 2.0, 3.3, 4.0, 5.5, 7.0]
    var q = _from[8](inside)
    var tensor_values = spline(q).to_host()
    for i in range(8):
        var array_value = eval_a[P, 5](xa, ya, moments, P(inside[i]))
        assert_almost_equal(
            Float64(array_value.v), Float64(tensor_values[i]), atol=1e-11
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
