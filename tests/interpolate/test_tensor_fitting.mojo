"""Tests for `numax.interpolate.chebyshev` and `numax.interpolate.grid`:
the Chebyshev least-squares fit and `RegularGridInterpolator` over
`Tensor`, each against NumPy's and SciPy's own values.
"""

from std.testing import TestSuite, assert_almost_equal, assert_true

from max.gpu.host import DeviceContext

from numax.core.array import Static
from numax.interpolate import Chebyshev, RegularGridInterpolator, chebval

comptime dtype = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _from[n: Int](values: List[Float64]) raises -> Static[dtype, n]:
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype](values[i]))
    return Static[dtype, n](_cpu(), out^)


def _matrix[
    rows: Int, cols: Int
](values: List[Float64]) raises -> Static[dtype, rows, cols]:
    var out = List[Scalar[dtype]](capacity=rows * cols)
    for i in range(rows * cols):
        out.append(Scalar[dtype](values[i]))
    return Static[dtype, rows, cols](_cpu(), out^)


def _assert_close[
    n: Int
](got: Static[dtype, n], want: List[Float64], atol: Float64 = 1e-11) raises:
    var values = got.to_host()
    for i in range(n):
        assert_almost_equal(Float64(values[i]), want[i], atol=atol)


# ------------------------------------------------------------- Chebyshev


def test_chebyshev_fit_matches_numpy() raises:
    """`numpy.polynomial.chebyshev.Chebyshev.fit(x, y, 3)` on five
    samples: the coefficients, the domain, and the series at ten points
    including two outside the data."""
    var x = _from[5]([0.0, 1.0, 2.5, 4.0, 7.0])
    var y = _from[5]([1.0, 3.0, 2.0, 5.0, -1.0])
    var series = Chebyshev[dtype, 4].fit(x, y)
    _assert_close(
        series.coefficients,
        [
            2.2120282873752797,
            -0.32744269385049257,
            -1.9535655946759714,
            -0.874215011442375,
        ],
        atol=1e-10,
    )
    assert_almost_equal(Float64(series.a), 0.0)
    assert_almost_equal(Float64(series.b), 7.0)
    var q = _from[10]([-0.5, 0.0, 0.7, 1.0, 2.0, 3.3, 4.0, 5.5, 7.0, 7.5])
    _assert_close(
        series(q),
        [
            1.6591164859505914,
            1.460120397992176,
            1.6192603919186854,
            1.807095264464728,
            2.739562430904135,
            4.022333958545458,
            4.403547632232365,
            3.5488647934120516,
            -0.9431950125935593,
            -3.534312236889898,
        ],
        atol=1e-10,
    )


def test_chebyshev_fit_with_as_many_terms_as_points_interpolates() raises:
    """Degree 4 through five points is the interpolating polynomial: NumPy's
    coefficients, and the samples reproduced at the knots."""
    var x = _from[5]([0.0, 1.0, 2.5, 4.0, 7.0])
    var y = _from[5]([1.0, 3.0, 2.0, 5.0, -1.0])
    var series = Chebyshev[dtype, 5].fit(x, y)
    _assert_close(
        series.coefficients,
        [
            3.8216435185185187,
            1.5747685185185183,
            -1.7175925925925912,
            -2.5747685185185207,
            -2.1040509259259275,
        ],
        atol=1e-9,
    )
    _assert_close(series(x), [1.0, 3.0, 2.0, 5.0, -1.0], atol=1e-9)


def test_chebval_matches_numpy_on_the_natural_domain() raises:
    """`chebval([0.1, 0.5, -0.3, 1, -1], [1, -2, 0.5, 0.25])`: raw
    coefficients, no domain map, `T_k(±1) = (±1)^k` at the ends."""
    var x = _from[5]([0.1, 0.5, -0.3, 1.0, -1.0])
    var c = _from[4]([1.0, -2.0, 0.5, 0.25])
    _assert_close(
        chebval(x, c),
        [0.23599999999999993, -0.5, 1.388, -0.25, 3.25],
        atol=1e-14,
    )


def test_chebval_agrees_with_the_series_on_the_natural_domain() raises:
    """A `Chebyshev` on `[-1, 1]` and `chebval` with the same coefficients
    are one computation: the convenience and the primitive agree."""
    var c = _from[4]([1.0, -2.0, 0.5, 0.25])
    var series = Chebyshev[dtype, 4](
        _from[4]([1.0, -2.0, 0.5, 0.25]), -1.0, 1.0
    )
    var x = _from[3]([0.1, 0.5, -0.3])
    var raw = chebval(x, c).to_host()
    var mapped = series(x).to_host()
    for i in range(3):
        assert_almost_equal(Float64(raw[i]), Float64(mapped[i]), atol=1e-15)


# ------------------------------------------------- RegularGridInterpolator


def _grid_x() -> List[Float64]:
    return [0.0, 1.0, 3.0]


def _grid_y() -> List[Float64]:
    return [0.0, 2.0, 3.0, 4.5]


def _grid_values() -> List[Float64]:
    return [1.0, 2.0, 0.5, -1.0, 3.0, 1.0, 4.0, 2.0, 0.0, -2.0, 1.5, 3.0]


def _points() -> List[Float64]:
    """Six `(x, y)` pairs: cell interiors, a grid node, the far corner,
    and two ties for the nearest rule."""
    return [0.5, 1.0, 2.0, 2.5, 0.0, 0.0, 3.0, 4.5, 1.5, 3.75, 2.5, 0.5]


def test_regular_grid_linear_matches_scipy() raises:
    """`scipy.interpolate.RegularGridInterpolator((x, y), v)(pts)` with the
    default linear method on a grid whose axes are not uniformly spaced."""
    var x = _from[3](_grid_x())
    var y = _from[4](_grid_y())
    var v = _matrix[3, 4](_grid_values())
    var grid = RegularGridInterpolator(x, y, v)
    var pts = _matrix[6, 2](_points())
    _assert_close(grid(pts), [1.75, 1.125, 1.0, 3.0, 2.8125, 0.25], atol=1e-14)


def test_regular_grid_nearest_matches_scipy() raises:
    """`method="nearest"`: the closer node per axis, and on a tie the lower
    one, which is what `(1.5, 3.75)` and `(2.5, 0.5)` check."""
    var x = _from[3](_grid_x())
    var y = _from[4](_grid_y())
    var v = _matrix[3, 4](_grid_values())
    var grid = RegularGridInterpolator(x, y, v, "nearest")
    var pts = _matrix[6, 2](_points())
    _assert_close(grid(pts), [1.0, 1.0, 1.0, 3.0, 4.0, 0.0], atol=1e-15)


def test_regular_grid_out_of_bounds_follows_scipy() raises:
    """`bounds_error=True` raises; `bounds_error=False` fills NaN, or the
    given `fill_value`; `extrapolate=True` (SciPy's `fill_value=None`)
    applies the edge cell's bilinear rule past the grid."""
    var x = _from[3](_grid_x())
    var y = _from[4](_grid_y())
    var v = _matrix[3, 4](_grid_values())
    var outside = _matrix[2, 2]([-1.0, 1.0, 3.5, 5.0])

    var strict = RegularGridInterpolator(x, y, v)
    var raised = False
    try:
        _ = strict(outside)
    except:
        raised = True
    assert_true(raised)

    var filled = RegularGridInterpolator(x, y, v, bounds_error=False)
    var nans = filled(outside).to_host()
    assert_true(Float64(nans[0]) != Float64(nans[0]))
    assert_true(Float64(nans[1]) != Float64(nans[1]))

    var sentinel = RegularGridInterpolator(
        x, y, v, bounds_error=False, fill_value=Scalar[dtype](-7.0)
    )
    _assert_close(sentinel(outside), [-7.0, -7.0], atol=1e-15)

    var extended = RegularGridInterpolator(
        x, y, v, bounds_error=False, extrapolate=True
    )
    _assert_close(extended(outside), [1.0, 4.041666666666666], atol=1e-13)


def test_regular_grid_unknown_method_raises() raises:
    var x = _from[3](_grid_x())
    var y = _from[4](_grid_y())
    var v = _matrix[3, 4](_grid_values())
    var raised = False
    try:
        var grid = RegularGridInterpolator(x, y, v, "cubic")
        _ = grid^
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
