"""Tests for `interpn` against SciPy in three dimensions -- linear,
nearest, out of range raising, filled and extrapolated -- and against
`numpy.interp` in one, where the two are the same rule."""

from std.math import isnan
from max.gpu.host import DeviceContext
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_raises,
    assert_true,
)

from numax.core.tensor import Static
from numax.interpolate import interpn

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _axes() raises -> Static[f64, 9]:
    return Static[f64, 9](
        [0.0, 0.5, 2.0, -1.0, 0.0, 1.0, 3.0, 0.0, 1.0], _cpu()
    )


def _values() raises -> Static[f64, 3, 4, 2]:
    return Static[f64, 3, 4, 2](
        [
            0.0,
            -0.3894183423086505,
            0.644217687237691,
            0.2955202066613395,
            0.9854497299884601,
            0.8414709848078964,
            0.8632093666488739,
            0.9916648104524687,
            0.963558185417193,
            0.7833269096274834,
            0.9092974268256817,
            1.099573603041505,
            0.4273798802338298,
            0.94570521217672,
            -0.2555411020268308,
            0.44112000805986773,
            0.5155013718214642,
            0.8084964038195901,
            -0.1577456941432482,
            0.4392493292139824,
            -0.7568024953079283,
            -0.04252044329485244,
            -0.9999232575641008,
            -0.3161659367494545,
        ],
        _cpu(),
    )


def _xi() raises -> Static[f64, 5, 3]:
    return Static[f64, 5, 3](
        [
            0.3,
            -0.5,
            0.2,
            1.9,
            2.9,
            0.9,
            0.0,
            0.0,
            0.0,
            2.0,
            3.0,
            1.0,
            1.2,
            0.4,
            0.5,
        ],
        _cpu(),
    )


def _close(got: List[Float64], want: List[Float64]) raises:
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-14)


def test_linear_and_nearest_match_scipy() raises:
    _close(
        interpn(_axes(), _values(), _xi()).to_host(),
        [
            0.6617782822305674,
            -0.31977289421671834,
            0.644217687237691,
            -0.3161659367494545,
            0.4326888092761853,
        ],
    )
    _close(
        interpn(_axes(), _values(), _xi(), "nearest").to_host(),
        [
            0.963558185417193,
            -0.3161659367494545,
            0.644217687237691,
            -0.3161659367494545,
            0.9092974268256817,
        ],
    )


def test_out_of_range() raises:
    var far = Static[f64, 1, 3]([2.5, 0.0, 0.0], _cpu())
    with assert_raises(contains="out of bounds"):
        _ = interpn(_axes(), _values(), far)
    var filled = interpn(_axes(), _values(), far, bounds_error=False).to_host()
    assert_true(isnan(filled[0]))
    var ext = interpn(
        _axes(), _values(), far, bounds_error=False, extrapolate=True
    ).to_host()
    assert_almost_equal(ext[0], -0.5134267344662248, atol=1e-14)


def test_one_dimension_is_interp() raises:
    var x = Static[f64, 4]([0.0, 1.0, 3.0, 4.0], _cpu())
    var y = Static[f64, 4]([1.0, -1.0, 2.0, 0.5], _cpu())
    var q = Static[f64, 3, 1]([0.5, 2.0, 3.9], _cpu())
    _close(interpn(x, y, q).to_host(), [0.0, 0.5, 0.6500000000000001])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
