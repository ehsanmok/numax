"""Tests for `cumulative_simpson` and `romb` against SciPy's digits:
uniform and non-uniform spacing, odd and even sample counts, the
two-sample trapezoid fallback and `initial`; Romberg on a smooth
integrand and on two samples, and its sample-count check."""

from std.math import sin
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_almost_equal, assert_raises

from numax.core.tensor import Static
from numax.integrate import cumulative_simpson, romb

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _y() raises -> Static[f64, 7]:
    return Static[f64, 7]([1.0, 2.5, 0.3, 4.0, 2.2, -1.0, 3.1], _cpu())


def _close(got: List[Float64], want: List[Float64]) raises:
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-13)


def test_uniform_matches_scipy() raises:
    _close(
        cumulative_simpson(_y(), 0.5).to_host(),
        [
            1.0291666666666666,
            1.8833333333333333,
            3.1875,
            4.966666666666667,
            4.9625,
            5.183333333333334,
        ],
    )
    var even = Static[f64, 6]([1.0, 2.5, 0.3, 4.0, 2.2, -1.0], _cpu())
    _close(
        cumulative_simpson(even, 0.5).to_host(),
        [
            1.0291666666666666,
            1.8833333333333333,
            3.1875,
            4.966666666666667,
            5.325,
        ],
    )
    var two = Static[f64, 2]([1.0, 2.5], _cpu())
    _close(cumulative_simpson(two, 0.5).to_host(), [0.875])
    var with_zero = cumulative_simpson[initial=True](_y(), 0.5).to_host()
    assert_almost_equal(with_zero[0], 0.0, atol=0.0)
    assert_almost_equal(with_zero[6], 5.183333333333334, atol=1e-13)


def test_nonuniform_matches_scipy() raises:
    var x = Static[f64, 7]([0.0, 0.3, 0.9, 1.0, 1.6, 2.5, 2.7], _cpu())
    _close(
        cumulative_simpson(_y(), x).to_host(),
        [
            0.5683333333333334,
            1.7550000000000003,
            1.97952380952381,
            5.896666666666669,
            3.7796212121212176,
            3.9604629629629686,
        ],
    )


def test_romb_matches_scipy() raises:
    var values = List[Float64](capacity=17)
    for i in range(17):
        values.append(sin(2.0 * Float64(i) / 16.0))
    var z = Static[f64, 17](values^, _cpu())
    assert_almost_equal(romb(z, 2.0 / 16.0), 1.4161468365139083, atol=1e-15)
    var pair = Static[f64, 2]([1.0, 3.0], _cpu())
    assert_almost_equal(romb(pair, 0.5), 1.0, atol=0.0)
    var bad = Static[f64, 6]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], _cpu())
    with assert_raises(contains="power of 2"):
        _ = romb(bad)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
