"""Tests for `peak_widths`, `argrelextrema`, `argrelmax` and `argrelmin`
against SciPy, on a growing, beating sinusoid with peaks of many shapes:
the widths at half prominence and at the bases (`rel_height = 1`, where
the crossings are the bases themselves), and relative extrema at orders
1 and 3, with a non-strict comparator under `"wrap"`.
"""

from std.math import cos, sin
from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.signal import (
    argrelextrema,
    argrelmax,
    argrelmin,
    find_peaks,
    peak_widths,
)

comptime f64 = DType.float64


def _x() raises -> Static[f64, 60]:
    var values = List[Scalar[f64]](capacity=60)
    for i in range(60):
        var t = Float64(i)
        values.append(sin(0.4 * t) * (1 + 0.1 * t) + 0.3 * cos(1.7 * t))
    return Static[f64, 60](values^, DeviceContext(api="cpu"))


def _close(got: List[Float64], want: List[Float64]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-12)


def _same(got: List[Int64], want: List[Int]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(Int(got[i]), want[i])


def test_peak_widths_match_scipy() raises:
    var peaks = find_peaks(_x())
    var r = peak_widths(_x(), peaks)
    _close(
        r.widths.to_host(),
        [
            3.098664361506539,
            7.083451851749341,
            7.360723832339257,
            7.641957022645485,
        ],
    )
    _close(
        r.width_heights.to_host(),
        [
            0.9801111456815274,
            0.3334993841641505,
            0.21810471249590968,
            0.3530565418388125,
        ],
    )
    _close(
        r.left_ips.to_host(),
        [
            2.5426846331378155,
            16.18625545056375,
            31.655869899138178,
            47.26491343796667,
        ],
    )
    _close(
        r.right_ips.to_host(),
        [
            5.641348994644354,
            23.26970730231309,
            39.016593731477435,
            54.906870460612154,
        ],
    )
    var b = peak_widths(_x(), peaks, 1.0)
    _close(
        b.widths.to_host(),
        [
            7.667625194454732,
            12.558988142782457,
            13.543967374963309,
            14.274105575231815,
        ],
    )
    _close(b.left_ips.to_host(), [6.188063084342383e-16, 13.0, 28.0, 43.0])


def test_relative_extrema_match_scipy() raises:
    _same(argrelmax(_x()).to_host(), [4, 19, 36, 51])
    _same(argrelmin[order=3](_x()).to_host(), [13, 28, 43])
    _same(
        argrelextrema[comparator="greater_equal", order=2, mode="wrap"](
            _x()
        ).to_host(),
        [4, 19, 36, 51],
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
