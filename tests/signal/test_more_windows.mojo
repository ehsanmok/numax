"""Tests for the `tukey`, `gaussian`, `flattop`, `nuttall` and `chebwin`
windows in `numax.signal.windows`.

Each against SciPy, symmetric and periodic, at odd and even lengths --
`chebwin`'s transform takes a different path for each parity -- plus
`tukey`'s two limits and the two new `get_window` names.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from numax.signal import (
    boxcar,
    chebwin,
    flattop,
    gaussian,
    get_window,
    hann,
    nuttall,
    tukey,
)

comptime f64 = DType.float64


def _close(got: List[Float64], want: List[Float64]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-12)


def test_windows_match_scipy() raises:
    _close(
        tukey[f64, 9](0.5).to_host(),
        [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0],
    )
    _close(
        tukey[f64, 10](0.3, False).to_host(),
        [
            0.0,
            0.7500000000000002,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.7500000000000004,
        ],
    )
    _close(
        gaussian[f64, 8](2.0).to_host(),
        [
            0.2162651668298873,
            0.45783336177161427,
            0.7548396019890073,
            0.9692332344763441,
            0.9692332344763441,
            0.7548396019890073,
            0.45783336177161427,
            0.2162651668298873,
        ],
    )
    _close(
        gaussian[f64, 7](1.5, False).to_host(),
        [
            0.06572852861653047,
            0.24935220877729622,
            0.6065306597126334,
            0.9459594689067654,
            0.9459594689067654,
            0.6065306597126334,
            0.24935220877729622,
        ],
    )
    _close(
        flattop[f64, 9]().to_host(),
        [
            -0.0004210510000000013,
            -0.026872193286334545,
            -0.05473684,
            0.4441353572863345,
            1.000000003,
            0.4441353572863345,
            -0.05473684,
            -0.026872193286334545,
            -0.0004210510000000013,
        ],
    )
    _close(
        flattop[f64, 8](False).to_host(),
        [
            -0.0004210510000000013,
            -0.026872193286334545,
            -0.05473684,
            0.4441353572863345,
            1.000000003,
            0.4441353572863345,
            -0.05473684,
            -0.026872193286334545,
        ],
    )
    _close(
        nuttall[f64, 7]().to_host(),
        [
            0.0003628000000000381,
            0.06133449999999996,
            0.5292298,
            1.0,
            0.5292298000000002,
            0.06133450000000014,
            0.0003628000000000381,
        ],
    )
    _close(
        nuttall[f64, 8](False).to_host(),
        [
            0.0003628000000000381,
            0.025205566515401824,
            0.22698240000000006,
            0.7019582334845982,
            1.0,
            0.7019582334845982,
            0.22698240000000006,
            0.025205566515401824,
        ],
    )
    _close(
        chebwin[f64, 9](50.0).to_host(),
        [
            0.07744700253116996,
            0.27604701066744125,
            0.5835232105523307,
            0.87760372828767,
            1.0,
            0.87760372828767,
            0.5835232105523307,
            0.27604701066744125,
            0.07744700253116996,
        ],
    )
    _close(
        chebwin[f64, 10](60.0).to_host(),
        [
            0.044313249478011095,
            0.1888932622106905,
            0.45729052434559336,
            0.777467958894373,
            1.0,
            1.0,
            0.777467958894373,
            0.45729052434559336,
            0.1888932622106905,
            0.044313249478011095,
        ],
    )
    _close(
        chebwin[f64, 8](45.0, False).to_host(),
        [
            0.09867497829870583,
            0.30900616988174484,
            0.611403034667267,
            0.8875384097049369,
            1.0,
            0.8875384097049369,
            0.611403034667267,
            0.30900616988174484,
        ],
    )


def test_tukey_limits_and_get_window() raises:
    _close(tukey[f64, 8](0.0).to_host(), boxcar[f64, 8]().to_host())
    _close(tukey[f64, 8](1.0).to_host(), hann[f64, 8]().to_host())
    _close(
        get_window[f64, 8]("flattop").to_host(),
        flattop[f64, 8](False).to_host(),
    )
    _close(
        get_window[f64, 8]("nuttall").to_host(),
        nuttall[f64, 8](False).to_host(),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
