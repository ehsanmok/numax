"""Tests for `solve_ivp`'s `t_eval`, `dense_output` and events: output on
the requested times of an oscillator whose solution is `cos`, the dense
interpolant between steps, a falling body stopped by a terminal event at
the closed-form landing time, and the zero crossings of `cos` found by a
non-terminal event in each direction."""

from std.math import cos as cos_f64, pi, sqrt
from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.integrate import solve_ivp

comptime dtype = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def oscillator(
    t: Scalar[dtype], y: Static[dtype, 2], ctx: DeviceContext
) raises -> Static[dtype, 2]:
    var h = y.to_host()
    return Static[dtype, 2]([h[1], -h[0]], ctx)


def falling(
    t: Scalar[dtype], y: Static[dtype, 2], ctx: DeviceContext
) raises -> Static[dtype, 2]:
    """Height and velocity under `g = 9.81`."""
    var h = y.to_host()
    return Static[dtype, 2]([h[1], Scalar[dtype](-9.81)], ctx)


def height(
    t: Scalar[dtype], y: Static[dtype, 2], ctx: DeviceContext
) raises -> Scalar[dtype]:
    return y.to_host()[0]


def position(
    t: Scalar[dtype], y: Static[dtype, 2], ctx: DeviceContext
) raises -> Scalar[dtype]:
    return y.to_host()[0]


def _times(values: List[Float64]) raises -> Static[dtype, 5]:
    return Static[dtype, 5](values.copy(), _cpu())


def test_t_eval_on_the_oscillator() raises:
    var y0 = Static[dtype, 2]([1.0, 0.0], _cpu())
    var times: List[Float64] = [0.0, 0.7, 2.0, 4.5, 6.0]
    var r = solve_ivp[f=oscillator](0.0, y0, 6.0, _times(times))
    assert_equal(r.status, 0)
    assert_equal(len(r.t), 5)
    var y = r.y.to_host()
    # `y` is `2 x 5`: row 0 is the position, row 1 the velocity.
    for j in range(5):
        assert_almost_equal(r.t[j], times[j], atol=0.0)
        assert_almost_equal(y[j], cos_f64(times[j]), atol=1e-7)


def test_dense_output_between_steps() raises:
    var y0 = Static[dtype, 2]([1.0, 0.0], _cpu())
    var times: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0]
    var r = solve_ivp[f=oscillator, dense_output=True](
        0.0, y0, 4.0, _times(times)
    )
    for i in range(41):
        var t = 0.1 * Float64(i)
        var at = r.sol(t).to_host()
        assert_almost_equal(at[0], cos_f64(t), atol=1e-7)


def test_terminal_event_stops_at_landing() raises:
    var y0 = Static[dtype, 2]([10.0, 0.0], _cpu())
    var times: List[Float64] = [0.0, 0.5, 1.0, 2.0, 3.0]
    var r = solve_ivp[f=falling, event=height, terminal=True, direction=-1](
        0.0, y0, 10.0, _times(times)
    )
    assert_equal(r.status, 1)
    assert_equal(len(r.t_events), 1)
    var landing = sqrt(2.0 * 10.0 / 9.81)
    assert_almost_equal(r.t_events[0], landing, atol=1e-12)
    var ye = r.y_events.to_host()
    assert_almost_equal(ye[0], 0.0, atol=1e-10)
    # The `t_eval` points past the landing (2.0 and 3.0) are not reported.
    assert_equal(len(r.t), 3)


def test_crossings_by_direction() raises:
    var y0 = Static[dtype, 2]([1.0, 0.0], _cpu())
    var times: List[Float64] = [0.0, 2.0, 4.0, 6.0, 10.0]
    # cos falls through zero at pi/2 and 5 pi/2, rises at 3 pi/2 and 7 pi/2.
    var down = solve_ivp[f=oscillator, event=position, direction=-1](
        0.0, y0, 10.0, _times(times)
    )
    assert_equal(len(down.t_events), 2)
    assert_almost_equal(down.t_events[0], pi / 2, atol=1e-7)
    assert_almost_equal(down.t_events[1], 5 * pi / 2, atol=1e-7)
    var both = solve_ivp[f=oscillator, event=position](
        0.0, y0, 10.0, _times(times)
    )
    assert_equal(len(both.t_events), 3)
    assert_almost_equal(both.t_events[1], 3 * pi / 2, atol=1e-7)
    assert_equal(both.status, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
