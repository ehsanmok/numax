"""Tests for `iirnotch`, `iirpeak`, `bilinear`, `group_delay` and
`lfiltic`, each against SciPy: a 60 Hz notch at `fs = 1000`, a peak in
Nyquist units, the bilinear transform of a second-order analog filter at
`fs = 10`, a Butterworth's group delay over half and the whole circle
(the latter at `fs = 100`, away from the near-singular `w = pi`), and `lfiltic` with and without past inputs,
with a `y` shorter than the state.
"""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.signal import (
    bilinear,
    butter,
    group_delay,
    iirnotch,
    iirpeak,
    lfilter,
    lfiltic,
)

comptime f64 = DType.float64


def _close(
    got: List[Float64], want: List[Float64], tol: Float64 = 1e-12
) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=tol)


def test_notch_and_peak_match_scipy() raises:
    var n = iirnotch[f64](60.0, 30.0, 1000.0)
    _close(
        n.b.to_host(),
        [0.9937559649536571, -1.8479418578501994, 0.9937559649536571],
    )
    _close(n.a.to_host(), [1.0, -1.8479418578501994, 0.9875119299073143])
    var p = iirpeak[f64](0.3, 5.0)
    _close(p.b.to_host(), [0.08636402701376222, 0.0, -0.08636402701376222])
    _close(p.a.to_host(), [1.0, -1.07404350177039, 0.8272719459724756])


def test_bilinear_matches_scipy() raises:
    var cpu = DeviceContext(api="cpu")
    var b = Static[f64, 2]([1.0, 0.5], cpu)
    var a = Static[f64, 3]([1.0, 2.0, 3.0], cpu)
    var tf = bilinear(b, a, 10.0)
    _close(
        tf.b.to_host(),
        [0.04627539503386004, 0.0022573363431151227, -0.04401805869074492],
    )
    _close(tf.a.to_host(), [1.0, -1.7923250564334088, 0.8194130925507901])


def test_group_delay_matches_scipy() raises:
    var tf = butter[f64, 4](0.3)
    var g = group_delay[worN=8](tf.b, tf.a)
    _close(
        g.w.to_host(),
        [
            0.0,
            0.39269908169872414,
            0.7853981633974483,
            1.1780972450961724,
            1.5707963267948966,
            1.9634954084936207,
            2.356194490192345,
            2.748893571891069,
        ],
    )
    _close(
        g.gd.to_host(),
        [
            2.5642742009703348,
            2.867548108216157,
            4.398264685675803,
            3.0534085351261755,
            1.528161948247341,
            1.0158564992924823,
            0.795044893544782,
            0.6950429270339438,
        ],
        1e-9,
    )
    # Five points round the circle, which skips `w = pi`: the zeros at
    # `z = -1` make the delay near-singular there, where SciPy warns.
    var whole = group_delay[worN=5, whole=True](tf.b, tf.a, 100.0)
    _close(whole.w.to_host(), [0.0, 20.0, 40.0, 60.0, 80.0])
    _close(
        whole.gd.to_host(),
        [
            2.5642742009703348,
            2.5901945705611036,
            0.7446093862466361,
            0.744609386246637,
            2.5901945705611,
        ],
        1e-9,
    )


def test_lfiltic_matches_scipy() raises:
    var cpu = DeviceContext(api="cpu")
    var tf = butter[f64, 4](0.3)
    var y = Static[f64, 3]([0.5, -0.25, 1.0], cpu)
    _close(
        lfiltic(tf.b, tf.a, y).to_host(),
        [
            1.5885061251949917,
            -0.8351045691857438,
            0.26125095032012596,
            -0.03809853230516621,
        ],
    )
    var y2 = Static[f64, 2]([0.5, -0.25], cpu)
    var x = Static[f64, 4]([1.0, 2.0, -1.0, 0.5], cpu)
    _close(
        lfiltic(tf.b, tf.a, y2, x).to_host(),
        [
            1.3361403896961206,
            -0.5175883664257481,
            0.372629014081509,
            -0.019535521678269035,
        ],
    )


def _x() raises -> Static[f64, 40]:
    var values = List[Scalar[f64]](capacity=40)
    for i in range(40):
        values.append(sin(0.3 * Float64(i)) + 0.1 * Float64(i))
    return Static[f64, 40](values^, DeviceContext(api="cpu"))


def test_lfilter_with_zi_matches_scipy_and_chains() raises:
    # scipy: signal.lfilter(b, a, x, zi=[0.5, -0.2, 0.1, 0.3])
    var tf = butter[f64, 4](0.3)
    var cpu = DeviceContext(api="cpu")
    var zi = Static[f64, 4]([0.5, -0.2, 0.1, 0.3], cpu)
    var r = lfilter(tf.b, tf.a, _x(), zi)
    _close(
        r.y.to_host(),
        [
            0.5,
            0.592541471413493,
            0.43628203307819713,
            0.5924230403546804,
            0.8424423123001129,
            1.0449360837791464,
            1.2034647404877494,
            1.3516100478974336,
            1.484061480721518,
            1.5627409395460619,
            1.5534263165832471,
            1.4505779948380124,
            1.277401405156498,
            1.0709881738065044,
            0.8674889790722181,
            0.6952447618418506,
            0.5750661560170097,
            0.522828563620277,
            0.5505771570158471,
            0.6653705857385921,
            0.8672491389394155,
            1.1480158232241695,
            1.4916163952579842,
            1.875897980541392,
            2.275064897579449,
            2.6622399663215255,
            3.0118446583471696,
            3.301729843692156,
            3.515026888323579,
            3.6416213440869343,
            3.679093871759612,
            3.632987420632113,
            3.5163382019017253,
            3.348507717331408,
            3.153436361163274,
            2.9574915695154824,
            2.787109438878227,
            2.666437549788036,
            2.6151833902365893,
            2.6468565839418448,
        ],
    )
    _close(
        r.zf.to_host(),
        [
            2.7032699757864522,
            -1.6991134167190272,
            1.3693550105146437,
            -0.14343167006181173,
        ],
    )
    # Two halves, the second started from the first's `zf`, are one call.
    var whole = lfilter(
        tf.b, tf.a, _x(), Static[f64, 4]([0.0, 0.0, 0.0, 0.0], cpu)
    )
    var xs = _x().to_host()
    var first = List[Scalar[f64]]()
    var second = List[Scalar[f64]]()
    for i in range(20):
        first.append(xs[i])
        second.append(xs[20 + i])
    var h1 = lfilter(
        tf.b,
        tf.a,
        Static[f64, 20](first^, cpu),
        Static[f64, 4]([0.0, 0.0, 0.0, 0.0], cpu),
    )
    var h2 = lfilter(tf.b, tf.a, Static[f64, 20](second^, cpu), h1.zf)
    var w = whole.y.to_host()
    var y1 = h1.y.to_host()
    var y2 = h2.y.to_host()
    for i in range(20):
        assert_almost_equal(y1[i], w[i], atol=1e-13)
        assert_almost_equal(y2[i], w[20 + i], atol=1e-13)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
