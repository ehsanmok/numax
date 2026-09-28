"""Tests for the N-d transforms in `numax.fft`: `fftn`, `ifftn`, `rfftn`,
`irfftn` and `irfft2`.

`fftn` and `rfftn` against NumPy on a `2 x 3 x 4` volume (a Bluestein
extent beside two powers of two) and `rfftn` again with an odd last axis;
the inverses by the round trip they exist for, including the odd output
width `irfftn[n=5]`; and the claim that `fftn` at rank 2 is `fft2`.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static, zeros
from numax.fft import fft2, fftn, ifftn, irfft2, irfftn, rfft2, rfftn

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _volume() raises -> Static[f64, 2, 3, 4]:
    var values = List[Scalar[f64]](capacity=24)
    for i in range(24):
        values.append(Float64(i * i % 7) - 2.0)
    return Static[f64, 2, 3, 4](values^, _cpu())


def _close(got: List[Float64], want: List[Float64]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-12)


def test_fftn_matches_numpy() raises:
    var out = fftn((_volume(), zeros[f64, 2, 3, 4](_cpu())))
    _close(
        out[0].to_host(),
        [
            -1.0,
            2.0,
            -3.0,
            2.0,
            -4.0,
            -4.4641016151377535,
            6.0,
            2.464101615137754,
            -4.0,
            2.464101615137754,
            6.0,
            -4.4641016151377535,
            -1.0,
            -10.0,
            -3.0,
            -10.0,
            2.0,
            -7.0,
            0.0,
            -7.0,
            2.0,
            -7.0,
            0.0,
            -7.0,
        ],
    )
    _close(
        out[1].to_host(),
        [
            0.0,
            1.0,
            0.0,
            -1.0,
            -3.464101615137755,
            -2.464101615137754,
            -3.4641016151377544,
            -4.4641016151377535,
            3.464101615137755,
            4.4641016151377535,
            3.4641016151377544,
            2.464101615137754,
            0.0,
            -11.0,
            0.0,
            11.0,
            6.928203230275509,
            7.0,
            0.0,
            -7.0,
            -6.928203230275509,
            7.0,
            0.0,
            -7.0,
        ],
    )


def test_ifftn_inverts_fftn() raises:
    var back = ifftn(fftn((_volume(), zeros[f64, 2, 3, 4](_cpu()))))
    _close(back[0].to_host(), _volume().to_host())
    var im = back[1].to_host()
    for i in range(24):
        assert_almost_equal(im[i], 0.0, atol=1e-13)


def test_fftn_at_rank_two_is_fft2() raises:
    var values = List[Scalar[f64]](capacity=15)
    for i in range(15):
        values.append(Float64(i * 3 % 8) - 3.0)
    var a = fftn(
        (Static[f64, 3, 5](values.copy(), _cpu()), zeros[f64, 3, 5](_cpu()))
    )
    var b = fft2(
        (Static[f64, 3, 5](values.copy(), _cpu()), zeros[f64, 3, 5](_cpu()))
    )
    _close(a[0].to_host(), b[0].to_host())
    _close(a[1].to_host(), b[1].to_host())


def test_rfftn_matches_numpy() raises:
    var out = rfftn(_volume())
    assert_equal(out[0].dim_at(2), 3)
    _close(
        out[0].to_host(),
        [
            -1.0,
            2.0,
            -3.0,
            -4.0,
            -4.4641016151377535,
            6.0,
            -4.0,
            2.464101615137754,
            6.0,
            -1.0,
            -10.0,
            -3.0,
            2.0,
            -7.0,
            0.0,
            2.0,
            -7.0,
            0.0,
        ],
    )
    _close(
        out[1].to_host(),
        [
            0.0,
            1.0,
            0.0,
            -3.464101615137755,
            -2.464101615137754,
            -3.4641016151377544,
            3.464101615137755,
            4.4641016151377535,
            3.4641016151377544,
            0.0,
            -11.0,
            0.0,
            6.928203230275509,
            7.0,
            0.0,
            -6.928203230275509,
            7.0,
            0.0,
        ],
    )


def test_rfftn_with_an_odd_last_axis_matches_numpy() raises:
    var values = List[Scalar[f64]](capacity=30)
    for i in range(30):
        values.append(Float64(i * 5 % 11) - 4.0)
    var out = rfftn(Static[f64, 2, 3, 5](values^, _cpu()))
    _close(
        out[0].to_host(),
        [
            31.0,
            -9.0,
            -9.0,
            10.0,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
            -3.0,
            -17.798373876248846,
            6.798373876248845,
            0.0,
            -25.720873339894876,
            -31.09800006813722,
            0.0,
            10.519247216143722,
            -8.700373808111625,
        ],
    )
    _close(
        out[1].to_host(),
        [
            0.0,
            6.604395050930092,
            -6.432881625776282,
            6.928203230275509,
            -8.357218352443704,
            42.80272505659539,
            -6.928203230275509,
            3.4179106100682635,
            11.975037210825773,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )


def test_irfftn_inverts_rfftn_at_even_and_odd_widths() raises:
    _close(irfftn(rfftn(_volume())).to_host(), _volume().to_host())
    var values = List[Scalar[f64]](capacity=30)
    for i in range(30):
        values.append(Float64(i * 5 % 11) - 4.0)
    var odd = irfftn[n=5](rfftn(Static[f64, 2, 3, 5](values.copy(), _cpu())))
    _close(odd.to_host(), values)


def test_irfft2_inverts_rfft2() raises:
    var values = List[Scalar[f64]](capacity=15)
    for i in range(15):
        values.append(Float64(i * 3 % 8) - 3.0)
    var half = rfft2(Static[f64, 3, 5](values.copy(), _cpu()))
    _close(
        half[0].to_host(),
        [
            6.0,
            -0.9721359549995796,
            7.97213595499958,
            -4.5,
            -8.836227706141226,
            -2.646955149129135,
            -4.5,
            -0.6916363388591931,
            -15.825180805870446,
        ],
    )
    var back = irfft2[n=5](half^)
    _close(back.to_host(), values)
    var rank2 = irfftn[n=5](rfftn(Static[f64, 3, 5](values.copy(), _cpu())))
    _close(rank2.to_host(), values)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
