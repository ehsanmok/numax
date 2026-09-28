"""Tests for `dctn`, `idctn`, `dstn` and `idstn` in `numax.fft.trig`.

Against SciPy on a `2 x 3 x 4` volume -- every axis a different length,
two of them not powers of two -- at types I, II and IV and under
`"ortho"`; each inverse by the round trip for every type; and the claim
that at rank 1 it is the 1-D transform.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static, copy
from numax.fft import dct, dctn, dst, dstn, idctn, idstn

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
        assert_almost_equal(got[i], want[i], atol=1e-11)


def test_dctn_and_dstn_match_scipy() raises:
    _close(
        dctn(_volume()).to_host(),
        [
            -8.0,
            -1.2681013422488583,
            16.97056274847714,
            -17.843539979101305,
            -13.856406460551028,
            24.05020003079641,
            -6.2803698347351e-16,
            32.45901844601404,
            -40.0,
            -36.58376301653854,
            2.51214793389404e-15,
            15.153490804093511,
            -5.656854249492376,
            -0.8966830583359302,
            -83.99999999999999,
            -12.617288119595798,
            -9.797958971132708,
            -44.80577613071087,
            0.0,
            18.559160145993136,
            39.59797974644666,
            -25.868626910316028,
            0.0,
            10.715136106222513,
        ],
    )
    _close(
        dctn[norm="ortho"](_volume()).to_host(),
        [
            -0.20412414523193154,
            -0.04575866570669407,
            0.6123724356957946,
            -0.6438732881393744,
            -0.5000000000000002,
            1.2273066309858227,
            -7.850462293418875e-17,
            1.6564173488400287,
            -1.4433756729740643,
            -1.8669073387796202,
            1.570092458683775e-16,
            0.7732983394163809,
            -0.20412414523193104,
            -0.04575866570669401,
            -4.286607049870561,
            -0.6438732881393741,
            -0.49999999999999967,
            -3.2335783637895035,
            0.0,
            1.339392013277814,
            2.0207259421636903,
            -1.8669073387796193,
            0.0,
            0.7732983394163807,
        ],
    )
    _close(
        dstn[type=4](_volume()).to_host(),
        [
            5.295973449254971,
            -22.792199196052202,
            22.43959416104298,
            -1.8696713082909195,
            -5.541237912538331,
            23.665814090857275,
            18.218370972220548,
            21.123836581214267,
            -26.194946037792853,
            -12.74672851337235,
            -4.523550589308287,
            6.490539933688971,
            -3.2225790627264086,
            38.38499038465641,
            -19.955163879593982,
            -32.57512393499954,
            -9.2205243864455,
            -7.833111991000093,
            -26.640581206558767,
            -3.87325865060638,
            29.200093932428043,
            -26.80072718354425,
            -32.65363162056655,
            2.4226991370760627,
        ],
    )
    _close(
        dctn[type=1](_volume()).to_host(),
        [
            1.0,
            6.0,
            4.0,
            -9.0,
            -3.0,
            2.0,
            0.0,
            11.0,
            -15.0,
            -14.0,
            0.0,
            7.0,
            5.0,
            6.0,
            -28.0,
            -9.0,
            -3.0,
            -14.0,
            0.0,
            7.0,
            21.0,
            -14.0,
            0.0,
            7.0,
        ],
    )
    _close(
        dstn[type=1](_volume()).to_host(),
        [
            -2.1234627860790667,
            8.537945725932605,
            13.085367288853941,
            -15.60218443358358,
            -10.661408512011672,
            20.728677090930226,
            -2.5168171447296395,
            40.128816903317045,
            -31.590871177384443,
            -42.25227968939612,
            -5.126492657346642,
            7.6433098020762795,
            9.690456528525658,
            8.537945725932602,
            -92.8189355046792,
            -15.60218443358358,
            -10.661408512011668,
            -74.62985958408171,
            -2.516817144729636,
            17.617720013107466,
            45.37977907547655,
            -42.25227968939612,
            -5.604990196949299,
            7.643309802076281,
        ],
    )


def test_inverses_round_trip_every_type() raises:
    var x = _volume().to_host()
    _close(idctn[type=1](dctn[type=1](_volume())).to_host(), x)
    _close(idctn[type=2](dctn[type=2](_volume())).to_host(), x)
    _close(idctn[type=3](dctn[type=3](_volume())).to_host(), x)
    _close(idctn[type=4](dctn[type=4](_volume())).to_host(), x)
    _close(idstn[type=1](dstn[type=1](_volume())).to_host(), x)
    _close(
        idstn[type=2, norm="ortho"](
            dstn[type=2, norm="ortho"](_volume())
        ).to_host(),
        x,
    )
    _close(idstn[type=3](dstn[type=3](_volume())).to_host(), x)
    _close(
        idstn[type=4, norm="forward"](
            dstn[type=4, norm="forward"](_volume())
        ).to_host(),
        x,
    )


def test_rank_one_is_the_one_dimensional_transform() raises:
    var v = Static[f64, 6]([1.0, -2.0, 0.5, 3.0, 0.0, 1.5], _cpu())
    _close(dctn(copy(v)).to_host(), dct(copy(v)).to_host())
    _close(dstn[type=3](copy(v)).to_host(), dst[type=3](copy(v)).to_host())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
