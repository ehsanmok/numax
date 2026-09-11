"""Tests for `numax.fft.trig`: `dct`/`idct`/`dst`/`idst` over `Tensor`.

Every forward value is `scipy.fft`'s own output, all four types under
`"backward"` and `"ortho"`, at a power-of-two length and at an odd one --
so the tables behind each type are checked against the definition rather
than against each other, and the odd length exercises the Bluestein path
the `2N`-point DFT takes when `N` is odd. The round trips then check the
inverse pairing and every placement of the scale.
"""

from std.testing import TestSuite, assert_almost_equal

from max.gpu.host import DeviceContext

from numax.core.array import Static
from numax.fft import dct, dst, idct, idst

comptime dtype = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _from[n: Int](values: List[Float64]) raises -> Static[dtype, n]:
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype](values[i]))
    return Static[dtype, n](_cpu(), out^)


def _v() -> List[Float64]:
    return [1.0, 2.0, 3.0, 4.0]


def _w() -> List[Float64]:
    """Five samples: odd, so every `2N`-point DFT is Bluestein's."""
    return [0.5, -1.25, 2.0, 0.75, -0.5]


def _assert_close[
    n: Int
](got: Static[dtype, n], want: List[Float64], atol: Float64 = 1e-12) raises:
    var values = got.to_host()
    for i in range(n):
        assert_almost_equal(Float64(values[i]), want[i], atol=atol)


def test_dct_defaults_to_type_two() raises:
    """`dct(x)` is `dct[type=2](x)` is `scipy.fft.dct([1, 2, 3, 4])`."""
    var want: List[Float64] = [
        20.0,
        -6.308644059797899,
        0.0,
        -0.4483415291679651,
    ]
    _assert_close(dct(_from[4](_v())), want)
    _assert_close(dct[type=2](_from[4](_v())), want)


def test_dct_types_match_scipy() raises:
    """`scipy.fft.dct([1, 2, 3, 4], type=t)` for `t` in I..IV."""
    _assert_close(dct[type=1](_from[4](_v())), [15.0, -4.0, 0.0, -1.0])
    _assert_close(
        dct[type=3](_from[4](_v())),
        [
            11.999626276085149,
            -9.102943217749221,
            2.6176618435106507,
            -1.51434490184658,
        ],
    )
    _assert_close(
        dct[type=4](_from[4](_v())),
        [
            10.181592984263283,
            -9.446695610035624,
            5.010298174943416,
            -4.689564857456725,
        ],
    )


def test_dct_ortho_matches_scipy() raises:
    """`scipy.fft.dct([1, 2, 3, 4], type=t, norm="ortho")` -- the boundary
    `sqrt(2)` tweaks land on the input side for I and III and the output
    side for I and II, which is what a wrong placement would get wrong."""
    _assert_close(
        dct[type=1, norm="ortho"](_from[4](_v())),
        [
            4.927992798267446,
            -2.1402990980327403,
            0.845509893628814,
            -0.6473946022019633,
        ],
    )
    _assert_close(
        dct[type=2, norm="ortho"](_from[4](_v())),
        [5.0, -2.230442497387663, 0.0, -0.15851266778110729],
    )
    _assert_close(
        dct[type=3, norm="ortho"](_from[4](_v())),
        [
            4.3889551651687695,
            -3.071929829606556,
            1.0719298296065565,
            -0.3889551651687705,
        ],
    )
    _assert_close(
        dct[type=4, norm="ortho"](_from[4](_v())),
        [
            3.5997367212269715,
            -3.3399112628306895,
            1.7714079076345361,
            -1.6580115557608877,
        ],
    )


def test_dst_types_match_scipy() raises:
    """`scipy.fft.dst([1, 2, 3, 4], type=t)` for `t` in I..IV."""
    _assert_close(
        dst[type=1](_from[4](_v())),
        [
            15.388417685876266,
            -6.881909602355868,
            3.6327126400268037,
            -1.624598481164532,
        ],
    )
    _assert_close(
        dst(_from[4](_v())),
        [13.065629648763766, -5.65685424949238, 5.41196100146197, -4.0],
    )
    _assert_close(
        dst[type=3](_from[4](_v())),
        [
            13.137071184544089,
            -1.6199144044217753,
            0.723231346085845,
            -0.5197830649482906,
        ],
    )
    _assert_close(
        dst[type=4](_from[4](_v())),
        [
            15.447561493151783,
            -0.4469333786714663,
            1.0031506944070392,
            0.4083909335848669,
        ],
    )


def test_dst_ortho_matches_scipy() raises:
    """`scipy.fft.dst([1, 2, 3, 4], type=t, norm="ortho")`."""
    _assert_close(
        dst[type=1, norm="ortho"](_from[4](_v())),
        [
            4.866244947338651,
            -2.1762508994828216,
            1.1487646027368057,
            -0.5137431483730079,
        ],
    )
    _assert_close(
        dst[type=2, norm="ortho"](_from[4](_v())),
        [4.619397662556434, -2.0, 1.9134171618254485, -1.0],
    )
    _assert_close(
        dst[type=3, norm="ortho"](_from[4](_v())),
        [
            5.230442497387663,
            -1.1585126677811075,
            0.841487332218893,
            -0.7695575026123374,
        ],
    )
    _assert_close(
        dst[type=4, norm="ortho"](_from[4](_v())),
        [
            5.461537742301907,
            -0.15801481139860443,
            0.35466732928360567,
            0.14438799925648219,
        ],
    )


def test_odd_length_matches_scipy() raises:
    """All eight types at `N = 5`, where the `2N`-point DFT is Bluestein's
    and DCT-I's `2(N-1) = 8` and DST-I's `2(N+1) = 12` are not both
    powers of two: `scipy.fft.dct`/`dst` of `[.5, -1.25, 2, .75, -.5]`."""
    _assert_close(
        dct[type=1](_from[5](_w())),
        [3.0, -1.8284271247461898, -4.0, 3.82842712474619, 5.0],
    )
    _assert_close(
        dct[type=2](_from[5](_w())),
        [
            3.0,
            -0.44902797657958526,
            -3.6909830056250525,
            4.97979656976556,
            4.8090169943749475,
        ],
    )
    _assert_close(
        dct[type=3](_from[5](_w())),
        [
            1.9310875708256687,
            -2.8230988882987558,
            -4.5,
            2.9689969220490715,
            4.923014395424016,
        ],
    )
    _assert_close(
        dct[type=4](_from[5](_w())),
        [
            2.1131504394394978,
            -3.3560487743115597,
            0.0,
            6.877296697722295,
            1.7956397122575984,
        ],
    )
    _assert_close(
        dst[type=1](_from[5](_w())),
        [
            3.1339745962155616,
            -1.732050807568877,
            -4.0,
            5.196152422706632,
            4.866025403784438,
        ],
    )
    _assert_close(
        dst[type=2](_from[5](_w())),
        [
            3.1909830056250525,
            -2.6286555605956674,
            -4.309016994374947,
            4.2532540417602,
            5.0,
        ],
    )
    _assert_close(
        dst[type=3](_from[5](_w())),
        [
            3.0022066155862843,
            -0.7142341973018571,
            -3.500000000000001,
            5.804404141051331,
            3.08796332816319,
        ],
    )
    _assert_close(
        dst[type=4](_from[5](_w())),
        [
            2.1987067861249683,
            1.469551599625914,
            -5.656854249492381,
            0.3891885726750084,
            5.2062115611626965,
        ],
    )


def test_idct_and_idst_match_scipy() raises:
    """`scipy.fft.idct([1, 2, 3, 4], type=2)` and `idst(..., type=3)`: the
    paired type's forward transform over `2N`, checked as values rather
    than only as a round trip."""
    _assert_close(
        idct(_from[4](_v())),
        [
            1.4999532845106436,
            -1.1378679022186526,
            0.32720773043883133,
            -0.1892931127308225,
        ],
    )
    _assert_close(
        idst[type=3](_from[4](_v())),
        [
            1.6332037060954707,
            -0.7071067811865475,
            0.6764951251827462,
            -0.5,
        ],
    )


def test_idct_inverts_dct_for_every_type_and_norm() raises:
    """`idct(dct(x)) == x` at `N = 5` for types I-IV under all three
    norms: the type pairing and every placement of the `1/M`."""
    _assert_close(idct[type=1](dct[type=1](_from[5](_w()))), _w())
    _assert_close(idct[type=2](dct[type=2](_from[5](_w()))), _w())
    _assert_close(idct[type=3](dct[type=3](_from[5](_w()))), _w())
    _assert_close(idct[type=4](dct[type=4](_from[5](_w()))), _w())
    _assert_close(
        idct[type=1, norm="ortho"](dct[type=1, norm="ortho"](_from[5](_w()))),
        _w(),
    )
    _assert_close(
        idct[type=2, norm="ortho"](dct[type=2, norm="ortho"](_from[5](_w()))),
        _w(),
    )
    _assert_close(
        idct[type=3, norm="ortho"](dct[type=3, norm="ortho"](_from[5](_w()))),
        _w(),
    )
    _assert_close(
        idct[type=4, norm="ortho"](dct[type=4, norm="ortho"](_from[5](_w()))),
        _w(),
    )
    _assert_close(
        idct[type=1, norm="forward"](
            dct[type=1, norm="forward"](_from[5](_w()))
        ),
        _w(),
    )
    _assert_close(
        idct[type=2, norm="forward"](
            dct[type=2, norm="forward"](_from[5](_w()))
        ),
        _w(),
    )
    _assert_close(
        idct[type=3, norm="forward"](
            dct[type=3, norm="forward"](_from[5](_w()))
        ),
        _w(),
    )
    _assert_close(
        idct[type=4, norm="forward"](
            dct[type=4, norm="forward"](_from[5](_w()))
        ),
        _w(),
    )


def test_idst_inverts_dst_for_every_type_and_norm() raises:
    """`idst(dst(x)) == x` at `N = 5` for types I-IV under all three
    norms."""
    _assert_close(idst[type=1](dst[type=1](_from[5](_w()))), _w())
    _assert_close(idst[type=2](dst[type=2](_from[5](_w()))), _w())
    _assert_close(idst[type=3](dst[type=3](_from[5](_w()))), _w())
    _assert_close(idst[type=4](dst[type=4](_from[5](_w()))), _w())
    _assert_close(
        idst[type=1, norm="ortho"](dst[type=1, norm="ortho"](_from[5](_w()))),
        _w(),
    )
    _assert_close(
        idst[type=2, norm="ortho"](dst[type=2, norm="ortho"](_from[5](_w()))),
        _w(),
    )
    _assert_close(
        idst[type=3, norm="ortho"](dst[type=3, norm="ortho"](_from[5](_w()))),
        _w(),
    )
    _assert_close(
        idst[type=4, norm="ortho"](dst[type=4, norm="ortho"](_from[5](_w()))),
        _w(),
    )
    _assert_close(
        idst[type=1, norm="forward"](
            dst[type=1, norm="forward"](_from[5](_w()))
        ),
        _w(),
    )
    _assert_close(
        idst[type=2, norm="forward"](
            dst[type=2, norm="forward"](_from[5](_w()))
        ),
        _w(),
    )
    _assert_close(
        idst[type=3, norm="forward"](
            dst[type=3, norm="forward"](_from[5](_w()))
        ),
        _w(),
    )
    _assert_close(
        idst[type=4, norm="forward"](
            dst[type=4, norm="forward"](_from[5](_w()))
        ),
        _w(),
    )


def test_forward_norm_is_backward_scaled_by_the_inverse_factor() raises:
    """`dct[norm="forward"]` is `dct[norm="backward"] / (2N)` -- the same
    `1/M` that `idct[norm="backward"]` applies, moved to this side."""
    var forward = dct[norm="forward"](_from[4](_v())).to_host()
    var backward = dct(_from[4](_v())).to_host()
    for k in range(4):
        assert_almost_equal(
            Float64(forward[k]), Float64(backward[k]) / 8.0, atol=1e-14
        )


def test_dct_of_a_constant_is_a_dc_spike() raises:
    """DCT-II of `[c, c, c, c]` is `[2Nc, 0, 0, 0]`: every cosine sum past
    `k = 0` cancels exactly, the cleanest check on the post-twiddle."""
    var constant: List[Float64] = [1.5, 1.5, 1.5, 1.5]
    _assert_close(dct(_from[4](constant)), [12.0, 0.0, 0.0, 0.0])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
