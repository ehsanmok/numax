"""Tests for the Schur-based matrix functions over `Tensor`: `sqrtm`,
`logm`, `funm`, `cosm`, `sinm` and `fractional_matrix_power`, against
`scipy.linalg` on two matrices -- one with a real spectrum, one with two
complex pairs -- and against the identities that define them (`X @ X ==
a`, `expm(logm(a)) == a`, `cos^2 + sin^2 == I`), plus the repeated and
defective eigenvalue cases the recurrences are chosen for."""

from std.testing import TestSuite, assert_almost_equal, assert_raises

from max.gpu.host import DeviceContext

from numax.core.array import Static, zeros
from numax.core.numeric import FloatLike
from numax.linalg import (
    cosm,
    expm,
    fractional_matrix_power,
    funm,
    logm,
    matmul,
    sinm,
    sqrtm,
)

comptime dtype = DType.float64


def _matrix[n: Int](values: List[Float64]) raises -> Static[dtype, n, n]:
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, n, n](ctx)
    var host = a.to_host()
    for i in range(n * n):
        host[i] = Scalar[dtype](values[i])
    a.copy_from_host(host)
    return a^


def _real_spectrum() raises -> Static[dtype, 4, 4]:
    """Eigenvalues 0.80, 4.08, 5.27, 9.84 -- the general matrix of the
    eigen tests shifted by `3 I` so every eigenvalue is positive."""
    return _matrix[4](
        [
            7.0,
            1.0,
            -2.0,
            2.0,
            1.0,
            5.0,
            0.0,
            1.0,
            -2.0,
            0.0,
            6.0,
            -2.0,
            2.0,
            1.0,
            -2.0,
            2.0,
        ]
    )


def _complex_pairs() raises -> Static[dtype, 4, 4]:
    """Eigenvalues `1 +- 2.449i` and `2 +- 2i`."""
    return _matrix[4](
        [
            1.0,
            -3.0,
            0.5,
            0.0,
            2.0,
            1.0,
            0.0,
            1.5,
            0.0,
            0.0,
            2.0,
            -4.0,
            0.0,
            0.0,
            1.0,
            2.0,
        ]
    )


def _assert_matrix(
    mut got: Static[dtype, 4, 4], want: List[Float64], atol: Float64
) raises:
    var host = got.to_host()
    for i in range(16):
        assert_almost_equal(Float64(host[i]), want[i], atol=atol)


def test_sqrtm_matches_scipy_and_squares_back() raises:
    var a = _real_spectrum()
    var root = sqrtm(a)
    _assert_matrix(
        root,
        [
            2.5727352410123583,
            0.18559589366729365,
            -0.3576095701879981,
            0.46765686054721683,
            0.1855958936672936,
            2.210885685807157,
            0.045549590277341845,
            0.2747069044862566,
            -0.3576095701879981,
            0.045549590277341956,
            2.3669156757624847,
            -0.5174464358417614,
            0.46765686054721683,
            0.2747069044862569,
            -0.5174464358417616,
            1.1992007185810705,
        ],
        1e-12,
    )
    var a2 = _real_spectrum()
    var again = sqrtm(a2)
    var square = matmul(root, again)
    _assert_matrix(
        square,
        [
            7.0,
            1.0,
            -2.0,
            2.0,
            1.0,
            5.0,
            0.0,
            1.0,
            -2.0,
            0.0,
            6.0,
            -2.0,
            2.0,
            1.0,
            -2.0,
            2.0,
        ],
        1e-12,
    )


def test_sqrtm_of_complex_pairs_is_real_and_matches_scipy() raises:
    var b = _complex_pairs()
    var root = sqrtm(b)
    _assert_matrix(
        root,
        [
            1.3501391245098762,
            -1.1109966171408638,
            0.11857338540631825,
            0.2159537447250708,
            0.7406644114272426,
            1.3501391245098762,
            -0.07756990179043619,
            0.42708015656863835,
            0.0,
            0.0,
            1.5537739740300374,
            -1.2871885058111652,
            0.0,
            0.0,
            0.3217971264527913,
            1.5537739740300374,
        ],
        1e-12,
    )


def test_sqrtm_handles_repeated_and_defective_eigenvalues() raises:
    # A Jordan block: the Parlett recurrence would divide by zero here; the
    # Bjorck-Hammarling one gives [[2, 1/4], [0, 2]].
    var j = _matrix[2]([4.0, 1.0, 0.0, 4.0])
    var root = sqrtm(j).to_host()
    assert_almost_equal(Float64(root[0]), 2.0, atol=1e-14)
    assert_almost_equal(Float64(root[1]), 0.25, atol=1e-14)
    assert_almost_equal(Float64(root[2]), 0.0, atol=1e-14)
    assert_almost_equal(Float64(root[3]), 2.0, atol=1e-14)
    # A scalar matrix: repeated eigenvalue, no coupling.
    var two = _matrix[3]([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0])
    var r2 = sqrtm(two).to_host()
    for i in range(3):
        for k in range(3):
            var want = 1.4142135623730951 if i == k else 0.0
            assert_almost_equal(Float64(r2[i * 3 + k]), want, atol=1e-14)


def test_logm_matches_scipy_and_inverts_expm() raises:
    var a = _real_spectrum()
    var log_a = logm(a)
    _assert_matrix(
        log_a,
        [
            1.8203103453884637,
            0.12903400074021665,
            -0.2396021550279023,
            0.4895115717347354,
            0.12903400074021665,
            1.557244827565126,
            0.09142550294123086,
            0.33868004698005,
            -0.23960215502790255,
            0.09142550294123097,
            1.6313331536084617,
            -0.604400437471401,
            0.48951157173473525,
            0.33868004698005033,
            -0.604400437471401,
            0.12691011048820963,
        ],
        1e-12,
    )
    var b = _complex_pairs()
    var log_b = logm(b)
    _assert_matrix(
        log_b,
        [
            0.9729550745276572,
            -1.4491176910934915,
            0.05609185760473093,
            0.46767873601697896,
            0.9660784607289941,
            0.9729550745276572,
            -0.1634625818218598,
            0.3816556892149109,
            0.0,
            0.0,
            1.0397207708399179,
            -1.5707963267948966,
            0.0,
            0.0,
            0.3926990816987243,
            1.0397207708399179,
        ],
        1e-12,
    )
    var back = expm(log_b)
    _assert_matrix(
        back,
        [
            1.0,
            -3.0,
            0.5,
            0.0,
            2.0,
            1.0,
            0.0,
            1.5,
            0.0,
            0.0,
            2.0,
            -4.0,
            0.0,
            0.0,
            1.0,
            2.0,
        ],
        1e-12,
    )
    # The Jordan block again: logm([[4, 1], [0, 4]]) = [[ln 4, 1/4], [0, ln 4]].
    var j = _matrix[2]([4.0, 1.0, 0.0, 4.0])
    var lj = logm(j).to_host()
    assert_almost_equal(Float64(lj[0]), 1.3862943611198906, atol=1e-13)
    assert_almost_equal(Float64(lj[1]), 0.25, atol=1e-13)
    assert_almost_equal(Float64(lj[2]), 0.0, atol=1e-13)


def _exp_f[T: FloatLike](x: T) -> T:
    return x.exp()


def test_funm_at_exp_agrees_with_expm_including_complex_blocks() raises:
    var b = _complex_pairs()
    var through_funm = funm[f=_exp_f](b)
    _assert_matrix(
        through_funm,
        [
            -2.0928207548056434,
            -2.124555501929299,
            -2.270048124039064,
            -5.102892086309344,
            1.4163703346195329,
            -2.092820754805643,
            2.2939614758105233,
            -2.50310561812712,
            0.0,
            0.0,
            -3.0749323206393235,
            -13.437699394856793,
            0.0,
            0.0,
            3.359424848714199,
            -3.0749323206393235,
        ],
        1e-12,
    )


def test_funm_raises_on_a_coupled_repeated_eigenvalue() raises:
    var j = _matrix[2]([4.0, 1.0, 0.0, 4.0])
    with assert_raises(contains="Jordan"):
        _ = funm[f=_exp_f](j)


def test_cosm_and_sinm_match_scipy_and_satisfy_the_pythagorean_identity() raises:
    var b = _complex_pairs()
    var c = cosm(b)
    _assert_matrix(
        c,
        [
            3.152332430558969,
            5.923865917822582,
            -2.7803642601589162,
            0.9453035354972279,
            -3.9492439452150547,
            3.1523324305589706,
            -0.17463946942522343,
            -5.884888205518509,
            0.0,
            0.0,
            -1.565625835315743,
            6.595789672622473,
            0.0,
            0.0,
            -1.6489474181556183,
            -1.5656258353157435,
        ],
        1e-11,
    )
    var b2 = _complex_pairs()
    var s = sinm(b2)
    _assert_matrix(
        s,
        [
            4.909466878032708,
            -3.8036705636189216,
            0.2892087420906246,
            6.4012701258096705,
            2.535780375745949,
            4.909466878032708,
            -2.4782449587860476,
            0.6567923911016105,
            0.0,
            0.0,
            3.4209548611170133,
            3.01861297064723,
            0.0,
            0.0,
            -0.7546532426618079,
            3.4209548611170133,
        ],
        1e-11,
    )
    var b3 = _complex_pairs()
    var b4 = _complex_pairs()
    var c2 = cosm(b3)
    var s2 = sinm(b4)
    var cc = matmul(c, c2)
    var ss = matmul(s, s2)
    var hc = cc.to_host()
    var hs = ss.to_host()
    for i in range(4):
        for k in range(4):
            var want = 1.0 if i == k else 0.0
            assert_almost_equal(
                Float64(hc[i * 4 + k]) + Float64(hs[i * 4 + k]),
                want,
                atol=1e-10,
            )


def test_fractional_matrix_power_matches_scipy_and_sqrtm() raises:
    var a = _real_spectrum()
    var p = fractional_matrix_power(a, 0.3)
    _assert_matrix(
        p,
        [
            1.749286087439507,
            0.07389062245809613,
            -0.14026345006113408,
            0.21335435288485383,
            0.07389062245809619,
            1.6045264840710174,
            0.028794007601591676,
            0.13396036327016686,
            -0.1402634500611341,
            0.028794007601591676,
            1.6602612053007884,
            -0.24664451379380098,
            0.21335435288485383,
            0.133960363270167,
            -0.24664451379380103,
            1.07948427108867,
        ],
        1e-12,
    )
    var a3 = _real_spectrum()
    var a4 = _real_spectrum()
    var half = fractional_matrix_power(a3, 0.5).to_host()
    var root = sqrtm(a4).to_host()
    for i in range(16):
        assert_almost_equal(Float64(half[i]), Float64(root[i]), atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
