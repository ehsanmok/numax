"""Tests for `eigh(a, b)` and `eigvalsh(a, b)`, the symmetric-definite
pencil: SciPy's eigenvalues, the residual `a X = b X Lambda`, the
normalization `X^T b X = I`, agreement between the two spellings, and
`b = I` reducing to the standard `eigh`."""

from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_almost_equal

from numax.core.tensor import Static, eye, transpose
from numax.linalg import eigh, eigvalsh, matmul

comptime f64 = DType.float64


def _a() raises -> Static[f64, 4, 4]:
    return Static[f64, 4, 4](
        [
            4.0,
            1.0,
            -2.0,
            0.5,
            1.0,
            3.0,
            0.2,
            1.0,
            -2.0,
            0.2,
            5.0,
            -1.0,
            0.5,
            1.0,
            -1.0,
            2.0,
        ],
        DeviceContext(api="cpu"),
    )


def _b() raises -> Static[f64, 4, 4]:
    return Static[f64, 4, 4](
        [
            2.0,
            0.3,
            0.0,
            0.1,
            0.3,
            1.5,
            0.2,
            0.0,
            0.0,
            0.2,
            3.0,
            0.4,
            0.1,
            0.0,
            0.4,
            1.2,
        ],
        DeviceContext(api="cpu"),
    )


def test_eigenvalues_match_scipy() raises:
    # scipy.linalg.eigh(a, b)[0].
    var want: List[Float64] = [
        0.557941064584682,
        1.475322424006927,
        2.078059433439138,
        3.516553408430477,
    ]
    var got = eigh(_a(), _b()).values.to_host()
    var only = eigvalsh(_a(), _b()).to_host()
    for i in range(4):
        assert_almost_equal(got[i], want[i], atol=1e-13)
        assert_almost_equal(only[i], got[i], atol=1e-14)


def test_residual_and_b_orthonormality() raises:
    var e = eigh(_a(), _b())
    var w = e.values.to_host()
    var ax = matmul(_a(), e.vectors).to_host()
    var bx = matmul(_b(), e.vectors).to_host()
    for i in range(4):
        for j in range(4):
            assert_almost_equal(ax[i * 4 + j], bx[i * 4 + j] * w[j], atol=1e-12)
    var gram = matmul(transpose(e.vectors), matmul(_b(), e.vectors)).to_host()
    for i in range(4):
        for j in range(4):
            assert_almost_equal(
                gram[i * 4 + j], 1.0 if i == j else 0.0, atol=1e-13
            )


def test_identity_b_is_the_standard_problem() raises:
    var plain = eigh(_a()).values.to_host()
    var pencil = eigh(
        _a(), eye[4, f64](ctx=DeviceContext(api="cpu"))
    ).values.to_host()
    for i in range(4):
        assert_almost_equal(pencil[i], plain[i], atol=1e-14)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
