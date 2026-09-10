"""Tests for the vector `norm` overload and the spectral thin wrappers
`eigvalsh`, `svdvals` and `matrix_rank`.

Two claims carry the weight.

The vector `norm` at `ord=2` must equal `nrm2` exactly -- it is the same
reduction under a second name, so agreement is the specification and not an
approximation.

`matrix_rank` returns `T` rather than `Int`, which is a deliberate
divergence from NumPy. `test_matrix_rank_counts_per_lane` is why: at
`Plain[dtype, w]` one call is `w` independent matrices whose ranks need not
agree, and a single `Int` could not describe them.
"""

from std.collections import Array
from std.math import sqrt as sqrt_f64
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax import Plain
from numax.core.array import Static
from numax.linalg import inf, neg_inf, norm, nrm2
from numax.linalg.array import eigh, eigvalsh, matrix_rank, svd, svdvals

comptime dtype = DType.float64
comptime P = Plain[DType.float64, 1]
comptime P2 = Plain[DType.float64, 2]


def _v(ctx: DeviceContext) raises -> Static[dtype, 4]:
    """(3, -4, 12, -1): 2-norm 13, 1-norm 20, inf 12, -inf 1."""
    return Static[dtype, 4](ctx, [3.0, -4.0, 12.0, -1.0])


# --- the vector norm --------------------------------------------------------


def test_vector_two_norm_is_exactly_nrm2() raises:
    var ctx = DeviceContext(api="cpu")
    var a = _v(ctx)
    var by_norm = norm[dtype, 4, 2](a)
    var b = _v(ctx)
    var by_nrm2 = nrm2[dtype, 4](b)
    assert_equal(Float64(by_norm), Float64(by_nrm2))


def test_vector_two_norm_value() raises:
    var ctx = DeviceContext(api="cpu")
    var a = _v(ctx)
    # 9 + 16 + 144 + 1 = 170.
    assert_almost_equal(
        Float64(norm[dtype, 4, 2](a)), sqrt_f64(170.0), atol=1e-12
    )


def test_vector_two_norm_is_the_default() raises:
    var ctx = DeviceContext(api="cpu")
    var a = _v(ctx)
    var defaulted = Float64(norm[dtype, 4](a))
    var b = _v(ctx)
    var explicit = Float64(norm[dtype, 4, 2](b))
    assert_equal(defaulted, explicit)


def test_vector_one_norm() raises:
    var ctx = DeviceContext(api="cpu")
    var a = _v(ctx)
    assert_almost_equal(Float64(norm[dtype, 4, 1](a)), 20.0, atol=1e-12)


def test_vector_infinity_norms() raises:
    var ctx = DeviceContext(api="cpu")
    var a = _v(ctx)
    assert_almost_equal(Float64(norm[dtype, 4, inf](a)), 12.0, atol=1e-12)
    var b = _v(ctx)
    assert_almost_equal(Float64(norm[dtype, 4, neg_inf](b)), 1.0, atol=1e-12)


def test_the_matrix_overload_still_resolves() raises:
    """Rank selects the overload, so a 2x2 must still reach the matrix
    `norm` and get its Frobenius default rather than the vector one."""
    var ctx = DeviceContext(api="cpu")
    var m = Static[dtype, 2, 2](ctx, [3.0, 0.0, 0.0, 4.0])
    assert_almost_equal(Float64(norm[dtype, 2](m)), 5.0, atol=1e-12)


# --- eigvalsh and svdvals ---------------------------------------------------


def _symmetric() -> Array[P, 9]:
    """diag(2, 3, 6) rotated: eigenvalues are exactly 2, 3 and 6."""
    var out = Array[P, 9](fill=P.constant(0.0))
    var values = [4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 3.0]
    for i in range(9):
        out[i] = P.constant(values[i])
    return out^


def test_eigvalsh_is_eigh_without_the_vectors() raises:
    var a = _symmetric()
    var only_values = eigvalsh[P, 3](a)
    var b = _symmetric()
    var both = eigh[P, 3](b)
    for i in range(3):
        assert_almost_equal(only_values[i].v, both[0][i].v, atol=0.0)


def test_svdvals_is_svd_without_the_vectors() raises:
    var a = _symmetric()
    var only_values = svdvals[P, 3](a)
    var b = _symmetric()
    var full = svd[P, 3](b)
    for i in range(3):
        assert_almost_equal(only_values[i].v, full[1][i].v, atol=0.0)


def test_eigenvalues_sum_to_the_trace() raises:
    """An independent check that `eigvalsh` returns eigenvalues at all."""
    var a = _symmetric()
    var values = eigvalsh[P, 3](a)
    var total = 0.0
    for i in range(3):
        total += values[i].v
    # trace of the matrix above is 4 + 4 + 3.
    assert_almost_equal(total, 11.0, atol=1e-9)


# --- matrix_rank ------------------------------------------------------------


def test_matrix_rank_of_a_full_rank_matrix() raises:
    var a = _symmetric()
    assert_almost_equal(matrix_rank[P, 3](a).v, 3.0, atol=0.0)


def test_matrix_rank_of_a_rank_deficient_matrix() raises:
    """Row 2 is row 0 doubled and row 3 is their sum, so the rank is 1."""
    var out = Array[P, 9](fill=P.constant(0.0))
    var values = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0]
    for i in range(9):
        out[i] = P.constant(values[i])
    assert_almost_equal(matrix_rank[P, 3](out).v, 1.0, atol=0.0)


def test_matrix_rank_counts_per_lane() raises:
    """The reason the return type is `T` and not `Int`. Lane 0 carries a
    full-rank matrix and lane 1 a rank-1 one, and one call must report both
    -- which a single `Int` could not."""
    var a = Array[P2, 9](fill=P2.constant(0.0))
    var full = [4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 3.0]
    var deficient = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0]
    for i in range(9):
        a[i] = P2(SIMD[DType.float64, 2](full[i], deficient[i]))

    var rank = matrix_rank[P2, 3](a)
    assert_almost_equal(Float64(rank.v[0]), 3.0, atol=0.0)
    assert_almost_equal(Float64(rank.v[1]), 1.0, atol=0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
