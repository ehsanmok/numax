"""Tests for the `Tensor` tier of `numax.linalg`, the half MAX computes.

The load-bearing claim is agreement: `matmul` over a `Tensor` and `matmul`
over an `Array[T, n*n]` are the same word, resolved by argument type, and
they must produce the same numbers. Everything else here checks that the
MAX kernel was handed the shape it was supposed to get -- a product against
a hand-computed reference, a matvec against the equivalent matmul, a band
against the triangle it names.
"""

from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_almost_equal, assert_raises

from numax import Plain
from numax.core.array import Shaped, zeros_dyn
from numax.core.array import zeros as array_zeros
from numax.core.array import transpose, tril, triu
from std.collections import Array

from numax.linalg import (
    asum,
    axpy,
    batched_matmul,
    cholesky,
    det,
    dot,
    lu_factor,
    matmul,
    matvec,
    nrm2,
    outer,
    solve,
)

comptime P = Plain[DType.float64, 1]


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def test_matmul_matches_hand_computed_product() raises:
    var ctx = _cpu()
    var a = Shaped[DType.float64, 2, 3](ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var b = Shaped[DType.float64, 3, 2](ctx, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    var c = matmul(a, b)
    var got = c.to_host()
    assert_almost_equal(Float64(got[0]), 58.0, atol=1e-12)
    assert_almost_equal(Float64(got[1]), 64.0, atol=1e-12)
    assert_almost_equal(Float64(got[2]), 139.0, atol=1e-12)
    assert_almost_equal(Float64(got[3]), 154.0, atol=1e-12)


def test_tensor_matmul_agrees_with_array_matmul() raises:
    """The two spellings of `matmul` are one name over two storages.

    Same entries, same product, so overload resolution is picking a
    different implementation of the same operation rather than a different
    operation.
    """
    var ctx = _cpu()
    var entries: List[Float64] = [
        2.0,
        -1.0,
        0.5,
        3.0,
        1.0,
        4.0,
        -2.0,
        0.25,
        6.0,
    ]

    var ta = Shaped[DType.float64, 3, 3](ctx, entries.copy())
    var tb = Shaped[DType.float64, 3, 3](ctx, entries.copy())
    var tensor_product = matmul(ta, tb).to_host()

    var aa = array_zeros[P, 9]()
    var ab = array_zeros[P, 9]()
    for i in range(9):
        aa[i] = P(entries[i])
        ab[i] = P(entries[i])
    var array_product = matmul[P, 3](aa, ab)

    for i in range(9):
        assert_almost_equal(
            Float64(tensor_product[i]),
            Float64(array_product[i].v),
            atol=1e-12,
        )


def test_matmul_dynamic_agrees_with_static() raises:
    """The run-time-shaped overload computes what the compile-time one does."""
    var ctx = _cpu()
    var entries: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    var sa = Shaped[DType.float64, 2, 3](ctx, entries.copy())
    var sb = Shaped[DType.float64, 3, 2](ctx, entries.copy())
    var static_product = matmul(sa, sb).to_host()

    var da = zeros_dyn[DType.float64, 2](2, 3, ctx=ctx)
    var db = zeros_dyn[DType.float64, 2](3, 2, ctx=ctx)
    da.copy_from_host(entries.copy())
    db.copy_from_host(entries.copy())
    var dynamic_product = matmul(da, db).to_host()

    for i in range(4):
        assert_almost_equal(
            Float64(static_product[i]), Float64(dynamic_product[i]), atol=1e-12
        )


def test_matmul_dynamic_rejects_mismatched_shapes() raises:
    """The check the static overload gets from the type system for free."""
    var ctx = _cpu()
    var a = zeros_dyn[DType.float64, 2](2, 3, ctx=ctx)
    var b = zeros_dyn[DType.float64, 2](4, 2, ctx=ctx)
    with assert_raises(contains="matmul shape mismatch"):
        _ = matmul(a, b)


def test_matvec_agrees_with_matmul_against_a_column() raises:
    """`matvec` is a matmul at `n == 1`, so it had better agree with one."""
    var ctx = _cpu()
    var a = Shaped[DType.float64, 2, 3](ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var x = Shaped[DType.float64, 3](ctx, [2.0, -1.0, 0.5])
    var by_matvec = matvec(a, x).to_host()

    var column = Shaped[DType.float64, 3, 1](ctx, [2.0, -1.0, 0.5])
    var by_matmul = matmul(a, column).to_host()

    for i in range(2):
        assert_almost_equal(
            Float64(by_matvec[i]), Float64(by_matmul[i]), atol=1e-12
        )
    assert_almost_equal(Float64(by_matvec[0]), 1.5, atol=1e-12)
    assert_almost_equal(Float64(by_matvec[1]), 6.0, atol=1e-12)


def test_batched_matmul_is_one_product_per_leading_index() raises:
    var ctx = _cpu()
    var a = Shaped[DType.float64, 2, 2, 2](
        ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    )
    var b = Shaped[DType.float64, 2, 2, 2](
        ctx, [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]
    )
    var got = batched_matmul(a, b).to_host()

    # First batch multiplies by the identity, the second by twice it.
    var want: List[Float64] = [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]
    for i in range(8):
        assert_almost_equal(Float64(got[i]), want[i], atol=1e-12)


def test_batched_matmul_agrees_with_per_matrix_matmul() raises:
    """One launch over the batch equals one call per matrix."""
    var ctx = _cpu()
    var first: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    var second: List[Float64] = [5.0, 6.0, 7.0, 8.0]

    var batch_a = Shaped[DType.float64, 2, 2, 2](
        ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    )
    var batch_b = Shaped[DType.float64, 2, 2, 2](
        ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    )
    var batched = batched_matmul(batch_a, batch_b).to_host()

    var a0 = Shaped[DType.float64, 2, 2](ctx, first.copy())
    var b0 = Shaped[DType.float64, 2, 2](ctx, first.copy())
    var single0 = matmul(a0, b0).to_host()
    var a1 = Shaped[DType.float64, 2, 2](ctx, second.copy())
    var b1 = Shaped[DType.float64, 2, 2](ctx, second.copy())
    var single1 = matmul(a1, b1).to_host()

    for i in range(4):
        assert_almost_equal(
            Float64(batched[i]), Float64(single0[i]), atol=1e-12
        )
        assert_almost_equal(
            Float64(batched[4 + i]), Float64(single1[i]), atol=1e-12
        )


def test_tril_and_triu_split_the_matrix_at_the_diagonal() raises:
    """Lower plus upper counts the diagonal twice and nothing else."""
    var ctx = _cpu()
    var entries: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    var a = Shaped[DType.float64, 3, 3](ctx, entries.copy())
    var lower = tril(a).to_host()
    var b = Shaped[DType.float64, 3, 3](ctx, entries.copy())
    var upper = triu(b).to_host()

    for row in range(3):
        for col in range(3):
            var i = row * 3 + col
            var sum = Float64(lower[i]) + Float64(upper[i])
            var doubled = entries[i] * 2.0 if row == col else entries[i]
            assert_almost_equal(sum, doubled, atol=1e-12)
            if row < col:
                assert_almost_equal(Float64(lower[i]), 0.0, atol=1e-12)
            if row > col:
                assert_almost_equal(Float64(upper[i]), 0.0, atol=1e-12)


def test_tril_and_triu_accept_a_non_square_matrix() raises:
    """The MAX band kernel is rectangular; the host loop it replaced was not.

    A 2x3 has one entry below the diagonal and three above it, so this
    would not have compiled against the square-only `n, n` signature.
    """
    var ctx = _cpu()
    var a = Shaped[DType.float64, 2, 3](ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var lower = tril(a).to_host()
    var b = Shaped[DType.float64, 2, 3](ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var upper = triu(b).to_host()

    var want_lower: List[Float64] = [1.0, 0.0, 0.0, 4.0, 5.0, 0.0]
    var want_upper: List[Float64] = [1.0, 2.0, 3.0, 0.0, 5.0, 6.0]
    for i in range(6):
        assert_almost_equal(Float64(lower[i]), want_lower[i], atol=1e-12)
        assert_almost_equal(Float64(upper[i]), want_upper[i], atol=1e-12)


def _spd_5x5() raises -> List[Float64]:
    """A symmetric positive definite 5x5, diagonally dominant by hand."""
    return [
        9.0,
        3.0,
        1.0,
        2.0,
        0.0,
        3.0,
        10.0,
        2.0,
        1.0,
        1.0,
        1.0,
        2.0,
        8.0,
        3.0,
        2.0,
        2.0,
        1.0,
        3.0,
        11.0,
        4.0,
        0.0,
        1.0,
        2.0,
        4.0,
        7.0,
    ]


def test_cholesky_reconstructs_the_matrix() raises:
    """`L @ L.T` is `A` again, which is the whole claim of a Cholesky.

    Run at `block=2` on a 5x5 so the blocked loop takes three steps with a
    ragged last one, and the MAX trailing update runs for real rather than
    being skipped by a single-block shortcut.
    """
    var ctx = _cpu()
    var entries = _spd_5x5()
    var a = Shaped[DType.float64, 5, 5](ctx, entries.copy())
    var lower = cholesky[DType.float64, 5, False, 2](a)
    var upper = transpose(lower)
    var reconstructed = matmul(lower, upper).to_host()

    for i in range(25):
        assert_almost_equal(Float64(reconstructed[i]), entries[i], atol=1e-12)


def test_cholesky_is_lower_triangular() raises:
    var ctx = _cpu()
    var a = Shaped[DType.float64, 5, 5](ctx, _spd_5x5())
    var lower = cholesky[DType.float64, 5, False, 2](a).to_host()
    for row in range(5):
        for col in range(row + 1, 5):
            assert_almost_equal(Float64(lower[row * 5 + col]), 0.0, atol=1e-15)


def test_cholesky_blocking_does_not_change_the_answer() raises:
    """Every block size factors the same matrix into the same `L`.

    `block=5` on a 5x5 is the unblocked algorithm -- one panel, no trailing
    update, no MAX call -- so this pins the blocked path against it and
    would catch an off-by-one in the panel offsets or a trailing update
    applied to the wrong submatrix.
    """
    var ctx = _cpu()
    var a1 = Shaped[DType.float64, 5, 5](ctx, _spd_5x5())
    var unblocked = cholesky[DType.float64, 5, False, 5](a1).to_host()

    for bs in [1, 2, 3, 4]:
        var a2 = Shaped[DType.float64, 5, 5](ctx, _spd_5x5())
        var blocked: List[Scalar[DType.float64]]
        if bs == 1:
            blocked = cholesky[DType.float64, 5, False, 1](a2).to_host()
        elif bs == 2:
            blocked = cholesky[DType.float64, 5, False, 2](a2).to_host()
        elif bs == 3:
            blocked = cholesky[DType.float64, 5, False, 3](a2).to_host()
        else:
            blocked = cholesky[DType.float64, 5, False, 4](a2).to_host()
        for i in range(25):
            assert_almost_equal(
                Float64(blocked[i]), Float64(unblocked[i]), atol=1e-12
            )


def test_cholesky_agrees_with_the_array_tier() raises:
    """The two `cholesky` overloads factor the same matrix the same way."""
    var ctx = _cpu()
    var entries = _spd_5x5()
    var a = Shaped[DType.float64, 5, 5](ctx, entries.copy())
    var tensor_factor = cholesky[DType.float64, 5, False, 2](a).to_host()

    var lifted = array_zeros[P, 25]()
    for i in range(25):
        lifted[i] = P(entries[i])
    var array_factor = cholesky[P, 5](lifted)

    for i in range(25):
        assert_almost_equal(
            Float64(tensor_factor[i]),
            Float64(array_factor[i].v),
            atol=1e-12,
        )


def test_cholesky_rejects_a_matrix_that_is_not_positive_definite() raises:
    """Host-side, so it can raise where the tier-1 sibling must floor."""
    var ctx = _cpu()
    var a = Shaped[DType.float64, 2, 2](ctx, [1.0, 2.0, 2.0, 1.0])
    with assert_raises(contains="not positive definite"):
        _ = cholesky[DType.float64, 2, False, 2](a)


def _nonsymmetric_4x4() raises -> List[Float64]:
    return [
        4.0,
        1.0,
        2.0,
        0.5,
        1.0,
        5.0,
        0.0,
        2.0,
        2.0,
        0.0,
        6.0,
        1.0,
        0.5,
        2.0,
        1.0,
        7.0,
    ]


def test_solve_residual_is_at_machine_precision() raises:
    """`a @ x` reproduces `b`, which is the only thing a solve promises."""
    var ctx = _cpu()
    var entries = _nonsymmetric_4x4()
    var rhs: List[Float64] = [1.0, 2.0, 3.0, 4.0]

    var a = Shaped[DType.float64, 4, 4](ctx, entries.copy())
    var b = Shaped[DType.float64, 4](ctx, rhs.copy())
    var x = solve[DType.float64, 4, False, 2](a, b)

    var a_again = Shaped[DType.float64, 4, 4](ctx, entries.copy())
    var product = matvec(a_again, x).to_host()
    for i in range(4):
        assert_almost_equal(Float64(product[i]), rhs[i], atol=1e-12)


def test_solve_blocking_does_not_change_the_answer() raises:
    """`block=n` is the unblocked LU, so it pins the blocked path."""
    var ctx = _cpu()
    var entries = _nonsymmetric_4x4()
    var rhs: List[Float64] = [1.0, 2.0, 3.0, 4.0]

    var a1 = Shaped[DType.float64, 4, 4](ctx, entries.copy())
    var b1 = Shaped[DType.float64, 4](ctx, rhs.copy())
    var unblocked = solve[DType.float64, 4, False, 4](a1, b1).to_host()

    var a2 = Shaped[DType.float64, 4, 4](ctx, entries.copy())
    var b2 = Shaped[DType.float64, 4](ctx, rhs.copy())
    var blocked = solve[DType.float64, 4, False, 2](a2, b2).to_host()

    var a3 = Shaped[DType.float64, 4, 4](ctx, entries.copy())
    var b3 = Shaped[DType.float64, 4](ctx, rhs.copy())
    var single = solve[DType.float64, 4, False, 1](a3, b3).to_host()

    for i in range(4):
        assert_almost_equal(
            Float64(blocked[i]), Float64(unblocked[i]), atol=1e-12
        )
        assert_almost_equal(
            Float64(single[i]), Float64(unblocked[i]), atol=1e-12
        )


def test_solve_agrees_with_the_array_tier() raises:
    """Same name, same system, same answer.

    The matrix is chosen to have no zero pivots, so the unpivoted `Array`
    solve is entitled to agree; `test_lu_factor_handles_a_zero_leading_pivot`
    covers the case where it is not.
    """
    var ctx = _cpu()
    var entries = _nonsymmetric_4x4()
    var rhs: List[Float64] = [1.0, 2.0, 3.0, 4.0]

    var a = Shaped[DType.float64, 4, 4](ctx, entries.copy())
    var b = Shaped[DType.float64, 4](ctx, rhs.copy())
    var tensor_x = solve[DType.float64, 4, False, 2](a, b).to_host()

    var lifted_a = array_zeros[P, 16]()
    for i in range(16):
        lifted_a[i] = P(entries[i])
    var lifted_b = array_zeros[P, 4]()
    for i in range(4):
        lifted_b[i] = P(rhs[i])
    var array_x = solve[P, 4](lifted_a, lifted_b)

    for i in range(4):
        assert_almost_equal(
            Float64(tensor_x[i]), Float64(array_x[i].v), atol=1e-10
        )


def test_lu_factor_handles_a_zero_leading_pivot() raises:
    """The exchange matrix is what pivoting buys.

    `[[0, 1], [1, 0]]` is perfectly well conditioned with a determinant of
    -1, and the unpivoted `Array` tier cannot start on it at all. This is
    the claim that the `Tensor` tier pivots for real.
    """
    var ctx = _cpu()
    var a = Shaped[DType.float64, 2, 2](ctx, [0.0, 1.0, 1.0, 0.0])
    var factorization = lu_factor[DType.float64, 2, False, 2](a)

    assert_almost_equal(Float64(factorization.det()), -1.0, atol=1e-12)

    var b = Shaped[DType.float64, 2](ctx, [3.0, 5.0])
    var x = factorization.solve(b).to_host()
    assert_almost_equal(Float64(x[0]), 5.0, atol=1e-12)
    assert_almost_equal(Float64(x[1]), 3.0, atol=1e-12)


def test_lu_factor_is_reusable_across_right_hand_sides() raises:
    """One factorization, several solves -- the reason `lu_factor` is public.

    Each answer must match what a fresh `solve` on the same system gives,
    so reusing the factorization is not quietly different from redoing it.
    """
    var ctx = _cpu()
    var entries = _nonsymmetric_4x4()
    var a = Shaped[DType.float64, 4, 4](ctx, entries.copy())
    var factorization = lu_factor[DType.float64, 4, False, 2](a)

    var first_rhs: List[Float64] = [1.0, 0.0, 0.0, 0.0]
    var second_rhs: List[Float64] = [0.0, 2.0, 0.0, -1.0]

    var b1 = Shaped[DType.float64, 4](ctx, first_rhs.copy())
    var reused_1 = factorization.solve(b1).to_host()
    var b2 = Shaped[DType.float64, 4](ctx, second_rhs.copy())
    var reused_2 = factorization.solve(b2).to_host()

    var a1 = Shaped[DType.float64, 4, 4](ctx, entries.copy())
    var fresh_b1 = Shaped[DType.float64, 4](ctx, first_rhs.copy())
    var fresh_1 = solve[DType.float64, 4, False, 2](a1, fresh_b1).to_host()
    var a2 = Shaped[DType.float64, 4, 4](ctx, entries.copy())
    var fresh_b2 = Shaped[DType.float64, 4](ctx, second_rhs.copy())
    var fresh_2 = solve[DType.float64, 4, False, 2](a2, fresh_b2).to_host()

    for i in range(4):
        assert_almost_equal(
            Float64(reused_1[i]), Float64(fresh_1[i]), atol=1e-12
        )
        assert_almost_equal(
            Float64(reused_2[i]), Float64(fresh_2[i]), atol=1e-12
        )


def test_lu_factor_det_agrees_with_the_array_tier() raises:
    var ctx = _cpu()
    var entries = _nonsymmetric_4x4()
    var a = Shaped[DType.float64, 4, 4](ctx, entries.copy())
    var tensor_det = Float64(lu_factor[DType.float64, 4, False, 2](a).det())

    var lifted = array_zeros[P, 16]()
    for i in range(16):
        lifted[i] = P(entries[i])
    var array_det = Float64(det[P, 4](lifted).v)

    assert_almost_equal(tensor_det, array_det, atol=1e-9)


def _ramp[n: Int](start: Float64, step: Float64) -> List[Scalar[DType.float64]]:
    var values = List[Scalar[DType.float64]](capacity=n)
    for i in range(n):
        values.append(Scalar[DType.float64](start + step * Float64(i)))
    return values^


def _array_of[n: Int](values: List[Scalar[DType.float64]]) -> Array[P, n]:
    var out = array_zeros[P, n]()
    for i in range(n):
        out[i] = P(values[i])
    return out^


def test_tensor_dot_agrees_with_the_array_dot() raises:
    """The two spellings of `dot` are one word, so they must agree.

    Not bit for bit: the `Array` fold is strictly left to right and the
    `Tensor` one is MAX's reassociated `ReduceSum`, which is exactly what
    both docstrings claim. The tolerance is what that difference costs at
    this length, not a hedge.
    """
    comptime n = 64
    var ctx = _cpu()
    var xv = _ramp[n](0.5, 0.25)
    var yv = _ramp[n](-3.0, 0.125)
    var x = Shaped[DType.float64, n](ctx, xv.copy())
    var y = Shaped[DType.float64, n](ctx, yv.copy())

    var want = dot(_array_of[n](xv), _array_of[n](yv)).v
    assert_almost_equal(dot(x, y), want, atol=1e-9)


def test_tensor_nrm2_and_asum_agree_with_the_array_versions() raises:
    comptime n = 48
    var ctx = _cpu()
    var xv = _ramp[n](-5.0, 0.375)
    var x = Shaped[DType.float64, n](ctx, xv.copy())
    var arr = _array_of[n](xv)

    assert_almost_equal(nrm2(x), nrm2(arr).v, atol=1e-9)
    assert_almost_equal(asum(x), asum(arr).v, atol=1e-9)


def test_tensor_axpy_agrees_with_the_array_axpy() raises:
    """`alpha` is a run-time scalar riding `elementwise`'s capture list,
    which is the thing `numax.core.tensor.map` cannot express -- so this
    also pins that the fused form got the right `alpha`."""
    comptime n = 32
    var ctx = _cpu()
    var xv = _ramp[n](1.0, 0.5)
    var yv = _ramp[n](7.0, -0.25)
    var x = Shaped[DType.float64, n](ctx, xv.copy())
    var y = Shaped[DType.float64, n](ctx, yv.copy())
    var alpha = Scalar[DType.float64](-1.75)

    var got = axpy(alpha, x, y).to_host()
    var want = axpy(P(alpha), _array_of[n](xv), _array_of[n](yv))
    for i in range(n):
        assert_almost_equal(got[i], want[i].v, atol=1e-12)


def test_tensor_outer_agrees_with_the_array_outer() raises:
    comptime n = 8
    var ctx = _cpu()
    var av = _ramp[n](2.0, 0.5)
    var bv = _ramp[n](-1.0, 0.25)
    var a = Shaped[DType.float64, n](ctx, av.copy())
    var b = Shaped[DType.float64, n](ctx, bv.copy())

    var got = outer(a, b).to_host()
    var want = outer(_array_of[n](av), _array_of[n](bv))
    for i in range(n * n):
        assert_almost_equal(got[i], want[i].v, atol=1e-12)


def test_tensor_outer_accepts_a_rectangular_result() raises:
    """`numpy.outer` does not require the two vectors to match, and the
    `Array` overload cannot express that -- its result is `n * n`."""
    comptime m = 3
    comptime n = 5
    var ctx = _cpu()
    var a = Shaped[DType.float64, m](ctx, [1.0, 2.0, 3.0])
    var b = Shaped[DType.float64, n](ctx, [1.0, 10.0, 100.0, 1000.0, 10000.0])

    var got = outer(a, b).to_host()
    for i in range(m):
        for j in range(n):
            var want = Float64(i + 1) * (10.0**j)
            assert_almost_equal(got[i * n + j], Scalar[DType.float64](want))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
