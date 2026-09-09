"""Tests for the product names both `numax.linalg` tiers gained: `inner`,
`kron` and `matrix_power`.

Each is checked against the definition rather than against itself. `kron`
goes to a hand-written quadruple loop, `matrix_power` to repeated `matmul`,
and `inner` to `matmul` against an explicitly transposed operand -- the
point of `inner` being that it never materializes that transpose, so the
transposed spelling is the independent reference and not the
implementation.

The two tiers are also compared, since they share these names and a caller
picking a tier by import must get the same arithmetic.
"""

from std.collections import Array
from std.testing import TestSuite, assert_almost_equal, assert_true

from max.gpu.host import DeviceContext

from numax import FloatLike, Plain
from numax.core.array import Static, transpose
from numax.linalg import inner, kron, matmul, matrix_power
from numax.linalg.array import inner as array_inner
from numax.linalg.array import kron as array_kron
from numax.linalg.array import matrix_power as array_matrix_power

comptime dtype = DType.float64
comptime P = Plain[DType.float64, 1]


def _a(ctx: DeviceContext) raises -> Static[dtype, 2, 3]:
    return Static[dtype, 2, 3](ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def _b(ctx: DeviceContext) raises -> Static[dtype, 4, 3]:
    return Static[dtype, 4, 3](
        ctx, [1.0, 0.0, -1.0, 2.0, 1.0, 0.0, 0.0, 3.0, 1.0, -2.0, 1.0, 4.0]
    )


def _square(ctx: DeviceContext) raises -> Static[dtype, 3, 3]:
    return Static[dtype, 3, 3](
        ctx, [2.0, -1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 2.0, -2.0]
    )


# --- inner ------------------------------------------------------------------


def test_inner_equals_matmul_against_an_explicit_transpose() raises:
    """The independent reference: `inner(a, b)` must equal `a @ b.T` with
    the transpose actually materialized. `inner` never builds one."""
    var ctx = DeviceContext(api="cpu")
    var a = _a(ctx)
    var b = _b(ctx)
    var got = inner[dtype, 2, 3, 4](a, b).to_host()

    var a2 = _a(ctx)
    var b2 = _b(ctx)
    var b_t = transpose(b2)
    var want = matmul[dtype, 2, 3, 4](a2, b_t).to_host()

    for i in range(8):
        assert_almost_equal(Float64(got[i]), Float64(want[i]), atol=1e-12)


def test_inner_is_the_gram_matrix_of_the_rows() raises:
    """`inner(a, a)[i, j]` is the dot product of rows `i` and `j`, so the
    result is symmetric and its diagonal is the squared row norms."""
    var ctx = DeviceContext(api="cpu")
    var a = _a(ctx)
    var a2 = _a(ctx)
    var g = inner[dtype, 2, 3, 2](a, a2).to_host()

    # Row 0 is (1,2,3): 1+4+9 = 14. Row 1 is (4,5,6): 16+25+36 = 77.
    assert_almost_equal(Float64(g[0]), 14.0, atol=1e-12)
    assert_almost_equal(Float64(g[3]), 77.0, atol=1e-12)
    # 1*4 + 2*5 + 3*6 = 32, and symmetric.
    assert_almost_equal(Float64(g[1]), 32.0, atol=1e-12)
    assert_almost_equal(Float64(g[2]), 32.0, atol=1e-12)


def test_array_inner_agrees_with_the_tensor_tier() raises:
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, 3, 3](
        ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    )
    var b = _square(ctx)
    var want = inner[dtype, 3, 3, 3](a, b).to_host()

    var aa = Array[P, 9](fill=P.constant(0.0))
    var values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    for i in range(9):
        aa[i] = P.constant(values[i])
    var bb = Array[P, 9](fill=P.constant(0.0))
    var bvalues = [2.0, -1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 2.0, -2.0]
    for i in range(9):
        bb[i] = P.constant(bvalues[i])

    var got = array_inner[P, 3](aa, bb)
    for i in range(9):
        assert_almost_equal(got[i].v, Float64(want[i]), atol=1e-12)


# --- kron -------------------------------------------------------------------


def test_kron_matches_a_hand_written_loop() raises:
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, 2, 2](ctx, [1.0, 2.0, 3.0, 4.0])
    var b = Static[dtype, 2, 3](ctx, [5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    var got = kron[dtype, 2, 2, 2, 3](a, b).to_host()

    var a_host = [1.0, 2.0, 3.0, 4.0]
    var b_host = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    for i in range(2):
        for j in range(2):
            for r in range(2):
                for c in range(3):
                    var want = a_host[i * 2 + j] * b_host[r * 3 + c]
                    var row = i * 2 + r
                    var col = j * 3 + c
                    assert_almost_equal(
                        Float64(got[row * 6 + col]), want, atol=1e-12
                    )


def test_kron_with_the_identity_tiles_the_other_operand() raises:
    """`kron(I(2), b)` is `b` twice down the diagonal and zero elsewhere --
    the property that makes the Kronecker product useful for block
    systems."""
    var ctx = DeviceContext(api="cpu")
    var eye2 = Static[dtype, 2, 2](ctx, [1.0, 0.0, 0.0, 1.0])
    var b = Static[dtype, 2, 2](ctx, [1.0, 2.0, 3.0, 4.0])
    var got = kron[dtype, 2, 2, 2, 2](eye2, b).to_host()

    assert_almost_equal(Float64(got[0]), 1.0, atol=1e-12)
    assert_almost_equal(Float64(got[1]), 2.0, atol=1e-12)
    assert_almost_equal(Float64(got[2]), 0.0, atol=1e-12)
    assert_almost_equal(Float64(got[10]), 1.0, atol=1e-12)
    assert_almost_equal(Float64(got[15]), 4.0, atol=1e-12)


def test_array_kron_agrees_with_the_tensor_tier() raises:
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, 2, 2](ctx, [1.0, 2.0, 3.0, 4.0])
    var b = Static[dtype, 2, 2](ctx, [5.0, 6.0, 7.0, 8.0])
    var want = kron[dtype, 2, 2, 2, 2](a, b).to_host()

    var aa = Array[P, 4](fill=P.constant(0.0))
    var bb = Array[P, 4](fill=P.constant(0.0))
    var av = [1.0, 2.0, 3.0, 4.0]
    var bv = [5.0, 6.0, 7.0, 8.0]
    for i in range(4):
        aa[i] = P.constant(av[i])
        bb[i] = P.constant(bv[i])

    var got = array_kron[P, 2, 2](aa, bb)
    for i in range(16):
        assert_almost_equal(got[i].v, Float64(want[i]), atol=1e-12)


# --- matrix_power -----------------------------------------------------------


def test_matrix_power_three_equals_two_matmuls() raises:
    var ctx = DeviceContext(api="cpu")
    var a = _square(ctx)
    var got = matrix_power[dtype, 3, 3](a).to_host()

    var b = _square(ctx)
    var c = _square(ctx)
    var squared = matmul[dtype, 3, 3, 3](b, c)
    var d = _square(ctx)
    var want = matmul[dtype, 3, 3, 3](squared, d).to_host()

    for i in range(9):
        assert_almost_equal(Float64(got[i]), Float64(want[i]), atol=1e-10)


def test_matrix_power_zero_is_the_identity() raises:
    var ctx = DeviceContext(api="cpu")
    var a = _square(ctx)
    var got = matrix_power[dtype, 3, 0](a).to_host()
    for i in range(3):
        for j in range(3):
            var want = 1.0 if i == j else 0.0
            assert_almost_equal(Float64(got[i * 3 + j]), want, atol=1e-14)


def test_matrix_power_one_is_the_matrix() raises:
    var ctx = DeviceContext(api="cpu")
    var a = _square(ctx)
    var got = matrix_power[dtype, 3, 1](a).to_host()
    var want = _square(ctx).to_host()
    for i in range(9):
        assert_almost_equal(Float64(got[i]), Float64(want[i]), atol=1e-14)


def test_an_even_power_exercises_the_squaring_path() raises:
    """`power = 4` is two squarings and no odd-bit multiply, which is the
    branch a purely odd power never reaches."""
    var ctx = DeviceContext(api="cpu")
    var a = _square(ctx)
    var got = matrix_power[dtype, 3, 4](a).to_host()

    var b = _square(ctx)
    var c = _square(ctx)
    var squared = matmul[dtype, 3, 3, 3](b, c)
    var d = _square(ctx)
    var e = _square(ctx)
    var squared_again = matmul[dtype, 3, 3, 3](d, e)
    var want = matmul[dtype, 3, 3, 3](squared, squared_again).to_host()

    for i in range(9):
        assert_almost_equal(Float64(got[i]), Float64(want[i]), atol=1e-10)


def test_array_matrix_power_agrees_with_the_tensor_tier() raises:
    var ctx = DeviceContext(api="cpu")
    var a = _square(ctx)
    var want = matrix_power[dtype, 3, 5](a).to_host()

    var aa = Array[P, 9](fill=P.constant(0.0))
    var av = [2.0, -1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 2.0, -2.0]
    for i in range(9):
        aa[i] = P.constant(av[i])

    var got = array_matrix_power[P, 3, 5](aa)
    for i in range(9):
        assert_almost_equal(got[i].v, Float64(want[i]), atol=1e-9)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
