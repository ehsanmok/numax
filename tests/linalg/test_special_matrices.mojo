"""Tests for `numax.linalg.special_matrices`.

Each constructor is checked against its defining index rule, and then
against something independent wherever one exists — a companion matrix's
eigenvalues must be the polynomial's roots, a convolution matrix times a
vector must equal `numax.signal.convolve` of the same pair, and a circulant
must be the Toeplitz matrix its first column implies. Those are the
assertions that would catch an off-by-one the index rule and the test would
otherwise share.
"""

from std.collections import Array
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax import Plain
from numax.core.array import Static, to_array
from numax.linalg import (
    block_diag,
    circulant,
    companion,
    convolution_matrix,
    hankel,
    hilbert,
    khatri_rao,
    matvec,
    toeplitz,
)
from numax.linalg.array import eigvals
from numax.signal.array import convolve

comptime dtype = DType.float64
comptime P = Plain[DType.float64, 1]


# --- toeplitz ---------------------------------------------------------------


def test_toeplitz_two_argument_form() raises:
    var ctx = DeviceContext(api="cpu")
    var c = Static[dtype, 3](ctx, [1.0, 2.0, 3.0])
    var r = Static[dtype, 4](ctx, [9.0, 4.0, 5.0, 6.0])
    var got = toeplitz[dtype, 3, 4](c, r).to_host()

    # r[0] is ignored: the corner is c[0].
    var want = [
        1.0,
        4.0,
        5.0,
        6.0,
        2.0,
        1.0,
        4.0,
        5.0,
        3.0,
        2.0,
        1.0,
        4.0,
    ]
    for i in range(12):
        assert_almost_equal(Float64(got[i]), want[i], atol=0.0)


def test_toeplitz_symmetric_form() raises:
    var ctx = DeviceContext(api="cpu")
    var c = Static[dtype, 3](ctx, [1.0, 2.0, 3.0])
    var got = toeplitz[dtype, 3](c).to_host()
    var want = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]
    for i in range(9):
        assert_almost_equal(Float64(got[i]), want[i], atol=0.0)


def test_the_symmetric_form_equals_the_two_argument_one() raises:
    var ctx = DeviceContext(api="cpu")
    var c = Static[dtype, 3](ctx, [1.0, 2.0, 3.0])
    var one_arg = toeplitz[dtype, 3](c).to_host()
    var c2 = Static[dtype, 3](ctx, [1.0, 2.0, 3.0])
    var r2 = Static[dtype, 3](ctx, [1.0, 2.0, 3.0])
    var two_arg = toeplitz[dtype, 3, 3](c2, r2).to_host()
    for i in range(9):
        assert_almost_equal(Float64(one_arg[i]), Float64(two_arg[i]), atol=0.0)


# --- hankel -----------------------------------------------------------------


def test_hankel_is_constant_along_antidiagonals() raises:
    var ctx = DeviceContext(api="cpu")
    var c = Static[dtype, 3](ctx, [1.0, 2.0, 3.0])
    var r = Static[dtype, 3](ctx, [3.0, 4.0, 5.0])
    var got = toeplitz[dtype, 3, 3](c, r).to_host()
    var hc = Static[dtype, 3](ctx, [1.0, 2.0, 3.0])
    var hr = Static[dtype, 3](ctx, [3.0, 4.0, 5.0])
    var h = hankel[dtype, 3, 3](hc, hr).to_host()
    var want = [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]
    for i in range(9):
        assert_almost_equal(Float64(h[i]), want[i], atol=0.0)
    # And it is genuinely a different matrix from the Toeplitz one.
    assert_true(Float64(h[1]) != Float64(got[1]))


# --- circulant --------------------------------------------------------------


def test_circulant_rotates_each_column() raises:
    var ctx = DeviceContext(api="cpu")
    var c = Static[dtype, 3](ctx, [1.0, 2.0, 3.0])
    var got = circulant[dtype, 3](c).to_host()
    var want = [1.0, 3.0, 2.0, 2.0, 1.0, 3.0, 3.0, 2.0, 1.0]
    for i in range(9):
        assert_almost_equal(Float64(got[i]), want[i], atol=0.0)


def test_a_symmetric_circulant_is_its_own_toeplitz() raises:
    """When `c` reads the same forwards and backwards after the first
    entry, the circulant and the symmetric Toeplitz coincide -- an
    independent check on both index rules at once."""
    var ctx = DeviceContext(api="cpu")
    var c = Static[dtype, 4](ctx, [1.0, 5.0, 9.0, 5.0])
    var circ = circulant[dtype, 4](c).to_host()
    var c2 = Static[dtype, 4](ctx, [1.0, 5.0, 9.0, 5.0])
    var toep = toeplitz[dtype, 4](c2).to_host()
    for i in range(16):
        assert_almost_equal(Float64(circ[i]), Float64(toep[i]), atol=0.0)


# --- companion --------------------------------------------------------------


def test_companion_eigenvalues_are_the_polynomial_roots() raises:
    """The independent check that matters: `x^3 - 6x^2 + 11x - 6` has roots
    1, 2 and 3, so the companion matrix's eigenvalues must be those."""
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, 4](ctx, [1.0, -6.0, 11.0, -6.0])
    var c = companion[dtype, 4](a)

    var as_array = to_array[P](c)
    var roots = eigvals[P, 3](as_array)

    var found_one = False
    var found_two = False
    var found_three = False
    for i in range(3):
        var re = roots[i].re.v
        assert_true(abs(roots[i].im.v) < 1e-6, "the roots are real")
        if abs(re - 1.0) < 1e-6:
            found_one = True
        if abs(re - 2.0) < 1e-6:
            found_two = True
        if abs(re - 3.0) < 1e-6:
            found_three = True
    assert_true(found_one and found_two and found_three)


def test_companion_has_the_documented_shape() raises:
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, 4](ctx, [2.0, -4.0, 6.0, -8.0])
    var got = companion[dtype, 4](a).to_host()
    # First row is -a[1:]/a[0]; ones on the first subdiagonal.
    var want = [2.0, -3.0, 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    for i in range(9):
        assert_almost_equal(Float64(got[i]), want[i], atol=1e-14)


# --- hilbert ----------------------------------------------------------------


def test_hilbert_entries() raises:
    var ctx = DeviceContext(api="cpu")
    var got = hilbert[3, dtype](ctx).to_host()
    for i in range(3):
        for j in range(3):
            var want = 1.0 / Float64(i + j + 1)
            assert_almost_equal(Float64(got[i * 3 + j]), want, atol=1e-15)


def test_hilbert_is_symmetric() raises:
    var ctx = DeviceContext(api="cpu")
    var got = hilbert[5, dtype](ctx).to_host()
    for i in range(5):
        for j in range(5):
            assert_almost_equal(
                Float64(got[i * 5 + j]), Float64(got[j * 5 + i]), atol=0.0
            )


# --- block_diag -------------------------------------------------------------


def test_block_diag_places_the_blocks_and_zeros_the_rest() raises:
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, 1, 2](ctx, [1.0, 2.0])
    var b = Static[dtype, 2, 1](ctx, [3.0, 4.0])
    var got = block_diag[dtype, 1, 2, 2, 1](a, b).to_host()
    # 3x3: [[1,2,0],[0,0,3],[0,0,4]]
    var want = [1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0]
    for i in range(9):
        assert_almost_equal(Float64(got[i]), want[i], atol=0.0)


# --- khatri_rao -------------------------------------------------------------


def test_khatri_rao_is_kron_column_by_column() raises:
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, 2, 2](ctx, [1.0, 2.0, 3.0, 4.0])
    var b = Static[dtype, 2, 2](ctx, [5.0, 6.0, 7.0, 8.0])
    var got = khatri_rao[dtype, 2, 2, 2](a, b).to_host()

    var a_host = [1.0, 2.0, 3.0, 4.0]
    var b_host = [5.0, 6.0, 7.0, 8.0]
    for col in range(2):
        for i in range(2):
            for r in range(2):
                var want = a_host[i * 2 + col] * b_host[r * 2 + col]
                assert_almost_equal(
                    Float64(got[(i * 2 + r) * 2 + col]), want, atol=0.0
                )


# --- convolution_matrix -----------------------------------------------------


def test_convolution_matrix_times_a_vector_is_a_convolution() raises:
    """The claim the name makes: `C @ v == convolve(a, v)`. Checked against
    `numax.signal.convolve`, which shares no code with this."""
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, 3](ctx, [1.0, -2.0, 3.0])
    var c = convolution_matrix[dtype, 3, 4](a)
    var v = Static[dtype, 4](ctx, [2.0, 0.0, -1.0, 5.0])
    var got = matvec[dtype, 6, 4](c, v).to_host()

    var a_arr = Array[P, 3](fill=P.constant(0.0))
    var a_values = [1.0, -2.0, 3.0]
    for i in range(3):
        a_arr[i] = P.constant(a_values[i])
    var v_arr = Array[P, 4](fill=P.constant(0.0))
    var v_values = [2.0, 0.0, -1.0, 5.0]
    for i in range(4):
        v_arr[i] = P.constant(v_values[i])
    var want = convolve[P, 3, 4](a_arr, v_arr)

    for i in range(6):
        assert_almost_equal(Float64(got[i]), want[i].v, atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
