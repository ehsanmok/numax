"""Tests for `expm` at both tiers and the SPD `sqrtm`.

`expm` is checked against closed forms rather than against another
implementation of the same algorithm, because there is no independent one to
hand:

- a diagonal matrix, whose exponential is the elementwise `exp` of the
  diagonal and nothing else;
- a nilpotent matrix, where the power series terminates and the answer is a
  finite sum written out by hand;
- the 2x2 rotation generator, whose exponential is exactly the rotation
  matrix `[[cos t, -sin t], [sin t, cos t]]`;
- the group identity `expm(A) @ expm(-A) == I`, which no wrong answer
  satisfies by accident.

And the claim that makes the `Array` overload worth having separately: it is
tier 1, so it differentiates. `test_expm_differentiates_at_dual` checks the
derivative against a central difference.
"""

from std.collections import Array
from std.math import cos as cos_f64, exp as exp_f64, sin as sin_f64
from std.testing import TestSuite, assert_almost_equal, assert_true

from max.gpu.host import DeviceContext

from numax import Dual, FloatLike, Plain
from numax.core.array import Static
from numax.linalg import expm, matmul
from numax.linalg.array import expm as array_expm
from numax.linalg.array import matmul as array_matmul
from numax.linalg.array import sqrtm

comptime dtype = DType.float64
comptime P = Plain[DType.float64, 1]
comptime D = Dual[P]


def _as_array[n: Int](values: Array[Float64, n]) -> Array[P, n]:
    var out = Array[P, n](fill=P.constant(0.0))
    for i in range(n):
        out[i] = P.constant(values[i])
    return out^


# --- expm over Tensor -------------------------------------------------------


def test_expm_of_a_diagonal_matrix() raises:
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, 3, 3](
        ctx, [0.5, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0]
    )
    var got = expm[dtype, 3](a).to_host()
    var want = [
        exp_f64(0.5),
        0.0,
        0.0,
        0.0,
        exp_f64(-1.0),
        0.0,
        0.0,
        0.0,
        exp_f64(2.0),
    ]
    for i in range(9):
        assert_almost_equal(Float64(got[i]), want[i], atol=1e-13)


def test_expm_of_zero_is_the_identity() raises:
    var ctx = DeviceContext(api="cpu")
    var entries = List[Scalar[dtype]](length=9, fill=0)
    var a = Static[dtype, 3, 3](ctx, entries^)
    var got = expm[dtype, 3](a).to_host()
    for i in range(3):
        for j in range(3):
            var want = 1.0 if i == j else 0.0
            assert_almost_equal(Float64(got[i * 3 + j]), want, atol=1e-14)


def test_expm_of_a_nilpotent_matrix_is_a_finite_sum() raises:
    """`N` strictly upper triangular with `N^3 == 0`, so
    `expm(N) == I + N + N^2/2` exactly."""
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, 3, 3](
        ctx, [0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
    )
    var got = expm[dtype, 3](a).to_host()
    # N^2 has a single nonzero at (0,2): 1*3 = 3, so N^2/2 gives 1.5.
    var want = [
        1.0,
        1.0,
        2.0 + 1.5,
        0.0,
        1.0,
        3.0,
        0.0,
        0.0,
        1.0,
    ]
    for i in range(9):
        assert_almost_equal(Float64(got[i]), want[i], atol=1e-13)


def test_expm_of_the_rotation_generator_is_a_rotation() raises:
    """`[[0, -t], [t, 0]]` exponentiates to the rotation by `t`, exactly."""
    var ctx = DeviceContext(api="cpu")
    var t = 0.7
    var a = Static[dtype, 2, 2](ctx, [0.0, -t, t, 0.0])
    var got = expm[dtype, 2](a).to_host()
    assert_almost_equal(Float64(got[0]), cos_f64(t), atol=1e-14)
    assert_almost_equal(Float64(got[1]), -sin_f64(t), atol=1e-14)
    assert_almost_equal(Float64(got[2]), sin_f64(t), atol=1e-14)
    assert_almost_equal(Float64(got[3]), cos_f64(t), atol=1e-14)


def test_expm_a_times_expm_minus_a_is_the_identity() raises:
    """The group identity. A wrong answer does not satisfy it by
    accident, and it exercises the scaling path on a matrix whose norm
    forces several squarings."""
    var ctx = DeviceContext(api="cpu")
    var values = [3.0, -7.0, 2.0, 1.0, 4.0, -5.0, 6.0, 0.5, -2.0]
    var entries = List[Scalar[dtype]](capacity=9)
    for i in range(9):
        entries.append(Scalar[dtype](values[i]))
    var a = Static[dtype, 3, 3](ctx, entries^)
    var forward = expm[dtype, 3](a)

    var negated = List[Scalar[dtype]](capacity=9)
    for i in range(9):
        negated.append(Scalar[dtype](-values[i]))
    var b = Static[dtype, 3, 3](ctx, negated^)
    var backward = expm[dtype, 3](b)

    var product = matmul[dtype, 3, 3, 3](forward, backward).to_host()
    for i in range(3):
        for j in range(3):
            var want = 1.0 if i == j else 0.0
            assert_almost_equal(Float64(product[i * 3 + j]), want, atol=1e-9)


def test_expm_scales_for_a_large_norm() raises:
    """`||a||_1` well past the Pade threshold, so several squarings run.
    Checked against the diagonal closed form, which stays exact."""
    var ctx = DeviceContext(api="cpu")
    var a = Static[dtype, 2, 2](ctx, [20.0, 0.0, 0.0, -15.0])
    var got = expm[dtype, 2](a).to_host()
    assert_almost_equal(Float64(got[0]) / exp_f64(20.0), 1.0, atol=1e-11)
    assert_almost_equal(Float64(got[3]), exp_f64(-15.0), atol=1e-13)


# --- expm over Array --------------------------------------------------------


def test_array_expm_agrees_with_the_tensor_tier() raises:
    var ctx = DeviceContext(api="cpu")
    var values = [0.3, -0.7, 0.2, 0.1, 0.4, -0.5, 0.6, 0.05, -0.2]
    var entries = List[Scalar[dtype]](capacity=9)
    for i in range(9):
        entries.append(Scalar[dtype](values[i]))
    var a = Static[dtype, 3, 3](ctx, entries^)
    var want = expm[dtype, 3](a).to_host()

    var aa = Array[Float64, 9](fill=0.0)
    for i in range(9):
        aa[i] = values[i]
    var got = array_expm[P, 3](_as_array[9](aa))

    for i in range(9):
        assert_almost_equal(got[i].v, Float64(want[i]), atol=1e-10)


def test_array_expm_of_the_rotation_generator() raises:
    var t = 0.7
    var aa = Array[Float64, 4](fill=0.0)
    aa[0] = 0.0
    aa[1] = -t
    aa[2] = t
    aa[3] = 0.0
    var got = array_expm[P, 2](_as_array[4](aa))
    assert_almost_equal(got[0].v, cos_f64(t), atol=1e-13)
    assert_almost_equal(got[1].v, -sin_f64(t), atol=1e-13)


def test_expm_differentiates_at_dual() raises:
    """The reason the `Array` overload takes a fixed squaring count. `expm`
    is tier 1 there, so it runs at `Dual` and produces the derivative of a
    matrix exponential -- with no adjoint rule written anywhere.

    Differentiating `expm(t * N)` with respect to `t` at `t = 1`, for a
    nilpotent `N`, has the closed form `N + t N^2` -- checked here against
    a central difference as well, so a wrong closed form cannot hide.
    """
    var base = Array[Float64, 4](fill=0.0)
    base[0] = 0.0
    base[1] = 1.0
    base[2] = 0.0
    base[3] = 0.0

    # Seed `t = 1` with derivative 1, and scale the matrix by it.
    var seeded = Array[D, 4](fill=D(P.constant(0.0), P.constant(0.0)))
    var t = D(P.constant(1.0), P.one())
    for i in range(4):
        seeded[i] = t * D(P.constant(base[i]), P.constant(0.0))
    var differentiated = array_expm[D, 2](seeded^)

    # For this `N`, `expm(t N) = I + t N`, so d/dt at the (0,1) entry is 1.
    assert_almost_equal(differentiated[1].value.v, 1.0, atol=1e-12)
    assert_almost_equal(differentiated[1].deriv.v, 1.0, atol=1e-9)
    assert_almost_equal(differentiated[0].deriv.v, 0.0, atol=1e-9)

    # And against a central difference on the plain overload.
    var h = 1e-5
    var plus = Array[Float64, 4](fill=0.0)
    var minus = Array[Float64, 4](fill=0.0)
    for i in range(4):
        plus[i] = (1.0 + h) * base[i]
        minus[i] = (1.0 - h) * base[i]
    var up = array_expm[P, 2](_as_array[4](plus))
    var down = array_expm[P, 2](_as_array[4](minus))
    var difference = (up[1].v - down[1].v) / (2 * h)
    assert_almost_equal(differentiated[1].deriv.v, difference, atol=1e-6)


# --- sqrtm ------------------------------------------------------------------


def test_sqrtm_squares_back_to_the_matrix() raises:
    """The defining property, checked directly rather than against a
    reference factorization."""
    var values = Array[Float64, 9](fill=0.0)
    var entries = [4.0, 1.0, 0.0, 1.0, 5.0, 1.0, 0.0, 1.0, 3.0]
    for i in range(9):
        values[i] = entries[i]
    var root = sqrtm[P, 3](_as_array[9](values))
    var squared = array_matmul[P, 3](root, root)
    for i in range(9):
        assert_almost_equal(squared[i].v, entries[i], atol=1e-8)


def test_sqrtm_of_a_diagonal_matrix() raises:
    var values = Array[Float64, 4](fill=0.0)
    values[0] = 9.0
    values[1] = 0.0
    values[2] = 0.0
    values[3] = 16.0
    var root = sqrtm[P, 2](_as_array[4](values))
    assert_almost_equal(root[0].v, 3.0, atol=1e-9)
    assert_almost_equal(root[3].v, 4.0, atol=1e-9)
    assert_almost_equal(root[1].v, 0.0, atol=1e-9)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
