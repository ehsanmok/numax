"""Tests for `to_array`/`to_tensor`, the seam between numax's two arrays.

`Tensor` carries shape and device; `Array[T, n]` carries the `FloatLike`
conformer. Nothing composed across them before these two existed, so what
is checked here is the composition itself: a tensor factored by
`numax.linalg` and lowered back, and a lift at `Dual` that still
differentiates.
"""

from std.collections import Array
from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax import Dual, Plain
from numax.core.array import Tensor, eye, to_array, to_tensor
from numax.linalg import cholesky, det

comptime dtype = DType.float64
comptime P = Plain[dtype, 1]


def _matrix() raises -> Tensor[dtype, 2, 2]:
    """`[[4, 2], [2, 3]]` -- symmetric positive definite, so it has a
    Cholesky factor and a determinant of 8."""
    var ctx = DeviceContext(api="cpu")
    var values = List[Scalar[dtype]](capacity=4)
    for v in [4.0, 2.0, 2.0, 3.0]:
        values.append(Scalar[dtype](v))
    return Tensor[dtype, 2, 2](ctx, values^)


def test_rank_1_round_trips() raises:
    var ctx = DeviceContext(api="cpu")
    var values = List[Scalar[dtype]](capacity=3)
    for v in [1.5, -2.0, 0.25]:
        values.append(Scalar[dtype](v))
    var xs = Tensor[dtype, 3](ctx, values^)

    var lifted = to_array[P](xs)
    var back = to_tensor(lifted, ctx)
    for i in range(3):
        assert_almost_equal(back[i], xs[i])


def test_rank_2_round_trips_row_major() raises:
    var ctx = DeviceContext(api="cpu")
    var a = _matrix()
    var back = to_tensor(to_array[P](a), ctx)
    for i in range(4):
        assert_almost_equal(back[i], a[i])


def test_a_tensor_can_be_factored_and_lowered_back() raises:
    # The composition the seam exists for: a tensor goes into numax.linalg
    # and the factor comes back as a tensor.
    var ctx = DeviceContext(api="cpu")
    var a = _matrix()
    var lower = cholesky[P, 2](to_array[P](a))
    var factor = to_tensor(lower, ctx)

    # L is lower triangular with L[0,0] = 2, and L @ L.T == A.
    assert_almost_equal(factor[0], Scalar[dtype](2.0))
    assert_almost_equal(factor[1], Scalar[dtype](0.0))
    assert_almost_equal(factor[2], Scalar[dtype](1.0))
    assert_almost_equal(
        factor[3] * factor[3] + factor[2] * factor[2], Scalar[dtype](3.0)
    )


def test_the_identity_lifts_to_a_determinant_of_one() raises:
    var ctx = DeviceContext(api="cpu")
    var i3 = eye[dtype, 3](ctx)
    assert_almost_equal(det[P, 3](to_array[P](i3)).v, Scalar[dtype](1.0))


def test_lifting_at_dual_still_differentiates() raises:
    # `to_array` lifts through `T.constant`, so a Dual arrives with a zero
    # derivative and the caller seeds the one it wants. Seeding A[0,0] and
    # differentiating det gives the cofactor, which is 3 for this matrix.
    var a = _matrix()
    var lifted = to_array[Dual[P]](a)
    lifted[0] = Dual[P](P.constant(4.0), P.one())
    assert_almost_equal(det[Dual[P], 2](lifted).deriv.v, Scalar[dtype](3.0))


def test_lifted_values_match_the_source_elements() raises:
    var a = _matrix()
    var lifted = to_array[P](a)
    for i in range(4):
        assert_almost_equal(lifted[i].v, a[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
