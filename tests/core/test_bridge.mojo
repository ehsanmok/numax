"""Tests for the seams out of `Tensor`: `to_array`/`to_tensor` between
numax's two arrays, `view`/`from_view` between owned storage and a borrowed
one, and `dynamic`/`static_view` between a shape the compiler can see and
one it cannot.

`Tensor` carries shape and device; `Array[T, n]` carries the `FloatLike`
conformer. Nothing composed across them before these two existed, so what
is checked here is the composition itself: a tensor factored by
`numax.linalg` and lowered back, and a lift at `Dual` that still
differentiates.
"""

from std.collections import Array
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from layout.tile_layout import row_major
from linalg.matmul import matmul as max_matmul
from nn.reshape import reshape as nn_reshape
from std.utils import IndexList

from numax import Dual, Plain
from numax.core.array import (
    Dynamic,
    Shaped,
    Tensor,
    eye,
    to_array,
    to_tensor,
    zeros,
    zeros_dyn,
)
from numax.linalg import cholesky, det
from numax.core.tensor import map_strided
from numax.linalg import matmul as numax_matmul

comptime dtype = DType.float64
comptime P = Plain[dtype]


def _matrix() raises -> Shaped[dtype, 2, 2]:
    """`[[4, 2], [2, 3]]` -- symmetric positive definite, so it has a
    Cholesky factor and a determinant of 8."""
    var ctx = DeviceContext(api="cpu")
    var values = List[Scalar[dtype]](capacity=4)
    for v in [4.0, 2.0, 2.0, 3.0]:
        values.append(Scalar[dtype](v))
    return Shaped[dtype, 2, 2](ctx, values^)


def test_rank_1_round_trips() raises:
    var ctx = DeviceContext(api="cpu")
    var values = List[Scalar[dtype]](capacity=3)
    for v in [1.5, -2.0, 0.25]:
        values.append(Scalar[dtype](v))
    var xs = Shaped[dtype, 3](ctx, values^)

    var lifted = to_array[P](xs)
    var back = to_tensor[dtype, 3](lifted, ctx)
    for i in range(3):
        assert_almost_equal(back[i], xs[i])


def test_rank_2_round_trips_row_major() raises:
    var ctx = DeviceContext(api="cpu")
    var a = _matrix()
    var back = to_tensor[dtype, 2, 2](to_array[P](a), ctx)
    for i in range(4):
        assert_almost_equal(back[i], a[i])


def test_a_tensor_can_be_factored_and_lowered_back() raises:
    # The composition the seam exists for: a tensor goes into numax.linalg
    # and the factor comes back as a tensor.
    var ctx = DeviceContext(api="cpu")
    var a = _matrix()
    var lower = cholesky[P, 2](to_array[P](a))
    var factor = to_tensor[dtype, 2, 2](lower, ctx)

    # L is lower triangular with L[0,0] = 2, and L @ L.T == A.
    assert_almost_equal(factor[0], Scalar[dtype](2.0))
    assert_almost_equal(factor[1], Scalar[dtype](0.0))
    assert_almost_equal(factor[2], Scalar[dtype](1.0))
    assert_almost_equal(
        factor[3] * factor[3] + factor[2] * factor[2], Scalar[dtype](3.0)
    )


def test_the_identity_lifts_to_a_determinant_of_one() raises:
    var ctx = DeviceContext(api="cpu")
    var i3 = eye[3](ctx)
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


def test_from_view_round_trips_the_elements() raises:
    var a = _matrix()
    var back = Shaped[dtype, 2, 2].from_view(a.view(), a.context())
    for i in range(4):
        assert_almost_equal(back[i], a[i])


def test_from_view_copies_rather_than_aliases() raises:
    var a = _matrix()
    var back = Shaped[dtype, 2, 2].from_view(a.view(), a.context())
    var v = a.view()
    v[0, 0] = 99.0
    assert_almost_equal(a[0], Scalar[dtype](99.0))
    assert_almost_equal(back[0], Scalar[dtype](4.0))


def test_widening_a_shape_and_naming_it_again_is_the_identity() raises:
    var a = _matrix()
    var elements = a.to_host()

    var widened = a^.dynamic()
    assert_equal(widened.dim_at(0), 2)
    assert_equal(widened.dim_at(1), 2)

    var narrowed = widened^.static_view[2, 2]()
    assert_equal(narrowed.num_elements, 4)
    for i in range(4):
        assert_equal(narrowed[i], elements[i])


def test_naming_the_wrong_shape_raises_rather_than_reading_past_the_end() raises:
    var ctx = DeviceContext(api="cpu")
    var a = zeros_dyn[dtype, 2](2, 3, ctx=ctx)
    var raised = False
    try:
        _ = a^.static_view[3, 2]()
    except:
        raised = True
    assert_true(raised)


def test_a_named_run_time_shape_matches_the_static_tensor_element_for_element() raises:
    var ctx = DeviceContext(api="cpu")
    var built_dynamic = zeros_dyn[dtype, 2](2, 3, ctx=ctx)
    var built_static = zeros[dtype, 2, 3](ctx)
    for i in range(6):
        built_dynamic[i] = Scalar[dtype](i) * 0.5
        built_static[i] = Scalar[dtype](i) * 0.5

    var named = built_dynamic^.static_view[2, 3]()
    var from_named = named.view()
    var from_static = built_static.view()
    for r in range(2):
        for c in range(3):
            assert_equal(from_named[r, c], from_static[r, c])


def test_a_rank_3_tensor_lifts_and_lowers() raises:
    # The case the old rank-1-and-square pair of overloads could not
    # express at all: an Array is flat, so only the element count matters.
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, 2, 3, 2](ctx)
    for i in range(12):
        a[i] = Scalar[dtype](i) - 3.0

    var back = to_tensor[dtype, 2, 3, 2](to_array[P](a), ctx)
    for i in range(12):
        assert_almost_equal(back[i], a[i])


def test_one_matrix_multiplied_through_both_layers_agrees() raises:
    # MAX's matmul takes a TileTensor of raw dtype; numax's takes an Array
    # of a FloatLike. The same matrix reaches both through this seam, and
    # the answers have to be the same one.
    var ctx = DeviceContext(api="cpu")
    var a = _matrix()
    var b = _matrix()
    var through_max = zeros[dtype, 2, 2](ctx)
    max_matmul[target="cpu"](through_max.view(), a.view(), b.view())
    ctx.synchronize()

    var through_numax = to_tensor[dtype, 2, 2](
        numax_matmul[P, 2](to_array[P](a), to_array[P](b)), ctx
    )
    for i in range(4):
        assert_almost_equal(through_numax[i], through_max[i])


def _double[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return x + x


def test_a_max_kernels_output_view_feeds_a_numax_walk() raises:
    # `nn.reshape` hands back a view whose extents are run-time values and
    # whose strides are not the row-major pattern a static walk requires.
    # Nothing in numax could consume that before the strided walker; now it
    # goes straight in, and the result lands in an ordinary numax tensor.
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, 2, 3](ctx)
    for i in range(6):
        a[i] = Scalar[dtype](i)

    var reshaped = nn_reshape[output_rank=2](a.view(), IndexList[2](3, 2))
    assert_equal(type_of(reshaped).all_dims_known, False)

    var out = zeros_dyn[dtype, 2](3, 2, ctx=ctx)
    map_strided[step=_double](reshaped, out.view())
    for i in range(6):
        assert_almost_equal(out[i], Scalar[dtype](2 * i))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
