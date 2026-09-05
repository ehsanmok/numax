"""Tests for a `Tensor` whose extents arrive at run time.

`Shaped[dtype, *dims]` and `Dynamic[dtype, rank]` are the same struct at two
layout types, so what is checked here is that the second one is a real
tensor and not a second-class one: it allocates from values only known at
run time, reports its own shape, carries elements through the same host
path, and feeds the same transforms as a static tensor -- with the two
agreeing element for element where they describe the same shape.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax.core.array import (
    Dynamic,
    Shaped,
    empty_dyn,
    full_dyn,
    ones_dyn,
    zeros_dyn,
    zeros_like,
)
from numax.core.elementwise import sqrt
from numax.stats import sum as tensor_sum

comptime dtype = DType.float64


def _rows() -> Int:
    """A row count the compiler cannot see, so the shape is genuinely a
    run-time one rather than a literal that happens to be spelled late."""
    var n = 0
    for i in range(5):
        n += i % 2
    return n + 2  # 4


def test_a_dynamic_tensor_takes_its_shape_at_run_time() raises:
    var ctx = DeviceContext(api="cpu")
    var a = zeros_dyn[dtype, 2](_rows(), 3, ctx=ctx)
    assert_equal(a.size(), 12)
    assert_equal(a.dim[0](), 4)
    assert_equal(a.dim[1](), 3)
    assert_equal(type_of(a).rank, 2)


def test_staticness_is_a_property_of_the_layout_type() raises:
    assert_true(Shaped[dtype, 4, 3].LayoutType.all_dims_known)
    assert_true(not Dynamic[dtype, 2].LayoutType.all_dims_known)


def test_elements_round_trip_through_the_host() raises:
    var ctx = DeviceContext(api="cpu")
    var a = zeros_dyn[dtype, 1](6, ctx=ctx)
    for i in range(6):
        a[i] = Scalar[dtype](i * i)
    var host = a.to_host()
    for i in range(6):
        assert_almost_equal(host[i], Scalar[dtype](i * i))


def test_the_fill_factories_agree_with_their_static_siblings() raises:
    var ctx = DeviceContext(api="cpu")
    var ones = ones_dyn[dtype, 2](2, 3, ctx=ctx)
    var sevens = full_dyn[dtype, 2](7.0, 2, 3, ctx=ctx)
    var blank = empty_dyn[dtype, 2](2, 3, ctx=ctx)
    for i in range(6):
        assert_almost_equal(ones[i], Scalar[dtype](1.0))
        assert_almost_equal(sevens[i], Scalar[dtype](7.0))
        assert_almost_equal(blank[i], Scalar[dtype](0.0))


def test_a_transform_gives_the_same_answer_at_either_shape() raises:
    # The claim the layout-generic migration rests on: a kernel written
    # once runs over both, and the two do not disagree.
    var ctx = DeviceContext(api="cpu")
    var values = List[Scalar[dtype]](capacity=6)
    for i in range(6):
        values.append(Scalar[dtype](i + 1))

    var static_a = Shaped[dtype, 2, 3](ctx, values.copy())
    var dynamic_a = zeros_dyn[dtype, 2](2, 3, ctx=ctx)
    dynamic_a.copy_from_host(values)

    var static_r = sqrt(static_a)
    var dynamic_r = sqrt(dynamic_a)
    for i in range(6):
        assert_almost_equal(dynamic_r[i], static_r[i])
    assert_almost_equal(tensor_sum(dynamic_a), tensor_sum(static_a))


def test_zeros_like_keeps_a_run_time_shape() raises:
    var ctx = DeviceContext(api="cpu")
    var a = ones_dyn[dtype, 2](3, 5, ctx=ctx)
    var b = zeros_like(a)
    assert_equal(b.size(), 15)
    assert_equal(b.dim[1](), 5)
    assert_almost_equal(b[0], Scalar[dtype](0.0))


def test_naming_a_compile_time_shape_is_required_to_reach_an_array() raises:
    # `Array`'s length is a parameter, so a run-time-shaped tensor has
    # nothing to lift into. The refusal is the contract, so it is pinned.
    var raised = False
    try:
        _ = Dynamic[dtype, 1]._static_layout()
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
