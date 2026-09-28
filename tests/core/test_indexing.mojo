"""Indexing and slicing a `Tensor`: `a[i:j]` and `a[i:j, k:l]` as views,
`block[m, n](r, c)`, rank-3 scalar access, and bounds checks.

A slice is a `TensorView` over the tensor's own storage, so the tests
check it reads the right elements, that writing through it lands in the
tensor, and that the NumPy rules for omitted and negative ends hold;
every scalar index outside its axis raises rather than reading past the
buffer.
"""

from std.testing import TestSuite, assert_equal, assert_raises

from max.gpu.host import DeviceContext

from numax.core.ops import add
from numax.linalg import cholesky
from numax.core.tensor import Static, copy, slice

comptime f32 = DType.float32


def _grid() raises -> Static[f32, 4, 5]:
    var values = List[Scalar[f32]](capacity=20)
    for i in range(20):
        values.append(Float32(i))
    return Static[f32, 4, 5](values^, DeviceContext(api="cpu"))


def test_row_slices_are_views() raises:
    var a = _grid()
    var v = a[1:3]
    assert_equal(v.dim_at(0), 2)
    assert_equal(v.dim_at(1), 5)
    var got = v.to_host()
    for i in range(10):
        assert_equal(got[i], Float32(5 + i))
    var tail = a[-1:]
    assert_equal(tail.dim_at(0), 1)
    assert_equal(tail.to_host()[0], 15.0)
    var whole = a[:]
    assert_equal(whole.dim_at(0), 4)
    var summed = add(v, v).to_host()
    assert_equal(summed[0], 10.0)


def test_slicing_agrees_with_the_slice_function() raises:
    """`a[1:3]` and `a[1:3, 2:4]` are views of what the copying `slice`
    returns."""
    var a = _grid()
    var rows = a[1:3]
    var got = rows.to_host()
    var want = slice(a, [1, 0], [3, 5]).to_host()
    for i in range(len(want)):
        assert_equal(got[i], want[i])
    var box = a[1:3, 2:4]
    var bg = box.to_host()
    var bw = slice(a, [1, 2], [3, 4]).to_host()
    for i in range(len(bw)):
        assert_equal(bg[i], bw[i])


def test_writes_through_a_view_land_in_the_tensor() raises:
    var a = _grid()
    var v = a[2:3]
    var ones = List[Scalar[f32]](length=5, fill=1.0)
    v.copy_from_host(ones)
    assert_equal(a[2, 0], 1.0)
    assert_equal(a[2, 4], 1.0)
    assert_equal(a[1, 4], 9.0)


def test_boxes_are_strided_views() raises:
    var a = _grid()
    var box = a[1:3, 2:4]
    var got = box.to_host()
    assert_equal(len(got), 4)
    assert_equal(got[0], 7.0)
    assert_equal(got[1], 8.0)
    assert_equal(got[2], 12.0)
    assert_equal(got[3], 13.0)
    var owned = copy(box).to_host()
    for i in range(4):
        assert_equal(owned[i], got[i])
    with assert_raises(contains="strided"):
        _ = add(box, box)


def test_block_is_an_element_offset_view() raises:
    var a = _grid()
    var b = a.block[2, 2](1, 3)
    var got = b.to_host()
    assert_equal(got[0], 8.0)
    assert_equal(got[1], 9.0)
    assert_equal(got[2], 13.0)
    assert_equal(got[3], 14.0)
    with assert_raises(contains="does not fit"):
        _ = a.block[2, 2](3, 3)


def test_a_block_is_statically_shaped_for_factorizations() raises:
    """`cholesky` wants a compile-time shape; `block` gives one over the
    parent's storage, and the factor matches an owned copy's."""
    var big = Static[f32, 3, 3](
        [4.0, 2.0, 9.0, 2.0, 3.0, 9.0, 9.0, 9.0, 9.0], DeviceContext(api="cpu")
    )
    var own = Static[f32, 2, 2]([4.0, 2.0, 2.0, 3.0], DeviceContext(api="cpu"))
    var got = cholesky(big.block[2, 2](0, 0)).to_host()
    var want = cholesky(own).to_host()
    for i in range(4):
        assert_equal(got[i], want[i])


def test_slices_reject_steps_and_bad_bounds() raises:
    var a = _grid()
    with assert_raises(contains="step"):
        _ = a[0:4:2]


def test_scalar_indices_are_bounds_checked() raises:
    var a = _grid()
    with assert_raises(contains="out of range"):
        _ = a[20]
    with assert_raises(contains="axis 1"):
        _ = a[0, 5]
    with assert_raises(contains="axis 0"):
        a[4, 0] = 1.0


def test_rank_three_indexing() raises:
    var values = List[Scalar[f32]](capacity=24)
    for i in range(24):
        values.append(Float32(i))
    var t = Static[f32, 2, 3, 4](values^, DeviceContext(api="cpu"))
    assert_equal(t[1, 2, 3], 23.0)
    assert_equal(t[0, 1, 2], 6.0)
    t[1, 0, 0] = -1.0
    assert_equal(t[12], -1.0)
    with assert_raises(contains="axis 2"):
        _ = t[0, 0, 4]


def test_a_wrong_length_initializer_raises() raises:
    """The constructor and both `copy_from_host`s check the list's length
    rather than reading or writing past it."""
    with assert_raises(contains="5 values for 6 elements"):
        _ = Static[f32, 2, 3](
            [1.0, 2.0, 3.0, 4.0, 5.0], DeviceContext(api="cpu")
        )
    var a = _grid()
    with assert_raises(contains="3 values for 20 elements"):
        a.copy_from_host([1.0, 2.0, 3.0])
    var v = a[0:1]
    with assert_raises(contains="2 values for 5 elements"):
        v.copy_from_host([1.0, 2.0])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
