"""Tests for `numax.core.tensorlike`: the `TensorLike` bound, the borrowed
`View`, and `Tensor`'s tracked-origin `view()`.

The claims checked are the ones the design rests on: one generic routine
runs unchanged on an owned `Tensor` and on a `View`; a `View` over a
sub-block of a tensor is zero-copy and writes land in the parent; a
read-only routine can take a `Tensor` by borrow and still call `view()`;
the tracked view erases to `MutAnyOrigin` where a kernel asks for it; and
the compile-time helpers `dim` and `is_row_major` report what they claim,
including refusing a strided block.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from layout import Coord, TileTensor

from numax.core.array import (
    Static,
    Tensor,
    arange,
    reshape,
    transpose,
    zeros,
    zeros_dyn,
)
from numax.core.elementwise import exp
from numax.core.logic import greater
from numax.core.ops import add, multiply
from numax.core.sorting import extract, sort
from numax.core.tensor import add_combine, map, reduce
from numax.core.tensorlike import TensorLike, View, dim, is_row_major

comptime f64 = DType.float64


def _total[T: TensorLike](a: T) raises -> Float64:
    """Sum every element through the layout, on any conformer."""
    var v = a.view()
    var acc = Float64(0)
    for i in range(dim[T, 0]):
        for j in range(dim[T, 1]):
            acc += Float64(v.load[1](Coord(i, j)))
    return acc


def _fill[T: TensorLike](mut a: T, value: Float64):
    """Write `value` into every element, on any conformer."""
    var v = a.view()
    for i in range(dim[T, 0]):
        for j in range(dim[T, 1]):
            v.store(Coord(i, j), Scalar[T.dtype](value))


def _double[w: Int](x: SIMD[f64, w]) -> SIMD[f64, w]:
    return x * 2


def test_tensor_and_view_conform_and_agree() raises:
    var m = reshape[rows=3, cols=4](arange[12, f64]())
    var w = View(m.view())
    assert_equal(_total(m), 66.0)
    assert_equal(_total(w), 66.0)
    assert_equal(dim[type_of(m), 0], 3)
    assert_equal(dim[type_of(w), 1], 4)


def test_view_over_a_block_writes_into_the_parent() raises:
    var a = zeros[f64, 4, 4]()
    var block = View(a.view().tile[2, 2](0, 0), a.context())
    _fill(block, 7.0)
    # The top-left quadrant changed and nothing else did.
    assert_equal(a[0, 0], 7.0)
    assert_equal(a[1, 1], 7.0)
    assert_equal(a[0, 2], 0.0)
    assert_equal(a[2, 0], 0.0)
    assert_equal(a[3, 3], 0.0)
    assert_equal(_total(block), 28.0)


def test_readonly_routine_can_view_a_borrowed_tensor() raises:
    # `view()` used to take `mut self`; a borrowed argument could not call it.
    def readonly(x: Static[f64, 2, 3]) raises -> Float64:
        return _total(x)

    var a = reshape[rows=2, cols=3](arange[6, f64]())
    assert_equal(readonly(a), 15.0)


def test_tracked_view_erases_to_any_origin_for_kernels() raises:
    var a = arange[4, f64]()
    var b = zeros[f64, 4]()
    # The kernel tier spells `MutAnyOrigin`; the tracked view converts at
    # the call site with inference intact.
    map[step=_double](a.view(), b.view())
    assert_equal(b[3], 6.0)
    var v: TileTensor[f64, type_of(a).LayoutType, MutAnyOrigin] = a.view()
    v.store(Coord(0), Float64(10))
    assert_equal(a[0], 10.0)
    assert_equal(reduce[combine=add_combine[f64]](a.view(), 0.0), 16.0)


def test_view_without_a_context_is_a_host_view() raises:
    var a = zeros[f64, 2, 2]()
    var w = View(a.view())
    assert_equal(w.context().api(), "cpu")
    assert_equal(w.size(), 4)
    assert_equal(w.dim[0](), 2)


def test_is_row_major_refuses_a_strided_block() raises:
    var a = zeros[f64, 4, 4]()
    var block = View(a.view().tile[2, 2](0, 0))
    var whole = View(a.view())
    assert_true(is_row_major[type_of(a)])
    assert_true(is_row_major[type_of(whole)])
    assert_false(is_row_major[type_of(block)])
    var d = zeros_dyn[f64, 2](3, 3)
    assert_true(is_row_major[type_of(d)])


def test_tensor_device_type_is_the_erased_view() raises:
    comptime T = Static[f64, 3, 3]
    assert_true(
        T.device_type == TileTensor[f64, T.LayoutType, MutAnyOrigin],
        "a Tensor crosses the launch boundary as its MutAnyOrigin view",
    )
    assert_equal(T.get_type_name(), "Tensor[float64, rank=2]")


def test_core_surface_accepts_a_view_and_agrees_with_the_tensor() raises:
    # One routine, two conformers: the public surface takes either.
    var a = arange[6, f64]()
    var m = reshape[rows=2, cols=3](arange[6, f64]())
    var v = View(a.view())
    var vm = View(m.view())
    var e_t = exp(a).to_host()
    var e_v = exp(v).to_host()
    for i in range(6):
        assert_equal(e_t[i], e_v[i])
    var s_t = add(a, a).to_host()
    var s_v = add(v, v).to_host()
    for i in range(6):
        assert_equal(s_t[i], s_v[i])
    var t_v = transpose(vm)
    assert_equal(t_v[2, 1], 5.0)
    assert_equal(sort(vm)[5], 5.0)
    var mask = greater(a, multiply(a, 0.0))
    assert_equal(extract(mask, v).size(), 5)


def test_view_to_host_reads_a_strided_block_in_its_own_order() raises:
    var m = reshape[rows=4, cols=4](arange[16, f64]())
    var block = View(m.view().tile[2, 2](1, 1))
    var got = block.to_host()
    # Rows 2..3, columns 2..3 of the 4x4 arange: 10 11 / 14 15.
    assert_equal(got[0], 10.0)
    assert_equal(got[1], 11.0)
    assert_equal(got[2], 14.0)
    assert_equal(got[3], 15.0)
    var values = List[Float64]()
    for i in range(4):
        values.append(Float64(-i))
    block.copy_from_host(values)
    assert_equal(m[2, 3], -1.0)
    assert_equal(m[3, 2], -2.0)
    assert_equal(m[0, 0], 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
