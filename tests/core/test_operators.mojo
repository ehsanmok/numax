"""`Tensor`'s reflected, in-place, power, comparison and `@` operators,
each against the free function it spells, and the compile-time shape
check the mixed-operand overloads make (its failures live in
`tests_compile_fail/`, since a program that does not build cannot run).

The operators are sugar: `2.0 - a` must be `numax.core.ops`'s reflected
subtract, `a > 0` must be `greater(a, 0)`, `a += b` must leave `a` equal
to `add(a, b)`. Every test asserts the two spellings agree element for
element, and that a comparison returns a `bool` tensor of `a`'s shape.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from layout.tile_layout import row_major

from max.gpu.host import DeviceContext

from numax.core.logic import (
    equal,
    greater,
    greater_equal,
    less,
    less_equal,
    not_equal,
)
from numax.core.ops import add, divide, multiply, power, subtract
from numax.core._drive import _broadcasts_statically
from numax.core.tensor import Dynamic, Static, _dyn_shape
from numax.linalg import matmul

comptime f64 = DType.float64


def _a() raises -> Static[f64, 2, 3]:
    return Static[f64, 2, 3](
        [1.0, -2.0, 3.5, 0.0, 4.0, -0.5], DeviceContext(api="cpu")
    )


def _b() raises -> Static[f64, 2, 3]:
    return Static[f64, 2, 3](
        [2.0, -2.0, 1.0, 0.5, 4.0, 3.0], DeviceContext(api="cpu")
    )


def _same(got: List[Scalar[f64]], want: List[Scalar[f64]]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def _same_mask(
    got: List[Scalar[DType.bool]], want: List[Scalar[DType.bool]]
) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def test_reflected_operators_put_the_scalar_on_the_left() raises:
    _same((2.0 + _a()).to_host(), add(_a(), 2.0).to_host())
    _same((2.0 * _a()).to_host(), multiply(_a(), 2.0).to_host())
    var diff = (2.0 - _a()).to_host()
    var quot = (3.0 / _b()).to_host()
    var a = _a().to_host()
    var b = _b().to_host()
    for i in range(6):
        assert_equal(diff[i], 2.0 - a[i])
        assert_equal(quot[i], 3.0 / b[i])


def test_power_operators() raises:
    var p = Static[f64, 2, 3](
        [1.0, 2.0, 0.5, 3.0, 1.5, 0.25], DeviceContext(api="cpu")
    )
    var q = Static[f64, 2, 3](
        [1.0, 2.0, 0.5, 3.0, 1.5, 0.25], DeviceContext(api="cpu")
    )
    _same((p**q).to_host(), power(p, q).to_host())
    _same((p**2.0).to_host(), power(p, 2.0).to_host())
    var base = (2.0**p).to_host()
    var pv = p.to_host()
    for i in range(6):
        assert_equal(base[i], 2.0 ** pv[i])


def test_in_place_operators_equal_the_binary_ones() raises:
    var a = _a()
    a += _b()
    _same(a.to_host(), add(_a(), _b()).to_host())
    a -= 1.0
    _same(a.to_host(), subtract(add(_a(), _b()), 1.0).to_host())
    var m = _a()
    m *= _b()
    _same(m.to_host(), multiply(_a(), _b()).to_host())
    m /= 2.0
    _same(m.to_host(), divide(multiply(_a(), _b()), 2.0).to_host())
    var e = _b()
    e **= 2.0
    _same(e.to_host(), power(_b(), 2.0).to_host())


def test_comparisons_return_bool_masks() raises:
    var mask = _a() > 0.0
    assert_equal(mask.dim[0](), 2)
    assert_equal(mask.dim[1](), 3)
    _same_mask(mask.to_host(), greater(_a(), 0.0).to_host())
    _same_mask((_a() < _b()).to_host(), less(_a(), _b()).to_host())
    _same_mask((_a() <= _b()).to_host(), less_equal(_a(), _b()).to_host())
    _same_mask((_a() >= 0.0).to_host(), greater_equal(_a(), 0.0).to_host())
    _same_mask((_a() == _b()).to_host(), equal(_a(), _b()).to_host())
    _same_mask((_a() != 4.0).to_host(), not_equal(_a(), 4.0).to_host())
    var got = (_a() > 0.0).to_host()
    var values = _a().to_host()
    for i in range(6):
        assert_true(got[i] == (values[i] > 0.0))


def test_matmul_operator_is_matmul() raises:
    var a = _a()
    var b = Static[f64, 3, 4](
        [1.0, 0.5, -1.0, 2.0, 0.0, 3.0, 1.5, -2.0, 4.0, 1.0, 0.25, 0.5],
        DeviceContext(api="cpu"),
    )
    var product = a @ b
    assert_equal(product.dim[0](), 2)
    assert_equal(product.dim[1](), 4)
    _same(product.to_host(), matmul(a, b).to_host())


def test_the_static_broadcast_check_is_numpys_rule() raises:
    """`_broadcasts_statically` rejects only what NumPy rejects, and defers
    to run time whenever an extent is not in the type."""
    comptime L23 = type_of(row_major[2, 3]())
    comptime L13 = type_of(row_major[1, 3]())
    comptime L3 = type_of(row_major[3]())
    comptime L45 = type_of(row_major[4, 5]())
    comptime L5 = type_of(row_major[5]())
    comptime L413 = type_of(row_major[4, 1, 3]())
    comptime Ldyn = type_of(row_major(_dyn_shape[2](4, 5)))
    assert_true(_broadcasts_statically[L23, L23]())
    assert_true(_broadcasts_statically[L23, L13]())
    assert_true(_broadcasts_statically[L23, L3]())
    assert_true(_broadcasts_statically[L413, L23]())
    assert_true(not _broadcasts_statically[L23, L45]())
    assert_true(not _broadcasts_statically[L23, L5]())
    assert_true(_broadcasts_statically[L23, Ldyn]())


def test_broadcastable_static_shapes_still_compile_and_agree() raises:
    """A row and a vector against a matrix take the mixed overload, which
    the check lets through, and agree with the free function."""
    var row = Static[f64, 1, 3]([10.0, 20.0, 30.0], DeviceContext(api="cpu"))
    var vec = Static[f64, 3]([1.0, 2.0, 3.0], DeviceContext(api="cpu"))
    _same((_a() + row).to_host(), add(_a(), row).to_host())
    _same((_a() * vec).to_host(), multiply(_a(), vec).to_host())
    _same_mask((_a() > vec).to_host(), greater(_a(), vec).to_host())


def test_a_run_time_shape_mismatch_still_raises() raises:
    """Where an extent is not in the type the check defers, and the
    broadcasting free function raises naming the axis, as before."""
    var d = Dynamic[f64, 2](
        row_major(_dyn_shape[2](4, 5)),
        List[Scalar[f64]](length=20, fill=1.0),
        DeviceContext(api="cpu"),
    )
    with assert_raises(contains="do not broadcast"):
        _ = _a() + d


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
