"""`Tensor`'s reflected, in-place, power, comparison and `@` operators,
each against the free function it spells.

The operators are sugar: `2.0 - a` must be `numax.core.ops`'s reflected
subtract, `a > 0` must be `greater(a, 0)`, `a += b` must leave `a` equal
to `add(a, b)`. Every test asserts the two spellings agree element for
element, and that a comparison returns a `bool` tensor of `a`'s shape.
"""

from std.testing import TestSuite, assert_equal, assert_true

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
from numax.core.tensor import Static
from numax.linalg import matmul

comptime f64 = DType.float64


def _a() raises -> Static[f64, 2, 3]:
    return Static[f64, 2, 3](
        DeviceContext(api="cpu"), [1.0, -2.0, 3.5, 0.0, 4.0, -0.5]
    )


def _b() raises -> Static[f64, 2, 3]:
    return Static[f64, 2, 3](
        DeviceContext(api="cpu"), [2.0, -2.0, 1.0, 0.5, 4.0, 3.0]
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
        DeviceContext(api="cpu"), [1.0, 2.0, 0.5, 3.0, 1.5, 0.25]
    )
    var q = Static[f64, 2, 3](
        DeviceContext(api="cpu"), [1.0, 2.0, 0.5, 3.0, 1.5, 0.25]
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
        DeviceContext(api="cpu"),
        [1.0, 0.5, -1.0, 2.0, 0.0, 3.0, 1.5, -2.0, 4.0, 1.0, 0.25, 0.5],
    )
    var product = a @ b
    assert_equal(product.dim[0](), 2)
    assert_equal(product.dim[1](), 4)
    _same(product.to_host(), matmul(a, b).to_host())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
