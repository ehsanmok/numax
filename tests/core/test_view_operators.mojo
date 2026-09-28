"""`TensorView`'s operators and printing, against the same on the `Tensor`
it borrows.

A view is the same data under another type, so every operator on it must
give what the operator gives on the owner, and `print(v)` must be
`print(a)`'s format rather than MAX's tile printer.
"""

from std.testing import TestSuite, assert_equal, assert_true

from max.gpu.host import DeviceContext

from numax.core.logic import greater
from numax.core.ops import add
from numax.core.tensor import Static
from numax.core.tensorlike import TensorView

comptime f64 = DType.float64


def _a() raises -> Static[f64, 2, 3]:
    return Static[f64, 2, 3](
        DeviceContext(api="cpu"), [1.0, -2.0, 3.5, 0.5, 4.0, -0.5]
    )


def _same(got: List[Scalar[f64]], want: List[Scalar[f64]]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def test_view_arithmetic_matches_the_tensor() raises:
    var a = _a()
    var b = _a()
    var v = TensorView(a.tile(), a.context())
    var w = TensorView(b.tile(), b.context())
    _same((v + w).to_host(), (_a() + _a()).to_host())
    _same((v - 1.0).to_host(), (_a() - 1.0).to_host())
    _same((v * w).to_host(), (_a() * _a()).to_host())
    _same((v / 2.0).to_host(), (_a() / 2.0).to_host())
    _same((2.0 - v).to_host(), (2.0 - _a()).to_host())
    _same((3.0 / v).to_host(), (3.0 / _a()).to_host())
    _same((2.0 * v).to_host(), (2.0 * _a()).to_host())
    _same((-v).to_host(), (-_a()).to_host())
    _same((v**2.0).to_host(), (_a() ** 2.0).to_host())


def test_view_comparisons_match_the_tensor() raises:
    var a = _a()
    var v = TensorView(a.tile(), a.context())
    var got = (v > 0.0).to_host()
    var want = (_a() > 0.0).to_host()
    for i in range(6):
        assert_equal(got[i], want[i])
    var eq = (v == v).to_host()
    for i in range(6):
        assert_true(eq[i])


def test_mixed_operands_broadcast_like_the_free_functions() raises:
    """A tensor and a view, and a matrix and a row, meet through the
    broadcasting free functions."""
    var a = _a()
    var v = TensorView(a.tile(), a.context())
    _same((_a() + v).to_host(), add(_a(), v).to_host())
    var row = Static[f64, 3](DeviceContext(api="cpu"), [10.0, 20.0, 30.0])
    var summed = _a() + row
    assert_equal(summed.dim[0](), 2)
    assert_equal(summed.dim[1](), 3)
    _same(summed.to_host(), add(_a(), row).to_host())
    var mask = (v > row).to_host()
    var want = greater(v, row).to_host()
    for i in range(6):
        assert_equal(mask[i], want[i])


def test_a_view_prints_like_its_tensor() raises:
    var a = _a()
    var v = TensorView(a.tile(), a.context())
    assert_equal(String(v), String(a))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
