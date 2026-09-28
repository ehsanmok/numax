"""`max` and `min` at `gpu=True` over infinite extremes, against NumPy.

The device fold starts from the same finite identity as the host one,
so the repair has to run there too: an all-`-inf` tensor, a matrix with
one all-`-inf` row, and a row that really holds the finite edge value,
which must be left alone.
"""

from std.testing import TestSuite, assert_equal
from std.utils.numerics import inf, neg_inf

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import max, min

comptime f32 = DType.float32


def test_whole_tensor_extremes_on_the_device() raises:
    var gpu = DeviceContext()
    var low = Static[f32, 300](gpu)
    var values = List[Scalar[f32]](length=300, fill=neg_inf[f32]())
    low.copy_from_host(values)
    assert_equal(max[gpu=True](low), neg_inf[f32]())
    var high = Static[f32, 3]([inf[f32](), inf[f32](), inf[f32]()], gpu)
    assert_equal(min[gpu=True](high), inf[f32]())


def test_axis_extremes_on_the_device() raises:
    var gpu = DeviceContext()
    var edge = Scalar[f32].MIN_FINITE
    var m = Static[f32, 3, 2](
        [
            neg_inf[f32](),
            1.0,
            neg_inf[f32](),
            neg_inf[f32](),
            edge,
            neg_inf[f32](),
        ],
        gpu,
    )
    var rows = max[axis=1, gpu=True](m).to_host()
    assert_equal(rows[0], 1.0)
    assert_equal(rows[1], neg_inf[f32]())
    assert_equal(rows[2], edge)
    var h = Static[f32, 2, 2]([inf[f32](), inf[f32](), 2.0, inf[f32]()], gpu)
    var cols = min[axis=0, gpu=True](h).to_host()
    assert_equal(cols[0], 2.0)
    assert_equal(cols[1], inf[f32]())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
