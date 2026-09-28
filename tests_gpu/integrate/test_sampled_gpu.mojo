"""`trapezoid` and `simpson` over samples at `gpu=True`, against the host:
uniform `dx` and sample points `x`, at an odd and an even count (the even
one exercises Simpson's last-interval correction)."""

from std.testing import TestSuite, assert_almost_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.integrate import simpson, trapezoid

comptime f32 = DType.float32


def _y[m: Int](ctx: DeviceContext) raises -> Static[f32, m]:
    var values = List[Scalar[f32]](capacity=m)
    for i in range(m):
        var t = Float32(i) / Float32(m - 1)
        values.append(t * t * (3.0 - t))
    return Static[f32, m](ctx, values^)


def _x[m: Int](ctx: DeviceContext) raises -> Static[f32, m]:
    var values = List[Scalar[f32]](capacity=m)
    for i in range(m):
        var t = Float32(i) / Float32(m - 1)
        values.append(t + 0.1 * t * t)
    return Static[f32, m](ctx, values^)


def _check[m: Int]() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var dx = Scalar[f32](0.01)
    assert_almost_equal(
        trapezoid[gpu=True](_y[m](gpu), dx),
        trapezoid(_y[m](cpu), dx),
        rtol=1e-5,
    )
    assert_almost_equal(
        simpson[gpu=True](_y[m](gpu), dx), simpson(_y[m](cpu), dx), rtol=1e-5
    )
    assert_almost_equal(
        trapezoid[gpu=True](_y[m](gpu), _x[m](gpu)),
        trapezoid(_y[m](cpu), _x[m](cpu)),
        rtol=1e-5,
    )
    assert_almost_equal(
        simpson[gpu=True](_y[m](gpu), _x[m](gpu)),
        simpson(_y[m](cpu), _x[m](cpu)),
        rtol=1e-5,
    )


def test_odd_count_on_the_device() raises:
    _check[1001]()


def test_even_count_on_the_device() raises:
    _check[1000]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
