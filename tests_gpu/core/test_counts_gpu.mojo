"""`all`, `any`, `count_nonzero`, `any_nonzero` and `all_nonzero` at
`gpu=True`: a device count (one flag launch and one `ReduceSum`) against
the host's short-circuiting loop."""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from max.gpu.host import DeviceContext

from numax.core.logic import all, allclose, any, array_equal, greater
from numax.core.sorting import all_nonzero, any_nonzero, count_nonzero
from numax.core.tensor import Static

comptime f32 = DType.float32
comptime n = 5000


def _data(ctx: DeviceContext, zero_every: Int) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(
            Float32(0) if (i + 1) % zero_every == 0 else Float32(i % 7) - 3.5
        )
    return Static[f32, n](ctx, values^)


def test_count_nonzero_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    for every in [3, 97, n + 1]:
        assert_equal(
            count_nonzero[gpu=True](_data(gpu, every)),
            count_nonzero(_data(cpu, every)),
        )
    assert_true(any_nonzero[gpu=True](_data(gpu, 3)))
    assert_false(all_nonzero[gpu=True](_data(gpu, 3)))
    assert_true(all_nonzero[gpu=True](_data(gpu, n + 1)))


def test_nan_counts_and_negative_zero_does_not() raises:
    var gpu = DeviceContext()
    var values: List[Scalar[f32]] = [
        0.0,
        -0.0,
        Float32.MAX * 2 - Float32.MAX * 2,
        1.0,
    ]
    assert_equal(count_nonzero[gpu=True](Static[f32, 4](gpu, values^)), 2)


def test_all_and_any_of_a_device_mask() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var a = _data(gpu, 3)
    var b = _data(gpu, 5)
    var mask = greater[gpu=True](a, b)
    var host_mask = greater(_data(cpu, 3), _data(cpu, 5))
    assert_equal(all[gpu=True](mask), all(host_mask))
    assert_equal(any[gpu=True](mask), any(host_mask))
    var none = greater[gpu=True](a, a)
    assert_false(any[gpu=True](none))
    assert_true(array_equal[gpu=True](a, _data(gpu, 3)))
    assert_false(array_equal[gpu=True](a, b))
    assert_true(allclose[gpu=True](a, _data(gpu, 3)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
