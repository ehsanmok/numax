"""`select` and the NaN-ignoring reductions at `gpu=True`, against the host.

Each `nan*` reduction is `isnan`, a `select` against a fill, and a MAX
monoid; with `select` on the device the whole composition stays there and
only the scalar answer crosses back.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
)

from max.gpu.host import DeviceContext

from numax.core.logic import isnan
from numax.core.sorting import select
from numax.core.tensor import Static
from numax.stats import nanmax, nanmean, nanmin, nanprod, nanstd, nansum, nanvar

comptime f32 = DType.float32
comptime n = 3000


def _with_nans(ctx: DeviceContext) raises -> Static[f32, n]:
    var nan = Float32.MAX * 2 - Float32.MAX * 2
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(nan if i % 7 == 3 else Float32(i % 13) * 0.125 - 0.5)
    return Static[f32, n](values^, ctx)


def _small(ctx: DeviceContext) raises -> Static[f32, 20]:
    var nan = Float32.MAX * 2 - Float32.MAX * 2
    var values = List[Scalar[f32]](capacity=20)
    for i in range(20):
        values.append(nan if i % 5 == 0 else 1.0 + Float32(i % 3) * 0.01)
    return Static[f32, 20](values^, ctx)


def test_select_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = select[gpu=True](
        isnan[gpu=True](_with_nans(gpu)), _with_nans(gpu), _small_fill(gpu)
    )
    assert_false(d.on_host())
    var h = select(isnan(_with_nans(cpu)), _with_nans(cpu), _small_fill(cpu))
    var got = d.to_host()
    var want = h.to_host()
    for i in range(n):
        if want[i] == want[i]:
            assert_equal(got[i], want[i])


def _small_fill(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(-i))
    return Static[f32, n](values^, ctx)


def test_the_nan_reductions_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    assert_almost_equal(
        nansum[gpu=True](_with_nans(gpu)), nansum(_with_nans(cpu)), rtol=1e-5
    )
    assert_almost_equal(
        nanmean[gpu=True](_with_nans(gpu)), nanmean(_with_nans(cpu)), rtol=1e-5
    )
    assert_almost_equal(
        nanvar[gpu=True](_with_nans(gpu)), nanvar(_with_nans(cpu)), rtol=1e-4
    )
    assert_almost_equal(
        nanstd[gpu=True](_with_nans(gpu), 1),
        nanstd(_with_nans(cpu), 1),
        rtol=1e-4,
    )
    assert_equal(nanmin[gpu=True](_with_nans(gpu)), nanmin(_with_nans(cpu)))
    assert_equal(nanmax[gpu=True](_with_nans(gpu)), nanmax(_with_nans(cpu)))
    assert_almost_equal(
        nanprod[gpu=True](_small(gpu)), nanprod(_small(cpu)), rtol=1e-5
    )


def test_the_broadcasting_select_on_the_device() raises:
    """A `(3, 1)` mask, a `(3, 4)` `x` and a `(1, 4)` `y`: every operand
    stretched on some axis."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var mask_values: List[Scalar[DType.bool]] = [True, False, True]
    var x_values = List[Scalar[f32]](capacity=12)
    for i in range(12):
        x_values.append(Float32(i))
    var y_values: List[Scalar[f32]] = [-1.0, -2.0, -3.0, -4.0]
    var d = select[gpu=True](
        Static[DType.bool, 3, 1](mask_values.copy(), gpu),
        Static[f32, 3, 4](x_values.copy(), gpu),
        Static[f32, 1, 4](y_values.copy(), gpu),
    )
    assert_false(d.on_host())
    var h = select(
        Static[DType.bool, 3, 1](mask_values^, cpu),
        Static[f32, 3, 4](x_values^, cpu),
        Static[f32, 1, 4](y_values^, cpu),
    )
    var got = d.to_host()
    var want = h.to_host()
    assert_equal(len(got), 12)
    for i in range(12):
        assert_equal(got[i], want[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
