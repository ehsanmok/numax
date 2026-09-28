"""The NumPy-named core surface at `gpu=True`, against the host answer.

Every test builds one input on the host and a copy on the device, runs the
routine at `gpu=False` and `gpu=True`, and compares at `float32`: Metal has
no `double`, so `float32` is the width every device proves. Needs a real
accelerator; `pixi run tests-gpu` runs this tree, `pixi run tests` does not.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.elementwise import exp
from numax.core.ops import add
from numax.core.tensor import Static
from numax.stats import argmax
from numax.stats import sum as tensor_sum

comptime f32 = DType.float32
comptime n = 4096


def _wave(ctx: DeviceContext) raises -> Static[f32, n]:
    """`sin`-free but non-monotone data: a ramp folded at `n // 3`."""
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        var x = Float32(i % (n // 3)) / Float32(n) - Float32(0.1)
        values.append(x)
    return Static[f32, n](ctx, values^)


def _assert_close(got: List[Scalar[f32]], want: List[Scalar[f32]]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-6, rtol=1e-5)


def test_exp_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    _assert_close(
        exp[gpu=True](_wave(gpu)).to_host(), exp(_wave(cpu)).to_host()
    )


def test_add_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    _assert_close(
        add[gpu=True](_wave(gpu), _wave(gpu)).to_host(),
        add(_wave(cpu), _wave(cpu)).to_host(),
    )


def test_sum_and_argmax_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = tensor_sum[gpu=True](_wave(gpu))
    var h = tensor_sum(_wave(cpu))
    assert_almost_equal(d, h, atol=1e-2)
    assert_equal(argmax[gpu=True](_wave(gpu)), argmax(_wave(cpu)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
