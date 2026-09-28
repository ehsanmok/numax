"""The fallback policy end to end: a device tensor handed to a routine
left at `gpu=False` falls back to the host, and `set_fallback` decides
whether that warns, raises or stays quiet. Each test restores `"warn"`.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_raises

from max.gpu.host import DeviceContext

from numax.core import set_fallback
from numax.core.ops import add
from numax.core.tensor import Static

comptime f32 = DType.float32


def _pair(ctx: DeviceContext) raises -> Static[f32, 64]:
    var values = List[Scalar[f32]](capacity=64)
    for i in range(64):
        values.append(Float32(i))
    return Static[f32, 64](ctx, values^)


def test_raise_policy_rejects_a_mismatch() raises:
    var gpu = DeviceContext()
    set_fallback("raise")
    try:
        _ = add(_pair(gpu), _pair(gpu))
        set_fallback("warn")
        raise Error("expected a raise")
    except e:
        set_fallback("warn")
        assert_equal(String(e).startswith("numax: add ran on the host"), True)


def test_silent_and_warn_still_answer() raises:
    var gpu = DeviceContext()
    set_fallback("silent")
    var s = add(_pair(gpu), _pair(gpu)).to_host()
    set_fallback("warn")
    var w = add(_pair(gpu), _pair(gpu)).to_host()
    for i in range(64):
        assert_equal(s[i], Float32(2 * i))
        assert_equal(w[i], Float32(2 * i))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
