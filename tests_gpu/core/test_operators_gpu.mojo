"""`+ - * /` and unary `-` on GPU-context tensors follow the tensor to its
device. An operator cannot spell `gpu=True`, so it checks residency at run
time on a build with an accelerator; every result here must stay on the
device and equal the host's.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
)

from max.gpu.host import DeviceContext

from numax.core.tensor import Static

comptime f32 = DType.float32
comptime n = 1000


def _ramp(ctx: DeviceContext, scale: Float32) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(i % 17) * scale + 0.5)
    return Static[f32, n](ctx, values^)


def _assert_close(got: List[Scalar[f32]], want: List[Scalar[f32]]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-6, rtol=1e-6)


def test_the_binary_operators_stay_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var a = _ramp(gpu, 0.25)
    var b = _ramp(gpu, -0.5)
    var ha = _ramp(cpu, 0.25)
    var hb = _ramp(cpu, -0.5)
    var s = a + b
    assert_false(s.on_host())
    _assert_close(s.to_host(), (ha + hb).to_host())
    _assert_close((a - b).to_host(), (ha - hb).to_host())
    _assert_close((a * b).to_host(), (ha * hb).to_host())
    _assert_close((a / b).to_host(), (ha / hb).to_host())


def test_the_scalar_and_unary_operators_stay_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var a = _ramp(gpu, 0.25)
    var ha = _ramp(cpu, 0.25)
    var twice = a * 2.0
    assert_false(twice.on_host())
    _assert_close(twice.to_host(), (ha * 2.0).to_host())
    _assert_close((a + 1.5).to_host(), (ha + 1.5).to_host())
    _assert_close((a - 1.5).to_host(), (ha - 1.5).to_host())
    _assert_close((a / 4.0).to_host(), (ha / 4.0).to_host())
    var neg = -a
    assert_false(neg.on_host())
    _assert_close(neg.to_host(), (-ha).to_host())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
