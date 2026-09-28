"""`convolve2d` and `correlate2d` at `gpu=True`, against the host, at
`float32`: one lane per output element summing the same kernel products,
so the results agree to `float32` rounding of the sums."""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.signal import MODE_SAME, convolve2d, correlate2d

comptime f32 = DType.float32


def _image(ctx: DeviceContext) raises -> Static[f32, 64, 48]:
    var values = List[Scalar[f32]](capacity=64 * 48)
    for i in range(64 * 48):
        values.append(Float32((i * 37) % 29) * 0.1 - 1.4)
    return Static[f32, 64, 48](values^, ctx)


def _kernel(ctx: DeviceContext) raises -> Static[f32, 5, 4]:
    var values = List[Scalar[f32]](capacity=20)
    for i in range(20):
        values.append(Float32((i * 5) % 7) * 0.25 - 0.5)
    return Static[f32, 5, 4](values^, ctx)


def test_convolve2d_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = convolve2d[gpu=True](_image(gpu), _kernel(gpu))
    assert_false(d.on_host())
    var got = d.to_host()
    var want = convolve2d(_image(cpu), _kernel(cpu)).to_host()
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-4)
    var c = correlate2d[mode=MODE_SAME, boundary="symm", gpu=True](
        _image(gpu), _kernel(gpu)
    ).to_host()
    var ch = correlate2d[mode=MODE_SAME, boundary="symm"](
        _image(cpu), _kernel(cpu)
    ).to_host()
    for i in range(len(ch)):
        assert_almost_equal(c[i], ch[i], atol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
