"""The N-d transforms at `gpu=True`, against the host, at `float32`.

`fftn`/`ifftn` on a `4 x 6 x 8` volume (a Bluestein axis between two
powers of two), `rfftn`/`irfftn` at rank 3 with an odd last axis, and
`irfft2`, keeping to one non-power-of-two extent per transform because
each distinct Bluestein length is its own set of device kernels to
compile: each device result against the host's to `float32` rounding
scaled by the transform size, and every round trip back to its input.
"""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static, zeros
from numax.fft import fftn, ifftn, irfft2, irfftn, rfft2, rfftn

comptime f32 = DType.float32


def _values(count: Int) -> List[Scalar[f32]]:
    var values = List[Scalar[f32]](capacity=count)
    for i in range(count):
        values.append(Float32((i * 37) % 23) * 0.25 - 2.5)
    return values^


def _close(got: List[Float32], want: List[Float32], tol: Float64) raises:
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=tol)


def test_fftn_and_ifftn_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = fftn[gpu=True](
        (Static[f32, 4, 6, 8](_values(192), gpu), zeros[f32, 4, 6, 8](gpu))
    )
    assert_false(d[0].on_host())
    var h = fftn(
        (Static[f32, 4, 6, 8](_values(192), cpu), zeros[f32, 4, 6, 8](cpu))
    )
    _close(d[0].to_host(), h[0].to_host(), 1e-3)
    _close(d[1].to_host(), h[1].to_host(), 1e-3)
    var back = ifftn[gpu=True](d^)
    _close(back[0].to_host(), _values(192), 1e-5)


def test_rfftn_and_irfftn_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = rfftn[gpu=True](Static[f32, 4, 4, 5](_values(80), gpu))
    var h = rfftn(Static[f32, 4, 4, 5](_values(80), cpu))
    _close(d[0].to_host(), h[0].to_host(), 1e-3)
    _close(d[1].to_host(), h[1].to_host(), 1e-3)
    var back = irfftn[gpu=True, n=5](d^)
    _close(back.to_host(), _values(80), 1e-5)


def test_irfft2_on_the_device() raises:
    var gpu = DeviceContext()
    var half = rfft2[gpu=True](Static[f32, 4, 8](_values(32), gpu))
    var back = irfft2[gpu=True](half^)
    assert_false(back.on_host())
    _close(back.to_host(), _values(32), 1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
