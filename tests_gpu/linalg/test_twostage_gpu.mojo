"""`eigvalsh` and `svdvals` at `gpu=True` on the two-stage path: stage 1's
panels and products on the device, the band chase on the host, against
the host run at `float32`."""

from std.testing import TestSuite, assert_almost_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.linalg import eigvalsh, svdvals

comptime f32 = DType.float32


def _lcg(count: Int, seed: Int) -> List[Float64]:
    var out = List[Float64](capacity=count)
    var x = seed
    for _ in range(count):
        x = (x * 1103515245 + 12345) % 2147483648
        out.append(Float64(x) / 2147483648.0 - 0.5)
    return out^


def _symmetric(n: Int) -> List[Scalar[f32]]:
    var v = _lcg(n * (n + 1) // 2, 3)
    var a = List[Scalar[f32]](length=n * n, fill=0)
    var k = 0
    for i in range(n):
        for j in range(i + 1):
            a[i * n + j] = Scalar[f32](v[k])
            a[j * n + i] = Scalar[f32](v[k])
            k += 1
    return a^


def test_eigvalsh_two_stage_gpu() raises:
    """`n = 300` from a device matrix, against the host."""
    comptime n = 300
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = eigvalsh[gpu=True](Static[f32, n, n](_symmetric(n), gpu)).to_host()
    var h = eigvalsh(Static[f32, n, n](_symmetric(n), cpu)).to_host()
    for i in range(n):
        assert_almost_equal(Float64(d[i]), Float64(h[i]), atol=1e-4)


def test_svdvals_two_stage_gpu() raises:
    """`80 x 64` from a device matrix, against the host."""
    var v = _lcg(80 * 64, 9)
    var host = List[Scalar[f32]](capacity=80 * 64)
    for i in range(80 * 64):
        host.append(Scalar[f32](v[i]))
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = svdvals[gpu=True](Static[f32, 80, 64](host.copy(), gpu)).to_host()
    var h = svdvals(Static[f32, 80, 64](host^, cpu)).to_host()
    for i in range(64):
        assert_almost_equal(Float64(d[i]), Float64(h[i]), atol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
