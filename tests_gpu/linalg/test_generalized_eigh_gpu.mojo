"""`eigh(a, b)` and `eigvalsh(a, b)` at `gpu=True`, against the host, at
`float32`: the Cholesky, the three triangular solves and the standard
`eigh` all run on the device, and the device eigenvectors satisfy
`a X = b X Lambda`."""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.linalg import eigh, eigvalsh, matmul

comptime f32 = DType.float32
comptime n = 32


def _sym(
    ctx: DeviceContext, shift: Float32, seed: Float32
) raises -> Static[f32, n, n]:
    """`M + M^T + shift I` for a fixed `M`: symmetric, and positive
    definite once `shift` is large enough."""
    var m = List[Float32](capacity=n * n)
    for i in range(n * n):
        m.append(sin(Float32(i) * seed + 0.3))
    var values = List[Scalar[f32]](capacity=n * n)
    for r in range(n):
        for c in range(n):
            var v = m[r * n + c] + m[c * n + r]
            values.append(v + (shift if r == c else Float32(0)))
    return Static[f32, n, n](values^, ctx)


def test_generalized_eigh_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var da = _sym(gpu, 0.0, 1.37)
    var db = _sym(gpu, Float32(2 * n), 0.53)
    var d = eigh[gpu=True](da, db)
    assert_false(d.vectors.on_host())
    var h = eigvalsh(_sym(cpu, 0.0, 1.37), _sym(cpu, Float32(2 * n), 0.53))
    var dv = d.values.to_host()
    var hv = h.to_host()
    var only = eigvalsh[gpu=True](da, db).to_host()
    for i in range(n):
        assert_almost_equal(dv[i], hv[i], atol=1e-4)
        assert_almost_equal(only[i], dv[i], atol=1e-5)
    var ax = matmul[gpu=True](da, d.vectors).to_host()
    var bx = matmul[gpu=True](db, d.vectors).to_host()
    for i in range(n):
        for j in range(n):
            assert_almost_equal(ax[i * n + j], bx[i * n + j] * dv[j], atol=2e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
