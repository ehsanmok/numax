"""`schur` and `eigvals` at `gpu=True` above `n = 75`, where the host
iteration is the multishift QR with aggressive early deflation and
`schur`'s Schur vectors come back as `Q Z_h` through a device `matmul`:
`Z T Z^T = A` and the device spectrum's power sums against the host's,
at `float32`."""

from std.testing import TestSuite, assert_almost_equal, assert_true

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.linalg import eigvals, schur

comptime f32 = DType.float32
comptime n = 96


def _general(ctx: DeviceContext) raises -> Static[f32, n, n]:
    var v = List[Scalar[f32]](capacity=n * n)
    var x = 3
    for _ in range(n * n):
        x = (x * 1103515245 + 12345) % 2147483648
        v.append(Scalar[f32](Float64(x) / 2147483648.0 - 0.5))
    return Static[f32, n, n](v^, ctx)


def test_schur_gpu_reconstructs() raises:
    """`Z T Z^T` reproduces `A` from a device-resident `A`."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var f = schur[gpu=True](_general(gpu))
    var t = f.t.to_host()
    var z = f.z.to_host()
    var a = _general(cpu).to_host()
    var zt = List[Float64](length=n * n, fill=0)
    for i in range(n):
        for k in range(n):
            var zik = Float64(z[i * n + k])
            for j in range(n):
                zt[i * n + j] += zik * Float64(t[k * n + j])
    var worst = 0.0
    for i in range(n):
        for j in range(n):
            var s = 0.0
            for k in range(n):
                s += zt[i * n + k] * Float64(z[j * n + k])
            worst = max(worst, abs(s - Float64(a[i * n + j])))
    assert_true(worst < 1e-4)


def test_eigvals_gpu_matches_host() raises:
    """The same spectrum from a device and a host `A`, as power sums."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = eigvals[gpu=True](_general(gpu))
    var h = eigvals(_general(cpu))
    var dre = d.re.to_host()
    var dim_ = d.im.to_host()
    var hre = h.re.to_host()
    var him = h.im.to_host()
    var s1 = 0.0
    var s2 = 0.0
    var q1 = 0.0
    var q2 = 0.0
    for i in range(n):
        s1 += Float64(dre[i])
        s2 += Float64(hre[i])
        q1 += Float64(dre[i]) ** 2 + Float64(dim_[i]) ** 2
        q2 += Float64(hre[i]) ** 2 + Float64(him[i]) ** 2
    assert_almost_equal(s1, s2, atol=1e-3)
    assert_almost_equal(q1, q2, atol=1e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
