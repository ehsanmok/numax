"""`ldl` at `gpu=True`, against the host, at `float32`: `sytf2_lower` and
`ldl_unpack_lower` are single-block launches, the pivot scans on thread 0
and the interchanges and updates spread over the block, on an indefinite
matrix whose factorization takes `2 x 2` pivots."""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.testing import assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static, transpose
from numax.linalg import ldl, matmul

comptime f32 = DType.float32
comptime n = 40


def _indefinite(ctx: DeviceContext) raises -> Static[f32, n, n]:
    # Symmetric, with a small diagonal against larger off-diagonal
    # entries, so Bunch-Kaufman takes 2 x 2 blocks and interchanges; every
    # entry distinct, so no pivot choice is a tie.
    var values = List[Scalar[f32]](capacity=n * n)
    for i in range(n):
        for j in range(n):
            var lo = min(i, j)
            var hi = max(i, j)
            var v = sin(Float32(lo * n + hi) * 1.37 + 0.2)
            values.append(v * (0.05 if i == j else Float32(1)))
    return Static[f32, n, n](values^, ctx)


def test_ldl_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var dv = ldl[gpu=True](_indefinite(gpu))
    var hv = ldl(_indefinite(cpu))
    assert_false(dv.lu.on_host())
    var dp = dv.perm.to_host()
    var hp = hv.perm.to_host()
    for i in range(n):
        assert_equal(Int(dp[i]), Int(hp[i]))
    var dd = dv.d.to_host()
    var hd = hv.d.to_host()
    var dl = dv.lu.to_host()
    var hl = hv.lu.to_host()
    for i in range(n * n):
        assert_almost_equal(dd[i], hd[i], atol=1e-4, rtol=1e-3)
        assert_almost_equal(dl[i], hl[i], atol=1e-4, rtol=1e-3)
    var back = matmul[gpu=True](
        matmul[gpu=True](dv.lu, dv.d), transpose[gpu=True](dv.lu)
    ).to_host()
    var a = _indefinite(cpu).to_host()
    for i in range(n * n):
        assert_almost_equal(back[i], a[i], atol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
