"""The run-time-shape `eigvalsh` and `eigh` at `gpu=True`: a `Dynamic`
symmetric matrix on the device, reduced and diagonalized there at
`float32`, against the host run at the same `dtype`."""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal

from layout.tile_layout import row_major
from max.gpu.host import DeviceContext

from numax.core.tensor import Dynamic, _dyn_shape
from numax.linalg import eigh, eigvalsh

comptime f32 = DType.float32


def _symmetric(n: Int, ctx: DeviceContext) raises -> Dynamic[f32, 2]:
    var values = List[Scalar[f32]](capacity=n * n)
    for i in range(n):
        for j in range(n):
            var v = sin(Float64(min(i, j) * 31 + max(i, j) * 17) * 0.37)
            if i == j:
                v += Float64(i % 7) * 0.5
            values.append(Scalar[f32](v))
    return Dynamic[f32, 2](row_major(_dyn_shape[2](n, n)), values^, ctx)


def test_dynamic_eigh_gpu_matches_host() raises:
    """`n = 70`: two `sytrd` panels and a remainder. The values against
    the host's; the vectors up to sign, since each column is determined
    only up to one."""
    var n = 70
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var dev = eigh[gpu=True](_symmetric(n, gpu))
    var host = eigh(_symmetric(n, cpu))
    var dw = dev.values.to_host()
    var hw = host.values.to_host()
    var dv = dev.vectors.to_host()
    var hv = host.vectors.to_host()
    for j in range(n):
        assert_almost_equal(Float64(dw[j]), Float64(hw[j]), atol=1e-4)
        var sign = 1.0
        var dotted = 0.0
        for i in range(n):
            dotted += Float64(dv[i * n + j]) * Float64(hv[i * n + j])
        if dotted < 0:
            sign = -1.0
        for i in range(n):
            assert_almost_equal(
                sign * Float64(dv[i * n + j]), Float64(hv[i * n + j]), atol=2e-3
            )
    var only = eigvalsh[gpu=True](_symmetric(n, gpu)).to_host()
    for j in range(n):
        assert_almost_equal(Float64(only[j]), Float64(hw[j]), atol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
