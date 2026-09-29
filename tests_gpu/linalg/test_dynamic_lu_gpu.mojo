"""The run-time-shape `lu_factor`, `solve`, `det` and `cholesky` at
`gpu=True`: a
`Dynamic` matrix on the device, factored and solved there at `float32`,
against the host run at the same `dtype`."""

from std.testing import TestSuite, assert_almost_equal

from layout.tile_layout import row_major
from max.gpu.host import DeviceContext

from numax.core.tensor import Dynamic, _dyn_shape
from numax.linalg import cholesky, det, solve

comptime f32 = DType.float32


def _matrix(n: Int, ctx: DeviceContext) raises -> Dynamic[f32, 2]:
    var values = List[Scalar[f32]](capacity=n * n)
    for i in range(n):
        for j in range(n):
            var v = Float64((i * 7 + j * 3) % 5) * 0.25 - 0.5
            if i == j:
                v = Float64(n) + 1.0
            values.append(Scalar[f32](v))
    return Dynamic[f32, 2](row_major(_dyn_shape[2](n, n)), values^, ctx)


def _vector(n: Int, ctx: DeviceContext) raises -> Dynamic[f32, 1]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Scalar[f32](Float64(i) - 1.5))
    return Dynamic[f32, 1](row_major(_dyn_shape[1](n)), values^, ctx)


def test_dynamic_solve_gpu_matches_host() raises:
    """`n = 70`: two panels and a remainder, on the device and the host."""
    var n = 70
    var gpu_ctx = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var xd = solve[gpu=True](_matrix(n, gpu_ctx), _vector(n, gpu_ctx))
    var xh = solve(_matrix(n, cpu), _vector(n, cpu))
    var d = xd.to_host()
    var h = xh.to_host()
    for i in range(n):
        assert_almost_equal(Float64(d[i]), Float64(h[i]), atol=1e-5)
    var dd = det[gpu=True](_matrix(8, gpu_ctx))
    var dh = det(_matrix(8, cpu))
    assert_almost_equal(Float64(dd), Float64(dh), rtol=1e-4)


def test_dynamic_cholesky_gpu_matches_host() raises:
    """`n = 70` on the device against the host at `float32`."""
    var n = 70
    var gpu_ctx = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var ld = cholesky[gpu=True](_matrix(n, gpu_ctx)).to_host()
    var lh = cholesky(_matrix(n, cpu)).to_host()
    for e in range(n * n):
        assert_almost_equal(Float64(ld[e]), Float64(lh[e]), atol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
