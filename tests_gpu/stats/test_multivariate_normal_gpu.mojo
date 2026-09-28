"""`multivariate_normal.logpdf` at `gpu=True`, against the host, at
`float32`: the factorization, the triangular solve and the fold on the
device, over 256 points of a correlated 4-D normal."""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import multivariate_normal

comptime f32 = DType.float32
comptime m = 256


def _points(ctx: DeviceContext) raises -> Static[f32, m, 4]:
    var values = List[Scalar[f32]](capacity=m * 4)
    for i in range(m * 4):
        values.append(Float32((i * 37) % 23) * 0.2 - 2.2)
    return Static[f32, m, 4](values^, ctx)


def _cov(ctx: DeviceContext) raises -> Static[f32, 4, 4]:
    return Static[f32, 4, 4](
        [
            2.0,
            0.5,
            0.1,
            0.0,
            0.5,
            1.5,
            0.2,
            0.1,
            0.1,
            0.2,
            1.0,
            0.3,
            0.0,
            0.1,
            0.3,
            1.2,
        ],
        ctx,
    )


def test_logpdf_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var mean_g = Static[f32, 4]([0.5, -0.5, 0.0, 1.0], gpu)
    var mean_c = Static[f32, 4]([0.5, -0.5, 0.0, 1.0], cpu)
    var d = multivariate_normal.logpdf[gpu=True](
        _points(gpu), mean_g, _cov(gpu)
    )
    assert_false(d.on_host())
    var got = d.to_host()
    var want = multivariate_normal.logpdf(
        _points(cpu), mean_c, _cov(cpu)
    ).to_host()
    for i in range(m):
        assert_almost_equal(got[i], want[i], atol=1e-3, rtol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
