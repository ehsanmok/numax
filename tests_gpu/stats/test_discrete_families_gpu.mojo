"""The 0.3 discrete families at `gpu=True`, against the host, at
`float32`: the `pmf` and `cdf` `Tensor` overloads over integer points,
and `nbinom`/`hypergeom` draws by their means."""

from std.math import sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_false,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax.core.tensor import arange_n
from numax.stats import Generator, bernoulli, geom, hypergeom, nbinom

comptime f32 = DType.float32


def _close(got: List[Float32], want: List[Float32]) raises:
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-5, rtol=1e-4)


def test_pmf_and_cdf_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var kg = arange_n[16, f32](ctx=gpu)
    var kc = arange_n[16, f32](ctx=cpu)
    var d = nbinom.pmf[gpu=True](kg, 3.5, 0.4)
    assert_false(d.on_host())
    _close(d.to_host(), nbinom.pmf(kc, 3.5, 0.4).to_host())
    _close(
        nbinom.cdf[gpu=True](kg, 3.5, 0.4).to_host(),
        nbinom.cdf(kc, 3.5, 0.4).to_host(),
    )
    _close(geom.cdf[gpu=True](kg, 0.25).to_host(), geom.cdf(kc, 0.25).to_host())
    _close(
        bernoulli.pmf[gpu=True](kg, 0.3).to_host(),
        bernoulli.pmf(kc, 0.3).to_host(),
    )
    _close(
        hypergeom.pmf[gpu=True](kg, 30.0, 12.0, 8.0).to_host(),
        hypergeom.pmf(kc, 30.0, 12.0, 8.0).to_host(),
    )
    _close(
        hypergeom.cdf[gpu=True](kg, 30.0, 12.0, 8.0).to_host(),
        hypergeom.cdf(kc, 30.0, 12.0, 8.0).to_host(),
    )


def test_rvs_on_the_device() raises:
    var gpu = DeviceContext()
    var rng = Generator(seed=71)
    comptime n = 50000
    var a = nbinom.rvs[DType.int32, n, gpu=True](3.5, 0.4, rng, gpu).to_host()
    var b = hypergeom.rvs[DType.int32, n, gpu=True](
        30, 12, 8, rng, gpu
    ).to_host()
    var sa = 0.0
    var sb = 0.0
    for i in range(n):
        sa += Float64(a[i])
        sb += Float64(b[i])
    assert_true(
        abs(sa / Float64(n) - nbinom.mean(3.5, 0.4))
        < 5 * sqrt(nbinom.var(3.5, 0.4) / Float64(n))
    )
    assert_true(
        abs(sb / Float64(n) - hypergeom.mean(30.0, 12.0, 8.0))
        < 5 * sqrt(hypergeom.var(30.0, 12.0, 8.0) / Float64(n))
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
