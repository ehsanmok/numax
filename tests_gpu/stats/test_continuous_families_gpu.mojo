"""The 0.3 continuous families' `Tensor` overloads and `rvs` at
`gpu=True`, against the host, at `float32`: `cdf` and `ppf` for each over
the same inputs to `float32` rounding, and `rvs` by its draws' mean."""

from std.math import sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_false,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax.core.tensor import linspace
from numax.stats import (
    Generator,
    cauchy,
    laplace,
    logistic,
    lognorm,
    pareto,
    rayleigh,
    uniform_dist,
    weibull_min,
)

comptime f32 = DType.float32
comptime n = 512


def _close(got: List[Float32], want: List[Float32]) raises:
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=2e-5, rtol=2e-4)


def test_cdf_and_ppf_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var xg = linspace[n, f32](0.1, 6.0, ctx=gpu)
    var xc = linspace[n, f32](0.1, 6.0, ctx=cpu)
    var pg = linspace[n, f32](0.01, 0.99, ctx=gpu)
    var pc = linspace[n, f32](0.01, 0.99, ctx=cpu)
    var d = weibull_min.cdf[gpu=True](xg, 1.8, 2.5)
    assert_false(d.on_host())
    _close(d.to_host(), weibull_min.cdf(xc, 1.8, 2.5).to_host())
    _close(
        lognorm.cdf[gpu=True](xg, 0.6, 2.0).to_host(),
        lognorm.cdf(xc, 0.6, 2.0).to_host(),
    )
    _close(
        cauchy.cdf[gpu=True](xg, 0.5, 2.0).to_host(),
        cauchy.cdf(xc, 0.5, 2.0).to_host(),
    )
    _close(
        logistic.ppf[gpu=True](pg, -1.0, 0.8).to_host(),
        logistic.ppf(pc, -1.0, 0.8).to_host(),
    )
    _close(
        laplace.ppf[gpu=True](pg, 1.0, 1.5).to_host(),
        laplace.ppf(pc, 1.0, 1.5).to_host(),
    )
    _close(
        rayleigh.ppf[gpu=True](pg, 1.7).to_host(),
        rayleigh.ppf(pc, 1.7).to_host(),
    )
    _close(
        pareto.ppf[gpu=True](pg, 2.5, 1.5).to_host(),
        pareto.ppf(pc, 2.5, 1.5).to_host(),
    )
    _close(
        uniform_dist.ppf[gpu=True](pg, 1.0, 3.0).to_host(),
        uniform_dist.ppf(pc, 1.0, 3.0).to_host(),
    )


def test_rvs_on_the_device() raises:
    var gpu = DeviceContext()
    var rng = Generator(seed=51)
    comptime m = 50000
    var draws = weibull_min.rvs[f32, m, gpu=True](1.8, 2.5, rng, gpu).to_host()
    var total = 0.0
    for i in range(m):
        total += Float64(draws[i])
    var mean = weibull_min.mean(1.8, 2.5)
    assert_true(
        abs(total / Float64(m) - mean)
        < 5 * sqrt(weibull_min.var(1.8, 2.5) / Float64(m))
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
