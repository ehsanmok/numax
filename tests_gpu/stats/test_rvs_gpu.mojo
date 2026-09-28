"""`rvs` at `gpu=True`: `norm`, `gamma` and `binom` through `Generator`'s
device samplers, and `t` by inverting its `ppf` on the device, each
checked by the moments of its draws at `float32`."""

from std.math import sqrt
from std.testing import TestSuite, assert_false, assert_true

from max.gpu.host import DeviceContext

from numax.stats import Generator, binom, gamma, norm, t

comptime f32 = DType.float32
comptime n = 50000


def _check[
    dtype: DType
](values: List[Scalar[dtype]], mean: Float64, variance: Float64) raises:
    var total = 0.0
    for i in range(len(values)):
        total += Float64(values[i])
    var m = total / Float64(len(values))
    var sq = 0.0
    for i in range(len(values)):
        sq += (Float64(values[i]) - m) ** 2
    assert_true(abs(m - mean) < 5 * sqrt(variance / Float64(len(values))))
    assert_true(abs(sq / Float64(len(values) - 1) - variance) < 0.08 * variance)


def test_rvs_on_the_device() raises:
    var gpu = DeviceContext()
    var rng = Generator(seed=41)
    var x = norm.rvs[f32, n, gpu=True](1.0, 2.0, rng, gpu)
    assert_false(x.on_host())
    _check(x.to_host(), 1.0, 4.0)
    _check(gamma.rvs[f32, n, gpu=True](2.0, 3.0, rng, gpu).to_host(), 6.0, 18.0)
    _check(
        binom.rvs[DType.int32, n, gpu=True](30.0, 0.4, rng, gpu).to_host(),
        12.0,
        7.2,
    )
    _check(t.rvs[f32, n, gpu=True](9.0, rng, gpu).to_host(), 0.0, 9.0 / 7.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
