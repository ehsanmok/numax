"""`Generator.gamma`, `beta`, `poisson` and `binomial` at `gpu=True`: the
same samplers run one thread per element on the device at `float32`, so
their samples are checked against the distributions' moments rather than
the host's draws bit for bit (the device `log` may round differently and
move an acceptance)."""

from std.math import sqrt
from std.testing import TestSuite, assert_false, assert_true

from max.gpu.host import DeviceContext

from numax.stats import Generator

comptime f32 = DType.float32
comptime n = 100000


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
    var v = sq / Float64(len(values) - 1)
    assert_true(abs(m - mean) < 5 * sqrt(variance / Float64(len(values))))
    assert_true(abs(v - variance) < 0.06 * variance)


def test_samplers_on_the_device_have_the_right_moments() raises:
    var gpu = DeviceContext()
    var rng = Generator(seed=21)
    var g = rng.gamma[f32, n, gpu=True](3.5, 2.0, ctx=gpu)
    assert_false(g.on_host())
    _check(g.to_host(), 7.0, 14.0)
    _check(
        rng.beta[f32, n, gpu=True](2.0, 5.0, ctx=gpu).to_host(),
        2.0 / 7.0,
        10.0 / (49.0 * 8.0),
    )
    _check(
        rng.poisson[DType.int32, n, gpu=True](3.0, ctx=gpu).to_host(), 3.0, 3.0
    )
    _check(
        rng.poisson[DType.int32, n, gpu=True](250.0, ctx=gpu).to_host(),
        250.0,
        250.0,
    )
    _check(
        rng.binomial[DType.int32, n, gpu=True](1000, 0.4, ctx=gpu).to_host(),
        400.0,
        240.0,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
