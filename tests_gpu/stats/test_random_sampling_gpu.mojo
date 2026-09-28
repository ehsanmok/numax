"""`permutation`, weighted `choice` and `multivariate_normal` at
`gpu=True`: draws, sorts, gathers and the Cholesky product on the device,
checked by the properties they must have -- every index once, each
weight's frequency, the requested moments -- rather than bit for bit."""

from std.math import sqrt
from std.testing import TestSuite, assert_false, assert_true

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import Generator

comptime f32 = DType.float32


def test_sampling_on_the_device() raises:
    var gpu = DeviceContext()
    var rng = Generator(seed=31)
    var p = rng.permutation[4096, gpu=True](gpu)
    assert_false(p.on_host())
    var h = p.to_host()
    var seen = List[Bool](length=4096, fill=False)
    for i in range(4096):
        assert_true(not seen[Int(h[i])])
        seen[Int(h[i])] = True
    var a = Static[f32, 3]([10.0, 20.0, 30.0], gpu)
    var w = Static[f32, 3]([0.2, 0.5, 0.3], gpu)
    comptime n = 50000
    var draws = rng.choice[size=n, gpu=True](a, w).to_host()
    var counts = List[Int](length=3, fill=0)
    for i in range(n):
        counts[Int(draws[i] / 10) - 1] += 1
    var want = [0.2, 0.5, 0.3]
    for k in range(3):
        var freq = Float64(counts[k]) / Float64(n)
        assert_true(
            abs(freq - want[k]) < 5 * sqrt(want[k] * (1 - want[k]) / Float64(n))
        )
    var mean = Static[f32, 2]([1.0, -2.0], gpu)
    var cov = Static[f32, 2, 2]([2.0, 0.8, 0.8, 1.0], gpu)
    var x = rng.multivariate_normal[count=n, gpu=True](mean, cov).to_host()
    var m0 = 0.0
    var c01 = 0.0
    for i in range(n):
        m0 += Float64(x[2 * i])
    m0 /= Float64(n)
    for i in range(n):
        c01 += (Float64(x[2 * i]) - m0) * (Float64(x[2 * i + 1]) + 2.0)
    c01 /= Float64(n - 1)
    assert_true(abs(m0 - 1.0) < 0.03)
    assert_true(abs(c01 - 0.8) < 0.05)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
