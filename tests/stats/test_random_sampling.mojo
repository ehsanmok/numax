"""Tests for `Generator.permutation`, `shuffle`, `choice`, `spawn` and
`multivariate_normal`.

A permutation holds every index once; a shuffle keeps its multiset;
`choice` without replacement never repeats and with weights draws each
element at its weight's frequency; spawned children differ from one
another and are reproducible from the parent's seed; and multivariate
normal draws have the requested mean and covariance to a few standard
errors.
"""

from std.math import sqrt
from std.testing import TestSuite, assert_equal, assert_true

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import Generator

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def test_permutation_holds_every_index_once() raises:
    var rng = Generator(seed=1)
    var p = rng.permutation[1000]().to_host()
    var seen = List[Bool](length=1000, fill=False)
    for i in range(1000):
        assert_true(not seen[Int(p[i])])
        seen[Int(p[i])] = True
    var moved = 0
    for i in range(1000):
        if Int(p[i]) != i:
            moved += 1
    assert_true(moved > 900)


def test_shuffle_keeps_the_multiset() raises:
    var rng = Generator(seed=2)
    var values = List[Scalar[f64]](capacity=100)
    for i in range(100):
        values.append(Float64(i % 10))
    var x = Static[f64, 100](values^, _cpu())
    rng.shuffle(x)
    var counts = List[Int](length=10, fill=0)
    var h = x.to_host()
    for i in range(100):
        counts[Int(h[i])] += 1
    for k in range(10):
        assert_equal(counts[k], 10)


def test_choice_with_and_without_replacement() raises:
    var rng = Generator(seed=3)
    var values = List[Scalar[f64]](capacity=50)
    for i in range(50):
        values.append(Float64(i))
    var a = Static[f64, 50](values^, _cpu())
    var once = rng.choice[size=50, replace=False](a).to_host()
    var seen = List[Bool](length=50, fill=False)
    for i in range(50):
        assert_true(not seen[Int(once[i])])
        seen[Int(once[i])] = True
    var many = rng.choice[size=200](a).to_host()
    for i in range(200):
        assert_true(many[i] >= 0 and many[i] < 50)


def test_weighted_choice_follows_the_weights() raises:
    var rng = Generator(seed=4)
    var a = Static[f64, 3]([10.0, 20.0, 30.0], _cpu())
    var p = Static[f64, 3]([0.2, 0.5, 0.3], _cpu())
    comptime n = 100000
    var draws = rng.choice[size=n](a, p).to_host()
    var counts = List[Int](length=3, fill=0)
    for i in range(n):
        counts[Int(draws[i] / 10) - 1] += 1
    var want = [0.2, 0.5, 0.3]
    for k in range(3):
        var freq = Float64(counts[k]) / Float64(n)
        var se = sqrt(want[k] * (1 - want[k]) / Float64(n))
        assert_true(abs(freq - want[k]) < 5 * se)


def test_spawn_is_independent_and_reproducible() raises:
    var parent = Generator(seed=9)
    var kids = parent.spawn(3)
    var twin = Generator(seed=9)
    var again = twin.spawn(3)
    var x0 = kids[0].uniform[f64, 4]().to_host()
    var x1 = kids[1].uniform[f64, 4]().to_host()
    var y0 = again[0].uniform[f64, 4]().to_host()
    for i in range(4):
        assert_equal(x0[i], y0[i])
    assert_true(x0[0] != x1[0])


def test_multivariate_normal_has_the_requested_moments() raises:
    var rng = Generator(seed=5)
    var mean = Static[f64, 2]([1.0, -2.0], _cpu())
    var cov = Static[f64, 2, 2]([2.0, 0.8, 0.8, 1.0], _cpu())
    comptime n = 100000
    var x = rng.multivariate_normal[count=n](mean, cov).to_host()
    var m0 = 0.0
    var m1 = 0.0
    for i in range(n):
        m0 += x[2 * i]
        m1 += x[2 * i + 1]
    m0 /= Float64(n)
    m1 /= Float64(n)
    var c00 = 0.0
    var c01 = 0.0
    var c11 = 0.0
    for i in range(n):
        var a = x[2 * i] - m0
        var b = x[2 * i + 1] - m1
        c00 += a * a
        c01 += a * b
        c11 += b * b
    c00 /= Float64(n - 1)
    c01 /= Float64(n - 1)
    c11 /= Float64(n - 1)
    assert_true(abs(m0 - 1.0) < 5 * sqrt(2.0 / Float64(n)))
    assert_true(abs(m1 + 2.0) < 5 * sqrt(1.0 / Float64(n)))
    assert_true(abs(c00 - 2.0) < 0.05)
    assert_true(abs(c01 - 0.8) < 0.03)
    assert_true(abs(c11 - 1.0) < 0.03)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
