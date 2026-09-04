"""Tests for `numax.stats`.

`uniform`/`normal`/`exponential` are checked two ways: fixed-seed
reproducibility (two draws separated only by the same `seed(...)` call
must match exactly), and the sample mean/stddev of a large draw landing
within tolerance of the distribution's theoretical moments. `seed` itself
is exercised implicitly by every other test here.
"""

from std.testing import TestSuite, assert_almost_equal, assert_true

from max.gpu.host import DeviceContext

from numax.stats import exponential, normal, seed, uniform

comptime dtype = DType.float32


def test_seed_makes_uniform_draws_reproducible() raises:
    var ctx = DeviceContext(api="cpu")
    seed(7)
    var a = uniform[dtype, 8](0, 1, ctx=ctx)
    seed(7)
    var b = uniform[dtype, 8](0, 1, ctx=ctx)
    for i in range(8):
        assert_almost_equal(a[i], b[i])


def test_seed_makes_normal_draws_reproducible() raises:
    var ctx = DeviceContext(api="cpu")
    seed(11)
    var a = normal[dtype, 8](0, 1, ctx=ctx)
    seed(11)
    var b = normal[dtype, 8](0, 1, ctx=ctx)
    for i in range(8):
        assert_almost_equal(a[i], b[i])


def test_seed_makes_exponential_draws_reproducible() raises:
    var ctx = DeviceContext(api="cpu")
    seed(13)
    var a = exponential[dtype, 8](2, ctx=ctx)
    seed(13)
    var b = exponential[dtype, 8](2, ctx=ctx)
    for i in range(8):
        assert_almost_equal(a[i], b[i])


def test_uniform_draws_land_within_the_requested_range() raises:
    var ctx = DeviceContext(api="cpu")
    seed(1)
    comptime n = 5000
    var xs = uniform[dtype, n](-3, 5, ctx=ctx)
    for i in range(n):
        assert_true(xs[i] >= -3.0 and xs[i] < 5.0)


def test_uniform_sample_mean_matches_the_midpoint_within_tolerance() raises:
    var ctx = DeviceContext(api="cpu")
    seed(2)
    comptime n = 30_000
    var xs = uniform[dtype, n](0, 10, ctx=ctx)
    var total = Float64(0)
    for i in range(n):
        total += Float64(xs[i])
    var sample_mean = total / Float64(n)
    # Theoretical mean of Uniform(0, 10) is 5; the CLT puts the sample mean's
    # own standard deviation at stddev/sqrt(n) ~= 2.89/173 ~= 0.017, so 0.15
    # is a generous multi-sigma tolerance, not a tight bound.
    assert_true(
        abs(sample_mean - 5.0) < 0.15,
        msg=String("uniform sample mean drifted too far: ", sample_mean),
    )


def test_normal_sample_mean_and_stddev_match_the_parameters() raises:
    var ctx = DeviceContext(api="cpu")
    seed(3)
    comptime n = 30_000
    var xs = normal[dtype, n](2, 3, ctx=ctx)
    var total = Float64(0)
    for i in range(n):
        total += Float64(xs[i])
    var sample_mean = total / Float64(n)
    var sq_total = Float64(0)
    for i in range(n):
        var d = Float64(xs[i]) - sample_mean
        sq_total += d * d
    var sample_stddev = (sq_total / Float64(n)) ** 0.5
    assert_true(
        abs(sample_mean - 2.0) < 0.15,
        msg=String("normal sample mean drifted too far: ", sample_mean),
    )
    assert_true(
        abs(sample_stddev - 3.0) < 0.15,
        msg=String("normal sample stddev drifted too far: ", sample_stddev),
    )


def test_exponential_sample_mean_matches_its_scale() raises:
    var ctx = DeviceContext(api="cpu")
    seed(4)
    comptime n = 30_000
    var xs = exponential[dtype, n](2, ctx=ctx)
    var total = Float64(0)
    for i in range(n):
        total += Float64(xs[i])
    var sample_mean = total / Float64(n)
    # Exponential(scale=2) has mean 2 and stddev 2 -- the sample mean's own
    # stddev is 2/sqrt(30_000) ~= 0.0115, so 0.15 is again generous.
    assert_true(
        abs(sample_mean - 2.0) < 0.15,
        msg=String("exponential sample mean drifted too far: ", sample_mean),
    )


def test_exponential_draws_are_never_negative() raises:
    var ctx = DeviceContext(api="cpu")
    seed(5)
    comptime n = 5000
    var xs = exponential[dtype, n](1, ctx=ctx)
    for i in range(n):
        assert_true(xs[i] >= 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
