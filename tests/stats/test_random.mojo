"""Tests for `numax.stats.random`.

`uniform`/`normal`/`exponential` are checked two ways: fixed-seed
reproducibility (two draws separated only by the same `seed(...)` call
must match exactly), and the sample mean/stddev of a large draw landing
within tolerance of the distribution's theoretical moments. `seed` itself
is exercised implicitly by every other test here. The stream layout the
module docstring promises -- element `i` is word `i` of
`Random(seed, offset=i // 4).step()` -- is pinned against `std.random`'s
Philox directly, which is also what makes the host and device fills agree
bit for bit (the device half of that claim is checked on Metal by hand,
since this suite runs on the CPU).
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from std.random import Random

from numax.stats import (
    Generator,
    exponential,
    normal,
    randbool,
    randint,
    seed,
    uniform,
)

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


def test_two_generators_with_one_seed_agree() raises:
    var a = Generator(seed=7)
    var b = Generator(seed=7)
    var xs = a.uniform[dtype, 16]().to_host()
    var ys = b.uniform[dtype, 16]().to_host()
    for i in range(16):
        assert_equal(xs[i], ys[i])


def test_a_generator_advances_between_draws() raises:
    var g = Generator(seed=7)
    var first = g.uniform[dtype, 16]().to_host()
    var second = g.uniform[dtype, 16]().to_host()
    var identical = True
    for i in range(16):
        if first[i] != second[i]:
            identical = False
    assert_true(not identical, msg="two draws should not repeat one stream")


def test_a_generator_is_unaffected_by_the_global_seed() raises:
    # The property the module-level functions cannot offer.
    var g = Generator(seed=7)
    var expected = g.uniform[dtype, 8]().to_host()

    var h = Generator(seed=7)
    seed(999)
    _ = uniform[dtype, 8]()
    var actual = h.uniform[dtype, 8]().to_host()
    for i in range(8):
        assert_equal(actual[i], expected[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_the_stream_layout_is_philox_word_i() raises:
    """A `float32` element `i` is the top 24 bits of word `i % 4` of Philox
    step `i // 4`, scaled by `2^-24`; a `float64` element takes the two
    words of pair `i % 2` of step `i // 2`."""
    var rng = Generator(seed=99)
    var xs = rng.uniform[DType.float32, 12](0, 1).to_host()
    for i in range(12):
        var r = Random(seed=UInt64(99), offset=UInt64(i // 4))
        var word = r.step()[i % 4]
        var want = Float32(word >> 8) * 5.960464477539063e-08
        assert_equal(xs[i], want)
        assert_true(xs[i] >= 0.0 and xs[i] < 1.0)
    var rng64 = Generator(seed=99)
    var ys = rng64.uniform[DType.float64, 6](0, 1).to_host()
    for i in range(6):
        var r = Random(seed=UInt64(99), offset=UInt64(i // 2))
        var words = r.step()
        var pair = 2 * (i % 2)
        var bits = (UInt64(words[pair]) << 32) | UInt64(words[pair + 1])
        var want = Float64(bits >> 11) * 1.1102230246251565e-16
        assert_equal(ys[i], want)


def test_a_generator_ignores_the_global_seed() raises:
    seed(1)
    var ga = Generator(seed=5)
    var a = ga.uniform[dtype, 8]().to_host()
    seed(2)
    var gb = Generator(seed=5)
    var b = gb.uniform[dtype, 8]().to_host()
    for i in range(8):
        assert_equal(a[i], b[i])
    var gc = Generator(seed=6)
    var c = gc.uniform[dtype, 8]().to_host()
    var differ = 0
    for i in range(8):
        if a[i] != c[i]:
            differ += 1
    assert_true(differ >= 7, msg="two seeds gave the same stream")


def test_multi_dimensional_shapes_fill_every_element() raises:
    var ctx = DeviceContext(api="cpu")
    var rng = Generator(seed=3)
    var xs = rng.uniform[dtype, 4, 8](10, 11, ctx=ctx)
    assert_equal(xs.size(), 32)
    var flat = xs.to_host()
    for i in range(32):
        assert_true(flat[i] >= 10.0 and flat[i] < 11.0)


def test_randint_is_integral_and_in_range() raises:
    var ctx = DeviceContext(api="cpu")
    seed(8)
    comptime n = 4000
    var xs = randint[DType.int32, n](-3, 4, ctx=ctx).to_host()
    var seen = List[Int](length=7, fill=0)
    for i in range(n):
        var v = Int(xs[i])
        assert_true(v >= -3 and v < 4)
        seen[v + 3] += 1
    for k in range(7):
        # Each of the seven values should appear about n/7 ~= 571 times.
        assert_true(
            seen[k] > 400 and seen[k] < 750,
            msg=String("bucket ", k, ": ", seen[k]),
        )
    var fs = randint[DType.float64, 8](0, 100, ctx=ctx).to_host()
    for i in range(8):
        assert_equal(fs[i], Float64(Int(fs[i])))


def test_randbool_hits_its_probability() raises:
    var ctx = DeviceContext(api="cpu")
    seed(9)
    comptime n = 20_000
    var xs = randbool[DType.bool, n](0.3, ctx=ctx).to_host()
    var trues = 0
    for i in range(n):
        if xs[i]:
            trues += 1
    var fraction = Float64(trues) / Float64(n)
    # stddev of the fraction is sqrt(0.3 * 0.7 / 20000) ~= 0.0032.
    assert_true(abs(fraction - 0.3) < 0.02, msg=String("fraction ", fraction))
