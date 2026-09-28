"""The integer surface of `numax.core.ops` at `gpu=True`, against the host.

Integer results are exact on both targets, so the comparison is
equality: the bitwise operations and shifts at `int32`, and `gcd`/`lcm`
over a spread of signed pairs including zeros -- the fixed Euclid step
count is what lets `gcd` launch unchanged, so it is the one to watch.
"""

from std.testing import TestSuite, assert_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.core.ops import (
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    gcd,
    lcm,
    left_shift,
    right_shift,
)

comptime i32 = DType.int32
comptime n = 257


def _values(ctx: DeviceContext, seed: Int) raises -> Static[i32, n]:
    var values = List[Scalar[i32]](capacity=n)
    for i in range(n):
        values.append(Int32(((i * 7919 + seed) % 2003) - 1001))
    return Static[i32, n](values^, ctx)


def _counts(ctx: DeviceContext) raises -> Static[i32, n]:
    var values = List[Scalar[i32]](capacity=n)
    for i in range(n):
        values.append(Int32(i % 20))
    return Static[i32, n](values^, ctx)


def test_bitwise_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = bitwise_and[gpu=True](_values(gpu, 1), _values(gpu, 5))
    assert_false(d.on_host())
    var got = d.to_host()
    var want = bitwise_and(_values(cpu, 1), _values(cpu, 5)).to_host()
    var got_or = bitwise_or[gpu=True](
        _values(gpu, 1), _values(gpu, 5)
    ).to_host()
    var want_or = bitwise_or(_values(cpu, 1), _values(cpu, 5)).to_host()
    var got_xor = bitwise_xor[gpu=True](
        _values(gpu, 1), Scalar[i32](0x5A5A)
    ).to_host()
    var want_xor = bitwise_xor(_values(cpu, 1), Scalar[i32](0x5A5A)).to_host()
    for i in range(n):
        assert_equal(got[i], want[i])
        assert_equal(got_or[i], want_or[i])
        assert_equal(got_xor[i], want_xor[i])


def test_shifts_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var left = left_shift[gpu=True](_values(gpu, 3), _counts(gpu)).to_host()
    var want_left = left_shift(_values(cpu, 3), _counts(cpu)).to_host()
    var right = right_shift[gpu=True](_values(gpu, 3), _counts(gpu)).to_host()
    var want_right = right_shift(_values(cpu, 3), _counts(cpu)).to_host()
    for i in range(n):
        assert_equal(left[i], want_left[i])
        assert_equal(right[i], want_right[i])


def test_gcd_and_lcm_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var g = gcd[gpu=True](_values(gpu, 2), _values(gpu, 11))
    assert_false(g.on_host())
    var got = g.to_host()
    var want = gcd(_values(cpu, 2), _values(cpu, 11)).to_host()
    var got_l = lcm[gpu=True](_values(gpu, 2), _counts(gpu)).to_host()
    var want_l = lcm(_values(cpu, 2), _counts(cpu)).to_host()
    for i in range(n):
        assert_equal(got[i], want[i])
        assert_equal(got_l[i], want_l[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
