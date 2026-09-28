"""`logaddexp`, `logaddexp2`, `sinc`, `heaviside` and `nan_to_num` at
`gpu=True`, against the host, at `float32`.

The device math library and the host one may differ in the last place,
so the transcendental three are held to a few `float32` ulp; `heaviside`
and `nan_to_num` only select and copy, so they must agree exactly,
NaN and the infinities included.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.testing import assert_false, assert_true
from std.utils.numerics import inf, nan, neg_inf

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.core.elementwise import (
    heaviside,
    logaddexp,
    logaddexp2,
    nan_to_num,
    sinc,
)

comptime f32 = DType.float32
comptime n = 300


def _xs(ctx: DeviceContext, shift: Float32) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        if i % 50 == 7:
            values.append(nan[f32]())
        elif i % 50 == 11:
            values.append(inf[f32]())
        elif i % 50 == 13:
            values.append(neg_inf[f32]())
        elif i % 25 == 3:
            values.append(0.0)
        else:
            values.append(Float32(i % 37) * 0.37 - 6.0 + shift)
    return Static[f32, n](values^, ctx)


def _close(got: Float32, want: Float32) raises:
    if want != want:
        assert_true(got != got)
    elif want == inf[f32]() or want == neg_inf[f32]():
        assert_equal(got, want)
    else:
        assert_almost_equal(got, want, atol=1e-5, rtol=1e-5)


def test_logaddexp_pair_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = logaddexp[gpu=True](_xs(gpu, 0.0), _xs(gpu, 1.5))
    assert_false(d.on_host())
    var got = d.to_host()
    var want = logaddexp(_xs(cpu, 0.0), _xs(cpu, 1.5)).to_host()
    var got2 = logaddexp2[gpu=True](_xs(gpu, 0.0), _xs(gpu, -2.0)).to_host()
    var want2 = logaddexp2(_xs(cpu, 0.0), _xs(cpu, -2.0)).to_host()
    for i in range(n):
        _close(got[i], want[i])
        _close(got2[i], want2[i])


def test_sinc_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var got = sinc[gpu=True](_xs(gpu, 0.0)).to_host()
    var want = sinc(_xs(cpu, 0.0)).to_host()
    for i in range(n):
        _close(got[i], want[i])


def test_heaviside_and_nan_to_num_on_the_device_are_exact() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var h0g = Static[f32, 1]([0.5], gpu)
    var h0c = Static[f32, 1]([0.5], cpu)
    var step = heaviside[gpu=True](_xs(gpu, 0.0), h0g).to_host()
    var want_step = heaviside(_xs(cpu, 0.0), h0c).to_host()
    var clean = nan_to_num[gpu=True](
        _xs(gpu, 0.0), -1.0, Float32(7.0)
    ).to_host()
    var want_clean = nan_to_num(_xs(cpu, 0.0), -1.0, Float32(7.0)).to_host()
    for i in range(n):
        if want_step[i] != want_step[i]:
            assert_true(step[i] != step[i])
        else:
            assert_equal(step[i], want_step[i])
        assert_equal(clean[i], want_clean[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
