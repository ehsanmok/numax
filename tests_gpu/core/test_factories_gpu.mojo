"""The generator factories fill on the device when handed a GPU context.

`eye`, `identity`, `tri`, `linspace`, `logspace`, `arange_n` and `arange`
used to build a host `List` and upload it; on a GPU context they now fill
the buffer where it lives. Each result must stay on the device and match
the host factory: exactly for the integer-valued ones, to a float32 ulp or
two for `linspace` and `logspace`, whose step and power are rounded in a
different order than the host's.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
)

from max.gpu.host import DeviceContext

from numax.core.tensor import (
    arange,
    arange_n,
    eye,
    identity,
    linspace,
    logspace,
    tri,
)

comptime f32 = DType.float32


def _assert_close(
    got: List[Scalar[f32]], want: List[Scalar[f32]], rtol: Float64
) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-6, rtol=rtol)


def test_eye_identity_and_tri_fill_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var e = eye[7, f32](gpu)
    assert_false(e.on_host())
    _assert_close(e.to_host(), eye[7, f32](cpu).to_host(), 0)
    _assert_close(
        identity[5, f32](gpu).to_host(), identity[5, f32](cpu).to_host(), 0
    )
    var t = tri[f32, 6](gpu)
    assert_false(t.on_host())
    _assert_close(t.to_host(), tri[f32, 6](cpu).to_host(), 0)


def test_the_ranges_fill_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var l = linspace[101, f32](-2.0, 3.0, ctx=gpu)
    assert_false(l.on_host())
    _assert_close(
        l.to_host(), linspace[101, f32](-2.0, 3.0, ctx=cpu).to_host(), 1e-6
    )
    _assert_close(
        logspace[33, f32](-1.0, 2.0, 2.5, ctx=gpu).to_host(),
        logspace[33, f32](-1.0, 2.0, 2.5, ctx=cpu).to_host(),
        1e-5,
    )
    var n = arange_n[50, f32](3.0, 0.5, ctx=gpu)
    assert_false(n.on_host())
    _assert_close(
        n.to_host(), arange_n[50, f32](3.0, 0.5, ctx=cpu).to_host(), 0
    )
    var r = arange[f32](1.0, 9.0, 0.75, ctx=gpu)
    assert_false(r.on_host())
    _assert_close(
        r.to_host(), arange[f32](1.0, 9.0, 0.75, ctx=cpu).to_host(), 0
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
