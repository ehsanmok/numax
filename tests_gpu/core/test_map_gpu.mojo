"""`map[gpu=True]` over `Tensor`s launches itself: the grid is sized in
`numax`, so the call site passes two device tensors and nothing else, and
the answer matches the host walk."""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax import Plain, gaussian
from numax.core.functional import add_step, map
from numax.core.tensor import Static

comptime f32 = DType.float32
comptime n = 1000


def _step[w: Int](x: SIMD[f32, w]) -> SIMD[f32, w]:
    return gaussian(Plain[f32, w](x)).v


def _values(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(i) * 0.01 - 5.0)
    return Static[f32, n](values^, ctx)


def test_map_launches_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var xs = _values(gpu)
    var ys = Static[f32, n](gpu)
    map[step=_step, gpu=True](xs, ys)
    assert_false(ys.on_host())
    var hx = _values(cpu)
    var hy = Static[f32, n](cpu)
    map[step=_step](hx, hy)
    var got = ys.to_host()
    var want = hy.to_host()
    for i in range(n):
        assert_almost_equal(got[i], want[i], atol=1e-6)
    var zs = Static[f32, n](gpu)
    map[width=4, step=add_step[f32, _], gpu=True](xs, xs, zs)
    var gz = zs.to_host()
    var hvals = hx.to_host()
    for i in range(n):
        assert_almost_equal(gz[i], 2 * hvals[i], atol=1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
