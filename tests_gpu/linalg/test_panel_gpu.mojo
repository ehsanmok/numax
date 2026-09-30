"""`pack_block` at `gpu=True`, against the host, at `float32`, from a corner
that is not a multiple of the vector width.

A wide load claiming more alignment than the block's offset and row stride
give is a `CUDA_ERROR_MISALIGNED_ADDRESS` on NVIDIA and is tolerated on
Metal and the host, so the odd offsets and odd extents here are the point:
each one is misaligned for every vector width, and `cols` is a multiple of
the lane count so the copy actually runs wide."""

from std.testing import TestSuite, assert_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static, zeros
from numax.linalg.panel import pack_block

comptime f32 = DType.float32
comptime n = 37
comptime rows = 17
comptime cols = 16


def _matrix(ctx: DeviceContext) raises -> Static[f32, n, n]:
    var values = List[Scalar[f32]](capacity=n * n)
    for i in range(n * n):
        values.append(Float32(i))
    return Static[f32, n, n](values^, ctx)


def test_pack_block_from_an_odd_corner_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var a = _matrix(gpu)
    var dst = zeros[f32, rows, cols](gpu)
    pack_block[target="gpu"](a.tile(), dst.tile(), 3, 5, rows, cols, gpu)
    gpu.synchronize()
    assert_false(dst.on_host())

    var h = _matrix(cpu)
    var want = zeros[f32, rows, cols](cpu)
    pack_block(h.tile(), want.tile(), 3, 5, rows, cols, cpu)
    var got = dst.to_host()
    var expected = want.to_host()
    for i in range(rows * cols):
        assert_equal(got[i], expected[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
