"""`lu`'s `(P, L, U)` at `gpu=True`, against the host, at `float32`: the
split, the permutation replay and the scatter are device launches over the
device factorization, and `P @ L @ U` reconstructs the input."""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.linalg import lu, matmul

comptime f32 = DType.float32
comptime n = 48


def _a(ctx: DeviceContext) raises -> Static[f32, n, n]:
    # Deterministic and non-symmetric, every entry distinct so no two
    # pivot candidates tie, with rows scrambled enough that partial
    # pivoting swaps.
    var values = List[Scalar[f32]](capacity=n * n)
    for i in range(n * n):
        values.append(sin(Float32(i) * 1.37 + Float32(i % 7) * 0.71))
    return Static[f32, n, n](values^, ctx)


def _close(
    got: List[Scalar[f32]], want: List[Scalar[f32]], atol: Float64
) raises:
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=atol)


def test_lu_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = lu[gpu=True](_a(gpu))
    var h = lu[block=16](_a(cpu))
    assert_false(d.p.on_host())
    _close(d.p.to_host(), h.p.to_host(), 0.0)
    _close(d.l.to_host(), h.l.to_host(), 1e-4)
    _close(d.u.to_host(), h.u.to_host(), 1e-4)
    var back = matmul[gpu=True](matmul[gpu=True](d.p, d.l), d.u).to_host()
    _close(back, _a(cpu).to_host(), 1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
