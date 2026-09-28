"""The blocked `numax.linalg` factorizations at `gpu=True`, against the
host answer, at `float32` (Metal has no `double`). Needs a real
accelerator; run by `pixi run tests-gpu`.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.linalg import cholesky, matmul, solve

comptime f32 = DType.float32
comptime n = 96


def _spd(ctx: DeviceContext) raises -> Static[f32, n, n]:
    """`M M^T + n I` for a fixed, non-symmetric `M`: symmetric positive
    definite and comfortably conditioned."""
    var m = List[Float32](capacity=n * n)
    for i in range(n * n):
        m.append(Float32((i * 7919) % 97) / 97.0 - 0.5)
    var values = List[Scalar[f32]](capacity=n * n)
    for r in range(n):
        for c in range(n):
            var acc = Float32(n) if r == c else Float32(0)
            for k in range(n):
                acc += m[r * n + k] * m[c * n + k]
            values.append(acc)
    return Static[f32, n, n](ctx, values^)


def _rhs(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(i % 5) - 2.0)
    return Static[f32, n](ctx, values^)


def _assert_close(
    got: List[Scalar[f32]], want: List[Scalar[f32]], atol: Float64
) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=atol)


def test_matmul_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    _assert_close(
        matmul[gpu=True](_spd(gpu), _spd(gpu)).to_host(),
        matmul(_spd(cpu), _spd(cpu)).to_host(),
        atol=1e-1,
    )


def test_cholesky_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    _assert_close(
        cholesky[gpu=True](_spd(gpu)).to_host(),
        cholesky(_spd(cpu)).to_host(),
        atol=1e-4,
    )


def test_solve_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    _assert_close(
        solve[gpu=True](_spd(gpu), _rhs(gpu)).to_host(),
        solve(_spd(cpu), _rhs(cpu)).to_host(),
        atol=1e-5,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
