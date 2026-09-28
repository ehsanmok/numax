"""The blocked `numax.linalg` factorizations at `gpu=True`, against the
host answer, at `float32` (Metal has no `double`). Needs a real
accelerator; run by `pixi run tests-gpu`.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.linalg import (
    NORM_INF,
    NORM_NEG_INF,
    cholesky,
    matmul,
    norm,
    rq,
    solve,
    solve_triangular,
)

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
    return Static[f32, n, n](values^, ctx)


def _rhs(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(i % 5) - 2.0)
    return Static[f32, n](values^, ctx)


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


def test_the_extremal_norms_on_the_device_match_the_host() raises:
    """The vector infinity norms fold on the device now, not on the host."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    assert_equal(
        norm[ord=NORM_INF, gpu=True](_rhs(gpu)), norm[ord=NORM_INF](_rhs(cpu))
    )
    assert_equal(
        norm[ord=NORM_NEG_INF, gpu=True](_rhs(gpu)),
        norm[ord=NORM_NEG_INF](_rhs(cpu)),
    )


def test_rq_on_the_device_matches_the_host() raises:
    """`rq`'s reversals are device flips now; both factors match the
    host's."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = rq[gpu=True](_spd(gpu))
    var h = rq(_spd(cpu))
    _assert_close(d.r.to_host(), h.r.to_host(), atol=1e-2)
    _assert_close(d.q.to_host(), h.q.to_host(), atol=1e-4)


def test_transposed_triangular_solves_on_the_device_match_the_host() raises:
    """`solve_triangular[trans=True]` against the Cholesky factor, stored
    lower, with a vector and a matrix right-hand side: `L^T x = b` on both
    targets."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var dl = cholesky[gpu=True](_spd(gpu))
    var hl = cholesky(_spd(cpu))
    _assert_close(
        solve_triangular[trans=True, gpu=True](dl, _rhs(gpu)).to_host(),
        solve_triangular[trans=True](hl, _rhs(cpu)).to_host(),
        1e-4,
    )
    _assert_close(
        solve_triangular[trans=True, gpu=True](dl, _spd(gpu)).to_host(),
        solve_triangular[trans=True](hl, _spd(cpu)).to_host(),
        1e-3,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
