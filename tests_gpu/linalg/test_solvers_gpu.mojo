"""`solve_sylvester`, `solve_continuous_lyapunov`,
`solve_discrete_lyapunov` and `solve_continuous_are` at `gpu=True`, against the host, at `float32`:
the Schur forms, the per-column `trsyl_column` launches and the products
all run on the device, on matrices whose Schur forms have `2 x 2` blocks
on both sides."""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static, eye
from numax.linalg import (
    matmul,
    solve_continuous_are,
    solve_continuous_lyapunov,
    solve_discrete_lyapunov,
    solve_sylvester,
)

comptime f32 = DType.float32


def _rot[
    n: Int
](
    ctx: DeviceContext, seed: Float32, shift: Float32, scale: Float32
) raises -> Static[f32, n, n]:
    var values = List[Scalar[f32]](capacity=n * n)
    for i in range(n):
        for j in range(n):
            var skew = Float32(j - i) * 0.7
            var diag = shift if i == j else Float32(0)
            values.append(
                scale * (skew + diag + 0.3 * sin(Float32(i * n + j) * seed))
            )
    return Static[f32, n, n](values^, ctx)


def _rhs[n: Int, m: Int](ctx: DeviceContext) raises -> Static[f32, n, m]:
    var values = List[Scalar[f32]](capacity=n * m)
    for i in range(n * m):
        values.append(sin(Float32(i) * 0.9))
    return Static[f32, n, m](values^, ctx)


def _close(
    got: List[Scalar[f32]], want: List[Scalar[f32]], atol: Float64
) raises:
    # Relative as well as absolute: entries reach 10 here, and the two
    # targets reach different, equally valid Schur forms at `float32`, so
    # the answers differ by that rounding carried through each -- up to a
    # few units in the fourth digit. The residual below is the tight check.
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=atol, rtol=1e-3)


def test_solvers_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    # One size throughout: each distinct size instantiates the device
    # `schur` again, and that is most of this file's compile time.
    var dx = solve_sylvester[gpu=True](
        _rot[10](gpu, 1.3, -2.0, 1.0),
        _rot[10](gpu, 0.7, 1.0, 1.0),
        _rhs[10, 10](gpu),
    )
    assert_false(dx.on_host())
    var hx = solve_sylvester(
        _rot[10](cpu, 1.3, -2.0, 1.0),
        _rot[10](cpu, 0.7, 1.0, 1.0),
        _rhs[10, 10](cpu),
    )
    _close(dx.to_host(), hx.to_host(), 1e-4)
    var lhs = matmul[gpu=True](_rot[10](gpu, 1.3, -2.0, 1.0), dx).to_host()
    var right = matmul[gpu=True](dx, _rot[10](gpu, 0.7, 1.0, 1.0)).to_host()
    var q = _rhs[10, 10](cpu).to_host()
    for i in range(10 * 10):
        assert_almost_equal(lhs[i] + right[i], q[i], atol=5e-4)

    _close(
        solve_continuous_lyapunov[gpu=True](
            _rot[10](gpu, 0.9, -1.5, 1.0), _rhs[10, 10](gpu)
        ).to_host(),
        solve_continuous_lyapunov(
            _rot[10](cpu, 0.9, -1.5, 1.0), _rhs[10, 10](cpu)
        ).to_host(),
        1e-4,
    )
    # A contraction, so the Stein equation's solution is bounded.
    _close(
        solve_discrete_lyapunov[gpu=True](
            _rot[10](gpu, 0.9, 0.0, 0.08), _rhs[10, 10](gpu)
        ).to_host(),
        solve_discrete_lyapunov(
            _rot[10](cpu, 0.9, 0.0, 0.08), _rhs[10, 10](cpu)
        ).to_host(),
        1e-4,
    )


def test_care_on_the_device_matches_the_host() raises:
    """The sign iteration's LUs, products and closing QR on the device:
    the same `X` as the host, to `float32` rounding."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var dx = solve_continuous_are[gpu=True](
        _rot[10](gpu, 0.9, 0.5, 1.0),
        _rot[10](gpu, 0.4, 2.0, 0.5),
        eye[10, f32](ctx=gpu),
        eye[10, f32](ctx=gpu),
    )
    assert_false(dx.on_host())
    var hx = solve_continuous_are(
        _rot[10](cpu, 0.9, 0.5, 1.0),
        _rot[10](cpu, 0.4, 2.0, 0.5),
        eye[10, f32](ctx=cpu),
        eye[10, f32](ctx=cpu),
    )
    _close(dx.to_host(), hx.to_host(), 1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
