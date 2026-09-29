"""`solve_bvp` at `gpu=True`, against the host, at `float32`: Bratu's
problem from the upper guess, with `fun` and `bc` written as device
kernels, so the collocation, the difference Jacobians, the Newton system
and its factorization, the residual estimate and the mesh refinement all
run on the device."""

from std.math import exp
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.testing import assert_false

from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from numax.core.tensor import Dynamic, Static, _dyn_shape
from numax.integrate import solve_bvp

comptime f32 = DType.float32


def bratu(
    x: Dynamic[f32, 1], y: Dynamic[f32, 2], ctx: DeviceContext
) raises -> Dynamic[f32, 2]:
    var m = y.dim[1]()
    var out = Dynamic[f32, 2](row_major(_dyn_shape[2](2, m)), ctx)
    var yp = y.tile()
    var op = out.tile()

    @always_inline
    def body[w: Int, alignment: Int = 1](coord: Coord) {var yp, var op, var m}:
        var j = coord_to_index_list(coord)[0]
        op.ptr[unsafe_offset=j] = yp.ptr[unsafe_offset=m + j]
        op.ptr[unsafe_offset=m + j] = -exp(yp.ptr[unsafe_offset=j])

    if y.on_host():
        elementwise[simd_width=1, target="cpu"](body, Coord(m), ctx)
    else:
        elementwise[simd_width=1, target="gpu"](body, Coord(m), ctx)
    ctx.synchronize()
    return out^


def ends_zero(
    ya: Dynamic[f32, 1], yb: Dynamic[f32, 1], ctx: DeviceContext
) raises -> Dynamic[f32, 1]:
    var out = Dynamic[f32, 1](row_major(_dyn_shape[1](2)), ctx)
    var ap = ya.tile()
    var bp = yb.tile()
    var op = out.tile()

    @always_inline
    def body[w: Int, alignment: Int = 1](coord: Coord) {var ap, var bp, var op}:
        op.ptr[unsafe_offset=0] = ap.ptr[unsafe_offset=0]
        op.ptr[unsafe_offset=1] = bp.ptr[unsafe_offset=0]

    if ya.on_host():
        elementwise[simd_width=1, target="cpu"](body, Coord(1), ctx)
    else:
        elementwise[simd_width=1, target="gpu"](body, Coord(1), ctx)
    ctx.synchronize()
    return out^


def _problem(
    ctx: DeviceContext,
) raises -> Tuple[Dynamic[f32, 1], Dynamic[f32, 2]]:
    var x = Dynamic[f32, 1](row_major(_dyn_shape[1](5)), ctx)
    x.copy_from_host([0.0, 0.25, 0.5, 0.75, 1.0])
    var y = Dynamic[f32, 2](row_major(_dyn_shape[2](2, 5)), ctx)
    y.copy_from_host([3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return (x^, y^)


def test_bvp_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var dp = _problem(gpu)
    var d = solve_bvp[fun=bratu, bc=ends_zero, gpu=True](dp[0], dp[1])
    var hp = _problem(cpu)
    var h = solve_bvp[fun=bratu, bc=ends_zero](hp[0], hp[1])
    assert_false(d.y.on_host())
    assert_equal(d.status, 0)
    assert_equal(d.x.size(), h.x.size())
    var q = Static[f32, 3]([0.25, 0.5, 0.75], gpu)
    var got = d.sol(q).to_host()
    var qh = Static[f32, 3]([0.25, 0.5, 0.75], cpu)
    var want = h.sol(qh).to_host()
    for i in range(3):
        assert_almost_equal(got[i], want[i], atol=1e-3)
    # SciPy's float64 upper solution at the midpoint.
    assert_almost_equal(got[1], 4.091478915149736, atol=1e-2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
