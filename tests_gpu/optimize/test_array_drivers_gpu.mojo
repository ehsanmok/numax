"""`numax.optimize`'s `Array` tier's converge-to-tolerance drivers inside a device
kernel body at `float32`, against the same kernel on the CPU.

Every lane of one launch solves its own problem -- the root of `x^2 - c`
for its own `c`, the minimum of a shifted parabola and of Rosenbrock, a
two-parameter fit -- with each driver at `dtype=float32`, so the test
checks the drivers compile for the device (Metal rejects any `double`),
converge at `float32`'s default tolerances, and agree with the host.
"""

from std.collections import Array
from std.testing import TestSuite, assert_almost_equal

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from numax.core.numeric import FloatLike
from numax.core.tensor import Static
from numax.optimize import (
    bfgs,
    bisect_tol,
    brent,
    brentq,
    cg,
    curve_fit,
    fminbound,
    golden,
    halley_tol,
    least_squares,
    nelder_mead,
    newton_tol,
    secant,
)

comptime f32 = DType.float32
comptime lanes = 32
comptime columns = 13


def _two_minus[U: FloatLike](x: U) -> U:
    return x * x - U.constant(2.0)


def _parabola[U: FloatLike](x: U) -> U:
    var d = x - U.constant(3.0)
    return d * d + U.constant(2.0)


def _bowl[U: FloatLike](v: Array[U, 2]) -> U:
    var a = v[0] - U.constant(1.0)
    var b = v[1] + U.constant(2.0)
    return a * a + U.constant(4.0) * b * b


def _line_residuals[U: FloatLike](p: Array[U, 2]) -> Array[U, 3]:
    var r = Array[U, 3](fill=U.constant(0.0))
    comptime for i in range(3):
        var x = U.constant(Float64(i))
        r[i] = p[0] * x + p[1] - (U.constant(2.0) * x + U.constant(1.0))
    return r^


def _line[U: FloatLike](x: U, p: Array[U, 2]) -> U:
    return p[0] * x + p[1]


def _run[target: StaticString](ctx: DeviceContext) raises -> List[Scalar[f32]]:
    var out = Static[f32, lanes, columns]._uninitialized(ctx)
    var ov = out.tile()

    @always_inline
    def body[width: Int, alignment: Int = 1](coord: Coord) {var ov}:
        var lane = coord_to_index_list(coord)[0]
        var shift = Float32(lane) * 0.01
        var ok = Float32(1)
        var r1 = brentq[_two_minus, dtype=f32](Float32(0.0), Float32(3.0))
        var r2 = newton_tol[_two_minus, dtype=f32](Float32(1.0) + shift)
        var r3 = halley_tol[_two_minus, dtype=f32](Float32(1.0) + shift)
        var r4 = secant[_two_minus, dtype=f32](Float32(1.0) + shift)
        var r5 = bisect_tol[_two_minus, dtype=f32](Float32(0.0), Float32(3.0))
        var m1 = brent[_parabola, dtype=f32]()
        var m2 = golden[_parabola, dtype=f32]()
        var m3 = fminbound[_parabola, dtype=f32](Float32(0.0), Float32(5.0))
        var start = Array[Float32, 2](fill=shift)
        var v1 = bfgs[2, _bowl, dtype=f32](start.copy())
        var v2 = cg[2, _bowl, dtype=f32](start.copy())
        var v3 = nelder_mead[2, _bowl, dtype=f32](start.copy())
        var v4 = least_squares[2, 3, _line_residuals, dtype=f32](start.copy())
        var xs = Array[Float32, 3](fill=0)
        var ys = Array[Float32, 3](fill=0)
        for i in range(3):
            xs[i] = Float32(i)
            ys[i] = 2 * Float32(i) + 1
        var v5 = curve_fit[2, 3, _line, dtype=f32](xs, ys, start.copy())
        for converged in [
            r1.converged,
            r2.converged,
            r3.converged,
            r4.converged,
            r5.converged,
            m1.converged,
            m2.converged,
            m3.converged,
            v1.converged,
            v2.converged,
            v3.converged,
            v4.converged,
            v5.converged,
        ]:
            if not converged:
                ok = 0
        ov.store[1](Coord(lane, 0), r1.x)
        ov.store[1](Coord(lane, 1), r2.x)
        ov.store[1](Coord(lane, 2), r3.x)
        ov.store[1](Coord(lane, 3), r4.x)
        ov.store[1](Coord(lane, 4), r5.x)
        ov.store[1](Coord(lane, 5), m1.x)
        ov.store[1](Coord(lane, 6), m2.x)
        ov.store[1](Coord(lane, 7), m3.x)
        ov.store[1](Coord(lane, 8), v1.x[1])
        ov.store[1](Coord(lane, 9), v2.x[1])
        ov.store[1](Coord(lane, 10), v4.x[0])
        ov.store[1](Coord(lane, 11), v5.x[1])
        ov.store[1](Coord(lane, 12), ok)

    elementwise[simd_width=1, target=target](body, Coord(lanes), ctx)
    ctx.synchronize()
    return out.to_host()


def test_array_drivers_run_inside_a_device_kernel() raises:
    var d = _run["gpu"](DeviceContext())
    var h = _run["cpu"](DeviceContext(api="cpu"))
    var want: List[Float64] = [
        1.4142135,
        1.4142135,
        1.4142135,
        1.4142135,
        1.4142135,
        3.0,
        3.0,
        3.0,
        -2.0,
        -2.0,
        2.0,
        1.0,
        1.0,
    ]
    for lane in range(lanes):
        for c in range(columns):
            var at = lane * columns + c
            assert_almost_equal(
                Float64(d[at]), Float64(h[at]), atol=2e-3, rtol=1e-3
            )
            assert_almost_equal(Float64(d[at]), want[c], atol=2e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
