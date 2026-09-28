"""The `Array` tier inside a device kernel body, against the same kernel
on the CPU.

Every lane of one `elementwise` launch builds its own 3x3 matrix and runs
the register-resident routines on it: the tier-2 `lu_factor` and its
`solve`, `inverse`, `pinv`, `matrix_rank`, `expm`, `sqrtm`, `eigvals`,
`eigh`, and alongside them `rk4`, `dopri5_step`, `firwin` and `hann`.
The same body at `target="cpu"` is the reference, so the test checks
that each routine compiles for the device (Metal rejects any `double`)
and agrees there at `float32`.
"""

from std.collections import Array
from std.testing import TestSuite, assert_almost_equal

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from numax.core.numeric import FloatLike
from numax.core.plain import Plain
from numax.core.tensor import Static
from numax.integrate import dopri5_step, rk4
from numax.linalg.array import (
    eigh,
    eigvals,
    expm,
    inverse,
    lu_factor,
    matrix_rank,
    pinv,
    sqrtm,
)
from numax.signal import firwin, hann

comptime f32 = DType.float32
comptime P = Plain[f32, 1]
comptime lanes = 64
comptime columns = 14


def _decay[U: FloatLike](t: U, y: U) -> U:
    return -y


def _run[target: StaticString](ctx: DeviceContext) raises -> List[Scalar[f32]]:
    var out = Static[f32, lanes, columns]._uninitialized(ctx)
    var ov = out.tile()

    @always_inline
    def body[width: Int, alignment: Int = 1](coord: Coord) {var ov}:
        var lane = coord_to_index_list(coord)[0]
        var a = Array[P, 9](fill=P(SIMD[f32, 1](0)))
        for i in range(9):
            var v = Float32((i * 7 + lane * 3) % 11) - 5.0
            if i % 4 == 0:
                v += 12.0
            a[i] = P(SIMD[f32, 1](v * 0.1))
        var s = Array[P, 9](fill=P(SIMD[f32, 1](0)))
        for i in range(3):
            for j in range(3):
                s[i * 3 + j] = a[i * 3 + j] + a[j * 3 + i]
        var b = Array[P, 3](fill=P(SIMD[f32, 1](1)))
        b[1] = P(SIMD[f32, 1](Float32(lane) * 0.1))
        var x = lu_factor[f32, 3](a.copy()).solve(b)
        ov.store[1](Coord(lane, 0), x[0].v[0])
        ov.store[1](Coord(lane, 1), x[2].v[0])
        ov.store[1](Coord(lane, 2), inverse[P, 3](a.copy())[4].v[0])
        ov.store[1](Coord(lane, 3), pinv[P, 3](a.copy())[4].v[0])
        ov.store[1](Coord(lane, 4), matrix_rank[P, 3](a.copy()).v[0])
        ov.store[1](Coord(lane, 5), expm[P, 3](a.copy())[4].v[0])
        ov.store[1](Coord(lane, 6), sqrtm[P, 3](s.copy())[4].v[0])
        var w = eigvals[P, 3](a.copy())
        ov.store[1](Coord(lane, 7), (w[0].re + w[1].re + w[2].re).v[0])
        ov.store[1](Coord(lane, 8), eigh[P, 3](s.copy())[0][2].v[0])
        var y0 = P(SIMD[f32, 1](1.0 + Float32(lane) * 0.01))
        ov.store[1](
            Coord(lane, 9),
            rk4[P, _decay](P.constant(0.0), y0.copy(), P.constant(1.0)).v[0],
        )
        ov.store[1](
            Coord(lane, 10),
            dopri5_step[P, _decay](P.constant(0.0), y0.copy(), P.constant(0.1))[
                0
            ].v[0],
        )
        var cutoff = P(SIMD[f32, 1](0.1 + Float32(lane % 8) * 0.1))
        var taps = firwin[P, 9](cutoff)
        ov.store[1](Coord(lane, 11), taps[4].v[0])
        ov.store[1](Coord(lane, 12), taps[1].v[0])
        ov.store[1](Coord(lane, 13), hann[P, 7]()[2].v[0])

    elementwise[simd_width=1, target=target](body, Coord(lanes), ctx)
    ctx.synchronize()
    return out.to_host()


def test_array_tier_runs_inside_a_device_kernel() raises:
    var d = _run["gpu"](DeviceContext())
    var h = _run["cpu"](DeviceContext(api="cpu"))
    for i in range(len(h)):
        assert_almost_equal(Float64(d[i]), Float64(h[i]), atol=1e-4, rtol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
