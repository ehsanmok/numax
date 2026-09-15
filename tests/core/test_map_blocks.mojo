"""`numax.core.tensor.map_blocks`: one small problem per lane.

The claim under test is that a lane receives its *whole* problem and
nothing else: the `k_in` values at one batch index, in row order, and that
whatever `step` computes from them lands in the `k_out` values at that same
index. So every test here has an independent host answer -- a scalar loop,
or the `Array`-tier algorithm run one problem at a time -- and asserts the
batched walk agrees with it.

`batch = 1003` throughout, deliberately not a multiple of any lane count:
the host path walks `simd_width_of[dtype]()` problems per step and MAX runs
the ragged tail at `w = 1`, so a `step` that reads its block wrongly at one
width and rightly at the other cannot pass.

CPU only, because the tests aggregate runs on GPU-less CI. The `gpu=True`
path of the same function is exercised on real hardware by
`examples/advanced/batched_solve.mojo`.
"""

from max.gpu.host import DeviceContext
from std.collections import Array
from std.testing import TestSuite, assert_almost_equal, assert_equal

from numax.core.array import Static
from numax.core.dual import Dual
from numax.core.plain import Plain
from numax.core.tensor import map_blocks
from numax.linalg.array import solve

comptime dtype = DType.float64
comptime batch = 1003
comptime n = 4
comptime k_in = n * n + n


def _value(p: Int, j: Int) -> Float64:
    """Entry `j` of problem `p`'s generator matrix, from a hash."""
    var h = (p * n * n + j) * 2654435761 % 100003
    return Float64(h) / 100003.0 - 0.5


def _ramp(rows: Int) -> List[Scalar[dtype]]:
    """A `(rows, batch)` block where entry `j` of problem `p` is
    `100*j + p` -- distinct everywhere, so a misread block shows up."""
    var values = List[Scalar[dtype]](length=rows * batch, fill=0)
    for j in range(rows):
        for p in range(batch):
            values[j * batch + p] = Scalar[dtype](100 * j + p)
    return values^


def _systems() -> List[Scalar[dtype]]:
    """`batch` SPD systems packed `(k_in, batch)`: `A = M M^T + 4 I` row
    major, then `b`."""
    var values = List[Scalar[dtype]](length=k_in * batch, fill=0)
    for p in range(batch):
        for r in range(n):
            for c in range(n):
                var total = 4.0 if r == c else 0.0
                for k in range(n):
                    total += _value(p, r * n + k) * _value(p, c * n + k)
                values[(r * n + c) * batch + p] = Scalar[dtype](total)
            values[(n * n + r) * batch + p] = Scalar[dtype](
                1.0 + 0.25 * Float64(r)
            )
    return values^


def _identity_step[
    w: Int
](block: Array[SIMD[dtype, w], 3]) -> Array[SIMD[dtype, w], 3]:
    return block.copy()


def _sum_step[
    w: Int
](block: Array[SIMD[dtype, w], 5]) -> Array[SIMD[dtype, w], 1]:
    var total = block[0]
    comptime for j in range(1, 5):
        total += block[j]
    var out = Array[SIMD[dtype, w], 1](uninitialized=True)
    out[0] = total
    return out^


def _solve_step[
    w: Int
](block: Array[SIMD[dtype, w], k_in]) -> Array[SIMD[dtype, w], n]:
    comptime P = Plain[dtype, w]
    var a = Array[P, n * n](uninitialized=True)
    comptime for j in range(n * n):
        a[j] = P(block[j])
    var b = Array[P, n](uninitialized=True)
    comptime for j in range(n):
        b[j] = P(block[n * n + j])

    var x = solve[P, n](a^, b^)
    var out = Array[SIMD[dtype, w], n](uninitialized=True)
    comptime for j in range(n):
        out[j] = x[j].v
    return out^


def _sensitivity_step[
    w: Int
](block: Array[SIMD[dtype, w], k_in]) -> Array[SIMD[dtype, w], n]:
    comptime P = Plain[dtype, w]
    comptime D = Dual[P]
    var a = Array[D, n * n](uninitialized=True)
    comptime for j in range(n * n):
        a[j] = D(P(block[j]), P.constant(1.0 if j == 0 else 0.0))
    var b = Array[D, n](uninitialized=True)
    comptime for j in range(n):
        b[j] = D(P(block[n * n + j]), P.constant(0.0))

    var x = solve[D, n](a^, b^)
    var out = Array[SIMD[dtype, w], n](uninitialized=True)
    comptime for j in range(n):
        out[j] = x[j].deriv.v
    return out^


def _host_solve(a: List[Float64], b: List[Float64]) -> List[Float64]:
    """One problem through the same `Array`-tier `solve`, on the host."""
    comptime H = Plain[dtype]
    var matrix = Array[H, n * n](uninitialized=True)
    for j in range(n * n):
        matrix[j] = H.constant(a[j])
    var vector = Array[H, n](uninitialized=True)
    for j in range(n):
        vector[j] = H.constant(b[j])

    var x = solve[H, n](matrix^, vector^)
    var out = List[Float64](length=n, fill=0.0)
    for j in range(n):
        out[j] = Float64(x[j].v)
    return out^


def test_an_identity_step_round_trips_every_block() raises:
    """Each lane sees its own column and writes it back unchanged -- at a
    `batch` that is not a lane multiple, so the tail runs too."""
    var ctx = DeviceContext(api="cpu")
    var values = _ramp(3)
    var xs = Static[dtype, 3, batch](ctx, values.copy())
    var ys = Static[dtype, 3, batch]._uninitialized(ctx)

    map_blocks[step=_identity_step](xs.view(), ys.view(), ctx)

    var out = ys.to_host()
    for i in range(3 * batch):
        assert_equal(out[i], values[i])
    _ = xs^
    _ = ys^


def test_a_block_sum_equals_a_host_loop() raises:
    """`k_out = 1`: five rows folded to one, against the same fold written
    as a scalar loop over the same packing."""
    var ctx = DeviceContext(api="cpu")
    var values = _ramp(5)
    var xs = Static[dtype, 5, batch](ctx, values.copy())
    var ys = Static[dtype, 1, batch]._uninitialized(ctx)

    map_blocks[step=_sum_step](xs.view(), ys.view(), ctx)

    var out = ys.to_host()
    for p in range(batch):
        var expected = Scalar[dtype](0)
        for j in range(5):
            expected += values[j * batch + p]
        assert_equal(out[p], expected)
    _ = xs^
    _ = ys^


def test_one_array_tier_solve_per_lane_matches_the_host_solve() raises:
    """The claim the primitive exists for: `numax.linalg.array.solve` runs
    inside the lane, and agrees with the same `solve` called once per
    problem on the host."""
    var ctx = DeviceContext(api="cpu")
    var values = _systems()
    var xs = Static[dtype, k_in, batch](ctx, values.copy())
    var ys = Static[dtype, n, batch]._uninitialized(ctx)

    map_blocks[step=_solve_step](xs.view(), ys.view(), ctx)

    var out = ys.to_host()
    for p in range(0, batch, 37):
        var a = List[Float64](length=n * n, fill=0.0)
        var b = List[Float64](length=n, fill=0.0)
        for j in range(n * n):
            a[j] = Float64(values[j * batch + p])
        for j in range(n):
            b[j] = Float64(values[(n * n + j) * batch + p])

        var expected = _host_solve(a, b)
        for j in range(n):
            assert_almost_equal(
                Float64(out[j * batch + p]), expected[j], atol=1e-12
            )
    _ = xs^
    _ = ys^


def test_a_dual_step_differentiates_the_solve_it_runs() raises:
    """`dx/dA00` from one `Dual` pass equals a central difference of the
    same `solve`, per lane."""
    var ctx = DeviceContext(api="cpu")
    var values = _systems()
    var xs = Static[dtype, k_in, batch](ctx, values.copy())
    var ys = Static[dtype, n, batch]._uninitialized(ctx)

    map_blocks[step=_sensitivity_step](xs.view(), ys.view(), ctx)

    var out = ys.to_host()
    comptime step = 1.0e-5
    for p in range(0, batch, 37):
        var a = List[Float64](length=n * n, fill=0.0)
        var b = List[Float64](length=n, fill=0.0)
        for j in range(n * n):
            a[j] = Float64(values[j * batch + p])
        for j in range(n):
            b[j] = Float64(values[(n * n + j) * batch + p])

        a[0] += step
        var up = _host_solve(a, b)
        a[0] -= 2.0 * step
        var down = _host_solve(a, b)

        for j in range(n):
            var fd = (up[j] - down[j]) / (2.0 * step)
            assert_almost_equal(Float64(out[j * batch + p]), fd, atol=1e-8)
    _ = xs^
    _ = ys^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
