"""Tests for `numax.tensor.map`, driving `FloatLike` kernels over `TileTensor`.

These mirror `test_special.mojo`'s value/derivative checks, but through the
`TileTensor` walk `examples/gaussian.mojo` and `examples/gaussian_gpu.mojo`
actually use, rather than calling the kernel directly on a single `FloatLike`
value.
"""

from layout import Coord, TileTensor
from max.gpu.host import DeviceContext
from layout.tile_layout import row_major
from std.math import exp
from std.sys.info import simd_width_of
from std.testing import TestSuite, assert_almost_equal, assert_true

from numax import Compensated, Dual, Plain, gaussian
from numax.tensor import (
    add_combine,
    add_step,
    map,
    map_threaded,
    mul_step,
    reduce,
)

comptime dtype = DType.float32
comptime n = 16
comptime width = simd_width_of[dtype]()


def gaussian_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(Plain[dtype, w](x)).v


def gaussian_deriv_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(
        Dual[Plain[dtype, w]](Plain[dtype, w](x), Plain[dtype, w](1))
    ).deriv.v


def gaussian_compensated_value_step[
    w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(Compensated[dtype, w](x, 0)).value


def gaussian_compensated_error_step[
    w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(Compensated[dtype, w](x, 0)).error


def test_map_at_native_width_matches_direct_call() raises:
    # n is a multiple of `width`, so this only exercises the vectorized path.
    comptime layout = row_major[n]()
    var xs_storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        xs_storage.append(Scalar[dtype](i) * 0.25 - 2.0)
    var xs = TileTensor(xs_storage, layout)
    var ys_storage = List[Scalar[dtype]](length=n, fill=0)
    var ys = TileTensor(ys_storage, layout)

    map[width=width, step=gaussian_step](xs, ys)

    for i in range(n):
        var expected = gaussian(Plain[dtype, 1](xs[i])).v
        assert_almost_equal(ys[i], expected)


def test_map_with_remainder_covers_tail_loop() raises:
    # n is deliberately not a multiple of `width`, so the scalar tail loop
    # has to pick up whatever the vectorized pass can't cover evenly.
    comptime odd_n = n + 1
    comptime layout = row_major[odd_n]()
    var xs_storage = List[Scalar[dtype]](capacity=odd_n)
    for i in range(odd_n):
        xs_storage.append(Scalar[dtype](i) * 0.2 - 2.0)
    var xs = TileTensor(xs_storage, layout)
    var ys_storage = List[Scalar[dtype]](length=odd_n, fill=0)
    var ys = TileTensor(ys_storage, layout)

    map[width=width, step=gaussian_step](xs, ys)

    for i in range(odd_n):
        var expected = gaussian(Plain[dtype, 1](xs[i])).v
        assert_almost_equal(ys[i], expected)


def test_map_dual_derivative_matches_closed_form() raises:
    comptime odd_n = n + 1
    comptime layout = row_major[odd_n]()
    var xs_storage = List[Scalar[dtype]](capacity=odd_n)
    for i in range(odd_n):
        xs_storage.append(Scalar[dtype](i) * 0.2 - 2.0)
    var xs = TileTensor(xs_storage, layout)
    var dydx_storage = List[Scalar[dtype]](length=odd_n, fill=0)
    var dydx = TileTensor(dydx_storage, layout)

    map[width=width, step=gaussian_deriv_step](xs, dydx)

    for i in range(odd_n):
        var x0 = Float64(xs[i])
        var y0 = exp(-(x0 * x0))
        var expected_d = -2.0 * x0 * y0
        assert_almost_equal(
            dydx[i], SIMD[dtype, 1](expected_d), atol=1e-5, rtol=1e-4
        )


def test_map_walks_a_multidimensional_tensor_via_coalesce() raises:
    # `map` accepts any rank directly -- no manual `.coalesce()` needed at
    # the call site. `n` (16) factors as 2*2*4, so this is a genuine rank-3,
    # non-rank-1 tensor, not a rank-1 tensor merely labeled otherwise.
    comptime layout = row_major[2, 2, 4]()
    var xs_storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        xs_storage.append(Scalar[dtype](i) * 0.25 - 2.0)
    var xs = TileTensor(xs_storage, layout)
    var ys_storage = List[Scalar[dtype]](length=n, fill=0)
    var ys = TileTensor(ys_storage, layout)

    map[width=width, step=gaussian_step](xs, ys)

    var xs_flat = xs.coalesce()
    var ys_flat = ys.coalesce()
    for i in range(n):
        var expected = gaussian(Plain[dtype, 1](xs_flat[i])).v
        assert_almost_equal(ys_flat[i], expected)


def test_map_compensated_beats_plain_for_small_x() raises:
    # Same margin `test_compensated.mojo` checks for a single value, now
    # produced by the tensor walk instead of a direct call.
    comptime layout = row_major[n]()
    var xs_storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        xs_storage.append(Scalar[dtype](i) * 0.25 - 2.0)
    var xs = TileTensor(xs_storage, layout)
    var plain_storage = List[Scalar[dtype]](length=n, fill=0)
    var comp_value_storage = List[Scalar[dtype]](length=n, fill=0)
    var comp_error_storage = List[Scalar[dtype]](length=n, fill=0)
    var ys_plain = TileTensor(plain_storage, layout)
    var ys_comp_value = TileTensor(comp_value_storage, layout)
    var ys_comp_error = TileTensor(comp_error_storage, layout)

    map[width=width, step=gaussian_step](xs, ys_plain)
    map[width=width, step=gaussian_compensated_value_step](xs, ys_comp_value)
    map[width=width, step=gaussian_compensated_error_step](xs, ys_comp_error)

    var found_improvement = False
    for i in range(n):
        var x0 = Float64(xs[i])
        var reference = exp(-(x0 * x0))
        var plain_err = abs(Float64(ys_plain[i]) - reference)
        var comp_err = abs(
            (Float64(ys_comp_value[i]) + Float64(ys_comp_error[i])) - reference
        )
        if plain_err > 0:
            assert_true(comp_err <= plain_err)
            if comp_err < plain_err:
                found_improvement = True

    assert_true(
        found_improvement,
        msg="expected compensated to beat plain for at least one point",
    )


def gaussian_weighted_step[
    w: Int
](x: SIMD[dtype, w], weight: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """A two-input kernel that is genuinely two-input: it needs a second
    buffer, not just more arithmetic on the first."""
    return (gaussian(Plain[dtype, w](x)) * Plain[dtype, w](weight)).v


def test_binary_map_adds_two_tensors() raises:
    comptime layout = row_major[n]()
    var lhs_storage = List[Scalar[dtype]](capacity=n)
    var rhs_storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        lhs_storage.append(Scalar[dtype](i) * 0.5)
        rhs_storage.append(Scalar[dtype](i) * -0.125 + 3.0)
    var lhs = TileTensor(lhs_storage, layout)
    var rhs = TileTensor(rhs_storage, layout)
    var out_storage = List[Scalar[dtype]](length=n, fill=0)
    var out = TileTensor(out_storage, layout)

    map[width=width, step=add_step[dtype, _]](lhs, rhs, out)

    for i in range(n):
        assert_almost_equal(out[i], lhs[i] + rhs[i])


def test_binary_map_multiplies_with_a_remainder() raises:
    # Not a multiple of `width`, so the scalar tail runs here too.
    comptime odd_n = n + 3
    comptime layout = row_major[odd_n]()
    var lhs_storage = List[Scalar[dtype]](capacity=odd_n)
    var rhs_storage = List[Scalar[dtype]](capacity=odd_n)
    for i in range(odd_n):
        lhs_storage.append(Scalar[dtype](i) * 0.3 - 1.0)
        rhs_storage.append(Scalar[dtype](i) * 0.7 + 0.25)
    var lhs = TileTensor(lhs_storage, layout)
    var rhs = TileTensor(rhs_storage, layout)
    var out_storage = List[Scalar[dtype]](length=odd_n, fill=0)
    var out = TileTensor(out_storage, layout)

    map[width=width, step=mul_step[dtype, _]](lhs, rhs, out)

    for i in range(odd_n):
        assert_almost_equal(out[i], lhs[i] * rhs[i], atol=1e-6)


def test_binary_map_runs_a_floatlike_kernel_over_both_inputs() raises:
    comptime layout = row_major[n]()
    var xs_storage = List[Scalar[dtype]](capacity=n)
    var ws_storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        xs_storage.append(Scalar[dtype](i) * 0.25 - 2.0)
        ws_storage.append(Scalar[dtype](i) * 0.1)
    var xs = TileTensor(xs_storage, layout)
    var ws = TileTensor(ws_storage, layout)
    var out_storage = List[Scalar[dtype]](length=n, fill=0)
    var out = TileTensor(out_storage, layout)

    map[width=width, step=gaussian_weighted_step](xs, ws, out)

    for i in range(n):
        var expected = gaussian(Plain[dtype, 1](xs[i])).v * ws[i]
        assert_almost_equal(out[i], expected, atol=1e-6)


def test_binary_map_coalesces_a_multidimensional_tensor() raises:
    comptime rows = 2
    comptime cols = 3
    comptime depth = 4
    comptime total = rows * cols * depth
    comptime layout = row_major[rows, cols, depth]()
    var lhs_storage = List[Scalar[dtype]](capacity=total)
    var rhs_storage = List[Scalar[dtype]](capacity=total)
    for i in range(total):
        lhs_storage.append(Scalar[dtype](i) * 0.5)
        rhs_storage.append(Scalar[dtype](total - i))
    var lhs = TileTensor(lhs_storage, layout)
    var rhs = TileTensor(rhs_storage, layout)
    var out_storage = List[Scalar[dtype]](length=total, fill=0)
    var out = TileTensor(out_storage, layout)

    map[width=width, step=add_step[dtype, _]](lhs, rhs, out)

    var lhs_flat = lhs.coalesce()
    var rhs_flat = rhs.coalesce()
    var out_flat = out.coalesce()
    for i in range(total):
        assert_almost_equal(out_flat[i], lhs_flat[i] + rhs_flat[i])


def test_map_threaded_matches_the_serial_walk() raises:
    # A size that is neither a multiple of the SIMD width nor of any likely
    # thread-chunk size, so a mishandled remainder on either path shows up.
    comptime m = 100003
    comptime layout = row_major[m]()
    var xs = List[Scalar[dtype]](length=m, fill=0)
    for i in range(m):
        xs[i] = Scalar[dtype](i) * 0.0001 - 5.0
    var serial = List[Scalar[dtype]](length=m, fill=0)
    var threaded = List[Scalar[dtype]](length=m, fill=0)

    map[width=width, step=gaussian_step](
        TileTensor(xs, layout), TileTensor(serial, layout)
    )
    map_threaded[width=width, step=gaussian_step](
        TileTensor(xs, layout),
        TileTensor(threaded, layout),
        DeviceContext(api="cpu"),
    )
    for i in range(m):
        assert_almost_equal(serial[i], threaded[i])


def test_map_threaded_coalesces_a_rank_three_tensor() raises:
    comptime layout = row_major[4, 8, 16]()
    comptime m = 4 * 8 * 16
    var xs = List[Scalar[dtype]](length=m, fill=0)
    for i in range(m):
        xs[i] = Scalar[dtype](i) * 0.01
    var serial = List[Scalar[dtype]](length=m, fill=0)
    var threaded = List[Scalar[dtype]](length=m, fill=0)

    map[width=width, step=gaussian_step](
        TileTensor(xs, layout), TileTensor(serial, layout)
    )
    map_threaded[width=width, step=gaussian_step](
        TileTensor(xs, layout),
        TileTensor(threaded, layout),
        DeviceContext(api="cpu"),
    )
    for i in range(m):
        assert_almost_equal(serial[i], threaded[i])


# ------------------------------------------------------------------
# Runtime-shape overloads. Each of these checks the dynamic path against
# the static one on the same values, rather than against a hand-computed
# expectation -- the two are supposed to be the same walk, so agreeing with
# each other is the property worth testing.
# ------------------------------------------------------------------


def test_dynamic_map_agrees_with_the_static_map() raises:
    var xs_storage = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        xs_storage[i] = Scalar[dtype](i) * 0.25 - 2.0

    # Static: comptime row_major[n]().
    var static_out = List[Scalar[dtype]](length=n, fill=0)
    var xs_static = TileTensor(xs_storage, row_major[n]())
    var ys_static = TileTensor(static_out, row_major[n]())
    map[width=width, step=gaussian_step](xs_static, ys_static)

    # Dynamic: row_major(Coord(...)), and a rank-2 shape at that, so the
    # internal flattening is exercised rather than trivially satisfied.
    var dyn_out = List[Scalar[dtype]](length=n, fill=0)
    var xs_dyn = TileTensor(xs_storage, row_major(Coord(4, 4)))
    var ys_dyn = TileTensor(dyn_out, row_major(Coord(4, 4)))
    map[width=width, step=gaussian_step](xs_dyn, ys_dyn)

    for i in range(n):
        assert_almost_equal(dyn_out[i], static_out[i])


def test_dynamic_map_handles_a_width_that_does_not_divide_n() raises:
    # 11 elements at the native width leaves a tail; dropping it would show
    # up as an unwritten output element.
    comptime odd_n = 11
    var xs_storage = List[Scalar[dtype]](length=odd_n, fill=0)
    for i in range(odd_n):
        xs_storage[i] = Scalar[dtype](i) * 0.5 - 1.0
    var out = List[Scalar[dtype]](length=odd_n, fill=-99)
    var xs = TileTensor(xs_storage, row_major(Coord(odd_n)))
    var ys = TileTensor(out, row_major(Coord(odd_n)))
    map[width=width, step=gaussian_step](xs, ys)
    for i in range(odd_n):
        assert_almost_equal(out[i], exp(-(xs_storage[i] * xs_storage[i])))


def test_dynamic_binary_map_agrees_with_the_static_one() raises:
    var lhs = List[Scalar[dtype]](length=n, fill=0)
    var rhs = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        lhs[i] = Scalar[dtype](i)
        rhs[i] = Scalar[dtype](2 * i + 1)

    var static_out = List[Scalar[dtype]](length=n, fill=0)
    map[width=width, step=add_step[dtype, _]](
        TileTensor(lhs, row_major[n]()),
        TileTensor(rhs, row_major[n]()),
        TileTensor(static_out, row_major[n]()),
    )

    var dyn_out = List[Scalar[dtype]](length=n, fill=0)
    map[width=width, step=add_step[dtype, _]](
        TileTensor(lhs, row_major(Coord(2, 8))),
        TileTensor(rhs, row_major(Coord(2, 8))),
        TileTensor(dyn_out, row_major(Coord(2, 8))),
    )

    for i in range(n):
        assert_almost_equal(dyn_out[i], static_out[i])


def test_dynamic_reduce_agrees_with_the_static_reduce() raises:
    var storage = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        storage[i] = Scalar[dtype](i) * 0.125

    var static_total = reduce[combine=add_combine[dtype]](
        TileTensor(storage, row_major[n]()), 0
    )
    var dyn_total = reduce[combine=add_combine[dtype]](
        TileTensor(storage, row_major(Coord(4, 4))), 0
    )
    # Both fold left-to-right over the same order, so this is exact, not
    # approximate.
    assert_true(dyn_total == static_total)


def test_dynamic_map_writes_through_to_the_underlying_storage() raises:
    # The dynamic path builds its flat view from `ptr_at_offset`, so a
    # regression that copied instead of aliasing would leave the caller's
    # buffer untouched and every other assertion here would still pass.
    var xs_storage = List[Scalar[dtype]](length=4, fill=1)
    var ys_storage = List[Scalar[dtype]](length=4, fill=0)
    var xs = TileTensor(xs_storage, row_major(Coord(2, 2)))
    var ys = TileTensor(ys_storage, row_major(Coord(2, 2)))
    map[width=1, step=gaussian_step](xs, ys)
    for i in range(4):
        assert_almost_equal(ys_storage[i], Scalar[dtype](exp(-1.0)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
