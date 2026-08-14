"""Tests for `ember.tensor.map`/`map_simd`, driving `FloatLike` kernels over
`TileTensor`.

These mirror `test_special.mojo`'s value/derivative checks, but through the
`TileTensor` walk `examples/gaussian.mojo` and `examples/gaussian_gpu.mojo`
actually use, rather than calling the kernel directly on a single `FloatLike`
value.
"""

from layout import TileTensor
from layout.tile_layout import row_major
from std.math import exp
from std.sys.info import simd_width_of
from std.testing import TestSuite, assert_almost_equal, assert_true

from ember import Compensated, Dual, Plain, gaussian
from ember.tensor import map, map_simd

comptime dtype = DType.float32
comptime n = 16
comptime width = simd_width_of[dtype]()


def wrap_plain(x: SIMD[dtype, 1]) -> Plain[dtype, 1]:
    return Plain[dtype, 1](x)


def unwrap_plain(p: Plain[dtype, 1]) -> SIMD[dtype, 1]:
    return p.v


def wrap_dual(x: SIMD[dtype, 1]) -> Dual[Plain[dtype, 1]]:
    return Dual[Plain[dtype, 1]](Plain[dtype, 1](x), Plain[dtype, 1](1))


def unwrap_dual_deriv(d: Dual[Plain[dtype, 1]]) -> SIMD[dtype, 1]:
    return d.deriv.v


def wrap_compensated(x: SIMD[dtype, 1]) -> Compensated[dtype, 1]:
    return Compensated[dtype, 1](x, 0)


def unwrap_compensated_value(c: Compensated[dtype, 1]) -> SIMD[dtype, 1]:
    return c.value


def unwrap_compensated_error(c: Compensated[dtype, 1]) -> SIMD[dtype, 1]:
    return c.error


def test_map_plain_matches_direct_call() raises:
    comptime layout = row_major[n]()
    var xs_storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        xs_storage.append(Scalar[dtype](i) * 0.25 - 2.0)
    var xs = TileTensor(xs_storage, layout)
    var ys_storage = List[Scalar[dtype]](length=n, fill=0)
    var ys = TileTensor(ys_storage, layout)

    map[
        dtype=dtype,
        T=Plain[dtype, 1],
        width=1,
        kernel=gaussian[Plain[dtype, 1]],
        wrap=wrap_plain,
        unwrap=unwrap_plain,
    ](xs, ys)

    for i in range(n):
        var expected = gaussian(Plain[dtype, 1](xs[i])).v
        assert_almost_equal(ys[i], expected)


def test_map_dual_derivative_matches_closed_form() raises:
    comptime layout = row_major[n]()
    var xs_storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        xs_storage.append(Scalar[dtype](i) * 0.25 - 2.0)
    var xs = TileTensor(xs_storage, layout)
    var dydx_storage = List[Scalar[dtype]](length=n, fill=0)
    var dydx = TileTensor(dydx_storage, layout)

    map[
        dtype=dtype,
        T=Dual[Plain[dtype, 1]],
        width=1,
        kernel=gaussian[Dual[Plain[dtype, 1]]],
        wrap=wrap_dual,
        unwrap=unwrap_dual_deriv,
    ](xs, dydx)

    for i in range(n):
        var x0 = Float64(xs[i])
        var y0 = exp(-(x0 * x0))
        var expected_d = -2.0 * x0 * y0
        assert_almost_equal(
            dydx[i], SIMD[dtype, 1](expected_d), atol=1e-5, rtol=1e-4
        )


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

    map[
        dtype=dtype,
        T=Plain[dtype, 1],
        width=1,
        kernel=gaussian[Plain[dtype, 1]],
        wrap=wrap_plain,
        unwrap=unwrap_plain,
    ](xs, ys_plain)
    map[
        dtype=dtype,
        T=Compensated[dtype, 1],
        width=1,
        kernel=gaussian[Compensated[dtype, 1]],
        wrap=wrap_compensated,
        unwrap=unwrap_compensated_value,
    ](xs, ys_comp_value)
    map[
        dtype=dtype,
        T=Compensated[dtype, 1],
        width=1,
        kernel=gaussian[Compensated[dtype, 1]],
        wrap=wrap_compensated,
        unwrap=unwrap_compensated_error,
    ](xs, ys_comp_error)

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


def gaussian_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(Plain[dtype, w](x)).v


def gaussian_deriv_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(
        Dual[Plain[dtype, w]](Plain[dtype, w](x), Plain[dtype, w](1))
    ).deriv.v


def test_map_simd_at_native_width_matches_direct_call() raises:
    # n is a multiple of `width`, so this only exercises the vectorized path.
    comptime layout = row_major[n]()
    var xs_storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        xs_storage.append(Scalar[dtype](i) * 0.25 - 2.0)
    var xs = TileTensor(xs_storage, layout)
    var ys_storage = List[Scalar[dtype]](length=n, fill=0)
    var ys = TileTensor(ys_storage, layout)

    map_simd[width=width, step=gaussian_step](xs, ys)

    for i in range(n):
        var expected = gaussian(Plain[dtype, 1](xs[i])).v
        assert_almost_equal(ys[i], expected)


def test_map_simd_with_remainder_covers_tail_loop() raises:
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

    map_simd[width=width, step=gaussian_step](xs, ys)

    for i in range(odd_n):
        var expected = gaussian(Plain[dtype, 1](xs[i])).v
        assert_almost_equal(ys[i], expected)


def test_map_simd_dual_derivative_matches_closed_form() raises:
    comptime odd_n = n + 1
    comptime layout = row_major[odd_n]()
    var xs_storage = List[Scalar[dtype]](capacity=odd_n)
    for i in range(odd_n):
        xs_storage.append(Scalar[dtype](i) * 0.2 - 2.0)
    var xs = TileTensor(xs_storage, layout)
    var dydx_storage = List[Scalar[dtype]](length=odd_n, fill=0)
    var dydx = TileTensor(dydx_storage, layout)

    map_simd[width=width, step=gaussian_deriv_step](xs, dydx)

    for i in range(odd_n):
        var x0 = Float64(xs[i])
        var y0 = exp(-(x0 * x0))
        var expected_d = -2.0 * x0 * y0
        assert_almost_equal(
            dydx[i], SIMD[dtype, 1](expected_d), atol=1e-5, rtol=1e-4
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
