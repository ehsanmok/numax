"""Tests for `numax.linalg.panel`, the blocked-factorization primitives.

Each of these checks the *defining identity* of one routine rather than
its output against a table, because that is what a caller relies on:
`potrf_diag` must give an `L` with `L @ L.T == A` on the lower triangle,
`trsm_right_lower_t` must give an `X` with `X @ L.T == B`, and
`getrf_panel` must give `L @ U == P @ A` on the columns it factored. A
table would also pass if the routine were transposed.

`pack_block` is checked against the strided block it copies, since the
whole reason it exists is that MAX's `matmul` will not read that block in
place.
"""

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_almost_equal, assert_equal

from numax.core.array import Static, zeros, zeros_dyn
from numax.linalg.common import _Dense
from numax.linalg.panel import (
    getrf2,
    _PANEL_THREADS,
    getrf_panel,
    pack_block,
    potrf_diag,
    trsm_left_lower_unit,
    trsm_right_lower_t,
)

comptime dtype = DType.float64


def _pivot_matrix[n: Int]() raises -> List[Scalar[dtype]]:
    """A general matrix whose column magnitudes are well separated, so the
    largest entry in any column is unambiguous and two implementations that
    search it must agree on the pivot exactly rather than by luck."""
    var values = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        for j in range(n):
            var v = Float64((i * 7 + j * 3) % 11) + 1.0
            if i == j:
                v = v + Float64(n)
            values[i * n + j] = Scalar[dtype](v * (1.0 + 0.25 * Float64(i)))
    return values^


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _spd(n: Int) -> List[Scalar[dtype]]:
    """A symmetric positive definite `n x n`: `B.T @ B + n * I` for a `B`
    with no zero entries, so every pivot is comfortably positive."""
    var b = List[Float64](length=n * n, fill=0)
    for i in range(n):
        for j in range(n):
            b[i * n + j] = Float64((i * 7 + j * 3) % 5) - 2.0 + 0.25

    var out = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        for j in range(n):
            var total = Float64(0)
            for p in range(n):
                total += b[p * n + i] * b[p * n + j]
            if i == j:
                total += Float64(n)
            out[i * n + j] = Scalar[dtype](total)
    return out^


def test_potrf_diag_factors_the_whole_block() raises:
    """At `nb == n` this is an unblocked Cholesky, so `L @ L.T == A`."""
    comptime n = 6
    var ctx = _cpu()
    var original = _spd(n)
    var a = Static[dtype, n, n](ctx, original.copy())
    var info = zeros[DType.int32, 1](ctx)

    potrf_diag(a.view(), info.view(), 0, n)

    assert_equal(Int(info.to_host()[0]), 0)
    var l = a.to_host()
    for i in range(n):
        for j in range(i + 1):
            var total = Float64(0)
            for p in range(j + 1):
                total += Float64(l[i * n + p]) * Float64(l[j * n + p])
            assert_almost_equal(total, Float64(original[i * n + j]), atol=1e-10)


def test_potrf_diag_leaves_the_upper_triangle_alone() raises:
    """Only the lower triangle is a factor; the rest is the caller's to
    zero, and this asserts it is untouched rather than garbage."""
    comptime n = 5
    var ctx = _cpu()
    var original = _spd(n)
    var a = Static[dtype, n, n](ctx, original.copy())
    var info = zeros[DType.int32, 1](ctx)

    potrf_diag(a.view(), info.view(), 0, n)

    var got = a.to_host()
    for i in range(n):
        for j in range(i + 1, n):
            assert_almost_equal(
                Float64(got[i * n + j]), Float64(original[i * n + j]), atol=0
            )


def test_potrf_diag_factors_an_offset_block() raises:
    """The block is named by `(k, nb)`, so factoring the bottom-right
    corner must leave everything left of and above it exactly as it was."""
    comptime n = 6
    comptime k = 2
    comptime nb = 4
    var ctx = _cpu()
    var original = _spd(n)
    var a = Static[dtype, n, n](ctx, original.copy())
    var info = zeros[DType.int32, 1](ctx)

    potrf_diag(a.view(), info.view(), k, nb)

    var got = a.to_host()
    for i in range(n):
        for j in range(n):
            if i >= k and j >= k and j <= i:
                continue
            assert_almost_equal(
                Float64(got[i * n + j]), Float64(original[i * n + j]), atol=0
            )

    for i in range(nb):
        for j in range(i + 1):
            var total = Float64(0)
            for p in range(j + 1):
                total += Float64(got[(k + i) * n + k + p]) * Float64(
                    got[(k + j) * n + k + p]
                )
            assert_almost_equal(
                total, Float64(original[(k + i) * n + k + j]), atol=1e-10
            )


def test_potrf_diag_flags_a_matrix_that_is_not_positive_definite() raises:
    """`info` carries the 1-based column of the first bad pivot, and the
    routine returns a finite answer rather than faulting."""
    comptime n = 3
    var ctx = _cpu()
    # Diagonal entry (1, 1) is too small for the column below it.
    var a = Static[dtype, n, n](
        ctx, [4.0, 2.0, 2.0, 2.0, 1.0, 0.0, 2.0, 0.0, 9.0]
    )
    var info = zeros[DType.int32, 1](ctx)

    potrf_diag(a.view(), info.view(), 0, n)

    # Column 0 factors to 2; the update leaves column 1's pivot at exactly
    # 1 - 1 == 0, so the first failure is column 1, reported 1-based.
    assert_equal(Int(info.to_host()[0]), 2)


def test_trsm_right_lower_t_solves_against_the_diagonal_block() raises:
    """`X @ L.T == B` for the panel below a factored diagonal block."""
    comptime n = 7
    comptime nb = 3
    var ctx = _cpu()
    var original = _spd(n)
    var a = Static[dtype, n, n](ctx, original.copy())
    var info = zeros[DType.int32, 1](ctx)

    potrf_diag(a.view(), info.view(), 0, nb)
    trsm_right_lower_t(a.view(), 0, nb, n, ctx)
    ctx.synchronize()

    var got = a.to_host()
    for i in range(nb, n):
        for j in range(nb):
            var total = Float64(0)
            for p in range(j + 1):
                total += Float64(got[i * n + p]) * Float64(got[j * n + p])
            assert_almost_equal(total, Float64(original[i * n + j]), atol=1e-10)


def test_trsm_left_lower_unit_solves_the_block_row() raises:
    """`L @ Y == B` with `L` unit-diagonal, which is LU's `U12` step."""
    comptime n = 6
    comptime nb = 3
    var ctx = _cpu()
    var original = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        for j in range(n):
            original[i * n + j] = Scalar[dtype](
                Float64((i * 5 + j * 2) % 7) + 1.0
            )
    var a = Static[dtype, n, n](ctx, original.copy())

    trsm_left_lower_unit(a.view(), 0, nb, n, ctx)
    ctx.synchronize()

    var got = a.to_host()
    for i in range(nb):
        for col in range(nb, n):
            # Row `i` of `L @ Y`, with `L`'s diagonal implicit.
            var total = Float64(got[i * n + col])
            for p in range(i):
                total += Float64(original[i * n + p]) * Float64(
                    got[p * n + col]
                )
            assert_almost_equal(
                total, Float64(original[i * n + col]), atol=1e-10
            )


def test_getrf_panel_gives_lu_of_the_columns_it_factored() raises:
    """`L @ U == P @ A` over the panel's columns, with `P` rebuilt from the
    recorded interchanges."""
    comptime n = 5
    comptime nb = 3
    var ctx = _cpu()
    var original = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        for j in range(n):
            original[i * n + j] = Scalar[dtype](
                Float64((i * 3 + j * 7) % 11) - 5.0 + 0.5
            )
    var a = Static[dtype, n, n](ctx, original.copy())
    var pivots = zeros[DType.int32, n + _PANEL_THREADS](ctx)
    var info = zeros[DType.int32, 1](ctx)

    getrf_panel(a.view(), pivots.view(), info.view(), 0, nb, n)

    assert_equal(Int(info.to_host()[0]), 0)

    # Apply the same interchanges to a copy of `A`.
    var permuted = original.copy()
    var swaps = pivots.to_host()
    for j in range(nb):
        var other = Int(swaps[j])
        if other != j:
            for c in range(n):
                var keep = permuted[j * n + c]
                permuted[j * n + c] = permuted[other * n + c]
                permuted[other * n + c] = keep

    var got = a.to_host()
    for i in range(n):
        for j in range(nb):
            var total = Float64(0)
            for p in range(min(i, j + 1)):
                total += Float64(got[i * n + p]) * Float64(got[p * n + j])
            if i <= j:
                total += Float64(got[i * n + j])
            assert_almost_equal(total, Float64(permuted[i * n + j]), atol=1e-10)


def test_getrf_panel_picks_the_largest_pivot() raises:
    """Partial pivoting, so a column whose largest entry is not on the
    diagonal has to interchange -- this is the property that makes LU
    usable on a matrix with a zero in the pivot position."""
    comptime n = 3
    var ctx = _cpu()
    var a = Static[dtype, n, n](
        ctx, [0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    )
    var pivots = zeros[DType.int32, n + _PANEL_THREADS](ctx)
    var info = zeros[DType.int32, 1](ctx)

    getrf_panel(a.view(), pivots.view(), info.view(), 0, 1, n)

    # Column 0 is (0, 4, 7): the largest magnitude is row 2.
    assert_equal(Int(pivots.to_host()[0]), 2)
    assert_equal(Int(info.to_host()[0]), 0)


def test_pack_block_copies_a_strided_block_densely() raises:
    """The block MAX's `matmul` refuses to read in place, made dense."""
    comptime n = 6
    comptime rows = 3
    comptime cols = 2
    var ctx = _cpu()
    var values = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n * n):
        values[i] = Scalar[dtype](i)
    var a = Static[dtype, n, n](ctx, values.copy())
    var dst = zeros[dtype, rows, cols](ctx)

    pack_block(a.view(), dst.view(), 2, 3, rows, cols, ctx)
    ctx.synchronize()

    var got = dst.to_host()
    for i in range(rows):
        for j in range(cols):
            assert_almost_equal(
                Float64(got[i * cols + j]),
                Float64(values[(2 + i) * n + 3 + j]),
                atol=0,
            )


def test_pack_block_copies_a_width_the_simd_lanes_do_not_divide() raises:
    """A block whose column count is prime, so the widened copy's ragged
    tail runs rather than being an even number of vectors.

    `pack_block` launches its untransposed copy at the native SIMD width,
    so `cols` divisible by that width would exercise only the whole-vector
    path. Seven is prime and smaller than any SIMD width this runs at, so
    the tail is the whole copy at some widths and part of it at others.
    """
    comptime n = 16
    comptime rows = 5
    comptime cols = 7
    var ctx = _cpu()
    var values = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n * n):
        values[i] = Scalar[dtype](i)
    var a = Static[dtype, n, n](ctx, values.copy())
    var dst = zeros[dtype, rows, cols](ctx)

    pack_block(a.view(), dst.view(), 3, 2, rows, cols, ctx)
    ctx.synchronize()

    var got = dst.to_host()
    for i in range(rows):
        for j in range(cols):
            assert_almost_equal(
                Float64(got[i * cols + j]),
                Float64(values[(3 + i) * n + 2 + j]),
                atol=0,
            )


def test_pack_block_transposed_copies_a_ragged_width() raises:
    """The same ragged shape through the transposed branch, which stays
    scalar because consecutive columns of `dst` read down a column of `a`.

    Here so that widening the untransposed launch cannot silently widen
    this one too: a strided gather read a vector at a time would return
    neighbouring rows instead of the column, and the assertion below is
    what catches it.
    """
    comptime n = 16
    comptime rows = 5
    comptime cols = 7
    var ctx = _cpu()
    var values = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n * n):
        values[i] = Scalar[dtype](i)
    var a = Static[dtype, n, n](ctx, values.copy())
    var dst = zeros[dtype, rows, cols](ctx)

    pack_block[trans=True](a.view(), dst.view(), 3, 2, rows, cols, ctx)
    ctx.synchronize()

    var got = dst.to_host()
    for i in range(rows):
        for j in range(cols):
            assert_almost_equal(
                Float64(got[i * cols + j]),
                Float64(values[(2 + j) * n + 3 + i]),
                atol=0,
            )


def test_getrf2_agrees_with_the_single_block_panel_at_every_base() raises:
    """The recursive panel and the one it replaces factor the same columns
    into the same `L`, `U` and pivots.

    `base >= nb` takes the base case on the first test, which *is*
    `getrf_panel`, so that run is the control every other base is pinned
    against. The factor is compared with a tolerance because the recursion
    reassociates -- its rank-one updates become a GEMM -- but the pivots are
    compared exactly: the matrix below has well-separated column magnitudes,
    so no two candidates can legitimately tie and a different pivot means a
    different search, not a different rounding.

    `nb = 7` is prime, so every level of the split is ragged and `half`
    never equals `nb - half`.
    """
    comptime n = 12
    comptime nb = 7
    var ctx = _cpu()

    var want_a = Static[dtype, n, n](ctx, _pivot_matrix[n]())
    var want_p = zeros[DType.int32, n + _PANEL_THREADS](ctx)
    var want_i = zeros[DType.int32, 1](ctx)
    var left0 = zeros_dyn[dtype, 2](n, nb, ctx=ctx)
    var right0 = zeros_dyn[dtype, 2](nb, n, ctx=ctx)
    var prod0 = zeros_dyn[dtype, 2](n, n, ctx=ctx)
    var wl0: _Dense[dtype] = TileTensor(
        left0.view().ptr_at_offset(Coord(0, 0)), row_major(Coord(n, nb))
    )
    var wr0: _Dense[dtype] = TileTensor(
        right0.view().ptr_at_offset(Coord(0, 0)), row_major(Coord(nb, n))
    )
    var wp0: _Dense[dtype] = TileTensor(
        prod0.view().ptr_at_offset(Coord(0, 0)), row_major(Coord(n, n))
    )
    getrf2(
        want_a.view(),
        want_p.view(),
        want_i.view(),
        wl0,
        wr0,
        wp0,
        0,
        nb,
        n,
        nb,
        ctx,
    )
    ctx.synchronize()
    var want = want_a.to_host()
    var want_pivots = want_p.to_host()
    _ = left0^
    _ = right0^
    _ = prod0^

    for base in [1, 2, 3, 4, 5]:
        var got_a = Static[dtype, n, n](ctx, _pivot_matrix[n]())
        var got_p = zeros[DType.int32, n + _PANEL_THREADS](ctx)
        var got_i = zeros[DType.int32, 1](ctx)
        var left = zeros_dyn[dtype, 2](n, nb, ctx=ctx)
        var right = zeros_dyn[dtype, 2](nb, n, ctx=ctx)
        var prod = zeros_dyn[dtype, 2](n, n, ctx=ctx)
        var wl: _Dense[dtype] = TileTensor(
            left.view().ptr_at_offset(Coord(0, 0)), row_major(Coord(n, nb))
        )
        var wr: _Dense[dtype] = TileTensor(
            right.view().ptr_at_offset(Coord(0, 0)), row_major(Coord(nb, n))
        )
        var wp: _Dense[dtype] = TileTensor(
            prod.view().ptr_at_offset(Coord(0, 0)), row_major(Coord(n, n))
        )
        getrf2(
            got_a.view(),
            got_p.view(),
            got_i.view(),
            wl,
            wr,
            wp,
            0,
            nb,
            n,
            base,
            ctx,
        )
        ctx.synchronize()
        var got = got_a.to_host()
        var got_pivots = got_p.to_host()

        # Every column, not just the panel's: interchanges are applied
        # across the full width as they are found, so the right recursion
        # permutes rows of already-written `L` and of the untouched
        # trailing columns. A replay bug shows up outside the panel.
        for i in range(n * n):
            assert_almost_equal(Float64(got[i]), Float64(want[i]), atol=1e-10)
        for j in range(nb):
            assert_equal(Int(got_pivots[j]), Int(want_pivots[j]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
