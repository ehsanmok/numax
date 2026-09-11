"""Tests for `numax.core.array`'s NumPy-named creation/manipulation surface.

Every creation routine is checked for shape (`num_elements`) and content;
`*_like` is checked to match its source's dtype/shape; the manipulation
functions (`transpose`, `squeeze`, `stack`, `reshape`, `ravel`,
`concatenate`, `split`) are checked against a hand-computed expected
result, not just "it runs". The `DType` aliases are checked against the
`DType` each is meant to name, since a mistyped one is otherwise silent.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.dtypes import (
    bf16,
    bool,
    f8e3m4,
    f8e4m3fn,
    f8e4m3fnuz,
    f8e5m2,
    f8e5m2fnuz,
    f16,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
)
from numax.core.array import (
    to_array,
    Static,
    Tensor,
    arange,
    concatenate,
    empty,
    empty_like,
    expand_dims,
    eye,
    full,
    full_like,
    linspace,
    logspace,
    moveaxis,
    ones,
    ones_like,
    ravel,
    repeat,
    reshape,
    roll,
    split,
    squeeze,
    stack,
    swapaxes,
    tile,
    transpose,
    zeros,
    zeros_like,
)

from numax.core.plain import Plain

comptime dtype = DType.float32
comptime P = Plain[DType.float64, 1]


def test_zeros_has_the_requested_shape_and_content() raises:
    var ctx = DeviceContext(api="cpu")
    var z = zeros[dtype, 2, 3](ctx)
    assert_equal(z.num_elements, 6)
    for i in range(6):
        assert_equal(z[i], Scalar[dtype](0))


def test_ones_is_filled_with_one() raises:
    var ctx = DeviceContext(api="cpu")
    var o = ones[dtype, 4](ctx)
    assert_equal(o.num_elements, 4)
    for i in range(4):
        assert_equal(o[i], Scalar[dtype](1))


def test_full_is_filled_with_the_given_value() raises:
    var ctx = DeviceContext(api="cpu")
    var f = full[dtype, 3](7, ctx=ctx)
    for i in range(3):
        assert_equal(f[i], Scalar[dtype](7))


def test_empty_is_zero_initialized_for_memory_safety() raises:
    var ctx = DeviceContext(api="cpu")
    # numax's `empty` documents zero-init rather than true uninitialized
    # memory -- checked directly rather than merely asserted in the
    # docstring.
    var e = empty[dtype, 5](ctx)
    for i in range(5):
        assert_equal(e[i], Scalar[dtype](0))


def test_eye_is_the_identity_matrix() raises:
    var ctx = DeviceContext(api="cpu")
    var m = eye[3, dtype](ctx)
    var v = m.view()
    for r in range(3):
        for c in range(3):
            var expected = Scalar[dtype](1) if r == c else Scalar[dtype](0)
            assert_equal(v[r, c], expected)


def test_linspace_matches_numpy_endpoints_and_spacing() raises:
    var ctx = DeviceContext(api="cpu")
    var ls = linspace[5, dtype](0, 1, ctx=ctx)
    var expected = [0.0, 0.25, 0.5, 0.75, 1.0]
    for i in range(5):
        assert_almost_equal(ls[i], Scalar[dtype](expected[i]))


def test_linspace_of_one_point_returns_start() raises:
    var ctx = DeviceContext(api="cpu")
    var ls = linspace[1, dtype](3, 9, ctx=ctx)
    assert_equal(ls[0], Scalar[dtype](3))


def test_logspace_matches_base_to_the_linspace_power() raises:
    var ctx = DeviceContext(api="cpu")
    var lg = logspace[3, dtype](0, 2, ctx=ctx)
    var expected = [1.0, 10.0, 100.0]
    for i in range(3):
        assert_almost_equal(lg[i], Scalar[dtype](expected[i]))


def test_zeros_like_matches_source_shape() raises:
    var ctx = DeviceContext(api="cpu")
    var src = zeros[dtype, 2, 3](ctx)
    var z = zeros_like(src)
    assert_equal(z.num_elements, src.num_elements)
    for i in range(6):
        assert_equal(z[i], Scalar[dtype](0))


def test_ones_like_matches_source_shape() raises:
    var ctx = DeviceContext(api="cpu")
    var src = zeros[dtype, 4](ctx)
    var o = ones_like(src)
    for i in range(4):
        assert_equal(o[i], Scalar[dtype](1))


def test_full_like_uses_the_given_fill_value() raises:
    var ctx = DeviceContext(api="cpu")
    var src = zeros[dtype, 3](ctx)
    var f = full_like(src, Scalar[dtype](5))
    for i in range(3):
        assert_equal(f[i], Scalar[dtype](5))


def test_empty_like_matches_source_shape() raises:
    var ctx = DeviceContext(api="cpu")
    var src = zeros[dtype, 2, 2](ctx)
    var _unused = empty_like(src)
    assert_equal(Static[dtype, 2, 2].num_elements, src.num_elements)


def test_transpose_swaps_rows_and_columns() raises:
    var ctx = DeviceContext(api="cpu")
    var m = full[dtype, 2, 3](0, ctx=ctx)
    var v = m.view()
    v[0, 0] = 1
    v[0, 1] = 2
    v[0, 2] = 3
    v[1, 0] = 4
    v[1, 1] = 5
    v[1, 2] = 6

    var t = transpose(m)
    var tv = t.view()
    for r in range(2):
        for c in range(3):
            assert_equal(tv[c, r], v[r, c])


def test_transpose_is_its_own_inverse() raises:
    var ctx = DeviceContext(api="cpu")
    var m = full[dtype, 3, 2](0, ctx=ctx)
    var v = m.view()
    var counter = 0
    for r in range(3):
        for c in range(2):
            v[r, c] = Scalar[dtype](counter)
            counter += 1

    var t = transpose(m)
    var tt = transpose(t)
    var v2 = tt.view()
    for r in range(3):
        for c in range(2):
            assert_equal(v2[r, c], v[r, c])


def test_transpose_default_is_the_host_path() raises:
    """`transpose` gained a `gpu` parameter, and `False` has to still mean
    what the unparameterized spelling meant.

    The `True` branch is a different implementation -- an `elementwise`
    gather rather than `linalg.transpose`, because every path MAX's kernel
    can reach is host code -- so it
    cannot be checked here: no CI runner has a device. What is checkable is
    that adding the parameter did not move the default off the MAX call.
    """
    var ctx = DeviceContext(api="cpu")
    var m = full[dtype, 2, 3](0, ctx=ctx)
    var v = m.view()
    var counter = 0
    for r in range(2):
        for c in range(3):
            v[r, c] = Scalar[dtype](counter)
            counter += 1

    var implicit = transpose(m)
    var explicit = transpose[dtype, 2, 3, False](m)
    var iv = implicit.view()
    var ev = explicit.view()
    for r in range(3):
        for c in range(2):
            assert_equal(iv[r, c], ev[r, c])


def test_squeeze_drops_a_leading_size_one_axis() raises:
    var ctx = DeviceContext(api="cpu")
    var row = full[dtype, 1, 4](0, ctx=ctx)
    var rv = row.view()
    for i in range(4):
        rv[0, i] = Scalar[dtype](i)

    var sq = squeeze(row)
    assert_equal(sq.num_elements, 4)
    for i in range(4):
        assert_equal(sq[i], Scalar[dtype](i))


def test_squeeze_drops_a_trailing_size_one_axis() raises:
    var ctx = DeviceContext(api="cpu")
    var col = full[dtype, 4, 1](0, ctx=ctx)
    var cv = col.view()
    for i in range(4):
        cv[i, 0] = Scalar[dtype](i * 2)

    var sq = squeeze(col)
    assert_equal(sq.num_elements, 4)
    for i in range(4):
        assert_equal(sq[i], Scalar[dtype](i * 2))


def test_stack_along_axis_zero() raises:
    var ctx = DeviceContext(api="cpu")
    var a = linspace[3, dtype](0, 2, ctx=ctx)
    var b = linspace[3, dtype](10, 12, ctx=ctx)
    var st = stack(a, b)
    var sv = st.view()
    for i in range(3):
        assert_equal(sv[0, i], a[i])
        assert_equal(sv[1, i], b[i])


def test_tensor_survives_the_call_that_built_it() raises:
    var ctx = DeviceContext(api="cpu")
    # The whole reason `Tensor` exists rather than returning a bare
    # `TileTensor`: the value returned by a factory function must remain
    # valid after the function that built it has returned and its locals
    # have been destroyed. Allocate a bunch of unrelated memory afterward
    # to make a use-after-free regression likely to show up as corruption
    # rather than silently passing.
    var t = full[dtype, 64](42, ctx=ctx)
    var junk = List[List[Scalar[dtype]]]()
    for i in range(256):
        junk.append(List[Scalar[dtype]](length=64, fill=Scalar[dtype](i)))
    for i in range(64):
        assert_equal(t[i], Scalar[dtype](42))


def test_arange_starts_at_start_and_steps_by_step() raises:
    var ctx = DeviceContext(api="cpu")
    var a = arange[5, dtype](2, 3, ctx=ctx)
    assert_equal(a.num_elements, 5)
    for i in range(5):
        assert_equal(a[i], Scalar[dtype](2 + 3 * i))


def test_arange_defaults_to_zero_start_unit_step() raises:
    var ctx = DeviceContext(api="cpu")
    var a = arange[4, dtype](ctx=ctx)
    for i in range(4):
        assert_equal(a[i], Scalar[dtype](i))


def test_reshape_preserves_row_major_element_order() raises:
    var ctx = DeviceContext(api="cpu")
    var a = arange[6, dtype](ctx=ctx)
    var m = reshape[rows=2, cols=3](a)
    assert_equal(m.dim[0](), 2)
    assert_equal(m.dim[1](), 3)
    # Row-major: flat index i lands at (i // 3, i % 3), so the flat read
    # back out has to match the source exactly.
    for i in range(6):
        assert_equal(m[i], Scalar[dtype](i))


def test_ravel_inverts_reshape() raises:
    var ctx = DeviceContext(api="cpu")
    var a = arange[6, dtype](ctx=ctx)
    var m = reshape[rows=3, cols=2](a)
    var flat = ravel(m)
    assert_equal(flat.num_elements, 6)
    for i in range(6):
        assert_equal(flat[i], a[i])


def test_ravel_flattens_a_2d_tensor_row_by_row() raises:
    var ctx = DeviceContext(api="cpu")
    var m = zeros[dtype, 2, 2](ctx)
    var v = m.view()
    v[0, 0] = 1
    v[0, 1] = 2
    v[1, 0] = 3
    v[1, 1] = 4
    var flat = ravel(m)
    assert_equal(flat[0], Scalar[dtype](1))
    assert_equal(flat[1], Scalar[dtype](2))
    assert_equal(flat[2], Scalar[dtype](3))
    assert_equal(flat[3], Scalar[dtype](4))


def test_concatenate_joins_two_rank1_tensors_end_to_end() raises:
    var ctx = DeviceContext(api="cpu")
    var a = arange[3, dtype](ctx=ctx)
    var b = arange[2, dtype](100, ctx=ctx)
    var c = concatenate(a, b)
    assert_equal(c.num_elements, 5)
    for i in range(3):
        assert_equal(c[i], a[i])
    for i in range(2):
        assert_equal(c[3 + i], b[i])


def test_split_inverts_concatenate() raises:
    var ctx = DeviceContext(api="cpu")
    var a = arange[3, dtype](ctx=ctx)
    var b = arange[4, dtype](50, ctx=ctx)
    var joined = concatenate(a, b)
    var parts = split[at=3](joined)
    assert_equal(parts[0].num_elements, 3)
    assert_equal(parts[1].num_elements, 4)
    for i in range(3):
        assert_equal(parts[0][i], a[i])
    for i in range(4):
        assert_equal(parts[1][i], b[i])


def test_split_at_an_endpoint_gives_one_empty_side() raises:
    var ctx = DeviceContext(api="cpu")
    var a = arange[3, dtype](ctx=ctx)
    var parts = split[at=0](a)
    assert_equal(parts[0].num_elements, 0)
    assert_equal(parts[1].num_elements, 3)
    for i in range(3):
        assert_equal(parts[1][i], a[i])


def test_rank_2_indexing_matches_the_flat_index() raises:
    var m = zeros[dtype, 2, 3]()
    m[1, 2] = Scalar[dtype](7)
    # (1, 2) is flat index 1*3 + 2 = 5.
    assert_almost_equal(m[5], Scalar[dtype](7))
    assert_almost_equal(m[1, 2], Scalar[dtype](7))


def test_rank_2_indexing_agrees_with_the_view() raises:
    var m = zeros[dtype, 3, 2]()
    var counter = 0
    for r in range(3):
        for c in range(2):
            m[r, c] = Scalar[dtype](counter)
            counter += 1
    var v = m.view()
    for r in range(3):
        for c in range(2):
            assert_almost_equal(m[r, c], v[r, c])


# ------------------------------------------------------------------
# The same factory names at the conformer layer
# ------------------------------------------------------------------


def test_array_factories_fill_what_they_say() raises:
    var z = zeros[P, 3]()
    var o = ones[P, 3]()
    var f = full[P, 3](2.5)
    for i in range(3):
        assert_almost_equal(z[i].v, 0.0)
        assert_almost_equal(o[i].v, 1.0)
        assert_almost_equal(f[i].v, 2.5)


def test_array_eye_agrees_with_lifting_the_tensor_one() raises:
    # The two spellings of the same identity: the conformer factory exists
    # so a caller reaching for `numax.linalg` does not have to build a
    # tensor and immediately copy it out again.
    var direct = eye[P, 3]()
    var lifted = to_array[P](eye[3](DeviceContext(api="cpu")))
    for i in range(9):
        assert_almost_equal(direct[i].v, lifted[i].v)


def test_the_tensor_factories_still_resolve() raises:
    # The conformer overloads share their names with the tensor ones, so
    # this fails if adding them shadowed rather than overloaded.
    var ctx = DeviceContext(api="cpu")
    assert_equal(zeros[dtype, 4](ctx).size(), 4)
    assert_equal(ones[dtype, 2, 3](ctx).size(), 6)
    assert_equal(eye[3](ctx).size(), 9)


def test_every_dtype_alias_names_the_dtype_it_looks_like() raises:
    # Spelled out rather than derived: a typo in one alias is exactly what
    # this catches, and deriving the expected side from the same table
    # would reproduce the typo.
    assert_equal(f16, DType.float16)
    assert_equal(bf16, DType.bfloat16)
    assert_equal(f32, DType.float32)
    assert_equal(f64, DType.float64)
    assert_equal(f8e3m4, DType.float8_e3m4)
    assert_equal(f8e4m3fn, DType.float8_e4m3fn)
    assert_equal(f8e4m3fnuz, DType.float8_e4m3fnuz)
    assert_equal(f8e5m2, DType.float8_e5m2)
    assert_equal(f8e5m2fnuz, DType.float8_e5m2fnuz)
    assert_equal(i8, DType.int8)
    assert_equal(i16, DType.int16)
    assert_equal(i32, DType.int32)
    assert_equal(i64, DType.int64)
    assert_equal(u8, DType.uint8)
    assert_equal(u16, DType.uint16)
    assert_equal(u32, DType.uint32)
    assert_equal(u64, DType.uint64)
    assert_equal(bool, DType.bool)


def _rank_three() raises -> Static[dtype, 2, 3, 4]:
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, 2, 3, 4](ctx)
    for i in range(24):
        a[i] = Scalar[dtype](i)
    return a^


def test_transpose_permutes_axes_at_rank_three() raises:
    # numpy.transpose(a, (2, 0, 1)) on a (2, 3, 4) gives a (4, 2, 3), and
    # a[i, j, k] lands at out[k, i, j].
    var a = _rank_three()

    var p = transpose(a, 2, 0, 1)
    assert_equal(p.dim_at(0), 4)
    assert_equal(p.dim_at(1), 2)
    assert_equal(p.dim_at(2), 3)

    var source = a.to_host()
    var out = p.to_host()
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert_equal(
                    out[(k * 2 + i) * 3 + j], source[(i * 3 + j) * 4 + k]
                )


def test_transpose_at_rank_two_still_keeps_its_extents_in_the_type() raises:
    # The rank-n overload must not shadow the rank-2 one, which is the only
    # one whose result shape is compile-time.
    var ctx = DeviceContext(api="cpu")
    var m = full[dtype, 2, 3](0, ctx=ctx)
    m[1] = 7
    var t: Static[dtype, 3, 2] = transpose(m)
    assert_equal(t.view()[1, 0], 7)


def test_transpose_rejects_a_bad_permutation() raises:
    var a = _rank_three()
    var repeated = False
    try:
        _ = transpose(a, 0, 0, 1)
    except:
        repeated = True
    assert_equal(repeated, True)

    var wrong_count = False
    try:
        _ = transpose(a, 1, 0)
    except:
        wrong_count = True
    assert_equal(wrong_count, True)


def test_swapaxes_exchanges_exactly_two_axes() raises:
    # numpy.swapaxes(a, 0, 2) on a (2, 3, 4) gives a (4, 3, 2).
    var a = _rank_three()
    var s = swapaxes(a, 0, 2)
    assert_equal(s.dim_at(0), 4)
    assert_equal(s.dim_at(1), 3)
    assert_equal(s.dim_at(2), 2)

    var source = a.to_host()
    var out = s.to_host()
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert_equal(
                    out[(k * 3 + j) * 2 + i], source[(i * 3 + j) * 4 + k]
                )


def test_moveaxis_is_not_swapaxes() raises:
    # Moving axis 0 to position 2 of a (2, 3, 4) gives a (3, 4, 2) -- the
    # order (1, 2, 0). Swapping them would give a (4, 3, 2) instead, which
    # is the distinction this test exists for.
    var a = _rank_three()
    var moved = moveaxis(a, 0, 2)
    assert_equal(moved.dim_at(0), 3)
    assert_equal(moved.dim_at(1), 4)
    assert_equal(moved.dim_at(2), 2)

    var swapped = swapaxes(a, 0, 2)
    assert_equal(swapped.dim_at(0), 4)


def test_swapaxes_accepts_negative_axes() raises:
    var a = _rank_three()
    var s = swapaxes(a, -1, 0)
    assert_equal(s.dim_at(0), 4)
    assert_equal(s.dim_at(2), 2)


def _grid2[
    rows: Int, cols: Int
](values: List[Float64]) raises -> Static[dtype, rows, cols]:
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, rows, cols](ctx)
    for i in range(rows * cols):
        a[i] = Scalar[dtype](values[i])
    return a^


def test_concatenate_along_an_axis_sums_only_that_extent() raises:
    # numpy.concatenate((a, b), axis=1) on a (2, 3) and a (2, 2) gives a
    # (2, 5) whose rows are [1, 2, 3, 7, 8] and [4, 5, 6, 9, 10].
    var a = _grid2[2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var b = _grid2[2, 2]([7.0, 8.0, 9.0, 10.0])

    var joined = concatenate[axis=1](a, b)
    assert_equal(joined.dim_at(0), 2)
    assert_equal(joined.dim_at(1), 5)
    var out = joined.to_host()
    var expected = [1.0, 2.0, 3.0, 7.0, 8.0, 4.0, 5.0, 6.0, 9.0, 10.0]
    for i in range(10):
        assert_equal(out[i], Scalar[dtype](expected[i]))


def test_concatenate_along_axis_zero_stacks_rows() raises:
    var a = _grid2[2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var b = _grid2[1, 3]([7.0, 8.0, 9.0])

    var joined = concatenate[axis=0](a, b)
    assert_equal(joined.dim_at(0), 3)
    assert_equal(joined.dim_at(1), 3)
    assert_equal(joined.to_host()[6], 7.0)


def test_concatenate_rejects_a_mismatched_other_axis() raises:
    var a = _grid2[2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var b = _grid2[3, 2]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var raised = False
    try:
        _ = concatenate[axis=1](a, b)
    except:
        raised = True
    assert_equal(raised, True)


def test_stack_at_an_axis_adds_one_rather_than_growing_one() raises:
    # numpy.stack((a, c), axis=1) on two (2, 3) gives a (2, 2, 3), where
    # concatenating them at axis=1 would give a (2, 6).
    var a = _grid2[2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var c = _grid2[2, 3]([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    var stacked = stack[axis=1](a, c)
    assert_equal(stacked.dim_at(0), 2)
    assert_equal(stacked.dim_at(1), 2)
    assert_equal(stacked.dim_at(2), 3)
    var out = stacked.to_host()
    var expected = [
        1.0,
        2.0,
        3.0,
        10.0,
        20.0,
        30.0,
        4.0,
        5.0,
        6.0,
        40.0,
        50.0,
        60.0,
    ]
    for i in range(12):
        assert_equal(out[i], Scalar[dtype](expected[i]))


def test_stack_at_the_trailing_position_interleaves() raises:
    # axis == rank appends the new axis, so (3,) and (3,) give a (3, 2).
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, 3](ctx)
    var b = zeros[dtype, 3](ctx)
    for i in range(3):
        a[i] = Scalar[dtype](i + 1)
        b[i] = Scalar[dtype](10 * (i + 1))

    var stacked = stack[axis=1](a, b)
    assert_equal(stacked.dim_at(0), 3)
    assert_equal(stacked.dim_at(1), 2)
    var out = stacked.to_host()
    assert_equal(out[0], 1.0)
    assert_equal(out[1], 10.0)
    assert_equal(out[2], 2.0)


def test_split_at_an_axis_inverts_concatenate() raises:
    # The claim worth pinning: the two are one operation read each way.
    var a = _grid2[2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var b = _grid2[2, 2]([7.0, 8.0, 9.0, 10.0])
    var joined = concatenate[axis=1](a, b)

    var parts = split[axis=1](joined, 3)
    assert_equal(parts[0].dim_at(1), 3)
    assert_equal(parts[1].dim_at(1), 2)

    var head = parts[0].to_host()
    var source = a.to_host()
    for i in range(6):
        assert_equal(head[i], source[i])
    var tail = parts[1].to_host()
    var other = b.to_host()
    for i in range(4):
        assert_equal(tail[i], other[i])


def test_split_rejects_a_cut_past_the_axis() raises:
    var a = _grid2[2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var raised = False
    try:
        _ = split[axis=1](a, 4)
    except:
        raised = True
    assert_equal(raised, True)


def test_the_rank_one_concatenate_still_keeps_its_length_in_the_type() raises:
    # The axis overloads must not shadow the rank-1 ones, whose result
    # length is compile-time.
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, 2](ctx)
    var b = zeros[dtype, 3](ctx)
    a[0] = 1
    b[0] = 9
    var joined: Static[dtype, 5] = concatenate(a, b)
    assert_equal(joined.to_host()[2], 9.0)


def test_expand_dims_inserts_a_size_one_axis() raises:
    # numpy.expand_dims on a (2, 3) at axis=1 gives a (2, 1, 3); the
    # elements keep their row-major order.
    var a = _grid2[2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    var e = expand_dims[axis=1](a)
    assert_equal(e.dim_at(0), 2)
    assert_equal(e.dim_at(1), 1)
    assert_equal(e.dim_at(2), 3)
    var out = e.to_host()
    var source = a.to_host()
    for i in range(6):
        assert_equal(out[i], source[i])


def test_roll_shifts_cyclically_and_inverts() raises:
    # numpy.roll([[1,2,3],[4,5,6]], 1, axis=1) == [[3,1,2],[6,4,5]]
    var a = _grid2[2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    var rolled = roll[axis=1](a, 1)
    var expected = [3.0, 1.0, 2.0, 6.0, 4.0, 5.0]
    for i in range(6):
        assert_equal(rolled[i], Scalar[dtype](expected[i]))

    # The opposite shift puts it back, which is what "cyclic" has to mean.
    var back = roll[axis=1](rolled, -1)
    var source = a.to_host()
    for i in range(6):
        assert_equal(back[i], source[i])


def test_roll_along_the_outer_axis() raises:
    var a = _grid2[2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var rolled = roll[axis=0](a, 1)
    assert_equal(rolled[0], 4.0)
    assert_equal(rolled[3], 1.0)


def test_tile_repeats_the_whole_block() raises:
    # numpy.tile on a (2, 3) by (2, 1) gives a (4, 3) that is `a` above `a`.
    var a = _grid2[2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    var tiled = tile(a, 2, 1)
    assert_equal(tiled.dim_at(0), 4)
    assert_equal(tiled.dim_at(1), 3)
    var out = tiled.to_host()
    var source = a.to_host()
    for i in range(6):
        assert_equal(out[i], source[i])
        assert_equal(out[6 + i], source[i])


def test_repeat_repeats_each_element_not_the_block() raises:
    # The distinction from `tile`: numpy.repeat(a, 2, axis=1) gives
    # [[1, 1, 2, 2, 3, 3], ...] where numpy.tile(a, (1, 2)) gives
    # [[1, 2, 3, 1, 2, 3], ...].
    var a = _grid2[2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    var repeated = repeat[axis=1](a, 2)
    assert_equal(repeated.dim_at(0), 2)
    assert_equal(repeated.dim_at(1), 6)
    var expected = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    var out = repeated.to_host()
    for i in range(6):
        assert_equal(out[i], Scalar[dtype](expected[i]))

    var tiled = tile(a, 1, 2)
    var tiled_out = tiled.to_host()
    var tiled_expected = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    for i in range(6):
        assert_equal(tiled_out[i], Scalar[dtype](tiled_expected[i]))


def test_tile_rejects_a_count_per_missing_axis() raises:
    var a = _grid2[2, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var raised = False
    try:
        _ = tile(a, 2)
    except:
        raised = True
    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
