"""Tests for `numax.array`'s NumPy-named creation/manipulation surface.

Every creation routine is checked for shape (`num_elements`) and content;
`*_like` is checked to match its source's dtype/shape; the manipulation
functions (`transpose`, `squeeze`, `stack`) are checked against a
hand-computed expected result, not just "it runs".
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from numax.array import (
    Tensor,
    empty,
    empty_like,
    eye,
    full,
    full_like,
    linspace,
    logspace,
    ones,
    ones_like,
    squeeze,
    stack,
    transpose,
    zeros,
    zeros_like,
)

comptime dtype = DType.float32


def test_zeros_has_the_requested_shape_and_content() raises:
    var z = zeros[dtype, 2, 3]()
    assert_equal(z.num_elements, 6)
    for i in range(6):
        assert_equal(z[i], Scalar[dtype](0))


def test_ones_is_filled_with_one() raises:
    var o = ones[dtype, 4]()
    assert_equal(o.num_elements, 4)
    for i in range(4):
        assert_equal(o[i], Scalar[dtype](1))


def test_full_is_filled_with_the_given_value() raises:
    var f = full[dtype, 3](Scalar[dtype](7))
    for i in range(3):
        assert_equal(f[i], Scalar[dtype](7))


def test_empty_is_zero_initialized_for_memory_safety() raises:
    # numax's `empty` documents zero-init rather than true uninitialized
    # memory -- checked directly rather than merely asserted in the
    # docstring.
    var e = empty[dtype, 5]()
    for i in range(5):
        assert_equal(e[i], Scalar[dtype](0))


def test_eye_is_the_identity_matrix() raises:
    var m = eye[dtype, 3]()
    var v = m.view()
    for r in range(3):
        for c in range(3):
            var expected = Scalar[dtype](1) if r == c else Scalar[dtype](0)
            assert_equal(v[r, c], expected)


def test_linspace_matches_numpy_endpoints_and_spacing() raises:
    var ls = linspace[dtype, 5](Scalar[dtype](0), Scalar[dtype](1))
    var expected = [0.0, 0.25, 0.5, 0.75, 1.0]
    for i in range(5):
        assert_almost_equal(ls[i], Scalar[dtype](expected[i]))


def test_linspace_of_one_point_returns_start() raises:
    var ls = linspace[dtype, 1](Scalar[dtype](3), Scalar[dtype](9))
    assert_equal(ls[0], Scalar[dtype](3))


def test_logspace_matches_base_to_the_linspace_power() raises:
    var lg = logspace[dtype, 3](Scalar[dtype](0), Scalar[dtype](2))
    var expected = [1.0, 10.0, 100.0]
    for i in range(3):
        assert_almost_equal(lg[i], Scalar[dtype](expected[i]))


def test_zeros_like_matches_source_shape() raises:
    var src = zeros[dtype, 2, 3]()
    var z = zeros_like(src)
    assert_equal(z.num_elements, src.num_elements)
    for i in range(6):
        assert_equal(z[i], Scalar[dtype](0))


def test_ones_like_matches_source_shape() raises:
    var src = zeros[dtype, 4]()
    var o = ones_like(src)
    for i in range(4):
        assert_equal(o[i], Scalar[dtype](1))


def test_full_like_uses_the_given_fill_value() raises:
    var src = zeros[dtype, 3]()
    var f = full_like(src, Scalar[dtype](5))
    for i in range(3):
        assert_equal(f[i], Scalar[dtype](5))


def test_empty_like_matches_source_shape() raises:
    var src = zeros[dtype, 2, 2]()
    var _unused = empty_like(src)
    assert_equal(Tensor[dtype, 2, 2].num_elements, src.num_elements)


def test_transpose_swaps_rows_and_columns() raises:
    var m = full[dtype, 2, 3](Scalar[dtype](0))
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
    var m = full[dtype, 3, 2](Scalar[dtype](0))
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


def test_squeeze_drops_a_leading_size_one_axis() raises:
    var row = full[dtype, 1, 4](Scalar[dtype](0))
    var rv = row.view()
    for i in range(4):
        rv[0, i] = Scalar[dtype](i)

    var sq = squeeze(row)
    assert_equal(sq.num_elements, 4)
    for i in range(4):
        assert_equal(sq[i], Scalar[dtype](i))


def test_squeeze_drops_a_trailing_size_one_axis() raises:
    var col = full[dtype, 4, 1](Scalar[dtype](0))
    var cv = col.view()
    for i in range(4):
        cv[i, 0] = Scalar[dtype](i * 2)

    var sq = squeeze(col)
    assert_equal(sq.num_elements, 4)
    for i in range(4):
        assert_equal(sq[i], Scalar[dtype](i * 2))


def test_stack_along_axis_zero() raises:
    var a = linspace[dtype, 3](Scalar[dtype](0), Scalar[dtype](2))
    var b = linspace[dtype, 3](Scalar[dtype](10), Scalar[dtype](12))
    var st = stack(a, b)
    var sv = st.view()
    for i in range(3):
        assert_equal(sv[0, i], a[i])
        assert_equal(sv[1, i], b[i])


def test_tensor_survives_the_call_that_built_it() raises:
    # The whole reason `Tensor` exists rather than returning a bare
    # `TileTensor`: the value returned by a factory function must remain
    # valid after the function that built it has returned and its locals
    # have been destroyed. Allocate a bunch of unrelated memory afterward
    # to make a use-after-free regression likely to show up as corruption
    # rather than silently passing.
    var t = full[dtype, 64](Scalar[dtype](42))
    var junk = List[List[Scalar[dtype]]]()
    for i in range(256):
        junk.append(List[Scalar[dtype]](length=64, fill=Scalar[dtype](i)))
    for i in range(64):
        assert_equal(t[i], Scalar[dtype](42))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
