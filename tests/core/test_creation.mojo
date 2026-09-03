"""Tests for the creation and manipulation names added for NumPy parity.

Each is checked against a hand-written expected matrix in row-major order,
including the shapes that are compile-time functions of the inputs
(`vander`'s columns, `vstack`/`hstack`'s joined extent, `diagflat`'s
square).
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.array import (
    Tensor,
    copy,
    diag,
    diagflat,
    diagonal,
    flip,
    geomspace,
    hstack,
    identity,
    meshgrid,
    tri,
    tril,
    triu,
    vander,
    vstack,
    zeros,
)

comptime dtype = DType.float64


def _t[n: Int](values: List[Float64]) raises -> Tensor[dtype, n]:
    var ctx = DeviceContext(api="cpu")
    var elements = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        elements.append(Scalar[dtype](values[i]))
    return Tensor[dtype, n](ctx, elements^)


def _m[
    rows: Int, cols: Int
](values: List[Float64]) raises -> Tensor[dtype, rows, cols]:
    var ctx = DeviceContext(api="cpu")
    var elements = List[Scalar[dtype]](capacity=rows * cols)
    for i in range(rows * cols):
        elements.append(Scalar[dtype](values[i]))
    return Tensor[dtype, rows, cols](ctx, elements^)


def test_geomspace_is_a_geometric_progression() raises:
    var ctx = DeviceContext(api="cpu")
    var g = geomspace[dtype, 4](ctx, 1.0, 1000.0).to_host()
    assert_almost_equal(g[0], 1.0)
    assert_almost_equal(g[1], 10.0)
    assert_almost_equal(g[2], 100.0)
    assert_almost_equal(g[3], 1000.0)


def test_geomspace_of_one_point_returns_start() raises:
    var ctx = DeviceContext(api="cpu")
    var g = geomspace[dtype, 1](ctx, 7.0, 9.0).to_host()
    assert_almost_equal(g[0], 7.0)


def test_identity_matches_eye() raises:
    var ctx = DeviceContext(api="cpu")
    var i3 = identity[dtype, 3](ctx).to_host()
    for r in range(3):
        for c in range(3):
            var expected = 1.0 if r == c else 0.0
            assert_equal(i3[r * 3 + c], Scalar[dtype](expected))


def test_diag_and_diagonal_are_inverses() raises:
    var v = _t[3]([1.0, 2.0, 3.0])
    var m = diag(v)
    var values = m.to_host()
    assert_equal(values[0], 1.0)
    assert_equal(values[4], 2.0)
    assert_equal(values[8], 3.0)
    assert_equal(values[1], 0.0)
    var back = diagonal(m).to_host()
    assert_equal(back[0], 1.0)
    assert_equal(back[1], 2.0)
    assert_equal(back[2], 3.0)


def test_diagflat_squares_a_flattened_input() raises:
    var m = _m[2, 2]([1.0, 2.0, 3.0, 4.0])
    var flat = diagflat(m)
    assert_equal(flat.num_elements, 16)
    var values = flat.to_host()
    assert_equal(values[0], 1.0)
    assert_equal(values[5], 2.0)
    assert_equal(values[10], 3.0)
    assert_equal(values[15], 4.0)


def test_tri_is_lower_triangular_ones() raises:
    var ctx = DeviceContext(api="cpu")
    var t = tri[dtype, 3](ctx).to_host()
    assert_equal(t[0], 1.0)
    assert_equal(t[1], 0.0)
    assert_equal(t[3], 1.0)
    assert_equal(t[4], 1.0)
    assert_equal(t[8], 1.0)


def test_tril_and_triu_partition_the_matrix() raises:
    var m = _m[3, 3]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    var lower = tril(m).to_host()
    var upper = triu(m).to_host()
    var original = m.to_host()
    # the diagonal appears in both, everything else in exactly one
    for i in range(9):
        var r = i // 3
        var c = i % 3
        if r == c:
            assert_equal(lower[i], original[i])
            assert_equal(upper[i], original[i])
        elif r > c:
            assert_equal(lower[i], original[i])
            assert_equal(upper[i], 0.0)
        else:
            assert_equal(lower[i], 0.0)
            assert_equal(upper[i], original[i])


def test_vander_has_decreasing_powers() raises:
    var x = _t[3]([1.0, 2.0, 3.0])
    var v = vander[dtype, 3, 3](x)
    assert_equal(v.num_elements, 9)
    var values = v.to_host()
    # row for 2.0 is [4, 2, 1]
    assert_almost_equal(values[3], 4.0)
    assert_almost_equal(values[4], 2.0)
    assert_almost_equal(values[5], 1.0)


def test_meshgrid_broadcasts_both_directions() raises:
    var x = _t[3]([1.0, 2.0, 3.0])
    var y = _t[2]([10.0, 20.0])
    var grids = meshgrid(x, y)
    var xx = grids[0].to_host()
    var yy = grids[1].to_host()
    assert_equal(grids[0].num_elements, 6)
    assert_equal(xx[0], 1.0)
    assert_equal(xx[1], 2.0)
    assert_equal(xx[3], 1.0)
    assert_equal(yy[0], 10.0)
    assert_equal(yy[3], 20.0)


def test_flip_reverses() raises:
    var a = _t[4]([1.0, 2.0, 3.0, 4.0])
    var f = flip(a).to_host()
    assert_equal(f[0], 4.0)
    assert_equal(f[3], 1.0)


def test_copy_is_independent_of_its_source() raises:
    var a = _t[3]([1.0, 2.0, 3.0])
    var b = copy(a)
    b[0] = 99.0
    assert_equal(a.to_host()[0], 1.0)
    assert_equal(b.to_host()[0], 99.0)


def test_vstack_joins_rows() raises:
    var a = _m[1, 3]([1.0, 2.0, 3.0])
    var b = _m[2, 3]([4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    var stacked = vstack(a, b)
    assert_equal(stacked.num_elements, 9)
    var values = stacked.to_host()
    assert_equal(values[0], 1.0)
    assert_equal(values[3], 4.0)
    assert_equal(values[8], 9.0)


def test_hstack_joins_columns() raises:
    var a = _m[2, 1]([1.0, 3.0])
    var b = _m[2, 2]([10.0, 20.0, 30.0, 40.0])
    var joined = hstack(a, b)
    assert_equal(joined.num_elements, 6)
    var values = joined.to_host()
    assert_equal(values[0], 1.0)
    assert_equal(values[1], 10.0)
    assert_equal(values[2], 20.0)
    assert_equal(values[3], 3.0)
    assert_equal(values[4], 30.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
