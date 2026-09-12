"""Tests for `cross`, `tensordot`, `tensorsolve` and `tensorinv` over
`Tensor`: each against the definition written out as loops on the host,
and against the sibling it reduces to (`tensordot` at `axes=1` on two
matrices is `matmul`; `axes=0` on two vectors is `outer`)."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

from max.gpu.host import DeviceContext

from numax.core.array import Static, zeros
from numax.linalg import (
    cross,
    inverse,
    matmul,
    outer,
    tensordot,
    tensorinv,
    tensorsolve,
)

comptime dtype = DType.float64


def _filled[*dims: Int](values: List[Float64]) raises -> Static[dtype, *dims]:
    var ctx = DeviceContext(api="cpu")
    var out = zeros[dtype, *dims](ctx)
    var host = out.to_host()
    for i in range(len(values)):
        host[i] = Scalar[dtype](values[i])
    out.copy_from_host(host)
    return out^


def _ramp[
    *dims: Int
](count: Int, offset: Float64) raises -> Static[dtype, *dims]:
    var values = List[Float64](capacity=count)
    for i in range(count):
        values.append(Float64(i) * 0.5 + offset)
    return _filled[*dims](values)


def test_cross_of_the_axes_and_antisymmetry() raises:
    var x = _filled[3]([1.0, 0.0, 0.0])
    var y = _filled[3]([0.0, 1.0, 0.0])
    var z = cross(x, y).to_host()
    assert_equal(Float64(z[0]), 0.0)
    assert_equal(Float64(z[1]), 0.0)
    assert_equal(Float64(z[2]), 1.0)
    var a = _filled[3]([1.0, 2.0, 3.0])
    var b = _filled[3]([-4.0, 0.5, 2.0])
    var ab = cross(a, b).to_host()
    # (2*2 - 3*0.5, 3*(-4) - 1*2, 1*0.5 - 2*(-4)) = (2.5, -14, 8.5)
    assert_almost_equal(Float64(ab[0]), 2.5)
    assert_almost_equal(Float64(ab[1]), -14.0)
    assert_almost_equal(Float64(ab[2]), 8.5)
    var a2 = _filled[3]([1.0, 2.0, 3.0])
    var b2 = _filled[3]([-4.0, 0.5, 2.0])
    var ba = cross(b2, a2).to_host()
    for i in range(3):
        assert_equal(Float64(ba[i]), -Float64(ab[i]))


def test_cross_on_rows() raises:
    var a = _filled[2, 3]([1.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    var b = _filled[2, 3]([0.0, 1.0, 0.0, -4.0, 0.5, 2.0])
    var c = cross(a, b).to_host()
    var want: List[Float64] = [0.0, 0.0, 1.0, 2.5, -14.0, 8.5]
    for i in range(6):
        assert_almost_equal(Float64(c[i]), want[i])


def test_tensordot_at_one_axis_is_matmul_and_at_zero_is_outer() raises:
    var a = _ramp[2, 3](6, 1.0)
    var b = _ramp[3, 4](12, -2.0)
    var a2 = _ramp[2, 3](6, 1.0)
    var b2 = _ramp[3, 4](12, -2.0)
    var contracted = tensordot[axes=1](a, b)
    assert_equal(contracted.dim_at(0), 2)
    assert_equal(contracted.dim_at(1), 4)
    var direct = matmul(a2, b2).to_host()
    var got = contracted.to_host()
    for i in range(8):
        assert_almost_equal(Float64(got[i]), Float64(direct[i]), atol=1e-12)
    var u = _ramp[2](2, 1.0)
    var v = _ramp[3](3, 0.25)
    var u2 = _ramp[2](2, 1.0)
    var v2 = _ramp[3](3, 0.25)
    var outer_form = tensordot[axes=0](u, v)
    assert_equal(outer_form.dim_at(0), 2)
    assert_equal(outer_form.dim_at(1), 3)
    var reference = outer(u2, v2).to_host()
    var host = outer_form.to_host()
    for i in range(6):
        assert_equal(Float64(host[i]), Float64(reference[i]))


def test_tensordot_contracts_two_axes_of_a_rank_three_tensor() raises:
    var a = _ramp[2, 3, 4](24, 0.0)
    var b = _ramp[3, 4](12, 1.0)
    var c = tensordot[axes=2](a, b)
    assert_equal(c.dim_at(0), 2)
    var ah = a.to_host()
    var bh = b.to_host()
    var ch = c.to_host()
    for i in range(2):
        var acc = 0.0
        for j in range(3):
            for k in range(4):
                acc += Float64(ah[i * 12 + j * 4 + k]) * Float64(bh[j * 4 + k])
        assert_almost_equal(Float64(ch[i]), acc, atol=1e-12)
    # One axis of a rank-3 against a matrix: (2, 3, 4) x (4, 5) -> (2, 3, 5).
    var a3 = _ramp[2, 3, 4](24, 0.0)
    var m = _ramp[4, 5](20, -1.0)
    var d = tensordot[axes=1](a3, m)
    assert_equal(d.dim_at(0), 2)
    assert_equal(d.dim_at(1), 3)
    assert_equal(d.dim_at(2), 5)
    var mh = m.to_host()
    var dh = d.to_host()
    for i in range(2):
        for j in range(3):
            for l in range(5):
                var acc = 0.0
                for k in range(4):
                    acc += Float64(ah[i * 12 + j * 4 + k]) * Float64(
                        mh[k * 5 + l]
                    )
                assert_almost_equal(
                    Float64(dh[i * 15 + j * 5 + l]), acc, atol=1e-12
                )


def test_tensordot_raises_on_mismatched_contracted_extents() raises:
    var a = _ramp[2, 3](6, 0.0)
    var b = _ramp[4, 2](8, 0.0)
    with assert_raises(contains="contracted extent"):
        _ = tensordot[axes=1](a, b)


def test_tensorsolve_recovers_a_known_solution() raises:
    """`a` is `(2, 3, 6)`, read as `6 x 6`; `b = tensordot(a, x, 1)` for a
    known `x`, and `tensorsolve` gives `x` back at shape `(6,)`."""
    var values = List[Float64](capacity=36)
    for i in range(6):
        for j in range(6):
            values.append((3.0 if i == j else 0.0) + 0.1 * Float64(i + 2 * j))
    var a = _filled[2, 3, 6](values)
    var x_true: List[Float64] = [1.0, -2.0, 0.5, 3.0, 0.0, -1.0]
    var b_values = List[Float64](capacity=6)
    for i in range(6):
        var acc = 0.0
        for j in range(6):
            acc += values[i * 6 + j] * x_true[j]
        b_values.append(acc)
    var b = _filled[2, 3](b_values)
    var x = tensorsolve(a, b)
    assert_equal(x.dim_at(0), 6)
    var xh = x.to_host()
    for i in range(6):
        assert_almost_equal(Float64(xh[i]), x_true[i], atol=1e-10)


def test_tensorinv_inverts_with_respect_to_tensordot() raises:
    """`a` is `(4, 2, 2)` with `ind=1`: `tensorinv(a)` has shape `(2, 2, 4)`
    and `tensordot(tensorinv(a), a, 1)` is the identity read as `(2, 2, 2,
    2)`."""
    var values = List[Float64](capacity=16)
    for i in range(4):
        for j in range(4):
            values.append(
                (2.0 if i == j else 0.0) + 0.25 * Float64(i) - 0.1 * Float64(j)
            )
    var a = _filled[4, 2, 2](values)
    var inv = tensorinv[ind=1](a)
    assert_equal(inv.dim_at(0), 2)
    assert_equal(inv.dim_at(1), 2)
    assert_equal(inv.dim_at(2), 4)
    var square = _filled[4, 4](values)
    var direct = inverse(square).to_host()
    var got = inv.to_host()
    for i in range(16):
        assert_almost_equal(Float64(got[i]), Float64(direct[i]), atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
