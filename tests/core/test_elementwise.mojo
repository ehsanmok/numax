"""Tests for `numax.core.elementwise`.

Each family is checked against a hand-computed value rather than against
`std.math` re-applied, so the test would catch a wrong `std.math` name being
wired to a numax name. Identities (`exp`/`log`, `sin`/`arcsin`,
`sinh`/`arcsinh`) cover the round trips; `diff`/`gradient` are checked
against their NumPy definitions including the one-sided endpoints.

The imports come from `numax` and `numax.core` rather than from
`numax.core.elementwise`, so a function that exists but is re-exported
nowhere fails here instead of being reachable only by its module path.
`tanh` is the exception: the root `tanh` is `numax.special.activations`'
scalar one, so the tensor form comes from `numax.core`.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core import Shaped, Tensor, tanh
from numax import (
    abs,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    cbrt,
    ceil,
    clip,
    copysign,
    cos,
    cosh,
    diff,
    exp,
    exp2,
    expm1,
    floor,
    gradient,
    hypot,
    log,
    log10,
    log1p,
    log2,
    maximum,
    minimum,
    round,
    rsqrt,
    sin,
    sinh,
    sqrt,
    tan,
    trunc,
)

comptime dtype = DType.float64


def _t[n: Int](values: List[Float64]) raises -> Shaped[dtype, n]:
    var ctx = DeviceContext(api="cpu")
    var elements = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        elements.append(Scalar[dtype](values[i]))
    return Shaped[dtype, n](ctx, elements^)


def test_exponentials_match_hand_computed_values() raises:
    var a = _t[3]([0.0, 1.0, 2.0])
    var e = exp(a).to_host()
    assert_almost_equal(e[0], 1.0)
    assert_almost_equal(e[1], 2.718281828459045)
    var e2 = exp2(a).to_host()
    assert_almost_equal(e2[0], 1.0)
    assert_almost_equal(e2[2], 4.0)
    var em1 = expm1(_t[1]([1e-10])).to_host()
    assert_almost_equal(em1[0], 1e-10, atol=1e-20)


def test_logarithms_match_hand_computed_values() raises:
    var a = _t[3]([1.0, 8.0, 100.0])
    assert_almost_equal(log(a).to_host()[0], 0.0)
    assert_almost_equal(log2(a).to_host()[1], 3.0)
    assert_almost_equal(log10(a).to_host()[2], 2.0)
    assert_almost_equal(log1p(_t[1]([1e-10])).to_host()[0], 1e-10, atol=1e-20)


def test_exp_and_log_round_trip() raises:
    var a = _t[4]([0.5, 1.0, 2.0, 7.5])
    var back = log(exp(a)).to_host()
    var original = a.to_host()
    for i in range(4):
        assert_almost_equal(back[i], original[i])


def test_roots() raises:
    var a = _t[3]([4.0, 9.0, 27.0])
    assert_almost_equal(sqrt(a).to_host()[0], 2.0)
    assert_almost_equal(sqrt(a).to_host()[1], 3.0)
    assert_almost_equal(cbrt(a).to_host()[2], 3.0)
    assert_almost_equal(rsqrt(_t[1]([4.0])).to_host()[0], 0.5)


def test_trig_at_known_angles() raises:
    var a = _t[2]([0.0, 1.5707963267948966])
    assert_almost_equal(sin(a).to_host()[0], 0.0)
    assert_almost_equal(sin(a).to_host()[1], 1.0)
    assert_almost_equal(cos(a).to_host()[0], 1.0)
    assert_almost_equal(tan(_t[1]([0.7853981633974483])).to_host()[0], 1.0)


def test_inverse_trig_round_trips() raises:
    var a = _t[3]([-0.5, 0.0, 0.5])
    var back = sin(arcsin(a)).to_host()
    var original = a.to_host()
    for i in range(3):
        assert_almost_equal(back[i], original[i])
    assert_almost_equal(arccos(_t[1]([1.0])).to_host()[0], 0.0)
    assert_almost_equal(arctan(_t[1]([1.0])).to_host()[0], 0.7853981633974483)


def test_hyperbolic_and_inverses() raises:
    var a = _t[3]([-1.0, 0.0, 1.0])
    assert_almost_equal(sinh(a).to_host()[1], 0.0)
    assert_almost_equal(cosh(a).to_host()[1], 1.0)
    assert_almost_equal(tanh(a).to_host()[1], 0.0)
    var back = sinh(arcsinh(a)).to_host()
    var original = a.to_host()
    for i in range(3):
        assert_almost_equal(back[i], original[i])
    assert_almost_equal(arccosh(_t[1]([1.0])).to_host()[0], 0.0)
    assert_almost_equal(arctanh(_t[1]([0.0])).to_host()[0], 0.0)


def test_rounding_family_splits_on_sign() raises:
    var a = _t[4]([1.7, -1.7, 2.5, -2.5])
    var f = floor(a).to_host()
    var c = ceil(a).to_host()
    var t = trunc(a).to_host()
    assert_equal(f[0], 1.0)
    assert_equal(f[1], -2.0)
    assert_equal(c[0], 2.0)
    assert_equal(c[1], -1.0)
    assert_equal(t[0], 1.0)
    assert_equal(t[1], -1.0)
    # ties to even, matching numpy.round
    var r = round(a).to_host()
    assert_equal(r[2], 2.0)
    assert_equal(r[3], -2.0)


def test_abs() raises:
    var a = _t[3]([-2.5, 0.0, 2.5])
    var abs_ = abs(a).to_host()
    assert_equal(abs_[0], 2.5)
    assert_equal(abs_[1], 0.0)
    assert_equal(abs_[2], 2.5)


def test_binary_families() raises:
    var a = _t[2]([3.0, 1.0])
    var b = _t[2]([4.0, -1.0])
    assert_almost_equal(hypot(a, b).to_host()[0], 5.0)
    var one = _t[1]([1.0])
    var also_one = _t[1]([1.0])
    assert_almost_equal(arctan2(one, also_one).to_host()[0], 0.7853981633974483)
    var signed = copysign(a, b).to_host()
    assert_equal(signed[0], 3.0)
    assert_equal(signed[1], -1.0)


def test_maximum_and_minimum_are_elementwise() raises:
    var a = _t[3]([1.0, 5.0, 3.0])
    var b = _t[3]([4.0, 2.0, 3.0])
    var hi = maximum(a, b).to_host()
    var lo = minimum(a, b).to_host()
    assert_equal(hi[0], 4.0)
    assert_equal(hi[1], 5.0)
    assert_equal(lo[0], 1.0)
    assert_equal(lo[1], 2.0)
    assert_equal(hi[2], 3.0)
    assert_equal(lo[2], 3.0)


def test_clip_confines_to_the_interval() raises:
    var a = _t[4]([-5.0, 0.0, 0.5, 5.0])
    var clipped = clip(a, -1.0, 1.0).to_host()
    assert_equal(clipped[0], -1.0)
    assert_equal(clipped[1], 0.0)
    assert_equal(clipped[2], 0.5)
    assert_equal(clipped[3], 1.0)


def test_elementwise_preserves_rank() raises:
    var ctx = DeviceContext(api="cpu")
    var a = Shaped[dtype, 2, 3](ctx)
    var result = exp(a)
    assert_equal(result.num_elements, 6)
    assert_equal(result.rank, 2)
    # exp(0) == 1 everywhere
    var values = result.to_host()
    for i in range(6):
        assert_almost_equal(values[i], 1.0)


def test_diff_is_one_shorter_and_matches_numpy() raises:
    var a = _t[4]([1.0, 4.0, 9.0, 16.0])
    var d = diff(a)
    assert_equal(d.num_elements, 3)
    var values = d.to_host()
    assert_equal(values[0], 3.0)
    assert_equal(values[1], 5.0)
    assert_equal(values[2], 7.0)


def test_gradient_is_central_inside_and_one_sided_at_the_ends() raises:
    var a = _t[4]([1.0, 2.0, 4.0, 7.0])
    var g = gradient(a).to_host()
    assert_almost_equal(g[0], 1.0)
    assert_almost_equal(g[1], 1.5)
    assert_almost_equal(g[2], 2.5)
    assert_almost_equal(g[3], 3.0)


def test_gradient_scales_with_spacing() raises:
    var a = _t[3]([0.0, 1.0, 2.0])
    var g = gradient(a, 0.5).to_host()
    for i in range(3):
        assert_almost_equal(g[i], 2.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
