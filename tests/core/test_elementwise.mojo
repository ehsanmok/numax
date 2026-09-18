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

The routing tests at the bottom make the other claim: that driving a name
through `numax.core._drive` returns exactly what the scalar loop it replaced
returned, bit for bit, across negatives, both zeros, both infinities and
NaN; that a run-time-shaped `Dynamic` gives the same answer as the `Static`
of the same extents; and that the threaded path above `_THREADED_FROM`
agrees with the serial one below it. Those are checked against `std.math`
re-applied on purpose -- the question is whether the driver preserves the
scalar answer, not which scalar function was wired up.
"""

from std.math import (
    acos as _std_acos,
    acosh as _std_acosh,
    asin as _std_asin,
    asinh as _std_asinh,
    atan as _std_atan,
    atan2 as _std_atan2,
    atanh as _std_atanh,
    cbrt as _std_cbrt,
    ceil as _std_ceil,
    copysign as _std_copysign,
    cos as _std_cos,
    cosh as _std_cosh,
    exp2 as _std_exp2,
    expm1 as _std_expm1,
    floor as _std_floor,
    hypot as _std_hypot,
    isnan,
    log10 as _std_log10,
    log1p as _std_log1p,
    log2 as _std_log2,
    nan,
    remainder as _std_remainder,
    round as _std_round,
    rsqrt as _std_rsqrt,
    sin as _std_sin,
    sinh as _std_sinh,
    sqrt as _std_sqrt,
    tan as _std_tan,
    tanh as _std_tanh,
    trunc as _std_trunc,
)
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax.core import Dynamic, Static, Tensor, tanh, zeros_dyn
from numax.core.libm import exp as _std_exp
from numax.core.libm import log as _std_log
from numax.core._drive import (
    _THREADED_FROM,
    binary_scalar,
    binary_to,
    broadcast_binary_to,
    unary_to,
)
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
    degrees,
    diff,
    exp,
    exp2,
    expm1,
    floor,
    fmax,
    fmin,
    gradient,
    hypot,
    log,
    log10,
    log1p,
    log2,
    maximum,
    minimum,
    power,
    radians,
    reciprocal,
    remainder,
    rint,
    round,
    rsqrt,
    sign,
    sin,
    sinh,
    sqrt,
    square,
    tan,
    trunc,
)

comptime dtype = DType.float64


def _t[n: Int](values: List[Float64]) raises -> Static[dtype, n]:
    var ctx = DeviceContext(api="cpu")
    var elements = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        elements.append(Scalar[dtype](values[i]))
    return Static[dtype, n](ctx, elements^)


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
    var a = Static[dtype, 2, 3](ctx)
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


def _m2[
    rows: Int, cols: Int
](values: List[Float64]) raises -> Static[dtype, rows, cols]:
    var ctx = DeviceContext(api="cpu")
    var elements = List[Scalar[dtype]](capacity=rows * cols)
    for i in range(rows * cols):
        elements.append(Scalar[dtype](values[i]))
    return Static[dtype, rows, cols](ctx, elements^)


def test_maximum_broadcasts_a_row_across_a_matrix() raises:
    # numpy: np.maximum([[1, 5, 3], [7, 2, 9]], [4, 4, 4])
    var a = _m2[2, 3]([1.0, 5.0, 3.0, 7.0, 2.0, 9.0])
    var floor_ = _t[3]([4.0, 4.0, 4.0])

    var got = maximum(a, floor_)
    assert_equal(got.dim_at(0), 2)
    assert_equal(got.dim_at(1), 3)
    var out = got.to_host()
    var expected = [4.0, 5.0, 4.0, 7.0, 4.0, 9.0]
    for i in range(6):
        assert_almost_equal(out[i], Scalar[dtype](expected[i]))


def test_hypot_broadcasts_a_column_against_a_row() raises:
    # A (2, 1) against a (2,) gives a (2, 2): the classic 3-4-5 and 6-8-10.
    var col = _m2[2, 1]([3.0, 6.0])
    var row = _t[2]([4.0, 8.0])

    var got = hypot(col, row).to_host()
    assert_almost_equal(got[0], 5.0)
    assert_almost_equal(got[3], 10.0)


def test_broadcast_minimum_agrees_with_the_same_shape_overload() raises:
    var a = _m2[2, 3]([1.0, 5.0, 3.0, 7.0, 2.0, 9.0])
    var thin = _m2[1, 3]([4.0, 4.0, 4.0])
    var wide = _m2[2, 3]([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

    var broadcast = minimum(a, thin).to_host()
    var direct = minimum(a, wide).to_host()
    for i in range(6):
        assert_equal(broadcast[i], direct[i])


# The routing tests.


def _edge() -> List[Float64]:
    """Negatives, both signed zeros, both infinities and a NaN -- every class
    a per-element launch has to carry through unchanged."""
    var inf = Float64.MAX * 2.0
    return [-3.5, -1.0, -0.0, 0.0, 0.5, 1.0, 2.5, inf, -inf, inf - inf]


def _nan_or(
    a: Scalar[dtype], b: Scalar[dtype], otherwise: Scalar[dtype]
) -> Scalar[dtype]:
    """NumPy's NaN rule for `maximum`/`minimum`: either operand NaN wins."""
    if isnan(a):
        return a
    if isnan(b):
        return b
    return otherwise


def _same(got: Scalar[dtype], want: Scalar[dtype]) raises:
    """Bit-for-bit agreement, with NaN compared by class rather than value."""
    if isnan(want):
        assert_true(isnan(got), "expected NaN")
    else:
        assert_equal(got, want)


def test_unary_routing_is_bit_exact_over_the_edge_cases() raises:
    """Every routed unary name, against the scalar call it wraps.

    The driver's job is to reproduce the scalar answer exactly, including
    on the inputs where that answer is a NaN or an infinity; the
    hand-computed tests above are what check the right scalar call was
    wired to the right numax name.
    """
    var a = _t[10](_edge())
    var x = a.to_host()

    var got_exp = exp(a).to_host()
    var got_exp2 = exp2(a).to_host()
    var got_expm1 = expm1(a).to_host()
    var got_log = log(a).to_host()
    var got_log2 = log2(a).to_host()
    var got_log10 = log10(a).to_host()
    var got_log1p = log1p(a).to_host()
    var got_sqrt = sqrt(a).to_host()
    var got_rsqrt = rsqrt(a).to_host()
    var got_cbrt = cbrt(a).to_host()
    var got_sin = sin(a).to_host()
    var got_cos = cos(a).to_host()
    var got_tan = tan(a).to_host()
    var got_arcsin = arcsin(a).to_host()
    var got_arccos = arccos(a).to_host()
    var got_arctan = arctan(a).to_host()
    var got_sinh = sinh(a).to_host()
    var got_cosh = cosh(a).to_host()
    var got_tanh = tanh(a).to_host()
    var got_arcsinh = arcsinh(a).to_host()
    var got_arccosh = arccosh(a).to_host()
    var got_arctanh = arctanh(a).to_host()
    var got_floor = floor(a).to_host()
    var got_ceil = ceil(a).to_host()
    var got_trunc = trunc(a).to_host()
    var got_round = round(a).to_host()
    var got_abs = abs(a).to_host()

    for i in range(10):
        _same(got_exp[i], _std_exp(x[i]))
        _same(got_exp2[i], _std_exp2(x[i]))
        _same(got_expm1[i], _std_expm1(x[i]))
        _same(got_log[i], _std_log(x[i]))
        _same(got_log2[i], _std_log2(x[i]))
        _same(got_log10[i], _std_log10(x[i]))
        _same(got_log1p[i], _std_log1p(x[i]))
        _same(got_sqrt[i], _std_sqrt(x[i]))
        _same(got_rsqrt[i], _std_rsqrt(x[i]))
        _same(got_cbrt[i], _std_cbrt(x[i]))
        _same(got_sin[i], _std_sin(x[i]))
        _same(got_cos[i], _std_cos(x[i]))
        _same(got_tan[i], _std_tan(x[i]))
        _same(got_arcsin[i], _std_asin(x[i]))
        _same(got_arccos[i], _std_acos(x[i]))
        _same(got_arctan[i], _std_atan(x[i]))
        _same(got_sinh[i], _std_sinh(x[i]))
        _same(got_cosh[i], _std_cosh(x[i]))
        _same(got_tanh[i], _std_tanh(x[i]))
        _same(got_arcsinh[i], _std_asinh(x[i]))
        _same(got_arccosh[i], _std_acosh(x[i]))
        _same(got_arctanh[i], _std_atanh(x[i]))
        _same(got_floor[i], _std_floor(x[i]))
        _same(got_ceil[i], _std_ceil(x[i]))
        _same(got_trunc[i], _std_trunc(x[i]))
        _same(got_round[i], _std_round(x[i]))
        _same(got_abs[i], x[i].__abs__())


def test_binary_routing_is_bit_exact_over_the_edge_cases() raises:
    """Every routed binary name, at one shape, against the scalar call."""
    var a = _t[10](_edge())
    var b = _t[10]([1.0, -2.0, 0.5, -0.0, 3.0, -1.5, 0.0, 2.0, -4.0, 1.0])
    var x = a.to_host()
    var y = b.to_host()

    var got_atan2 = arctan2(a, b).to_host()
    var got_hypot = hypot(a, b).to_host()
    var got_copysign = copysign(a, b).to_host()
    var got_remainder = remainder(a, b).to_host()
    var got_max = maximum(a, b).to_host()
    var got_min = minimum(a, b).to_host()
    var got_fmax = fmax(a, b).to_host()
    var got_fmin = fmin(a, b).to_host()

    for i in range(10):
        _same(got_atan2[i], _std_atan2(x[i], y[i]))
        _same(got_hypot[i], _std_hypot(x[i], y[i]))
        _same(got_copysign[i], _std_copysign(x[i], y[i]))
        _same(got_remainder[i], _std_remainder(x[i], y[i]))
        # `max`/`min` are IEEE `maxNum`/`minNum` and skip a NaN, which is
        # `fmax`/`fmin`'s contract. `maximum`/`minimum` follow NumPy and
        # propagate, so their reference has to say so.
        _same(got_fmax[i], max(x[i], y[i]))
        _same(got_fmin[i], min(x[i], y[i]))
        _same(got_max[i], _nan_or(x[i], y[i], max(x[i], y[i])))
        _same(got_min[i], _nan_or(x[i], y[i], min(x[i], y[i])))


def test_clip_diff_and_gradient_match_their_scalar_walks() raises:
    """The three routines with a body of their own rather than a driver."""
    var a = _t[8]([-5.0, -0.5, -0.0, 0.0, 0.25, 1.0, 4.0, 9.0])
    var x = a.to_host()

    var clipped = clip(a, -1.0, 2.0).to_host()
    for i in range(8):
        _same(clipped[i], min(max(x[i], -1.0), 2.0))

    var differences = diff(a).to_host()
    for i in range(7):
        _same(differences[i], x[i + 1] - x[i])

    var slopes = gradient(a, 0.25).to_host()
    _same(slopes[0], (x[1] - x[0]) / 0.25)
    _same(slopes[7], (x[7] - x[6]) / 0.25)
    for i in range(1, 7):
        _same(slopes[i], (x[i + 1] - x[i - 1]) / (0.25 + 0.25))


def test_a_dynamic_input_matches_the_static_result() raises:
    var ctx = DeviceContext(api="cpu")
    var values = List[Scalar[dtype]](capacity=6)
    for i in range(6):
        values.append(Scalar[dtype](i) * 0.75 - 2.0)

    var fixed = Static[dtype, 2, 3](ctx, values.copy())
    var runtime = zeros_dyn[dtype, 2](2, 3, ctx=ctx)
    runtime.copy_from_host(values)

    var from_static = exp(fixed).to_host()
    var from_dynamic = exp(runtime).to_host()
    assert_equal(len(from_dynamic), 6)
    for i in range(6):
        assert_equal(from_dynamic[i], from_static[i])


def test_the_threaded_path_agrees_with_the_serial_one() raises:
    """`_THREADED_FROM` is a performance switch, not a numerical one.

    The values stay well away from the denormal range: `elementwise`'s CPU
    backend runs its workers with flush-to-zero set, which the serial loop
    below the threshold does not, so a denormal result is the one thing the
    two paths may disagree about.
    """
    comptime n = 100003
    assert_true(n > _THREADED_FROM, "n must cross the threading threshold")
    var ctx = DeviceContext(api="cpu")
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(Scalar[dtype](i % 977) * 0.01 - 4.0)

    var big = Static[dtype, n](ctx, values.copy())
    var got = exp(big).to_host()
    for i in range(n):
        _same(got[i], _std_exp(values[i]))


# The drivers `elementwise.mojo` does not itself use. Pinned here so the
# whole of `_drive` is instantiated by the suite rather than only the part
# this module reaches.


def _truncate_op[w: Int](x: SIMD[dtype, w]) -> SIMD[DType.int32, w]:
    return x.cast[DType.int32]()


def _greater_op[
    w: Int
](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[DType.bool, w]:
    return a.gt(b)


def _plus_op[w: Int](a: SIMD[dtype, w], b: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return a + b


def test_unary_to_changes_dtype() raises:
    var a = _t[4]([-1.7, 0.0, 2.9, 5.5])
    var got = unary_to[
        type_of(a),
        DType.int32,
        op=_truncate_op,
        gpu=False,
        name="truncate",
    ](a).to_host()
    assert_equal(got[0], -1)
    assert_equal(got[1], 0)
    assert_equal(got[2], 2)
    assert_equal(got[3], 5)


def test_binary_to_writes_a_bool_result_everywhere() raises:
    var a = _t[4]([1.0, 2.0, 3.0, 4.0])
    var b = _t[4]([5.0, 6.0, 7.0, 8.0])
    var got = binary_to[
        type_of(a),
        DType.bool,
        op=_greater_op,
        gpu=False,
        name="greater",
    ](a, b).to_host()
    # All false: an uninitialized destination that the launch missed would
    # show here rather than in a result with true entries in it.
    for i in range(4):
        assert_equal(got[i], False)


def test_binary_scalar_captures_its_operand() raises:
    var a = _t[3]([1.0, 2.0, 3.0])
    var got = binary_scalar[
        type_of(a),
        op=_plus_op,
        gpu=False,
        name="add",
    ](a, 10.0).to_host()
    assert_equal(got[0], 11.0)
    assert_equal(got[1], 12.0)
    assert_equal(got[2], 13.0)


def test_broadcast_binary_to_stretches_a_row() raises:
    var a = _m2[2, 3]([1.0, 5.0, 3.0, 7.0, 2.0, 9.0])
    var row = _t[3]([4.0, 4.0, 4.0])
    var got = broadcast_binary_to[
        type_of(a),
        type_of(row),
        DType.bool,
        op=_greater_op,
        gpu=False,
        name="greater",
    ](a, row)
    assert_equal(got.dim_at(0), 2)
    assert_equal(got.dim_at(1), 3)
    var out = got.to_host()
    var expected = [False, True, False, True, False, True]
    for i in range(6):
        assert_equal(out[i], expected[i])


def test_sign_is_zero_at_both_zeros_and_nan_at_nan() raises:
    var a = _t[5]([2.5, -2.5, 0.0, -0.0, nan[dtype]()])
    var got = sign(a).to_host()
    assert_equal(got[0], 1.0)
    assert_equal(got[1], -1.0)
    assert_equal(got[2], 0.0)
    # `copysign(1, x)` would answer -1 here and 1 at NaN; NumPy does not.
    assert_equal(got[3], 0.0)
    assert_true(isnan(got[4]), "sign(nan) should be nan")


def test_square_and_reciprocal_match_hand_computed_values() raises:
    var a = _t[4]([2.0, -3.0, 0.5, 0.0])
    var sq = square(a).to_host()
    assert_almost_equal(sq[0], 4.0)
    assert_almost_equal(sq[1], 9.0)
    assert_almost_equal(sq[2], 0.25)
    assert_almost_equal(sq[3], 0.0)
    var r = reciprocal(a).to_host()
    assert_almost_equal(r[0], 0.5)
    assert_almost_equal(r[1], -1.0 / 3.0)
    assert_almost_equal(r[2], 2.0)
    assert_true(r[3] > 1e300, "reciprocal(0) should be an infinity")


def test_square_agrees_with_power_of_two() raises:
    var a = _t[4]([1.5, -2.25, 3.0, 0.125])
    var sq = square(a).to_host()
    var viapow = power(a, Scalar[dtype](2)).to_host()
    for i in range(4):
        assert_almost_equal(sq[i], viapow[i])


def test_degrees_and_radians_round_trip() raises:
    var a = _t[4](
        [0.0, 0.5235987755982988, 1.5707963267948966, -3.141592653589793]
    )
    var d = degrees(a).to_host()
    assert_almost_equal(d[0], 0.0)
    assert_almost_equal(d[1], 30.0)
    assert_almost_equal(d[2], 90.0)
    assert_almost_equal(d[3], -180.0)
    var back = radians(degrees(a)).to_host()
    var orig = a.to_host()
    for i in range(4):
        assert_almost_equal(back[i], orig[i])


def test_rint_is_round_under_numpys_name() raises:
    var a = _t[6]([0.5, 1.5, 2.5, -0.5, -1.5, 2.4])
    var r = rint(a).to_host()
    var viaround = round(a).to_host()
    for i in range(6):
        assert_equal(r[i], viaround[i])
    # half to even, not half away from zero
    assert_equal(r[0], 0.0)
    assert_equal(r[1], 2.0)
    assert_equal(r[2], 2.0)


def test_maximum_propagates_nan_and_fmax_ignores_it() raises:
    var a = _t[3]([nan[dtype](), 1.0, 5.0])
    var b = _t[3]([1.0, nan[dtype](), 2.0])

    var mx = maximum(a, b).to_host()
    assert_true(isnan(mx[0]), "maximum(nan, 1) should be nan")
    assert_true(isnan(mx[1]), "maximum(1, nan) should be nan")
    assert_equal(mx[2], 5.0)

    var mn = minimum(a, b).to_host()
    assert_true(isnan(mn[0]), "minimum(nan, 1) should be nan")
    assert_true(isnan(mn[1]), "minimum(1, nan) should be nan")
    assert_equal(mn[2], 2.0)

    # `fmax`/`fmin` are the hardware instruction: the NaN is skipped.
    var fx = fmax(a, b).to_host()
    assert_equal(fx[0], 1.0)
    assert_equal(fx[1], 1.0)
    assert_equal(fx[2], 5.0)

    var fm = fmin(a, b).to_host()
    assert_equal(fm[0], 1.0)
    assert_equal(fm[1], 1.0)
    assert_equal(fm[2], 2.0)


def test_fmax_broadcasts_a_row_like_maximum() raises:
    var a = _m2[2, 3]([1.0, 5.0, 3.0, 7.0, 2.0, 9.0])
    var row = _t[3]([4.0, 4.0, 4.0])
    var wide = _m2[2, 3]([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    var broadcast = fmax(a, row).to_host()
    var direct = fmax(a, wide).to_host()
    for i in range(6):
        assert_equal(broadcast[i], direct[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
