"""Tests for the combinatorial gamma-family functions, the
information-theoretic elementwise functions, and `logsumexp` at both
tiers, each against `scipy.special`'s values and conventions."""

from std.collections import Array
from std.math import log as _log
from std.testing import TestSuite, assert_almost_equal, assert_true

from max.gpu.host import DeviceContext

from numax import Dual, Plain
from numax.core.array import Static
from numax.special import (
    comb,
    entr,
    factorial,
    gammasgn,
    kl_div,
    logit,
    logsumexp,
    perm,
    poch,
    rel_entr,
    xlog1py,
    xlogy,
)

comptime dtype = DType.float64
comptime P = Plain[dtype, 1]
comptime D = Dual[P]


def pv(x: Float64) -> P:
    return P.constant(x)


def s(x: P) -> Float64:
    return Float64(x.v)


def test_combinatorics_match_scipy() raises:
    """`comb`, `perm`, `factorial` at integer and real arguments, `poch` at
    positive and negative `z`, and `gammasgn` on both sides of the poles."""
    assert_almost_equal(s(comb(pv(10.0), pv(3.0))), 120.0, atol=1e-10)
    assert_almost_equal(
        s(comb(pv(7.5), pv(2.2))), 28.68916543518843, atol=1e-10
    )
    assert_almost_equal(s(perm(pv(10.0), pv(3.0))), 720.0, atol=1e-10)
    assert_almost_equal(
        s(perm(pv(7.5), pv(2.2))), 69.54154666305169, atol=1e-10
    )
    assert_almost_equal(s(factorial(pv(5.0))), 120.0, atol=1e-10)
    assert_almost_equal(s(factorial(pv(4.5))), 52.34277778455352, atol=1e-10)
    assert_almost_equal(s(poch(pv(3.5), pv(2.0))), 15.75, atol=1e-10)
    assert_almost_equal(s(poch(pv(-2.5), pv(3.0))), -1.875, atol=1e-10)
    assert_almost_equal(
        s(poch(pv(0.3), pv(1.7))), 0.33427275256419053, atol=1e-10
    )
    assert_almost_equal(s(gammasgn(pv(-2.5))), -1.0)
    assert_almost_equal(s(gammasgn(pv(-1.5))), 1.0)
    assert_almost_equal(s(gammasgn(pv(2.5))), 1.0)
    assert_almost_equal(s(gammasgn(pv(-0.5))), -1.0)


def test_xlogy_and_xlog1py_apply_the_zero_convention() raises:
    assert_almost_equal(s(xlogy(pv(0.0), pv(0.0))), 0.0)
    assert_almost_equal(s(xlogy(pv(0.0), pv(5.0))), 0.0)
    assert_almost_equal(
        s(xlogy(pv(2.0), pv(3.0))), 2.1972245773362196, atol=1e-14
    )
    var minus_inf = s(xlogy(pv(2.0), pv(0.0)))
    assert_true(minus_inf < -1e300)
    assert_almost_equal(s(xlog1py(pv(0.0), pv(-1.0))), 0.0)
    assert_almost_equal(
        s(xlog1py(pv(2.0), pv(3.0))), 2.772588722239781, atol=1e-14
    )


def test_entropy_terms_match_scipy() raises:
    """`entr`, `rel_entr` and `kl_div` at interior points and at the
    zero conventions, plus `entr`'s `-inf` for a negative argument."""
    assert_almost_equal(s(entr(pv(0.5))), 0.34657359027997264, atol=1e-14)
    assert_almost_equal(s(entr(pv(0.0))), 0.0)
    assert_almost_equal(s(entr(pv(1.0))), 0.0, atol=1e-15)
    assert_true(s(entr(pv(-0.5))) < -1e300)
    assert_almost_equal(
        s(rel_entr(pv(0.5), pv(0.25))), 0.34657359027997264, atol=1e-14
    )
    assert_almost_equal(s(rel_entr(pv(0.0), pv(0.3))), 0.0)
    assert_true(s(rel_entr(pv(0.4), pv(0.0))) > 1e300)
    assert_almost_equal(
        s(kl_div(pv(0.5), pv(0.25))), 0.09657359027997264, atol=1e-14
    )
    assert_almost_equal(s(kl_div(pv(0.0), pv(0.3))), 0.3, atol=1e-15)
    assert_almost_equal(s(kl_div(pv(0.7), pv(0.7))), 0.0, atol=1e-15)


def test_logit_inverts_the_sigmoid() raises:
    assert_almost_equal(s(logit(pv(0.25))), -1.0986122886681098, atol=1e-14)
    assert_almost_equal(s(logit(pv(0.5))), 0.0, atol=1e-15)
    # logit(expit(-1)) == -1
    assert_almost_equal(s(logit(pv(0.2689414213699951))), -1.0, atol=1e-13)


def test_information_functions_differentiate() raises:
    """`d/dx xlogy(x, y) = log(y)` and `d/dx entr(x) = -log(x) - 1` at
    `Dual`, with no adjoint rule written."""
    var x = D(pv(2.0), pv(1.0))
    var y = D(pv(3.0), pv(0.0))
    assert_almost_equal(Float64(xlogy(x, y).deriv.v), _log(3.0), atol=1e-14)
    var e = entr(D(pv(0.5), pv(1.0)))
    assert_almost_equal(Float64(e.deriv.v), -_log(0.5) - 1.0, atol=1e-14)


def test_logsumexp_over_an_array_is_stable_and_differentiates() raises:
    """`logsumexp([1, 2, 3])` against SciPy, the `[-1000, -1000.5]` case a
    naive `log(sum(exp))` underflows on, and the derivative -- the
    softmax -- at `Dual`."""
    var xs = Array[P, 3](fill=pv(0.0))
    xs[0] = pv(1.0)
    xs[1] = pv(2.0)
    xs[2] = pv(3.0)
    assert_almost_equal(s(logsumexp(xs)), 3.40760596444438, atol=1e-13)
    var deep = Array[P, 2](fill=pv(0.0))
    deep[0] = pv(-1000.0)
    deep[1] = pv(-1000.5)
    assert_almost_equal(s(logsumexp(deep)), -999.5259230158199, atol=1e-12)
    var seeded = Array[D, 3](fill=D.constant(0.0))
    seeded[0] = D(pv(1.0), pv(1.0))
    seeded[1] = D(pv(2.0), pv(0.0))
    seeded[2] = D(pv(3.0), pv(0.0))
    var softmax_0 = 1.0 / (1.0 + 2.718281828459045 + 7.38905609893065)
    assert_almost_equal(
        Float64(logsumexp(seeded).deriv.v), softmax_0, atol=1e-13
    )


def test_logsumexp_over_a_tensor_matches_scipy_and_the_array_tier() raises:
    """The `Tensor` overload through MAX's `OnlineLogSumExp` monoid: SciPy's
    value on sixteen samples, the deep-negative case, and agreement with
    the `Array` overload on the same data."""
    var ctx = DeviceContext(api="cpu")
    var values = List[Scalar[dtype]](capacity=16)
    var packed = Array[P, 16](fill=pv(0.0))
    for i in range(16):
        var v = Float64(i) * 0.37 - 3.0
        values.append(Scalar[dtype](v))
        packed[i] = pv(v)
    var x = Static[dtype, 16](ctx, values^)
    assert_almost_equal(Float64(logsumexp(x)), 3.720865788282288, atol=1e-13)
    assert_almost_equal(Float64(logsumexp(x)), s(logsumexp(packed)), atol=1e-13)
    var deep = Static[dtype, 2](ctx, [-1000.0, -1000.5])
    assert_almost_equal(
        Float64(logsumexp(deep)), -999.5259230158199, atol=1e-12
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
