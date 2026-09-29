"""Tests for `linprog`: each problem against `scipy.optimize.linprog`'s
HiGHS optimum (the bodies are generated from SciPy, not transcribed), the
infeasible and unbounded verdicts, the inequality-only overload against the
five-tensor spelling, the default `x >= 0`, `slack`/`con`, and `float32`."""

from std.math import inf as _inf
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

from numax.core.tensor import Static, zeros
from numax.optimize import Bounds, linprog

comptime dtype = DType.float64


def test_scipy_example() raises:
    """SciPy's docstring example: a free variable and a finite lower bound."""
    var c = Static[dtype, 2]([-1.0, 4.0])
    var a_ub = Static[dtype, 2, 2]([-3.0, 1.0, 1.0, 2.0])
    var b_ub = Static[dtype, 2]([6.0, 4.0])
    var a_eq = zeros[dtype, 0, 2]()
    var b_eq = zeros[dtype, 0]()
    var r = linprog(
        c,
        a_ub,
        b_ub,
        a_eq,
        b_eq,
        Bounds(
            [-_inf[DType.float64](), -3.0],
            [_inf[DType.float64](), _inf[DType.float64]()],
        ),
    )
    assert_equal(r.status, 0)
    assert_almost_equal(r.fun, -22.0, atol=1e-6)
    var x = r.x.to_host()
    var want: List[Float64] = [10.0, -3.0]
    for i in range(2):
        assert_almost_equal(Float64(x[i]), want[i], atol=1e-5)


def test_mixed_boxed() raises:
    """Random inequalities and equalities over the box [0, 3]: every row kind and the boxed-variable slack rows.
    """
    var c = Static[dtype, 5](
        [
            0.6672475608343279,
            1.438522591656152,
            -0.6756622510056528,
            0.20313861038960904,
            -0.46330757653841514,
        ]
    )
    var a_ub = Static[dtype, 7, 5](
        [
            0.0012301533574825742,
            0.2987455375084699,
            -0.2741378553622176,
            -0.8905918387572742,
            -0.45467078517172255,
            -0.9916465549964624,
            0.060143602597438485,
            1.3402152455545335,
            -0.49220651855132963,
            -0.6204748998199404,
            0.4898420501851982,
            0.35688700816006075,
            0.10541424899789856,
            -0.9304680447082047,
            -0.02925182246327349,
            0.6953031944582878,
            -1.344214547285082,
            -0.45761576104021817,
            -1.901222739800844,
            -1.289537739784976,
            -1.8417350377917323,
            -0.23509113107468127,
            -1.2674464814437032,
            0.2712643588217015,
            0.15675108662422516,
            -0.18693094462995438,
            -2.516759710820513,
            -0.5386928958466366,
            -0.048500945401071985,
            0.11330898600330756,
            -1.5301357655053935,
            -0.47775327603393064,
            -0.9785190780566395,
            -0.8088372394255993,
            1.0608986233860787,
        ]
    )
    var b_ub = Static[dtype, 7](
        [
            -0.4952264968252784,
            0.9630557674458077,
            1.0360926174940466,
            -2.6228989828818694,
            -3.165025747066963,
            -2.1606220299277767,
            -2.6108646062211176,
        ]
    )
    var a_eq = Static[dtype, 2, 5](
        [
            0.11935402569658124,
            -0.6414703941072214,
            2.000416546342423,
            0.7622597120847118,
            -1.1992889021052233,
            0.07451622877146342,
            0.5766895836701853,
            -0.1887821253507493,
            0.682910267195206,
            -0.06651732014941557,
        ]
    )
    var b_eq = Static[dtype, 2]([2.7690785487712426, 0.48042058562658285])
    var r = linprog(
        c,
        a_ub,
        b_ub,
        a_eq,
        b_eq,
        Bounds([0.0, 0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 3.0, 3.0, 3.0]),
    )
    assert_equal(r.status, 0)
    assert_almost_equal(r.fun, -1.6626251514723984, atol=1e-6)
    var x = r.x.to_host()
    var want: List[Float64] = [
        1.2601012411672476,
        0.2983870017727438,
        2.6890511256166345,
        1.3495809168954467,
        3.0,
    ]
    for i in range(5):
        assert_almost_equal(Float64(x[i]), want[i], atol=1e-5)


def test_equality_only() raises:
    """Equalities alone under mixed bounds: an upper-only variable, a lower-only one, two boxed.
    """
    var c = Static[dtype, 4](
        [
            0.12726841122583082,
            -1.18719452785014,
            -0.5793015965026732,
            -0.1961959728044967,
        ]
    )
    var a_ub = zeros[dtype, 0, 4]()
    var b_ub = zeros[dtype, 0]()
    var a_eq = Static[dtype, 2, 4](
        [
            0.8987638721004078,
            1.145222007454132,
            -1.323527792484255,
            -0.7946423659870495,
            0.6469034225734218,
            -1.9924197841744944,
            -0.46316986495236695,
            -0.09728692567008902,
        ]
    )
    var b_eq = Static[dtype, 2]([0.15552360800878262, -2.763131777925366])
    var r = linprog(
        c,
        a_ub,
        b_ub,
        a_eq,
        b_eq,
        Bounds(
            [-1.0, -2.0, 0.0, -_inf[DType.float64]()],
            [2.0, 2.0, _inf[DType.float64](), 4.0],
        ),
    )
    assert_equal(r.status, 0)
    assert_almost_equal(r.fun, -4.021253312027007, atol=1e-6)
    var x = r.x.to_host()
    var want: List[Float64] = [
        2.0,
        -2.0,
        26.968767472545437,
        -45.734225499710845,
    ]
    for i in range(4):
        assert_almost_equal(Float64(x[i]), want[i], atol=1e-5)


def test_infeasible() raises:
    """With x >= 0, x0 + x1 <= -1 has no feasible point: status 2."""
    var c = Static[dtype, 2]([1.0, 1.0])
    var a_ub = Static[dtype, 1, 2]([1.0, 1.0])
    var b_ub = Static[dtype, 1]([-1.0])
    var a_eq = zeros[dtype, 0, 2]()
    var b_eq = zeros[dtype, 0]()
    var r = linprog(
        c,
        a_ub,
        b_ub,
        a_eq,
        b_eq,
        Bounds([0.0, 0.0], [_inf[DType.float64](), _inf[DType.float64]()]),
    )
    assert_equal(r.status, 2)


def test_unbounded() raises:
    """Minimizing -x0 with x0 - x1 <= 1 over x >= 0 runs off to infinity: status 3.
    """
    var c = Static[dtype, 2]([-1.0, 0.0])
    var a_ub = Static[dtype, 1, 2]([1.0, -1.0])
    var b_ub = Static[dtype, 1]([1.0])
    var a_eq = zeros[dtype, 0, 2]()
    var b_eq = zeros[dtype, 0]()
    var r = linprog(
        c,
        a_ub,
        b_ub,
        a_eq,
        b_eq,
        Bounds([0.0, 0.0], [_inf[DType.float64](), _inf[DType.float64]()]),
    )
    assert_equal(r.status, 3)


def test_overloads_agree() raises:
    """The three-tensor overload is the five-tensor one with no equality
    rows, and omitted bounds are SciPy's `(0, None)`."""
    var c = Static[dtype, 3]([-3.0, -1.0, -2.0])
    var a = Static[dtype, 2, 3]([1.0, 1.0, 1.0, 2.0, 0.5, 1.0])
    var b = Static[dtype, 2]([4.0, 5.0])
    var short = linprog(c, a, b)
    var full = linprog(
        c,
        a,
        b,
        zeros[dtype, 0, 3](),
        zeros[dtype, 0](),
        Bounds(0.0, _inf[DType.float64]()),
    )
    assert_equal(short.status, 0)
    assert_equal(full.status, 0)
    assert_almost_equal(short.fun, full.fun, atol=1e-12)
    var xs = short.x.to_host()
    var xf = full.x.to_host()
    for i in range(3):
        assert_almost_equal(Float64(xs[i]), Float64(xf[i]), atol=1e-12)
        assert_true(Float64(xs[i]) > -1e-9)
    assert_equal(short.con.size(), 0)
    assert_true(short.success)


def test_slack_and_con() raises:
    """`slack` is `b_ub - A_ub x` and `con` is `b_eq - A_eq x`: nonnegative
    and zero at the optimum."""
    var c = Static[dtype, 2]([1.0, 2.0])
    var a_ub = Static[dtype, 1, 2]([-1.0, -1.0])
    var b_ub = Static[dtype, 1]([-1.0])
    var a_eq = Static[dtype, 1, 2]([1.0, -1.0])
    var b_eq = Static[dtype, 1]([0.25])
    var r = linprog(c, a_ub, b_ub, a_eq, b_eq)
    assert_equal(r.status, 0)
    var x = r.x.to_host()
    assert_almost_equal(Float64(x[0]), 0.625, atol=1e-7)
    assert_almost_equal(Float64(x[1]), 0.375, atol=1e-7)
    assert_almost_equal(Float64(r.slack.to_host()[0]), 0.0, atol=1e-7)
    assert_almost_equal(Float64(r.con.to_host()[0]), 0.0, atol=1e-7)


def test_float32() raises:
    """At `float32` the default tolerance is `1e-5` and the optimum of
    SciPy's example holds to it."""
    var c = Static[DType.float32, 2]([-1.0, 4.0])
    var a = Static[DType.float32, 2, 2]([-3.0, 1.0, 1.0, 2.0])
    var b = Static[DType.float32, 2]([6.0, 4.0])
    var r = linprog(
        c,
        a,
        b,
        Bounds(
            [-_inf[DType.float64](), -3.0],
            [_inf[DType.float64](), _inf[DType.float64]()],
        ),
    )
    assert_equal(r.status, 0)
    assert_almost_equal(r.fun, -22.0, atol=1e-3)


def test_shape_errors() raises:
    """A column-count mismatch and inverted bounds raise, naming `linprog`."""
    var c = Static[dtype, 2]([1.0, 1.0])
    var a = Static[dtype, 1, 3]([1.0, 1.0, 1.0])
    var b = Static[dtype, 1]([1.0])
    with assert_raises(contains="linprog"):
        _ = linprog(c, a, b)
    var a2 = Static[dtype, 1, 2]([1.0, 1.0])
    with assert_raises(contains="lb <= ub"):
        _ = linprog(c, a2, b, Bounds(1.0, 0.0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
