"""Tests for `milp`: each problem against `scipy.optimize.milp`'s HiGHS
optimum (the bodies are generated from SciPy, not transcribed), the
integer-infeasible verdict, `milp` with no integer variable against
`linprog`, the node limit, and the argument checks."""

from std.math import inf as _inf
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

from numax.core.tensor import Static, zeros
from numax.optimize import Bounds, LinearConstraint, linprog, milp

comptime dtype = DType.float64


def test_scipy_example() raises:
    """SciPy's docstring example; its two optima tie, so only `fun` is compared.
    """
    var c = Static[dtype, 2]([-0.0, -1.0])
    var a = Static[dtype, 3, 2]([-1.0, 1.0, 3.0, 2.0, 2.0, 3.0])
    var lo = Static[dtype, 3](
        [-_inf[DType.float64](), -_inf[DType.float64](), -_inf[DType.float64]()]
    )
    var hi = Static[dtype, 3]([1.0, 12.0, 12.0])
    var r = milp(
        c,
        integrality=[1, 1],
        bounds=Bounds(
            [0.0, 0.0], [_inf[DType.float64](), _inf[DType.float64]()]
        ),
        constraints=[LinearConstraint(a, lo, hi)],
    )
    assert_equal(r.status, 0)
    assert_almost_equal(r.fun, -2.0, atol=1e-6)
    assert_almost_equal(r.mip_gap, 0.0, atol=1e-9)


def test_knapsack() raises:
    """A 0-1 knapsack of eight items: every variable binary, one capacity row.
    """
    var c = Static[dtype, 8](
        [-21.0, -10.0, -19.0, -37.0, -24.0, -7.0, -23.0, -9.0]
    )
    var a = Static[dtype, 1, 8]([5.0, 5.0, 16.0, 11.0, 13.0, 13.0, 15.0, 3.0])
    var lo = Static[dtype, 1]([-_inf[DType.float64]()])
    var hi = Static[dtype, 1]([36.45])
    var r = milp(
        c,
        integrality=[1, 1, 1, 1, 1, 1, 1, 1],
        bounds=Bounds(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        constraints=[LinearConstraint(a, lo, hi)],
    )
    assert_equal(r.status, 0)
    assert_almost_equal(r.fun, -92.0, atol=1e-6)
    assert_almost_equal(r.mip_gap, 0.0, atol=1e-9)
    var x = r.x.to_host()
    var want: List[Float64] = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    for i in range(8):
        assert_almost_equal(Float64(x[i]), want[i], atol=1e-5)


def test_mixed_integer() raises:
    """Three integer and two continuous variables in the box [-1, 4], two-sided rows and one equality row.
    """
    var c = Static[dtype, 5](
        [
            0.7468856162565439,
            -1.8473247989741095,
            1.5665487746995206,
            -0.09643216015562055,
            0.6803784532741461,
        ]
    )
    var a = Static[dtype, 5, 5](
        [
            -0.13656633397682774,
            -0.3790985670748533,
            0.46311015859758675,
            0.824513527530113,
            -0.20252987069345152,
            -0.15278617857019708,
            0.685698610809258,
            -0.8703406419471712,
            -1.5143835037313955,
            0.39498186274953,
            -0.6705658236878794,
            -1.9203405901180286,
            -0.8140536639453595,
            -0.467597558892747,
            -1.1932024774322612,
            -1.4924638840630338,
            0.03663782694480509,
            0.8972492567277476,
            -0.23313207796045685,
            -0.7435960295088448,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    )
    var lo = Static[dtype, 5](
        [
            -0.9602761224435798,
            -_inf[DType.float64](),
            -6.858052416348139,
            -5.289890315971711,
            7.0,
        ]
    )
    var hi = Static[dtype, 5](
        [
            1.9104923062221448,
            -1.5280525836382255,
            -4.449238876938654,
            -2.9426952516254348,
            7.0,
        ]
    )
    var r = milp(
        c,
        integrality=[1, 0, 1, 1, 0],
        bounds=Bounds(
            [-1.0, -1.0, -1.0, -1.0, -1.0], [4.0, 4.0, 4.0, 4.0, 4.0]
        ),
        constraints=[LinearConstraint(a, lo, hi)],
    )
    assert_equal(r.status, 0)
    assert_almost_equal(r.fun, -6.308084597019524, atol=1e-6)
    assert_almost_equal(r.mip_gap, 0.0, atol=1e-9)
    var x = r.x.to_host()
    var want: List[Float64] = [
        2.0,
        2.8525244465883,
        -1.0,
        4.0,
        -0.8525244465883003,
    ]
    for i in range(5):
        assert_almost_equal(Float64(x[i]), want[i], atol=1e-5)


def test_integer_infeasible() raises:
    """2 x0 = 1 has the relaxation's solution 0.5 and no integer one: status 2.
    """
    var c = Static[dtype, 1]([1.0])
    var a = Static[dtype, 1, 1]([2.0])
    var lo = Static[dtype, 1]([1.0])
    var hi = Static[dtype, 1]([1.0])
    var r = milp(
        c,
        integrality=[1],
        bounds=Bounds([0.0], [10.0]),
        constraints=[LinearConstraint(a, lo, hi)],
    )
    assert_equal(r.status, 2)


def test_continuous_matches_linprog() raises:
    """With every variable continuous, `milp` is one relaxation and agrees
    with `linprog` on the same rows."""
    var c = Static[dtype, 3]([-3.0, -1.0, -2.0])
    var a = Static[dtype, 2, 3]([1.0, 1.0, 1.0, 2.0, 0.5, 1.0])
    var b = Static[dtype, 2]([4.0, 5.0])
    var lp = linprog(c, a, b)
    var mip = milp(
        c, constraints=[LinearConstraint(a, -_inf[DType.float64](), 5.0)]
    )
    var tight = milp(c, constraints=[LinearConstraint(a, zeros[dtype, 2](), b)])
    assert_equal(tight.status, 0)
    assert_equal(tight.mip_node_count, 1)
    assert_almost_equal(tight.fun, lp.fun, atol=1e-7)
    assert_equal(mip.status, 0)
    assert_true(mip.fun <= lp.fun + 1e-9)


def test_node_limit() raises:
    """A limit of one node stops after the fractional root: status 1, no
    point, and the root's bound as the dual bound."""
    var c = Static[dtype, 1]([-1.0])
    var a = Static[dtype, 1, 1]([2.0])
    var r = milp(
        c,
        integrality=[1],
        constraints=[LinearConstraint(a, -_inf[DType.float64](), 3.0)],
        node_limit=1,
    )
    assert_equal(r.status, 1)
    assert_equal(r.mip_node_count, 1)
    assert_equal(r.x.size(), 0)
    assert_almost_equal(r.mip_dual_bound, -1.5, atol=1e-6)


def test_argument_errors() raises:
    """Semi-continuous integrality and a column mismatch raise, naming
    `milp`."""
    var c = Static[dtype, 2]([1.0, 1.0])
    with assert_raises(contains="semi-continuous"):
        _ = milp(c, integrality=[2])
    var a = Static[dtype, 1, 3]([1.0, 1.0, 1.0])
    with assert_raises(contains="milp"):
        _ = milp(c, constraints=[LinearConstraint(a, 0.0, 1.0)])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
