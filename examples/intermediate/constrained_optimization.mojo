"""Constrained optimization over a `Tensor`: `linprog`, `milp` and
`minimize(method="slsqp")`, the three `scipy.optimize` entry points for a
problem with constraints.

```mojo
var lp = linprog(c, a_ub, b_ub)                                  # an LP
var ip = milp(c, integrality=[1], constraints=[LinearConstraint(a, lo, hi)])
var nl = minimize[f=f, jac=grad](x0, NonlinearConstraint[...](lb, ub))
```

A small production plan is solved three ways. As a linear program the
optimum is fractional; `milp` makes the quantities whole and pays for it
in profit, and reports how many relaxations the branch and bound took.
Then a nonlinear version -- the same profit with diminishing returns,
under a quadratic resource budget -- goes to SLSQP, whose answer is
checked against the constraint it must sit on.

Run: `pixi run example-constrained-optimization`
"""

from std.math import inf, sqrt

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    linprog,
    milp,
    minimize,
)

comptime dtype = DType.float64
comptime INF = inf[DType.float64]()


# Profit per unit of three products, and the two resources each uses.
def profit() raises -> Static[dtype, 3]:
    return Static[dtype, 3]([-5.0, -4.0, -3.0])  # negated: linprog minimizes


def usage() raises -> Static[dtype, 2, 3]:
    return Static[dtype, 2, 3]([3.0, 2.0, 4.0, 2.0, 5.0, 1.0])


def capacity() raises -> Static[dtype, 2]:
    return Static[dtype, 2]([10.0, 9.0])


# The nonlinear plan: profit with diminishing returns, sqrt(1 + x) - 1 per
# product, under a quadratic budget x^T x <= 4.
def returns(p: Static[dtype, 3], ctx: DeviceContext) raises -> Scalar[dtype]:
    var v = p.to_host()
    var total = Scalar[dtype](0)
    var weights: List[Float64] = [5.0, 4.0, 3.0]
    for i in range(3):
        total -= Scalar[dtype](weights[i]) * (sqrt(1 + v[i]) - 1)
    return total


def returns_grad(
    p: Static[dtype, 3], ctx: DeviceContext
) raises -> Static[dtype, 3]:
    var v = p.to_host()
    var weights: List[Float64] = [5.0, 4.0, 3.0]
    var g = List[Scalar[dtype]](capacity=3)
    for i in range(3):
        g.append(-Scalar[dtype](weights[i]) / (2 * sqrt(1 + v[i])))
    return Static[dtype, 3](g^, ctx)


def budget(p: Static[dtype, 3], ctx: DeviceContext) raises -> Static[dtype, 1]:
    var v = p.to_host()
    return Static[dtype, 1]([v[0] * v[0] + v[1] * v[1] + v[2] * v[2]], ctx)


def budget_jac(
    p: Static[dtype, 3], ctx: DeviceContext
) raises -> Static[dtype, 1, 3]:
    var v = p.to_host()
    return Static[dtype, 1, 3]([2 * v[0], 2 * v[1], 2 * v[2]], ctx)


def main() raises:
    # --- The linear program: max profit under both capacities, x >= 0.
    var lp = linprog(profit(), usage(), capacity())
    var x = lp.x.to_host()
    print("linprog:", lp.message)
    print("  x =", x[0], x[1], x[2], " profit =", -lp.fun)
    var slack = lp.slack.to_host()
    print("  unused capacity:", slack[0], slack[1])

    # --- Whole units only.
    var lo = Static[dtype, 2]([-INF, -INF])
    var ip = milp(
        profit(),
        integrality=[1],
        constraints=[LinearConstraint(usage(), lo, capacity())],
    )
    var xi = ip.x.to_host()
    print("milp:", ip.message)
    print("  x =", xi[0], xi[1], xi[2], " profit =", -ip.fun)
    print("  nodes:", ip.mip_node_count, " gap:", ip.mip_gap)

    # --- Diminishing returns under a quadratic budget.
    var x0 = Static[dtype, 3]([0.1, 0.1, 0.1])
    var nl = minimize[f=returns, jac=returns_grad](
        x0,
        NonlinearConstraint[dtype, 3, 1, budget, budget_jac](-INF, 4.0),
        bounds=Bounds(0.0, INF),
    )
    var xn = nl.x.to_host()
    print("slsqp: converged =", nl.converged, "in", nl.iterations, "iterations")
    print("  x =", xn[0], xn[1], xn[2], " value =", -nl.f_x)
    print(
        "  budget used:",
        xn[0] * xn[0] + xn[1] * xn[1] + xn[2] * xn[2],
        "of 4 (the optimum sits on the constraint)",
    )
