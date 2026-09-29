"""Sequential quadratic programming over `Tensor`: the engine behind
`minimize(method="slsqp")`, SciPy's `method="SLSQP"`, and the
`NonlinearConstraint` it takes.

**Tier 2.** Kraft's SLSQP at the level of its algorithm, with a different
subproblem solver:

1. **Subproblem.** At `x`, `min (1/2) d^T B d + g^T d` subject to the
   linearized constraints `J_eq d + c_eq = 0` and `J_in d + c_in >= 0`
   and the bounds `lb - x <= d <= ub - x`, solved exactly by Goldfarb and
   Idnani's dual active-set method (`numax.optimize._qp`), where Kraft's
   Fortran solves the same program as a least-squares problem through
   `LSEI`/`NNLS`. Both return the subproblem's minimizer and multipliers.
2. **Merit and line search.** Powell's `L1` merit `f + sum rho_i
   |violation_i|`, each `rho_i = max(|lambda_i|, (rho_i + |lambda_i|)/2)`,
   with Kraft's Armijo test (`0.1` of the predicted decrease) and his
   interpolated backtracking, at most ten trials, never below a tenth of
   the previous step.
3. **Hessian.** `B` starts at the identity and takes Powell's damped
   BFGS update on the Lagrangian's gradient, so it stays positive
   definite.
4. **Stopping.** Kraft's two tests: before the line search, the
   predicted change `|g^T d| + sum |lambda_i c_i|` and the total
   violation both below `tol`; after it, `|f - f_prev|` or `|d|` below
   `tol` with the violation below `tol`.

Iterates stay inside the bounds (the subproblem enforces them and a step
is a convex combination), so `f` and the constraints are only ever
evaluated there, as SciPy guarantees. `f`, the gradient and the
constraints are the caller's compile-time functions, evaluated on
`x0`'s device; the subproblem is `O(n^3)` host `Float64`, sized by the
variable count.

**Ceiling.** An inconsistent linearization -- constraints whose
linearization has no solution although the problem is feasible -- ends
the run with `converged=False` where Kraft's relaxes the subproblem with
an extra variable. `B` is dense, so `n` in the hundreds is the practical
limit, as it is for SciPy's.

## The MAX gate

Nothing: MAX has no optimizer of any kind. **Extend.**
"""

from std.math import inf as _inf, isnan as _isnan, sqrt as _sqrt

from max.gpu.host import DeviceContext

from ..core.tensor import Static, zeros
from .common import _as_tensor
from .linprog import Bounds, _host
from .milp import LinearConstraint
from ._qp import solve_qp


struct NonlinearConstraint[
    dtype: DType,
    n: Int,
    m: Int,
    fun: def(Static[dtype, n], DeviceContext) raises thin -> Static[dtype, m],
    jac: def(Static[dtype, n], DeviceContext) raises thin -> Static[
        dtype, m, n
    ],
](Copyable, Movable):
    """`lb <= fun(x) <= ub`, row by row, `scipy.optimize.NonlinearConstraint`.

    `fun` returns the `m` constraint values at a point and `jac` their `m x
    n` Jacobian; both are compile-time parameters, like `minimize`'s `f`
    and `jac`. A row with `lb == ub` is an equality; an infinite side is
    no bound.

    Parameters:
        dtype: The element type of the variables.
        n: The variable count.
        m: The constraint count.
        fun: The constraint values at a point.
        jac: Their Jacobian at a point.
    """

    var lb: List[Float64]
    """The `m` lower bounds."""
    var ub: List[Float64]
    """The `m` upper bounds."""

    def __init__(out self, lb: Float64, ub: Float64):
        """The same bounds on every row.

        Args:
            lb: The lower bound of every row; `-inf` for none.
            ub: The upper bound of every row; `inf` for none.
        """
        self.lb = List[Float64](length=Self.m, fill=lb)
        self.ub = List[Float64](length=Self.m, fill=ub)

    def __init__(out self, var lb: List[Float64], var ub: List[Float64]) raises:
        """Per-row bounds.

        Args:
            lb: The `m` lower bounds.
            ub: The `m` upper bounds.

        Raises:
            If either list is not `m` long.
        """
        if len(lb) != Self.m or len(ub) != Self.m:
            raise Error(
                "NonlinearConstraint: lb and ub must have ", Self.m, " entries"
            )
        self.lb = lb^
        self.ub = ub^


def _no_constraint[
    dtype: DType, n: Int
](x: Static[dtype, n], ctx: DeviceContext) raises -> Static[dtype, 0]:
    """No constraint rows: the nonlinear slot of a problem that has
    none."""
    return zeros[dtype, 0](ctx)


def _no_constraint_jac[
    dtype: DType, n: Int
](x: Static[dtype, n], ctx: DeviceContext) raises -> Static[dtype, 0, n]:
    return zeros[dtype, 0, n](ctx)


struct _Row(Copyable, Movable):
    """One standardized constraint: `sign * (source value) - offset`, as an
    equality or `>= 0`. `source < 0` names linear row `-source - 1`."""

    var source: Int
    var sign: Float64
    var offset: Float64

    def __init__(out self, source: Int, sign: Float64, offset: Float64):
        self.source = source
        self.sign = sign
        self.offset = offset


def _classify(
    lb: List[Float64],
    ub: List[Float64],
    base: Int,
    mut eq: List[_Row],
    mut ineq: List[_Row],
) raises:
    """`lb <= v <= ub` rows as `v - lb = 0`, `v - lb >= 0`, `ub - v >= 0`."""
    var infinity = _inf[DType.float64]()
    for i in range(len(lb)):
        if lb[i] > ub[i]:
            raise Error("minimize: a constraint row has lb > ub")
        if lb[i] == ub[i]:
            eq.append(_Row(base + i, 1.0, lb[i]))
            continue
        if lb[i] > -infinity:
            ineq.append(_Row(base + i, 1.0, lb[i]))
        if ub[i] < infinity:
            ineq.append(_Row(base + i, -1.0, -ub[i]))


def _violation(values: List[Float64], meq: Int) -> Float64:
    """The total violation: `|c|` on the equalities, `max(0, -c)` on the
    inequalities."""
    var total = 0.0
    for r in range(len(values)):
        total += abs(values[r]) if r < meq else max(0.0, -values[r])
    return total


def _merit(
    value: Float64, values: List[Float64], rho: List[Float64], meq: Int
) -> Float64:
    """Powell's `L1` merit, `f + sum rho_i violation_i`."""
    var total = value
    for r in range(len(values)):
        var v = abs(values[r]) if r < meq else max(0.0, -values[r])
        total += rho[r] * v
    return total


def _slsqp[
    dtype: DType,
    n: Int,
    m: Int,
    f: def(Static[dtype, n], DeviceContext) raises thin -> Scalar[dtype],
    jac: def(Static[dtype, n], DeviceContext) raises thin -> Static[dtype, n],
    cf: def(Static[dtype, n], DeviceContext) raises thin -> Static[dtype, m],
    cj: def(Static[dtype, n], DeviceContext) raises thin -> Static[dtype, m, n],
](
    var x: List[Float64],
    nl_lb: List[Float64],
    nl_ub: List[Float64],
    linear: List[LinearConstraint],
    bounds: Optional[Bounds],
    ctx: DeviceContext,
    tol: Float64,
    max_iter: Int,
) raises -> Tuple[List[Float64], Float64, Float64, Int, Bool]:
    """The run from `x`: the final point, `f` there, the Lagrangian's
    gradient infinity-norm, the iteration count and whether a stopping
    test was met. The module docstring has the steps."""
    var infinity = _inf[DType.float64]()
    var lo = List[Float64](length=n, fill=-infinity)
    var hi = List[Float64](length=n, fill=infinity)
    if bounds:
        ref box = bounds.value()
        if (len(box.lb) != 1 and len(box.lb) != n) or (
            len(box.ub) != 1 and len(box.ub) != n
        ):
            raise Error(
                "minimize: bounds must have 1 or ", n, " entries a side"
            )
        for j in range(n):
            lo[j] = box.lower(j)
            hi[j] = box.upper(j)
            if lo[j] > hi[j]:
                raise Error("minimize: bounds must satisfy lb <= ub")
            x[j] = min(max(x[j], lo[j]), hi[j])

    # Linear rows stacked after the nonlinear ones: one value vector, one
    # Jacobian.
    var lin_a = List[Float64]()
    var lin_lb = List[Float64]()
    var lin_ub = List[Float64]()
    for k in range(len(linear)):
        ref con = linear[k]
        if con.n != n:
            raise Error(
                "minimize: a LinearConstraint must have ", n, " columns"
            )
        for e in range(con.m * n):
            lin_a.append(con.a[e])
        for i in range(con.m):
            lin_lb.append(con.lb[i])
            lin_ub.append(con.ub[i])
    var n_lin = len(lin_lb)
    var eq = List[_Row]()
    var ineq = List[_Row]()
    _classify(nl_lb, nl_ub, 0, eq, ineq)
    _classify(lin_lb, lin_ub, m, eq, ineq)
    var meq = len(eq)
    var rows = List[_Row](capacity=meq + len(ineq))
    for i in range(meq):
        rows.append(eq[i].copy())
    for i in range(len(ineq)):
        rows.append(ineq[i].copy())
    var n_rows = len(rows)
    var n_bound = 0
    for j in range(n):
        if lo[j] > -infinity:
            n_bound += 1
        if hi[j] < infinity:
            n_bound += 1

    def evaluate(
        point: List[Float64],
    ) raises {imm} -> Tuple[Float64, List[Float64]]:
        var xs = _as_tensor[dtype, n](point, ctx)
        var value = Float64(f(xs, ctx))
        var values = List[Float64](capacity=n_rows)
        if m + n_lin > 0:
            var raw = List[Float64](capacity=m + n_lin)
            comptime if m > 0:
                var nl = _host(cf(xs, ctx))
                for i in range(m):
                    raw.append(nl[i])
            for i in range(n_lin):
                var s = 0.0
                for j in range(n):
                    s += lin_a[i * n + j] * point[j]
                raw.append(s)
            for r in range(n_rows):
                values.append(
                    rows[r].sign * raw[rows[r].source] - rows[r].offset
                )
        return (value, values^)

    def derivatives(
        point: List[Float64],
    ) raises {imm} -> Tuple[List[Float64], List[Float64]]:
        var xs = _as_tensor[dtype, n](point, ctx)
        var grad = _host(jac(xs, ctx))
        var jrows = List[Float64](length=n_rows * n, fill=0.0)
        if n_rows > 0:
            var raw = List[Float64](length=(m + n_lin) * n, fill=0.0)
            comptime if m > 0:
                var nl = _host(cj(xs, ctx))
                for e in range(m * n):
                    raw[e] = nl[e]
            for e in range(n_lin * n):
                raw[m * n + e] = lin_a[e]
            for r in range(n_rows):
                for j in range(n):
                    jrows[r * n + j] = (
                        rows[r].sign * raw[rows[r].source * n + j]
                    )
        return (grad^, jrows^)

    var hess = List[Float64](length=n * n, fill=0.0)
    for j in range(n):
        hess[j * n + j] = 1.0
    var rho = List[Float64](length=n_rows, fill=0.0)
    var ev = evaluate(x)
    var fx = ev[0]
    var cvals = ev[1].copy()
    var dv = derivatives(x)
    var grad = dv[0].copy()
    var jrows = dv[1].copy()
    var lam = List[Float64](length=n_rows, fill=0.0)
    var lam_bound = List[Float64](length=n, fill=0.0)
    var converged = False
    var iteration = 0
    while iteration < max_iter:
        iteration += 1
        # The subproblem's rows: the linearized constraints, then the
        # bounds on `d`.
        var total = n_rows + n_bound
        var cmat = List[Float64](length=total * n, fill=0.0)
        var rhs = List[Float64](length=total, fill=0.0)
        var bound_var = List[Int](capacity=n_bound)
        var bound_sign = List[Float64](capacity=n_bound)
        for r in range(n_rows):
            for j in range(n):
                cmat[r * n + j] = jrows[r * n + j]
            rhs[r] = -cvals[r]
        var row = n_rows
        for j in range(n):
            if lo[j] > -infinity:
                cmat[row * n + j] = 1.0
                rhs[row] = lo[j] - x[j]
                bound_var.append(j)
                bound_sign.append(1.0)
                row += 1
            if hi[j] < infinity:
                cmat[row * n + j] = -1.0
                rhs[row] = x[j] - hi[j]
                bound_var.append(j)
                bound_sign.append(-1.0)
                row += 1
        var neg_grad = List[Float64](capacity=n)
        for j in range(n):
            neg_grad.append(-grad[j])
        var qp = solve_qp(hess, neg_grad, cmat, rhs, meq, n)
        if not qp.feasible:
            break
        var d = qp.x.copy()
        for r in range(n_rows):
            lam[r] = qp.multipliers[r]
        for j in range(n):
            lam_bound[j] = 0.0
        for k in range(n_bound):
            lam_bound[bound_var[k]] += (
                bound_sign[k] * qp.multipliers[n_rows + k]
            )

        var h1 = 0.0
        for j in range(n):
            h1 += grad[j] * d[j]
        h1 = abs(h1)
        for r in range(n_rows):
            h1 += abs(lam[r] * cvals[r])
        var h2 = _violation(cvals, meq)
        if h1 < tol and h2 < tol:
            converged = True
            break

        for r in range(n_rows):
            rho[r] = max(abs(lam[r]), 0.5 * (rho[r] + abs(lam[r])))

        var phi0 = _merit(fx, cvals, rho, meq)
        var slope = 0.0
        for j in range(n):
            slope += grad[j] * d[j]
        for r in range(n_rows):
            var v = abs(cvals[r]) if r < meq else max(0.0, -cvals[r])
            slope -= rho[r] * v
        var alpha = 1.0
        var trial = List[Float64](length=n, fill=0.0)
        var fnew = fx
        var cnew = cvals.copy()
        for line in range(10):
            for j in range(n):
                trial[j] = min(max(x[j] + alpha * d[j], lo[j]), hi[j])
            var et = evaluate(trial)
            fnew = et[0]
            cnew = et[1].copy()
            var phi = _merit(fnew, cnew, rho, meq)
            var predicted = alpha * slope
            if _isnan(phi):
                alpha = 0.1 * alpha
                continue
            if phi - phi0 <= 0.1 * predicted or slope >= 0 or line == 9:
                break
            var shrink = predicted / (2.0 * (predicted - (phi - phi0)))
            alpha = alpha * max(0.1, min(shrink, 0.5))

        var s = List[Float64](capacity=n)
        var step_norm = 0.0
        for j in range(n):
            s.append(trial[j] - x[j])
            step_norm += s[j] * s[j]
        step_norm = _sqrt(step_norm)
        var dn = derivatives(trial)
        var grad_new = dn[0].copy()
        var jrows_new = dn[1].copy()
        # Powell's damped BFGS on the Lagrangian's gradient.
        var y = List[Float64](length=n, fill=0.0)
        for j in range(n):
            var diff = grad_new[j] - grad[j]
            for r in range(n_rows):
                diff -= lam[r] * (jrows_new[r * n + j] - jrows[r * n + j])
            y[j] = diff
        var bs = List[Float64](length=n, fill=0.0)
        var sbs = 0.0
        var sy = 0.0
        for i in range(n):
            var acc = 0.0
            for j in range(n):
                acc += hess[i * n + j] * s[j]
            bs[i] = acc
            sbs += s[i] * acc
            sy += s[i] * y[i]
        if sbs > 0:
            if sy < 0.2 * sbs:
                var theta = 0.8 * sbs / (sbs - sy)
                for j in range(n):
                    y[j] = theta * y[j] + (1.0 - theta) * bs[j]
                sy = 0.2 * sbs
            if sy > 0:
                for i in range(n):
                    for j in range(n):
                        hess[i * n + j] += (
                            y[i] * y[j] / sy - bs[i] * bs[j] / sbs
                        )
        var f_prev = fx
        x = trial^
        fx = fnew
        cvals = cnew^
        grad = grad_new^
        jrows = jrows_new^
        var h3 = _violation(cvals, meq)
        if (abs(fx - f_prev) < tol or step_norm < tol) and h3 < tol:
            converged = True
            break

    # The Lagrangian's gradient at the last point, reported as `grad_norm`.
    var kkt = 0.0
    for j in range(n):
        var v = grad[j] - lam_bound[j]
        for r in range(n_rows):
            v -= lam[r] * jrows[r * n + j]
        kkt = max(kkt, abs(v))
    return (x^, fx, kkt, iteration, converged)
