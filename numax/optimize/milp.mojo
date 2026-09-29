"""Mixed-integer linear programming over `Tensor`: `milp`,
`scipy.optimize.milp`, with the `LinearConstraint` it takes and the
`MilpResult` it returns.

**Tier 2.** Branch and bound over `linprog`'s interior-point relaxation:

1. **Relaxation.** Each node is the LP with the node's bounds, solved by
   the same homogeneous self-dual method (so at `gpu=True` each node's
   normal matrix is formed and factored on the device). An infeasible
   node is dropped, and one whose bound cannot beat the incumbent is
   pruned before and after its solve.
2. **Branching.** On the integer variable farthest from an integer
   (beyond `100 tol`, and at least `1e-6`), into `x_j <= floor(v)` and
   `x_j >= ceil(v)`; the side `v` is nearer is explored first, depth
   first, so an incumbent appears early and prunes the rest.
3. **Incumbent.** A relaxation integral on every integer variable is a
   candidate; its integer entries are rounded, and it replaces the
   incumbent when its objective is lower.

`mip_dual_bound` is the smallest bound among the nodes still open (the
incumbent's objective once the tree is exhausted), and `mip_gap` is
`(fun - mip_dual_bound) / |fun|`, both as SciPy reports them.

**Ceiling.** Plain depth-first branch and bound: no presolve, cutting
planes, heuristics or warm start, all of which HiGHS brings, so the node
count grows exponentially on hard instances where HiGHS's does not;
`node_limit` bounds it and a run that reaches it returns status `1` with
the best point found. An interior-point relaxation stops at the center of
an optimal face rather than a vertex, so a problem with tied optima
branches more than a simplex-based one would. Semi-continuous and
semi-integer variables (`integrality` `2` and `3`) are not taken.

## The MAX gate

Nothing: MAX has no optimizer of any kind. **Extend.**
"""

from std.math import ceil as _ceil, floor as _floor, inf as _inf

from max.gpu.host import DeviceContext

from ..core._drive import _check_device, _notice
from ..core.tensorlike import TensorLike
from ..core.tensor import Dynamic
from .linprog import Bounds, _dot, _host, _lp_host, _to_device


struct LinearConstraint(Copyable, Movable):
    """`lb <= A x <= ub`, row by row, `scipy.optimize.LinearConstraint`.

    A row with `lb == ub` is an equality; an infinite side is no bound.
    Held on the host as `Float64`, read once when the constraint is built.
    """

    var a: List[Float64]
    """The `m x n` matrix, row-major."""
    var lb: List[Float64]
    """The `m` lower bounds."""
    var ub: List[Float64]
    """The `m` upper bounds."""
    var m: Int
    """Rows."""
    var n: Int
    """Columns: the variable count."""

    def __init__[
        A: TensorLike, L: TensorLike, U: TensorLike
    ](out self, a: A, lb: L, ub: U) raises where (
        A.LayoutType.rank == 2
        and L.LayoutType.rank == 1
        and U.LayoutType.rank == 1
    ):
        """Per-row bounds from tensors.

        Parameters:
            A: The tensor type of the matrix.
            L: The tensor type of the lower bounds.
            U: The tensor type of the upper bounds.

        Args:
            a: The `m x n` matrix.
            lb: The `m` lower bounds.
            ub: The `m` upper bounds.

        Raises:
            If `lb` or `ub` is not `m` long, or a read-back fails.
        """
        self.m = a.dim_at(0)
        self.n = a.dim_at(1)
        self.a = _host(a)
        self.lb = _host(lb)
        self.ub = _host(ub)
        if len(self.lb) != self.m or len(self.ub) != self.m:
            raise Error(
                "LinearConstraint: lb and ub must have ", self.m, " entries"
            )

    def __init__[
        A: TensorLike
    ](out self, a: A, lb: Float64, ub: Float64) raises where (
        A.LayoutType.rank == 2
    ):
        """The same bounds on every row.

        Parameters:
            A: The tensor type of the matrix.

        Args:
            a: The `m x n` matrix.
            lb: The lower bound of every row; `-inf` for none.
            ub: The upper bound of every row; `inf` for none.

        Raises:
            If the read-back fails.
        """
        self.m = a.dim_at(0)
        self.n = a.dim_at(1)
        self.a = _host(a)
        self.lb = List[Float64](length=self.m, fill=lb)
        self.ub = List[Float64](length=self.m, fill=ub)


struct MilpResult[dtype: DType](Movable):
    """What `milp` returns, SciPy's `OptimizeResult` for `milp`."""

    var x: Dynamic[Self.dtype, 1]
    """The best point found, on `c`'s device; integer entries are exact
    integers. Empty when no feasible point was found."""
    var fun: Float64
    """`c^T x`; `nan` when no feasible point was found."""
    var status: Int
    """SciPy's code: `0` optimal, `1` node limit, `2` infeasible, `3`
    unbounded, `4` other (a relaxation hit numerical trouble)."""
    var message: String
    """SciPy's message for `status`."""
    var success: Bool
    """`status == 0`."""
    var mip_node_count: Int
    """Relaxations solved."""
    var mip_dual_bound: Float64
    """A lower bound on the optimum."""
    var mip_gap: Float64
    """`(fun - mip_dual_bound) / |fun|`."""

    def __init__(
        out self,
        var x: Dynamic[Self.dtype, 1],
        fun: Float64,
        status: Int,
        mip_node_count: Int,
        mip_dual_bound: Float64,
        mip_gap: Float64,
    ):
        """Build from the parts; `message` and `success` follow `status`.

        Args:
            x: The best point.
            fun: Its objective.
            status: SciPy's status code.
            mip_node_count: The node count.
            mip_dual_bound: The dual bound.
            mip_gap: The relative gap.
        """
        self.x = x^
        self.fun = fun
        self.status = status
        self.message = _message(status)
        self.success = status == 0
        self.mip_node_count = mip_node_count
        self.mip_dual_bound = mip_dual_bound
        self.mip_gap = mip_gap


def _message(status: Int) -> String:
    if status == 0:
        return "Optimization terminated successfully."
    if status == 1:
        return "Node limit reached."
    if status == 2:
        return "Problem is infeasible."
    if status == 3:
        return "Problem is unbounded."
    return "A relaxation encountered numerical difficulties."


struct _Node(Copyable, Movable):
    var lb: List[Float64]
    var ub: List[Float64]
    var bound: Float64

    def __init__(
        out self, var lb: List[Float64], var ub: List[Float64], bound: Float64
    ):
        self.lb = lb^
        self.ub = ub^
        self.bound = bound


def _milp[
    dtype: DType, gpu: Bool
](
    c: List[Float64],
    a_ub: List[Float64],
    b_ub: List[Float64],
    a_eq: List[Float64],
    b_eq: List[Float64],
    var lb: List[Float64],
    var ub: List[Float64],
    integer: List[Bool],
    node_limit: Int,
    tol: Float64,
    ctx: DeviceContext,
) raises -> MilpResult[dtype] where dtype.is_floating_point():
    var n = len(c)
    var int_tol = max(1e-6, 100.0 * tol)
    var infinity = _inf[DType.float64]()
    for j in range(n):
        if integer[j]:
            lb[j] = _ceil(lb[j] - int_tol)
            ub[j] = _floor(ub[j] + int_tol)
    var stack = List[_Node]()
    stack.append(_Node(lb^, ub^, -infinity))
    var incumbent = infinity
    var best = List[Float64]()
    var nodes = 0
    var trouble = False
    var status = 0
    while len(stack) > 0:
        if nodes >= node_limit:
            status = 1
            break
        var node = stack.pop()
        var slack = 1e-9 * max(1.0, abs(incumbent))
        if node.bound >= incumbent - slack:
            continue
        var infeasible_box = False
        for j in range(n):
            if node.lb[j] > node.ub[j]:
                infeasible_box = True
        if infeasible_box:
            continue
        nodes += 1
        var lp = _lp_host[dtype, gpu](
            c,
            a_ub,
            b_ub,
            a_eq,
            b_eq,
            Bounds(node.lb.copy(), node.ub.copy()),
            tol,
            1000,
            ctx,
        )
        var lp_status = lp[1]
        if lp_status == 2:
            continue
        if lp_status == 3:
            status = 3
            best = List[Float64]()
            break
        if lp_status != 0:
            trouble = True
            continue
        var x = lp[0].copy()
        var value = _dot(c, x)
        if value >= incumbent - slack:
            continue
        var branch = -1
        var widest = int_tol
        for j in range(n):
            if integer[j]:
                var frac = abs(x[j] - _floor(x[j] + 0.5))
                if frac > widest:
                    widest = frac
                    branch = j
        if branch < 0:
            for j in range(n):
                if integer[j]:
                    x[j] = _floor(x[j] + 0.5)
            incumbent = _dot(c, x)
            best = x^
            continue
        var v = x[branch]
        var down_ub = node.ub.copy()
        down_ub[branch] = _floor(v)
        var up_lb = node.lb.copy()
        up_lb[branch] = _ceil(v)
        var down = _Node(node.lb.copy(), down_ub^, value)
        var up = _Node(up_lb^, node.ub.copy(), value)
        # The nearer side last, so it is popped first.
        if v - _floor(v) < 0.5:
            stack.append(up^)
            stack.append(down^)
        else:
            stack.append(down^)
            stack.append(up^)

    var found = len(best) > 0
    if status == 0 and not found:
        status = 4 if trouble else 2
    elif status == 0 and trouble:
        status = 4
    var dual = incumbent
    if status == 1:
        for k in range(len(stack)):
            dual = min(dual, stack[k].bound)
    var fun = incumbent if found else Float64(0) / Float64(0)
    var gap = 0.0
    if found:
        gap = (incumbent - dual) / max(abs(incumbent), 1e-300)
    elif status == 3:
        dual = -infinity
    return MilpResult[dtype](
        _to_device[dtype](best, ctx), fun, status, nodes, dual, gap
    )


def milp[
    C: TensorLike, gpu: Bool = False
](
    c: C,
    integrality: List[Int] = List[Int](),
    bounds: Optional[Bounds] = None,
    constraints: List[LinearConstraint] = List[LinearConstraint](),
    node_limit: Int = 10000,
    tol: Optional[Float64] = None,
) raises -> MilpResult[C.dtype] where (
    C.dtype.is_floating_point() and C.LayoutType.rank == 1
):
    """`min c^T x` over `constraints` and `bounds` with the variables
    `integrality` marks restricted to integers. `scipy.optimize.milp(c,
    integrality=..., bounds=..., constraints=...)`.

    The module docstring has the algorithm and its ceiling.

    Parameters:
        C: The tensor type of the length-`n` cost vector.
        gpu: Solve every relaxation's normal equations on `c`'s device; a
            residency mismatch falls back to the host with a notice.

    Args:
        c: The cost vector.
        integrality: `0` continuous, `1` integer, per variable, or one
            entry for all; every variable continuous when empty.
        bounds: The per-variable bounds; `x >= 0` when omitted, SciPy's
            default.
        constraints: The linear constraints, each `lb <= A x <= ub`.
        node_limit: The most relaxations to solve.
        tol: The relaxations' tolerance; `linprog`'s default when omitted.

    Returns:
        A `MilpResult`: `x`, `fun`, SciPy's `status` and `message`,
        `success`, `mip_node_count`, `mip_dual_bound` and `mip_gap`.

    Raises:
        On an `integrality` entry other than `0` or `1`, mismatched
        shapes, bounds with `lb > ub`, or a device failure.
    """
    var n = c.size()
    if n == 0:
        raise Error("milp: c must have at least one element")
    var integer = List[Bool](capacity=n)
    if (
        len(integrality) != 0
        and len(integrality) != 1
        and len(integrality) != n
    ):
        raise Error("milp: integrality must have 0, 1 or ", n, " entries")
    for j in range(n):
        var code = 0
        if len(integrality) == 1:
            code = integrality[0]
        elif len(integrality) == n:
            code = integrality[j]
        if code != 0 and code != 1:
            raise Error(
                "milp: integrality must be 0 or 1; semi-continuous (2) and"
                " semi-integer (3) variables are not supported"
            )
        integer.append(code == 1)
    var box = bounds.value().copy() if bounds else Bounds(
        0.0, _inf[DType.float64]()
    )
    if (len(box.lb) != 1 and len(box.lb) != n) or (
        len(box.ub) != 1 and len(box.ub) != n
    ):
        raise Error("milp: bounds must have 1 or ", n, " entries a side")
    var lb = List[Float64](capacity=n)
    var ub = List[Float64](capacity=n)
    for j in range(n):
        lb.append(box.lower(j))
        ub.append(box.upper(j))
    var a_ub = List[Float64]()
    var b_ub = List[Float64]()
    var a_eq = List[Float64]()
    var b_eq = List[Float64]()
    var infinity = _inf[DType.float64]()
    for k in range(len(constraints)):
        ref con = constraints[k]
        if con.n != n:
            raise Error("milp: a constraint must have ", n, " columns")
        for i in range(con.m):
            if con.lb[i] > con.ub[i]:
                raise Error("milp: a constraint row has lb > ub")
            if con.lb[i] == con.ub[i]:
                for j in range(n):
                    a_eq.append(con.a[i * n + j])
                b_eq.append(con.ub[i])
                continue
            if con.ub[i] < infinity:
                for j in range(n):
                    a_ub.append(con.a[i * n + j])
                b_ub.append(con.ub[i])
            if con.lb[i] > -infinity:
                for j in range(n):
                    a_ub.append(-con.a[i * n + j])
                b_ub.append(-con.lb[i])
    var t = tol.value() if tol else (1e-8 if C.dtype == DType.float64 else 1e-5)
    var ctx = c.context()
    var ch = _host(c)
    if _check_device[C, gpu](c):
        return _milp[C.dtype, gpu](
            ch, a_ub, b_ub, a_eq, b_eq, lb^, ub^, integer, node_limit, t, ctx
        )
    _notice[gpu]("milp")
    return _milp[C.dtype, False](
        ch, a_ub, b_ub, a_eq, b_eq, lb^, ub^, integer, node_limit, t, ctx
    )
