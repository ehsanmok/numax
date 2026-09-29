"""Linear programming over `Tensor`: `linprog`, `scipy.optimize.linprog`,
with the `Bounds` and `LinprogResult` it takes and returns.

**Tier 2.** SciPy's `method="interior-point"` (`_linprog_ip.py`, the
homogeneous self-dual method of Andersen and Andersen that SciPy shipped
before HiGHS replaced it), transcribed step for step:

1. **Standard form.** `min c^T x` over `A_ub x <= b_ub`, `A_eq x = b_eq`
   and `lb <= x <= ub` becomes `min c'^T x' + c0` over `A x' = b`, `x' >=
   0`: a finite lower bound shifts its variable to zero, a finite upper
   bound on a variable with a finite lower one adds a row and a slack, an
   upper bound alone flips the variable's sign, a free variable splits into
   two nonnegative ones, and each inequality row gains a slack.
2. **Homogeneous self-dual iteration.** From the blind start `x = z = 1`,
   `y = 0`, `tau = kappa = 1`: Mehrotra's predictor-corrector direction,
   whose two symmetric solves share one normal matrix `M = A diag(x / z)
   A^T`, a step to `0.99995` of the boundary, and SciPy's five indicators
   -- primal, dual and gap residuals relative to the blind start's, the
   relative objective gap and the path parameter.
3. **Verdict.** Converged when the first three indicators are below
   `tol`; infeasible or unbounded when `tau` collapses against `kappa`
   (Andersen and Andersen's certificate: the sign of `b^T y` decides
   which); status `4` when the normal matrix is singular or the direction
   is not finite.

At `gpu=True` the `O(m^2 N)` term -- the scaled `A diag(d)` and the
product with `A^T` through `numax.linalg.matmul` -- and the `O(m^3)`
factorization of `M` (`numax.linalg`'s blocked LU at a run-time order)
run on `c`'s device, each solve against it too. The `O(m N)` matrix-vector
products, the step length and the indicators are host `Float64`, which is
also what keeps the stopping test honest when the device is `float32`.

**Ceiling.** Dense throughout, where SciPy's `sparse=True` factored a
sparse `M`. There is no presolve: equality rows must be linearly
independent (SciPy's `presolve=False` contract), and a redundant row makes
`M` singular and ends the run with status `4`. At `float32` the normal
matrix loses the last digits as `x / z` spreads, and the default `tol` is
`1e-5` there, against SciPy's `1e-8` at `float64`. The HiGHS methods
SciPy now defaults to are simplex and a sparse IPM with crossover, so a
degenerate problem's `x` may differ from HiGHS's vertex while `fun`
agrees.

## The MAX gate

Nothing: MAX has no optimizer of any kind. The dense products delegate to
`linalg.matmul` through `numax.linalg`. **Extend.**
"""

from std.math import inf as _inf, isnan as _isnan, sqrt as _sqrt

from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core._drive import _check_device, _notice
from ..core.tensorlike import TensorLike
from ..core.tensor import Dynamic, _dyn_shape, _same_order, asarray
from ..linalg.blas import matmul
from ..linalg.lu import _LURuntime, _lu_factor_runtime


def _target[gpu: Bool]() -> StaticString:
    comptime if gpu:
        return "gpu"
    else:
        return "cpu"


struct Bounds(Copyable, Movable):
    """Per-variable bounds `lb <= x <= ub`, `scipy.optimize.Bounds`.

    Either side may be infinite (`std.math.inf`). Held on the host: bounds
    are read once, when a problem is put into standard form. A length-1
    `lb` or `ub` applies to every variable, the broadcast SciPy's scalar
    `Bounds(0, 1)` means.
    """

    var lb: List[Float64]
    """The lower bounds, one per variable or one for all."""
    var ub: List[Float64]
    """The upper bounds, one per variable or one for all."""

    def __init__(out self, lb: Float64, ub: Float64):
        """The same bounds on every variable.

        Args:
            lb: The lower bound; `-inf` for none.
            ub: The upper bound; `inf` for none.
        """
        self.lb = [lb]
        self.ub = [ub]

    def __init__(out self, var lb: List[Float64], var ub: List[Float64]):
        """Per-variable bounds from host lists.

        Args:
            lb: The lower bounds.
            ub: The upper bounds.
        """
        self.lb = lb^
        self.ub = ub^

    def __init__[
        L: TensorLike, U: TensorLike
    ](out self, lb: L, ub: U) raises where (
        L.LayoutType.rank == 1 and U.LayoutType.rank == 1
    ):
        """Per-variable bounds from two rank-1 tensors, read back once.

        Parameters:
            L: The tensor type of `lb`.
            U: The tensor type of `ub`.

        Args:
            lb: The lower bounds.
            ub: The upper bounds.

        Raises:
            If reading either tensor back fails.
        """
        self.lb = _host(lb)
        self.ub = _host(ub)

    def lower(self, i: Int) -> Float64:
        """Variable `i`'s lower bound, broadcasting a length-1 list."""
        return self.lb[0] if len(self.lb) == 1 else self.lb[i]

    def upper(self, i: Int) -> Float64:
        """Variable `i`'s upper bound, broadcasting a length-1 list."""
        return self.ub[0] if len(self.ub) == 1 else self.ub[i]


struct LinprogResult[dtype: DType](Movable):
    """What `linprog` returns, SciPy's `OptimizeResult` for `linprog`."""

    var x: Dynamic[Self.dtype, 1]
    """The solution, on `c`'s device."""
    var fun: Float64
    """`c^T x`."""
    var slack: Dynamic[Self.dtype, 1]
    """`b_ub - A_ub x`, nonnegative at a feasible `x`."""
    var con: Dynamic[Self.dtype, 1]
    """`b_eq - A_eq x`, zero at a feasible `x`."""
    var status: Int
    """SciPy's code: `0` success, `1` iteration limit, `2` infeasible,
    `3` unbounded, `4` numerical difficulties."""
    var message: String
    """SciPy's message for `status`."""
    var nit: Int
    """Iterations taken."""
    var success: Bool
    """`status == 0`."""

    def __init__(
        out self,
        var x: Dynamic[Self.dtype, 1],
        fun: Float64,
        var slack: Dynamic[Self.dtype, 1],
        var con: Dynamic[Self.dtype, 1],
        status: Int,
        nit: Int,
    ):
        """Build from the parts; `message` and `success` follow `status`.

        Args:
            x: The solution.
            fun: The objective at it.
            slack: The inequality slacks.
            con: The equality residuals.
            status: SciPy's status code.
            nit: The iteration count.
        """
        self.x = x^
        self.fun = fun
        self.slack = slack^
        self.con = con^
        self.status = status
        self.message = _message(status)
        self.nit = nit
        self.success = status == 0


def _message(status: Int) -> String:
    if status == 0:
        return "Optimization terminated successfully."
    if status == 1:
        return "The iteration limit was reached before the algorithm converged."
    if status == 2:
        return (
            "The algorithm terminated successfully and determined that the"
            " problem is infeasible."
        )
    if status == 3:
        return (
            "The algorithm terminated successfully and determined that the"
            " problem is unbounded."
        )
    return (
        "Numerical difficulties were encountered before the problem converged."
    )


def _host[T: TensorLike](t: T) raises -> List[Float64]:
    """Every element of `t`, row-major, as host `Float64`s."""
    var out = List[Float64](capacity=t.size())
    if t.size() == 0:
        return out^
    var flat = _same_order(t, row_major(_dyn_shape[1](t.size())))
    var h = flat.to_host[DType.float64]()
    for i in range(len(h)):
        out.append(Float64(h[i]))
    return out^


# ------------------------------------------------------------ host algebra


def _dot(a: List[Float64], b: List[Float64]) -> Float64:
    var s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def _norm(a: List[Float64]) -> Float64:
    return _sqrt(_dot(a, a))


def _ax(a: List[Float64], m: Int, n: Int, x: List[Float64]) -> List[Float64]:
    """`A x` for the row-major `m x n` `a`."""
    var out = List[Float64](length=m, fill=0.0)
    for i in range(m):
        var s = 0.0
        for j in range(n):
            s += a[i * n + j] * x[j]
        out[i] = s
    return out^


def _atx(a: List[Float64], m: Int, n: Int, y: List[Float64]) -> List[Float64]:
    """`A^T y` for the row-major `m x n` `a`."""
    var out = List[Float64](length=n, fill=0.0)
    for i in range(m):
        var yi = y[i]
        if yi != 0:
            for j in range(n):
                out[j] += a[i * n + j] * yi
    return out^


# ------------------------------------------------------------ standard form


struct _Standard(Movable):
    """`min c^T x + c0` over `A x = b`, `x >= 0`, and the map back to the
    caller's variables: `x_orig[j] = shift[j] + sign[j] x[j] - x[twin[j]]`
    with `twin[j] = -1` for a variable that did not split."""

    var a: List[Float64]
    var b: List[Float64]
    var c: List[Float64]
    var c0: Float64
    var m: Int
    var n: Int
    var shift: List[Float64]
    var sign: List[Float64]
    var twin: List[Int]

    def __init__(
        out self,
        var a: List[Float64],
        var b: List[Float64],
        var c: List[Float64],
        c0: Float64,
        m: Int,
        n: Int,
        var shift: List[Float64],
        var sign: List[Float64],
        var twin: List[Int],
    ):
        self.a = a^
        self.b = b^
        self.c = c^
        self.c0 = c0
        self.m = m
        self.n = n
        self.shift = shift^
        self.sign = sign^
        self.twin = twin^


def _place(
    mut a: List[Float64],
    mut b: List[Float64],
    row: Int,
    total_cols: Int,
    source: List[Float64],
    r: Int,
    rhs: Float64,
    n: Int,
    sign: List[Float64],
    twin: List[Int],
    shift: List[Float64],
):
    """Row `r` of `source` written as standard-form row `row`, its
    right-hand side moved by the shifts."""
    var acc = rhs
    for j in range(n):
        var v = source[r * n + j]
        a[row * total_cols + j] = sign[j] * v
        if twin[j] >= 0:
            a[row * total_cols + twin[j]] = -v
        acc -= v * shift[j]
    b[row] = acc


def _standard_form(
    c: List[Float64],
    a_ub: List[Float64],
    b_ub: List[Float64],
    a_eq: List[Float64],
    b_eq: List[Float64],
    bounds: Bounds,
) raises -> _Standard:
    var n = len(c)
    var m_ub = len(b_ub)
    var m_eq = len(b_eq)
    var shift = List[Float64](length=n, fill=0.0)
    var sign = List[Float64](length=n, fill=1.0)
    var twin = List[Int](length=n, fill=-1)
    # Rows `x'_j + s = ub_j - lb_j` for the doubly bounded variables.
    var boxed = List[Int]()
    var box_width = List[Float64]()
    var columns = n
    for j in range(n):
        var lo = bounds.lower(j)
        var hi = bounds.upper(j)
        if lo > hi:
            raise Error("linprog: bounds must satisfy lb <= ub, variable ", j)
        if lo > -_inf[DType.float64]():
            shift[j] = lo
            if hi < _inf[DType.float64]():
                boxed.append(j)
                box_width.append(hi - lo)
        elif hi < _inf[DType.float64]():
            shift[j] = hi
            sign[j] = -1.0
        else:
            twin[j] = columns
            columns += 1
    var n_box = len(boxed)
    var total_cols = columns + m_ub + n_box
    var m = m_ub + m_eq + n_box
    var a = List[Float64](length=m * total_cols, fill=0.0)
    var b = List[Float64](length=m, fill=0.0)
    var cs = List[Float64](length=total_cols, fill=0.0)
    var c0 = 0.0
    for j in range(n):
        cs[j] = sign[j] * c[j]
        if twin[j] >= 0:
            cs[twin[j]] = -c[j]
        c0 += c[j] * shift[j]

    for r in range(m_ub):
        _place(a, b, r, total_cols, a_ub, r, b_ub[r], n, sign, twin, shift)
        a[r * total_cols + columns + r] = 1.0
    for r in range(m_eq):
        _place(
            a, b, m_ub + r, total_cols, a_eq, r, b_eq[r], n, sign, twin, shift
        )
    for k in range(n_box):
        var row = m_ub + m_eq + k
        a[row * total_cols + boxed[k]] = 1.0
        a[row * total_cols + columns + m_ub + k] = 1.0
        b[row] = box_width[k]
    return _Standard(a^, b^, cs^, c0, m, total_cols, shift^, sign^, twin^)


# ------------------------------------------------------ the device pieces


struct _Normal[dtype: DType, gpu: Bool](
    Movable where dtype.is_floating_point()
):
    """`A` and `A^T` resident on the device, and the one product per
    iteration, `A diag(d) A^T`, factored there."""

    var a: Dynamic[Self.dtype, 2]
    var at: Dynamic[Self.dtype, 2]
    var m: Int
    var n: Int

    def __init__(
        out self, a: List[Float64], m: Int, n: Int, ctx: DeviceContext
    ) raises:
        var rows = List[Scalar[Self.dtype]](capacity=m * n)
        var cols = List[Scalar[Self.dtype]](capacity=m * n)
        for e in range(m * n):
            rows.append(Scalar[Self.dtype](a[e]))
        for j in range(n):
            for i in range(m):
                cols.append(Scalar[Self.dtype](a[i * n + j]))
        self.a = Dynamic[Self.dtype, 2](row_major(_dyn_shape[2](m, n)), ctx)
        self.a.copy_from_host(rows^)
        self.at = Dynamic[Self.dtype, 2](row_major(_dyn_shape[2](n, m)), ctx)
        self.at.copy_from_host(cols^)
        self.m = m
        self.n = n

    def factor(
        mut self, d: List[Float64]
    ) raises -> _LURuntime[
        Self.dtype, Self.gpu
    ] where Self.dtype.is_floating_point():
        """The LU of `A diag(d) A^T`."""
        var ctx = self.a.context()
        var dv_host = List[Scalar[Self.dtype]](capacity=self.n)
        for j in range(self.n):
            dv_host.append(Scalar[Self.dtype](d[j]))
        var dv = asarray(dv_host^, ctx)
        var scaled = Dynamic[Self.dtype, 2](
            row_major(_dyn_shape[2](self.m, self.n)), ctx
        )
        var ap = self.a.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
        var dp = dv.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
        var sp = scaled.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
        var n = self.n

        @always_inline
        def scale[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var ap, var dp, var sp, var n}:
            var e = coord_to_index_list(coord)[0]
            sp[unsafe_offset=e] = ap[unsafe_offset=e] * dp[unsafe_offset=e % n]

        elementwise[simd_width=1, target=_target[Self.gpu]()](
            scale, Coord(self.m * self.n), ctx
        )
        ctx.synchronize()
        var product = matmul[gpu=Self.gpu](scaled, self.at)
        _ = dv^
        _ = scaled^
        return _lu_factor_runtime[Self.dtype, Self.gpu](product)


def _sym_solve[
    dtype: DType, gpu: Bool
](
    s: _Standard,
    dinv: List[Float64],
    mut lu: Optional[_LURuntime[dtype, gpu]],
    r1: List[Float64],
    r2: List[Float64],
) raises -> Tuple[List[Float64], List[Float64]] where dtype.is_floating_point():
    """SciPy's `_sym_solve`: `v = M^-1 (r2 + A (Dinv r1))` through the
    device factorization, then `u = Dinv (A^T v - r1)`."""
    var m = s.m
    var n = s.n
    var scaled = List[Float64](capacity=n)
    for j in range(n):
        scaled.append(dinv[j] * r1[j])
    var r = _ax(s.a, m, n, scaled)
    for i in range(m):
        r[i] += r2[i]
    var v = List[Float64]()
    var atv = List[Float64](length=n, fill=0.0)
    if m > 0:
        v = _solve(lu.value(), r)
        atv = _atx(s.a, m, n, v)
    var u = List[Float64](capacity=n)
    for j in range(n):
        u.append(dinv[j] * (atv[j] - r1[j]))
    return (u^, v^)


def _solve[
    dtype: DType, gpu: Bool
](mut lu: _LURuntime[dtype, gpu], r: List[Float64]) raises -> List[
    Float64
] where dtype.is_floating_point():
    """`M^-1 r` through the device factorization, read back."""
    var ctx = lu.factored.context()
    var host = List[Scalar[dtype]](capacity=len(r))
    for i in range(len(r)):
        host.append(Scalar[dtype](r[i]))
    var rhs = asarray(host^, ctx)
    var x = lu.solve(rhs)
    var back = x.to_host[DType.float64]()
    var out = List[Float64](capacity=len(r))
    for i in range(len(back)):
        out.append(Float64(back[i]))
    return out^


# ---------------------------------------------------------------- the solve


def _finite(v: List[Float64]) -> Bool:
    for i in range(len(v)):
        if _isnan(v[i]) or abs(v[i]) == _inf[DType.float64]():
            return False
    return True


def _step(
    x: List[Float64],
    d_x: List[Float64],
    z: List[Float64],
    d_z: List[Float64],
    tau: Float64,
    d_tau: Float64,
    kappa: Float64,
    d_kappa: Float64,
    alpha0: Float64,
) -> Float64:
    """SciPy's `_get_step`: `alpha0` of the longest step keeping `x`, `z`,
    `tau` and `kappa` nonnegative, at most `1`."""
    var alpha = 1.0
    for i in range(len(x)):
        if d_x[i] < 0:
            alpha = min(alpha, alpha0 * x[i] / -d_x[i])
        if d_z[i] < 0:
            alpha = min(alpha, alpha0 * z[i] / -d_z[i])
    if d_tau < 0:
        alpha = min(alpha, alpha0 * tau / -d_tau)
    if d_kappa < 0:
        alpha = min(alpha, alpha0 * kappa / -d_kappa)
    return alpha


struct _Indicators(Copyable, Movable):
    var rho_p: Float64
    var rho_d: Float64
    var rho_a: Float64
    var rho_g: Float64
    var rho_mu: Float64

    def __init__(
        out self,
        rho_p: Float64,
        rho_d: Float64,
        rho_a: Float64,
        rho_g: Float64,
        rho_mu: Float64,
    ):
        self.rho_p = rho_p
        self.rho_d = rho_d
        self.rho_a = rho_a
        self.rho_g = rho_g
        self.rho_mu = rho_mu


def _indicators(
    s: _Standard,
    x: List[Float64],
    y: List[Float64],
    z: List[Float64],
    tau: Float64,
    kappa: Float64,
) -> _Indicators:
    """SciPy's `_indicators`: the residuals relative to the blind start's."""
    var m = s.m
    var n = s.n
    var ones = List[Float64](length=n, fill=1.0)
    var a1 = _ax(s.a, m, n, ones)
    var rp0 = List[Float64](capacity=m)
    for i in range(m):
        rp0.append(s.b[i] - a1[i])
    var rd0 = List[Float64](capacity=n)
    var csum = 0.0
    for j in range(n):
        rd0.append(s.c[j] - 1.0)
        csum += s.c[j]
    var rg0 = abs(1.0 + csum)

    var ax = _ax(s.a, m, n, x)
    var rp = List[Float64](capacity=m)
    for i in range(m):
        rp.append(s.b[i] * tau - ax[i])
    var aty = _atx(s.a, m, n, y)
    var rd = List[Float64](capacity=n)
    for j in range(n):
        rd.append(s.c[j] * tau - aty[j] - z[j])
    var cx = _dot(s.c, x)
    var by = _dot(s.b, y)
    var rg = abs(kappa + cx - by)
    var mu = (_dot(x, z) + tau * kappa) / Float64(n + 1)
    return _Indicators(
        _norm(rp) / max(1.0, _norm(rp0)),
        _norm(rd) / max(1.0, _norm(rd0)),
        abs(cx - by) / (tau + abs(by)),
        rg / max(1.0, rg0),
        mu,
    )


def _ip_hsd[
    dtype: DType, gpu: Bool
](
    s: _Standard, ctx: DeviceContext, tol: Float64, max_iter: Int
) raises -> Tuple[List[Float64], Int, Int] where dtype.is_floating_point():
    """SciPy's `_ip_hsd` with `pc=True`, `ip=False`, `alpha0=0.99995`:
    the standard-form solution `x / tau`, the status and the iteration
    count."""
    comptime alpha0 = 0.99995
    var m = s.m
    var n = s.n
    var x = List[Float64](length=n, fill=1.0)
    var z = List[Float64](length=n, fill=1.0)
    var y = List[Float64](length=m, fill=0.0)
    var tau = 1.0
    var kappa = 1.0
    var normal = _Normal[dtype, gpu](s.a, m, n, ctx)
    var ind = _indicators(s, x, y, z, tau, kappa)
    var go = ind.rho_p > tol or ind.rho_d > tol or ind.rho_a > tol
    var status = 0
    var iteration = 0
    while go:
        iteration += 1
        # _get_delta: one normal matrix, a predictor and a corrector.
        var ax = _ax(s.a, m, n, x)
        var aty = _atx(s.a, m, n, y)
        var r_p = List[Float64](capacity=m)
        for i in range(m):
            r_p.append(s.b[i] * tau - ax[i])
        var r_d = List[Float64](capacity=n)
        for j in range(n):
            r_d.append(s.c[j] * tau - aty[j] - z[j])
        var r_g = _dot(s.c, x) - _dot(s.b, y) + kappa
        var mu = (_dot(x, z) + tau * kappa) / Float64(n + 1)
        var dinv = List[Float64](capacity=n)
        for j in range(n):
            dinv.append(x[j] / z[j])

        var failed = False
        var p = List[Float64]()
        var q = List[Float64]()
        var lu = Optional[_LURuntime[dtype, gpu]]()
        if m > 0:
            lu = normal.factor(dinv)
            if lu.value().singular():
                failed = True

        var d_x = List[Float64](length=n, fill=0.0)
        var d_y = List[Float64](length=m, fill=0.0)
        var d_z = List[Float64](length=n, fill=0.0)
        var d_tau = 0.0
        var d_kappa = 0.0
        if not failed:
            var pq = _sym_solve(s, dinv, lu, s.c, s.b)
            p = pq[0].copy()
            q = pq[1].copy()
            failed = not (_finite(p) and _finite(q))
        var gamma = 0.0
        var corrections = 0
        while not failed and corrections <= 1:
            var eta = 1.0 - gamma
            var rhatd = List[Float64](capacity=n)
            var rhatxs = List[Float64](capacity=n)
            for j in range(n):
                var xs = gamma * mu - x[j] * z[j]
                if corrections == 1:
                    xs -= d_x[j] * d_z[j]
                rhatxs.append(xs)
                rhatd.append(eta * r_d[j] - xs / x[j])
            var rhatp = List[Float64](capacity=m)
            for i in range(m):
                rhatp.append(eta * r_p[i])
            var rhatg = eta * r_g
            var rhattk = gamma * mu - tau * kappa
            if corrections == 1:
                rhattk -= d_tau * d_kappa
            var uv = _sym_solve(s, dinv, lu, rhatd, rhatp)
            var u = uv[0].copy()
            var v = uv[1].copy()
            d_tau = (rhatg + rhattk / tau - (-_dot(s.c, u) + _dot(s.b, v))) / (
                kappa / tau + (-_dot(s.c, p) + _dot(s.b, q))
            )
            for j in range(n):
                d_x[j] = u[j] + p[j] * d_tau
                d_z[j] = (rhatxs[j] - z[j] * d_x[j]) / x[j]
            for i in range(m):
                d_y[i] = v[i] + q[i] * d_tau
            d_kappa = (rhattk - kappa * d_tau) / tau
            if not (_finite(d_x) and _finite(d_z) and _finite([d_tau])):
                failed = True
                break
            var alpha = _step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, 1.0)
            gamma = (1.0 - alpha) ** 2 * min(0.1, 1.0 - alpha)
            corrections += 1
        _ = lu^
        if failed:
            status = 4
            break

        var alpha = _step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, alpha0)
        for j in range(n):
            x[j] += alpha * d_x[j]
            z[j] += alpha * d_z[j]
        for i in range(m):
            y[i] += alpha * d_y[i]
        tau += alpha * d_tau
        kappa += alpha * d_kappa

        ind = _indicators(s, x, y, z, tau, kappa)
        go = ind.rho_p > tol or ind.rho_d > tol or ind.rho_a > tol
        var inf1 = (
            ind.rho_p < tol
            and ind.rho_d < tol
            and ind.rho_g < tol
            and tau < tol * max(1.0, kappa)
        )
        var inf2 = ind.rho_mu < tol and tau < tol * min(1.0, kappa)
        if inf1 or inf2:
            status = 2 if _dot(s.b, y) > tol else 3
            break
        elif go and iteration >= max_iter:
            status = 1
            break
    var x_hat = List[Float64](capacity=n)
    for j in range(n):
        x_hat.append(x[j] / tau)
    return (x_hat^, status, iteration)


def _to_device[
    dtype: DType
](values: List[Float64], ctx: DeviceContext) raises -> Dynamic[dtype, 1]:
    var host = List[Scalar[dtype]](capacity=len(values))
    for i in range(len(values)):
        host.append(Scalar[dtype](values[i]))
    return asarray(host^, ctx)


def _linprog[
    dtype: DType, gpu: Bool
](
    c: List[Float64],
    a_ub: List[Float64],
    b_ub: List[Float64],
    a_eq: List[Float64],
    b_eq: List[Float64],
    bounds: Bounds,
    tol: Float64,
    max_iter: Int,
    ctx: DeviceContext,
) raises -> LinprogResult[dtype] where dtype.is_floating_point():
    var n = len(c)
    var s = _standard_form(c, a_ub, b_ub, a_eq, b_eq, bounds)
    var solved = _ip_hsd[dtype, gpu](s, ctx, tol, max_iter)
    var xs = solved[0].copy()
    var x = List[Float64](capacity=n)
    for j in range(n):
        var v = s.shift[j] + s.sign[j] * xs[j]
        if s.twin[j] >= 0:
            v -= xs[s.twin[j]]
        x.append(v)
    var ax_ub = _ax(a_ub, len(b_ub), n, x)
    var slack = List[Float64](capacity=len(b_ub))
    for i in range(len(b_ub)):
        slack.append(b_ub[i] - ax_ub[i])
    var ax_eq = _ax(a_eq, len(b_eq), n, x)
    var con = List[Float64](capacity=len(b_eq))
    for i in range(len(b_eq)):
        con.append(b_eq[i] - ax_eq[i])
    return LinprogResult[dtype](
        _to_device[dtype](x, ctx),
        _dot(c, x),
        _to_device[dtype](slack, ctx),
        _to_device[dtype](con, ctx),
        solved[1],
        solved[2],
    )


def linprog[
    C: TensorLike,
    AU: TensorLike,
    BU: TensorLike,
    AE: TensorLike,
    BE: TensorLike,
    gpu: Bool = False,
](
    c: C,
    a_ub: AU,
    b_ub: BU,
    a_eq: AE,
    b_eq: BE,
    bounds: Optional[Bounds] = None,
    tol: Optional[Float64] = None,
    max_iter: Int = 1000,
) raises -> LinprogResult[C.dtype] where (
    C.dtype.is_floating_point()
    and C.LayoutType.rank == 1
    and AU.dtype == C.dtype
    and AU.LayoutType.rank == 2
    and BU.dtype == C.dtype
    and BU.LayoutType.rank == 1
    and AE.dtype == C.dtype
    and AE.LayoutType.rank == 2
    and BE.dtype == C.dtype
    and BE.LayoutType.rank == 1
):
    """`min c^T x` over `A_ub x <= b_ub`, `A_eq x = b_eq` and `bounds`.
    `scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
    method="interior-point")`.

    A problem with no inequalities (or no equalities) passes a zero-row
    `A` and a length-0 `b` for that side; the three-tensor overload is
    the inequality-only spelling. The module docstring has the algorithm
    and its ceiling.

    Parameters:
        C: The tensor type of the length-`n` cost vector.
        AU: The tensor type of the `m_ub x n` inequality matrix.
        BU: The tensor type of the length-`m_ub` inequality bounds.
        AE: The tensor type of the `m_eq x n` equality matrix.
        BE: The tensor type of the length-`m_eq` equality right-hand side.
        gpu: Form and factor the normal matrix on `c`'s device; a
            residency mismatch falls back to the host with a notice.

    Args:
        c: The cost vector.
        a_ub: The inequality matrix.
        b_ub: The inequality bounds.
        a_eq: The equality matrix.
        b_eq: The equality right-hand side.
        bounds: The per-variable bounds; `x >= 0` when omitted, SciPy's
            default `(0, None)`.
        tol: The tolerance on the relative residuals; `1e-8` at `float64`
            and `1e-5` otherwise.
        max_iter: The iteration cap.

    Returns:
        A `LinprogResult`: `x`, `fun`, `slack`, `con`, SciPy's `status`
        and `message`, `nit` and `success`.

    Raises:
        On mismatched shapes, bounds with `lb > ub`, or a device failure.
    """
    var n = c.size()
    if n == 0:
        raise Error("linprog: c must have at least one element")
    if a_ub.dim_at(1) != n and a_ub.dim_at(0) != 0:
        raise Error("linprog: A_ub must have ", n, " columns")
    if a_ub.dim_at(0) != b_ub.size():
        raise Error("linprog: A_ub and b_ub disagree on the row count")
    if a_eq.dim_at(1) != n and a_eq.dim_at(0) != 0:
        raise Error("linprog: A_eq must have ", n, " columns")
    if a_eq.dim_at(0) != b_eq.size():
        raise Error("linprog: A_eq and b_eq disagree on the row count")
    var box = bounds.value().copy() if bounds else Bounds(
        0.0, _inf[DType.float64]()
    )
    if (len(box.lb) != 1 and len(box.lb) != n) or (
        len(box.ub) != 1 and len(box.ub) != n
    ):
        raise Error("linprog: bounds must have 1 or ", n, " entries a side")
    var t = tol.value() if tol else (1e-8 if C.dtype == DType.float64 else 1e-5)
    var ctx = c.context()
    var ch = _host(c)
    var auh = _host(a_ub)
    var buh = _host(b_ub)
    var aeh = _host(a_eq)
    var beh = _host(b_eq)
    if _check_device[C, gpu](c):
        return _linprog[C.dtype, gpu](
            ch, auh, buh, aeh, beh, box, t, max_iter, ctx
        )
    _notice[gpu]("linprog")
    return _linprog[C.dtype, False](
        ch, auh, buh, aeh, beh, box, t, max_iter, ctx
    )


def linprog[
    C: TensorLike,
    AU: TensorLike,
    BU: TensorLike,
    gpu: Bool = False,
](
    c: C,
    a_ub: AU,
    b_ub: BU,
    bounds: Optional[Bounds] = None,
    tol: Optional[Float64] = None,
    max_iter: Int = 1000,
) raises -> LinprogResult[C.dtype] where (
    C.dtype.is_floating_point()
    and C.LayoutType.rank == 1
    and AU.dtype == C.dtype
    and AU.LayoutType.rank == 2
    and BU.dtype == C.dtype
    and BU.LayoutType.rank == 1
):
    """`min c^T x` over `A_ub x <= b_ub` and `bounds`, no equalities.
    `scipy.optimize.linprog(c, A_ub, b_ub, bounds=bounds)`.

    The five-tensor overload with a zero-row `A_eq`.

    Parameters:
        C: The tensor type of the length-`n` cost vector.
        AU: The tensor type of the `m_ub x n` inequality matrix.
        BU: The tensor type of the length-`m_ub` inequality bounds.
        gpu: Form and factor the normal matrix on `c`'s device.

    Args:
        c: The cost vector.
        a_ub: The inequality matrix.
        b_ub: The inequality bounds.
        bounds: The per-variable bounds; `x >= 0` when omitted.
        tol: The tolerance on the relative residuals.
        max_iter: The iteration cap.

    Returns:
        A `LinprogResult`, with an empty `con`.

    Raises:
        On mismatched shapes, bounds with `lb > ub`, or a device failure.
    """
    var ctx = c.context()
    var a_eq = Dynamic[C.dtype, 2](row_major(_dyn_shape[2](0, c.size())), ctx)
    var b_eq = Dynamic[C.dtype, 1](row_major(_dyn_shape[1](0)), ctx)
    return linprog[gpu=gpu](c, a_ub, b_ub, a_eq, b_eq, bounds, tol, max_iter)
