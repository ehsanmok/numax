"""Unconstrained minimization over `numax.core.array.Tensor`.

**This module is tier 2.** It iterates to a tolerance and its driver loop
runs on the host, so nothing here is launchable inside a kernel body.
`numax.optimize.array` is the `FloatLike` tier; this one is `Plain`-only and
exists for objectives whose argument vector is too large to sit in
registers.

## Why the gradient is an argument here and not there

`numax.optimize.array.minimize` takes no Jacobian and no `jac`: it evaluates
the objective at `Gradient[_P, n_vars]`, and one call returns the value
together with every partial derivative, exactly, by the chain rule.

That is structurally unavailable at this tier. A `Tensor` owns a
`DeviceBuffer[dtype]` and `DType` is MAX's closed enum of machine scalars,
so no conformer -- `Gradient` included -- can live in one. The choice is
therefore between a caller-supplied gradient and a finite difference, and
this tier takes the gradient: a forward difference would cost `n_vars + 1`
objective evaluations and cap accuracy near `sqrt(eps)`, which is a silent
accuracy regression against the sibling function of the same name.
Supplying `jac` explicitly keeps the two tiers honest about being the same
algorithm on different data. `numax.optimize.least_squares` takes its
Jacobian as an argument for the identical reason.

## Where the work is, per method

The methods differ in what they store, and at this tier that is the whole
decision:

- **`"bfgs"`** keeps an `n_vars x n_vars` inverse Hessian on the device and
  updates it with MAX's kernels -- one `matvec` for `H y`, three `outer`
  products for the rank-two correction, and elementwise combines. So the
  `O(n^2)` arithmetic per iteration is MAX's and never crosses to the host,
  but the `O(n^2)` *memory* is real and is what eventually bounds the size
  of problem this method fits.
- **`"l-bfgs"`** keeps the last `memory` (default ten) pairs `(s, y)` and
  applies the inverse Hessian implicitly by the two-loop recursion, `O(m
  n)` work and memory per iteration and no matrix anywhere. Limited memory
  is the shape a `Tensor` tier is for: it is what SciPy reaches for at
  size, and the method to use when `n_vars` makes `"bfgs"`'s matrix the
  problem.
- **`"cg"`** stores one direction vector. Every operation in its driver is
  `O(n_vars)`, so its per-iteration cost is dominated by the caller's own
  `f` and `jac` -- which are device tensors and where the work belongs.
- **`"powell"`** is derivative-free: Powell's direction-set method, a
  one-dimensional Brent minimization along each of `n_vars` directions per
  iteration and a direction replaced by the net displacement. It is the
  method for an objective with no usable gradient -- noisy, a simulation, a
  black box -- and takes no `jac`; `f` alone is evaluated, `O(n_vars)`
  line minimizations of a few evaluations each per iteration, on device
  tensors the driver stages.

Either way the driver's bookkeeping is a host-side walk over vectors of
length `n_vars`, so this tier pays for itself when evaluating `f` and `jac`
is the expense -- which is the shape a `Tensor`-sized problem has -- and not
when it is not. For a handful of variables the `Array` tier is both faster
and differentiable.

## Bounds

The overload taking `lower` and `upper` minimizes over the box `lower <= x
<= upper` with the same four methods, by projection: the direction has its
outward components at an active bound zeroed, every trial point is clamped
into the box, and convergence is on the *projected* gradient -- the
gradient with those same components removed, which is what is zero at a
constrained minimum. With bounds the gradient methods all backtrack on the
Armijo condition along the projected path rather than running the Wolfe
search, since a curvature condition along a kinked path is not the right
test. This is projected gradient descent with a quasi-Newton or conjugate
direction, which converges but is not L-BFGS-B: that method's generalized
Cauchy point and subspace minimization are the upgrade, and the reason
SciPy's `L-BFGS-B` takes fewer iterations on a problem with many active
bounds.

`"nelder-mead"` is deliberately absent. Its simplex is `n_vars + 1` points
of `n_vars` entries each, and comparing function values across all of them
every iteration is exactly the shape a `Tensor` tier exists to not be. It
stays in `numax.optimize.array`. General constraints are out of scope.
"""

from max.gpu.host import DeviceContext

from ..core.array import Static, eye
from ..core.ops import add, multiply
from ..linalg.blas import matvec, outer

from .common import _as_tensor


struct TensorMinimizeResult[dtype: DType, n_vars: Int](Movable):
    """What a `Tensor`-tier minimization returns.

    `x` is the argument vector, `f_x` the objective at it, and `grad_norm`
    the infinity-norm of the gradient there -- the quantity the convergence
    test actually looks at, reported so a caller can see how close a
    non-converged run got.

    The field names are `MinimizeResult`'s, so the two tiers read the same.
    `TensorFitResult` next door says `cost` rather than `f_x` because a
    least-squares cost is `sum(r**2) / 2` and genuinely a different
    quantity; an objective value is not.

    Movable but not `Copyable`: `x` owns a `DeviceBuffer`.
    """

    var x: Static[Self.dtype, Self.n_vars]
    var f_x: Float64
    var grad_norm: Float64
    var iterations: Int
    var converged: Bool

    def __init__(
        out self,
        var x: Static[Self.dtype, Self.n_vars],
        f_x: Float64,
        grad_norm: Float64,
        iterations: Int,
        converged: Bool,
    ):
        """Field-wise, spelled out because a `Movable`-only struct gets no
        generated one: `x` owns a `DeviceBuffer` and so cannot be copied."""
        self.x = x^
        self.f_x = f_x
        self.grad_norm = grad_norm
        self.iterations = iterations
        self.converged = converged


def _to_list[dtype: DType, n: Int](t: Static[dtype, n]) raises -> List[Float64]:
    """A device vector read back as host `Float64`s.

    The drivers keep their vectors on the host; this is the one direction
    that crosses. It is `O(n)` and happens twice per iteration, against the
    `O(n^2)` the device does for `"bfgs"` and the caller's own evaluations
    for either method.
    """
    var host = t.to_host()
    var out = List[Float64](capacity=n)
    for i in range(n):
        out.append(Float64(host[i]))
    return out^


def _infinity_norm(values: List[Float64], n: Int) -> Float64:
    var largest = 0.0
    for i in range(n):
        largest = max(largest, abs(values[i]))
    return largest


def _slope_along[
    dtype: DType,
    n: Int,
    f: def(Static[dtype, n], DeviceContext) raises thin -> Scalar[dtype],
    jac: def(Static[dtype, n], DeviceContext) raises thin -> Static[dtype, n],
](
    x: List[Float64],
    direction: List[Float64],
    alpha: Float64,
    mut value: Float64,
    ctx: DeviceContext,
) raises -> Float64:
    """`phi'(alpha)` for `phi(alpha) = f(x + alpha * direction)`, with
    `phi(alpha)` left in `value`. Two of the caller's evaluations, `f` and
    `jac`, at one trial point."""
    var trial = List[Float64](length=n, fill=0.0)
    for i in range(n):
        trial[i] = x[i] + alpha * direction[i]
    var point = _as_tensor[dtype, n](trial, ctx)
    value = Float64(f(point, ctx))
    var grad = _to_list[dtype, n](jac(point, ctx))
    var slope = 0.0
    for i in range(n):
        slope += grad[i] * direction[i]
    return slope


def _wolfe_step[
    dtype: DType,
    n: Int,
    f: def(Static[dtype, n], DeviceContext) raises thin -> Scalar[dtype],
    jac: def(Static[dtype, n], DeviceContext) raises thin -> Static[dtype, n],
](
    x: List[Float64],
    direction: List[Float64],
    f_x: Float64,
    directional: Float64,
    first_guess: Float64,
    ctx: DeviceContext,
) raises -> Float64:
    """A step length satisfying the **strong Wolfe** conditions, or `0` if
    none was found. Nocedal & Wright algorithms 3.5 and 3.6, and the same
    search `numax.optimize.array`'s `cg` uses -- its docstring carries the
    full reasoning.

    The short version, because it is the difference between a CG that
    converges and one that does not: a backtracking search on the Armijo
    condition alone can only ever *shorten* its trial step. BFGS does not
    care, since `H` supplies the scale of a good step; CG has no curvature
    model to lengthen one for free, so in a curved valley it stalls with the
    gradient still large. The curvature condition (`c2 = 0.1`) is what fixes
    that, and it costs a `jac` evaluation per trial -- which is why only
    `"cg"` pays for it here.
    """
    comptime c1 = 1e-4
    comptime c2 = 0.1

    var lo = 0.0
    var lo_value = f_x
    var hi = 0.0
    var bracketed = False

    var previous = 0.0
    var previous_value = f_x
    var alpha = first_guess

    # Phase 1: bracket.
    for attempt in range(40):
        var value = f_x
        var slope = _slope_along[dtype, n, f, jac](
            x, direction, alpha, value, ctx
        )
        if value > f_x + c1 * alpha * directional or (
            attempt > 0 and value >= previous_value
        ):
            lo = previous
            lo_value = previous_value
            hi = alpha
            bracketed = True
            break
        if abs(slope) <= -c2 * directional:
            return alpha
        if slope >= 0:
            lo = alpha
            lo_value = value
            hi = previous
            bracketed = True
            break
        previous = alpha
        previous_value = value
        alpha = alpha * 2

    if not bracketed:
        return 0

    # Phase 2: zoom, by bisection -- it cannot leave the bracket, and the
    # extra iterations are cheap next to an evaluation of `f`.
    for _ in range(60):
        var mid = (lo + hi) / 2
        if mid <= 0:
            return 0
        var value = f_x
        var slope = _slope_along[dtype, n, f, jac](
            x, direction, mid, value, ctx
        )
        if value > f_x + c1 * mid * directional or value >= lo_value:
            hi = mid
        else:
            if abs(slope) <= -c2 * directional:
                return mid
            if slope * (hi - lo) >= 0:
                hi = lo
            lo = mid
            lo_value = value

    # The bracket collapsed without meeting the curvature condition; `lo` is
    # still Armijo-acceptable wherever it is not the origin.
    return lo


def _bfgs_update[
    dtype: DType, n: Int, gpu: Bool
](
    var h: Static[dtype, n, n],
    s_host: List[Float64],
    y_host: List[Float64],
    sy: Float64,
    ctx: DeviceContext,
) raises -> Static[dtype, n, n] where dtype.is_floating_point():
    """One BFGS rank-two correction, on the device.

    `H <- (I - s y^T / sy) H (I - y s^T / sy) + s s^T / sy`, expanded so
    that no `n x n` product is needed -- only `H y`, which is a `matvec`,
    and three outer products:

    `H <- H - (s (Hy)^T + (Hy) s^T) / sy + (1 + y^T H y / sy) s s^T / sy`

    That expansion is what keeps this `O(n^2)` rather than `O(n^3)`. Every
    term is a MAX kernel and the matrix never leaves the device; only the
    two scalars `sy` and `y^T H y` come back, and the second of those is a
    dot product of two host vectors rather than a launch of its own.
    """
    var s = _as_tensor[dtype, n](s_host, ctx)
    var y = _as_tensor[dtype, n](y_host, ctx)

    var hy = matvec[dtype, n, n, gpu](h, y)
    var hy_host = _to_list[dtype, n](hy)

    var yhy = 0.0
    for i in range(n):
        yhy += y_host[i] * hy_host[i]

    var s_hy = outer[dtype, n, n, gpu](s, hy)
    var hy_s = outer[dtype, n, n, gpu](hy, s)

    # `s s^T` needs a second binding for the same vector: `outer` takes both
    # operands mutably, because `.view()` cannot build a writable
    # `TileTensor` from an immutable one, and Mojo rejects passing one
    # binding through two `mut` arguments. A second materialization of the
    # same host vector is the cost, and it is `O(n)` against the `O(n^2)`
    # product it feeds.
    var s_again = _as_tensor[dtype, n](s_host, ctx)
    var s_s = outer[dtype, n, n, gpu](s, s_again)

    var correction = multiply(add(s_hy, hy_s), Scalar[dtype](-1.0 / sy))
    var rank_one = multiply(s_s, Scalar[dtype]((1 + yhy / sy) / sy))
    return add(add(h, correction), rank_one)


def _no_jac[
    dtype: DType, n: Int
](x: Static[dtype, n], ctx: DeviceContext) raises -> Static[dtype, n]:
    """The `jac` a caller who chose `"powell"` does not have to write; any
    other method raises here rather than silently differencing."""
    raise Error(
        "minimize: this method needs a gradient; pass `jac`, or choose"
        " method='powell', which is derivative-free"
    )


def _clamp(
    mut x: List[Float64], lower: List[Float64], upper: List[Float64], n: Int
):
    for i in range(n):
        x[i] = min(max(x[i], lower[i]), upper[i])


def _projected_gradient_norm(
    x: List[Float64],
    grad: List[Float64],
    lower: List[Float64],
    upper: List[Float64],
    n: Int,
    bounded: Bool,
) -> Float64:
    """The infinity norm of the gradient with the components that push
    against an active bound removed -- zero exactly at a constrained
    minimum."""
    var largest = 0.0
    for i in range(n):
        var g = grad[i]
        if bounded:
            if (x[i] <= lower[i] and g > 0) or (x[i] >= upper[i] and g < 0):
                g = 0
        largest = max(largest, abs(g))
    return largest


def _lbfgs_direction(
    grad: List[Float64],
    s_hist: List[List[Float64]],
    y_hist: List[List[Float64]],
    rho: List[Float64],
    n: Int,
) -> List[Float64]:
    """`-H g` by the two-loop recursion over the stored pairs, with `H0 =
    (s^T y / y^T y) I` from the most recent pair (Nocedal & Wright
    algorithm 7.4)."""
    var q = List[Float64](capacity=n)
    for i in range(n):
        q.append(grad[i])
    var m = len(s_hist)
    var alphas = List[Float64](length=m, fill=0.0)
    var k = m - 1
    while k >= 0:
        var dot = 0.0
        for i in range(n):
            dot += s_hist[k][i] * q[i]
        alphas[k] = rho[k] * dot
        for i in range(n):
            q[i] -= alphas[k] * y_hist[k][i]
        k -= 1
    var gamma = 1.0
    if m > 0:
        var sy = 0.0
        var yy = 0.0
        for i in range(n):
            sy += s_hist[m - 1][i] * y_hist[m - 1][i]
            yy += y_hist[m - 1][i] * y_hist[m - 1][i]
        if yy > 0:
            gamma = sy / yy
    for i in range(n):
        q[i] *= gamma
    for k2 in range(m):
        var dot = 0.0
        for i in range(n):
            dot += y_hist[k2][i] * q[i]
        var beta = rho[k2] * dot
        for i in range(n):
            q[i] += s_hist[k2][i] * (alphas[k2] - beta)
    for i in range(n):
        q[i] = -q[i]
    return q^


def _line_value[
    dtype: DType,
    n: Int,
    f: def(Static[dtype, n], DeviceContext) raises thin -> Scalar[dtype],
](
    x: List[Float64],
    direction: List[Float64],
    alpha: Float64,
    ctx: DeviceContext,
) raises -> Float64:
    var trial = List[Float64](length=n, fill=0.0)
    for i in range(n):
        trial[i] = x[i] + alpha * direction[i]
    var point = _as_tensor[dtype, n](trial, ctx)
    return Float64(f(point, ctx))


def _brent_line[
    dtype: DType,
    n: Int,
    f: def(Static[dtype, n], DeviceContext) raises thin -> Scalar[dtype],
](
    x: List[Float64],
    direction: List[Float64],
    lo: Float64,
    hi: Float64,
    ctx: DeviceContext,
) raises -> Tuple[Float64, Float64]:
    """Brent's one-dimensional minimization of `alpha -> f(x + alpha d)`
    on `[lo, hi]`, golden section with parabolic steps (Numerical Recipes
    `brent`), to `sqrt(eps)` in `alpha`. Returns `(alpha, f)`."""
    comptime golden = 0.3819660112501051
    comptime rel = 1.4901161193847656e-08
    var a = lo
    var b = hi
    var v = a + golden * (b - a)
    var w = v
    var xm = v
    var fv = _line_value[dtype, n, f](x, direction, v, ctx)
    var fw = fv
    var fx = fv
    var d = 0.0
    var e = 0.0
    for _ in range(60):
        var mid = 0.5 * (a + b)
        var tol1 = rel * abs(xm) + 1e-12
        var tol2 = 2 * tol1
        if abs(xm - mid) <= tol2 - 0.5 * (b - a):
            break
        var parabolic = False
        if abs(e) > tol1:
            var r = (xm - w) * (fx - fv)
            var q = (xm - v) * (fx - fw)
            var pp = (xm - v) * q - (xm - w) * r
            q = 2 * (q - r)
            if q > 0:
                pp = -pp
            q = abs(q)
            var etemp = e
            e = d
            if not (
                abs(pp) >= abs(0.5 * q * etemp)
                or pp <= q * (a - xm)
                or pp >= q * (b - xm)
            ):
                d = pp / q
                var u = xm + d
                if u - a < tol2 or b - u < tol2:
                    d = tol1 if mid - xm >= 0 else -tol1
                parabolic = True
        if not parabolic:
            e = (a if xm >= mid else b) - xm
            d = golden * e
        var u = xm + d if abs(d) >= tol1 else xm + (tol1 if d >= 0 else -tol1)
        var fu = _line_value[dtype, n, f](x, direction, u, ctx)
        if fu <= fx:
            if u >= xm:
                a = xm
            else:
                b = xm
            v = w
            fv = fw
            w = xm
            fw = fx
            xm = u
            fx = fu
        else:
            if u < xm:
                a = u
            else:
                b = u
            if fu <= fw or w == xm:
                v = w
                fv = fw
                w = u
                fw = fu
            elif fu <= fv or v == xm or v == w:
                v = u
                fv = fu
    return (xm, fx)


def _feasible_interval(
    x: List[Float64],
    direction: List[Float64],
    lower: List[Float64],
    upper: List[Float64],
    n: Int,
    bounded: Bool,
    reach: Float64,
) -> Tuple[Float64, Float64]:
    """The `alpha` range on which `x + alpha d` stays in the box, capped
    at `[-reach, reach]`."""
    var lo = -reach
    var hi = reach
    if bounded:
        for i in range(n):
            if direction[i] > 0:
                hi = min(hi, (upper[i] - x[i]) / direction[i])
                lo = max(lo, (lower[i] - x[i]) / direction[i])
            elif direction[i] < 0:
                hi = min(hi, (lower[i] - x[i]) / direction[i])
                lo = max(lo, (upper[i] - x[i]) / direction[i])
    return (lo, hi)


def _bracket_line[
    dtype: DType,
    n: Int,
    f: def(Static[dtype, n], DeviceContext) raises thin -> Scalar[dtype],
](
    x: List[Float64],
    direction: List[Float64],
    f_x: Float64,
    lo_bound: Float64,
    hi_bound: Float64,
    ctx: DeviceContext,
) raises -> Tuple[Float64, Float64]:
    """A bracket `[a, c]` around the *nearest* minimum of `alpha -> f(x +
    alpha d)` from `alpha = 0`, by golden-ratio expansion outward
    (Numerical Recipes `mnbrak`), clipped to `[lo_bound, hi_bound]`. The
    nearest one, rather than the lowest on a fixed interval, is what keeps
    Powell descending into the valley it is in."""
    comptime gold = 1.618033988749895
    var a = 0.0
    var fa = f_x
    var b = min(1.0, hi_bound)
    if b <= 0:
        b = max(-1.0, lo_bound)
    var fb = _line_value[dtype, n, f](x, direction, b, ctx)
    if fb > fa:
        # Downhill is the other way.
        var t = a
        a = b
        b = t
        var ft = fa
        fa = fb
        fb = ft
    var c = b + gold * (b - a)
    c = min(max(c, lo_bound), hi_bound)
    var fc = _line_value[dtype, n, f](x, direction, c, ctx)
    for _ in range(60):
        if fb <= fc or c == b:
            break
        a = b
        fa = fb
        b = c
        fb = fc
        var next = b + gold * (b - a)
        next = min(max(next, lo_bound), hi_bound)
        if next == c:
            break
        c = next
        fc = _line_value[dtype, n, f](x, direction, c, ctx)
    return (min(a, c), max(a, c))


def _powell[
    dtype: DType,
    n_vars: Int,
    f: def(Static[dtype, n_vars], DeviceContext) raises thin -> Scalar[dtype],
](
    x0: Static[dtype, n_vars],
    tol: Float64,
    max_iter: Int,
    lower: List[Float64],
    upper: List[Float64],
    bounded: Bool,
) raises -> TensorMinimizeResult[dtype, n_vars] where dtype.is_floating_point():
    """Powell's direction-set method (Numerical Recipes `powell`): a Brent
    line minimization along each direction, then the direction of largest
    decrease replaced by the net displacement, with the extrapolation test
    that keeps the set from going degenerate. Stops when an iteration's
    relative decrease in `f` falls below `tol`; `grad_norm` reports that
    decrease, there being no gradient."""
    var ctx = x0.context()
    var x = _to_list[dtype, n_vars](x0)
    if bounded:
        _clamp(x, lower, upper, n_vars)
    var f_x = Float64(f(_as_tensor[dtype, n_vars](x, ctx), ctx))
    var directions = List[List[Float64]](capacity=n_vars)
    for i in range(n_vars):
        var unit = List[Float64](length=n_vars, fill=0.0)
        unit[i] = 1.0
        directions.append(unit^)
    var last_decrease = 0.0
    for iteration in range(max_iter):
        var start = x.copy()
        var f_start = f_x
        var biggest = 0.0
        var biggest_index = 0
        for i in range(n_vars):
            var before = f_x
            var span = _feasible_interval(
                x, directions[i], lower, upper, n_vars, bounded, 1e6
            )
            var bracket = _bracket_line[dtype, n_vars, f](
                x, directions[i], f_x, span[0], span[1], ctx
            )
            var found = _brent_line[dtype, n_vars, f](
                x, directions[i], bracket[0], bracket[1], ctx
            )
            for k in range(n_vars):
                x[k] += found[0] * directions[i][k]
            if bounded:
                _clamp(x, lower, upper, n_vars)
            f_x = found[1]
            if before - f_x > biggest:
                biggest = before - f_x
                biggest_index = i
        last_decrease = (
            2 * abs(f_start - f_x) / (abs(f_start) + abs(f_x) + 1e-300)
        )
        if last_decrease <= tol:
            return TensorMinimizeResult[dtype, n_vars](
                _as_tensor[dtype, n_vars](x, ctx),
                f_x,
                last_decrease,
                iteration + 1,
                True,
            )
        # The extrapolated point and Powell's replacement test.
        var displacement = List[Float64](capacity=n_vars)
        var extrapolated = List[Float64](capacity=n_vars)
        for k in range(n_vars):
            displacement.append(x[k] - start[k])
            extrapolated.append(2 * x[k] - start[k])
        if bounded:
            _clamp(extrapolated, lower, upper, n_vars)
        var f_extra = Float64(
            f(_as_tensor[dtype, n_vars](extrapolated, ctx), ctx)
        )
        if f_extra < f_start:
            var t = (
                2
                * (f_start - 2 * f_x + f_extra)
                * (f_start - f_x - biggest) ** 2
                - biggest * (f_start - f_extra) ** 2
            )
            if t < 0:
                var span = _feasible_interval(
                    x, displacement, lower, upper, n_vars, bounded, 1e6
                )
                var bracket = _bracket_line[dtype, n_vars, f](
                    x, displacement, f_x, span[0], span[1], ctx
                )
                var found = _brent_line[dtype, n_vars, f](
                    x, displacement, bracket[0], bracket[1], ctx
                )
                for k in range(n_vars):
                    x[k] += found[0] * displacement[k]
                if bounded:
                    _clamp(x, lower, upper, n_vars)
                f_x = found[1]
                directions[biggest_index] = directions[n_vars - 1].copy()
                directions[n_vars - 1] = displacement^
    return TensorMinimizeResult[dtype, n_vars](
        _as_tensor[dtype, n_vars](x, ctx), f_x, last_decrease, max_iter, False
    )


def minimize[
    dtype: DType,
    n_vars: Int,
    f: def(Static[dtype, n_vars], DeviceContext) raises thin -> Scalar[dtype],
    jac: def(Static[dtype, n_vars], DeviceContext) raises thin -> Static[
        dtype, n_vars
    ] = _no_jac[dtype, n_vars],
    method: StaticString = "bfgs",
    gpu: Bool = False,
    memory: Int = 10,
](
    x0: Static[dtype, n_vars],
    tol: Optional[Float64] = None,
    max_iter: Optional[Int] = None,
) raises -> TensorMinimizeResult[dtype, n_vars] where dtype.is_floating_point():
    """Minimize `f` from `x0`. `scipy.optimize.minimize` over `Tensor`.

    | `method` | Stores | Reach for it when |
    | --- | --- | --- |
    | `"bfgs"` (default) | an `n x n` inverse Hessian, on the device | the curvature is worth `O(n^2)` memory |
    | `"l-bfgs"` | the last `memory` `(s, y)` pairs | `n_vars` makes that matrix the problem and curvature still matters |
    | `"cg"` | one direction vector | the same size, with a strong-Wolfe search instead of curvature pairs |
    | `"powell"` | `n_vars` directions | there is no usable gradient; `jac` is not needed |

    `f` returns the objective at a point and `jac` its gradient there. Both
    are compile-time parameters, so neither can capture run-time data -- the
    module docstring says why the gradient is an argument at this tier when
    the `Array` tier needs none. `"powell"` is the one method that does not
    read `jac`, and `jac` may be left at its default for it; the default
    raises if any other method is chosen without one.

    `tol` and `max_iter` default per method, matching the `Array` tier's
    `minimize`; both methods here test `max|grad| < 1e-8`, and the defaults
    are stated per method anyway so that adding one with a different
    stopping rule cannot silently change these.

    `x0` is borrowed rather than consumed, and deliberately: it is what
    keeps the `DeviceContext` this minimization allocates on alive for the
    whole loop. See `numax.optimize.common`.

    A run that exhausts `max_iter`, or whose line search cannot find a
    downhill step, returns `converged=False` with the best point it reached
    rather than raising. See `minimize` in `numax.optimize.array` for why an
    unrecognized `method` raises rather than failing to compile.
    """
    var empty = List[Float64]()
    comptime if method == "bfgs" or method == "cg" or method == "l-bfgs":
        return _descend[dtype, n_vars, f, jac, method, gpu, memory](
            x0,
            tol.value() if tol else 1e-8,
            max_iter.value() if max_iter else 200,
            empty,
            empty,
            False,
        )
    elif method == "powell":
        return _powell[dtype, n_vars, f](
            x0,
            tol.value() if tol else 1e-10,
            max_iter.value() if max_iter else 200,
            empty,
            empty,
            False,
        )
    else:
        raise Error(
            "minimize: unknown method '",
            method,
            (
                "'; expected 'bfgs', 'l-bfgs', 'cg' or 'powell'. 'nelder-mead'"
                " is Array-tier only -- see the module docstring."
            ),
        )


def minimize[
    dtype: DType,
    n_vars: Int,
    f: def(Static[dtype, n_vars], DeviceContext) raises thin -> Scalar[dtype],
    jac: def(Static[dtype, n_vars], DeviceContext) raises thin -> Static[
        dtype, n_vars
    ] = _no_jac[dtype, n_vars],
    method: StaticString = "l-bfgs",
    gpu: Bool = False,
    memory: Int = 10,
](
    x0: Static[dtype, n_vars],
    lower: Static[dtype, n_vars],
    upper: Static[dtype, n_vars],
    tol: Optional[Float64] = None,
    max_iter: Optional[Int] = None,
) raises -> TensorMinimizeResult[dtype, n_vars] where dtype.is_floating_point():
    """Minimize `f` over the box `lower <= x <= upper`.
    `scipy.optimize.minimize(..., bounds=...)` over `Tensor`, by
    projection; the module docstring's *Bounds* section has the method and
    its ceiling against L-BFGS-B. The same four methods, defaulting to
    `"l-bfgs"` here as SciPy does with bounds; `tol` is on the projected
    gradient (or, for `"powell"`, the relative decrease).
    """
    var lo = _to_list[dtype, n_vars](lower)
    var hi = _to_list[dtype, n_vars](upper)
    comptime if method == "bfgs" or method == "cg" or method == "l-bfgs":
        return _descend[dtype, n_vars, f, jac, method, gpu, memory](
            x0,
            tol.value() if tol else 1e-8,
            max_iter.value() if max_iter else 200,
            lo,
            hi,
            True,
        )
    elif method == "powell":
        return _powell[dtype, n_vars, f](
            x0,
            tol.value() if tol else 1e-10,
            max_iter.value() if max_iter else 200,
            lo,
            hi,
            True,
        )
    else:
        raise Error(
            "minimize: unknown method '",
            method,
            "'; expected 'bfgs', 'l-bfgs', 'cg' or 'powell'.",
        )


def _descend[
    dtype: DType,
    n_vars: Int,
    f: def(Static[dtype, n_vars], DeviceContext) raises thin -> Scalar[dtype],
    jac: def(Static[dtype, n_vars], DeviceContext) raises thin -> Static[
        dtype, n_vars
    ],
    method: StaticString,
    gpu: Bool,
    memory: Int,
](
    x0: Static[dtype, n_vars],
    tol: Float64,
    max_iter: Int,
    lower: List[Float64],
    upper: List[Float64],
    bounded: Bool,
) raises -> TensorMinimizeResult[dtype, n_vars] where dtype.is_floating_point():
    """The shared driver. The gradient methods take the same steps --
    evaluate, test, choose a direction, back-track, update -- and differ
    only in how the direction is produced, which is the `comptime if`
    below. With `bounded`, the direction is projected, every trial point is
    clamped, and the test is on the projected gradient.

    One loop rather than three because the line search, the convergence
    test, the failure reporting and the `List`/`Tensor` staging are
    identical and were the bulk of each.
    """
    var ctx = x0.context()

    var x = _to_list[dtype, n_vars](x0)
    if bounded:
        _clamp(x, lower, upper, n_vars)
    var point = _as_tensor[dtype, n_vars](x, ctx)
    var f_x = Float64(f(point, ctx))
    var grad = _to_list[dtype, n_vars](jac(point, ctx))
    var grad_norm = _projected_gradient_norm(
        x, grad, lower, upper, n_vars, bounded
    )

    # `"l-bfgs"`'s memory: the last `memory` pairs and their `1 / s^T y`.
    var s_hist = List[List[Float64]]()
    var y_hist = List[List[Float64]]()
    var rho = List[Float64]()

    # `"bfgs"`'s inverse Hessian, starting at the identity so the first step
    # is steepest descent. `"cg"` never touches it, and a 1x1 placeholder
    # would still have to be a `Tensor`, so it is allocated either way and
    # the comment is the honest accounting: `"cg"` pays one `n x n`
    # allocation it does not use.
    var h = eye[n_vars, dtype](ctx)

    # `"cg"`'s carried direction, and the previous step and slope that set
    # the next trial length.
    var direction = List[Float64](capacity=n_vars)
    for i in range(n_vars):
        direction.append(-grad[i])
    var previous_step = 0.0
    var previous_directional = 0.0

    for iteration in range(max_iter):
        if grad_norm < tol:
            return TensorMinimizeResult[dtype, n_vars](
                _as_tensor[dtype, n_vars](x, ctx),
                f_x,
                grad_norm,
                iteration,
                True,
            )

        # With bounds the quasi-Newton model lives in the free subspace: the
        # gradient it sees and the pairs it stores have the components at an
        # active bound zeroed, so a blocked coordinate cannot feed curvature
        # into a direction that then has to be projected away.
        var reduced = List[Float64](capacity=n_vars)
        for i in range(n_vars):
            var g = grad[i]
            if bounded and (
                (x[i] <= lower[i] and g > 0) or (x[i] >= upper[i] and g < 0)
            ):
                g = 0
            reduced.append(g)

        comptime if method == "bfgs":
            # p = -H g, on the device.
            var g = _as_tensor[dtype, n_vars](reduced, ctx)
            var hg = _to_list[dtype, n_vars](
                matvec[dtype, n_vars, n_vars, gpu](h, g)
            )
            for i in range(n_vars):
                direction[i] = -hg[i]
        elif method == "l-bfgs":
            direction = _lbfgs_direction(reduced, s_hist, y_hist, rho, n_vars)

        # At an active bound, a component pointing out of the box is
        # dropped: the projected direction.
        if bounded:
            for i in range(n_vars):
                if (x[i] <= lower[i] and direction[i] < 0) or (
                    x[i] >= upper[i] and direction[i] > 0
                ):
                    direction[i] = 0

        var directional = 0.0
        for i in range(n_vars):
            directional += grad[i] * direction[i]

        # Rounding can leave a carried direction pointing uphill; steepest
        # descent cannot, unless the gradient is zero, which the tolerance
        # test above already caught.
        if directional >= 0:
            directional = 0
            for i in range(n_vars):
                direction[i] = -grad[i]
                if bounded and (
                    (x[i] <= lower[i] and direction[i] < 0)
                    or (x[i] >= upper[i] and direction[i] > 0)
                ):
                    direction[i] = 0
                directional += grad[i] * direction[i]
            if directional >= 0:
                # Nothing to move: every descent component is blocked.
                return TensorMinimizeResult[dtype, n_vars](
                    _as_tensor[dtype, n_vars](x, ctx),
                    f_x,
                    grad_norm,
                    iteration,
                    True,
                )

        # BFGS's `H` carries the scale, so a unit step is the right first
        # trial. CG carries none, so the trial length comes from the
        # previous step rescaled by the ratio of the slopes -- Nocedal &
        # Wright (3.60) -- with a first iteration that moves `x` by about
        # one unit.
        var step: Float64
        comptime if method == "bfgs" or method == "l-bfgs":
            step = 1
        else:
            if previous_step > 0 and directional < 0:
                step = previous_step * previous_directional / directional
            else:
                var longest = 0.0
                for i in range(n_vars):
                    longest = max(longest, abs(direction[i]))
                step = 1 / longest if longest > 1 else 1

        var candidate = List[Float64](length=n_vars, fill=0.0)
        var accepted = False
        var f_candidate = f_x

        # CG and L-BFGS need the curvature condition: it is what guarantees
        # `s^T y > 0` so the pair can be stored, and what lengthens a step
        # the model made too short -- Armijo alone accepts a unit step along
        # a tiny direction and crawls. See `_wolfe_step`.
        var use_wolfe = (method == "cg" or method == "l-bfgs") and not bounded
        if use_wolfe:
            step = _wolfe_step[dtype, n_vars, f, jac](
                x, direction, f_x, directional, step, ctx
            )
            if step > 0:
                accepted = True
                for i in range(n_vars):
                    candidate[i] = x[i] + step * direction[i]
                var moved_trial = _as_tensor[dtype, n_vars](candidate, ctx)
                f_candidate = Float64(f(moved_trial, ctx))
        else:
            # Backtrack on Armijo alone: BFGS's `H` carries the scale of a
            # good step, so a unit trial that is only ever halved is enough
            # and costs no `jac` evaluations inside the search. With bounds
            # every method takes this path, along the projected path `P(x +
            # step d)` with the decrease measured against the move actually
            # made.
            for _ in range(60):
                var predicted = 0.0
                for i in range(n_vars):
                    candidate[i] = x[i] + step * direction[i]
                    if bounded:
                        candidate[i] = min(
                            max(candidate[i], lower[i]), upper[i]
                        )
                    predicted += grad[i] * (candidate[i] - x[i])
                var trial = _as_tensor[dtype, n_vars](candidate, ctx)
                f_candidate = Float64(f(trial, ctx))
                if f_candidate <= f_x + 1e-4 * predicted and predicted < 0:
                    accepted = True
                    break
                step = step / 2

        if not accepted:
            return TensorMinimizeResult[dtype, n_vars](
                _as_tensor[dtype, n_vars](x, ctx),
                f_x,
                grad_norm,
                iteration + 1,
                False,
            )

        var moved = _as_tensor[dtype, n_vars](candidate, ctx)
        var grad_new = _to_list[dtype, n_vars](jac(moved, ctx))

        var s = List[Float64](capacity=n_vars)
        var y = List[Float64](capacity=n_vars)
        var sy = 0.0
        for i in range(n_vars):
            var si = candidate[i] - x[i]
            var yi = grad_new[i] - grad[i]
            if bounded and (
                (candidate[i] <= lower[i] and grad_new[i] > 0)
                or (candidate[i] >= upper[i] and grad_new[i] < 0)
            ):
                si = 0
                yi = 0
            s.append(si)
            y.append(yi)
            sy += si * yi

        comptime if method == "bfgs":
            # Skip the update when the curvature condition failed: the
            # formula divides by `sy`, and a non-positive value would make
            # `H` indefinite. Keeping the previous `H` costs one slower
            # step; a NaN-filled one costs the whole run.
            if sy > 0:
                h = _bfgs_update[dtype, n_vars, gpu](h^, s, y, sy, ctx)
        elif method == "l-bfgs":
            if sy > 1e-300:
                if len(s_hist) == memory:
                    _ = s_hist.pop(0)
                    _ = y_hist.pop(0)
                    _ = rho.pop(0)
                s_hist.append(s^)
                y_hist.append(y^)
                rho.append(1.0 / sy)
        else:
            # Polak-Ribiere, clamped at zero, with a restart every `n_vars`
            # iterations. Both safeguards matter more than the formula:
            # unclamped PR can fail to converge, and conjugacy decays on
            # anything that is not a quadratic.
            var numerator = 0.0
            var denominator = 0.0
            for i in range(n_vars):
                numerator += grad_new[i] * (grad_new[i] - grad[i])
                denominator += grad[i] * grad[i]
            var beta = 0.0
            if denominator > 0 and (iteration + 1) % n_vars != 0:
                beta = max(Float64(0), numerator / denominator)
            for i in range(n_vars):
                direction[i] = -grad_new[i] + beta * direction[i]
            previous_step = step
            previous_directional = directional

        for i in range(n_vars):
            x[i] = candidate[i]
            grad[i] = grad_new[i]
        f_x = f_candidate
        grad_norm = _projected_gradient_norm(
            x, grad, lower, upper, n_vars, bounded
        )

    return TensorMinimizeResult[dtype, n_vars](
        _as_tensor[dtype, n_vars](x, ctx),
        f_x,
        grad_norm,
        max_iter,
        False,
    )
