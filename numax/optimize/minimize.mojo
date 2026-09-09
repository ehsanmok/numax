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

The two methods differ in what they store, and at this tier that is the
whole decision:

- **`"bfgs"`** keeps an `n_vars x n_vars` inverse Hessian on the device and
  updates it with MAX's kernels -- one `matvec` for `H y`, three `outer`
  products for the rank-two correction, and elementwise combines. So the
  `O(n^2)` arithmetic per iteration is MAX's and never crosses to the host,
  but the `O(n^2)` *memory* is real and is what eventually bounds the size
  of problem this method fits.
- **`"cg"`** stores one direction vector. Every operation in its driver is
  `O(n_vars)`, so its per-iteration cost is dominated by the caller's own
  `f` and `jac` -- which are device tensors and where the work belongs. This
  is the method to reach for when `n_vars` is large.

Either way the driver's bookkeeping is a host-side walk over vectors of
length `n_vars`, so this tier pays for itself when evaluating `f` and `jac`
is the expense -- which is the shape a `Tensor`-sized problem has -- and not
when it is not. For a handful of variables the `Array` tier is both faster
and differentiable.

`"nelder-mead"` is deliberately absent. Its simplex is `n_vars + 1` points
of `n_vars` entries each, and comparing function values across all of them
every iteration is exactly the shape a `Tensor` tier exists to not be. It
stays in `numax.optimize.array`.

Bounds and constraints are out of scope, the same scope the `Array` tier
states.
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


def minimize[
    dtype: DType,
    n_vars: Int,
    f: def(Static[dtype, n_vars], DeviceContext) raises thin -> Scalar[dtype],
    jac: def(Static[dtype, n_vars], DeviceContext) raises thin -> Static[
        dtype, n_vars
    ],
    method: StaticString = "bfgs",
    gpu: Bool = False,
](
    x0: Static[dtype, n_vars],
    tol: Optional[Float64] = None,
    max_iter: Optional[Int] = None,
) raises -> TensorMinimizeResult[dtype, n_vars] where dtype.is_floating_point():
    """Minimize `f` from `x0`. `scipy.optimize.minimize` over `Tensor`.

    | `method` | Stores | Reach for it when |
    | --- | --- | --- |
    | `"bfgs"` (default) | an `n x n` inverse Hessian, on the device | the curvature is worth `O(n^2)` memory |
    | `"cg"` | one direction vector | `n_vars` makes that matrix the problem |

    `f` returns the objective at a point and `jac` its gradient there. Both
    are compile-time parameters, so neither can capture run-time data -- the
    module docstring says why the gradient is an argument at this tier when
    the `Array` tier needs none.

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
    comptime if method == "bfgs" or method == "cg":
        return _descend[dtype, n_vars, f, jac, method, gpu](
            x0,
            tol.value() if tol else 1e-8,
            max_iter.value() if max_iter else 200,
        )
    else:
        raise Error(
            "minimize: unknown method '",
            method,
            (
                "'; expected 'bfgs' or 'cg'. 'nelder-mead' is Array-tier only"
                " -- see the module docstring."
            ),
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
](
    x0: Static[dtype, n_vars],
    tol: Float64,
    max_iter: Int,
) raises -> TensorMinimizeResult[dtype, n_vars] where dtype.is_floating_point():
    """The shared driver. Both methods take the same steps -- evaluate,
    test, choose a direction, back-track, update -- and differ only in how
    the direction is produced, which is the one `comptime if` below.

    One loop rather than two because the line search, the convergence test,
    the failure reporting and the `List`/`Tensor` staging are identical and
    were the bulk of both.
    """
    var ctx = x0.context()

    var x = _to_list[dtype, n_vars](x0)
    var point = _as_tensor[dtype, n_vars](x, ctx)
    var f_x = Float64(f(point, ctx))
    var grad = _to_list[dtype, n_vars](jac(point, ctx))
    var grad_norm = _infinity_norm(grad, n_vars)

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

        comptime if method == "bfgs":
            # p = -H g, on the device.
            var g = _as_tensor[dtype, n_vars](grad, ctx)
            var hg = _to_list[dtype, n_vars](
                matvec[dtype, n_vars, n_vars, gpu](h, g)
            )
            for i in range(n_vars):
                direction[i] = -hg[i]

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
                directional -= grad[i] * grad[i]

        # BFGS's `H` carries the scale, so a unit step is the right first
        # trial. CG carries none, so the trial length comes from the
        # previous step rescaled by the ratio of the slopes -- Nocedal &
        # Wright (3.60) -- with a first iteration that moves `x` by about
        # one unit.
        var step: Float64
        comptime if method == "bfgs":
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

        comptime if method == "cg":
            # CG needs the curvature condition; see `_wolfe_step`.
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
            # BFGS backtracks on Armijo alone: `H` already carries the scale
            # of a good step, so a unit trial that is only ever halved is
            # enough and costs no `jac` evaluations inside the search.
            for _ in range(60):
                for i in range(n_vars):
                    candidate[i] = x[i] + step * direction[i]
                var trial = _as_tensor[dtype, n_vars](candidate, ctx)
                f_candidate = Float64(f(trial, ctx))
                if f_candidate <= f_x + 1e-4 * step * directional:
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
            s.append(candidate[i] - x[i])
            y.append(grad_new[i] - grad[i])
            sy += s[i] * y[i]

        comptime if method == "bfgs":
            # Skip the update when the curvature condition failed: the
            # formula divides by `sy`, and a non-positive value would make
            # `H` indefinite. Keeping the previous `H` costs one slower
            # step; a NaN-filled one costs the whole run.
            if sy > 0:
                h = _bfgs_update[dtype, n_vars, gpu](h^, s, y, sy, ctx)
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
        grad_norm = _infinity_norm(grad, n_vars)

    return TensorMinimizeResult[dtype, n_vars](
        _as_tensor[dtype, n_vars](x, ctx),
        f_x,
        grad_norm,
        max_iter,
        False,
    )
