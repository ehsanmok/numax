"""Nonlinear least squares over `numax.core.array.Tensor`.

**This module is tier 2.** It iterates to a tolerance and its driver loop
runs on the host, so nothing here is launchable inside a kernel body.
`numax.optimize.array` is the `FloatLike` tier; this one is `Plain`-only and
exists for fits whose residual vector is too large to sit in registers.

## Why the Jacobian is an argument here and not there

`numax.optimize.array.least_squares` takes no Jacobian: it evaluates the
residuals at `Gradient[_P, n_params]`, and every residual comes back
carrying all `n_params` partial derivatives, so one call produces the exact
Jacobian by the chain rule.

That is structurally unavailable at this tier. A `Tensor` owns a
`DeviceBuffer[dtype]` and `DType` is MAX's closed enum of machine scalars,
so no conformer -- `Gradient` included -- can live in one. The choice is
therefore between a caller-supplied Jacobian and a finite difference, and
this tier takes the Jacobian: a forward difference would cost `n_params + 1`
residual evaluations and cap accuracy near `sqrt(eps)`, which is a silent
accuracy regression against the sibling function of the same name. Supplying
`J` explicitly keeps the two tiers honest about being the same algorithm on
different data.

## The algorithm

Levenberg-Marquardt, the same as the `Array` tier: a step that reduces the
cost is accepted and the damping relaxed, moving toward Gauss-Newton's
quadratic convergence; a step that does not is rejected and the damping
raised, shortening the step toward the gradient direction. That is what
converges from a poor start where plain Gauss-Newton diverges.

The damped step is solved as an **augmented least-squares problem** rather
than through the normal equations:

```
min || [ J        ] d  +  [ r ] ||
    || [ sqrt(l) I]       [ 0 ] ||
```

which `numax.linalg.lstsq` answers with a blocked, device-resident
Householder QR. The normal equations `(J^T J + l I) d = -J^T r` would square
the condition number of `J` and throw away half the significant digits of an
ill-conditioned fit; the augmented form has the same solution and the
conditioning of `J` itself.

**Where the work is.** The cost, the gradient norm and the assembly of the
augmented matrix are host-side walks over `n_resid * n_params` elements.
The QR is `O((m + n) n^2)` and goes to the device. So this tier pays for
itself when the residual vector is long -- many data points against few
parameters, which is the shape a fit usually has -- and not when it is
short. For a handful of residuals the `Array` tier is both faster and
differentiable.

Bounds, robust losses and sparse Jacobians are out of scope, the same
scope the `Array` tier states.
"""

from std.math import sqrt as _sqrt

from max.gpu.host import DeviceContext

from ..core.array import Static
from ..linalg.qr import lstsq

from .common import _as_tensor


struct TensorFitResult[dtype: DType, n_params: Int](Movable):
    """What a `Tensor`-tier fit returns.

    `x` is the parameter vector, `cost` is `sum(r**2) / 2` at it (SciPy's
    convention rather than the raw sum), and `grad_norm` is `max|J^T r|`,
    the first-order condition the convergence test actually looks at.

    Movable but not `Copyable`: `x` owns a `DeviceBuffer`.
    """

    var x: Static[Self.dtype, Self.n_params]
    var cost: Float64
    var grad_norm: Float64
    var iterations: Int
    var converged: Bool

    def __init__(
        out self,
        var x: Static[Self.dtype, Self.n_params],
        cost: Float64,
        grad_norm: Float64,
        iterations: Int,
        converged: Bool,
    ):
        """Field-wise, spelled out because a `Movable`-only struct gets no
        generated one: `x` owns a `DeviceBuffer` and so cannot be copied."""
        self.x = x^
        self.cost = cost
        self.grad_norm = grad_norm
        self.iterations = iterations
        self.converged = converged


def _damped_step[
    dtype: DType, n_resid: Int, n_params: Int, gpu: Bool, block: Int
](
    jacobian: List[Float64],
    residual: List[Float64],
    damping: Float64,
    ctx: DeviceContext,
) raises -> List[Float64] where (
    dtype.is_floating_point()
    and n_resid + n_params >= n_params
    and n_params >= 1
):
    """The Levenberg-Marquardt step, through the augmented least-squares
    system rather than the normal equations.

    `[J; sqrt(damping) I] d = [-r; 0]`, handed to `numax.linalg.lstsq`. The
    assembly is host-side and the solve is not; see the module docstring on
    where the work sits.
    """
    comptime rows = n_resid + n_params
    var scale = _sqrt(damping)

    var entries = List[Scalar[dtype]](capacity=rows * n_params)
    for k in range(n_resid):
        for j in range(n_params):
            entries.append(Scalar[dtype](jacobian[k * n_params + j]))
    for i in range(n_params):
        for j in range(n_params):
            entries.append(Scalar[dtype](scale if i == j else 0.0))

    var rhs = List[Scalar[dtype]](capacity=rows)
    for k in range(n_resid):
        rhs.append(Scalar[dtype](-residual[k]))
    for _ in range(n_params):
        rhs.append(Scalar[dtype](0))

    var a = Static[dtype, rows, n_params](ctx, entries^)
    var b = Static[dtype, rows](ctx, rhs^)
    var solved = lstsq[dtype, rows, n_params, gpu, block](a, b).to_host()

    var step = List[Float64](capacity=n_params)
    for j in range(n_params):
        step.append(Float64(solved[j]))
    return step^


def _cost_of(residual: List[Float64], count: Int) -> Float64:
    """`sum(r**2) / 2`, SciPy's convention for the least-squares cost."""
    var total = 0.0
    for k in range(count):
        total += residual[k] * residual[k]
    return total / 2


def least_squares[
    dtype: DType,
    n_params: Int,
    n_resid: Int,
    residuals: def(
        Static[dtype, n_params], DeviceContext
    ) raises thin -> Static[dtype, n_resid],
    jacobian: def(Static[dtype, n_params], DeviceContext) raises thin -> Static[
        dtype, n_resid, n_params
    ],
    gpu: Bool = False,
    block: Int = 16,
](
    x0: Static[dtype, n_params],
    tol: Float64 = 1e-10,
    max_iter: Int = 100,
) raises -> TensorFitResult[dtype, n_params] where (
    dtype.is_floating_point()
    and n_resid >= n_params
    and n_params >= 1
    # `_damped_step` restates this for `lstsq`; the solver does not carry a
    # caller's clause into a callee's own parameter list.
    and n_resid + n_params >= n_params
):
    """Minimize `sum(residuals(x)**2) / 2` by Levenberg-Marquardt.
    `scipy.optimize.least_squares` over `Tensor`.

    `residuals` returns the `n_resid` vector at a parameter vector and
    `jacobian` the `n_resid x n_params` matrix of its partials. Both are
    compile-time parameters, so neither can capture run-time data -- pass a
    fit's data through `curve_fit` below, which takes it as an argument.

    `n_resid >= n_params` is a `where` clause: an underdetermined fit has a
    solution space rather than a solution, and the augmented system would
    hide that rather than report it.

    Convergence is `max|J^T r| < tol`, reported in `grad_norm`. A fit that
    runs out of iterations, or whose damping loop cannot find a downhill
    step, returns `converged=False` with the best parameters it reached
    rather than raising.

    `x0` is borrowed rather than consumed, and deliberately: it is what
    keeps the `DeviceContext` this fit allocates on alive for the whole
    loop. See `numax.optimize.common`.

    The module docstring says why the Jacobian is an argument here when the
    `Array` tier's `least_squares` takes none, and when to prefer which.
    """
    var ctx = x0.context()
    var start = x0.to_host()
    var current = List[Float64](capacity=n_params)
    for j in range(n_params):
        current.append(Float64(start[j]))

    var cost = 0.0
    var grad_norm = 0.0
    var damping = 1e-3

    for iteration in range(max_iter):
        var point = _as_tensor[dtype, n_params](current, ctx)
        var r_host = residuals(point, ctx).to_host()
        var j_host = jacobian(point, ctx).to_host()

        var residual = List[Float64](capacity=n_resid)
        for k in range(n_resid):
            residual.append(Float64(r_host[k]))
        var jac = List[Float64](capacity=n_resid * n_params)
        for k in range(n_resid * n_params):
            jac.append(Float64(j_host[k]))

        cost = _cost_of(residual, n_resid)

        grad_norm = 0.0
        for j in range(n_params):
            var slope = 0.0
            for k in range(n_resid):
                slope += jac[k * n_params + j] * residual[k]
            grad_norm = max(grad_norm, abs(slope))
        if grad_norm < tol:
            return TensorFitResult[dtype, n_params](
                _as_tensor[dtype, n_params](current, ctx),
                cost,
                grad_norm,
                iteration,
                True,
            )

        var accepted = False
        var candidate = List[Float64](capacity=n_params)

        for _ in range(30):
            var step = _damped_step[dtype, n_resid, n_params, gpu, block](
                jac, residual, damping, ctx
            )
            candidate = List[Float64](capacity=n_params)
            for j in range(n_params):
                candidate.append(current[j] + step[j])

            var trial_point = _as_tensor[dtype, n_params](candidate, ctx)
            var trial_r = residuals(trial_point, ctx).to_host()
            var trial = List[Float64](capacity=n_resid)
            for k in range(n_resid):
                trial.append(Float64(trial_r[k]))

            if _cost_of(trial, n_resid) < cost:
                accepted = True
                damping = max(damping / 3, 1e-12)
                break
            damping = damping * 3

        if not accepted:
            return TensorFitResult[dtype, n_params](
                _as_tensor[dtype, n_params](current, ctx),
                cost,
                grad_norm,
                iteration + 1,
                False,
            )
        current = candidate^

    return TensorFitResult[dtype, n_params](
        _as_tensor[dtype, n_params](current, ctx),
        cost,
        grad_norm,
        max_iter,
        False,
    )


def curve_fit[
    dtype: DType,
    n_params: Int,
    n_points: Int,
    model: def(
        Static[dtype, n_points], Static[dtype, n_params], DeviceContext
    ) raises thin -> Static[dtype, n_points],
    model_jacobian: def(
        Static[dtype, n_points], Static[dtype, n_params], DeviceContext
    ) raises thin -> Static[dtype, n_points, n_params],
    gpu: Bool = False,
    block: Int = 16,
](
    xdata: Static[dtype, n_points],
    ydata: Static[dtype, n_points],
    p0: Static[dtype, n_params],
    tol: Float64 = 1e-10,
    max_iter: Int = 100,
) raises -> TensorFitResult[dtype, n_params] where (
    dtype.is_floating_point()
    and n_points >= n_params
    and n_params >= 1
    and n_points + n_params >= n_params
):
    """Fit `model(xdata, params)` to `ydata` by least squares.
    `scipy.optimize.curve_fit` over `Tensor`, first return value.

    `model` evaluates the whole curve at once -- `n_points` in, `n_points`
    out -- rather than a point at a time, because at this tier the data is
    already a tensor and a per-point call would be `n_points` launches.
    `model_jacobian` returns the partials of those values with respect to
    the parameters.

    This repeats `least_squares`'s loop rather than wrapping it, for the
    reason the `Array` tier's `curve_fit` records: `least_squares` takes its
    residuals as a compile-time parameter, and a non-capturing function
    cannot reach run-time data through one. Here the data is an argument to
    the fit and the model never has to close over it.
    """
    var ctx = p0.context()
    var start = p0.to_host()
    var observed = ydata.to_host()
    var current = List[Float64](capacity=n_params)
    for j in range(n_params):
        current.append(Float64(start[j]))

    var cost = 0.0
    var grad_norm = 0.0
    var damping = 1e-3

    for iteration in range(max_iter):
        var point = _as_tensor[dtype, n_params](current, ctx)
        var predicted = model(xdata, point, ctx).to_host()
        var j_host = model_jacobian(xdata, point, ctx).to_host()

        var residual = List[Float64](capacity=n_points)
        for k in range(n_points):
            residual.append(Float64(predicted[k]) - Float64(observed[k]))
        var jac = List[Float64](capacity=n_points * n_params)
        for k in range(n_points * n_params):
            jac.append(Float64(j_host[k]))

        cost = _cost_of(residual, n_points)

        grad_norm = 0.0
        for j in range(n_params):
            var slope = 0.0
            for k in range(n_points):
                slope += jac[k * n_params + j] * residual[k]
            grad_norm = max(grad_norm, abs(slope))
        if grad_norm < tol:
            return TensorFitResult[dtype, n_params](
                _as_tensor[dtype, n_params](current, ctx),
                cost,
                grad_norm,
                iteration,
                True,
            )

        var accepted = False
        var candidate = List[Float64](capacity=n_params)

        for _ in range(30):
            var step = _damped_step[dtype, n_points, n_params, gpu, block](
                jac, residual, damping, ctx
            )
            candidate = List[Float64](capacity=n_params)
            for j in range(n_params):
                candidate.append(current[j] + step[j])

            var trial_point = _as_tensor[dtype, n_params](candidate, ctx)
            var trial_p = model(xdata, trial_point, ctx).to_host()
            var trial = List[Float64](capacity=n_points)
            for k in range(n_points):
                trial.append(Float64(trial_p[k]) - Float64(observed[k]))

            if _cost_of(trial, n_points) < cost:
                accepted = True
                damping = max(damping / 3, 1e-12)
                break
            damping = damping * 3

        if not accepted:
            return TensorFitResult[dtype, n_params](
                _as_tensor[dtype, n_params](current, ctx),
                cost,
                grad_norm,
                iteration + 1,
                False,
            )
        current = candidate^

    return TensorFitResult[dtype, n_params](
        _as_tensor[dtype, n_params](current, ctx),
        cost,
        grad_norm,
        max_iter,
        False,
    )
