"""Roots of a square nonlinear system over `numax.core.array.Tensor`.
`scipy.optimize.root`.

**This module is tier 2.** It iterates to a tolerance with a host driver
loop; the linear algebra of each step is MAX's. `numax.optimize.array.root`
is the `FloatLike` tier for a handful of unknowns, and reads its Jacobian
off `Gradient`; this one takes `jac` as an argument for the reason
`numax.optimize.minimize` gives -- a `Tensor` cannot hold a conformer.

Two methods:

- `"newton"` (the default) solves `J(x) d = -F(x)` through
  `numax.linalg.solve` -- blocked, pivoted LU on the device -- and
  backtracks on `||F||^2` until the step is a decrease. Quadratic near a
  root with a nonsingular Jacobian, and the natural method when the system
  is square and the Jacobian is in hand.
- `"lm"` is `numax.optimize.least_squares` applied to `F` itself, the same
  reuse the `Array` tier makes: a square system's root is a zero of
  `sum(F^2)`, and Levenberg-Marquardt gets there from a poor start where a
  Newton step diverges. Its damped step goes through `lstsq`'s
  device-resident QR.

**Check `residual_norm`, not only `converged`.** A system with no root
still has points where `||F||` stops falling; `residual_norm` near zero is
what says a root was found. `"hybr"`, SciPy's default Powell dogleg, is
not here: it is a trust-region policy rather than a variation on either of
these, and either of these answers the same systems.
"""

from std.math import sqrt as _sqrt

from max.gpu.host import DeviceContext

from ..core.array import Static
from ..linalg.basic import solve

from .common import _as_tensor
from .least_squares import least_squares
from .minimize import _infinity_norm, _to_list


struct TensorRootResult[dtype: DType, n: Int](Movable):
    """What `root` returns: the point, the infinity norm of `F` there, the
    iteration count and whether the tolerance was met. Movable but not
    `Copyable`: `x` owns a `DeviceBuffer`."""

    var x: Static[Self.dtype, Self.n]
    var residual_norm: Float64
    var iterations: Int
    var converged: Bool

    def __init__(
        out self,
        var x: Static[Self.dtype, Self.n],
        residual_norm: Float64,
        iterations: Int,
        converged: Bool,
    ):
        self.x = x^
        self.residual_norm = residual_norm
        self.iterations = iterations
        self.converged = converged


def _squared_norm(values: List[Float64], n: Int) -> Float64:
    var total = 0.0
    for i in range(n):
        total += values[i] * values[i]
    return total


def root[
    dtype: DType,
    n: Int,
    f: def(Static[dtype, n], DeviceContext) raises thin -> Static[dtype, n],
    jac: def(Static[dtype, n], DeviceContext) raises thin -> Static[
        dtype, n, n
    ],
    method: StaticString = "newton",
    gpu: Bool = False,
](
    x0: Static[dtype, n],
    tol: Optional[Float64] = None,
    max_iter: Optional[Int] = None,
) raises -> TensorRootResult[dtype, n] where (
    dtype.is_floating_point()
    and n >= 1
    # `least_squares` restates its own clauses and the prover does not carry
    # a caller's into a callee's parameter list, so they are spelled here.
    and n >= n
    and n + n >= n
):
    """Solve `f(x) = 0` for a vector `x` from `x0`. `scipy.optimize.root`
    over `Tensor`; the module docstring has the two methods.

    `f` returns the residual vector at a point and `jac` its `n x n`
    Jacobian there; both are compile-time parameters. Convergence is
    `max|f(x)| < tol` (default `1e-10`), within `max_iter` (default `100`)
    iterations. A run that exhausts them, or whose line search cannot find
    a decrease in `||f||`, returns `converged=False` with the best point it
    reached. `x0` is borrowed rather than consumed, for the reason
    `numax.optimize.common` gives.
    """
    var ctx = x0.context()
    var tolerance = tol.value() if tol else 1e-10
    var limit = max_iter.value() if max_iter else 100

    comptime if method == "lm":
        var fit = least_squares[dtype, n, n, f, jac, gpu](x0, tolerance, limit)
        var at = f(fit.x, ctx)
        var norm = _infinity_norm(_to_list[dtype, n](at), n)
        # `TensorFitResult` is `Movable` and cannot be taken apart field by
        # field, so the solution is re-staged from its host copy: `O(n)`.
        var solution = _as_tensor[dtype, n](_to_list[dtype, n](fit.x), ctx)
        return TensorRootResult[dtype, n](
            solution^, norm, fit.iterations, norm < tolerance
        )
    elif method == "newton":
        var x = _to_list[dtype, n](x0)
        var point = _as_tensor[dtype, n](x, ctx)
        var residual = _to_list[dtype, n](f(point, ctx))
        var norm = _infinity_norm(residual, n)
        for iteration in range(limit):
            if norm < tolerance:
                return TensorRootResult[dtype, n](
                    _as_tensor[dtype, n](x, ctx), norm, iteration, True
                )
            var here = _as_tensor[dtype, n](x, ctx)
            var jacobian = jac(here, ctx)
            var negated = List[Float64](capacity=n)
            for i in range(n):
                negated.append(-residual[i])
            var rhs = _as_tensor[dtype, n](negated, ctx)
            var direction = _to_list[dtype, n](
                solve[dtype, n, gpu](jacobian, rhs)
            )

            # Backtrack on `||F||^2` until the step is a real decrease.
            var current = _squared_norm(residual, n)
            var step = 1.0
            var accepted = False
            var candidate = List[Float64](length=n, fill=0.0)
            var next_residual = List[Float64]()
            for _ in range(40):
                for i in range(n):
                    candidate[i] = x[i] + step * direction[i]
                var trial = _as_tensor[dtype, n](candidate, ctx)
                next_residual = _to_list[dtype, n](f(trial, ctx))
                if (
                    _squared_norm(next_residual, n)
                    <= (1 - 2e-4 * step) * current
                ):
                    accepted = True
                    break
                step = step / 2
            if not accepted:
                return TensorRootResult[dtype, n](
                    _as_tensor[dtype, n](x, ctx), norm, iteration + 1, False
                )
            x = candidate^
            residual = next_residual^
            norm = _infinity_norm(residual, n)
        return TensorRootResult[dtype, n](
            _as_tensor[dtype, n](x, ctx), norm, limit, norm < tolerance
        )
    else:
        raise Error(
            "root: unknown method '",
            method,
            (
                "'; expected 'newton' or 'lm'. 'hybr' is not implemented -- see"
                " the module docstring."
            ),
        )
