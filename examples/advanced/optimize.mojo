"""Minimization and root finding where the derivative is exact, not estimated.

`numax.optimize` is tier 2: it loops until it converges, branches on data,
and runs on the host. The *objective*, though, is an ordinary `FloatLike`
kernel -- written once, generic over its conformer -- so the optimizer can
evaluate it at whichever type it needs:

- `Plain` for the value, in the line search.
- `Dual` for one derivative, in `newton_tol`.
- `Gradient[..., n]` for all `n` partials at once, in `bfgs`.

That is the difference between this and a SciPy port. SciPy's `minimize`
either takes a hand-written `jac` or estimates one by finite differences,
and a finite difference cannot do better than about `eps**(2/3)` relative
accuracy no matter how the step is chosen -- truncation error and
cancellation error pull in opposite directions. Forward-mode AD has neither
term: the derivative is carried alongside the value by the chain rule, so it
is as accurate as the value itself.

This example does three things: it finds a root two ways and shows they
agree, it minimizes Rosenbrock from the classic bad starting point, and it
measures the exact gradient against the best finite difference a sweep of
step sizes can produce.

Run with `pixi run example-optimize`. CPU only -- nothing here is
GPU-launchable, which is what "tier 2" means.
"""

from std.collections import Array

from numax import Dual, FloatLike, Gradient, Plain
from numax.optimize import bfgs, brentq, newton_tol
from numax.optimize import newton

comptime P = Plain[DType.float64, 1]
comptime G = Gradient[P, 2]


def cos_minus_x[U: FloatLike](x: U) -> U:
    """`cos(x) - x`. Its root is the Dottie number, 0.739085..."""
    return x.cos() - x


def rosenbrock[U: FloatLike](v: Array[U, 2]) -> U:
    """`(1 - x)**2 + 100*(y - x**2)**2`.

    Minimum 0 at `(1, 1)`, at the bottom of a curved, narrow valley. The
    standard test precisely because steepest descent zig-zags across the
    valley instead of following it, so reaching `(1, 1)` says the curvature
    information is being used.
    """
    var a = U.one() - v[0]
    var b = v[1] - (v[0] * v[0])
    return a * a + U.constant(100.0) * b * b


def value_at(x: Float64, y: Float64) -> Float64:
    """`rosenbrock` at `Plain` -- the same kernel, no derivative."""
    var v = Array[P, 2](fill=P.constant(0.0))
    v[0] = P.constant(x)
    v[1] = P.constant(y)
    return Float64(rosenbrock[P](v^).v)


def main() raises:
    print("=== One kernel, three conformers ===")
    print()

    # The same `rosenbrock` body, evaluated three ways at (0.5, 0.5).
    var plain_v = Array[P, 2](fill=P.constant(0.0))
    plain_v[0] = P.constant(0.5)
    plain_v[1] = P.constant(0.5)
    print("Plain     f(0.5, 0.5)      =", rosenbrock[P](plain_v^))

    var dual_v = Array[Dual[P], 2](fill=Dual[P].constant(0.0))
    dual_v[0] = Dual[P](P.constant(0.5), P.one())
    dual_v[1] = Dual[P].constant(0.5)
    var d = rosenbrock[Dual[P]](dual_v^)
    print("Dual      df/dx            =", d.deriv)

    var grad_v = Array[G, 2](fill=G.constant(0.0))
    grad_v[0] = G.variable(0.5, 0)
    grad_v[1] = G.variable(0.5, 1)
    var g = rosenbrock[G](grad_v^)
    print(
        "Gradient  [df/dx, df/dy]   =",
        g.grad[0],
        g.grad[1],
    )
    print("           (analytic: -51.0, 50.0)")
    print()

    print("=== Root finding: cos(x) - x ===")
    print()

    # Tier 1: fixed 20 Newton steps, GPU-launchable, no convergence test.
    var tier1 = newton[P, cos_minus_x](P.constant(0.5))
    print("numax.optimize.newton  (tier 1, fixed 20 steps) =", tier1)

    # Tier 2: the same mathematics, iterating until the step is tiny, and
    # reporting whether it got there.
    var tier2 = newton_tol[cos_minus_x](0.5)
    print(
        "numax.optimize.newton_tol (tier 2)           =",
        tier2.x,
        " converged:",
        tier2.converged,
        " iterations:",
        tier2.iterations,
    )

    # Brent: no derivative at all, but it cannot leave the bracket, so it
    # cannot diverge the way Newton can.
    var bracketed = brentq[cos_minus_x](0.0, 2.0)
    print(
        "numax.optimize.brentq     (tier 2, bracketed) =",
        bracketed.x,
        " iterations:",
        bracketed.iterations,
    )
    print("           (bisecting [0, 2] to 1e-12 would take ~41)")
    print()

    print("=== BFGS on Rosenbrock, from the classic (-1.2, 1) ===")
    print()

    var start = Array[Float64, 2](fill=0)
    start[0] = -1.2
    start[1] = 1.0
    var minimized = bfgs[2, rosenbrock](start)
    print("x        =", minimized.x[0], minimized.x[1], " (exact: 1, 1)")
    print("f(x)     =", minimized.f_x, " (exact: 0)")
    print("max|grad|=", minimized.grad_norm)
    print(
        "converged:",
        minimized.converged,
        " iterations:",
        minimized.iterations,
    )
    print()

    print("=== Exact gradient vs. the best finite difference ===")
    print()

    # df/dx of Rosenbrock at (0.5, 0.5) is
    #   -2*(1 - x) - 400*x*(y - x*x) = -1.0 - 200*0.25 = -51.0
    comptime analytic = -51.0
    var exact = Float64(g.grad[0].v)
    print("analytic          :", analytic)
    print("forward-mode AD   :", exact, " error:", abs(exact - analytic))
    print()
    print("central differences, swept over step size:")

    var best_error = 1e30
    var best_h = 0.0
    var h = 1e-1
    for _ in range(12):
        var fd = (value_at(0.5 + h, 0.5) - value_at(0.5 - h, 0.5)) / (2 * h)
        var err = abs(fd - analytic)
        if err < best_error:
            best_error = err
            best_h = h
        print("  h =", h, " estimate =", fd, " error =", err)
        h = h / 10

    print()
    print("best finite-difference error:", best_error, " at h =", best_h)
    print("forward-mode AD error       :", abs(exact - analytic))
    print()
    print("The sweep is U-shaped: large h loses to truncation error, small h")
    print("loses to cancellation. AD has neither term, so there is no step")
    print("size to tune and nothing to trade off.")
