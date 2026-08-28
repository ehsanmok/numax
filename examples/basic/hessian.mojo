"""A full Hessian from `Gradient[Dual[Plain], n]` -- no second-order code.

`Dual` nested in itself already gives a second derivative of a
single-variable kernel. `Gradient` nested over `Dual` extends that to
several variables: seed variable `i` with `Dual(x_i, u_i)` for a direction
`u`, and `grad[i].deriv` comes back as `sum_j H[i,j] * u_j`. Choosing `u =
e_k` reads off one column of the Hessian, so `n` calls (or one call per
column, here two) assemble the whole matrix -- while a general `u` gives a
Hessian-vector product with no matrix formed at all.

Neither `numax/dual.mojo` nor `numax/gradient.mojo` has a line of code
about second derivatives. This works because both are written against
`Inner: FloatLike` rather than against `SIMD`, so each one's chain rule
composes with whatever the other does.
"""

from numax import Dual, Gradient, Plain

comptime dtype = DType.float64
comptime width = 1
comptime P = Plain[dtype, width]
comptime DP = Dual[P]
comptime G = Gradient[DP, 2]


def seed(value: Float64, direction: Float64, index: Int) -> G:
    return G.variable(DP(P.constant(value), P.constant(direction)), index)


def main():
    print("--- Gradient[Dual[Plain], 2]: value, gradient, and Hessian ---")
    print("f(x, y) = x^2*y + sin(x*y), at (x, y) = (2, 3)")
    print()

    # Column 0 of the Hessian: seed the direction vector u = (1, 0).
    var x0 = seed(2.0, 1.0, 0)
    var y0 = seed(3.0, 0.0, 1)
    var f0 = x0 * x0 * y0 + (x0 * y0).sin()

    # Column 1: u = (0, 1).
    var x1 = seed(2.0, 0.0, 0)
    var y1 = seed(3.0, 1.0, 1)
    var f1 = x1 * x1 * y1 + (x1 * y1).sin()

    print("f       =", f0.value.value.v)
    print("df/dx   =", f0.grad[0].value.v, " (closed form: 2xy + y*cos(xy))")
    print("df/dy   =", f0.grad[1].value.v, " (closed form: x^2 + x*cos(xy))")
    print()
    print("Hessian:")
    print(
        "  d2f/dx2  =", f0.grad[0].deriv.v, " (closed form: 2y - y^2*sin(xy))"
    )
    print(
        "  d2f/dxdy =",
        f0.grad[1].deriv.v,
        " (closed form: 2x + cos(xy) - xy*sin(xy))",
    )
    print("  d2f/dydx =", f1.grad[0].deriv.v, " (equal to d2f/dxdy)")
    print("  d2f/dy2  =", f1.grad[1].deriv.v, " (closed form: -x^2*sin(xy))")
    print()

    # A general direction gives H @ u in one pass, no matrix assembled.
    var xv = seed(2.0, 0.6, 0)
    var yv = seed(3.0, -1.4, 1)
    var fv = xv * xv * yv + (xv * yv).sin()
    print("Hessian-vector product with u = (0.6, -1.4), in one pass:")
    print("  (H @ u)[0] =", fv.grad[0].deriv.v)
    print("  (H @ u)[1] =", fv.grad[1].deriv.v)
