"""Root-finding and quadrature, both written against `FloatLike`.

Two things this shows that a conventional numerics library can't:

1. `newton` takes only `f`. No derivative is supplied, because the solver
   evaluates `f` at `Dual` internally and reads the derivative off the
   result. The same `f` you'd write for a plain evaluation is the whole
   input.
2. Running the *integrator* at `Dual` differentiates the integral itself.
   `d/db integral(f, a, b) = f(b)` comes out of the quadrature with no
   special support for it anywhere in `numax.integrate`.

The Gauss-Legendre nodes are themselves a small demonstration of the same
composability: they're roots of a Legendre polynomial (`numax.special.legendre`)
found by numax's own Newton solver (`numax.optimize`) differentiating through
numax's own `Dual`, evaluated at compile time so none of that costs
anything at run time.

Run with: `pixi run example-quadrature`
"""

from std.math import cos as cos_f64
from std.math import exp as exp_f64
from std.math import sin as sin_f64

from numax import (
    Dual,
    FloatLike,
    Plain,
    chebyshev_t,
    gauss_legendre,
    hermite_h,
    laguerre_l,
    legendre_p,
    newton,
    simpson,
    trapezoid,
)

comptime dtype = DType.float64
comptime width = 1
comptime P = Plain[dtype, width]
comptime D = Dual[P]


def scalar(x: Float64) -> P:
    return P(SIMD[dtype, width](x))


def show(x: P) -> Float64:
    return Float64(x.v)


def cos_minus_x[U: FloatLike](x: U) -> U:
    """`cos(x) - x`, whose root is the Dottie number."""
    return x.cos() + (-x)


def bell[U: FloatLike](x: U) -> U:
    """`exp(-x^2)`, integrated below."""
    return (-(x * x)).exp()


def sine[U: FloatLike](x: U) -> U:
    return x.sin()


def main() raises:
    print("=== Root-finding: only f is supplied ===")
    var dottie = newton[f=cos_minus_x](scalar(1.0))
    print("root of cos(x) - x  =", show(dottie))
    print("residual            =", show(cos_minus_x(dottie)))
    print()

    print("=== Gauss-Legendre against a uniform grid ===")
    var pi = 3.141592653589793
    var exact = 2.0
    var g8 = show(gauss_legendre[f=sine, n=8](scalar(0.0), scalar(pi)))
    var s64 = show(simpson[f=sine, num_panels=32](scalar(0.0), scalar(pi)))
    var t64 = show(trapezoid[f=sine, num_intervals=64](scalar(0.0), scalar(pi)))
    print("integral of sin over [0, pi], exact =", exact)
    print("  gauss-legendre,  8 points :", g8, " error", abs(g8 - exact))
    print("  simpson,        64 points :", s64, " error", abs(s64 - exact))
    print("  trapezoid,      64 points :", t64, " error", abs(t64 - exact))
    print()

    print("=== Differentiating through the integral ===")
    # F(b) = integral of exp(-t^2) from 0 to b. The fundamental theorem of
    # calculus says F'(b) = exp(-b^2); nothing in the quadrature knows that.
    var b = 1.25
    var seeded = gauss_legendre[f=bell, n=16](
        D.constant(0.0), D(scalar(b), scalar(1.0))
    )
    print("F(b)          =", show(seeded.value), " at b =", b)
    print("F'(b)         =", show(seeded.deriv))
    print("exp(-b^2)     =", exp_f64(-b * b))
    print()

    print("=== Orthogonal polynomials at x = 0.6 ===")
    var x = scalar(0.6)
    print("P_5(x) =", show(legendre_p(5, x)))
    print("H_5(x) =", show(hermite_h(5, x)))
    print("L_5(x) =", show(laguerre_l(5, x)))
    print("T_5(x) =", show(chebyshev_t(5, x)))
    # T_n(cos(theta)) = cos(n*theta) is the identity behind Chebyshev fits.
    var theta = 0.9
    print(
        "T_5(cos(0.9)) =",
        show(chebyshev_t(5, scalar(cos_f64(theta)))),
        " vs cos(4.5) =",
        cos_f64(5.0 * theta),
    )
