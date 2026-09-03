"""`numax.core.gradient.Gradient[Inner, n_vars]`: forward-mode automatic
differentiation across several input variables at once, rather than
`Dual`'s single derivative.

Seed each input with `Gradient.variable(x_i, i)` -- value `x_i`, a one-hot
gradient at position `i` -- and a kernel of several inputs, built purely
from `+`/`*`/`/`/`exp`/`sin`/etc., returns every partial derivative from
one call, the same way `Dual` returns one. `variable`'s `Float64` overload
builds `Inner` from the literal via `Inner.constant` internally, so a
plain number seeds a variable directly with no `SIMD`/`Inner`-construction
helper needed at the call site.
"""

from numax import Gradient, Plain

comptime dtype = DType.float64
comptime width = 1
comptime G2 = Gradient[Plain[dtype, width], 2]


def main():
    print("--- Gradient[Plain, 2]: two partial derivatives in one pass ---")
    print("f(x, y) = x^2*y + sin(x*y), at (x, y) = (2, 3)")

    var x = G2.variable(2.0, 0)
    var y = G2.variable(3.0, 1)
    var f = x * x * y + (x * y).sin()

    print("f(2, 3)     =", f.value.v)
    print("df/dx(2, 3) =", f.grad[0].v, " (closed form: 2xy + y*cos(xy))")
    print("df/dy(2, 3) =", f.grad[1].v, " (closed form: x^2 + x*cos(xy))")
