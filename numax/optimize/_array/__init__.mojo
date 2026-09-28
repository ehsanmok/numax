"""Private: the register tier of `numax.optimize`, over `Array[T, n]` and
`FloatLike` scalars.

Nothing outside numax imports this package by name. `numax.optimize`
exports its names, and each name it shares with the `Tensor` tier is an
overload in the `Tensor`-tier module that forwards here. The algorithms,
their tier and their bounds are documented on the functions in
`optimize.mojo`, `solve.mojo`.
"""

from .optimize import (
    ArrayMinimizeResult,
    OptimizeResult,
    bfgs,
    bisect_tol,
    brent,
    brentq,
    cg,
    curve_fit,
    fminbound,
    golden,
    halley_tol,
    least_squares,
    minimize,
    minimize_scalar,
    nelder_mead,
    newton_tol,
    root,
    root_scalar,
    secant,
)
from .solve import bisection, halley, newton
