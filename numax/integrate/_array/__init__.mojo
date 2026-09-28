"""Private: the register tier of `numax.integrate`, over `Array[T, n]` and
`FloatLike` scalars.

Nothing outside numax imports this package by name. `numax.integrate`
exports its names, and each name it shares with the `Tensor` tier is an
overload in the `Tensor`-tier module that forwards here. The algorithms,
their tier and their bounds are documented on the functions in
`ode.mojo`, `quadrature.mojo`.
"""

from .ode import dopri5, dopri5_step, dopri5_with_error, rk4, rk4_system
from .quadrature import gauss_legendre, simpson, trapezoid
