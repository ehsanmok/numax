"""numax.integrate: quadrature and initial-value problem solvers.

Split the same way as `numax.optimize`. `quadrature`'s Gauss-Legendre,
Simpson and trapezoid rules and `ode`'s `rk4`/`dopri5` steps take a fixed
number of nodes or steps, so they are tier 1 and GPU-launchable.
`quad`/`quad_vec`/`solve_ivp`/`solve_ivp_stiff` adapt until they hit a
tolerance, so they
are tier 2, `Plain`-only and host-side.

```mojo
from numax.integrate import gauss_legendre, rk4, quad, solve_ivp
```

Because the integrand is a `FloatLike` kernel, differentiating through an
integral is just calling the same quadrature at `Dual`.
"""

from .integrate import (
    IVPResult,
    QuadResult,
    quad,
    quad_vec,
    solve_ivp,
    solve_ivp_stiff,
)
from .ode import dopri5, dopri5_step, dopri5_with_error, rk4, rk4_system
from .quadrature import gauss_legendre, simpson, trapezoid
