"""numax.integrate: quadrature and initial-value problem solvers.

```mojo
from numax.integrate import trapezoid, simpson, cumulative_trapezoid, quad, solve_ivp
```

Two tiers, one import each, the same split `numax.linalg`, `numax.fft` and
`numax.optimize` make:

| Import | Holds | Good for |
| --- | --- | --- |
| `numax.integrate` | `Tensor`, `Plain`-only, tier 2 | samples the caller already holds: `trapezoid(y, dx)`, `simpson(y, x)`, `cumulative_trapezoid` -- `scipy.integrate`'s own signatures; and a system whose state is a `Tensor`: `rk4_system`, `dopri5`, `solve_ivp`, every stage an `elementwise` launch on the state's device |
| `numax.integrate.array` | `Array[T, n]` and `FloatLike`, tier 1 | a `FloatLike` integrand: `gauss_legendre[f](a, b)`, `simpson[f](a, b)`, `trapezoid[f](a, b)`, and the fixed-step `rk4`/`dopri5` |

This surface is the `Tensor` one. It also carries the adaptive scalar
drivers -- `quad`, `quad_vec`, the scalar `solve_ivp`, `solve_ivp_stiff` --
which take a `FloatLike` integrand but iterate to a tolerance on the host,
so they are tier 2 and belong to neither tier's type; they stay where they
were. `solve_ivp` is therefore two overloads under one name: a `Float64`
state and a `Tensor` one, with the same controller, and a test pins them
step for step at `n == 1`.

Because the integrand of the `array` tier is a `FloatLike` kernel,
differentiating through an integral is just calling the same quadrature at
`Dual`. The sample-taking rules here have no such property -- they never
see the function -- and are the ones a caller reaches for with data rather
than a formula.
"""

from .integrate import (
    IVPResult,
    QuadResult,
    TensorIVPResult,
    quad,
    quad_vec,
    solve_ivp,
    solve_ivp_stiff,
)
from .ode import TensorStep, dopri5, dopri5_step, rk4_system
from .quadrature import cumulative_trapezoid, simpson, trapezoid
