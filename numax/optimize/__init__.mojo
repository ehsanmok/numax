"""numax.optimize: minimization, root finding and nonlinear fitting.

```mojo
from numax.optimize import least_squares, curve_fit
```

Two tiers, one import each, the same split `numax.linalg` and `numax.fft`
make:

| Import | Holds | Good for |
| --- | --- | --- |
| `numax.optimize` | `Tensor`, `Plain`-only, tier 2 | fits whose residual vector is long: the damped step goes through `numax.linalg.lstsq`'s blocked device-resident QR |
| `numax.optimize.array` | `Array[T, n]`, `FloatLike`-generic | everything else, including `minimize`, and the only tier that takes no Jacobian -- it reads one off `Gradient` exactly |

This surface is the `Tensor` one, and so far it carries `least_squares`,
`curve_fit` and the `TensorFitResult` they return. Scalar root finding
(`newton`, `halley`, `bisection`, `brentq`, `newton_tol`), `minimize`
(`bfgs`, `cg`, `nelder_mead`) and `minimize_scalar` (`brent`, `golden`,
`fminbound`) are `Array`-tier only: they work on a handful of scalars, which is the shape a
`Tensor` exists to not be.
"""

from .least_squares import TensorFitResult, curve_fit, least_squares
