"""numax.optimize: minimization, root finding and nonlinear fitting.

```mojo
from numax.optimize import least_squares, curve_fit
```

Two tiers, one import each, the same split `numax.linalg` and `numax.fft`
make:

| Import | Holds | Good for |
| --- | --- | --- |
| `numax.optimize` | `Tensor`, `Plain`-only, tier 2 | objectives and fits sized for a tensor: `minimize` keeps BFGS's inverse Hessian on the device, and a fit's damped step goes through `numax.linalg.lstsq`'s blocked device-resident QR |
| `numax.optimize.array` | `Array[T, n]`, `FloatLike`-generic | everything else, including `minimize`, and the only tier that takes no Jacobian -- it reads one off `Gradient` exactly |

This surface is the `Tensor` one: `minimize` with `TensorMinimizeResult`,
and `least_squares`/`curve_fit` with the `TensorFitResult` they return. Both
take the derivative as a compile-time parameter -- `jac` and `jacobian` --
because a `dtype`-monomorphic tensor cannot hold a `Gradient`. Scalar root finding
(`root_scalar` over `brentq`, `bisect_tol`, `newton_tol`, `halley_tol` and
`secant`), the vector `root`, `minimize` (`bfgs`, `cg`, `nelder_mead`) and `minimize_scalar`
(`brent`, `golden`, `fminbound`) are `Array`-tier only: they work on a handful of scalars, which is the shape a
`Tensor` exists to not be.
"""

from .least_squares import TensorFitResult, curve_fit, least_squares
from .minimize import TensorMinimizeResult, minimize
