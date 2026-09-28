"""numax.optimize: minimization, root finding and nonlinear fitting.

```mojo
from numax.optimize import least_squares, curve_fit
```

Two tiers under one import. A name the two tiers share is one function
with two overloads, picked by the argument -- a `Tensor` takes the device
tier, an `Array[T, n]` of `FloatLike` values the register tier -- and the
register tier's own names are exported here too:

| Tier | Holds | Good for |
| --- | --- | --- |
| `Tensor`, `Plain`-only, tier 2 | objectives and fits sized for a tensor: `minimize` (`bfgs`, `l-bfgs`, `cg`, `powell`, with or without box bounds) keeps BFGS's inverse Hessian on the device, `root` solves its Newton step through `numax.linalg.solve`, a fit's damped step goes through `numax.linalg.lstsq`'s blocked device-resident QR, and `nnls`/`lsq_linear` form their normal equations there |
| `Array[T, n]`, `FloatLike`-generic | a handful of scalars, one problem per thread if need be: `minimize` over an `Array` (`bfgs`, `cg`, `nelder_mead`), `root`, `least_squares`/`curve_fit`, `minimize_scalar` (`brent`, `golden`, `fminbound`), `root_scalar` (`brentq`, `bisect_tol`, `newton_tol`, `halley_tol`, `secant`) and the fixed-iteration `newton`/`halley`/`bisection` -- the only tier that takes no Jacobian, since it reads one off `Gradient` exactly |

The `Tensor` tier's results: `minimize` with `MinimizeResult`,
`root` with `RootResult`, `least_squares`/`curve_fit` with
`FitResult`, and `nnls`/`lsq_linear` with `LinearResult`. The
gradient methods take the derivative as a compile-time parameter -- `jac`
and `jacobian` -- because a `dtype`-monomorphic tensor cannot hold a
`Gradient`; `"powell"` is the one that takes none. Scalar root finding
(`root_scalar` over `brentq`, `bisect_tol`, `newton_tol`, `halley_tol` and
`secant`), `nelder_mead` and `minimize_scalar` (`brent`, `golden`,
`fminbound`) are `Array`-tier only: they work on a handful of scalars,
which is the shape a `Tensor` exists to not be.
"""

from .least_squares import FitResult, curve_fit, least_squares
from .linear import LinearResult, lsq_linear, nnls
from .minimize import MinimizeResult, minimize
from .root import RootResult, root
from ._array import (
    ArrayMinimizeResult,
    OptimizeResult,
    bfgs,
    bisect_tol,
    bisection,
    brent,
    brentq,
    cg,
    fminbound,
    golden,
    halley,
    halley_tol,
    minimize_scalar,
    nelder_mead,
    newton,
    newton_tol,
    root_scalar,
    secant,
)
