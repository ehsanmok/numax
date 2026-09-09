"""`numax.optimize` over `Array[T, n]`: the register-resident,
`FloatLike`-generic tier.

```mojo
from numax.optimize.array import newton, brentq, bfgs, least_squares
```

**Opt-in, and one import away rather than in the flat surface.** It shares
`least_squares` and `curve_fit` with the `Tensor` tier, so a caller gets one
tier per import and the surface never resolves an overload by a type the
reader has to look up. `numax.optimize`, `numax.prelude` and `numax` itself
export the `Tensor` tier; this subpackage is the other one, the same split
`numax.linalg.array` and `numax.fft.array` make.

| Module | Holds |
| --- | --- |
| `solve` | `newton`, `halley`, `bisection` |
| `optimize` | `newton_tol`, `brentq`, `bfgs`, `nelder_mead`, `least_squares`, `curve_fit`, `OptimizeResult`, `MinimizeResult` |

Two halves, split by whether the iteration count is known up front.
`solve`'s `newton`/`halley`/`bisection` run a fixed number of steps with no
data-dependent branching, so they are **tier 1** and GPU-launchable inside a
kernel body. The rest converge to a tolerance and are **tier 2**,
`Plain`-only and host-side.

## Why this tier exists at all

The objective is an ordinary `FloatLike` kernel, so `bfgs` evaluates it at
`Gradient` and gets every partial derivative exactly -- there is no `jac`
argument to pass. `least_squares` and `curve_fit` get the whole Jacobian the
same way, from one call per iteration rather than the `n_params + 1` a
finite difference would cost. That is the property the `Tensor` tier cannot
have: a `Tensor` is `dtype`-monomorphic, so no `Gradient` fits in one, and
`numax.optimize.least_squares` takes the Jacobian as an argument instead.

`nelder_mead` ignores the derivative on purpose, for objectives built from
branchless blends whose kinks make it misleading.

## The cost of the split, stated plainly

Mojo deprecates importing one name from two modules, so a file that wants
both tiers of a name has to alias one of them:

```mojo
from numax.optimize import least_squares                 # Tensor
from numax.optimize.array import least_squares as lsq_a  # Array
```

That is the price of a flat surface that means exactly one thing. Files
wanting only one tier -- which is nearly all of them -- pay nothing.
"""

from .optimize import (
    MinimizeResult,
    OptimizeResult,
    bfgs,
    brentq,
    curve_fit,
    least_squares,
    nelder_mead,
    newton_tol,
)
from .solve import bisection, halley, newton
