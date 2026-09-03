"""numax.optimize: minimization and scalar root finding.

Two halves, split by whether the iteration count is known up front.
`solve`'s `newton`/`halley`/`bisection` run a fixed number of steps with
no data-dependent branching, so they are tier 1 and GPU-launchable. The
rest converge to a tolerance and are tier 2, `Plain`-only and host-side.

```mojo
from numax.optimize import newton, brentq, bfgs
```

The objective is an ordinary `FloatLike` kernel, so `bfgs` evaluates it
at `Gradient` and gets every partial derivative exactly -- there is no
`jac` argument to pass.
"""

from .optimize import (
    MinimizeResult,
    OptimizeResult,
    bfgs,
    brentq,
    newton_tol,
)
from .solve import bisection, halley, newton
