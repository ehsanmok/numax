"""`numax.integrate` over `Array[T, n]` and `FloatLike` scalars: the
register-resident, `FloatLike`-generic tier.

```mojo
from numax.integrate.array import gauss_legendre, simpson, trapezoid
```

**Opt-in, and one import away rather than in the flat surface.** `simpson`
and `trapezoid` are the same names `numax.integrate` exports over `Tensor`,
so a caller gets one tier per import and the surface never resolves an
overload by a type the reader has to look up. `numax.integrate`,
`numax.prelude` and `numax` itself export the `Tensor` tier; this subpackage
is the other one, the same split `numax.linalg.array`, `numax.fft.array` and
`numax.optimize.array` make.

| Module | Holds |
| --- | --- |
| `quadrature` | `gauss_legendre`, `simpson`, `trapezoid` -- a `FloatLike` integrand `f` over `[a, b]` at a fixed node count |

## Why this tier exists at all

The integrand is a `FloatLike` kernel, so integrating at `Dual`
differentiates the integral -- with respect to a limit or a parameter --
and integrating at `Compensated` recovers the digits a long sum loses.
Every node count is a compile-time parameter, so the whole quadrature is
**tier 1**: fixed work, no per-lane branch, launchable inside a GPU thread,
which is how `examples/advanced/ode.mojo` runs a thousand integrals one per
thread.

## Where the two tiers part, and the spelling that follows

This tier integrates a *function*: `simpson[f](a, b)` samples `f` itself on
its own grid. The `Tensor` tier integrates *samples* the caller already
holds: `simpson(y, dx)` and `simpson(y, x)`, which is exactly the signature
`scipy.integrate.simpson` has. Same name, same rule, different input --
and SciPy's spelling is the `Tensor` one, so `from numax import trapezoid`
means what `scipy.integrate.trapezoid` means. `docs/parity.md` records the
function-taking forms here as the divergent spelling.

## The cost of the split, stated plainly

Mojo deprecates importing one name from two modules, so a file that wants
both tiers of a name has to alias one of them:

```mojo
from numax.integrate import simpson                   # Tensor, over samples
from numax.integrate.array import simpson as simpson_f  # Array, over f
```

That is the price of a flat surface that means exactly one thing.
"""

from .quadrature import gauss_legendre, simpson, trapezoid
