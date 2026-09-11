"""`numax.interpolate` over `Array[T, n]` and `FloatLike` scalars: the
register-resident, `FloatLike`-generic tier.

```mojo
from numax.interpolate.array import horner, CubicSpline, Chebyshev
```

**Opt-in, and one import away rather than in the flat surface.** `horner`
and `CubicSpline` are the same names `numax.interpolate` exports over
`Tensor`, so a caller gets one tier per import and the surface never
resolves an overload by a type the reader has to look up. `numax.interpolate`,
`numax.prelude` and `numax` itself export the `Tensor` tier; this subpackage
is the other one, the same split `numax.linalg.array`, `numax.fft.array`,
`numax.optimize.array` and `numax.integrate.array` make.

| Module | Holds |
| --- | --- |
| `interp` | `horner`; `cubic_spline_moments`/`cubic_spline_eval` and the `CubicSpline` object over them; `chebyshev_fit`/`chebyshev_eval` and the `Chebyshev` object, which fit a `FloatLike` *function* at Chebyshev nodes |

## Why this tier exists at all

Every routine is a `FloatLike` kernel with a compile-time size, so a spline
evaluated at `Dual` differentiates, a Chebyshev fit of `f` at `Dual`
carries derivatives with respect to whatever the interval depends on, and
all of it runs per SIMD lane inside a GPU thread. **Tier 1** throughout,
at the price the module docstring states: the spline scans every interval
and blends rather than searching, `O(n)` per point, which is right for the
register-resident sizes it is for and wrong for a spline over thousands of
knots -- the `Tensor` tier's case.

## Where the two tiers part

Size, and what is being interpolated. This tier holds a handful of knots
in registers and fits a *function*; the `Tensor` tier holds a device
buffer of samples, searches it, and fits *data* -- `interp`, non-uniform
`CubicSpline`, `PchipInterpolator`, `Akima1DInterpolator`, a least-squares
`Chebyshev.fit(x, y)` rather than a nodal one.

## The cost of the split, stated plainly

Mojo deprecates importing one name from two modules, so a file that wants
both tiers of a name has to alias one of them:

```mojo
from numax.interpolate import CubicSpline                  # Tensor
from numax.interpolate.array import CubicSpline as CubicSplineA  # Array
```

That is the price of a flat surface that means exactly one thing.
"""

from .interp import (
    Chebyshev,
    CubicSpline,
    chebyshev_eval,
    chebyshev_fit,
    cubic_spline_eval,
    cubic_spline_moments,
    horner,
)
