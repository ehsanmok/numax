"""numax.interpolate: interpolation and polynomial evaluation.

```mojo
from numax.interpolate import interp, CubicSpline, Chebyshev, RegularGridInterpolator
```

Two tiers under one import. A name that exists in both -- `horner` --
is one function with two overloads, picked by the argument: a `Tensor`
takes the device tier, an `Array[T, n]` of `FloatLike` values takes the
register tier. The register tier's own names come from here too.

| Tier | Holds | Good for |
| --- | --- | --- |
| `Tensor`, `Plain`-only, tier 2 | `interp` (NumPy's linear lookup), `horner`, the legacy `numpy.poly*` family (`polyval`, `polyder`, `polyint`, `roots`, `polyfit`, all descending-coefficient where `horner` is ascending), and the cubic splines on knots that need not be uniform -- `CubicSpline` with SciPy's `bc_type`s, `PchipInterpolator`, `Akima1DInterpolator`, `CubicHermiteSpline` -- each evaluating any derivative order and integrating; `Chebyshev.fit(x, y)`, the least-squares series through `numax.linalg.lstsq`, with `chebval`; and `RegularGridInterpolator` on a 2-D rectilinear grid, linear or nearest | a device buffer of samples queried at a tensor of points |
| `Array[T, n]` and `FloatLike`, tier 1 | `horner`, `cubic_spline_moments`/`cubic_spline_eval` and the natural `ArrayCubicSpline`, `chebyshev_fit`/`chebyshev_eval` and `ArrayChebyshev` for a `FloatLike` function | a handful of knots in registers; differentiates at `Dual` and runs per SIMD lane inside a kernel |

What separates the tiers is the interval
search: over a tensor a query bisects the knots, a data-dependent branch
that a per-lane `FloatLike` kernel cannot make, which is why the `Array`
spline scans every interval and blends instead.

MAX has no interpolation at arbitrary points -- its `nn.resize_*` kernels
resample a whole image onto a fixed grid by a scale factor -- so both tiers
are numax's own; `numax/interpolate/interp.mojo` records the gate.
"""

from .interp import (
    horner,
    interp,
    polyder,
    polyfit,
    polyint,
    polyval,
    roots,
)
from .spline import (
    Akima1DInterpolator,
    CubicHermiteSpline,
    CubicSpline,
    PchipInterpolator,
)
from .chebyshev import Chebyshev, chebval
from .grid import RegularGridInterpolator
from ._array import (
    ArrayChebyshev,
    ArrayCubicSpline,
    chebyshev_eval,
    chebyshev_fit,
    cubic_spline_eval,
    cubic_spline_moments,
)
