"""numax.interpolate: polynomial and spline interpolation.

```mojo
from numax.interpolate import horner, CubicSpline, Chebyshev
```

Horner evaluation, natural cubic splines (over `numax.linalg`'s
tridiagonal solve) and Chebyshev series. Tier 1, 1-D.

`CubicSpline` and `Chebyshev` are the `scipy.interpolate`-shaped objects:
built once, called many times, with `__call__` doing the evaluation. They
wrap `cubic_spline_moments`/`cubic_spline_eval` and
`chebyshev_fit`/`chebyshev_eval`, which stay public because they are what a
GPU-launchable kernel calls -- the objects are the convenience over them,
not a replacement.
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
