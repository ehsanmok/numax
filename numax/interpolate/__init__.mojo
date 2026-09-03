"""numax.interpolate: polynomial and spline interpolation.

```mojo
from numax.interpolate import horner, cubic_spline_moments, chebyshev_fit
```

Horner evaluation, natural cubic splines (`cubic_spline_moments` then
`cubic_spline_eval`, over `numax.linalg`'s tridiagonal solve), and
Chebyshev fit/eval. Tier 1, 1-D.
"""

from .interp import (
    chebyshev_eval,
    chebyshev_fit,
    cubic_spline_eval,
    cubic_spline_moments,
    horner,
)
