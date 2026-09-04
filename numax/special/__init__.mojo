"""numax.special: special functions and activations.

Each one is written once against `FloatLike`, so the same source is
differentiable (`Dual`), extra-precise (`Compensated`), complex
(`Complex`) or interval-bounded (`Interval`) depending on what
instantiates it -- and every one is tier 1, launchable inside a GPU
thread.

```mojo
from numax.special import gamma, j0, erf, gaussian
```

| Module | Contents |
|---|---|
| `erf` | `erf`, `erfc` |
| `gamma` | `gamma`, `lgamma`, `digamma`, `gammainc`, `gammaincc` |
| `beta` | `beta`, `betainc`, `betaincc` |
| `bessel` | `j0`, `j1`, `y0`, `y1` |
| `elliptic` | `elliptic_k`, `elliptic_e` |
| `lambertw` | `lambertw`, `lambertw_m1` |
| `legendre`, `orthopoly` | `legendre_p`; Chebyshev `T`/`U`, Hermite `H`, Laguerre `L` |
| `activations` | `gaussian`, `sigmoid`, `swish`, `tanh`, `relu`, `leaky_relu`, `gelu`, `softmax` |

Every approximation documents its own error bound, and `pixi run
accuracy` measures it against checked-in mpmath references at 50 digits.
"""

from .activations import (
    gaussian,
    gelu,
    leaky_relu,
    relu,
    sigmoid,
    softmax,
    swish,
    tanh,
)
from .bessel import j0, j1, y0, y1
from .beta import beta, betainc, betaincc
from .elliptic import elliptic_e, elliptic_k
from .erf import erf, erfc
from .gamma import digamma, gamma, gammainc, gammaincc, lgamma
from .lambertw import lambertw, lambertw_m1
from .legendre import legendre_p
from .orthopoly import chebyshev_t, chebyshev_u, hermite_h, laguerre_l
