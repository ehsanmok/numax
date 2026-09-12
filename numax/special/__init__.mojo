"""numax.special: special functions and activations.

Each one is written once against `FloatLike`, so the same source is
differentiable (`Dual`), extra-precise (`Compensated`), complex
(`Complex`) or interval-bounded (`Interval`) depending on what
instantiates it -- and every one is tier 1, launchable inside a GPU
thread, except the `Tensor` overload of `logsumexp`, which is a device
reduction and says so.

```mojo
from numax.special import gamma, j0, erf, gaussian
```

| Module | Contents |
|---|---|
| `erf` | `erf`, `erfc`, `erfinv`, `erfcinv` |
| `gamma` | `gamma`, `lgamma`, `digamma`, `gammainc`, `gammaincc`, `gammasgn`, `factorial`, `comb`, `perm`, `poch` |
| `expint` | `exp1`, `expi`, `expn`, `sici`, `fresnel` -- series against continued fractions, the complex ones over `Complex[T]` |
| `zeta` | `zeta(s)`, `zeta(s, q)` -- one Euler-Maclaurin sum on both sides of the pole |
| `hyper` | `hyp1f1`, `hyp2f1` -- the series, with Kummer's and Pfaff's transformations for the negative side |
| `information` | `xlogy`, `xlog1py`, `entr`, `rel_entr`, `kl_div`, `logit` -- the `0 log 0` conventions as blends |
| `logsumexp` | `logsumexp` over an `Array` (tier 1) and over a `Tensor` through MAX's `OnlineLogSumExp` monoid (tier 2, the one delegation here) |
| `beta` | `beta`, `betainc`, `betaincc` |
| `bessel` | `j0`, `j1`, `y0`, `y1` by A&S polynomials; `jv`, `yv`, `iv`, `kv`, `ive`, `kve`, `spherical_jn`, `spherical_yn` of any real order by Temme's method, fixed-depth continued fractions and held-or-taken recurrences |
| `airy` | `airy` -- `(Ai, Ai', Bi, Bi')`, series in the middle and the Bessel forms on both sides |
| `struve` | `struve` -- the Bessel-function series over Miller's recurrence to `x = 40`, the asymptotic expansion past it |
| `owens` | `owens_t` -- two fixed Gauss-Legendre rules after the argument reduction |
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
from .airy import airy
from .bessel import (
    iv,
    ive,
    j0,
    j1,
    jv,
    kv,
    kve,
    spherical_jn,
    spherical_yn,
    y0,
    y1,
    yv,
)
from .beta import beta, betainc, betaincc
from .elliptic import elliptic_e, elliptic_k
from .erf import erf, erfc, erfcinv, erfinv
from .gamma import (
    comb,
    digamma,
    factorial,
    gamma,
    gammainc,
    gammaincc,
    gammasgn,
    lgamma,
    perm,
    poch,
)
from .expint import exp1, expi, expn, fresnel, sici
from .hyper import hyp1f1, hyp2f1
from .information import entr, kl_div, logit, rel_entr, xlog1py, xlogy
from .logsumexp import logsumexp
from .zeta import zeta
from .lambertw import lambertw, lambertw_m1
from .legendre import legendre_p
from .orthopoly import chebyshev_t, chebyshev_u, hermite_h, laguerre_l
from .owens import owens_t
from .struve import struve
