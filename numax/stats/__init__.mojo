"""numax.stats: descriptive statistics, distributions, and sampling.

```mojo
from numax.stats import mean, variance, norm, uniform, seed
```

| Module | Contents |
|---|---|
| `statistics` | `sum`, `mean`, `median`, `mode`, `prod`, `cumsum`, `cumprod`, `variance`, `stddev`, `min`/`max`, `argmin`/`argmax` |
| `distributions` | `norm`, `gamma`, `beta`, `chi2`, `t`, `f`, `expon`, `binom`, `poisson` -- each a namespace with `.pdf`/`.pmf`, `.cdf` and `.ppf`, spelled the way `scipy.stats` spells them |
| `random` | `uniform`, `normal`, `exponential`, `randint`, `randbool`, `seed`, and `Generator` for a named reproducible stream |

Every reduction takes a `Tensor` and covers every element.
`mean`/`variance`/`stddev`/`cumsum` also have a `FloatLike`-generic form
over `List[T]`, so calling them at `Compensated` recovers the precision a
long float32 summation loses. `sum`, `prod`, `min`, `max` and `mean` each
carry a second overload folding one axis -- `sum(a)` and `sum[axis=k](a)`,
matching `numpy.sum(a)` and `numpy.sum(a, axis=k)`.
No `Random[FloatLike]` conformer: sampling is not differentiable, so the
trait contract does not fit.

Tier 2 over tensors (`Plain`-only, host-side, free to branch on data); the
`FloatLike`-generic `List[T]` reductions and every distribution function
are tier 1.
"""

from .distributions import (
    norm,
    expon,
    gamma,
    chi2,
    beta,
    t,
    f,
    poisson,
    binom,
)
from .random import (
    Generator,
    exponential,
    normal,
    randbool,
    randint,
    seed,
    uniform,
)
from .statistics import (
    argmax,
    argmin,
    cumprod,
    cumsum,
    max,
    mean,
    median,
    min,
    mode,
    prod,
    stddev,
    sum,
    variance,
)
