"""numax.stats: descriptive statistics, distributions, and sampling.

```mojo
from numax.stats import mean, variance, normal_cdf, uniform, seed
```

| Module | Contents |
|---|---|
| `statistics` | `sum`, `mean`, `median`, `mode`, `prod`, `cumsum`, `cumprod`, `variance`, `stddev`, `min`/`max`, `argmin`/`argmax` |
| `distributions` | pdf/cdf/quantile for normal, gamma, beta, chi-squared, Student t, F, exponential, binomial and Poisson |
| `random` | `uniform`, `normal`, `exponential`, `randint`, `randbool`, `seed` |

`mean`/`variance`/`stddev`/`cumsum` also have a `FloatLike`-generic form
over `List[T]`, so calling them at `Compensated` recovers the precision a
long float32 summation loses. Reductions cover every element -- there is
no `axis=` yet; `numax.core.tensor.reduce_axis` is the axis-wise route.
No `Random[FloatLike]` conformer: sampling is not differentiable, so the
trait contract does not fit.
"""

from .distributions import (
    beta_cdf,
    beta_pdf,
    beta_quantile,
    binomial_cdf,
    binomial_pmf,
    chi2_cdf,
    chi2_pdf,
    chi2_quantile,
    exponential_cdf,
    exponential_pdf,
    f_cdf,
    f_pdf,
    gamma_cdf,
    gamma_pdf,
    gamma_quantile,
    normal_cdf,
    normal_pdf,
    normal_quantile,
    poisson_cdf,
    poisson_pmf,
    student_t_cdf,
    student_t_pdf,
    student_t_quantile,
)
from .random import exponential, normal, randbool, randint, seed, uniform
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
