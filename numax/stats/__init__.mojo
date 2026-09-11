"""numax.stats: descriptive statistics, distributions, and sampling.

```mojo
from numax.stats import mean, variance, norm, uniform, seed
```

| Module | Contents |
|---|---|
| `statistics` | `sum`, `mean`, `median`, `mode`, `prod`, `cumsum`, `cumprod`, `variance`, `stddev`, `variance_axis`, `min`/`max`, `argmin`/`argmax`, `ptp`, `average`, `moment` |
| `quantiles` | `quantile`/`percentile` under every NumPy `method`, `nanquantile`/`nanpercentile`/`nanmedian`, `iqr` -- host-side, one sort |
| `histograms` | `histogram` (uniform, ranged, explicit edges, density, weights), `histogram2d`, `histogramdd`, `bincount`, `digitize` -- NumPy's edge rules, host-side |
| `correlation` | `cov`, `corrcoef`, `pearsonr`, `spearmanr`, `kendalltau`, `linregress`, `rankdata`, `zscore` -- SciPy's p-values through `t.sf`/`norm.sf`, host-side |
| `descriptive` | `skew`, `kurtosis`, `sem`, `gmean`, `hmean`, `entropy`, `trim_mean`, `describe` -- SciPy's bias corrections and conventions, host-side |
| `hypothesis` | `ttest_1samp`/`ttest_ind`/`ttest_rel`, `chisquare`, `ks_1samp`, `f_oneway`, `mannwhitneyu` -- each a statistic and a tail of `t`/`chi2`/`f`/`norm`, host-side |
| `nanfunctions` | `nansum`, `nanprod`, `nanmean`, `nanvar`, `nanstd`, `nanmin`, `nanmax` -- `isnan`, `select` and the plain reductions, composed |
| `distributions` | `norm`, `gamma`, `beta`, `chi2`, `t`, `f`, `expon`, `binom`, `poisson` -- each a namespace with the eight `scipy.stats` methods, `.pdf`/`.pmf`, `.logpdf`/`.logpmf`, `.cdf`, `.logcdf`, `.sf`, `.logsf`, `.ppf` and `.isf`, spelled the way `scipy.stats` spells them |
| `random` | `uniform`, `normal`, `exponential`, `randint`, `randbool`, `seed`, and `Generator` for a named reproducible stream |

Every reduction takes a `Tensor` and covers every element.
`mean`/`variance`/`stddev`/`cumsum` also have a `FloatLike`-generic form
over `List[T]`, so calling them at `Compensated` recovers the precision a
long float32 summation loses. `sum`, `prod`, `min`, `max`, `mean`, `median`, `mode`, `argmin`,
`argmax`, `cumsum` and `cumprod` each carry a second overload taking one axis -- `sum(a)` and
`sum[axis=k](a)`, matching `numpy.sum(a)` and `numpy.sum(a, axis=k)`. The
axis `argmin`/`argmax` return a tensor of positions *along that axis*
where the whole-tensor ones return a single flat `Int`, which is NumPy's
split too. `cumsum`/`cumprod` are scans rather than reductions, so their
axis forms keep `xs`'s shape and their no-axis forms flatten, again
matching NumPy.
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
    average,
    cumprod,
    cumsum,
    max,
    mean,
    median,
    min,
    mode,
    moment,
    prod,
    ptp,
    stddev,
    sum,
    variance,
    variance_axis,
)
from .quantiles import (
    iqr,
    nanmedian,
    nanpercentile,
    nanquantile,
    percentile,
    quantile,
)
from .histograms import (
    Histogram,
    Histogram2D,
    HistogramDD,
    bincount,
    digitize,
    histogram,
    histogram2d,
    histogramdd,
)
from .correlation import (
    CorrelationResult,
    LinregressResult,
    corrcoef,
    cov,
    kendalltau,
    linregress,
    pearsonr,
    rankdata,
    spearmanr,
    zscore,
)
from .descriptive import (
    Description,
    describe,
    entropy,
    gmean,
    hmean,
    kurtosis,
    sem,
    skew,
    trim_mean,
)
from .hypothesis import (
    TestResult,
    chisquare,
    f_oneway,
    ks_1samp,
    mannwhitneyu,
    ttest_1samp,
    ttest_ind,
    ttest_rel,
)
from .nanfunctions import (
    nanmax,
    nanmean,
    nanmin,
    nanprod,
    nanstd,
    nansum,
    nanvar,
)
