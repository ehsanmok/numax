"""The names most programs need, in one import.

```mojo
from numax.prelude import *
```

brings the conformers, `Tensor` and its creation and manipulation surface
-- including the rank-generic half, `transpose`/`swapaxes`/`moveaxis`,
`expand_dims`/`roll`/`tile`/`repeat`, `broadcast_shapes` and
`take_along_axis` -- the elementwise math, the comparisons, the constants,
the seam to the `Array` layer, and the entry points of `numax.special`,
`numax.linalg` (factorizations, spectra, matrix functions, the structured
constructors), `numax.optimize`, `numax.integrate`, `numax.interpolate`,
`numax.fft`, `numax.signal`, `numax.stats` and `numax.io` that a program
reaches for first.

This module declares no tier of its own -- it re-exports, and each name
carries the tier of the module that defines it.

**What is deliberately not here.** `numax.stats`'s `sum`, `prod`, `min`,
`max`, and `numax.core`'s `abs`, `all`, `any`, `round` and `copysign` all
share a name with a Mojo builtin. A module-level definition *replaces* that
builtin for the rest of the importing file rather than overloading it, so a
star-import carrying them would silently break `min(1, 2)` in the caller's
own code. They are one qualified import away -- `from numax import abs`,
`from numax.stats import sum` -- and that import is then the caller's
explicit choice.

The nine `scipy.stats` distribution namespaces are not here either:
`gamma` and `beta` would collide with the special functions of those names.
Reach for them as `numax.stats.norm`, `numax.stats.chi2`, and so on.
`numax.core.array.slice` stays out on the same principle, since `slice` is
what a reader expects to mean Mojo's own slicing. And `hilbert` is here as
the `scipy.linalg` matrix only; the `scipy.signal` transform of the same
name is `numax.signal.hilbert`, because one name means one thing on the
flat surface.

Import the subpackage instead when you want everything: `from numax import
...` is the full flat surface, and `from numax.linalg import ...` is one
subsystem.
"""

# The trait and its conformers.
from .core.numeric import FloatLike
from .core.plain import Plain
from .core.dtypes import (
    bf16,
    bool,
    f8e3m4,
    f8e4m3fn,
    f8e4m3fnuz,
    f8e5m2,
    f8e5m2fnuz,
    f16,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
)
from .core.dual import Dual
from .core.gradient import Gradient
from .core.compensated import Compensated
from .core.decimal import Decimal
from .core.complex import Complex
from .core.interval import Interval
from .core.constants import e, e_at, pi, pi_at

# The tensor, its creation and manipulation surface, and the seam to
# `Array[T, n]`.
from .core.array import (
    Dynamic,
    Static,
    Tensor,
    arange,
    broadcast_shapes,
    broadcast_to,
    asarray,
    atleast_1d,
    atleast_2d,
    concatenate,
    concatenate_dyn,
    copy,
    diag,
    diagflat,
    diagonal,
    empty,
    empty_dyn,
    empty_like,
    expand_dims,
    eye,
    flip,
    full,
    full_dyn,
    full_like,
    geomspace,
    hstack,
    identity,
    linspace,
    logspace,
    meshgrid,
    moveaxis,
    ones,
    ones_dyn,
    ones_like,
    ravel,
    repeat,
    reshape,
    reshape_dyn,
    roll,
    split,
    split_dyn,
    stack_dyn,
    squeeze,
    stack,
    swapaxes,
    tile,
    to_array,
    to_tensor,
    transpose,
    tri,
    pad,
    pad_constant,
    pad_edge,
    pad_reflect,
    tril,
    triu,
    vander,
    vstack,
    zeros,
    zeros_dyn,
    zeros_like,
)

# Arithmetic and elementwise math.
from .core.ops import (
    add,
    astype,
    divide,
    floor_divide,
    invert,
    mod,
    multiply,
    negative,
    power,
    subtract,
)
from .core.elementwise import (
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    cbrt,
    ceil,
    clip,
    cos,
    cosh,
    diff,
    exp,
    exp2,
    expm1,
    floor,
    gradient,
    hypot,
    log,
    log10,
    log1p,
    log2,
    maximum,
    minimum,
    remainder,
    rsqrt,
    sin,
    sinh,
    sqrt,
    tan,
    trunc,
)

# Comparisons, masks, sorting.
from .core.logic import (
    allclose,
    array_equal,
    equal,
    greater,
    greater_equal,
    isclose,
    isfinite,
    isinf,
    isnan,
    less,
    less_equal,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    not_equal,
)
from .core.sorting import (
    argsort,
    argwhere,
    count_nonzero,
    extract,
    nonzero,
    put,
    searchsorted,
    select,
    sort,
    take,
    take_along_axis,
    top_k,
    unique,
)

# The special functions reached for first.
from .special.activations import (
    gaussian,
    gelu,
    leaky_relu,
    relu,
    sigmoid,
    softmax,
    swish,
    tanh,
)
from .special.erf import erf, erfc, erfcinv, erfinv
from .special.gamma import digamma, gamma, lgamma
from .special.expint import exp1, expi, expn, fresnel, sici
from .special.hyper import hyp1f1, hyp2f1
from .special.zeta import zeta
from .special.airy import airy
from .special.bessel import iv, jv, kv, yv
from .special.information import entr, kl_div, logit, rel_entr, xlog1py, xlogy
from .special.logsumexp import logsumexp
from .special.beta import beta

# Dense linear algebra over `Tensor`. The `Array` tier is
# `numax.linalg.array` and shares these names, so it stays out of the
# prelude for the same reason the builtin-shadowing reductions do.
from .linalg import (
    TensorLU,
    TensorQR,
    asum,
    axpy,
    batched_matmul,
    block_diag,
    cholesky,
    cholesky_solve,
    circulant,
    companion,
    cond,
    cosm,
    cross,
    det,
    eigh,
    eigvals,
    eigvalsh,
    expm,
    fractional_matrix_power,
    funm,
    hankel,
    hessenberg,
    hilbert,
    schur,
    dot,
    fro,
    inf,
    inner,
    inverse,
    kron,
    logm,
    lu_factor,
    matmul,
    matrix_power,
    matrix_rank,
    matvec,
    norm,
    nrm2,
    outer,
    lstsq,
    pinv,
    qr_factor,
    sinm,
    slogdet,
    solve,
    solve_banded,
    solve_toeplitz,
    solve_triangular,
    sqrtm,
    svd,
    svdvals,
    sytrd,
    tensordot,
    tensorinv,
    tensorsolve,
    toeplitz,
    trace,
)

# Statistics and sampling. The builtin-shadowing reductions are excluded --
# see this module's docstring.
from .stats.quantiles import (
    iqr,
    nanmedian,
    nanpercentile,
    nanquantile,
    percentile,
    quantile,
)
from .stats.histograms import (
    bincount,
    digitize,
    histogram,
    histogram2d,
    histogramdd,
)
from .stats.correlation import (
    corrcoef,
    cov,
    kendalltau,
    linregress,
    pearsonr,
    rankdata,
    spearmanr,
    zscore,
)
from .stats.descriptive import (
    describe,
    entropy,
    gmean,
    hmean,
    kurtosis,
    sem,
    skew,
    trim_mean,
)
from .stats.hypothesis import (
    chisquare,
    f_oneway,
    ks_1samp,
    mannwhitneyu,
    ttest_1samp,
    ttest_ind,
    ttest_rel,
)
from .stats.nanfunctions import (
    nanmax,
    nanmean,
    nanmin,
    nanprod,
    nanstd,
    nansum,
    nanvar,
)
from .stats.statistics import (
    average,
    moment,
    ptp,
    argmax,
    argmin,
    cumprod,
    cumsum,
    mean,
    median,
    mode,
    stddev,
    variance,
)
from .stats.random import (
    Generator,
    exponential,
    normal,
    randbool,
    randint,
    seed,
    uniform,
)

# Algorithms.
from .optimize.least_squares import curve_fit, least_squares
from .optimize.linear import lsq_linear, nnls
from .optimize.minimize import minimize
from .optimize.root import root
from .integrate.integrate import quad, quad_vec, solve_ivp
from .integrate.quadrature import cumulative_trapezoid, simpson, trapezoid
from .integrate.ode import dopri5, rk4_system
from .interpolate.interp import horner, interp
from .interpolate.spline import (
    Akima1DInterpolator,
    CubicHermiteSpline,
    CubicSpline,
    PchipInterpolator,
)
from .interpolate.chebyshev import Chebyshev, chebval
from .interpolate.grid import RegularGridInterpolator
from .fft.fft import (
    Spectrum,
    fft,
    fft2,
    fftfreq,
    fftshift,
    ifft,
    ifft2,
    ifftshift,
    irfft,
    next_fast_len,
    rfft,
    rfft2,
    rfftfreq,
)
from .fft.trig import dct, dst, idct, idst

# Signal processing. `hilbert` the transform is not here -- see the
# docstring.
from .signal.convolution import convolve, correlate, fftconvolve
from .signal.windows import (
    bartlett,
    blackman,
    boxcar,
    get_window,
    hamming,
    hann,
    kaiser,
)
from .signal.filters import (
    detrend,
    filtfilt,
    firwin,
    lfilter,
    medfilt,
    resample,
    savgol_filter,
    sosfilt,
)
from .signal.spectral import periodogram, spectrogram, stft, welch
from .signal.peaks import find_peaks
from .signal.design import butter, freqz

# I/O.
from .io.io import nmx
from .io.npy import numpy
