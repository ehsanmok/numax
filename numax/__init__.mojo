"""numax: one kernel, several meanings.

NumPy and SciPy's ground, in Mojo, on MAX. Special functions, linear
algebra, quadrature, ODE solvers, FFTs, distributions, and a NumPy-named
array surface, built on MAX's `TileTensor` and kernels.

**One kernel, several meanings.** Every function is written once against
the `FloatLike` trait. The type you call it with decides what comes back:
a value (`Plain`), a derivative (`Dual`), extra precision
(`Compensated`), exact base-10 fixed point (`Decimal`), a complex result
(`Complex`), a full gradient (`Gradient`), or an interval bound
(`Interval`). They nest, so autodiff, precision and complex arithmetic
compose instead of each needing its own copy of every kernel.

**One tensor, every device.** `Tensor` owns a MAX `DeviceBuffer`, so the
`DeviceContext` passed to a factory decides host or device memory.
Nothing else changes, and `.view()` yields the `TileTensor` every MAX
kernel takes.

```mojo
from numax import Dual, FloatLike, Plain

def g[T: FloatLike](x: T) -> T:          # written once, no `dtype`
    return (-(x * x)).exp()

def main():
    comptime P = Plain[DType.float32, 1]

    print(g(P(0.5)).v)                   # 0.7788008  -- just the value
    var d = g(Dual[P](P(0.5), P(1.0)))   # seed the derivative with 1
    print(d.value.v, d.deriv.v)          # 0.7788008 -0.7788008
```

## Layout

Subpackages follow NumPy/SciPy naming, so a NumPy or SciPy import has an
obvious counterpart. Each one re-exports its own public surface, and this
root package re-exports all of them, so both spellings work:

```mojo
from numax import Dual, cholesky, quad     # flat, everything in one place
from numax.linalg import cholesky          # or by subsystem
```

| Subpackage | Contents |
|---|---|
| `numax.core` | `FloatLike` and its conformers, `Tensor` creation and manipulation, arithmetic and operators, elementwise math, comparisons and logic, sorting and searching, `pi`/`e`. The tensor engine itself -- `map`/`reduce`/`reduce_axis`/`broadcast_op_rows` -- is `numax.core.tensor` |
| `numax.special` | Γ and B, `erf`, Bessel `J`/`Y`, Lambert `W`, elliptic `K`/`E`, orthogonal polynomials, activations |
| `numax.linalg` | `cholesky`, `lu`, `qr`, `eigh`, `svd`, `solve`, `inverse`, `pinv`, `det`, `trace`, `cond`, norms, `dot`/`nrm2`/`outer`, `matmul`, `tridiagonal_solve` |
| `numax.optimize` | `newton`/`halley`/`bisection` at a fixed iteration count; `newton_tol`, `brentq`, `bfgs` to a tolerance |
| `numax.integrate` | Gauss-Legendre/Simpson/trapezoid and `rk4`/`dopri5` at a fixed step; `quad`, `quad_vec`, `solve_ivp` adaptively |
| `numax.interpolate` | Horner, cubic splines, Chebyshev fits |
| `numax.fft` | `fft`/`ifft`, `rfft`/`irfft`, `fft2`, `fftfreq`, `fftshift`, circular convolution |
| `numax.signal` | `convolve`, `correlate`, Hann/Hamming/Blackman windows |
| `numax.stats` | `sum`/`mean`/`median`/`mode`/`argmax`..., and pdf/cdf/quantile for nine families, plus `uniform`/`normal`/`exponential`/`randint`/`randbool`/`seed` |
| `numax.io` | NumPy `.npy` interchange (`npy_load`/`npy_save`, byte-identical to `numpy.save`), numax's own `NMX1` `save`/`load`, and `print_tensor` |

## The two tiers

**Tier 1** is everything with a fixed iteration count and no per-lane
branching, and therefore launchable inside a GPU thread: the special
functions, `linalg`, the fixed-step algorithms. **Tier 2** is `optimize`,
the adaptive half of `integrate`, `sorting`, `logic`, `elementwise` and
`ops`: `Plain`-only, host-side, free to loop or branch on data. Every
module docstring declares its tier, and tier 1 never calls tier 2.

Design rationale: `docs/architecture.md`. What numax absorbs from
NumPy/SciPy, routes to MAX, or leaves out: `docs/parity.md`.
"""

# Numeric types: the `FloatLike` trait and its conformers, plus the
# `Tensor` surface (creation, arithmetic, elementwise math, comparisons,
# sorting) built over `TileTensor` -- `numax.core`.
from .core.array import (
    Tensor,
    arange,
    concatenate,
    copy,
    diag,
    diagflat,
    diagonal,
    empty,
    empty_like,
    eye,
    flip,
    full,
    full_like,
    geomspace,
    hstack,
    identity,
    linspace,
    logspace,
    meshgrid,
    ones,
    ones_like,
    ravel,
    reshape,
    split,
    squeeze,
    stack,
    transpose,
    tri,
    tril,
    triu,
    vander,
    vstack,
    zeros,
    zeros_like,
)
from .core.compensated import Compensated
from .core.complex import Complex
from .core.constants import e, e_at, pi, pi_at
from .core.decimal import Decimal
from .core.dual import Dual
from .core.elementwise import (
    absolute,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    cbrt,
    clip,
    diff,
    exp,
    exp2,
    expm1,
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
    sqrt,
    tan,
)
from .core.gradient import Gradient
from .core.interval import Interval
from .core.logic import (
    all_true,
    allclose,
    any_true,
    array_equal,
    equal,
    greater,
    greater_equal,
    isclose,
    isfinite,
    isinf,
    isnan,
    isneginf,
    isposinf,
    less,
    less_equal,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    not_equal,
    to_mask,
)
from .core.numeric import FloatLike
from .core.ops import (
    add,
    invert,
    astype,
    divide,
    floor_divide,
    mod,
    multiply,
    negative,
    power,
    subtract,
)
from .core.plain import Plain
from .core.sorting import (
    all_nonzero,
    any_nonzero,
    argsort,
    count_nonzero,
    extract,
    nonzero,
    searchsorted,
    sort,
    unique,
    select,
)

# Special functions and activations -- `numax.special`.
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
from .special.bessel import bessel_j0, bessel_j1, bessel_y0, bessel_y1
from .special.beta import beta, betainc, betaincc
from .special.elliptic import elliptic_e, elliptic_k
from .special.erf import erf, erfc
from .special.gamma import digamma, gamma, gammainc, gammaincc, lgamma
from .special.lambertw import lambertw, lambertw_m1
from .special.legendre import legendre_p
from .special.orthopoly import chebyshev_t, chebyshev_u, hermite_h, laguerre_l

# Dense linear algebra -- `numax.linalg`.
from .linalg.linalg import (
    back_substitution,
    cholesky,
    cholesky_solve,
    cond,
    det,
    dot,
    eigh,
    forward_substitution,
    inverse,
    log_det_from_cholesky,
    lu,
    matmul,
    matvec,
    norm_1,
    norm_frobenius,
    norm_inf,
    nrm2,
    outer,
    pinv,
    qr,
    solve,
    svd,
    trace,
    tridiagonal_solve,
)

# Minimization and scalar root finding -- `numax.optimize`.
from .optimize.optimize import (
    MinimizeResult,
    OptimizeResult,
    bfgs,
    brentq,
    newton_tol,
)
from .optimize.solve import bisection, halley, newton

# Quadrature and ODE solvers -- `numax.integrate`.
from .integrate.integrate import (
    IVPResult,
    QuadResult,
    quad,
    quad_vec,
    solve_ivp,
)
from .integrate.ode import (
    dopri5,
    dopri5_step,
    dopri5_with_error,
    rk4,
    rk4_system,
)
from .integrate.quadrature import gauss_legendre, simpson, trapezoid

# Interpolation -- `numax.interpolate`.
from .interpolate.interp import (
    chebyshev_eval,
    chebyshev_fit,
    cubic_spline_eval,
    cubic_spline_moments,
    horner,
)

# Discrete Fourier transforms -- `numax.fft`.
from .fft.fft import (
    circular_convolve,
    fft,
    fft2,
    fftfreq,
    fftshift,
    ifft,
    ifft2,
    irfft,
    rfft,
    rfftfreq,
)

# Convolution, correlation, windows -- `numax.signal`.
from .signal.signal import (
    apply_window,
    blackman,
    convolve,
    convolve_same,
    correlate,
    hamming,
    hann,
)

# Statistics, distributions, sampling -- `numax.stats`.
from .stats.distributions import (
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
from .stats.random import exponential, normal, randbool, randint, seed, uniform
from .stats.statistics import (
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

# Tensor I/O -- `numax.io`.
from .io.io import load, print_tensor, save
from .io.npy import npy_load, npy_save
