"""numax: one kernel, several meanings.

A numerical computing library built on MAX: special functions, linear
algebra, quadrature, ODE solvers, FFTs, distributions, and a NumPy-named
array surface, written in Mojo against MAX's `TileTensor` and kernel
infrastructure.

**One kernel, several meanings.** Every function is written once against
the `FloatLike` trait. The type you call it with decides what comes back:
a value (`Plain`), a derivative (`Dual`), extra precision
(`Compensated`), exact base-10 fixed point (`Decimal`), a complex result
(`Complex`), a full gradient (`Gradient`), or an interval bound
(`Interval`). They nest, so autodiff, precision and complex arithmetic
compose instead of each needing its own copy of every kernel.

**One tensor, every device.** `Tensor` owns a MAX `DeviceBuffer`, so the
`DeviceContext` passed to a factory decides host or device memory: the
same kernel, any accelerator, unmodified. Nothing else changes, and
`.view()` yields the `TileTensor` every MAX kernel takes. Its shape lives
in its layout type, so `Static[f32, 2, 3]` and `Dynamic[f32, 2]` -- extents
compiled in, extents supplied at run time -- are one type, not two.

```mojo
from numax import Dual, FloatLike, Plain, f32

def g[T: FloatLike](x: T) -> T:              # written once, no `dtype`
    return (-(x * x)).exp()

def main():
    print(g(Plain[f32](0.5)).v)              # 0.7788008  -- just the value
    var d = g(Dual[Plain[f32]].seed(0.5))    # derivative seeded to 1
    print(d.value.v, d.deriv.v)              # 0.7788008 -0.7788008
```

## Layout

Subpackages follow NumPy/SciPy naming, so a NumPy or SciPy import has an
obvious counterpart. Each one re-exports its own public surface, and this
root package re-exports all of them, so both spellings work:

```mojo
from numax.prelude import *                # the common surface, one line
from numax import Dual, cholesky, quad     # flat, everything in one place
from numax.linalg import cholesky          # or by subsystem
from numax.linalg.array import cholesky    # ... or by tier
```

`numax.prelude` leaves out the names that would shadow a Mojo builtin
(`sum`, `min`, `max`, `abs`, `all`, `any`, `round`) so that a star import
is safe; its own docstring lists them and where to reach them.

| Subpackage | Contents |
|---|---|
| `numax.core` | `FloatLike` and its conformers, `Tensor` creation and manipulation, arithmetic and operators, elementwise math, comparisons and logic, sorting and searching, `pi`/`e`. The tensor engine itself -- `map`/`reduce`/`reduce_axis`/`broadcast_op_rows` -- is `numax.core.tensor` |
| `numax.special` | Γ and B, `erf`, Bessel `J`/`Y`, Lambert `W`, elliptic `K`/`E`, orthogonal polynomials, activations |
| `numax.linalg` | The `Tensor` tier, through MAX: `matmul`/`matvec`/`batched_matmul`/`inner` are MAX kernels, `cholesky`/`lu_factor`/`qr_factor`/`solve` are blocked with their `O(n^3)` update in MAX's GEMM, and `solve_triangular`/`cholesky_solve`/`lstsq`/`inverse`/`det`/`slogdet`/`norm`/`trace` build on those; `kron`/`matrix_power` and the `scipy.linalg` structured constructors (`toeplitz`, `hankel`, `circulant`, `companion`, `hilbert`, `block_diag`, `khatri_rao`, `convolution_matrix`), the banded and Toeplitz solves (`solve_banded`, `solveh_banded`, `cholesky_banded`, `solve_toeplitz`, `solve_circulant`) and `expm` sit beside them. `numax.linalg.array` is the `FloatLike`-generic tier, one import away because it shares these names: `cholesky`, `lu`, `qr`, `eigh`, `eigvals`, `eigvalsh`, `svd`, `svdvals`, `solve`, `lstsq`, `inverse`, `pinv`, `det`, `slogdet`, `trace`, `cond`, `matrix_rank`, norms, `dot`/`nrm2`/`outer`, `matmul`, `expm`, `sqrtm`, `tridiagonal_solve` |
| `numax.optimize` | `minimize` (`bfgs`, `cg`) and `least_squares`/`curve_fit` over `Tensor`, the fit's damped step through `numax.linalg.lstsq`; `numax.optimize.array` is the conformer tier and holds `newton`/`halley`/`bisection` at a fixed iteration count and `root_scalar` (`brentq`, `bisect_tol`, `newton_tol`, `halley_tol`, `secant`), `root`, `minimize` (`bfgs`, `cg`, `nelder_mead`), `minimize_scalar` (`brent`, `golden`, `fminbound`) and its own Jacobian-free `least_squares`/`curve_fit` to a tolerance |
| `numax.integrate` | `trapezoid`/`simpson`/`cumulative_trapezoid` over sampled `Tensor`s with `scipy.integrate`'s signatures; `quad`, `quad_vec`, `solve_ivp`, `solve_ivp_stiff` adaptively; `numax.integrate.array` is the `FloatLike` tier that integrates a function -- Gauss-Legendre, Simpson and trapezoid at a fixed node count, `rk4`/`dopri5` at a fixed step -- and differentiates at `Dual` |
| `numax.interpolate` | Horner, cubic splines, Chebyshev fits |
| `numax.fft` | `fft`/`ifft`, `rfft`/`irfft`, rectangular `fft2`/`ifft2`/`rfft2`, `fftshift`/`ifftshift`, `fftfreq`/`rfftfreq` over `Tensor` at any length -- radix-2 at a power of two, Bluestein otherwise -- device-resident across `log2(n) + 1` stages per axis; `dct`/`idct`/`dst`/`idst` types I-IV; `numax.fft.array` is the register-resident tier that differentiates, and adds circular convolution. MAX ships no forward transform at all |
| `numax.signal` | `convolve`, `correlate`, `lfilter`, `firwin`, Hann/Hamming/Blackman windows |
| `numax.stats` | `sum`/`mean`/`median`/`mode`/`argmax`..., the nine `scipy.stats`-shaped distribution namespaces (`numax.stats.norm.cdf`, ...), plus `uniform`/`normal`/`exponential`/`randint`/`randbool`/`seed` |
| `numax.io` | NumPy `.npy` interchange (`numpy.load`/`numpy.save`, byte-identical to `numpy.save`), and numax's own `NMX1` `nmx.save`/`nmx.load`. Printing is `print(a)`, since `Tensor` is `Writable` |

## The two tiers

**Tier 1** is everything with a fixed iteration count and no per-lane
branching, and therefore launchable inside a GPU thread: the special
functions, `numax.linalg.array`, the fixed-step algorithms. **Tier 2**
is `numax.linalg`'s `Tensor` tier (host-orchestrated, though `gpu=True` still
runs MAX's GPU kernels), `optimize`,
the adaptive half of `integrate`, `sorting`, `logic`, `elementwise` and
`ops`: `Plain`-only, host-side, free to loop or branch on data. Every
subpackage's docstring declares its tier, and tier 1 never calls tier 2.

Design rationale: `docs/architecture.md`. What numax absorbs from
NumPy/SciPy, routes to MAX, or leaves out: `docs/parity.md`.
"""

# Numeric types: the `FloatLike` trait and its conformers, plus the
# `Tensor` surface (creation, arithmetic, elementwise math, comparisons,
# sorting) built over `TileTensor` -- `numax.core`.
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
    roll,
    reshape_dyn,
    slice,
    split,
    split_dyn,
    stack_dyn,
    squeeze,
    stack,
    swapaxes,
    tile,
    transpose,
    to_array,
    to_tensor,
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
from .core.compensated import Compensated
from .core.complex import Complex
from .core.constants import e, e_at, pi, pi_at
from .core.decimal import Decimal
from .core.dual import Dual
from .core.elementwise import (
    abs,
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
    copysign,
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
    round,
    rsqrt,
    sin,
    sinh,
    sqrt,
    tan,
    trunc,
)
from .core.gradient import Gradient
from .core.interval import Interval
from .core.logic import (
    all,
    allclose,
    any,
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
from .core.sorting import (
    all_nonzero,
    any_nonzero,
    argsort,
    argwhere,
    count_nonzero,
    extract,
    nonzero,
    put,
    searchsorted,
    sort,
    take,
    take_along_axis,
    top_k,
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
from .special.bessel import j0, j1, y0, y1
from .special.beta import beta, betainc, betaincc
from .special.elliptic import elliptic_e, elliptic_k
from .special.erf import erf, erfc, erfcinv, erfinv
from .special.gamma import digamma, gamma, gammainc, gammaincc, lgamma
from .special.lambertw import lambertw, lambertw_m1
from .special.legendre import legendre_p
from .special.orthopoly import chebyshev_t, chebyshev_u, hermite_h, laguerre_l

# Dense linear algebra over `Tensor` -- `numax.linalg`. The
# `FloatLike`-generic `Array` tier is `numax.linalg.array`, deliberately not
# re-exported here: it shares these names, so a flat surface carrying both
# would resolve `cholesky` by a type the reader has to look up.
from .linalg import (
    TensorBidiagonal,
    TensorEigh,
    TensorLU,
    TensorQR,
    TensorSVD,
    TensorTridiagonal,
    asum,
    axpy,
    batched_matmul,
    block_diag,
    cholesky,
    cho_solve_banded,
    cholesky_banded,
    cholesky_solve,
    circulant,
    companion,
    cond,
    convolution_matrix,
    det,
    dot,
    eigh,
    eigvalsh,
    expm,
    fro,
    hankel,
    hilbert,
    inf,
    inner,
    inverse,
    khatri_rao,
    kron,
    lu_factor,
    matmul,
    matrix_power,
    matrix_rank,
    matvec,
    neg_inf,
    norm,
    nrm2,
    outer,
    lstsq,
    pinv,
    qr_factor,
    slogdet,
    solve,
    solve_banded,
    solve_circulant,
    solve_toeplitz,
    solve_triangular,
    solveh_banded,
    svd,
    svdvals,
    sytrd,
    toeplitz,
    trace,
)

# Minimization and least-squares fitting -- `numax.optimize`.
# The `Tensor` tier. The scalar root finders and minimizers, the vector
# `root`, and `nelder_mead` are `Array`-tier only, one import away at
# `numax.optimize.array`, which is also where the `Gradient`-exact
# `least_squares` and `curve_fit` live.
from .optimize.least_squares import TensorFitResult, curve_fit, least_squares
from .optimize.minimize import TensorMinimizeResult, minimize

# Quadrature and ODE solvers -- `numax.integrate`.
from .integrate.integrate import (
    IVPResult,
    QuadResult,
    TensorIVPResult,
    quad,
    quad_vec,
    solve_ivp,
    solve_ivp_stiff,
)
from .integrate.ode import TensorStep, dopri5, dopri5_step, rk4_system
from .integrate.quadrature import cumulative_trapezoid, simpson, trapezoid

# Interpolation -- `numax.interpolate`.
from .interpolate.interp import (
    Chebyshev,
    CubicSpline,
    chebyshev_eval,
    chebyshev_fit,
    cubic_spline_eval,
    cubic_spline_moments,
    horner,
)

# Discrete Fourier transforms -- `numax.fft`, the `Tensor` tier.
# `circular_convolve` is `Array`-tier only, one import away at
# `numax.fft.array`.
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

# Convolution, correlation, windows -- `numax.signal`.
from .signal.signal import (
    apply_window,
    blackman,
    convolve,
    correlate,
    firwin,
    hamming,
    hann,
    lfilter,
)

# Statistics, distributions, sampling -- `numax.stats`.
# The nine distribution namespaces (`norm`, `gamma`, `beta`, `chi2`, `t`,
# `f`, `expon`, `binom`, `poisson`) are reached as `numax.stats.norm` and
# are deliberately not re-exported here: `gamma` and `beta` would collide
# with the special functions of those names, and a root `t` or `f` names
# nothing a reader could guess.
from .stats.random import (
    Generator,
    exponential,
    normal,
    randbool,
    randint,
    seed,
    uniform,
)
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
from .io.io import nmx
from .io.npy import numpy
