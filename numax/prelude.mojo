"""The names most programs need, in one import.

```mojo
from numax.prelude import *
```

brings the conformers, `Tensor` and its creation and manipulation surface,
the elementwise math, the comparisons, the constants, the seam to the
`Array` layer, and the entry points of `numax.special`, `numax.linalg`,
`numax.stats` and `numax.io` that a program reaches for first.

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
what a reader expects to mean Mojo's own slicing.

Import the subpackage instead when you want everything: `from numax import
...` is the full flat surface, and `from numax.linalg import ...` is one
subsystem.
"""

# The trait and its conformers.
from .core.numeric import FloatLike
from .core.plain import Plain, f32, f64
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
    Shaped,
    Tensor,
    arange,
    broadcast_to,
    asarray,
    concatenate,
    concatenate_dyn,
    copy,
    diag,
    diagflat,
    diagonal,
    empty,
    empty_dyn,
    empty_like,
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
    ones,
    ones_dyn,
    ones_like,
    ravel,
    reshape,
    reshape_dyn,
    split,
    split_dyn,
    stack_dyn,
    squeeze,
    stack,
    to_array,
    to_tensor,
    transpose,
    tri,
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
    count_nonzero,
    extract,
    nonzero,
    searchsorted,
    select,
    sort,
    take,
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
from .special.erf import erf, erfc
from .special.gamma import digamma, gamma, lgamma
from .special.beta import beta

# Dense linear algebra.
from .linalg.linalg import (
    cholesky,
    det,
    dot,
    eigh,
    fro,
    inverse,
    lu,
    matmul,
    matvec,
    norm,
    qr,
    solve,
    svd,
    trace,
)

# Statistics and sampling. The builtin-shadowing reductions are excluded --
# see this module's docstring.
from .stats.statistics import (
    argmax,
    argmin,
    cumprod,
    cumsum,
    max_axis,
    mean,
    mean_axis,
    median,
    min_axis,
    mode,
    prod_axis,
    stddev,
    sum_axis,
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
from .optimize.solve import bisection, halley, newton
from .integrate.quadrature import gauss_legendre, simpson, trapezoid
from .integrate.ode import rk4
from .interpolate.interp import Chebyshev, CubicSpline, horner
from .fft.fft import fft, fftfreq, fftshift, ifft, ifftshift, irfft, rfft

# I/O.
from .io.io import nmx
from .io.npy import numpy
