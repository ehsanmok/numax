"""numax: one kernel, several meanings.

Write a numeric kernel once against `FloatLike`, and the type you instantiate
it with decides what you get: plain SIMD (`Plain`), a value paired with its
derivative (`Dual`), a value carried to roughly double precision
(`Compensated`), or exact base-10 fixed-point arithmetic (`Decimal`).
`numax` ships a growing library of kernels built the same way --
activations (`gaussian`, `sigmoid`, `swish`, `tanh`, `relu`, `leaky_relu`,
`gelu`, `softmax`) and special functions (`erf`, `erfc`, `gamma`, `lgamma`,
`digamma`, `gammainc`, `gammaincc`, `beta`, `betainc`, `betaincc`,
`legendre_p`, `bessel_j0`, `bessel_j1`, `bessel_y0`, `bessel_y1`,
`lambertw`, `lambertw_m1`, `elliptic_k`, `elliptic_e`) -- each one
differentiable and extra-precise with no changes of its own. `Complex`
adds a fifth conformer, a complex number over any of the other four.
`Gradient` generalizes `Dual` to several input variables at once.
"""

from .array import (
    Tensor,
    empty,
    empty_like,
    eye,
    full,
    full_like,
    linspace,
    logspace,
    ones,
    ones_like,
    squeeze,
    stack,
    transpose,
    zeros,
    zeros_like,
)
from .bessel import bessel_j0, bessel_j1, bessel_y0, bessel_y1
from .beta import beta, betainc, betaincc
from .compensated import Compensated
from .complex import Complex
from .decimal import Decimal
from .dual import Dual
from .elliptic import elliptic_e, elliptic_k
from .erf import erf, erfc
from .gamma import digamma, gamma, gammainc, gammaincc, lgamma
from .gradient import Gradient
from .io import load, print_tensor, save
from .lambertw import lambertw, lambertw_m1
from .legendre import legendre_p
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
from .fft import circular_convolve, fft, ifft
from .interp import (
    chebyshev_eval,
    chebyshev_fit,
    cubic_spline_eval,
    cubic_spline_moments,
    horner,
)
from .interval import Interval
from .linalg import (
    back_substitution,
    cholesky,
    cholesky_solve,
    det,
    forward_substitution,
    inverse,
    log_det_from_cholesky,
    lu,
    matmul,
    matvec,
    solve,
    tridiagonal_solve,
)
from .numeric import FloatLike
from .ode import dopri5, dopri5_with_error, rk4, rk4_system
from .orthopoly import chebyshev_t, chebyshev_u, hermite_h, laguerre_l
from .plain import Plain
from .quadrature import gauss_legendre, simpson, trapezoid
from .random import exponential, normal, seed, uniform
from .solve import bisection, halley, newton
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
from .special import (
    gaussian,
    gelu,
    leaky_relu,
    relu,
    sigmoid,
    softmax,
    swish,
    tanh,
)
