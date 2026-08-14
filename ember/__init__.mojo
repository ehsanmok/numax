"""Ember: one kernel, several meanings.

Write a numeric kernel once against `FloatLike`, and the type you instantiate
it with decides what you get: plain SIMD (`Plain`), a value paired with its
derivative (`Dual`), a value carried to roughly double precision
(`Compensated`), or exact base-10 fixed-point arithmetic (`Decimal`).
`ember` ships a growing library of kernels built the same way --
activations (`gaussian`, `sigmoid`, `swish`, `tanh`, `relu`, `leaky_relu`,
`gelu`, `softmax`) and special functions (`erf`, `erfc`, `gamma`, `lgamma`,
`gammainc`, `gammaincc`, `bessel_j0`, `lambertw`) -- each one differentiable
and extra-precise with no changes of its own.
"""

from .bessel import bessel_j0
from .compensated import Compensated
from .decimal import Decimal
from .dual import Dual
from .erf import erf, erfc
from .gamma import gamma, gammainc, gammaincc, lgamma
from .lambertw import lambertw
from .numeric import FloatLike
from .plain import Plain
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
