"""Private: the register tier of `numax.linalg`, over `Array[T, n]` and
`FloatLike` scalars.

Nothing outside numax imports this package by name. `numax.linalg`
exports its names, and each name it shares with the `Tensor` tier is an
overload in the `Tensor`-tier module that forwards here. The algorithms,
their tier and their bounds are documented on the functions in
`basic.mojo`, `blas.mojo`, `cholesky.mojo`, `eigen.mojo`, `lu.mojo`, `matfuncs.mojo`, `misc.mojo`, `qr.mojo`, `triangular.mojo`.
"""

from .basic import inverse, pinv, solve
from .blas import (
    asum,
    axpy,
    dot,
    inner,
    kron,
    matmul,
    matrix_power,
    matvec,
    nrm2,
    outer,
)
from .cholesky import cholesky, cholesky_solve, slogdet_cholesky
from .eigen import (
    eigh,
    eigvals,
    eigvalsh,
    hessenberg,
    matrix_rank,
    svd,
    svdvals,
)
from .lu import PivotedLU, det, lu, lu_factor, slogdet
from .matfuncs import expm, sqrtm
from .misc import cond, norm, trace
from .qr import lstsq, qr
from .triangular import (
    back_substitution,
    forward_substitution,
    tridiagonal_solve,
)
