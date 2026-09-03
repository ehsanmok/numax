"""numax.linalg: small dense linear algebra, generic over FloatLike.

Matrices are comptime-sized `Array[T, n*n]` in registers, not heap
allocations. That is what makes `cholesky` differentiable at `Dual` and
launchable inside a GPU thread; MAX's own `linalg` is the right call past
roughly 8x8, and it is monomorphic in a raw `dtype`, so no conformer
passes through it.

```mojo
from numax.linalg import cholesky, qr, solve, det, norm_frobenius
```

Factorizations (`cholesky`, `lu`, `qr`, `eigh`, `svd`), solves
(`solve`, `cholesky_solve`, `tridiagonal_solve`, the substitutions),
inverses (`inverse`, `pinv`), scalars (`det`, `trace`, `cond`,
`log_det_from_cholesky`), norms (`norm_1`, `norm_inf`,
`norm_frobenius`, `nrm2`) and products (`dot`, `outer`, `matvec`,
`matmul`). Tier 1.
"""

from .linalg import (
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
