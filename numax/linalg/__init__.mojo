"""numax.linalg: small dense linear algebra, generic over FloatLike.

Matrices are comptime-sized `Array[T, n*n]` in registers, not heap
allocations. That is what makes `cholesky` differentiable at `Dual` and
launchable inside a GPU thread; MAX's own `linalg` is the right call past
roughly 8x8, and it is monomorphic in a raw `dtype`, so no conformer
passes through it.

```mojo
from numax.linalg import cholesky, qr, solve, det, norm
```

Factorizations (`cholesky`, `lu`, `qr`, `eigh`, `eigvals`, `svd`), solves
(`solve`, `lstsq`, `cholesky_solve`, `tridiagonal_solve`, the
substitutions), inverses (`inverse`, `pinv`), scalars (`det`, `trace`,
`cond`, `slogdet_cholesky`), norms (`norm` at `fro`/`1`/`inf`, `nrm2`)
and products (`dot`, `outer`, `matvec`,
`matmul`). Tier 1, except `lu_factor` and the `PivotedLU` it returns:
choosing a pivot by magnitude is a branch on data, which buys the
matrices unpivoted `lu` cannot factor at the cost of the GPU.
"""

from .linalg import (
    back_substitution,
    cholesky,
    cholesky_solve,
    cond,
    det,
    dot,
    eigh,
    eigvals,
    forward_substitution,
    fro,
    inf,
    inverse,
    slogdet_cholesky,
    lstsq,
    lu,
    lu_factor,
    PivotedLU,
    matmul,
    matvec,
    norm,
    nrm2,
    outer,
    pinv,
    qr,
    solve,
    svd,
    trace,
    tridiagonal_solve,
)
