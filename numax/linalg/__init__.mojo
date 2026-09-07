"""numax.linalg: dense linear algebra at two tiers, one set of names.

The `Array[T, n*n]` tier is comptime-sized and lives in registers, not on
the heap. That is what makes `cholesky` differentiable at `Dual` and
launchable inside a GPU thread, and it is the right shape for the small
matrices that appear *inside* a per-element kernel.

The `Tensor` tier goes through MAX. `matmul`, `matvec` and
`batched_matmul` are MAX kernels over `TileTensor`, so they inherit its
whole dispatch tree -- Apple, NVIDIA, AMD, vendor BLAS -- without numax
naming an architecture. `cholesky`, `lu_factor` and `solve` are gaps MAX
does not fill -- it ships no factorization on `TileTensor` at all -- so
numax writes them blocked and sends the cubic term back through `matmul`.
This tier is `dtype`-monomorphic, which is why the `Array` tier exists
beside it rather than being replaced by it.

Names are shared where both tiers have the operation; overload resolution
picks by argument type. `to_tensor`/`to_array` cross between them.

```mojo
from numax.linalg import cholesky, qr, solve, det, norm, matmul
```

Over `Array`: factorizations (`cholesky`, `lu`, `qr`, `eigh`, `eigvals`,
`svd`), solves (`solve`, `lstsq`, `cholesky_solve`, `tridiagonal_solve`, the
substitutions), inverses (`inverse`, `pinv`), scalars (`det`, `trace`,
`cond`, `slogdet_cholesky`), norms (`norm` at `fro`/`1`/`inf`, `nrm2`, `asum`)
and products (`dot`, `axpy`, `outer`, `matvec`, `matmul`). Tier 1, except
`lu_factor` and the `PivotedLU` it returns: choosing a pivot by magnitude is
a branch on data, which buys the matrices unpivoted `lu` cannot factor at
the cost of the GPU.

Over `Tensor`: `matmul` (compile-time and run-time shapes), `matvec`,
`batched_matmul`, and blocked `cholesky`, `lu_factor` (returning a
reusable `TensorLU`) and `solve`. Each takes a `gpu: Bool` parameter that
chooses MAX's target and a `block` size that tunes the panel.
`tril`/`triu` are `numax.core`'s, also MAX-backed.
"""

from .linalg import (
    back_substitution,
    cholesky,
    cholesky_solve,
    cond,
    det,
    asum,
    axpy,
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
    TensorLU,
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
    batched_matmul,
)
