"""`numax.linalg` over `Array[T, n*n]`: the register-resident,
`FloatLike`-generic tier.

```mojo
from numax.linalg.array import cholesky, qr, eigh, lstsq
```

**Opt-in, and one import away rather than in the flat surface.** The names
here are the same names `numax.linalg` exports over `Tensor`, so a caller
gets one tier per import and the surface never resolves an overload by a
type the reader has to look up. `numax.linalg`, `numax.prelude` and
`numax` itself export the `Tensor` tier; this subpackage is the other one.

| Module | Holds |
| --- | --- |
| `blas` | `dot`, `nrm2`, `asum`, `axpy`, `outer`, `matvec`, `matmul`, `inner`, `kron`, `matrix_power` |
| `triangular` | `forward_substitution`, `back_substitution`, `tridiagonal_solve` |
| `cholesky` | `cholesky`, `cholesky_solve`, `slogdet_cholesky` |
| `lu` | `lu`, `lu_factor`, `PivotedLU`, `det`, `slogdet` |
| `qr` | `qr`, `lstsq` |
| `eigen` | `eigh`, `eigvals`, `eigvalsh`, `svd`, `svdvals`, `matrix_rank` |
| `matfuncs` | `expm` (tier 1, fixed squarings), `sqrtm` (symmetric positive definite) |
| `basic` | `solve`, `inverse`, `pinv` |
| `misc` | `norm`, `cond`, `trace` |

## Why this tier exists at all

It is not a fallback for small matrices, though it is faster there. It is
the only tier that **differentiates**: MAX's kernels are monomorphic in a
raw `dtype`, so no `Dual` passes through them, and `cholesky` at
`Dual[Plain]` gives the derivative of a factorization with no adjoint rule
written anywhere. It is also the only tier that **runs per SIMD lane
inside a GPU kernel body** -- an `Array` lives in registers, so a whole
`3 x 3` solve can sit inside a `map[gpu=True]` step, one problem per lane,
which is a shape the `Tensor` tier cannot express at all.

Most of it is tier 1 (fixed trip count, no branching). The exceptions are
declared in each module and in each function: `lu_factor`/`PivotedLU`,
`solve`, `inverse` and `pinv` pivot or go through `svd`.

## The cost of the split, stated plainly

Mojo deprecates importing one name from two modules, so a file that wants
both tiers of a name has to alias one of them:

```mojo
from numax.linalg import cholesky                  # Tensor
from numax.linalg.array import cholesky as chol_a  # Array
```

That is the price of a flat surface that means exactly one thing. Files
wanting only one tier -- which is nearly all of them -- pay nothing.

`fro` and `inf`, the `norm` selectors, are **not** re-exported here: they
are plain integer aliases shared by both tiers and live in
`numax.linalg.misc`, so `from numax.linalg import fro` is right whichever
tier's `norm` is being called.
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
from .eigen import eigh, eigvals, eigvalsh, matrix_rank, svd, svdvals
from .lu import PivotedLU, det, lu, lu_factor, slogdet
from .matfuncs import expm, sqrtm
from .misc import cond, norm, trace
from .qr import lstsq, qr
from .triangular import (
    back_substitution,
    forward_substitution,
    tridiagonal_solve,
)
