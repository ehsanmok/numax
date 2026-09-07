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

## Layout

Flat surface, one module per operation family -- the shape `scipy.linalg`
uses, where `scipy.linalg.lu` is the public name and `_decomp_lu` is where
it lives. Import from `numax.linalg` and never think about the split; the
modules matter when reading or extending.

| Module | Holds | SciPy counterpart |
| --- | --- | --- |
| `blas` | `matmul`, `matvec`, `batched_matmul`, `dot`, `nrm2`, `asum`, `axpy`, `outer` | `scipy.linalg.blas`, `numpy.linalg.matmul` |
| `triangular` | `forward_substitution`, `back_substitution`, `tridiagonal_solve` | `solve_triangular`, `solve_banded` |
| `cholesky` | `cholesky`, `cholesky_solve`, `slogdet_cholesky` | `_decomp_cholesky` |
| `lu` | `lu`, `lu_factor`, `PivotedLU`, `TensorLU`, `det` | `_decomp_lu` |
| `qr` | `qr`, `lstsq` | `_decomp_qr` |
| `eigen` | `eigh`, `eigvals`, `svd` | `_decomp`, `_decomp_svd` |
| `basic` | `solve`, `inverse`, `pinv` | `_basic` |
| `misc` | `norm`, `cond`, `trace`, `fro`, `inf` | `_misc` |

A name lives in exactly one module, because Mojo deprecates importing one
name from two of them. That is what decides the split: `matmul`, `matvec`,
`cholesky`, `lu_factor` and `solve` each carry both an `Array` and a
`Tensor` overload, so the tiers of an operation are neighbours in one file
rather than being separated into an "array module" and a "tensor module".
`common` holds the private helpers and exports nothing.

## What is where, by tier

Over `Array`: factorizations (`cholesky`, `lu`, `qr`, `eigh`, `eigvals`,
`svd`), solves (`solve`, `lstsq`, `cholesky_solve`, `tridiagonal_solve`, the
substitutions), inverses (`inverse`, `pinv`), scalars (`det`, `trace`,
`cond`, `slogdet_cholesky`), norms (`norm` at `fro`/`1`/`inf`, `nrm2`, `asum`)
and products (`dot`, `axpy`, `outer`, `matvec`, `matmul`). Tier 1, except
`lu_factor` and the `PivotedLU` it returns: choosing a pivot by magnitude is
a branch on data, which buys the matrices unpivoted `lu` cannot factor at
the cost of the GPU.

Over `Tensor`: `matmul` (compile-time and run-time shapes), `matvec`,
`batched_matmul`, the BLAS-1 five (`dot`, `nrm2`, `asum`, `axpy`,
`outer`), and blocked `cholesky`, `lu_factor` (returning a reusable
`TensorLU`) and `solve`. Each takes a `gpu: Bool` parameter that chooses
MAX's target, and the factorizations a `block` size that tunes the panel.
`cholesky` is device-resident -- its panel steps are `panel.mojo` kernels
addressing the matrix in place and its trailing update is fused into
`matmul`'s epilogue, so nothing crosses to the host between the copy in and
the copy out.
`tril`/`triu` are `numax.core`'s, also MAX-backed.

The BLAS-1 five are the newest and the plainest illustration of what
"MAX-first" buys: MAX names none of them, but `dot`/`nrm2`/`asum` are its
`ReduceSum` monoid over its `rowwise` scaffolder with the multiply, square
or magnitude fused into the per-tile transform, and `axpy`/`outer` are
`max.algorithm.elementwise` maps. numax writes the contribution and the
signature; MAX supplies the SIMD width, the CPU threading and the GPU
tiering.

`gpu` is a compile-time parameter rather than a look at `ctx.api()`
because MAX's `target` is a `StaticString`: deciding it at run time would
compile the GPU kernels into every CPU-only build. `map` and `reduce` in
`numax.core.tensor` take the same parameter for the same reason. The
`Tensor` overloads also take their operands mutably even though they only
read them -- `view()` hands back a `TileTensor` that can write, and a
mutable view cannot be built from an immutable binding.

## Not here yet

`qr`, `svd`, `eigh`, `cholesky_solve` have no `Tensor` overload, so past
the crossover in `docs/performance.md` they are genuinely missing rather
than one import away. MAX's own `linalg.qr_factorization` does not close
that gap: it is on the older `LayoutTensor`, and numax's interop is
`TileTensor` only. `tridiagonal_solve` will stay `Array`-only -- Thomas is
already linear and has nothing to hand a GEMM.
"""

from .basic import inverse, pinv, solve
from .blas import (
    asum,
    axpy,
    batched_matmul,
    dot,
    matmul,
    matvec,
    nrm2,
    outer,
)
from .cholesky import cholesky, cholesky_solve, slogdet_cholesky
from .eigen import eigh, eigvals, svd
from .lu import PivotedLU, TensorLU, det, lu, lu_factor
from .misc import cond, fro, inf, norm, trace
from .qr import lstsq, qr
from .triangular import (
    back_substitution,
    forward_substitution,
    tridiagonal_solve,
)
