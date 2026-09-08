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
| `triangular` | `solve_triangular`, `forward_substitution`, `back_substitution`, `tridiagonal_solve` | `solve_banded` |
| `cholesky` | `cholesky`, `cholesky_solve`, `slogdet_cholesky` | `_decomp_cholesky` |
| `lu` | `lu`, `lu_factor`, `PivotedLU`, `TensorLU`, `det` | `_decomp_lu` |
| `qr` | `qr`, `qr_factor`, `TensorQR`, `lstsq` | `_decomp_qr` |
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
`outer`), blocked `cholesky`, `lu_factor` (returning a reusable
`TensorLU`), `qr_factor` (returning a reusable `TensorQR`) and `solve`,
and the solves those unlock: `solve_triangular`, `cholesky_solve`,
`inverse`, `det` and the least-squares `TensorQR.solve`, plus the scalar
summaries `norm` (`fro`/`1`/`inf`) and `trace`. Each takes a
`gpu: Bool` parameter that chooses MAX's target, and everything blocked a
`block` size that tunes the panel.

The solves come in a vector and a matrix form, and the pair is not a
convenience: with one right-hand side the update between diagonal blocks
is a `gemv`, with several it is a matrix product and goes to
`linalg.matmul`. That is why `inverse` solves against the whole identity
in one call rather than looping the columns, and why `cholesky_solve` and
`TensorLU.solve` each have both spellings.
All three factorizations are device-resident -- their panel steps are
`panel.mojo` kernels addressing the matrix in place and their trailing
updates are fused into `matmul`'s epilogue, so nothing crosses to the host
between the copy in and the copy out. `TensorLU` holds its factors and its
pivot vector in device memory and carries `gpu` in its type, which is what
makes solving a device factorization from host code a compile error rather
than a device-pointer read.
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

`svd`, `eigh`, `eigvals`, `cond` and `pinv` have no `Tensor` overload, so
past the crossover in `docs/performance.md` they are genuinely missing
rather than one import away. `cond` and `pinv` are `svd`'s dependents and
move when it does. `tridiagonal_solve` will stay `Array`-only -- Thomas is
already linear and has nothing to hand a GEMM.

The spectral four stop here deliberately, and the reason is the shape of
the algorithms rather than the amount of work left. Every one of them is
two phases. The first is a reduction -- symmetric to tridiagonal, general
to bidiagonal or to Hessenberg -- and that phase is exactly the block
reflector `qr_factor` already runs: a panel of Householder vectors,
`larft_panel`'s `T`, a trailing update that is three GEMMs. It would be
MAX-first in the same way, and it is over half the arithmetic.

The second phase is not. Implicitly shifted QL/QR sweeps on the
tridiagonal, or Golub-Kahan sweeps on the bidiagonal, loop to a tolerance,
deflate on a data-dependent test, and do it on a band two entries wide.
There is no GEMM to hand anything to, the sweeps are sequential in a way
`O(n^2)` of total work spread over `O(n)` of them makes unfixable at this
level, and both properties are tier 2 by numax's own definition -- so a
`Tensor` `eigh` would be half a device-resident MAX kernel and half a host
loop that decides the runtime at exactly the sizes a `Tensor` tier is for.
Closing that half properly is LAPACK's multishift-with-aggressive-early-
deflation machinery, which is a research-grade item and not a missing
overload.

So the `Array` tier keeps `eigh`, `eigvals`, `svd`, `cond` and `pinv` for
matrices small enough to live in registers, where it also differentiates
them, and a caller with a large device-resident spectral problem gets an
honest no. When this resumes, the first commit is the reduction phase
alone -- `sytrd` on top of the existing block reflector, checkable by
asserting `Q^T A Q` is tridiagonal and similar to `A` -- with the sweep
following as declared tier 2.

`qr` is the one name where the two tiers are spelled differently.
`qr_factor` returns a `TensorQR` rather than a `(R, Q)` tuple, because a
`Tuple` of two `Tensor`s cannot be destructured in Mojo 1.0 -- `Tensor` is
`Movable`, tuple unpacking wants `ImplicitlyCopyable` -- so a tuple-shaped
`Tensor` overload of `qr` would hand back a pair no caller could take
apart. `TensorQR.r()` and `.q()` materialize either factor and
`.apply_q_transpose`/`.solve` skip `Q` entirely, which is LAPACK's split
and the more useful surface anyway.
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
from .qr import TensorQR, lstsq, qr, qr_factor
from .triangular import (
    back_substitution,
    forward_substitution,
    solve_triangular,
    tridiagonal_solve,
)
