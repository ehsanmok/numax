"""numax.linalg: dense linear algebra over `Tensor`, through MAX.

```mojo
from numax.linalg import cholesky, qr_factor, solve, det, norm, matmul
```

Every name here takes a `Tensor` and runs where that tensor lives.
`matmul`, `matvec` and `batched_matmul` are MAX kernels over `TileTensor`,
so they inherit its whole dispatch tree -- Apple, NVIDIA, AMD, vendor
BLAS -- without numax naming an architecture. `cholesky`, `lu_factor`,
`qr_factor` and `solve` are gaps MAX does not fill, since it ships no
factorization on `TileTensor` at all, so numax writes them blocked and
sends the cubic term back through `matmul`. This tier is
`dtype`-monomorphic.

The `FloatLike`-generic, register-resident half of the library is
`numax.linalg.array`, one import away and covering names this surface does
not have at all -- `eigh`, `eigvals`, `svd`, `qr` as a pair of factors,
`pinv`, `cond`, the substitutions, `tridiagonal_solve`. That
subpackage's docstring says why it exists and what the split costs;
`to_tensor`/`to_array` cross between the two.

## Layout

Flat surface, one module per operation family -- the shape `scipy.linalg`
uses, where `scipy.linalg.lu` is the public name and `_decomp_lu` is where
it lives. Import from `numax.linalg` and never think about the split; the
modules matter when reading or extending.

| Module | Holds | SciPy counterpart |
| --- | --- | --- |
| `blas` | `matmul`, `matvec`, `batched_matmul`, `inner`, `kron`, `matrix_power`, `dot`, `nrm2`, `asum`, `axpy`, `outer` | `scipy.linalg.blas`, `numpy.linalg.matmul` |
| `triangular` | `solve_triangular` | `_basic`'s `solve_triangular` |
| `banded` | `solve_banded`, `solveh_banded`, `cholesky_banded`, `cho_solve_banded`, `solve_toeplitz`, `solve_circulant` | `_banded`, `_solve_toeplitz` |
| `cholesky` | `cholesky`, `cholesky_solve` | `_decomp_cholesky` |
| `lu` | `lu_factor`, `TensorLU`, `det`, `slogdet` | `_decomp_lu` |
| `qr` | `qr_factor`, `TensorQR`, `lstsq` | `_decomp_qr` |
| `basic` | `solve`, `inverse` | `_basic` |
| `misc` | `norm` (matrix and vector), `trace`, `fro`, `inf`, `neg_inf` | `_misc` |
| `matfuncs` | `expm` | `_matfuncs` |
| `special_matrices` | `toeplitz`, `hankel`, `circulant`, `companion`, `hilbert`, `block_diag`, `khatri_rao`, `convolution_matrix` | `_special_matrices` |
| `panel` | the unblocked tile kernels the factorizations step with | LAPACK's `*2` routines |

`common` holds the private helpers and exports nothing. `array/` mirrors
this split for the other tier, so `array.cholesky` is the `Array`
`cholesky` and this `cholesky` is the `Tensor` one -- a name is defined
once per tier and never twice within one.

## What is here

`matmul` (compile-time and run-time shapes), `matvec`, `batched_matmul`,
the BLAS-1 five (`dot`, `nrm2`, `asum`, `axpy`, `outer`), blocked
`cholesky`, `lu_factor` (returning a reusable `TensorLU`), `qr_factor`
(returning a reusable `TensorQR`) and `solve`, the solves those unlock --
`solve_triangular`, `cholesky_solve`, `inverse`, `det`, `slogdet`, and the
least-squares `TensorQR.solve` -- and the scalar summaries `norm`
(`fro`/`1`/`inf` over a matrix, `2`/`1`/`inf`/`neg_inf` over a vector) and
`trace`. Each takes a `gpu: Bool` parameter that
chooses MAX's target, and everything blocked a `block` size that tunes the
panel.

The solves come in a vector and a matrix form, and the pair is not a
convenience: with one right-hand side the update between diagonal blocks
is a `gemv`, with several it is a matrix product and goes to
`linalg.matmul`. That is why `inverse` solves against the whole identity
in one call rather than looping the columns, and why `cholesky_solve` and
`TensorLU.solve` each have both spellings.

All three factorizations are device-resident -- their panel steps are
`panel.mojo` kernels addressing the matrix in place and their trailing
updates are fused into `matmul`'s epilogue, so nothing crosses to the host
between the copy in and the copy out. `TensorLU` and `TensorQR` hold their
factors in device memory and carry `gpu` in their type, which is what
makes solving a device factorization from host code a compile error rather
than a device-pointer read. `tril`/`triu` are `numax.core`'s, also
MAX-backed.

The BLAS-1 five are the plainest illustration of what "MAX-first" buys:
MAX names none of them, but `dot`/`nrm2`/`asum` are its `ReduceSum` monoid
over its `rowwise` scaffolder with the multiply, square or magnitude fused
into the per-tile transform, and `axpy`/`outer` are
`max.algorithm.elementwise` maps. numax writes the contribution and the
signature; MAX supplies the SIMD width, the CPU threading and the GPU
tiering.

`gpu` is a compile-time parameter rather than a look at `ctx.api()`
because MAX's `target` is a `StaticString`: deciding it at run time would
compile the GPU kernels into every CPU-only build. `map` and `reduce` in
`numax.core.tensor` take the same parameter for the same reason. These
overloads also take their operands mutably even though they only read them
-- `view()` hands back a `TileTensor` that can write, and a mutable view
cannot be built from an immutable binding.

## Not here yet

`svd`, `eigh`, `eigvals`, `cond` and `pinv` have no `Tensor` overload, so
past the crossover in `docs/performance.md` they are genuinely missing
rather than one import away; `numax.linalg.array` has them for matrices
small enough to live in registers. `cond` and `pinv` are `svd`'s
dependents and move when it does. `tridiagonal_solve` will stay
`Array`-only -- Thomas is already linear and has nothing to hand a GEMM.

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

When this resumes, the first commit is the reduction phase alone --
`sytrd` on top of the existing block reflector, checkable by asserting
`Q^T A Q` is tridiagonal and similar to `A` -- with the sweep following as
declared tier 2.

`qr` is the one operation the two tiers spell differently. `qr_factor`
returns a `TensorQR` rather than a `(R, Q)` tuple, because a `Tuple` of
two `Tensor`s cannot be destructured in Mojo 1.0 -- `Tensor` is `Movable`,
tuple unpacking wants `ImplicitlyCopyable` -- so a tuple-shaped overload
would hand back a pair no caller could take apart. `TensorQR.r()` and
`.q()` materialize either factor and `.apply_q_transpose`/`.solve` skip
`Q` entirely, which is LAPACK's split and the more useful surface anyway.
`numax.linalg.array.qr` is the tuple-returning one.
"""

from .banded import (
    cho_solve_banded,
    cholesky_banded,
    solve_banded,
    solve_circulant,
    solveh_banded,
    solve_toeplitz,
)
from .basic import inverse, solve
from .blas import (
    asum,
    axpy,
    batched_matmul,
    dot,
    inner,
    kron,
    matmul,
    matrix_power,
    matvec,
    nrm2,
    outer,
)
from .cholesky import cholesky, cholesky_solve
from .lu import TensorLU, det, lu_factor, slogdet
from .matfuncs import expm
from .misc import fro, inf, neg_inf, norm, trace
from .qr import TensorQR, lstsq, qr_factor
from .special_matrices import (
    block_diag,
    circulant,
    companion,
    convolution_matrix,
    hankel,
    hilbert,
    khatri_rao,
    toeplitz,
)
from .triangular import solve_triangular
