# Features

Complete inventory of what ships in [`numax/`](../numax/), walking
[`numax/__init__.mojo`](../numax/__init__.mojo) plus each subpackage. Every
entry is part of the public surface: `from numax import <name>` reaches all of
it flat, and `from numax.<subpackage> import <name>` reaches one subsystem.
Anything prefixed `_` is internal and excluded.

For the design rationale — the trait, the fixed-iteration invariant, the
tensor/GPU layer — see [`architecture.md`](architecture.md). For what numax
absorbs from NumPy/SciPy, routes to MAX, or leaves out, see
[`parity.md`](parity.md). Rendered API docs:
<https://ehsanmok.github.io/numax/>.

- [One import: `numax.prelude`](#one-import-numaxprelude)
- [The trait and its conformers](#the-trait-and-its-conformers)
- [The two tiers](#the-two-tiers)
- [`numax.core` — the tensor engine](#numaxcore--the-tensor-engine)
- [`numax.core` — arrays and the NumPy-named surface](#numaxcore--arrays-and-the-numpy-named-surface)
- [`numax.special`](#numaxspecial)
- [`numax.linalg`](#numaxlinalg)
- [`numax.optimize`](#numaxoptimize)
- [`numax.integrate`](#numaxintegrate)
- [`numax.interpolate`](#numaxinterpolate)
- [`numax.fft`](#numaxfft)
- [`numax.signal`](#numaxsignal)
- [`numax.stats`](#numaxstats)
- [`numax.io`](#numaxio)
- [Where rank stops](#where-rank-stops)
- [Accuracy](#accuracy)

## One import: `numax.prelude`

```mojo
from numax.prelude import *
```

brings the conformers, `Tensor` with its creation and manipulation surface,
the elementwise math and comparisons, `pi`/`e`, the `to_array`/`to_tensor`
seam, and the entry points of `numax.special`, `numax.linalg`,
`numax.stats`, `numax.integrate`, `numax.fft` and `numax.io` that a program
reaches for first.

What it leaves out, on purpose:

| Excluded | Why | Reach it as |
|---|---|---|
| `sum`, `prod`, `min`, `max` | Shadow Mojo builtins. A module-level definition *replaces* the builtin for the importing file rather than overloading it, so a star import carrying them would break `min(1, 2)` in the caller's own code | `from numax.stats import sum` |
| `abs`, `all`, `any`, `round`, `copysign` | Same | `from numax import abs` |
| `norm`, `expon`, `gamma`, `chi2`, `beta`, `t`, `f`, `poisson`, `binom` | The distribution namespaces: `gamma` and `beta` collide with the special functions of those names | `numax.stats.norm.cdf(...)` |

`from numax import ...` is the full flat surface and `from numax.<sub>
import ...` is one subsystem; the prelude is the convenience over both, not
a replacement for either.

## The trait and its conformers

`FloatLike` ([`numax/core/numeric.mojo`](../numax/core/numeric.mojo)) is
small: `+ * / -`, `exp`, `ln`, `sqrt`, `erf`, `erfc`, `sin`, `cos`, `abs`,
`copysign`, `floor`, `ceil`, `trunc`, `one()`, `constant()`. Every kernel in
the library is written against it once, and the conformer you instantiate
decides what a call returns.

| Type | What you get | Where |
|---|---|---|
| `Plain[dtype, width]` | Ordinary `SIMD`, at hardware speed — the baseline every kernel runs at unless you ask for something else. `map` over a `Plain` kernel measures 0.998x a hand-written raw-SIMD loop | [`core/plain.mojo`](../numax/core/plain.mojo) |
| `Dual[Inner]` | Forward-mode autodiff: `f(x)` and `f'(x)` from one call, by the chain rule built into the arithmetic. Nests for second derivatives | [`core/dual.mojo`](../numax/core/dual.mojo), [`gaussian.mojo`](../examples/basic/gaussian.mojo) |
| `Gradient[Inner, n_vars]` | Every `∂f/∂xᵢ` from one call; over `Dual` it is a full Hessian and Hessian-vector products | [`core/gradient.mojo`](../numax/core/gradient.mojo), [`hessian.mojo`](../examples/basic/hessian.mojo) |
| `Compensated[dtype, width]` | A value carried as `a + b`, with `b` holding the rounding error `a` lost — roughly double `dtype`'s precision, same algorithm | [`core/compensated.mojo`](../numax/core/compensated.mojo), [`statistics.mojo`](../examples/intermediate/statistics.mojo) |
| `Decimal[width, scale]` | Exact base-10 fixed point (`0.1 + 0.2 == 0.3`), scoped to modest magnitudes and single-digit `scale` | [`core/decimal.mojo`](../numax/core/decimal.mojo) |
| `Complex[Inner]` | Complex over any other conformer; `Complex[Dual[...]]` differentiates holomorphically | [`core/complex.mojo`](../numax/core/complex.mojo), [`complex.mojo`](../examples/basic/complex.mojo) |
| `Interval[Inner]` | An enclosure of every `f(x)` for `x` in `[lo, hi]`, including tight `sin`/`cos` over an interval | [`core/interval.mojo`](../numax/core/interval.mojo) |

Every conformer is `Writable`, so `print(d)` shows a `Dual` as its value and
derivative, an `Interval` as `[lo, hi]`, a `Complex` as `a + bi` — no `.v` to
reach the raw `SIMD` first. `.v` stays for the cases that want it: arithmetic
on the raw vector, and returning one from a `map` `step`.

`f32` and `f64` are `DType.float32` and `DType.float64`, exported so one name
serves both layers: `linspace[5, f64](...)` where a dtype belongs, `Plain[f64]`
where a conformer does. `Plain`'s `width` defaults to 1, so `Plain[f64]` is the
scalar and `Dual[Plain[f64]].seed(x)` is `x` with its derivative set to 1 — the
way nearly every single-variable differentiation starts.

They nest — `Complex[Dual[Plain[...]]]`, `Gradient[Dual[...], n]` — so
autodiff, precision and complex arithmetic compose instead of each needing its
own copy of every kernel. `pi`/`e` (and `pi_at`/`e_at`) are available at any
conformer from [`core/constants.mojo`](../numax/core/constants.mojo).

`numax.core.numeric` also exports the branchless helpers every
conformer-generic kernel is built from: `max_of`, `min_of`, `blend`,
`ge_indicator`, `guard_nonzero`, `default_erf_approx`.

## The two tiers

Every module declares its tier in its own docstring, and tier 1 never calls
tier 2. Where both make sense the library ships both, cross-referenced: a
fixed-iteration `newton` and a converge-to-tolerance `newton_tol` are
siblings rather than replacements.

| Tier | Rule | Who |
|---|---|---|
| **Tier 1** | Fixed iteration count, no per-lane branching — therefore launchable inside a GPU thread and usable at every conformer | The conformers, the tensor engine, `special`, `linalg`, `interpolate`, `fft`, `signal`, `optimize.solve`, `integrate`'s fixed-node quadrature and fixed-step ODE steps |
| **Tier 2** | Free to loop or branch on data; `Plain`-only, host-side | `ops`, `elementwise`, `logic`, `sorting`, `io`, `stats`'s tensor reductions, `optimize`'s converge-to-tolerance minimizers, `integrate`'s adaptive `quad`/`solve_ivp` |

## `numax.core` — the tensor engine

[`numax/core/tensor.mojo`](../numax/core/tensor.mojo) drives a `FloatLike`
kernel over a MAX `TileTensor`, CPU or GPU, chosen by one `gpu: Bool`
parameter. Shapes may be comptime or runtime: `map`/`reduce` have two
overloads under one name, picked by `where` clauses that are exact negations.
The static path can be launched on a GPU; the runtime one is CPU-only, since a
launch needs the extent in the type. There is no second tensor type — both are
`TileTensor`.

| Surface | Where |
|---|---|
| `map[step, width, gpu]` — walk one tensor into another at native SIMD width; `gpu=True` is the same source launched per element | [`gaussian.mojo`](../examples/basic/gaussian.mojo), [`gaussian_gpu.mojo`](../examples/advanced/gaussian_gpu.mojo) |
| `map_to`, `zip_to` — write into a caller-supplied output, and elementwise combine of two inputs | [`test_tensor.mojo`](../tests/core/test_tensor.mojo) |
| `map_threaded` — the same `step` spread across cores via `max.algorithm.elementwise` | [`bench_elementwise.mojo`](../bench/bench_elementwise.mojo) |
| `reduce[combine]`, `reduce_block_gpu` — whole-tensor folds, host and device | [`test_tensor_reduce.mojo`](../tests/core/test_tensor_reduce.mojo) |
| `reduce_rows`, `reduce_axis` — per-row and per-axis folds | [`softmax.mojo`](../examples/intermediate/softmax.mojo) |
| `broadcast_op_rows`, `broadcast_op_axis` — broadcast a folded result back along an axis | [`softmax.mojo`](../examples/intermediate/softmax.mojo) |
| `add_op`, `max_op`, `add_combine`, `max_combine`, `add_step`, `mul_step` — the ready-made steps and combiners | [`core/tensor.mojo`](../numax/core/tensor.mojo) |
| Composition: two `step`s fused into one closes both passes into a single walk — worth 1.99x on the GPU at every size | [`bench_fusion.mojo`](../bench/bench_fusion.mojo) |

`Tensor` ([`core/array.mojo`](../numax/core/array.mojo)) adds ownership only:
it owns a MAX `DeviceBuffer`, so the `DeviceContext` passed to a factory
decides host or device memory, and `.view()` yields the same `TileTensor`
either way. Reads go through `to_host()`/`copy_from_host()`, never
`DeviceBuffer.unsafe_ptr()`, which on CUDA returns a *device* pointer that
segfaults a host read.

## `numax.core` — arrays and the NumPy-named surface

| Area | Surface | Where |
|---|---|---|
| Creation | `zeros`, `ones`, `full`, `empty`, `eye`, `identity`, `arange`, `linspace`, `logspace`, `geomspace`, `meshgrid`, `copy`, and the `*_like` forms (`zeros_like`, `ones_like`, `full_like`, `empty_like`) — every one takes its `DeviceContext` last and optional, so `zeros[f32, 2, 3]()` allocates on the host and `zeros[f32, 2, 3](gpu)` on a device | [`core/array.mojo`](../numax/core/array.mojo), [`array_creation.mojo`](../examples/basic/array_creation.mojo) |
| Creation at a computed shape | `zeros_dyn`, `ones_dyn`, `full_dyn`, `empty_dyn` — rank in the type, extents as arguments (`zeros_dyn[f32, 2](r, c)`) — and `asarray`, which takes a `List` and is as long as the list is | [`core/array.mojo`](../numax/core/array.mojo) |
| Manipulation | `reshape`, `ravel`, `transpose`, `squeeze`, `flip`, `stack`, `vstack`, `hstack`, `concatenate`, `split` | [`core/array.mojo`](../numax/core/array.mojo) |
| Manipulation at a computed shape | `reshape_dyn`, `slice` (basic slicing at any rank), `broadcast_to` (NumPy's rules, right-aligned), `concatenate_dyn`, `split_dyn`, `stack_dyn` — all copy into compact storage rather than returning a view, since MAX has no stride-0 broadcast view and a view would borrow from a tensor these do not own | [`core/array.mojo`](../numax/core/array.mojo) |
| Indexing | `a[i]` flat on any rank, `a[r, c]` on a rank-2 tensor; `.view()[i, j, k]` is the general form. On a GPU each access stages its own host mapping — take one `to_host()` and index that instead | [`core/array.mojo`](../numax/core/array.mojo) |
| Printing | `print(a)` — `Tensor` conforms to `Writable`; `a.format(precision=8, threshold=..., edge_items=...)` is the same output with the defaults overridden | [`core/array.mojo`](../numax/core/array.mojo) |
| Conversion | `to_array`, `to_tensor` — the seam between `Tensor` (shape and device) and `Array[T, n]` (the `FloatLike` conformer layer `numax.linalg`, `numax.signal` and `numax.interpolate` take). Lifting works at any conformer; lowering is `Plain`-only, since `FloatLike` can build a value from a `Float64` but not read one back | [`core/array.mojo`](../numax/core/array.mojo), [`npy_to_cholesky.mojo`](../examples/intermediate/npy_to_cholesky.mojo) |
| Matrix builders | `diag`, `diagflat`, `diagonal`, `tri`, `tril`, `triu`, `vander`, `pad` | [`core/array.mojo`](../numax/core/array.mojo) |
| Arithmetic and operators | `add`, `subtract`, `multiply`, `divide`, `power`, `mod`, `floor_divide`, `negative`, `invert`, `astype` — tensor-tensor and tensor-scalar; `astype` is explicit because there is no dtype promotion | [`core/ops.mojo`](../numax/core/ops.mojo) |
| Elementwise math | `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `sqrt`, `rsqrt`, `cbrt`, `abs`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `floor`, `ceil`, `trunc`, `round`, `copysign`, `arcsin`, `arccos`, `arctan`, `arctan2`, `arcsinh`, `arccosh`, `arctanh`, `hypot`, `maximum`, `minimum`, `clip`, `remainder`, `diff`, `gradient` | [`core/elementwise.mojo`](../numax/core/elementwise.mojo) |
| Comparison and logic | `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`, `isclose`, `allclose`, `array_equal`, `isnan`, `isinf`, `isfinite`, `isposinf`, `isneginf`, `logical_and`, `logical_or`, `logical_not`, `logical_xor`, `all`, `any` — truth is a `Static[DType.bool]`, and `select`/`extract` take exactly that, so comparisons compose straight into a mask | [`core/logic.mojo`](../numax/core/logic.mojo) |
| Sorting, searching, masking | `sort`, `argsort`, `searchsorted`, `unique`, `nonzero`, `count_nonzero`, `all_nonzero`, `any_nonzero`, `extract`, `select`, `take`, `top_k` — `take` consumes what `nonzero` and `argsort` return, so `take(a, argsort(a))` is the sorted copy | [`core/sorting.mojo`](../numax/core/sorting.mojo) |

`argsort` routes into MAX's `nn.argsort`, `top_k` into `nn.top_k` (the one
name here with a device path, since MAX's kernel has one), and
`argmin`/`argmax` into
`nn.argmaxmin`; the rest walk a host copy, where `std.builtin.sort` is the
better route.

Three names differ from NumPy's, and only where Mojo leaves no choice:

| NumPy | numax | Why |
|---|---|---|
| `var` | `variance` | `var` is a keyword — it introduces a declaration |
| `std` | `stddev` | `std` is the standard library's package name, always in scope; the compiler rejects a `def std` outright |
| `where` | `select` | `where` introduces constraint clauses, and `mojo format` cannot parse it as an identifier at all |

Every other name a builtin would have collided with — `abs`, `all`, `any`,
`min`, `max`, `sum`, `prod`, `round` — keeps NumPy's spelling. Defining one
of those *replaces* the builtin for the importing file rather than
overloading it, which is the same trade `from numpy import abs` makes in
Python, and it is the caller's to make: `numax.prelude` leaves them out so
a star import cannot make it silently.

## `numax.special`

Every function here is tier 1 — fixed iteration count, GPU-launchable, and
differentiable or extra-precise through whichever conformer instantiates it.

| Module | Surface |
|---|---|
| [`erf`](../numax/special/erf.mojo) | `erf`, `erfc`, `erfinv`, `erfcinv` -- the inverses are a two-region guess plus three Newton steps against the trait's own `erf`/`erfc`, so every conformer's inverse is consistent with its forward function |
| [`gamma`](../numax/special/gamma.mojo) | `gamma`, `lgamma`, `digamma`, `gammainc`, `gammaincc` |
| [`beta`](../numax/special/beta.mojo) | `beta`, `betainc`, `betaincc` |
| [`bessel`](../numax/special/bessel.mojo) | `j0`, `j1`, `y0`, `y1` |
| [`elliptic`](../numax/special/elliptic.mojo) | `elliptic_k`, `elliptic_e` |
| [`lambertw`](../numax/special/lambertw.mojo) | `lambertw`, `lambertw_m1` |
| [`legendre`](../numax/special/legendre.mojo) | `legendre_p` |
| [`orthopoly`](../numax/special/orthopoly.mojo) | `chebyshev_t`, `chebyshev_u`, `hermite_h`, `laguerre_l` |
| [`activations`](../numax/special/activations.mojo) | `gaussian`, `sigmoid`, `swish`, `tanh`, `relu`, `leaky_relu`, `gelu`, `softmax` |

> Run it: `pixi run example-special-functions` ·
> [`special_functions.mojo`](../examples/intermediate/special_functions.mojo)

MAX's scalar `gamma`/`lgamma`/`j0`/`j1`/`y0`/`y1` exist but are CPU-only
libm — compiling one into a GPU kernel fails — which is why numax keeps its
own GPU-launchable versions.

## `numax.linalg`

Two tiers sharing one set of names, one tier per import.
`numax.linalg` is the `Tensor` tier; `numax.linalg.array` is the other one.

The `Array[T, n*n]` tier is comptime-sized and register-resident, not heap
allocated. That is what makes `cholesky` differentiable at `Dual` and
launchable inside a GPU thread, and it is the right shape for the small
matrices that appear *inside* a per-element kernel.

The `Tensor` tier goes through MAX and is the one to use past roughly 8x8
(see [performance.md](performance.md)). It is `dtype`-monomorphic, so no
conformer passes through it — which is why the `Array` tier exists beside
it rather than being replaced by it. `to_tensor`/`to_array` cross over.

The flat surface is the `Tensor` tier alone, so `from numax import solve`
means one thing. A file wanting both tiers of a name aliases one of them
(`from numax.linalg.array import cholesky as chol_a`), which is what
Mojo's one-owning-module-per-name rule costs here.

| Area | Surface — over `Tensor` |
|---|---|
| Products | `matmul` (comptime and run-time shapes), `matvec`, `batched_matmul` — MAX's `linalg.matmul`/`bmm` outright, so they inherit its Apple/NVIDIA/AMD/vendor-BLAS dispatch and numax names no architecture. `inner` is that same kernel under its `transpose_b` parameter, so a Gram matrix costs no materialized transpose; `matrix_power` is a squaring chain of it; `kron` is an `elementwise` map, MAX shipping no equivalent |
| Factorizations | `cholesky`, `lu_factor` (returning a reusable `TensorLU` with `solve`/`det`), `qr_factor` (returning a `TensorQR` with `r`/`q`/`apply_q_transpose`/`solve`), `solve` — MAX ships no factorization on `TileTensor`, so these are numax's, written blocked so the `O(n^3)` trailing update is a matrix product and goes back to `linalg.matmul`. All of them are device-resident: the matrix is copied in and the answer out, and the host touches nothing in between. The panel steps are [`linalg/panel.mojo`](../numax/linalg/panel.mojo) kernels and the trailing update is fused into `matmul`'s epilogue. `TensorLU` and `TensorQR` hold their factors in device memory and carry `gpu` in their type, so solving a device factorization from host code is a compile error |
| Solves | `solve_triangular` (`upper`/`unit`/`trans`, vector or matrix right-hand side), `cholesky_solve`, `TensorLU.solve` (vector or matrix), `inverse`, `det`, `slogdet` — the last for when the determinant itself leaves `float64`'s range, which it does for ordinary matrices long before the answer stops mattering. MAX ships no triangular solve at any size — `trsm` lives only in its private cuBLAS/rocBLAS bindings, which are per-vendor and so out of bounds — so these are numax's, blocked and device-resident like the factorizations. With one right-hand side the update between diagonal blocks is a `gemv`; with several it is a GEMM and goes to `linalg.matmul`, which is why `inverse` solves against the whole identity at once rather than column by column |
| Least squares | `lstsq`, and `TensorQR.solve` underneath it — `Q^T b` through the reflectors and then a back substitution against `R`, so `Q` is never formed. The `Array` tier's `lstsq` in the `Tensor` tier's shape, and the only overdetermined solve either tier has. `lstsq` factors and solves in one call; hold the `TensorQR` instead when several right-hand sides share one `A`, since the factorization is the expensive half |
| Matrix functions | `expm` — scaling and squaring with a degree-13 Padé approximant, so the whole algorithm is `linalg.matmul` plus one `lu_factor` solve and the matrix never leaves its device. Tier 2, because the squaring count comes from `norm(a)`. `logm`, `sqrtm` for a general matrix, `funm` and the rest are deferred behind a Schur decomposition numax does not have, and `numax/linalg/matfuncs.mojo` says what unblocks them |
| Banded and Toeplitz solves | `solve_banded` (partial pivoting, so the working band widens to `l + u`), `solveh_banded`, `cholesky_banded`, `cho_solve_banded`, `solve_toeplitz` (Levinson-Durbin, `O(n²)` and never materializing the matrix), `solve_circulant` (three FFTs and a division, `O(n log n)`, power-of-two `n`). SciPy's diagonal-ordered `ab` storage verbatim. **Tier 2, `Plain`-only, host-side and staying that way** — a banded elimination is `O(n · bandwidth²)` over `n` sequential column steps with no GEMM to hand anything to, so this is an entry surface rather than a faster path |
| Structured constructors | `toeplitz` (two-argument and symmetric), `hankel`, `circulant`, `companion`, `hilbert`, `block_diag`, `khatri_rao`, `convolution_matrix` — `scipy.linalg`'s `_special_matrices`, each one an `elementwise` map from an index rule. `companion`'s eigenvalues are its polynomial's roots, which is the bridge to `numax.linalg.array.eigvals`; `convolution_matrix` is the matrix `C` with `C @ v == convolve(a, v)`, which is what deconvolution needs and a function cannot be |
| Scalars | `norm` — a matrix overload (`fro`/`1`/`inf`) and a vector one (`2`/`1`/`inf`/`neg_inf`), selected by rank because `ord` does not mean the same thing on the two — plus `trace`, `dot`, `nrm2`, `asum` — MAX names none of them, and all five are its reductions underneath: `ReduceSum` under the `rowwise` scaffolder, with the square, the magnitude or the multiply fused into the per-tile transform. The Frobenius norm is one launch over the matrix read as the vector it already is; the induced norms are three (magnitudes, `sum_axis`, `max_axis`); `trace` is two (diagonal gather, fold) |
| Reductions | `sytrd` — a symmetric matrix to tridiagonal form, device-resident, returning a reusable `TensorTridiagonal` with `.q()`. Over half an `eigh`'s arithmetic and the half with a GEMM in it: four launches per column, the symmetric rank-two update issued as one `[v \| w] @ [w \| v]^T` product under `transpose_b=True` |
| Symmetric eigen | `eigvalsh` and `eigh`, returning a `TensorEigh` of ascending values and orthonormal column vectors. `sytrd` reduces device-resident, implicit QL sweeps the two diagonals on the host (tier 2, declared), and `eigh`'s vectors come back through one `matmul` against the reduction's `Q`. `eigvalsh` is `O(n^2)` past the reduction; `eigh` accumulates rotations at `O(n^3)` in scalar host code, the one named ceiling, with divide-and-conquer as the upgrade |
| SVD | `svdvals` and `svd`, returning a `TensorSVD` of `u`, descending `s` and `v` (columns; SciPy's `Vh` is `transpose(v)`), for any `m >= n`. `gebrd` reduces to bidiagonal form device-resident with every reflector through `matmul`, and the singular values are the eigenvalues of the Golub-Kahan tridiagonal, which the same implicit-QL sweep `eigh` uses diagonalizes -- so the SVD adds no new numerics, only a doubling of the host sweep at `vectors=True`, the named ceiling |
| SVD dependents | `pinv` (`V diag(1/s) U^T`, one `inner` product after the SVD, NumPy's `rcond`), `cond` (the 2-norm ratio, `inf` when singular), `matrix_rank` (NumPy's default tolerance, or an explicit one), and `lstsq[method="svd"]`, which returns the minimum-norm solution a rank-deficient system has where the `"qr"` default cannot |
| Not yet | `eigvals` over `Tensor`, which waits on a Hessenberg reduction and its own sweep — a sweep over a two-wide band, looping to a tolerance and deflating on a test of the data, with no GEMM to route to and tier 2 by definition. `numax/linalg/__init__.mojo` records which part of that is negligible (the eigenvalue-only sweep, `O(n^2)`) and which is not. MAX's `qr_factorization` is on the older `LayoutTensor`, which numax denies rather than bridges, so `qr_factor` is numax's own |

Every `Tensor` entry point takes `gpu: Bool` (which picks MAX's `target`)
and the factorizations take a `block` size; `block=n` recovers the
unblocked algorithm, which is what the tests pin the blocked path against.

| Area | Surface — over `Array[T, n*n]`, from `numax.linalg.array` |
|---|---|
| Factorizations | `cholesky`, `lu`, `qr`, `eigh`, `eigvals`, `svd`, and the values-only `eigvalsh`/`svdvals` |
| Solves | `solve`, `lstsq`, `cholesky_solve`, `tridiagonal_solve`, `forward_substitution`, `back_substitution` |
| Inverses | `inverse`, `pinv` |
| Scalars | `det`, `slogdet`, `trace`, `cond`, `slogdet_cholesky`, `matrix_rank` — the last returning `T` rather than `Int`, since one call at `Plain[dtype, w]` is `w` matrices whose ranks need not agree |
| Pivoted (tier 2) | `lu_factor` and the `PivotedLU` it returns, whose `solve` and `det` get the answers the unpivoted routines cannot |
| Norms | `norm` — `ord=fro` (default), `1` or `inf`, per `numpy.linalg.norm` — and `nrm2` |
| Products | `dot`, `outer`, `matvec`, `matmul`, `inner`, `kron`, `matrix_power` — the last three square-only, where the `Tensor` tier takes independent extents |
| BLAS-1 | `dot`, `nrm2`, `asum`, `axpy`, `outer`, at both tiers. MAX names no BLAS-1, so the `Tensor` overloads are built from what it does ship -- `ReduceSum` over `rowwise` for the three reductions, `elementwise` for the two maps, `gpu: Bool` picking the target. The `Array` overloads stay because no BLAS anywhere is generic over its scalar type, and only they run at `Dual` or `Compensated` |

All under [`numax/linalg/`](../numax/linalg/), one module per operation
family the way `scipy.linalg` splits `_decomp_lu`, `_decomp_cholesky` and
`_basic` behind a flat public namespace: `blas`, `triangular`, `cholesky`,
`lu`, `qr`, `basic`, `misc`, plus `panel` for the tile kernels the
factorizations step with. [`array/`](../numax/linalg/array/) mirrors that
split for the other tier and adds `eigen`, which is `Array`-only. Mojo
wants a single owning module per name, so a name is defined once per tier
and never twice within one. Every function's docstring records its own
error behaviour and which MAX kernel, if any, it delegates to; every
module's docstring records the tier and the MAX disposition for the
family.

The `Array` tier is tier 1 except the pivoted row: choosing a pivot by
magnitude is a data-dependent branch, so `PivotedLU` gives up the GPU and
the generic `T` in exchange for factoring matrices `lu` cannot start on.
The whole `Tensor` tier is tier 2 in the sense that it is host-orchestrated
and `dtype`-monomorphic, not in the sense of being CPU-bound — `gpu=True`
runs MAX's GPU kernels.

## `numax.optimize`

Two tiers, one import each. `numax.optimize` is the `Tensor` one and holds
the nonlinear fits; `numax.optimize.array` holds everything that works on a
handful of scalars, split in turn by whether the iteration count is known up
front.

| Surface — over `Tensor`, from `numax.optimize` | Tier | Where |
|---|---|---|
| `minimize` — `scipy.optimize.minimize`, `method="bfgs"` (an `n × n` inverse Hessian kept on the device) or `"cg"` (one direction vector), returning `TensorMinimizeResult` | 2 | [`optimize/minimize.mojo`](../numax/optimize/minimize.mojo) |
| `least_squares`, `curve_fit` — Levenberg-Marquardt, the damped step through `numax.linalg.lstsq`'s blocked device-resident QR rather than the normal equations, returning `TensorFitResult` | 2 | [`optimize/least_squares.mojo`](../numax/optimize/least_squares.mojo) |

This tier takes the derivative as an argument — `jac` for `minimize`,
`jacobian` for the fits — which the `Array` tier does not. A `Tensor` is
`dtype`-monomorphic, so no `Gradient` fits in one, and a finite difference
would be a silent accuracy regression against the sibling of the same name.
It earns its keep when the vectors are long: the driver's bookkeeping is
host-side and `O(n)`, while BFGS's rank-two update and the fits' QR are
neither.

`"nelder-mead"` is deliberately not a `Tensor` method. Its simplex is
`n + 1` points of `n` entries compared every iteration, which is the shape a
`Tensor` tier exists to not be; it stays in `numax.optimize.array`.

| Surface — over `Array[T, n]`, from `numax.optimize.array` | Tier | Where |
|---|---|---|
| `newton`, `halley`, `bisection` — fixed number of steps, no data-dependent branching | 1 | [`optimize/array/solve.mojo`](../numax/optimize/array/solve.mojo) |
| `root_scalar` — `scipy.optimize.root_scalar`, dispatching on `method=` to the five below | 2 | [`optimize/array/optimize.mojo`](../numax/optimize/array/optimize.mojo) |
| `root` — `scipy.optimize.root` for a square vector system, `method="lm"` over `least_squares` | 2 | [`optimize/array/optimize.mojo`](../numax/optimize/array/optimize.mojo) |
| `brentq`, `bisect_tol` — bracketed scalar root finding to a tolerance, returning `OptimizeResult` | 2 | [`optimize/array/optimize.mojo`](../numax/optimize/array/optimize.mojo) |
| `newton_tol`, `halley_tol` — from a single guess, with `f′` and `f″` exact from `Dual` and `Dual[Dual]` | 2 | [`optimize/array/optimize.mojo`](../numax/optimize/array/optimize.mojo) |
| `secant` — the one root finder here that uses no derivative at all, for objectives whose derivative lies | 2 | [`optimize/array/optimize.mojo`](../numax/optimize/array/optimize.mojo) |
| `minimize_scalar` — `scipy.optimize.minimize_scalar`, dispatching on `method=` to the three below | 2 | [`optimize/array/optimize.mojo`](../numax/optimize/array/optimize.mojo) |
| `brent`, `golden` — one-variable minimization from a downhill *direction*, which the search expands into a bracket | 2 | [`optimize/array/optimize.mojo`](../numax/optimize/array/optimize.mojo) |
| `fminbound` — the same engine constrained to `[lower, upper]`, which the answer may not leave | 2 | [`optimize/array/optimize.mojo`](../numax/optimize/array/optimize.mojo) |
| `minimize` — `scipy.optimize.minimize`, dispatching on `method=` to the three below | 2 | [`optimize/array/optimize.mojo`](../numax/optimize/array/optimize.mojo) |
| `bfgs` — quasi-Newton minimization to a tolerance, returning `MinimizeResult` | 2 | [`optimize/array/optimize.mojo`](../numax/optimize/array/optimize.mojo) |
| `cg` — Polak-Ribière conjugate gradients under a strong-Wolfe line search; one direction vector rather than an `n × n` inverse Hessian | 2 | [`optimize/array/optimize.mojo`](../numax/optimize/array/optimize.mojo) |
| `least_squares`, `curve_fit` — Levenberg-Marquardt, the second with the data as a runtime argument | 2 | [`optimize/array/optimize.mojo`](../numax/optimize/array/optimize.mojo) |
| `nelder_mead` — derivative-free simplex, for objectives whose gradient exists but should not be trusted | 2 | [`optimize/array/optimize.mojo`](../numax/optimize/array/optimize.mojo) |

The objective is an ordinary `FloatLike` kernel, so `bfgs` evaluates it at
`Gradient` and gets every partial derivative *exactly* — there is no `jac`
argument to pass. `least_squares` and `curve_fit` get the entire Jacobian
from one call per iteration for the same reason, where a forward difference
would cost `n_params + 1`. A central difference cannot beat about `ε^(2/3)` relative
accuracy; AD has neither the truncation nor the cancellation term.

`minimize`, `minimize_scalar` and `root_scalar` are the SciPy-shaped entry
points and spell their methods the way SciPy does — `minimize[2, rosenbrock, method="nelder-mead"](x0)`. `tol` and
`max_iter` default *per method* rather than globally, because the three do
not measure the same thing: `bfgs` and `cg` stop on `max|∇f| < 1e-8`,
`nelder_mead` on a simplex spread below `1e-10`; `brent` and `golden` use
`sqrt(eps)`, the floor a quadratic minimum puts on locating `x` at all,
while `fminbound` keeps SciPy's looser `1e-5` for that method.

`root_scalar` splits its arguments the way SciPy does and refuses to blur
them: a `bracket` is an interval the search may not leave, an `x0` is a
guess it may. A bracketing method handed only `x0`, or a guess method handed
a `bracket`, raises rather than quietly promising a guarantee it does not
have.

An unrecognized method raises rather than failing to compile, which is a Mojo limitation and not a
choice — a `where` clause cannot compare `StaticString`s, and an untaken
`comptime if` branch is still constraint-checked, so neither the signature
nor the dead branch can reject the value. What it will never do is fall
through to a different algorithm.

> Run it: `pixi run example-optimize` ·
> [`optimize.mojo`](../examples/advanced/optimize.mojo)

## `numax.integrate`

| Surface | Tier | Where |
|---|---|---|
| `trapezoid`, `simpson`, `cumulative_trapezoid` over sampled `Tensor`s — `trapezoid(y, dx)`, `simpson(y, x)`, exactly `scipy.integrate`'s signatures, SciPy's even-count Simpson correction included | 2 | [`integrate/quadrature.mojo`](../numax/integrate/quadrature.mojo) |
| `gauss_legendre`, `simpson`, `trapezoid` over a `FloatLike` function — fixed node count; the Gauss-Legendre nodes are Legendre roots found by numax's own Newton solver | 1 | [`integrate/array/quadrature.mojo`](../numax/integrate/array/quadrature.mojo) (`numax.integrate.array`) |
| `rk4_system`, `dopri5`, `dopri5_step`, `solve_ivp` over a `Tensor` state — every stage an `elementwise` launch on the state's device, `gpu=True` on the accelerator; `solve_ivp` shares the scalar controller and takes the same steps at `n == 1` | 2 | [`integrate/ode.mojo`](../numax/integrate/ode.mojo) |
| `rk4`, `rk4_system`, `dopri5_step`, `dopri5_with_error`, `dopri5` over a `FloatLike` right-hand side — fixed-step, one state or `n` register-resident components, with the embedded error estimate | 1 | [`integrate/array/ode.mojo`](../numax/integrate/array/ode.mojo) (`numax.integrate.array`) |
| `quad`, `quad_vec` — adaptive quadrature to a tolerance, returning `QuadResult` | 2 | [`integrate/integrate.mojo`](../numax/integrate/integrate.mojo) |
| `solve_ivp` (scalar state) — adaptive step-size control, returning `IVPResult` | 2 | [`integrate/integrate.mojo`](../numax/integrate/integrate.mojo) |
| `solve_ivp_stiff` — A-stable implicit trapezoid, for the problem whose step size stability rather than accuracy decides | 2 | [`integrate/integrate.mojo`](../numax/integrate/integrate.mojo) |

Because the integrand is a `FloatLike` kernel, differentiating through an
integral is just calling the same quadrature at `Dual`. 1024 ODE trajectories
run one per GPU thread with solution sensitivities from the same integrator.

> Run it: `pixi run example-quadrature` ·
> [`quadrature.mojo`](../examples/intermediate/quadrature.mojo) ·
> `pixi run example-ode` (needs a GPU) ·
> [`ode.mojo`](../examples/advanced/ode.mojo)

## `numax.interpolate`

| Surface | Where |
|---|---|
| `horner` — polynomial evaluation | [`interpolate/interp.mojo`](../numax/interpolate/interp.mojo) |
| `CubicSpline`, `Chebyshev` — the `scipy.interpolate`-shaped objects: built once, called many times, `__call__` evaluates. `Chebyshev[T, n].fit[f](a, b)` fits and keeps the coefficients | [`interpolate/interp.mojo`](../numax/interpolate/interp.mojo) |
| `cubic_spline_moments`, `cubic_spline_eval` — natural cubic splines, over `numax.linalg`'s tridiagonal solve. The pair the `CubicSpline` object wraps, kept public because they are what a GPU-launchable kernel calls | [`interpolate/interp.mojo`](../numax/interpolate/interp.mojo) |
| `chebyshev_fit`, `chebyshev_eval` — Chebyshev fit and evaluation | [`interpolate/interp.mojo`](../numax/interpolate/interp.mojo) |

Tier 1, 1-D.

## `numax.fft`

Radix-2 Cooley-Tukey, power-of-two by construction. MAX ships **no forward
FFT at all** — its only transform is `nn.irfft`, inverse-only,
last-axis-only and NVIDIA-only over the private `_cufft` — so there is
nothing to route to and both tiers here are numax's own.

Two tiers, one import each, the same split `numax.linalg` makes. The size
is what chooses: an `Array` holds its data in registers and a `Tensor` does
not.

| Surface — over `Tensor`, from `numax.fft` | Where |
|---|---|
| `fft`, `ifft` — complex forward and inverse, travelling as a `Spectrum` real/imaginary pair, since a `dtype`-monomorphic tensor cannot hold a `Complex` | [`fft/fft.mojo`](../numax/fft/fft.mojo) |
| `rfft` — real input, half spectrum | [`fft/fft.mojo`](../numax/fft/fft.mojo) |
| `fftfreq`, `rfftfreq` — frequency grids | [`fft/fft.mojo`](../numax/fft/fft.mojo) |

Tier 2: the stage loop is on the host and each of the `log2(n) + 1` stages
is a device kernel, so the data stays device-resident between them but
nothing here runs *inside* a kernel body. `gpu=True` is `float32` on Apple
silicon, which is Metal's limit on `double` rather than this module's.

| Surface — over `Array[Complex[T], n]`, from `numax.fft.array` | Where |
|---|---|
| `fft`, `ifft` — complex forward and inverse | [`fft/array/fft.mojo`](../numax/fft/array/fft.mojo) |
| `rfft`, `irfft` — real input, half spectrum | [`fft/array/fft.mojo`](../numax/fft/array/fft.mojo) |
| `fft2`, `ifft2` — square 2-D transforms | [`fft/array/fft.mojo`](../numax/fft/array/fft.mojo) |
| `fftfreq`, `rfftfreq`, `fftshift` — frequency grids and centring | [`fft/array/fft.mojo`](../numax/fft/array/fft.mojo) |
| `circular_convolve` — convolution in the transform domain | [`fft/array/fft.mojo`](../numax/fft/array/fft.mojo) |

Tier 1, and the only tier that differentiates: the butterfly is `Complex`
arithmetic over `FloatLike`, so `fft` at `Complex[Dual[Plain]]` returns the
transform and its derivative with no adjoint rule written anywhere.

## `numax.signal`

| Surface | Where |
|---|---|
| `convolve` (`mode=full`, the default, or `same`), `correlate` — direct sums over comptime-sized `Array`s | [`signal/signal.mojo`](../numax/signal/signal.mojo) |
| `hann`, `hamming`, `blackman`, `apply_window` | [`signal/signal.mojo`](../numax/signal/signal.mojo) |
| `lfilter` — the recursive difference equation a convolution cannot express; `firwin` — lowpass taps by the window method | [`signal/signal.mojo`](../numax/signal/signal.mojo) |

Tier 1; `numax.fft.circular_convolve` is the transform-domain route.

## `numax.stats`

| Area | Surface | Where |
|---|---|---|
| Reductions | `sum`, `prod`, `mean`, `median`, `mode`, `min`, `max`, `argmin`, `argmax`, `cumsum`, `cumprod`, `variance`, `stddev` — every one takes a `Tensor` and covers all of it, and `mean`/`variance`/`stddev` fold through MAX's `Welford` monoid on either target under a `gpu: Bool` parameter; `mean`, `variance`, `stddev` and `cumsum` also have a `FloatLike`-generic `List[T]` form | [`stats/statistics.mojo`](../numax/stats/statistics.mojo) |
| Reductions along one axis | `sum[axis=k](a)`, and the same second overload on `prod`, `min`, `max` and `mean`, plus `variance_axis[axis=k](a)` returning mean and variance from one traversal — `numpy.sum(a, axis=k)` and friends, at any rank above 1. The axis is a compile-time parameter and drops from the result, whose remaining extents are run-time values | [`stats/statistics.mojo`](../numax/stats/statistics.mojo) |
| Distributions | Nine `scipy.stats`-shaped namespaces — `norm`, `expon`, `gamma`, `chi2`, `beta`, `t`, `f`, `poisson`, `binom` — each with the eight `scipy.stats` methods: `.pdf` (`.pmf`) and `.logpdf` (`.logpmf`), `.cdf` and `.logcdf`, `.sf` and `.logsf`, `.ppf` and `.isf`. The discrete `.ppf`s return SciPy's smallest integer `k` with `cdf(k) >= p`, by a fixed-count branchless scan capped at a compile-time `max_k`. `.pdf`/`.pmf`, `.cdf` and `.ppf` each also take a `Tensor` with the parameters as scalars -- `norm.cdf(samples, mu, sigma)` -- driven by `map` at SIMD width on the host or one thread per element at `gpu=True`, so a batch of p-values or quantiles never leaves the device. Reached as `numax.stats.norm.cdf(...)`; not re-exported at the root, where `gamma` and `beta` are the special functions | [`stats/distributions.mojo`](../numax/stats/distributions.mojo) |
| Sampling | `uniform`, `normal`, `exponential`, `randint`, `randbool`, `seed`, and `Generator` — a named stream, so two generators built from one seed agree and neither is disturbed by what else touched the global RNG (`numpy.random.Generator`'s shape) | [`stats/random.mojo`](../numax/stats/random.mojo) |

`mean`/`variance`/`stddev`/`cumsum` also have a `FloatLike`-generic form over
`List[T]`, so calling them at `Compensated` matches a float64 reference where
`Plain` drifts — the one place the parity surface and the composable-type spine
meet. The distributions are built on `numax.special`'s incomplete gamma and
beta, so they inherit its accuracy bounds. There is no `Random[FloatLike]`
conformer: sampling is not differentiable, so the trait contract does not fit.

> Run it: `pixi run example-statistics` ·
> [`statistics.mojo`](../examples/intermediate/statistics.mojo) ·
> `pixi run example-random-ensemble` (needs a GPU) ·
> [`random_ensemble.mojo`](../examples/intermediate/random_ensemble.mojo)

## `numax.io`

| Surface | Where |
|---|---|
| `numpy.load` — read a `.npy` file `numpy.save` wrote, straight into a `Tensor`. No Python and no NumPy involved: `.npy` is a self-contained binary format, so this is a header parse plus a payload copy in the default `mojo` + `max` environment | [`io/npy.mojo`](../numax/io/npy.mojo), [`npy_interop.mojo`](../examples/basic/npy_interop.mojo) |
| `numpy.load_dyn` — the same read for a file whose shape you learn from the file: extents go into the layout, so the result is a `Dynamic` at the rank you named | [`io/npy.mojo`](../numax/io/npy.mojo), [`test_npy.mojo`](../tests/io/test_npy.mojo) |
| `numpy.save` — write a `.npy` file `numpy.load` opens. Byte-identical to what `numpy.save` would have written for the same array: NumPy's key order, its `, }` terminator, its 64-byte header alignment | [`io/npy.mojo`](../numax/io/npy.mojo), [`test_npy.mojo`](../tests/io/test_npy.mojo) |
| `nmx.save`, `nmx.load` — numax's own `NMX1` binary format: little-endian, dtype/rank/shape in the header and checked on load. The better choice between numax programs, since it carries the dtype name in full and has no Python literal to parse | [`io/io.mojo`](../numax/io/io.mojo) |


`numpy.load` and `nmx.load` are *typed*: `dtype` and `dims` are compile-time
parameters the caller supplies, matching every other `numax.core.array`
factory, and the load raises if the file disagrees rather than inferring a
shape from it. `numpy.load_dyn[dtype, rank]` is the other way round, for a
file whose extents you do not know in advance: it reads them from the header
into the layout and hands back a `Dynamic`. The rank still has to be right,
since it is the tensor's type.

`numpy.load` rejects, with a message naming the fix: `fortran_order: True`
(column-major, so the payload order is not numax's), a big-endian `descr` like
`'>f4'` (nothing in numax byte-swaps), a `descr` naming a different dtype, and
`.npz` archives, which are zip containers rather than `.npy` files —
`numpy.savez` output has to be unzipped, or re-saved per array with
`numpy.save`, first. `bfloat16` and the float8 formats have no NumPy dtype at
all, so both directions raise for them.

Tier 2, `Plain`-only, host-side.

> Run it: `pixi run example-npy-interop` ·
> [`npy_interop.mojo`](../examples/basic/npy_interop.mojo)

## Where rank stops

Rank is a compile-time variadic and `map`/`reduce` coalesce any contiguous
row-major tensor, but the surface above it is not uniformly rank-generic yet:

- `numax.stats`'s `sum`, `prod`, `min`, `max`, `mean`, `median`, `mode`,
  `argmin`, `argmax`, `cumsum` and `cumprod` each take an `axis=` through a
  second overload of its own name (`sum(a)` folds everything,
  `sum[axis=k](a)` folds one axis), and `numax.core.tensor.reduce_axis`
  folds an arbitrary `combine` the same way.
- `numax.core.sorting` flattens, except `take`/`take_along_axis`, which
  gather along an axis at any rank, and `searchsorted`, which takes a whole
  tensor of queries.
- `transpose` permutes any rank (`transpose(a, 2, 0, 1)`), with `swapaxes`
  and `moveaxis` beside it; `concatenate`/`split`/`stack` join or cut along
  any axis at any rank through a second overload taking `axis`, keeping the
  rank-1 ones for when the result length should stay compile-time;
  `expand_dims`, `roll`, `tile` and `repeat` take an axis at any rank;
  `reshape` targets rank 2 or 3.
- The SciPy-shaped algorithms are fixed-size `Array` kernels: `linalg` is
  matrices, `fft2` a square transform, `rk4_system` an `n`-component state,
  while `quad`, `solve_ivp`, the splines and the distributions are 1-D.
- Binary operations broadcast between two arbitrary shapes under NumPy's
  rules (`broadcast_shapes`), but there is no fancy indexing.
  `broadcast_op_axis` broadcasts only in the direction that pairs with a
  reduction, which is the route that avoids a copy.

An extent that depends on a value has somewhere to live: each dimension of a
`Tensor` is independently compile-time or run-time, so `unique`, `extract` and
`take` return a right-sized result. A transposed or sliced view is walked by
`map_strided` / `reduce_strided`, which address elements through their own
strides; the run-time and strided paths are CPU-only, because a GPU launch
needs the extent in the type.

## Accuracy

Every approximation documents an error bound, and `pixi run accuracy` measures
it against checked-in [mpmath](https://mpmath.org/) references at 50 digits
(`erf`'s A&S 7.1.26 bound of ~1.5e-7 measures 1.38e-07). One caveat: Mojo's
`std.math` `exp`/`log`/`erf` are not correctly rounded at `float64`, so every
function built on them inherits that floor — invisible at `float32`. Details:
[`bench/accuracy/README.md`](../bench/accuracy/README.md).
