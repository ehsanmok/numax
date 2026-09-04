# NumPy/SciPy parity: what numax absorbs, and what it doesn't

For every piece of NumPy/SciPy surface a caller might expect, this records
what numax does about it and why: absorb it, route it to MAX, or leave it out.

## The two axes

A piece is worth absorbing if it satisfies either:

1. **Composable types.** One kernel, several meanings — the `FloatLike` trait
   and its seven conformers. If running a routine at `Dual` or `Compensated`
   gives something no NumPy/SciPy equivalent can, that is a reason on its own.
2. **Parity entry surface, MAX-first.** A NumPy or SciPy caller should find a
   Mojo entry point of comparable shape, and numax should lean on what MAX
   already ships rather than rebuild on bare `SIMD`.

Each candidate is checked in that order: is there a MAX primitive to route to;
if not, does it compose out of what numax already has; if not, is it worth
writing for the entry surface alone.

## What MAX actually ships

Surveyed against `max 26.5`. Import roots are top-level `layout`, `linalg`,
`nn`, `algorithm`, plus `max.algorithm` and `max.gpu` — there is no
`max.linalg`, no `max.kernels`, and no `max.random`.

**Available, so numax routes to it:**

| Area | Names | Root |
|---|---|---|
| Matmul | `matmul`, `batched_matmul`, `gemv`, `grouped_matmul`, vendor cuBLAS/rocBLAS | `linalg` |
| Transpose | `transpose` (n-D), `matrix_band_part` | `linalg` |
| Reductions | `reduce_sum/max/min/product/mean`, `reduce_argmin`, `reduce_argmax` | `algorithm.reductions` |
| Drivers | `elementwise`, `parallelize`, `stencil` | `max.algorithm` |
| Shape ops | `reshape`, `slice`, `concat`, `split`, `tile`, `broadcast`, `pad`, `arange` | `nn` |
| Indexing | `gather`, `scatter_nd`, `index_tensor`, `arg_nonzero` | `nn` |
| Ordering | `argsort` (rank-1), `top_k`; host `sort`/`partition` | `nn`, `std.builtin.sort` |
| GPU | `DeviceContext`, buffers, streams, `GPUInfo` | `max.gpu.host` |
| RNG | `seed`, `rand`, `randn`, Philox `Random`/`NormalRandom` | `std.random` |
| Scalar math | `exp`, `log`, trig, `pow`, `sqrt`, `hypot`, **`erf` `erfc` `gamma` `lgamma` `j0` `j1` `y0` `y1`**, `floor`/`ceil`/`trunc` | `std.math` |

**Not available, so numax writes it:**

- **`scipy.linalg`, almost all of it.** One decomposition ships:
  `qr_factorization` (Householder, CPU-only, scalar loops, on the older
  `LayoutTensor`). No LU, Cholesky, SVD, eig, `solve`, triangular solve,
  inverse, determinant, matrix norm, or BLAS-1. No cuSOLVER bridge.
- **FFT.** Only `nn.irfft`: inverse real, last dimension, NVIDIA-only, a thin
  wrapper over the *private* `_cufft` package. No forward FFT anywhere.
- **Out-of-place tensor arithmetic and explicit broadcast.** `TileTensor` has
  in-place operators only and no `broadcast_to`.
- **Statistics and ordering past the basics.** No value-based or n-D sort, no
  `searchsorted`, no `partition`, no `unique`/`median`/`quantile`/`histogram`.
  `nn.cumsum` is CPU-only.
- **Distributions.** Uniform, normal and Gumbel only.
- **Everything algorithmic.** No quadrature, ODE solvers, optimizers, root
  finders, sparse matrices, polynomials, or signal-sense convolution.

Scalar `gamma`/`lgamma`/`j0`/`j1`/`y0`/`y1` exist but are **CPU-only libm**:
compiling one into a GPU kernel fails with "libm operations are only available
on CPU targets", which is why numax keeps its own GPU-launchable versions.

## Dispositions

| Area | Home | Notes |
|---|---|---|
| Array creation and manipulation | `numax/core/array.mojo` | `Plain`-only, comptime shape, a thin owner whose `.view()` is a `TileTensor`. `transpose` routes to `linalg.transpose`; `to_array`/`to_tensor` bridge to the `Array[T, n]` conformer layer |
| Elementwise math | `numax/core/elementwise.mojo` | `Plain`-only over `std.math`, rather than growing `FloatLike` by twenty methods across seven conformers |
| Arithmetic and operators | `numax/core/ops.mojo` | Tensor-tensor and tensor-scalar; `astype` is explicit because there is no dtype promotion |
| Comparison and logic | `numax/core/logic.mojo` | Truth is a `Tensor[DType.bool]`, so a comparison composes with `logical_and` |
| Statistics | `numax/stats/statistics.mojo` | `variance`/`stddev`/`cumsum`/`mean` are `FloatLike`-generic — at `Compensated` they match a float64 reference where `Plain` drifts. `argmin`/`argmax` route into `nn.argmaxmin` |
| Sorting, searching, masking | `numax/core/sorting.mojo` | Tier 2. `argsort` routes into `nn.argsort`; the rest walk a host copy, where `std.builtin.sort` is the better route |
| Small dense linalg | `numax/linalg/linalg.mojo` | `FloatLike`-generic over comptime `Array[T, n*n]`. Differentiability is the point; MAX's `matmul` is the call past ~8x8 |
| Root finding and minimization | `numax/optimize/solve.mojo`, `numax/optimize/optimize.mojo` | Fixed-iteration siblings in `solve` (tier 1), converge-to-tolerance in `optimize` (tier 2) |
| Quadrature and ODE | `numax/integrate/quadrature.mojo`, `numax/integrate/ode.mojo`, `numax/integrate/integrate.mojo` | Fixed-node and fixed-step are tier 1; adaptive is tier 2 |
| Transforms and signal | `numax/fft/fft.mojo`, `numax/signal/signal.mojo` | Power-of-two by construction. MAX has no forward FFT to route to |
| Tensor I/O | `numax/io/io.mojo`, `numax/io/npy.mojo` | Two formats: numax's own `NMX1` for numax-to-numax round trips (MAX ships no array I/O at all), and NumPy's `.npy` for interchange -- `numax.io.numpy.save` output is byte-identical to `numpy.save`, and `numax.io.numpy.load` reads `numpy.save` output, with no Python or NumPy dependency since the format is self-contained. `.npz` is out: it is a zip container |
| Random sampling | `numax/stats/random.mojo` | Over `std.random` on the host. No `Random[FloatLike]` conformer: RNG is not differentiable, so the trait contract does not fit |

Three names differ from NumPy's because Mojo will not allow them: `var` is a
keyword and `std` is the standard library's package (hence `variance`,
`stddev`), and `where` introduces constraint clauses (hence `select`).

### Not absorbed

A separate complex array type — `Complex` composes into every kernel already.
A `Backend` trait — the `gpu: Bool` parameter on `numax.core.tensor.map` covers the
same ground with no dispatch. ML primitives — `nn` ships softmax,
normalization, convolution and pooling, tuned, on both backends.

MAX's `nn` versions of `arange`/`reshape`/`concat`/`split` were checked and are
not usable as array functions: `nn.arange` returns one SIMD vector for an
index rather than filling a tensor, `nn.concat` wants a pre-sized output plus a
`DeviceContext` and one layout type across all inputs, and `nn.reshape`
returns a dynamically-laid-out `TileTensor`. They are graph-operator kernels.

## What is still missing

The array object itself. `Tensor` carries its shape at compile time, so
anything whose extent depends on a value — `reshape` to a computed shape, a
boolean mask, a right-sized `unique` — comes back as a full-length result plus
a count. Strided views, general broadcasting, slicing and fancy indexing all
wait on a runtime-shape owner, since a view over a slice is not row-major and
every driver in `numax.core.tensor` requires that it is.

What exists instead: `map`/`reduce` have runtime-shape overloads selected by a
`where` clause that is the exact negation of the static one, so a
`row_major(Coord(...))` tensor walks without `coalesce()` — CPU-only, because
a GPU launch needs the extent in the type.

Also absent, each a decision: sparse matrices, iterative solvers, distributed
execution, and dtype promotion. The first three are a different library's job;
the fourth is a compile error waiting to happen in a language that infers
parameters, so `astype` is explicit.
