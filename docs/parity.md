# NumPy/SciPy parity: what numax absorbs, and what it doesn't

This is the disposition table. For every piece of NumPy/SciPy surface that a
caller might expect, it records what numax does about it and why — absorb it,
route it to MAX, or leave it out.

It lives in `docs/` rather than in `.cursor/rules/` deliberately. The rules
directory is gitignored (`.gitignore`: "local agent config/rules, not shipped
with the repo"), so a design document there is invisible to anyone who clones
the repo — and several shipped modules cite these dispositions from their own
docstrings. What tracked code cites has to be tracked.

## The two axes

A piece is worth absorbing if it satisfies either:

1. **Composable types.** One kernel, several meanings — the `FloatLike` trait
   and its seven conformers (`Plain`, `Dual`, `Compensated`, `Decimal`,
   `Complex`, `Gradient`, `Interval`). If running a routine at `Dual` or
   `Compensated` gives something no NumPy/SciPy equivalent can, that is a
   reason on its own.
2. **Parity entry surface, MAX-first.** A NumPy or SciPy caller should find a
   Mojo entry point of comparable shape, and numax should lean on what MAX
   already ships rather than rebuild on bare `SIMD`.

Each candidate is checked in that order: is there a MAX primitive to route to;
if not, does it compose out of what numax already has; if not, is it worth
writing for the entry surface alone.

## What MAX actually ships

The MAX-first check is only as good as its facts, so they are recorded here
rather than assumed. Surveyed against `max 26.5`. Import roots are top-level
`layout`, `linalg`, `nn`, `algorithm`, plus `max.algorithm` and `max.gpu` —
there is no `max.linalg`, no `max.kernels`, and no `max.random`.

**Available, so numax routes to it rather than reimplementing:**

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

- **`scipy.linalg`, almost all of it.** MAX ships exactly one matrix
  decomposition — `qr_factorization` (Householder, CPU-only, scalar loops, on
  the older `LayoutTensor`). No LU, Cholesky, SVD, eig, `solve`, triangular
  solve, inverse, determinant, or matrix norm of any kind, and no BLAS-1 at
  all (no `dot`, `axpy`, `nrm2`). No cuSOLVER bridge.
- **FFT.** Only `nn.irfft`: inverse real FFT, last dimension, NVIDIA-only, a
  thin wrapper over the *private* `_cufft` FFI package. No forward FFT
  anywhere, no CPU FFT, nothing on AMD or Apple.
- **Out-of-place tensor arithmetic and explicit broadcast.** `TileTensor` has
  in-place operators only (`__iadd__`, `__imul__`, …) and no `broadcast_to`;
  broadcasting exists only implicitly inside binary ops.
- **Statistics and ordering past the basics.** No value-based or n-D sort, no
  `searchsorted`, no `partition`. `nn.cumsum` is CPU-only; no `cumprod`,
  `cummax`, `diff`. No `unique`, `bincount`, `histogram`, `median`, `quantile`,
  `cov`, `corrcoef`.
- **Distributions.** Uniform, normal and Gumbel. Nothing else.
- **Everything algorithmic.** No quadrature, ODE solvers, optimizers, root
  finders, sparse matrices, polynomials, or signal-sense
  `convolve`/`correlate`.

## Dispositions

### Absorbed — array creation and manipulation

Home: [`numax/array.mojo`](../numax/array.mojo). `Plain`-only; no new
`FloatLike` conformers.

`zeros`, `ones`, `full`, `empty`, `eye`, `linspace`, `logspace`, `arange`,
`zeros_like`/`ones_like`/`full_like`/`empty_like`, `reshape`, `ravel`,
`transpose`, `squeeze`, `stack`, `concatenate`, `split`.

Built over MAX's `layout` package as a thin owner whose `.view()` is a
`TileTensor`, not a competing array type. Three shapes here are set by the
language rather than by preference, and each is documented at its definition:
`arange` takes an element count rather than a `stop` (NumPy derives the count
from `stop`, which makes the extent depend on runtime values); `reshape` spells
out rank-2 and rank-3 rather than taking a `*new_dims` pack (a general form
needs to pair two variadic packs in one signature, and the element-count check
has to be provable to a `where` clause); `split` takes one index and returns
two tensors (`numpy.split` returns a variable-length list, making the *number*
of outputs a runtime value).

MAX's `nn` versions of `arange`/`reshape`/`concat`/`split` were checked and are
not usable as array functions: `nn.arange` returns one SIMD vector for a given
index rather than filling a tensor, `nn.concat` wants a pre-sized output plus a
`DeviceContext`, and `nn.reshape` returns a dynamically-laid-out `TileTensor`.
They are graph-operator kernels.

### Absorbed — statistics

Home: [`numax/statistics.mojo`](../numax/statistics.mojo). Composed from
`numax.tensor`'s `reduce`/`reduce_axis`/`reduce_rows` primitives.

`FloatLike`-generic (the axis-1 win): `variance`, `stddev`, `cumsum`, and one
`mean` overload. Long summations lose precision exactly the way `Compensated`
was built to recover, so running these at `Compensated` matches a float64
reference where `Plain` drifts.

`Plain`-only: `mean` (`TileTensor` overload), `median`, `mode`, `argmin`,
`argmax`, `sum`, `prod`, `cumprod`, `min`, `max`. `argmin`/`argmax` route into
`nn.argmaxmin` — numax writes no comparison logic of its own there.

Two names differ from NumPy's because Mojo will not allow them: `var` is a
keyword, and `std` is the standard library's package name. They are `variance`
and `stddev`.

### Absorbed — tensor I/O

Home: [`numax/io.mojo`](../numax/io.mojo). `Plain`-only. `save`, `load`,
`print_tensor` (with `precision`, `threshold`, `edge_items`). Binary format is
numax's own `NMX1`, not `.npy`. `FloatLike` was not built to carry
serialization, so this stays outside the trait entirely.

### Absorbed — random sampling

Home: [`numax/random.mojo`](../numax/random.mojo). `uniform`, `normal`,
`exponential`, `seed`. `Plain`-only, and **no `Random[FloatLike]` conformer on
purpose**: RNG is not mathematically differentiable, so seeding a `Dual`'s
derivative from a random draw has no well-defined meaning. Same scoping shape
as units in Track B — the trait contract does not fit the mathematics.

Host sampling is `std.random`, not MAX. `nn.rand_uniform`/`nn.rand_normal`
exist and were tried: both take their fill logic as an `OutputFn` parameter
bound to `RegisterPassable & ImplicitlyCopyable`, which a closure over a
caller's buffer does not satisfy. That is graph-op fusion machinery, not an
eager host API. GPU-consumed values use `std.random.philox.{Random,
NormalRandom}` directly — per-thread counter-based streams are what a
`map[gpu=True]` body needs, and wrapping them in a numax name would only
rename them.

### Absorbed — small dense linear algebra

Home: [`numax/linalg.mojo`](../numax/linalg.mojo). `FloatLike`-generic and
compile-time-sized: `matvec`, `matmul`, `cholesky`, `lu`,
`forward_substitution`, `back_substitution`, `solve`, `cholesky_solve`, `det`,
`log_det_from_cholesky`, `inverse`, `tridiagonal_solve`, `trace`,
`norm_frobenius`, `norm_1`, `norm_inf`, `qr`.

The payoff is differentiability, not speed. A differentiable Cholesky is what
Gaussian process marginal likelihoods, Kalman updates and multivariate normal
densities all bottom out in, and calling `cholesky` at `Dual` gives gradients
with no adjoint rule written anywhere. MAX's `matmul` and `qr_factorization`
are monomorphic in a raw `dtype`, so a `Dual` cannot pass through them — they
are exactly as differentiable as a BLAS call.

**Route to MAX past N.** Every function here is register-resident, so both
compile time and register pressure grow with `n`. Measured crossover for
`matmul` on an M3 Pro: `n = 8` scalar, `n = 16` against the 4-wide batched
form, ~130× by `n = 64`. See [`performance.md`](performance.md). For the
routines MAX has no counterpart to, the honest recommendation is to build the
large-matrix equivalent from `linalg.qr_factorization`, not to expect a
drop-in.

### Absorbed — root finding and minimization (tier 2)

Home: [`numax/optimize.mojo`](../numax/optimize.mojo). `newton_tol`,
`brentq`, `bfgs`. MAX ships no optimizer of any kind, so there is nothing to
route to.

This is the first tier-2 module: the drivers loop until they converge and
branch on data, which the fixed-iteration invariant forbids for
`FloatLike`-generic code. They are `Plain`-only, host-side, and not
GPU-launchable, and they say so. `numax.solve`'s fixed-iteration `newton`,
`halley` and `bisection` remain tier 1 and are not superseded — each
docstring names its counterpart.

The objective stays an ordinary `FloatLike` kernel, which is the point:
`bfgs` evaluates it at `Gradient[Plain[float64, 1], n_vars]` and gets every
partial derivative exactly, from one call, by the chain rule. No `jac`
argument, no step size, and none of the accuracy a finite difference gives
up — a central difference is capped near `eps**(2/3)` however carefully the
step is chosen. `examples/advanced/optimize.mojo` measures it: best finite
difference ~5e-10, AD exactly 0.

The driver is fixed at float64 rather than parameterized on `dtype`.
Convergence work belongs at the widest precision available, and Mojo will
not accept a struct instantiated with a *function-level* `DType` parameter
as a `FloatLike` type argument, so a per-call dtype would not compile at
all.

### Absorbed — adaptive integration (tier 2)

Home: [`numax/integrate.mojo`](../numax/integrate.mojo). `quad`, `quad_vec`.
MAX ships no quadrature at all.

Tier 2, and a clean illustration of why the split was needed:
`numax.quadrature`'s module docstring had ruled adaptive quadrature out
entirely, because the subdivision pattern is data-dependent. That reasoning
is right for `FloatLike`-generic code and was over-applied to the library as
a whole. `quad` composes out of the tier-1 rule -- each panel is integrated
whole with `gauss_legendre[n]` and again as two halves, and the difference
is the error estimate -- so there is no second quadrature rule to maintain.

On a smooth integrand the fixed rule is still better, and a test asserts it
in both directions so the tier-1 version is not quietly deprecated. On a
Lorentzian spike narrower than the node spacing, the 8-point rule is off by
a factor of six and `quad` is right to 1e-12.

`quad_vec` takes known breakpoints. Bisection can never land exactly on an
irrational kink location, so a feature at 1/3 keeps every straddling panel
inaccurate however small it gets: 26 panels blind, 2 when told.

### Absorbed — transforms and signal processing

Home: [`numax/fft.mojo`](../numax/fft.mojo) and
[`numax/signal.mojo`](../numax/signal.mojo). Both tier 1.

`fft`/`ifft`, `rfft`/`irfft`, `fft2`/`ifft2`, `fftfreq`/`rfftfreq`,
`fftshift`, `circular_convolve`; `convolve`, `convolve_same`, `correlate`,
the `hann`/`hamming`/`blackman` window family, `apply_window`.

MAX has almost nothing here, which makes this less of a parity veneer than
it looks. Its only transform is `nn.irfft` -- inverse real FFT, last
dimension, NVIDIA-only, over the private `_cufft` FFI package. No forward
FFT of any kind, no CPU FFT, nothing on Metal or AMD. **Recommendation:
treat numax's transform as the portable path and `_cufft` as at most an
optional NVIDIA fast path.** A private, single-vendor, inverse-only kernel
is not a foundation to route through, and the register-resident small-`n`
transform -- one that runs *inside* a kernel -- has no MAX counterpart at
all.

`nn.conv` exists but is the neural-network operator: NHWC layouts, filter
packing, batching, stride, dilation. `numpy.convolve` over a
one-dimensional sequence is a different function, and that is what
`numax.signal` provides. The direct `O(m*k)` sum is deliberate at these
sizes; `circular_convolve` is there for callers already in the frequency
domain.

Frequency-axis conventions match NumPy exactly, including `fftfreq`
reporting the Nyquist bin as *negative* -- for even `n` that bin genuinely
aliases, and matching NumPy matters more than picking a side.

### Not absorbed — sorting and searching as `FloatLike` kernels

`sort`, `argsort` and `searchsorted` are not public numax names operating on
`FloatLike`. A comparison sort runs a data-dependent number of comparisons and
branches per element, which the fixed-iteration invariant forbids for
`FloatLike`-generic code: a `Self` may hold a SIMD vector whose lanes disagree
about which branch they want, and there is no per-lane `select` on the trait.

This is a constraint on what numax writes *inside the trait*, not a ban on
sorting. MAX's own `nn.argsort` and `std.builtin.sort` have data-dependent
control flow and are fair game from `Plain`-only code —
`numax.statistics.median` and `mode` use exactly that route.

### Not absorbed — redundant with existing conformers

A separate complex array type (`Complex` composes into every kernel already),
a `Backend` trait (the `gpu: Bool` compile-time parameter on
`numax.tensor.map` covers the same ground with no dispatch), and ML
primitives (`nn` ships softmax, normalization, convolution and pooling,
tuned, on both backends).

### Deferred

Views over runtime-shaped storage are partly addressed:
`numax.tensor.map`/`reduce` now have runtime-shape overloads, selected by a
`where` clause that is the exact negation of the static one, so a
`row_major(Coord(...))` tensor walks without `coalesce()`. Boolean masking,
`where`/`nonzero` and `unique` build on that and are not written yet. Sparse
matrices are out of scope and named here so their absence is a decision.

## Where numax stands against NuMojo

[NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo) is the
other Mojo numerics library, and Track F was scoped by comparison against it.

**numax has, NuMojo does not:** the composable type layer (seven nesting
conformers), the special-function library (`erf`, the Gamma family, Bessel,
Lambert W, elliptic integrals, orthogonal polynomials, the Beta family), the
algorithms layer (`solve`, quadrature, ODE, interpolation, distributions),
differentiable linear algebra and FFT, GPU execution through `map[gpu=True]`
with every conformer running inside one thread, and an accuracy harness
checked against 50-digit mpmath references.

**NuMojo has, numax does not (or has only partly):** a full `NDArray` with
general broadcasting, slicing and printing; general n-dimensional
manipulation; sorting and searching; and a `Backend` trait. The
runtime-shape work above is the first half of closing the array-semantics
gap.
