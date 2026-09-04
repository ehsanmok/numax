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

They nest — `Complex[Dual[Plain[...]]]`, `Gradient[Dual[...], n]` — so
autodiff, precision and complex arithmetic compose instead of each needing its
own copy of every kernel. `pi`/`e` (and `pi_at`/`e_at`) are available at any
conformer from [`core/constants.mojo`](../numax/core/constants.mojo).

`numax.core.numeric` also exports the branchless helpers every
conformer-generic kernel is built from: `max_of`, `min_of`, `blend`,
`ge_indicator`, `guard_nonzero`, `default_erf_approx`.

## The two tiers

Every subpackage's docstring declares its tier, and tier 1 never calls
tier 2. Labelling each individual module is still in progress: 12 of the 38
modules carry the line today.

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
| Creation | `zeros`, `ones`, `full`, `empty`, `eye`, `identity`, `arange`, `linspace`, `logspace`, `geomspace`, `meshgrid`, `copy`, and the `*_like` forms (`zeros_like`, `ones_like`, `full_like`, `empty_like`) | [`core/array.mojo`](../numax/core/array.mojo), [`array_creation.mojo`](../examples/basic/array_creation.mojo) |
| Manipulation | `reshape`, `ravel`, `transpose`, `squeeze`, `flip`, `stack`, `vstack`, `hstack`, `concatenate`, `split` | [`core/array.mojo`](../numax/core/array.mojo) |
| Matrix builders | `diag`, `diagflat`, `diagonal`, `tri`, `tril`, `triu`, `vander` | [`core/array.mojo`](../numax/core/array.mojo) |
| Arithmetic and operators | `add`, `subtract`, `multiply`, `divide`, `power`, `mod`, `floor_divide`, `negative`, `invert`, `astype` — tensor-tensor and tensor-scalar; `astype` is explicit because there is no dtype promotion | [`core/ops.mojo`](../numax/core/ops.mojo) |
| Elementwise math | `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `sqrt`, `rsqrt`, `cbrt`, `abs`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `floor`, `ceil`, `trunc`, `round`, `copysign`, `arcsin`, `arccos`, `arctan`, `arctan2`, `arcsinh`, `arccosh`, `arctanh`, `hypot`, `maximum`, `minimum`, `clip`, `remainder`, `diff`, `gradient` | [`core/elementwise.mojo`](../numax/core/elementwise.mojo) |
| Comparison and logic | `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`, `isclose`, `allclose`, `array_equal`, `isnan`, `isinf`, `isfinite`, `isposinf`, `isneginf`, `logical_and`, `logical_or`, `logical_not`, `logical_xor`, `all`, `any` — truth is a `Tensor[DType.bool]`, and `select`/`extract` take exactly that, so comparisons compose straight into a mask | [`core/logic.mojo`](../numax/core/logic.mojo) |
| Sorting, searching, masking | `sort`, `argsort`, `searchsorted`, `unique`, `nonzero`, `count_nonzero`, `all_nonzero`, `any_nonzero`, `extract`, `select` | [`core/sorting.mojo`](../numax/core/sorting.mojo) |

`argsort` routes into MAX's `nn.argsort` and `argmin`/`argmax` into
`nn.argmaxmin`; the rest walk a host copy, where `std.builtin.sort` is the
better route. Three names differ from NumPy's because Mojo will not allow
them: `var` is a keyword and `std` is the standard library's package (hence
`variance`, `stddev`), and `where` introduces constraint clauses (hence
`select`).

## `numax.special`

Every function here is tier 1 — fixed iteration count, GPU-launchable, and
differentiable or extra-precise through whichever conformer instantiates it.

| Module | Surface |
|---|---|
| [`erf`](../numax/special/erf.mojo) | `erf`, `erfc` |
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

Matrices are comptime-sized `Array[T, n*n]` in registers, not heap
allocations. That is what makes `cholesky` differentiable at `Dual` and
launchable inside a GPU thread; MAX's own `linalg` is monomorphic in a raw
`dtype`, so no conformer passes through it, and it is the right call past
roughly 8x8.

| Area | Surface |
|---|---|
| Factorizations | `cholesky`, `lu`, `qr`, `eigh`, `svd` |
| Solves | `solve`, `cholesky_solve`, `tridiagonal_solve`, `forward_substitution`, `back_substitution` |
| Inverses | `inverse`, `pinv` |
| Scalars | `det`, `trace`, `cond`, `slogdet_cholesky` |
| Norms | `norm` — `ord=fro` (default), `1` or `inf`, per `numpy.linalg.norm` — and `nrm2` |
| Products | `dot`, `outer`, `matvec`, `matmul` |

All in [`linalg/linalg.mojo`](../numax/linalg/linalg.mojo); every function's
docstring records its own error behaviour and where MAX's kernel takes over.

## `numax.optimize`

Two halves, split by whether the iteration count is known up front.

| Surface | Tier | Where |
|---|---|---|
| `newton`, `halley`, `bisection` — fixed number of steps, no data-dependent branching | 1 | [`optimize/solve.mojo`](../numax/optimize/solve.mojo) |
| `newton_tol`, `brentq` — scalar root finding to a tolerance, returning `OptimizeResult` | 2 | [`optimize/optimize.mojo`](../numax/optimize/optimize.mojo) |
| `bfgs` — quasi-Newton minimization to a tolerance, returning `MinimizeResult` | 2 | [`optimize/optimize.mojo`](../numax/optimize/optimize.mojo) |

The objective is an ordinary `FloatLike` kernel, so `bfgs` evaluates it at
`Gradient` and gets every partial derivative *exactly* — there is no `jac`
argument to pass. A central difference cannot beat about `ε^(2/3)` relative
accuracy; AD has neither the truncation nor the cancellation term.

> Run it: `pixi run example-optimize` ·
> [`optimize.mojo`](../examples/advanced/optimize.mojo)

## `numax.integrate`

| Surface | Tier | Where |
|---|---|---|
| `gauss_legendre`, `simpson`, `trapezoid` — fixed node count; the Gauss-Legendre nodes are Legendre roots found by numax's own Newton solver | 1 | [`integrate/quadrature.mojo`](../numax/integrate/quadrature.mojo) |
| `rk4`, `rk4_system`, `dopri5_step`, `dopri5_with_error`, `dopri5` — fixed-step integration, one state or `n` components, with the embedded error estimate | 1 | [`integrate/ode.mojo`](../numax/integrate/ode.mojo) |
| `quad`, `quad_vec` — adaptive quadrature to a tolerance, returning `QuadResult` | 2 | [`integrate/integrate.mojo`](../numax/integrate/integrate.mojo) |
| `solve_ivp` — adaptive step-size control, returning `IVPResult` | 2 | [`integrate/integrate.mojo`](../numax/integrate/integrate.mojo) |

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
| `cubic_spline_moments`, `cubic_spline_eval` — natural cubic splines, over `numax.linalg`'s tridiagonal solve | [`interpolate/interp.mojo`](../numax/interpolate/interp.mojo) |
| `chebyshev_fit`, `chebyshev_eval` — Chebyshev fit and evaluation | [`interpolate/interp.mojo`](../numax/interpolate/interp.mojo) |

Tier 1, 1-D.

## `numax.fft`

Radix-2 Cooley-Tukey, power-of-two by construction, over `Complex` at any
conformer. MAX ships no forward FFT to route to.

| Surface | Where |
|---|---|
| `fft`, `ifft` — complex forward and inverse | [`fft/fft.mojo`](../numax/fft/fft.mojo) |
| `rfft`, `irfft` — real input, half spectrum | [`fft/fft.mojo`](../numax/fft/fft.mojo) |
| `fft2`, `ifft2` — square 2-D transforms | [`fft/fft.mojo`](../numax/fft/fft.mojo) |
| `fftfreq`, `rfftfreq`, `fftshift` — frequency grids and centring | [`fft/fft.mojo`](../numax/fft/fft.mojo) |
| `circular_convolve` — convolution in the transform domain | [`fft/fft.mojo`](../numax/fft/fft.mojo) |

## `numax.signal`

| Surface | Where |
|---|---|
| `convolve` (`mode=full`, the default, or `same`), `correlate` — direct sums over comptime-sized `Array`s | [`signal/signal.mojo`](../numax/signal/signal.mojo) |
| `hann`, `hamming`, `blackman`, `apply_window` | [`signal/signal.mojo`](../numax/signal/signal.mojo) |

Tier 1; `numax.fft.circular_convolve` is the transform-domain route.

## `numax.stats`

| Area | Surface | Where |
|---|---|---|
| Reductions | `sum`, `prod`, `mean`, `median`, `mode`, `min`, `max`, `argmin`, `argmax`, `cumsum`, `cumprod`, `variance`, `stddev` — every one takes a `Tensor` and covers all of it; `mean`, `variance`, `stddev` and `cumsum` also have a `FloatLike`-generic `List[T]` form | [`stats/statistics.mojo`](../numax/stats/statistics.mojo) |
| Distributions | Nine `scipy.stats`-shaped namespaces — `norm`, `expon`, `gamma`, `chi2`, `beta`, `t`, `f`, `poisson`, `binom` — each with `.pdf` (or `.pmf`), `.cdf` and, where defined, `.ppf`. Reached as `numax.stats.norm.cdf(...)`; not re-exported at the root, where `gamma` and `beta` are the special functions | [`stats/distributions.mojo`](../numax/stats/distributions.mojo) |
| Sampling | `uniform`, `normal`, `exponential`, `randint`, `randbool`, `seed` | [`stats/random.mojo`](../numax/stats/random.mojo) |

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
| `numpy.save` — write a `.npy` file `numpy.load` opens. Byte-identical to what `numpy.save` would have written for the same array: NumPy's key order, its `, }` terminator, its 64-byte header alignment | [`io/npy.mojo`](../numax/io/npy.mojo), [`test_npy.mojo`](../tests/io/test_npy.mojo) |
| `nmx.save`, `nmx.load` — numax's own `NMX1` binary format: little-endian, dtype/rank/shape in the header and checked on load. The better choice between numax programs, since it carries the dtype name in full and has no Python literal to parse | [`io/io.mojo`](../numax/io/io.mojo) |
| `print_tensor` — NumPy-style printing, truncating past a threshold | [`io/io.mojo`](../numax/io/io.mojo) |

Both loads are *typed*: `dtype` and `dims` are compile-time parameters the
caller supplies, matching every other `numax.core.array` factory, and the load
raises if the file disagrees rather than inferring a shape from it.

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

- `numax.stats` reduces over every element and has no `axis=`; axis-wise
  folding is `numax.core.tensor.reduce_axis`/`reduce_rows`.
- `numax.core.sorting` flattens.
- `transpose` is 2-D; `concatenate`/`split`/`stack` are rank-1; `reshape`
  targets rank 2 or 3.
- The SciPy-shaped algorithms are fixed-size `Array` kernels: `linalg` is
  matrices, `fft2` a square transform, `rk4_system` an `n`-component state,
  while `quad`, `solve_ivp`, the splines and the distributions are 1-D.
- There is no broadcasting, no strided views, no slicing or fancy indexing.
  Those wait on a runtime-shape owner, since a view over a slice is not
  row-major and every driver in `numax.core.tensor` requires that it is.

Anything whose extent depends on a value — `reshape` to a computed shape, a
boolean mask, a right-sized `unique` — comes back as a full-length result plus
a count.

## Accuracy

Every approximation documents an error bound, and `pixi run accuracy` measures
it against checked-in [mpmath](https://mpmath.org/) references at 50 digits
(`erf`'s A&S 7.1.26 bound of ~1.5e-7 measures 1.38e-07). One caveat: Mojo's
`std.math` `exp`/`log`/`erf` are not correctly rounded at `float64`, so every
function built on them inherits that floor — invisible at `float32`. Details:
[`bench/accuracy/README.md`](../bench/accuracy/README.md).
