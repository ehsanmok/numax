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
| Matmul | `matmul`, `batched_matmul`, `gemv`, `grouped_matmul`, vendor cuBLAS/rocBLAS, **Apple Accelerate `cblas_sgemm` on macOS at `float32`** | `linalg` |
| Transpose | `transpose` (n-D, **host only** -- see the defect note below), `matrix_band_part` | `linalg` |
| Reductions | `ReduceSum`/`ReduceMax`/`ReduceMin`/`ReduceProduct`, `MinMax`, `ArgMax`, `ArgMin`, `Welford` (online mean/variance), `OnlineLogSumExp` (flash softmax), driven by the `rowwise` CPU/GPU scaffolder | `algorithm.reduce_op`, `algorithm.rowwise` |
| Drivers | `elementwise`, `parallelize`, `stencil` | `max.algorithm` |
| Shape ops | `reshape`, `slice`, `concat`, `split`, `tile`, `broadcast`, `pad` (three modes on the host, **`constant` only on GPU** -- see the note below), `arange` | `nn`, `nn.pad_gpu` |
| Indexing | `gather`, `scatter_nd`, `index_tensor`, `arg_nonzero` | `nn` |
| Ordering | `argsort` (rank-1), `top_k` (any axis, CPU and GPU); host `sort`/`partition` | `nn`, `std.builtin.sort` |
| ML primitives | `softmax`/`logsoftmax`, `layer_norm`/`group_norm`/`rms_norm`, convolution, pooling | `nn` |
| GPU | `DeviceContext`, buffers, streams, `GPUInfo` | `max.gpu.host` |
| RNG | `seed`, `rand`, `randn`, Philox `Random`/`NormalRandom` | `std.random` |
| Scalar math | `exp`, `log`, trig, `pow`, `sqrt`, `hypot`, **`erf` `erfc` `gamma` `lgamma` `j0` `j1` `y0` `y1`**, `floor`/`ceil`/`trunc` | `std.math` |

The `algorithm` root is a **reduction library, not a function library**, which
is why its row names monoids rather than NumPy-shaped calls. A reduction is
authored as a `ReduceOp` conformer — inlined state, `__init__` for the
identity, `accumulate[w]` for the SIMD-tile fold, `join` to combine — and
driven by `rowwise.launch`. Its stated contract is one body for both targets:
the body never branches on `target`, because `rowwise.reduce`/`pjoin`/`once`/
`simd` each comptime-dispatch on `params.target`, so GPU primitives never
appear in CPU codegen and vice versa. That is the same CPU/GPU split
`numax.core.tensor` spells with its own `gpu: Bool` parameter, which makes
numax's reduce family a re-implementation of upstream scaffolding rather than
a gap it fills.

The convenience wrappers that *would* make this a function library —
`reduce_sum`, `reduce_mean` and friends — are in `algorithm.reductions`, which
is **26.6-dev only**: at the 26.5 pin `from algorithm.reductions import
reduce_sum` fails with "unable to locate module 'reductions'". Composing
`rowwise` with a `reduce_op` monoid is the supported spelling here, and it
loses nothing, since `Welford` and `OnlineLogSumExp` are already at the pin.

One dispatch fact worth knowing when reading any CPU benchmark on Apple
silicon: `linalg/matmul/cpu/apple_accelerate.mojo` routes to Apple's
`cblas_sgemm` whenever the target is macOS *and* every operand is `float32`.
So a `float32` CPU `matmul` on a Mac is Accelerate, and a comparison against
SciPy there is measuring call overhead rather than two kernels. At `float64`
the gate does not fire and MAX's own kernel runs, at 0.72x of Accelerate's
`dgemm` (264 against 365 GFLOP/s at `n = 1024` on an M3 Pro).

`pad` and `top_k` were searched at the pin for numax's own versions, and the
lesson of the two is that a kernel family is not one module. `nn.top_k` is a
whole delegate from a single entry point: it takes `largest`, a `target`, an
axis, a `sorted` flag and a required `DeviceContext`, and dispatches a
`parallelize`d CPU path against a real GPU one, so numax's `top_k` carries a
`gpu` parameter like the rest of the delegations.

`pad` is split across two modules that do not agree on a type. `nn.pad` is
host only — `pad_constant`, `pad_reflect` and `pad_repeat` (NumPy's `edge`)
take `TileTensor` in and out and have no `target` and no `DeviceContext`.
`nn.pad_gpu` covers the device, but **only** `pad_constant`, and it takes raw
`Pointer[Scalar[dtype]]` plus an `IndexList` shape rather than a tensor. So
numax's `pad` can carry a `gpu` parameter for the constant mode and cannot
for the other two, and its device path hands over a pointer where its host
path hands over a `.view()`. A pointer and a shape are not the denied
`LayoutTensor` bridge — they are what `Tensor`'s own `DeviceBuffer` already
holds — but this is the one delegation whose two halves are spelled
differently, and reading `nn/pad.mojo` alone says the GPU path does not
exist.

**Not available, so numax writes it:**

- **No symmetric or triangular BLAS-3 at all.** Searched at the `max ==26.5`
  pin across `linalg`, `nn`, `layout` and the top-level `algorithm` root:
  there is no `syrk`, `symm`, `trmm`, `trsm` or `potrf`. The only `syrk`
  symbols in the tree are unwired FFI declarations in the private
  `_cublas`/`_rocblas` shims, which are vendor- and architecture-specific and
  so denied on the same grounds as any other per-arch path. `matrix_band_part`
  is a materializing band *mask*, not a triangle-restricted product, and
  costs an extra full pass. Neither `grouped_matmul` (ragged `M` only, with
  `N`/`K` shared and static across groups, GPU-only) nor `batched_matmul`
  (uniform shapes) can batch differently-shaped block updates. So numax's
  blocked `cholesky` writes its own tiled symmetric trailing update --
  **extend**.
- **No Kronecker product, matrix power, or row-wise inner product.**
  Searched at the `max ==26.5` pin across `linalg`, `nn`, `algorithm` and
  `layout`: `kron`, `matrix_power`, `inner`, `vdot`, `tensordot` and
  `slogdet` return nothing -- numax now writes `slogdet` too, off the same
  LU diagonal `det` already reads, because `det`'s product of `n` entries
  overflows for ordinary matrices where a sum of logarithms does not. The
  only `outer_product` in the tree is
  `layout.math.outer_product_acc`, already denied above for being
  `LayoutTensor` and accumulate-only. So numax writes all three --
  **extend** for `kron` (an `elementwise` map over the output, MAX's idiom)
  and **delegate underneath** for the other two, since `inner` is
  `linalg.matmul` with its own `transpose_b` parameter and `matrix_power`
  is a squaring chain of `linalg.matmul` calls. `numpy.inner` at rank 1 is
  `dot`, which numax already has, so only the rank-2 form is added.
- **`vdot` and `multi_dot` are not absorbed, and each for a structural
  reason.** `vdot` differs from `dot` only by conjugating its first
  argument, and there is no complex `Tensor` -- so at this tier it would be
  a second name meaning exactly what `dot` means, which is the property
  `numax/prelude.mojo` protects. `multi_dot` chooses an association order
  for a chain of *differently shaped* products, and numax carries a
  tensor's shape in its type, so the chain cannot be held in one
  homogeneously typed container to begin with. `cross`, `tensordot`,
  `tensorsolve` and `tensorinv` wait on the general broadcasting and rank-n
  reductions listed under "What is still missing"; `matrix_transpose` is
  `numax.core.array.transpose`.
- **No matrix functions.** `expm`, `logm`, `sqrtm` and `funm` return
  nothing at the pin -- the only `expm` in the tree is a substring of
  `expm1` in `nn/activations.mojo`, which is elementwise and a different
  operation entirely. numax writes `expm` -- **extend**, and a clean one:
  scaling and squaring with a Pade approximant is matrix products plus one
  solve, so the cubic term goes to `linalg.matmul` and the shape is MAX's
  own. The rest are deferred behind a Schur decomposition, which is the
  same iterative sweep already deferred for `eigvals`;
  `numax/linalg/matfuncs.mojo` names the resumption commit.
- **No banded or Toeplitz solver.** Searched at the pin: `solve_banded`,
  `solveh_banded`, `cholesky_banded` and `solve_toeplitz` return nothing
  across the four roots, which follows from there being no dense triangular
  solve either. numax writes them -- **diverge**, and this is the one place
  that label is applied to something outside the `Array` tier, because the
  thing MAX cannot express is not a conformer but a *sequential* algorithm:
  a banded elimination is `O(n * bandwidth^2)` spread over `n` column steps
  that each touch a `bandwidth x bandwidth` corner, so there is no GEMM to
  send work to and no device residency to be had. `numax/linalg/banded.mojo`
  is therefore `Plain`-only and host-side by declaration, and says so at the
  top: it adds names and a shape, not speed. `solve_circulant` is the
  exception in that file -- a circulant is diagonalized by the DFT, so it is
  `O(n log n)` through `numax.fft` rather than an elimination at all, and it
  is why `linalg` depends on `fft`.
- **None of `scipy.linalg`'s structured constructors.** Searched at the pin
  across the four roots: `toeplitz`, `hankel`, `circulant`, `block_diag`,
  `companion` and the rest return nothing. These are the **Plain-only
  surface** outcome rather than extend or diverge -- MAX has no kernel,
  *and* instantiating a Hilbert matrix at `Dual` or `Interval` adds no
  meaning, since every entry is a constant of its index. So
  `numax/linalg/special_matrices.mojo` is `Tensor`-only with no `Array`
  sibling, and none is missing. `dft` stays out because its entries are
  complex and there is no complex `Tensor`; `invhilbert` stays out because
  its entries exceed what `float64` represents exactly, so computing them
  there and calling the result an inverse would be a claim numax cannot
  back. `pascal`, `invpascal`, `hadamard`, `helmert`, `fiedler`,
  `fiedler_companion` and `leslie` are simply not written yet -- each is a
  short index rule of the same shape as the eight that are, so they are a
  follow-up rather than a decision.
- **`scipy.linalg`, almost all of it.** One decomposition ships,
  `qr_factorization` (Householder, CPU-only, scalar loops), and it is on the
  older `LayoutTensor`, which numax denies rather than bridges — interop is
  `TileTensor` only. So: no usable LU, Cholesky, SVD, eig, `solve`, triangular
  solve, inverse, determinant, matrix norm, or BLAS-1, and no cuSOLVER bridge.
  numax's `cholesky`, `lu_factor`, `qr_factor` and `solve` over `Tensor` fill
  the gap blocked, sending the `O(n^3)` term back through `linalg.matmul`.
  The spectral reduction is now there too: `sytrd` reduces a symmetric
  matrix to tridiagonal form device-resident, with the symmetric rank-two
  update issued as the single product `[V | W] @ [W | V]^T` under
  `transpose_b=True` -- the identity that stands in for the `syr2k` above.
  `eigvalsh` and `eigh` sit on top of it: an implicit-QL sweep over the two
  diagonals, host-side and tier 2 by numax's definition, `O(n^2)` for values
  alone and `O(n^3)` of scalar rotations when the eigenvectors are
  accumulated -- the one ceiling in `eigh` with no MAX in it, named in the
  code with divide-and-conquer as the upgrade. The eigenvectors of `A` then
  come back as `Q Z` through one `matmul`. `svd` and `svdvals` take the
  same two phases over a bidiagonal form: `gebrd` reduces device-resident
  with alternating left and right reflectors, each a matrix-vector product
  and a rank-one update through `matmul`, and the singular values are the
  eigenvalues of the Golub-Kahan tridiagonal -- the `2n x 2n` symmetric
  tridiagonal with zero diagonal and off-diagonal `d_1, e_1, d_2, ...` --
  which the very same sweep diagonalizes, its eigenvectors interleaving
  `B`'s singular vectors. LAPACK's `dbdsvdx` takes that route too. `pinv`,
  `cond`, `matrix_rank` and `lstsq`'s `"svd"` method are its dependents --
  each one SVD and a few lines, delegating underneath. What is still missing
  over `Tensor` is the Hessenberg reduction `eigvals` starts from and its own
  sweep. `numax/linalg/__init__.mojo` carries the
  reasoning.
- **FFT.** Only `nn.irfft`: inverse real, last dimension, NVIDIA-only, a thin
  wrapper over the *private* `_cufft` package. No forward FFT anywhere, and
  nothing at all on Metal or AMD, so `numax.fft` over `Tensor` is an
  **extend** with no delegation underneath it.
- **Out-of-place tensor arithmetic and explicit broadcast.** `TileTensor` has
  in-place operators only and no `broadcast_to`.
- **Statistics and ordering past the basics.** No value-based or n-D sort, no
  `searchsorted`, no `partition`, no `unique`/`median`/`quantile`/`histogram`,
  and no `cumprod`. `nn.cumsum` *is* there and numax routes to it --
  `TileTensor` in and out, its axis a compile-time parameter, any axis rather
  than only the innermost, and a `float64` accumulator for a `float32` input,
  so it is both the MAX-first route and the more accurate one. It is
  **CPU-only**, and not by omission: the graph operator at
  `graph_compiler/builtin_kernels/reductions.mojo` takes a `DeviceContext` and
  drops it, so `mo.cumsum` has no GPU kernel either. numax's `cumsum` and
  `cumprod` are host-side for that reason; a device scan is a blocked
  Blelloch pass rather than a flag on this kernel.
- **Distributions.** Uniform, normal and Gumbel only.
- **Everything algorithmic.** No quadrature, ODE solvers, optimizers, root
  finders, sparse matrices, polynomials, or signal-sense convolution.

Scalar `gamma`/`lgamma`/`j0`/`j1`/`y0`/`y1` exist but are **CPU-only libm**:
compiling one into a GPU kernel fails with "libm operations are only available
on CPU targets", which is why numax keeps its own GPU-launchable versions.

## Dispositions

| Area | Home | Notes |
|---|---|---|
| Array creation and manipulation | `numax/core/array.mojo` | `Plain`-only, comptime shape, a thin owner whose `.view()` is a `TileTensor`. `transpose` routes to `linalg.transpose` on the host -- at rank 2 with the permutation in the type, and at any rank through `transpose(a, *axes)`, `swapaxes` and `moveaxis`, verified against an arbitrary rank-3 permutation at the pin -- and to an `elementwise` gather on a device, because every path `linalg.transpose` can reach is a host memcpy; `pad` routes to `nn.pad` on the host for all three NumPy modes and to `nn.pad_gpu` on a device for `constant`, the only mode MAX implements there; `to_array`/`to_tensor` bridge to the `Array[T, n]` conformer layer |
| Elementwise math | `numax/core/elementwise.mojo` | `Plain`-only over `std.math`, rather than growing `FloatLike` by twenty methods across seven conformers |
| Arithmetic and operators | `numax/core/ops.mojo` | Tensor-tensor and tensor-scalar; `astype` is explicit because there is no dtype promotion |
| Comparison and logic | `numax/core/logic.mojo` | Truth is a `Static[DType.bool]`, so a comparison composes with `logical_and` |
| Statistics | `numax/stats/statistics.mojo` | NumPy-named entry points only: `mean`/`variance`/`stddev` and the axis-wise `mean`/`variance_axis` fold through MAX's `Welford` monoid, `argmin`/`argmax` route into `nn.argmaxmin` -- **but only for the innermost axis**, which is the only one `nn/argmaxmin.mojo`'s `_argn` accepts ("axis other than innermost not supported yet"), and it wants an output of the input's own rank with that axis at extent 1. So the axis-wise `argmin`/`argmax` delegate at `axis == rank - 1`, where MAX parallelizes the outer rows, and numax walks the other axes itself; both return the first extremum among ties, NumPy's rule and MAX's. `median` and `mode` need a whole slice at once rather than a running accumulator, so their axis forms gather rather than fold. The nine distributions in `numax/stats/distributions.mojo` are `FloatLike` kernels with a `Tensor` overload of `pdf`/`pmf`, `cdf` and `ppf` beside each -- the kernel driven by `numax.core.tensor.map` with the distribution's parameters crossing the launch boundary as scalars, which is the case that `map`'s scalar-parameter overloads were added for. MAX ships uniform, normal and Gumbel *sampling* and no distribution function of any kind, so this is the Plain-only-surface outcome on top of tier-1 kernels. The `List[T]` forms of `variance`/`stddev`/`cumsum`/`mean` are `FloatLike`-generic — at `Compensated` they match a float64 reference where `Plain` drifts |
| Sorting, searching, masking | `numax/core/sorting.mojo` | Tier 2. `argsort` routes into `nn.argsort`, `top_k` into `nn.top_k`, and the axis-wise `take`/`take_along_axis` into `nn.gather`/`nn.gather_elements` -- the **tensor** overload of `nn.gather`, not the closure one whose `input_fn` origins are unreachable, so `take` carries a real `gpu` parameter. Those are the names with a device path; the rest walk a host copy, where `std.builtin.sort` is the better route. `searchsorted` has no MAX counterpart at all, so its vectorized form is numax's own binary search. `put` was searched against `nn.scatter_elements` and `nn.scatter_nd` and takes neither: both want the indices as a tensor shaped like the output slice rather than the flat list `nonzero` and `argsort` hand back, so delegating would mean building the very thing the caller is avoiding |
| Dense linalg | `numax/linalg/` (`blas`, `triangular`, `cholesky`, `lu`, `qr`, `basic`, `misc`, `panel`) and `numax/linalg/array/` (the same split plus `eigen`) | Two tiers sharing one set of names, one tier per import. `numax.linalg` is `Tensor`: `matmul`/`matvec`/`batched_matmul` delegate to `linalg.matmul`/`bmm`, and `cholesky`/`lu_factor`/`qr_factor`/`solve`/`lstsq` are blocked and device-resident, with the `O(n^3)` trailing update fused into `linalg.matmul`'s epilogue -- MAX's only factorization, `linalg.qr_factorization`, is `LayoutTensor`-only and so denied. `numax.linalg.array` is `FloatLike`-generic and register-resident, where differentiability is the point, and is the only tier with `eigh`/`eigvals`/`eigvalsh`/`svd`/`svdvals`. `PivotedLU`/`TensorLU` are the tier-2 exceptions, since a pivot choice is a branch on data |
| Root finding and minimization | `numax/optimize/least_squares.mojo` and `numax/optimize/array/{solve,optimize}.mojo` | Two tiers on the `numax.linalg` pattern. `numax.optimize.array` is the conformer tier: fixed-iteration siblings in `solve` (tier 1), converge-to-tolerance in `optimize` (tier 2), and its `least_squares`/`curve_fit` take the Jacobian from `Gradient` rather than a difference. Its `minimize`, `minimize_scalar` and `root_scalar` are the `scipy.optimize` entry points over `bfgs`/`cg`/`nelder_mead`, `brent`/`golden`/`fminbound` and `brentq`/`bisect_tol`/`newton_tol`/`halley_tol`/`secant`, with a vector `root` routed to its own `least_squares`, dispatching on a `StaticString` `method` the way SciPy spells it -- searched at the pin and recorded as the **Plain-only surface** outcome, since MAX ships no optimizer of any kind (see "Everything algorithmic" above) and instantiating a driver at `Dual` or `Interval` adds no meaning where the objective is already the generic half. `numax.optimize` is `Tensor` and holds `minimize` (`bfgs` keeping its inverse Hessian on the device through `matvec` and `outer`, `cg` keeping one vector) plus `least_squares`/`curve_fit`, for problems whose vectors are long; it takes the Jacobian as an argument, since a `dtype`-monomorphic tensor cannot hold a `Gradient`, and sends its damped step through the augmented least-squares system `numax.linalg.lstsq` solves rather than the normal equations |
| Quadrature and ODE | `numax/integrate/{quadrature,ode}.mojo` (`Tensor`), `numax/integrate/array/{quadrature,ode}.mojo`, `numax/integrate/integrate.mojo` | Two tiers on the `numax.linalg` pattern. `numax.integrate` is `Tensor`: `trapezoid`, `simpson` and `cumulative_trapezoid` over sampled values with `scipy.integrate`'s own signatures, `Plain`-only and host-side -- MAX ships no quadrature of any kind, and a weighted sum of samples at `Dual` means nothing, so this is the Plain-only-surface outcome. `rk4_system`, `dopri5` and `solve_ivp` over a `Tensor` state build every stage from one in-place `axpy` `elementwise` launch on the state's device -- the method-of-lines case, where the state is too large for registers -- and share the scalar `solve_ivp`'s controller. `numax.integrate.array` integrates a `FloatLike` *function* at a fixed node or step count and is tier 1. The adaptive drivers are tier 2, including `solve_ivp_stiff`, whose Newton iteration takes `df/dy` from `Dual` |
| Interpolation | `numax/interpolate/interp.mojo` (`Tensor`) and `numax/interpolate/array/interp.mojo` | Two tiers on the `numax.linalg` pattern. MAX has no interpolation at arbitrary query points: `nn.resize_linear`/`nn.resize_nearest_neighbor` (host-only) and `nn.resize_bicubic` (`target` + `DeviceContext`) resample a whole NCHW image onto a fixed output grid by a scale factor, which is a different operation from `numpy.interp`'s and `scipy.interpolate`'s value-at-these-points, so both tiers are **extend** with nothing underneath. Over `Tensor`, `interp` and the splines bisect the knots inside one `elementwise` launch -- the data-dependent branch the `Array` tier's scan-and-blend spline exists to avoid. `CubicSpline`, `PchipInterpolator`, `Akima1DInterpolator` and `CubicHermiteSpline` share SciPy's `PPoly` form and take non-uniform knots; `CubicSpline`'s slopes are the rows `scipy/interpolate/_cubic.py` assembles, solved by `numax.linalg.solve_banded` on the host. `RegularGridInterpolator` is where the comparison with the resize kernels is closest and still fails: they take scale factors and return the whole resampled image, it takes query points on a rectilinear grid; two dimensions only, the `n`-D form waiting on a caller. `Chebyshev.fit(x, y)` is the least-squares fit to data through `lstsq`, beside the `Array` tier's nodal fit of a function |
| Transforms and signal | `numax/fft/fft.mojo` and `numax/fft/array/fft.mojo`; `numax/signal/{convolution,windows}.mojo` (`Tensor`) and `numax/signal/array/signal.mojo` | Radix-2 at a power of two; over `Tensor`, Bluestein's chirp-z at any other length, three power-of-two transforms of `next_fast_len(2n - 1)` for one of length `n`. The `Array` tier stays power-of-two. `dct`/`dst` types I-IV over `Tensor` in `numax/fft/trig.mojo`, each a real projection of one complex DFT; MAX has no cosine or sine transform either. MAX's only transform is `nn.irfft` -- inverse-only, last-axis-only, NVIDIA-only -- so there is no forward FFT to route to at all and both tiers are numax's. Two tiers on the `numax.linalg` pattern: `numax.fft` is `Tensor`, `Plain`-only, a real/imaginary pair carried across `log2(n) + 1` device-resident stages, for sizes an `Array` cannot hold; `numax.fft.array` is the `FloatLike` tier that differentiates at `Complex[Dual]` and runs per SIMD lane inside a kernel body. Signal convolution was gated against `nn.conv`: `conv_gpu` is `TileTensor`, rank 1-3, asymmetric padding, so a 1-D `full` convolution is expressible as a `[1, W, 1]` image against an `[R, 1, 1]` filter -- and it is GPU-only with a pack-the-filter CPU sibling (`conv_nhwc_direct`), two paths for one operation, tiled over the channels and filters a signal has one each of. numax writes the direct sum as one `elementwise` launch and `fftconvolve` on `numax.fft`, which MAX has no counterpart to at all. The `Tensor` windows are host tables. `lfilter`/`firwin` cover IIR and FIR at the `Array` tier; filter *design* past a windowed sinc is out |
| Tensor I/O | `numax/io/io.mojo`, `numax/io/npy.mojo` | Two formats: numax's own `NMX1` for numax-to-numax round trips (MAX ships no array I/O at all), and NumPy's `.npy` for interchange -- `numax.io.numpy.save` output is byte-identical to `numpy.save`, and `numax.io.numpy.load` reads `numpy.save` output, with no Python or NumPy dependency since the format is self-contained. `.npz` is out: it is a zip container |
| Random sampling | `numax/stats/random.mojo` | Over `std.random` on the host. No `Random[FloatLike]` conformer: RNG is not differentiable, so the trait contract does not fit |

### Where the spelling differs, and why

Every name numax adds matches SciPy or NumPy exactly. A handful of older
ones do not, and each divergence is a decision rather than an oversight, so
they are listed here rather than aliased -- a second name for one thing is
the property `numax/prelude.mojo` exists to protect.

**Forced by Mojo.** `var` is a keyword and `std` is the standard library's
package, hence `variance` and `stddev`; `where` introduces constraint
clauses, hence `select`.

**Chosen.**

| SciPy / NumPy | numax | Why |
|---|---|---|
| `inv` | `inverse` | spelled out, in the same spirit as `variance`/`stddev` above |
| `cho_factor` / `cho_solve` | `cholesky` / `cholesky_solve` | ditto; the abbreviation saves four characters and costs a reader the expansion |
| `qr` returning `(Q, R)` | `qr_factor` returning `TensorQR` | forced at the `Tensor` tier: a `Tuple` of two `Tensor`s cannot be destructured in Mojo 1.0, since `Tensor` is `Movable` and tuple unpacking wants `ImplicitlyCopyable`. `numax.linalg.array.qr` *is* the tuple-returning one |
| `lu_solve(factor, b)` | `TensorLU.solve(b)` / `PivotedLU.solve(b)` | the factorization object owns its solve, so the pairing cannot be got wrong |
| `res.fun` / `res.nit` / `res.success` | `f_x` / `iterations` / `converged` | `converged` says what it means where `success` does not; the result structs' docstrings carry the rest |
| `scipy.fft.next_fast_len(n)` returning the next 2-3-5-7-11-smooth number | `next_fast_len(n)` returning the next power of two | same contract -- the smallest length `>= n` the engine transforms at full speed -- and a different engine: pocketfft has a radix per small prime, numax has radix 2 and Bluestein. The name is kept because a caller padding to it gets what the name promises; the set differs because the engine does |
| `scipy.integrate.trapezoid(y, x)` / `simpson(y, x)` over samples | the same names over `Tensor` take exactly that signature; `numax.integrate.array.trapezoid[f](a, b)` and `simpson[f](a, b)` integrate a *function* on their own grid | the `Array` tier is the divergent spelling: it exists so the rule can run at `Dual` inside a kernel, which a caller holding samples has no use for. Same rule, different input, one tier per import |

Names that would have been a second spelling of an existing one are absent
on purpose: `vdot` (`dot`, there being no complex `Tensor` to conjugate),
`numpy.linalg.norm(v, ord=0)` (`count_nonzero`), and `numpy.inner` at rank 1
(`dot` again).

### Not absorbed

A separate complex array type — `Complex` composes into every kernel already.
A `Backend` trait — the `gpu: Bool` parameter on `numax.core.tensor.map` covers the
same ground with no dispatch. ML primitives beyond `softmax` — `nn` ships
normalization, convolution and pooling, tuned, on both backends, and numax has
no reason to name them.

`softmax` is the one numax does name, because a NumPy/SciPy-shaped library is
expected to have it: `numax.special.activations.softmax` is a delegation, MAX's
`nn.softmax` with numax's tensors passed straight through. Only the CPU
overload is reachable at the pin, for the `input_fn`-origins reason
`numax/special/activations.mojo` records; a device-resident softmax is still
hand-launched from `numax.core.tensor`'s row primitives, which is what
`examples/intermediate/softmax.mojo` shows.

`max.algorithm.dual_elementwise` was checked as a way to fuse chained maps into
one launch and is denied twice over. It ships only the compile-time-parameter
form, so a closure fails with "failed to infer parameter `__origins__`" while a
non-capturing function cannot reach the tensors at all -- its body is handed
only a coordinate. And it fuses two *independent* elementwise operations into
one launch rather than two chained ones, so it does not address the chained
case even where it compiles.

`nn.tile` and `nn.repeat_interleave` were searched at the pin for `tile` and
`repeat`, and both are **delegates**: `nn.tile` is the ONNX `Tile` operator and
agrees with `numpy.tile` element for element, `nn.repeat_interleave` agrees with
`numpy.repeat`. Each carries one limit worth recording. `nn.tile` asserts
**rank 4 at most** and takes no `target` and no `DeviceContext`, so it is host
only; `nn.repeat_interleave` does take a context and so has a device path. There
is no `nn.roll` and no `nn.expand_dims`, and no `nn.gather` spelling of a cyclic
shift that avoids materializing the index tensor, so `roll` and `expand_dims`
are numax's own -- the **Plain-only surface** outcome, since a cyclic shift at
`Dual` or `Interval` means nothing a plain one does not.

MAX's `nn` versions of `arange`/`reshape`/`concat`/`split` were checked and are
not usable as array functions: `nn.arange` returns one SIMD vector for an
index rather than filling a tensor, `nn.concat` wants a pre-sized output plus a
`DeviceContext` and one layout type across all inputs, and `nn.reshape`
returns a dynamically-laid-out `TileTensor`. They are graph-operator kernels.

## What is still missing

Slicing as a first-class owned type, and fancy indexing.

**General broadcasting is no longer missing.** `broadcast_shapes` in
`numax/core/array.mojo` is NumPy's right-alignment rule written down once,
and every binary routine in `numax.core.ops`, `numax.core.elementwise` and
`numax.core.logic` has a second overload taking two shapes it accepts. The
broadcast walk reads through zero strides rather than materializing either
operand, so an `(n, n)` against an `(n,)` costs index arithmetic and not a
second square buffer. The result is a `Dynamic`, since the extents it
computes are run-time values; the same-shape overload still matches first
and still returns the input's own layout type, so a compile-time shape
survives where both operands have one. `broadcast_op_axis` remains the
route that pairs a broadcast with a reduction, which is what softmax and
per-axis normalization need.

`numpy.broadcast_arrays` has no counterpart and will not get one: it
returns a tuple of tensors, and a `Tuple` of two `Tensor`s cannot be
destructured in Mojo 1.0 -- the same constraint that makes `qr_factor`
return a `TensorQR` rather than a pair. `broadcast_to` covers the single-
operand case.

What exists: `Tensor` carries its shape in its layout type, with each
dimension independently compile-time or run-time, so an extent that depends
on a value has somewhere to live — `unique`, `extract` and `take` return a
right-sized tensor rather than a full-length one plus a count. Every walker
in `numax.core.tensor` has a run-time-shape overload selected by a `where`
clause that is the exact negation of the static one, and `map_strided` /
`reduce_strided` walk a transposed or sliced view that is not row-major at
all. The run-time paths are CPU-only, because a GPU launch needs the extent
in the type.

Also absent, each a decision: sparse matrices, iterative solvers, distributed
execution, and dtype promotion. The first three are a different library's job;
the fourth is a compile error waiting to happen in a language that infers
parameters, so `astype` is explicit.
