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
- **`scipy.linalg`, almost all of it.** One decomposition ships,
  `qr_factorization` (Householder, CPU-only, scalar loops), and it is on the
  older `LayoutTensor`, which numax denies rather than bridges — interop is
  `TileTensor` only. So: no usable LU, Cholesky, SVD, eig, `solve`, triangular
  solve, inverse, determinant, matrix norm, or BLAS-1, and no cuSOLVER bridge.
  numax's `cholesky`, `lu_factor`, `qr_factor` and `solve` over `Tensor` fill
  the gap blocked, sending the `O(n^3)` term back through `linalg.matmul`.
  `svd`, `eigh` and `eigvals` over `Tensor` stop short on purpose: their
  reduction phase is the block reflector numax now has, but their iterative
  phase is a sequential sweep over a two-wide band with data-dependent
  deflation, which no GEMM helps and which is tier 2 by numax's definition.
  `numax/linalg/__init__.mojo` carries the reasoning.
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
| Array creation and manipulation | `numax/core/array.mojo` | `Plain`-only, comptime shape, a thin owner whose `.view()` is a `TileTensor`. `transpose` routes to `linalg.transpose` on the host and to an `elementwise` gather on a device, because every path `linalg.transpose` can reach is a host memcpy; `pad` routes to `nn.pad` on the host for all three NumPy modes and to `nn.pad_gpu` on a device for `constant`, the only mode MAX implements there; `to_array`/`to_tensor` bridge to the `Array[T, n]` conformer layer |
| Elementwise math | `numax/core/elementwise.mojo` | `Plain`-only over `std.math`, rather than growing `FloatLike` by twenty methods across seven conformers |
| Arithmetic and operators | `numax/core/ops.mojo` | Tensor-tensor and tensor-scalar; `astype` is explicit because there is no dtype promotion |
| Comparison and logic | `numax/core/logic.mojo` | Truth is a `Static[DType.bool]`, so a comparison composes with `logical_and` |
| Statistics | `numax/stats/statistics.mojo` | NumPy-named entry points only: `mean`/`variance`/`stddev` and the axis-wise `mean`/`variance_axis` fold through MAX's `Welford` monoid, `argmin`/`argmax` route into `nn.argmaxmin`. The `List[T]` forms of `variance`/`stddev`/`cumsum`/`mean` are `FloatLike`-generic — at `Compensated` they match a float64 reference where `Plain` drifts |
| Sorting, searching, masking | `numax/core/sorting.mojo` | Tier 2. `argsort` routes into `nn.argsort`; the rest walk a host copy, where `std.builtin.sort` is the better route |
| Dense linalg | `numax/linalg/` (`blas`, `triangular`, `cholesky`, `lu`, `qr`, `basic`, `misc`, `panel`) and `numax/linalg/array/` (the same split plus `eigen`) | Two tiers sharing one set of names, one tier per import. `numax.linalg` is `Tensor`: `matmul`/`matvec`/`batched_matmul` delegate to `linalg.matmul`/`bmm`, and `cholesky`/`lu_factor`/`qr_factor`/`solve` are blocked and device-resident, with the `O(n^3)` trailing update fused into `linalg.matmul`'s epilogue -- MAX's only factorization, `linalg.qr_factorization`, is `LayoutTensor`-only and so denied. `numax.linalg.array` is `FloatLike`-generic and register-resident, where differentiability is the point, and is the only tier with `eigh`/`eigvals`/`svd`. `PivotedLU`/`TensorLU` are the tier-2 exceptions, since a pivot choice is a branch on data |
| Root finding and minimization | `numax/optimize/solve.mojo`, `numax/optimize/optimize.mojo` | Fixed-iteration siblings in `solve` (tier 1), converge-to-tolerance in `optimize` (tier 2). `least_squares`/`curve_fit` take the Jacobian from `Gradient` rather than a difference |
| Quadrature and ODE | `numax/integrate/quadrature.mojo`, `numax/integrate/ode.mojo`, `numax/integrate/integrate.mojo` | Fixed-node and fixed-step are tier 1; adaptive is tier 2, including `solve_ivp_stiff`, whose Newton iteration takes `df/dy` from `Dual` |
| Transforms and signal | `numax/fft/fft.mojo`, `numax/signal/signal.mojo` | Power-of-two by construction. MAX's only transform is `nn.irfft` -- inverse-only, last-axis-only, NVIDIA-only -- so there is no forward FFT to route to at all. `lfilter`/`firwin` cover IIR and FIR; filter *design* past a windowed sinc is out |
| Tensor I/O | `numax/io/io.mojo`, `numax/io/npy.mojo` | Two formats: numax's own `NMX1` for numax-to-numax round trips (MAX ships no array I/O at all), and NumPy's `.npy` for interchange -- `numax.io.numpy.save` output is byte-identical to `numpy.save`, and `numax.io.numpy.load` reads `numpy.save` output, with no Python or NumPy dependency since the format is self-contained. `.npz` is out: it is a zip container |
| Random sampling | `numax/stats/random.mojo` | Over `std.random` on the host. No `Random[FloatLike]` conformer: RNG is not differentiable, so the trait contract does not fit |

Three names differ from NumPy's because Mojo will not allow them: `var` is a
keyword and `std` is the standard library's package (hence `variance`,
`stddev`), and `where` introduces constraint clauses (hence `select`).

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

MAX's `nn` versions of `arange`/`reshape`/`concat`/`split` were checked and are
not usable as array functions: `nn.arange` returns one SIMD vector for an
index rather than filling a tensor, `nn.concat` wants a pre-sized output plus a
`DeviceContext` and one layout type across all inputs, and `nn.reshape`
returns a dynamically-laid-out `TileTensor`. They are graph-operator kernels.

## What is still missing

General broadcasting between two arbitrary shapes, slicing as a first-class
owned type, and fancy indexing. `broadcast_op_axis` broadcasts only in the
direction that pairs with a reduction, which is what softmax and per-axis
normalization need and less than NumPy does.

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
