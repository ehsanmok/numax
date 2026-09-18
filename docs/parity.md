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
| Ordering | `argsort` (rank-1), `top_k` (any axis, CPU and GPU); host `sort`/`partition` (the latter two-way, so quadratic on a near-constant range -- `numax.stats.quantiles` selects three ways instead) | `nn`, `std.builtin.sort` |
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

**Batched small problems (`map_blocks`)** were gated at the pin and in
`~/workspace/modular-oss` across every top-level kernel root.
`linalg/bmm.mojo`'s `batched_matmul` is the only batched kernel MAX ships:
`TileTensor` in and out, a `target`/`context` pair and an
`elementwise_epilogue_fn`, but monomorphic in a raw `dtype`, so no
conformer passes through it, and it is one fixed operation rather than a
user kernel mapped over blocks. `nn` has no per-lane block map -- its
operators run over whole tensors -- `algorithm` is the reduction library,
where `rowwise` folds a row to a value rather than handing a lane a
problem, and `layout` supplies `TileTensor` and no walk. Disposition:
**extend**. `numax.core.tensor.map_blocks` is written in MAX's idiom --
`TileTensor` in and out, `gpu` picking the `target`, a `DeviceContext`,
launched through `max.algorithm.elementwise` -- and gives one lane one
small problem out of a `(k, batch)` SoA layout, which is the shape
`docs/why.md`'s "many small problems" audience needs and the one thing
`map` could not express, since `map` hands a lane one scalar.

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
  homogeneously typed container to begin with. `cross` is an
  `elementwise` map (extend); `tensordot` in NumPy's integer form is one
  run-time-shape `linalg.matmul` on the two buffers retyped as matrices
  (delegate underneath), and `tensorsolve`/`tensorinv` are `solve` and
  `inverse` on a tensor retyped as the square matrix its element count
  says it is, so all four are delegations with a reshape around them;
  `matrix_transpose` is `numax.core.array.transpose`.
- **No matrix functions.** `expm`, `logm`, `sqrtm` and `funm` return
nothing at the pin -- the only `expm` in the tree is a substring of
`expm1` in `nn/activations.mojo`, which is elementwise and a different
operation entirely. numax writes all of them -- **extend**: `expm` by
scaling and squaring with every product in `linalg.matmul`, and `sqrtm`,
`logm`, `funm`, `cosm`, `sinm` and `fractional_matrix_power` on the real
Schur form `numax.linalg.schur` produces, a host recurrence over the
quasi-triangular factor between two `matmul`s. `numax/linalg/matfuncs.mojo`
has each algorithm and its ceiling.
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
  `fiedler_companion` and `leslie` are written too, each a short index
  rule of the same shape as the first eight -- `invpascal` in closed form
  rather than by inverting, `hadamard` as `(-1)^popcount(i & j)` rather
  than `log2 n` stackings.
- **`scipy.linalg`, almost all of it.** One decomposition ships,
  `qr_factorization` (Householder, CPU-only, scalar loops), and it is on the
  older `LayoutTensor`, which numax denies rather than bridges — interop is
  `TileTensor` only. So: no usable LU, Cholesky, SVD, eig, `solve`, triangular
  solve, inverse, determinant, matrix norm, or BLAS-1, and no cuSOLVER bridge.
  numax's `cholesky`, `lu_factor`, `qr_factor` and `solve` over `Tensor` fill
  the gap blocked, sending the `O(n^3)` term back through `linalg.matmul`.
  The spectral reduction is now there too, and it is **blocked**: `sytrd`
  reduces a symmetric matrix to tridiagonal form device-resident over
  `latrd` panels of `block` columns. Each column of a panel is brought up
  to date with the panel's own `V` and `W` rather than with the matrix,
  its `w` built from `2j + 1` reductions that also supply the scalar a
  separate `dot` used to synchronize for, and the panel's whole trailing
  update then goes out as the single product `[V | W] @ [W | V]^T` under
  `transpose_b=True` at `K = 2 * block` -- the identity that stands in for
  the `syr2k` above, at rank 64 instead of rank 2. The update is
  restricted to the rows and columns past the panel, because the rows
  inside it hold the packed reflectors. `block == 1` is the unblocked
  reduction exactly and `block == n` is one panel, both pinned by tests.
  What is left is the matrix-vector product `p = A v`, one per column:
  `2n^3` bandwidth-bound flops that blocking does not touch (half of
  LAPACK's `dsytrd` is BLAS-2 for the same reason), and the upgrade is a
  two-stage dense-to-banded reduction, not more blocking.
  `eigvalsh` and `eigh` sit on top of it: an implicit-QL sweep over the two
  diagonals, host-side and tier 2 by numax's definition, `O(n^2)` for values
  alone. The eigenvector accumulation that used to sit beside it as
  `O(n^3)` of scalar host rotations is **blocked now**: `eigh` takes a
  `block` (default 32), and `_RotationBatch` holds `block` sweeps' worth of
  rotations, reorders them into windows of mutually commuting ones (Lang
  1998 -- rotations on adjacent pairs `(i, i+1)` and `(j, j+1)` commute
  when `|i - j| >= 2`) and sends each window out as one `linalg.matmul` of
  a `w x w` rotation product against a `w x n` stripe. `Z` is held
  transposed on the device so that stripe is a contiguous row block, and
  the eigenvectors of `A` come back as `inner(q, zt)` under
  `transpose_b=True`, never transposing it. The host keeps
  `O(block * n^2)` of contiguous-row work building the products, and
  `block == 1` recovers the unblocked algorithm exactly. Forming the
  reduction's own `Q` -- LAPACK's `orgtr` and `orghr` -- is blocked too,
  and by reuse rather than by a second implementation: `sytrd` and
  `hessenberg` take the same `block` -- on `sytrd` it is the reduction
  panel as well -- and `.q()` runs `qr_factor`'s
  reverse panel walk over the packed reflectors through a view shifted one
  row down, since a tridiagonal reduction puts the implicit unit one row
  below a QR's. `svd` shares the same accumulation through **two**
  batches, one per side, and its singular vectors never touch the host
  either: `_bdsqr` leaves `U_B^T` and `V_B^T` device-resident, one
  `elementwise` gathers the rows into descending order and applies the
  sign fix, and `U = Q U_B`, `V = P V_B` come back through `inner` under
  `transpose_b=True` because the batches hold both factors transposed.
  `gebrd` takes the same `block` for both its own reduction and that walk,
  `.p()` reaching the walk through one transposing pack since the right
  reflectors are held as rows. `schur` shares the same batch at **reach two**,
  since a Francis reflector spans three columns where a Givens rotation
  spans two: the tag becomes `(k + 2 s) // block` applied in *increasing*
  order (the chase ascends and the window slides two columns down per
  sweep), `block // 2` sweeps ride in one batch, and the `2 x 2` real split
  rotation is pushed as a sweep of its own. `svd` and `svdvals` take the
  same two phases over a bidiagonal form, and that reduction is **blocked**
  too: `gebrd` runs `labrd` panels of `block` columns, each column brought
  up to date with the panel's own `V`, `Y`, `X` and `U` rather than with
  the matrix (`labrd_column` for the column the left reflector comes from,
  `labrd_row` for the row the right one comes from), the two corrections
  built by `labrd_y` and `labrd_x` with their `O((m + n) j)` terms threaded
  the way `latrd_w`'s are, and the whole panel then applied as the two
  products `V Y^T` and `X U` under `transpose_b=True`. That takes the
  whole-matrix traffic from four passes per column to two, since the two
  rank-one updates are what the panel defers; the two matrix-vector
  products stay, for the reason `sytrd`'s `A v` does. `block == 1` is the
  unblocked reduction and `block == n` one panel, both pinned by tests, and
  the per-column `taus.to_host()` synchronizations are gone -- the panel
  kernels read the scales on the device. The band iteration is
  `dbdsqr`'s: an implicit-shift QR on the bidiagonal itself, splitting at
  a negligible off-diagonal, taking Demmel and Kahan's zero shift wherever
  a nonzero one would cost relative accuracy and otherwise the smallest
  singular value of the trailing `2 x 2` (`las2`), and chasing the bulge
  with one rotation on the right and one on the left per column. The two
  streams go into two `_RotationBatch`es at `reach = 1` and **ascending**
  -- the chase runs top-down, so its index rises within a sweep where
  `_tql`'s falls -- which hold `U_B^T` and `V_B^T` device-resident; the
  descending order and `dbdsqr`'s sign fix are then one `elementwise`
  gather. The chase is top-down only, since one accumulator cannot carry
  both tag directions; `dbdsqr` picks per block and a matrix graded the
  other way costs sweeps here. **The `2n x 2n` Golub-Kahan doubling is
  gone**: `svd` used to embed `B` in the symmetric tridiagonal with zero
  diagonal and off-diagonal `d_1, e_1, d_2, ...`, whose eigenvalues are
  `+-sigma` and whose eigenvectors interleave the singular vectors, and
  run `_tql` there -- about `4n^2` rotations at width `2n` against `2n^2`
  at width `n` now. `_golub_kahan` stays private as the test oracle the
  new sweep is pinned against, and LAPACK's `dbdsvdx` is the route it
  took. `pinv`,
  `cond`, `matrix_rank` and `lstsq`'s `"svd"` method are its dependents --
  each one SVD and a few lines, delegating underneath. The general
  spectrum takes the same two phases over a Hessenberg form, and that
  reduction is **blocked** too: `hessenberg` runs `lahr2` panels of
  `block` columns, each column brought up to date with the panel's own
  reflectors on both sides by `lahr2_column` -- the deferred right factor
  `Y V[i, :]^T` first, then the left block reflector `V T^T (V^T b)` --
  and `lahr2_y` building the panel's `Y = A V T` and the matching column
  of `T` out of one shared set of `j` dot products, which is `larft`'s own
  recurrence. The panel then goes out as `A[:, k0+nb:] -= Y V[k0+nb:, :]^T`
  in one `transpose_b=True` GEMM with the subtraction in its epilogue and
  `A[k0+1:, k0+nb:] <- (I - V T^T V^T) A[k0+1:, k0+nb:]` through the same
  `_apply_block_reflector` QR uses, reached by the one-row-shifted view.
  That takes the whole-matrix traffic from four passes per column --
  `v^T A`, a rank-one subtract, `A v`, another -- to one, and the
  per-column `taus.to_host()` synchronization is gone with it. What is
  left is `p = A v`, `sytrd`'s ceiling again. `block == 1` is the
  unblocked reduction and `block == n` one panel, both pinned by tests.
  **That ceiling is the second 0.3 filing: the two-stage dense-to-banded
  reduction** (LAPACK's `sytrd_2stage` shape -- dense to banded by GEMMs,
  then banded to tridiagonal by a bulge chase), which is the only thing
  that removes a whole-matrix `matvec` per column, because no width of
  panel does. Measured on an M3 Pro at `n = 1024`, `float32`: `svd` and
  `svdvals` are 1,583 and 1,459 ms and `gebrd` is about 1,450 of both, so
  the reduction is 92% of a values-only SVD and the vectors cost 120 ms;
  `eigvalsh` is 142 ms at `block = 32` and 74.8 at `block = 8`, nearly all
  of it `sytrd`. `svd` at 0.056 of LAPACK is the row this would move.
  Then the Francis double-shift QR
  iteration -- EISPACK's `hqr2`, LAPACK's `dlahqr` -- runs on the host,
  `eigvals` reading the eigenvalues off its deflations and `schur` keeping
  the quasi-triangular `T` and accumulating the chase's transformations
  into `Z^T` device-resident, brought back through the reduction's `Q`
  under `transpose_b=True`. What is left on `schur`'s host side is `T`
  itself: the far-from-diagonal row and column updates can be deferred only
  one sweep at a time, because the next chase reads rows a deferred update
  would already have written, so batching them across sweeps needs many
  shifts per sweep. **Multishift QR with aggressive early deflation
  (`dhseqr` driving `dlaqr5`) is therefore filed for 0.3**, as a different
  algorithm rather than a different schedule; the interim is that `T`'s row
  update runs down three contiguous rows as one SIMD walk while its column
  update strides by `n` and stays scalar. The size of the filing, measured
  on an M3 Pro at `n = 1024`, `float32`: `eigvals` is 920 ms of which the
  blocked `lahr2` reduction is about 146, so five sixths of the call is the
  host Francis iteration, and `schur` is 1,056 ms on top of that. Those are
  0.15 and 0.16 of LAPACK where the factorizations reach 0.23 to 0.66, so
  this one item is most of the remaining spectral gap on the general side. `schur`'s own docstring carries
  the measured split, and `numax/linalg/__init__.mojo` the reasoning.
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
- **`logsumexp`.** No entry point -- `nn.softmax` with `logsoftmax=True` returns the normalized tensor, not the scalar -- but the fold is there: `algorithm.reduce_op.OnlineLogSumExp` is the flash-style `{max, sum exp(x - max)}` monoid, and `numax.special.logsumexp` over `Tensor` drives it through `rowwise` exactly as `numax.linalg.dot` drives `ReduceSum`. The monoid is delegated, the entry point extended.
- **Distributions.** `nn.random`'s draws are uniform, normal and Gumbel
  only. That is MAX's RNG surface, not numax's: `numax.stats` carries nine
  `scipy.stats` distribution namespaces with the full method set, listed in
  the dispositions table below.
- **Everything algorithmic.** No quadrature, ODE solvers, optimizers, root
  finders, sparse matrices, polynomials, or signal-sense convolution.

Scalar `gamma`/`lgamma`/`j0`/`j1`/`y0`/`y1` exist but are **CPU-only libm**:
compiling one into a GPU kernel fails with "libm operations are only available
on CPU targets", which is why numax keeps its own GPU-launchable versions.

## Dispositions

| Area | Home | Notes |
|---|---|---|
| Array creation and manipulation | `numax/core/array.mojo` | `Plain`-only, comptime shape, a thin owner whose `.view()` is a `TileTensor`. `transpose` routes to `linalg.transpose` on the host -- at rank 2 with the permutation in the type, and at any rank through `transpose(a, *axes)`, `swapaxes` and `moveaxis`, verified against an arbitrary rank-3 permutation at the pin -- and to an `elementwise` gather on a device, because every path `linalg.transpose` can reach is a host memcpy; `pad` routes to `nn.pad` on the host for all three NumPy modes and to `nn.pad_gpu` on a device for `constant`, the only mode MAX implements there; `to_array`/`to_tensor` bridge to the `Array[T, n]` conformer layer. Added in 0.2: `flatten` (`ravel` under NumPy's other name -- the view-versus-copy distinction that separates them there cannot exist where `Tensor` owns its storage), `dstack`, `atleast_3d` at each of its three ranks, `rot90` as two overloads because the parity of the quarter turn decides whether the extents swap, and `array_split`, which is the section-count form `split` could not express: a `List[Dynamic[dtype, rank]]` can carry a run-time number of outputs where a comptime-shaped tensor cannot, and the uneven division follows NumPy rather than raising the way `numpy.split` does. `rot90`'s `k` must already lie in `0 .. 3`, since `where` cannot fold `%` |
| Elementwise math | `numax/core/elementwise.mojo`, driven by the private `numax/core/_drive.mojo` | `Plain`-only over `std.math`, rather than growing `FloatLike` by twenty methods across seven conformers; `exp`, `log` and `erf` come from `numax/core/libm.mojo` at float64, where `std.math`'s are `1e5` to `2e8` ulp off. Gated against MAX and recorded as **extend**: `max.algorithm.functional.elementwise` is the scaffolder and numax supplies the body, the same shape `numax.stats.distributions` and `numax.stats.random` already use; `nn` ships fused activation and normalization operators rather than a NumPy-named `arctanh`, and there is no `elementwise` monoid set to route to the way `algorithm.rowwise` gives the reductions one. So the launch policy is numax's and the walk is MAX's. Every routine takes `gpu: Bool = False` last; `elementwise` computes its grid from a run-time `Coord`, so a `Dynamic` launches on a device exactly as a `Static` does and there is one signature per name. Below `1 << 16` elements the host path is a serial SIMD loop, since thread dispatch loses there. A call whose target and whose tensor's residency disagree runs the pre-0.2 host walk and prints one line on `stderr` naming the fast spelling. Added in 0.2: `sign` (two comparisons, not `copysign(1, x)`, which answers -1 at -0.0 and 1 at NaN where NumPy gives 0 and NaN), `square`, `reciprocal`, `rint`, `degrees`, `radians`, and `fmax`/`fmin`. The last pair came with a **correctness fix**: `max`/`min` on a `SIMD` are IEEE `maxNum`/`minNum` and skip a NaN, so `maximum`/`minimum` had been answering 1.0 for `maximum(nan, 1)` where NumPy answers nan. They now propagate on float dtypes and `fmax`/`fmin` are the spellings that want the bare instruction, which is what makes both names worth having |
| Arithmetic and operators | `numax/core/ops.mojo`, driven by the private `numax/core/_drive.mojo` | Tensor-tensor, tensor-scalar and tensor-tensor at two broadcastable shapes; `astype` is explicit because there is no dtype promotion, and it is the one name here whose result dtype differs from its input's, so it takes `_drive.unary_to`. Gated against MAX on the same terms as the elementwise row above and recorded as **extend**: `max.algorithm.functional.elementwise` is the scaffolder and numax supplies the body. Every routine takes `gpu: Bool = False` last and there is one signature per name, a `Dynamic` launching on a device exactly as a `Static` does. The operators on `Tensor` -- `+`, `-`, `*`, `/`, unary `-` -- have no parameter list to spell `[gpu=True]` in, so they forward at the default and a device tensor gets the retained host walk plus one line on `stderr` naming `add[gpu=True]` |
| Comparison and logic | `numax/core/logic.mojo`, driven by the private `numax/core/_drive.mojo` | Truth is a `Static[DType.bool]`, so a comparison composes with `logical_and`. Gated against MAX on the same terms as the two rows above and recorded as **extend**: `max.algorithm.functional.elementwise` is the scaffolder and numax supplies the body -- `nn` has no elementwise comparison operator and `algorithm`'s monoid set is reductions, not predicates. Every comparison and predicate takes `gpu: Bool = False` last, and the destination is a `DType.bool` tensor the launch writes through `TileTensor.store`, probed on Metal before the route was written. The comparisons are spelled as SIMD methods (`a.gt(b)`) because the operators are scalar-only at a width above one, and `not_equal` is `~a.eq(b)` because `SIMD.ne` is the *ordered* comparison and NumPy's is not. `all` and `any` return a `Bool` and stay host reads that short-circuit; `allclose` and `array_equal` are the device comparison plus that read |
| Statistics | `numax/stats/statistics.mojo`, over `numax/core/rowwise.mojo` | NumPy-named entry points only, and every reduction among them is now one of MAX's `algorithm.reduce_op` monoids under its `rowwise` scaffolder, reached through `numax.core.rowwise`: `mean`/`variance`/`stddev` and the axis-wise `mean`/`variance_axis` through `Welford`, `sum`/`prod`/`min`/`max` (whole-tensor through `reduce_all`, axis-wise through `sum_axis`/`prod_axis`/`min_axis`/`max_axis`) through `ReduceSum`/`ReduceProduct`/`ReduceMin`/`ReduceMax`, and the **whole-tensor `argmax`/`argmin` through `reduce_op.ArgMax`/`ArgMin`** -- the gate result recorded here: those monoids exist at the 26.5 pin as `ArgMax[dtype, W]`, carry the index beside the value, break ties to the lower index and skip NaN candidates, so numax writes no comparison logic and the input never leaves the device. The winning index is read from `acc_indices`, not the state's scalar `best_idx`, because `join_parallel` leaves it there so the cooperative and tiled tiers emit alike -- what MAX's own `algorithm.reductions.reduce_argmax` reads. Each of these takes `gpu: Bool = False` last, and a call whose target and whose tensor's residency disagree runs the pre-0.2 host walk plus one line on `stderr`, the policy `numax/core/_drive.mojo` sets. `sum` and `prod` are reassociated by the monoid and say so. The **axis-wise `argmin`/`argmax` stay on `nn.argmaxmin`**, MAX's only axis-taking entry point -- **but only for the innermost axis**, which is the only one `nn/argmaxmin.mojo`'s `_argn` accepts ("axis other than innermost not supported yet"), and it wants an output of the input's own rank with that axis at extent 1. So the axis-wise `argmin`/`argmax` delegate at `axis == rank - 1`, where MAX parallelizes the outer rows, and numax walks the other axes itself; both return the first extremum among ties, NumPy's rule and MAX's. `median` and `mode` need a whole slice at once rather than a running accumulator, so their axis forms gather rather than fold, and `cumsum`/`cumprod` are scans rather than folds, so all four stay host-side and declare it; `quantile`/`percentile` (all thirteen NumPy methods), their `nan*` forms and `median` are host-side for the same reason, but by **selection rather than a sort**: every one of the thirteen methods reads `v[lo] + w (v[hi] - v[lo])` with `hi` equal to `lo` or `lo + 1`, so one `O(n)` quickselect answers a quantile that used to cost `O(n log n)` (2^24 float32 at `q = 0.5`, M3 Pro: 1,096 ms to 151 ms, against NumPy's 53 ms). The gate result: MAX ships no partition, selection or quantile kernel in `linalg`, `nn`, `algorithm` or `layout`; the one device selection it has is `nn.top_k`, which `numax.core.sorting.top_k` already delegates to, and whose cost grows with `k` -- the median's `k = n / 2 + 1` is its worst case, so routing a median through it would order half the tensor to answer what one linear pass answers. The quickselect itself is numax's, not `std.builtin.sort.partition`'s: that function splits two ways, so a live range of equal values never shrinks and it goes quadratic (89 s for 2^18 equal elements, 12 s for 2^18 with one different, against 1 ms for distinct values), and a constant column is ordinary data. `numax/stats/quantiles.mojo`'s `_select_pair` splits three ways with a median-of-three pivot and a rounds budget that falls back to one sort. The vector-`q` overloads select per probability while `3 m < log2 n` and sort once above it. The `nan*` reductions are `isnan`, `select` and the plain reduction, composed rather than written, and stay host-side because `numax.core.sorting.select` does, even though the reduction underneath now runs on either target; `ptp`, `average` and `moment` likewise. `histogram`/`histogram2d`/`histogramdd`/`bincount`/`digitize` are host-side scatters -- `nn.gather_scatter`'s scatter stores rather than accumulates, so there is no bucketize to route to -- **`cov` and `corrcoef` run where the tensor lives**: the gate result is that MAX ships no covariance operator anywhere, but the `O(rows^2 n)` step inside one is a Gram matrix, so the disposition is extend-around-a-delegation -- `numax.stats.mean[axis=1]` (`Welford`) for the means, one `elementwise` to centre, and `linalg.matmul` with `transpose_b=True` for `C C^T`, which needs no transposed copy. `cov` scales by `1 / (n - ddof)` and `corrcoef` divides by the outer product of the diagonal's square roots, each one small `rows x rows` launch; the pair overload stacks its two vectors into a `2 x n` and reuses the matrix path, so there is one algorithm. Both take `gpu: Bool = False` last and fall back to the pre-0.2 host loop with a `stderr` line on a residency mismatch. Measured on an M3 Pro, `8 x 2^20` float32: 125 ms to 4.1 ms, against NumPy's 13.5 ms. The GEMM reassociates, so a `float32` covariance differs from the host loop's in the last bits and the matrix is no longer symmetric by construction -- `numpy.cov` computes it the same way. The rest of the correlation family (`pearsonr` through `linregress`, `rankdata`, `zscore`), the shape statistics (`skew` through `describe`) and the hypothesis tests (`ttest_*`, `chisquare`, `ks_1samp`, `f_oneway`, `mannwhitneyu`) are a few host sums plus a `t`/`chi2`/`f`/`norm` tail each, which is not worth a device pass. The nine distributions in `numax/stats/distributions.mojo` are `FloatLike` kernels with a `Tensor` overload of `pdf`/`pmf`, `cdf` and `ppf` beside each -- the kernel driven by `numax.core.tensor.map` with the distribution's parameters crossing the launch boundary as scalars, which is the case that `map`'s scalar-parameter overloads were added for. MAX ships uniform, normal and Gumbel *sampling* and no distribution function of any kind, so this is the Plain-only-surface outcome on top of tier-1 kernels. The `List[T]` forms of `variance`/`stddev`/`cumsum`/`mean` are `FloatLike`-generic — at `Compensated` they match a float64 reference where `Plain` drifts. Added in 0.2: `ks_2samp`, which walks both sorted samples advancing past ties on each side before reading the gap, and whose one-sided tails carry Hodges' finite-sample correction because that is what SciPy's `method="asymp"` uses and without it the tail reads 0.607 against SciPy's 0.472 for two samples of eight; and `wilcoxon`, dropping zero differences as SciPy's default `zero_method` does and taking average ranks on ties with the matching variance correction. Both are asymptotic only, the same declared divergence `ks_1samp` and `mannwhitneyu` carry |
| Sorting, searching, masking | `numax/core/sorting.mojo` | Tier 2. `argsort` routes into `nn.argsort`, `top_k` into `nn.top_k`, and the axis-wise `take`/`take_along_axis` into `nn.gather`/`nn.gather_elements` -- the **tensor** overload of `nn.gather`, not the closure one whose `input_fn` origins are unreachable, so `take` carries a real `gpu` parameter. Those are the names with a device path; the rest walk a host copy, where `std.builtin.sort` is the better route. `searchsorted` has no MAX counterpart at all, so its vectorized form is numax's own binary search. `put` was searched against `nn.scatter_elements` and `nn.scatter_nd` and takes neither: both want the indices as a tensor shaped like the output slice rather than the flat list `nonzero` and `argsort` hand back, so delegating would mean building the very thing the caller is avoiding. Added in 0.2: `compress`, whose condition is rank 1 and may be **shorter** than the tensor with the tail dropped, which is the one behavior separating it from `extract`; and `partition`/`argpartition`, which sort -- a full ordering satisfies the partition contract at every `kth` at once, so the answer is right at `O(n log n)` where introselect is `O(n)`. The upgrade is the three-way quickselect `numax/stats/quantiles.mojo` already runs, which cannot be called from `numax.core` because that package depends on no other, so sharing it means moving it down into a measured hot path |
| Dense linalg | `numax/linalg/` (`blas`, `triangular`, `cholesky`, `lu`, `qr`, `basic`, `misc`, `panel`) and `numax/linalg/array/` (the same split plus `eigen`) | Two tiers sharing one set of names, one tier per import. `numax.linalg` is `Tensor`: `matmul`/`matvec`/`batched_matmul` delegate to `linalg.matmul`/`bmm`, and `cholesky`/`lu_factor`/`qr_factor`/`solve`/`lstsq` are blocked and device-resident, with the `O(n^3)` trailing update fused into `linalg.matmul`'s epilogue -- MAX's only factorization, `linalg.qr_factorization`, is `LayoutTensor`-only and so denied. `numax.linalg.array` is `FloatLike`-generic and register-resident, where differentiability is the point. Both tiers now carry the spectra -- `eigh`/`eigvals`/`eigvalsh`/`svd`/`svdvals`, plus `sytrd`/`hessenberg`/`schur` and the Schur-form matrix functions on the `Tensor` side -- the `Tensor` tier's reductions blocked through `matmul` and its band iterations on the host, tier 2 and declared; the "What MAX actually ships" section above has the shape and `docs/performance.md` the measured cost of that split. `PivotedLU`/`TensorLU` are the tier-2 exceptions, since a pivot choice is a branch on data. **Known bug, 0.2, now guarded:** the device spelling of every spectral routine is wrong -- `sytrd[gpu=True]` disagrees with the host band at `n = 4` at every `block` including 1, so it is older than the `latrd` panel, and `svd[gpu=True]` raised instead of converging. The factorizations are unaffected. Nothing in numax, its tests, its examples or its benches named the spelling, which is why it survived. **`gpu=True` is now a compile error** on `sytrd`, `eigvalsh`, `eigh`, `hessenberg`, `eigvals`, `schur`, `gebrd`, `svdvals`, `svd`, the six `matfuncs` routines that reach `schur`, and `pinv`/`cond`/`matrix_rank`, each through a `comptime assert` in its body naming the routine. An assert rather than a `where` clause because a clause propagates: it would force `not gpu` onto every generic caller, costing `lstsq`'s working `"qr"` route its device spelling for the sake of its `"svd"` one. `docs/performance.md`'s "What is not measured" carries the diagnosis and there is no device spectral table until the path is fixed. Also here since 0.2: `orth`, `null_space` and `polar` off the same SVD (the first two returning a `Dynamic`, since their width is the rank and no type can see it), `tanm` solving `X C = S` through the transposed many-right-hand-sides form rather than inverting, and `rq` from the QR of a row-reversed transpose -- `J Rb^T J` is upper triangular because conjugating a lower triangular matrix by the reversal permutation makes it so, which is the whole trick and is why no `gerqf` is needed |
| Root finding and minimization | `numax/optimize/least_squares.mojo` and `numax/optimize/array/{solve,optimize}.mojo` | Two tiers on the `numax.linalg` pattern. `numax.optimize.array` is the conformer tier: fixed-iteration siblings in `solve` (tier 1), converge-to-tolerance in `optimize` (tier 2), and its `least_squares`/`curve_fit` take the Jacobian from `Gradient` rather than a difference. Its `minimize`, `minimize_scalar` and `root_scalar` are the `scipy.optimize` entry points over `bfgs`/`cg`/`nelder_mead`, `brent`/`golden`/`fminbound` and `brentq`/`bisect_tol`/`newton_tol`/`halley_tol`/`secant`, with a vector `root` routed to its own `least_squares`, dispatching on a `StaticString` `method` the way SciPy spells it -- searched at the pin and recorded as the **Plain-only surface** outcome, since MAX ships no optimizer of any kind (see "Everything algorithmic" above) and instantiating a driver at `Dual` or `Interval` adds no meaning where the objective is already the generic half. `numax.optimize` is `Tensor` and holds `minimize` (`bfgs` keeping its inverse Hessian on the device through `matvec` and `outer`, `cg` keeping one vector) plus `least_squares`/`curve_fit`, for problems whose vectors are long; it takes the Jacobian as an argument, since a `dtype`-monomorphic tensor cannot hold a `Gradient`, and sends its damped step through the augmented least-squares system `numax.linalg.lstsq` solves rather than the normal equations The `Tensor` tier's `minimize` now carries `bfgs`, `l-bfgs`, `cg` and `powell` with or without box bounds, `root` solves its Newton step through `numax.linalg.solve`, and `nnls`/`lsq_linear` form `A^T A` and `A^T b` on the device and run a projected Newton active-set iteration on the host -- all host drivers around MAX linear algebra, tier 2 by numax's definition |
| Quadrature and ODE | `numax/integrate/{quadrature,ode}.mojo` (`Tensor`), `numax/integrate/array/{quadrature,ode}.mojo`, `numax/integrate/integrate.mojo` | Two tiers on the `numax.linalg` pattern. `numax.integrate` is `Tensor`: `trapezoid`, `simpson` and `cumulative_trapezoid` over sampled values with `scipy.integrate`'s own signatures, `Plain`-only and host-side -- MAX ships no quadrature of any kind, and a weighted sum of samples at `Dual` means nothing, so this is the Plain-only-surface outcome. `rk4_system`, `dopri5` and `solve_ivp` over a `Tensor` state build every stage from one in-place `axpy` `elementwise` launch on the state's device -- the method-of-lines case, where the state is too large for registers -- and share the scalar `solve_ivp`'s controller. `numax.integrate.array` integrates a `FloatLike` *function* at a fixed node or step count and is tier 1. The adaptive drivers are tier 2, including `solve_ivp_stiff`, whose Newton iteration takes `df/dy` from `Dual`. Added in 0.2: `fixed_quad`, `gauss_legendre`'s `Float64` front door, and `dblquad` as a tensor-product rule over a **rectangle**. Both of `dblquad`'s divergences from SciPy -- constant inner bounds rather than `gfun`/`hfun`, fixed order rather than adaptive -- come from one constraint: `quad`'s integrand is a compile-time non-capturing function parameter, and the inner integral must capture the outer variable, so it cannot be handed to `quad` at all. Taking `f(x, y)` sidesteps it, in that argument order rather than SciPy's reversed `f(y, x)` |
| Interpolation | `numax/interpolate/interp.mojo` (`Tensor`) and `numax/interpolate/array/interp.mojo` | Two tiers on the `numax.linalg` pattern. MAX has no interpolation at arbitrary query points: `nn.resize_linear`/`nn.resize_nearest_neighbor` (host-only) and `nn.resize_bicubic` (`target` + `DeviceContext`) resample a whole NCHW image onto a fixed output grid by a scale factor, which is a different operation from `numpy.interp`'s and `scipy.interpolate`'s value-at-these-points, so both tiers are **extend** with nothing underneath. Over `Tensor`, `interp` and the splines bisect the knots inside one `elementwise` launch -- the data-dependent branch the `Array` tier's scan-and-blend spline exists to avoid. `CubicSpline`, `PchipInterpolator`, `Akima1DInterpolator` and `CubicHermiteSpline` share SciPy's `PPoly` form and take non-uniform knots; `CubicSpline`'s slopes are the rows `scipy/interpolate/_cubic.py` assembles, solved by `numax.linalg.solve_banded` on the host. `RegularGridInterpolator` is where the comparison with the resize kernels is closest and still fails: they take scale factors and return the whole resampled image, it takes query points on a rectilinear grid; two dimensions only, the `n`-D form waiting on a caller. `Chebyshev.fit(x, y)` is the least-squares fit to data through `lstsq`, beside the `Array` tier's nodal fit of a function. Added in 0.2: the legacy `numpy.poly*` family beside `horner` -- `polyval`, `polyder`, `polyint`, `roots` and `polyfit` -- all **descending**-coefficient where `horner` is ascending, because `polyfit`'s output has to feed `polyval` and `roots` without a reversal and `scipy.linalg.companion` is descending too. `roots` is `companion` fed to `eigvals`, which is how `numpy.roots` is implemented and what `companion`'s own docstring already pointed at; `polyfit` is `vander` fed to `lstsq`. So the whole family is two existing calls each under the names a caller looks for |
| Transforms and signal | `numax/fft/fft.mojo` and `numax/fft/array/fft.mojo`; `numax/signal/{convolution,windows}.mojo` (`Tensor`) and `numax/signal/array/signal.mojo` | Radix-2 and radix-4 at a power of two; over `Tensor`, Bluestein's chirp-z at any other length, three power-of-two transforms of `next_fast_len(2n - 1)` for one of length `n`. The `Array` tier stays power-of-two. `dct`/`dst` types I-IV over `Tensor` in `numax/fft/trig.mojo`, each a real projection of one complex DFT; MAX has no cosine or sine transform either. MAX's only transform is `nn.irfft` -- inverse-only, last-axis-only, NVIDIA-only -- so there is no forward FFT to route to at all and both tiers are numax's. Two tiers on the `numax.linalg` pattern: `numax.fft` is `Tensor`, `Plain`-only, a real/imaginary pair carried across `1 + ceil((log2(n) - 6) / 2)` device-resident launches -- one fused kernel doing the bit-reversal gather and the first six stages in registers, then one radix-4 kernel per remaining pair of stages, then a radix-2 kernel for an odd leftover, with the inverse `1/n` folded into the last kernel's stores rather than a pass of its own (7 launches at `n = 2^17` where one stage per launch plus a permutation was 18; `n <= 64` is a single kernel) -- for sizes an `Array` cannot hold; `numax.fft.array` is the `FloatLike` tier that differentiates at `Complex[Dual]` and runs per SIMD lane inside a kernel body. Signal convolution was gated against `nn.conv`: `conv_gpu` is `TileTensor`, rank 1-3, asymmetric padding, so a 1-D `full` convolution is expressible as a `[1, W, 1]` image against an `[R, 1, 1]` filter -- and it is GPU-only with a pack-the-filter CPU sibling (`conv_nhwc_direct`), two paths for one operation, tiled over the channels and filters a signal has one each of. numax writes the direct sum as one `elementwise` launch and `fftconvolve` on `numax.fft`, which MAX has no counterpart to at all. The `Tensor` windows are host tables, and so are the recursive filters `lfilter`/`filtfilt`/`sosfilt` -- a recurrence has no GEMM and no independent lanes, `numax.linalg.banded`'s grounds -- while `medfilt`, `detrend` and `savgol_filter` are a window per lane. Host-side there is still a spread of an order of magnitude, and 0.2 closed it: the recurrence reads and writes through `buffer.map_to_host()` (the accessor `to_host`/`copy_from_host` use, legitimate inside numax and nowhere else) instead of copying the signal into a `List[Float64]`, computes at `dtype` as SciPy does instead of widening, walks its coefficient and state `List`s through their own pointers, and runs in place -- `filtfilt` filters one extension buffer forwards then backwards over itself rather than materializing a reversed copy, `sosfilt` chains every section through the destination. M3 Pro, `float32`, `n = 2^20`: `lfilter` with a 32-tap FIR 99 ms to 10.6 ms against SciPy's 7.9 ms, `filtfilt` at order 4 from 32 ms to 12.4 ms against SciPy's 14.9 ms -- `filtfilt` is now ahead. The bounds check on a `List` index was the larger of the two costs, not the widening. `firwin` covers FIR design for every band shape at the `Tensor` tier and lowpass at the `Array` tier. **Reversed, and the follow-ups are done:** IIR design was recorded as out of scope because it needs complex poles and a `Tensor` holds no `Complex`; the design is a few dozen host `Float64` pairs and its result is real, so the whole of `scipy.signal.iirfilter`'s route is in -- `butter`, `cheby1`, `cheby2` and `ellip` in all four band shapes, the `iirfilter` front door over them, and `freqz`. The gate result for the record: **MAX has no filter design of any kind**, in `linalg`, `nn`, `algorithm` or `layout`; `nn` ships convolution and there is no prototype, no bilinear transform and no polynomial expansion anywhere to delegate to, so this is **extend** with nothing underneath. Two shapes are worth recording. The band forms double the order through `lp2bp`/`lp2bs`, and `TransferFunction` carries its order in its type while a `StaticString` cannot be constrained in a `where` clause, so the dispatch is the argument: a `Float64` edge designs at `order`, a `(low, high)` tuple at `2 * order`, and a `btype` the overload cannot serve raises. And `ellip` takes its complete elliptic integral from an arithmetic-geometric mean private to `design.mojo` rather than from the tier-1 `numax.special.elliptic_k`: the degree equation puts `K` under an exponential, and A&S 17.3.34's `2e-8` comes out as `1e-8` in the coefficients where the AGM gives `4e-13` -- a host tier-2 call site has no fixed-iteration obligation, so the two coexist for the two purposes. The spectral estimators frame a signal and transform every frame as one batch on the lane engine, which is the shape the `Tensor` tier exists for. Added in 0.2: `csd`, which is `welch` with `X conj(Y)` in place of `|X|^2` and so returns a complex pair, with `csd(x, x) == welch(x)` asserted rather than approximated; `coherence`, computing all three spectra in one lane pass over two framings because the ratio needs them at the same bin at once, and taking no `scaling` parameter since every factor cancels; `istft`, the least-squares overlap-add inverse, whose `nperseg` and `noverlap` are explicit parameters because the `keep` a caller receives is `stft`'s unevaluated `nperseg // 2 + 1` and the `where` prover folds neither `//` nor a `def` call, the same reason `irfft` takes its `n`; `decimate` in SciPy's `ftype="fir"` form with SciPy's own smaller `padlen`, which is what lets a signal only a few times longer than its taps be decimated at all; `peak_prominences`; `zpk2tf` by synthetic multiplication, discarding an imaginary part that is zero for any conjugate-closed root set, which is every real filter; and `oaconvolve`, which **is** `fftconvolve` -- SciPy's is the same convolution computed in blocks, so the answers agree and only the peak footprint differs |
| Tensor I/O | `numax/io/io.mojo`, `numax/io/npy.mojo` | Two formats: numax's own `NMX1` for numax-to-numax round trips (MAX ships no array I/O at all), and NumPy's `.npy` for interchange -- `numax.io.numpy.save` output is byte-identical to `numpy.save`, and `numax.io.numpy.load` reads `numpy.save` output, with no Python or NumPy dependency since the format is self-contained. `.npz` is out: it is a zip container |
| Random sampling | `numax/stats/random.mojo` | `std.random.philox.Random` -- MAX's own generator -- seeded per element from one scalar, so the fill is one `elementwise` body that runs threaded on the host or one thread per element on the device (`gpu=True`) and gives the same tensor on both. `nn.rand_uniform`/`nn.rand_normal` were checked and not called: their fill is an `OutputFn` bound to `RegisterPassable & ImplicitlyCopyable` and their seed a device pointer, graph-fusion machinery rather than an eager entry point. No `Random[FloatLike]` conformer: RNG is not differentiable, so the trait contract does not fit |

### Where the spelling differs, and why

Every name numax adds matches SciPy or NumPy exactly. A handful of older
ones do not, and each divergence is a decision rather than an oversight, so
they are listed here rather than aliased -- a second name for one thing is
the property `numax/prelude.mojo` exists to protect.

**Forced by Mojo.** `var` is a keyword and `std` is the standard library's
package, hence `variance` and `stddev`; `where` introduces constraint
clauses, hence `select`.

**Chosen.**

- `hilbert` on the flat surface is `scipy.linalg.hilbert`, the matrix; the
  `scipy.signal.hilbert` transform is `numax.signal.hilbert` and is not
  re-exported from `numax` or the prelude. SciPy carries the collision
  across two namespaces; numax's flat surface has one, and one name means
  one thing there, so the transform is reached by its subpackage.

| SciPy / NumPy | numax | Why |
|---|---|---|
| `inv` | `inverse` | spelled out, in the same spirit as `variance`/`stddev` above |
| `cho_factor` / `cho_solve` | `cholesky` / `cholesky_solve` | ditto; the abbreviation saves four characters and costs a reader the expansion |
| `qr` returning `(Q, R)` | `qr_factor` returning `TensorQR` | forced at the `Tensor` tier: a `Tuple` of two `Tensor`s cannot be destructured in Mojo 1.0, since `Tensor` is `Movable` and tuple unpacking wants `ImplicitlyCopyable`. `numax.linalg.array.qr` *is* the tuple-returning one |
| `lu_solve(factor, b)` | `TensorLU.solve(b)` / `PivotedLU.solve(b)` | the factorization object owns its solve, so the pairing cannot be got wrong |
| `res.fun` / `res.nit` / `res.success` | `f_x` / `iterations` / `converged` | `converged` says what it means where `success` does not; the result structs' docstrings carry the rest |
| `scipy.fft.next_fast_len(n)` returning the next 2-3-5-7-11-smooth number | `next_fast_len(n)` returning the next power of two | same contract -- the smallest length `>= n` the engine transforms at full speed -- and a different engine: pocketfft has a radix per small prime, numax has radix 2 and Bluestein. The name is kept because a caller padding to it gets what the name promises; the set differs because the engine does |
| `scipy.stats.ks_1samp` two-sided p-value by the exact Marsaglia-Tsang-Wang distribution; `mannwhitneyu` and `kendalltau` exact for small untied samples | the asymptotic Kolmogorov law for `ks_1samp`'s two-sided tail (SciPy's `method="asymp"`), the one-sided tails exact as SciPy's; `mannwhitneyu` and `kendalltau` always the tie-corrected normal approximation | the exact two-sided Kolmogorov law and the exact `U`/tau enumerations are substantial algorithms of their own, and at the sample sizes a device tensor holds the approximations agree with them; at `n = 16` `ks_1samp` differs in the second digit and the tests pin the asymptotic value on purpose |
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

Slicing as a first-class *owned* type, and fancy indexing. Borrowed slicing
is `View` over `a.view().tile[...]`/`.slice(...)`, which the `TensorLike`
bound lets a routine take in place of the tensor.

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
all. Those overloads are CPU-only because they launch through
`enqueue_function`, which needs the extent in the type — not because the
device does. `numax.core._drive` launches through
`max.algorithm.elementwise`, which computes its grid from a run-time
`Coord`, so the NumPy-named surface and `numax.stats`'s distributions run a
run-time-shaped tensor on the GPU.

Also absent, each a decision: sparse matrices, iterative solvers, distributed
execution, and dtype promotion. The first three are a different library's job;
the fourth is a compile error waiting to happen in a language that infers
parameters, so `astype` is explicit.

Reverse-mode autodiff is absent by measurement rather than by omission.
Every derivative conformer is forward-mode: `Dual` costs one pass per
direction and `Gradient[T, n]` carries all `n` partials through one pass,
at a cost that grows with `n` (sublinearly -- 32 variables cost about 12x
one, not 32x). A tape-based reverse mode was prototyped to a working,
numerically correct `FloatLike` conformer, ran inside a Metal kernel, and
was rejected head to head: 11x slower than `Gradient` at two variables,
break-even near sixteen, 2x ahead at thirty-two. Per-element kernels have
single-digit input counts, squarely where a tape loses. The trigger to
revisit is a real caller differentiating with respect to more than about
sixteen inputs, a fitted parameter vector say; the `Tensor` tier's
`minimize` takes `jac` explicitly for that reason, since a tensor cannot
hold a `Gradient`.

### The named gaps, one line each

Everything below was searched for during the 0.2 surface sweep and left
out on purpose. The rule this file runs on is that an unrecorded decision
did not happen, so each one says *why* rather than only *that*. A name
absent from both this list and the surface is an oversight and worth
reporting.

**NumPy.**

| Name | Disposition |
|---|---|
| `einsum` | **Out.** The subscript string is a parser plus a contraction planner, and every case a caller actually writes is already a name: `matmul`, `inner`, `outer`, `tensordot`, `trace`, `transpose`, `kron`. Revisit when a caller needs a contraction none of those spell. |
| `fftn`, `hfft` | **Deferred.** `fft`, `fft2`, `rfft`, `rfft2` and their inverses cover rank 1 and 2, which is what `numax.signal` consumes. `fftn` at arbitrary rank needs the transform applied along a run-time axis list, so the lane engine would take its axis as an argument rather than a parameter. `hfft` is `irfft` of a conjugate, one line, and waits with it. |
| `resize` | **Out.** NumPy's repeats-or-truncates semantics on a flat view; `reshape_dyn` plus `tile` says the same thing without the surprise that `resize` changes length. |
| `savetxt`, `loadtxt` | **Out.** Text I/O is a parser and a formatter, not numerics. `numax.io.numpy` reads and writes `.npy`, which is the interchange format that round-trips exactly; text does not. |
| `savez`, `.npz` | **Out**, recorded above: an archive of named arrays needs a zip container. |
| `eig` as one name | **Out.** `eigvals` returns the spectrum and `schur` the invariant subspaces; a combined `eig` would have to return non-orthogonal eigenvectors, which is the numerically worst of the three and the one LAPACK warns about. |
| `broadcast_arrays` | **Out**, recorded above: it returns a tuple of tensors and Mojo 1.0 cannot destructure one. |
| `vdot` | **Out**, recorded above: `dot` is the real case and there is no complex `Tensor`. |
| `fromfunction` | **Out.** `numax.core.tensor.map` over `arange` is the same thing with the kernel visible. |

**SciPy.**

| Name | Disposition |
|---|---|
| `tf2sos`, `sosfreqz`, `sosfiltfilt` | **Deferred, and it is a type rather than a name.** numax has no second-order-section representation: `butter` and friends return a `TransferFunction`, and its docstring already says an order above about eight should be run as a cascade it does not produce. `tf2sos` needs polynomial root-finding (now available as `roots`), conjugate pairing and gain distribution across sections. The right shape is an `Sos` struct beside `TransferFunction`, with `sosfilt` already here to consume it. |
| `zpk2sos`, `tf2zpk` | **Deferred** with the above, same missing type. `zpk2tf` is here, being the direction that needs no factoring. |
| `odeint` | **Out.** The name means LSODA -- automatic stiff/non-stiff switching -- and offering it over an explicit Runge-Kutta engine would misrepresent it. `solve_ivp` is the non-stiff route and `solve_ivp_stiff` sits beside it. |
| `romberg`, `quadrature` | **Out.** Both are extrapolation drivers superseded by adaptive Gauss-Kronrod in SciPy's own docs; `quad` is that driver and `fixed_quad` the non-adaptive one. |
| `dblquad` with variable bounds, `tplquad`, `nquad` | **Deferred.** `dblquad` over a rectangle is here; the general region needs a capturing integrand, which a compile-time function parameter cannot be. |
| `.rvs()` on the distribution namespaces | **Deferred, and the route is known.** Inverse-CDF sampling over `numax.stats.random.uniform` plus each namespace's own `ppf` gives it, and `ppf` already exists for all nine. What stops a one-line version is that the nine take different shape-parameter counts, so it is nine signatures rather than one generic, and partial coverage -- `norm.rvs` present, `gamma.rvs` absent -- is worse for a caller than none. `numax.stats.random` covers uniform, normal and exponential directly today. |
| `lognorm`, `weibull`, `cauchy`, `laplace`, `rayleigh`, `uniform` as distribution namespaces | **Deferred.** Nine families ship with the full method set. Each of these is a closed-form `pdf`/`cdf`/`ppf` triple and so is cheap; they wait on a caller rather than on a difficulty. |
| `shapiro`, `anderson` | **Out for now.** Both need tabulated critical values rather than a closed-form tail -- Royston's polynomial coefficients for one, Stephens' tables for the other -- which is checked-in reference data rather than an algorithm. `ks_1samp` against a fitted normal is the available goodness-of-fit test. |
| `linprog`, `differential_evolution`, `basinhopping`, `linear_sum_assignment` | **Out.** A simplex or interior-point LP, a population metaheuristic and the Hungarian algorithm are each a different field from the local, derivative-driven optimization `numax.optimize` covers. |
| `griddata`, `RBFInterpolator`, `BarycentricInterpolator` | **Deferred.** Scattered-data interpolation needs a spatial index (a Delaunay triangulation or a kd-tree) that numax has no data structure for. `RegularGridInterpolator` covers the gridded case at rank 2. |
| `interp1d`, `splrep`, `splev`, `make_interp_spline` | **Out as spellings.** `interp` is the linear case and `CubicSpline`, `PchipInterpolator` and `Akima1DInterpolator` are the cubic ones, each evaluating any derivative order. SciPy's own docs deprecate `interp1d` and recommend exactly those; a general-degree B-spline basis is the one real capability missing, and waits on a caller. |
| `polygamma`, `betaincinv`, `wofz` | **Deferred.** Each is a genuine new approximation with its own error analysis -- a recurrence plus asymptotic series, a Newton inversion of `betainc`, and the Faddeeva function -- rather than a composition of what is here. `digamma` covers the `n = 0` polygamma. |
| `eig_banded`, `expm_frechet`, `expm_cond`, `qz` | **Out**, each recorded at its own module: the banded eigen pipeline, the Frechet derivative of `expm`, its condition number, and the generalized Schur form. |
| `loadmat`, `savemat` | **Out.** MATLAB's container format; `.npy` is the interchange numax carries. |
| `bessel` filter design | **Deferred.** The other four IIR families ship. A Bessel design needs the reverse Bessel polynomial roots, which is a root-finder on a specific polynomial family rather than a variation on the analog prototypes already here. |
| `lfiltic`, `deconvolve`, `resample_poly` | **Deferred**, recorded at `numax/signal/filters.mojo`. |
