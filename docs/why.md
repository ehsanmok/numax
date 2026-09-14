# Why numax

> Companion to the top-level [README](../README.md). This page says what
> numax is for, what it can show that the alternatives cannot, and what it
> does not claim. Every claim here names the runnable file that proves it;
> the numbers live in [`performance.md`](performance.md) and only there.

numax is a Mojo library on MAX for writing a numerical algorithm once,
against a numeric type, and running it unchanged as a value, a derivative, a
bound or a higher-precision result, on the CPU's SIMD lanes or inside one GPU
thread, with NumPy's and SciPy's names on the surface.

That sentence is the whole positioning. It is not "a faster NumPy": the
elementwise benchmarks are good and they are secondary. The thing no other
library puts in one place is the intersection below.

## Three claims, each with its proof

### 1. Whole algorithms inside a kernel, at any numeric type

A decomposition, an ODE integrator or a Newton solve runs *inside one GPU
thread*, and *across SIMD lanes* on the CPU, from the same source, at any
`FloatLike` conformer. Not `exp(x)` mapped over a tensor: the entire
eigensolve, per thread.

- [`examples/advanced/quantum_well.mojo`](../examples/advanced/quantum_well.mojo):
  one function builds a 24x24 Hamiltonian and returns its lowest eigenvalue.
  The same function answers four questions, and all four also run one well
  per GPU thread over 256 wells: the energy at `Plain`, `dE0/dw` at `Dual`
  (the Hellmann-Feynman theorem, with no perturbation theory written), the
  frequency that hits a target energy through `newton` (each Newton step a
  `Dual` eigensolve, inside the thread), and the sweep itself.
- [`examples/advanced/ode.mojo`](../examples/advanced/ode.mojo): 1024
  trajectories, one `rk4` integration per thread, at `Plain` and again at
  `Dual` for the sensitivity to the initial condition.
- [`examples/advanced/gaussian_gpu.mojo`](../examples/advanced/gaussian_gpu.mojo):
  `Compensated` inside a kernel, recovering the rounding error `float32`
  drops.

What makes this possible is stated in every module docstring as a tier:
**tier 1** code has a fixed iteration count and no per-lane branch, so it is
launchable inside a GPU thread and usable at every conformer by construction.
Cyclic Jacobi at a fixed sweep count, RK4 at a fixed step, Newton at a fixed
iteration count. The tolerance-driven siblings exist beside them as tier 2,
`Plain`-only and host-side, and tier 1 never calls tier 2.

Who this is for: batched small eigensystems and factorizations, parameter
sweeps, per-trajectory ODE solves, an optimizer inside an outer simulation,
Monte Carlo with real per-sample numerical logic, sensitivities of special
functions, small estimation and control problems mapped across many threads.

### 2. Derivatives, bounds and precision through SciPy's algorithms, with no adjoint rule written

The numerical method itself is written against `FloatLike`, so the method,
not only the user's outer expression, takes on the type's meaning.

- `Dual` through `det`, `cholesky`, `solve`, `qr`, `eigh` and `lstsq`:
  [`tests/linalg/test_linalg.mojo`](../tests/linalg/test_linalg.mojo), and
  [`examples/intermediate/npy_to_cholesky.mojo`](../examples/intermediate/npy_to_cholesky.mojo)
  for `d(det A)/dA[0,0]` on a matrix NumPy saved.
- `Dual` through an integral and through every special function:
  [`examples/intermediate/quadrature.mojo`](../examples/intermediate/quadrature.mojo),
  [`examples/intermediate/special_functions.mojo`](../examples/intermediate/special_functions.mojo).
- `Gradient[Dual[Plain], n]` is a Hessian by nesting, `Complex[Dual[Plain]]`
  differentiates holomorphically:
  [`examples/basic/hessian.mojo`](../examples/basic/hessian.mojo),
  [`examples/basic/complex.mojo`](../examples/basic/complex.mojo).
- `Interval` and `Decimal` through the special functions:
  [`tests/special/test_enclosure.mojo`](../tests/special/test_enclosure.mojo)
  pins containment for `erf`, `erfc`, `gaussian`, `j0`, `j1` and exact
  decimal arithmetic through a kernel;
  [`examples/intermediate/enclosures.mojo`](../examples/intermediate/enclosures.mojo)
  shows the same, and shows `gamma` at `Interval` returning `[nan, nan]`,
  which is where this claim stops (see below).

SciPy differentiates by finite difference. Here the derivative falls out of
the same kernel, exactly, because the type carries it.

### 3. The missing `scipy` for MAX, built MAX-first

MAX ships `matmul`, batched and grouped GEMM, reductions, softmax, padding,
top-k, gather, cumsum and an inverse real FFT on `TileTensor`. It ships no
factorization, no eigensolver, no forward FFT, no quadrature, no optimizer,
no distribution. numax's rule is a gate, not a preference: before any kernel
is written, MAX is searched and the result recorded in
[`parity.md`](parity.md). What MAX has is **delegated**. What it lacks is
**extended** in MAX's idiom, `TileTensor` in and out, a `target` parameter,
a `DeviceContext`, the `O(n^3)` term pushed back into `linalg.matmul`, so
it stays upstreamable. Only what MAX cannot express, the register-resident
`Array[T, n]` tier that carries the conformers, **diverges**.

The result is the dense-LAPACK-shaped surface MAX does not have, on every
device `linalg.matmul` reaches, with no per-vendor code anywhere in numax:
`rg has_nvidia numax/` is empty, and the whole of numax's device
participation is `target="gpu"`.

Under all three, a habit worth naming once: CPU and GPU numbers are never
mixed into one comparison, every approximation documents an error bound that
`pixi run accuracy` checks against mpmath at 50 digits, `Plain` carries a
one-ulp `exp`/`ln`/`erf` at `float64`, and the README's performance section
has a paragraph titled "where this version is slow".

## Against the alternatives

No competitor lacks all of these capabilities. numax's value is the
intersection.

| Alternative | Where it is ahead | numax's distinction |
|---|---|---|
| NumPy/SciPy | maturity, breadth, dynamic indexing, sparse, the ecosystem | compiled kernels on any device; the algorithm, not only the expression, is polymorphic in its numeric type |
| CuPy | a mature CUDA drop-in with cuBLAS, cuFFT, cuSOLVER, cuSPARSE behind it | vendor-neutral through MAX; algorithms run *inside* kernels rather than being dispatched to opaque ones |
| JAX | reverse mode, `vmap`, sharding, program transformations | semantics as Mojo types (`Interval`, `Compensated`, `Decimal` have no JAX analogue); no tracer, ahead-of-time |
| PyTorch | reverse mode, the neural-network ecosystem | scientific-computing scope; explicit kernel construction |
| Julia + ForwardDiff (+ CUDA.jl) | the same generic-numerics model, nestable duals, a richer AD ecosystem | one device story across vendors via MAX; a *declared* per-function GPU-launchability tier; a SciPy-shaped catalogue in one AOT-compiled systems language |
| Arb/FLINT | rigorous ball arithmetic, arbitrary precision | enclosures compose with the kernel architecture, with the soundness caveat stated below |
| Numba/Triton | flexible custom compilation and kernel authoring | a library of algorithms, not a kernel-authoring tool |

Julia deserves the honest sentence: ForwardDiff's dual numbers do run inside
CUDA.jl kernels over `StaticArrays`. numax does not claim the primitive. It
claims the combination: the same kernel batching across SIMD lanes on the
CPU and across threads on Metal, CUDA or ROCm through one `target="gpu"`;
every tier-1 function documented as launchable rather than discovered to be;
and NumPy's and SciPy's names on top, in a language with no JIT.

## What numax does not claim

- **Speed of the spectral decompositions, the FFT, or the host-loop
  statistics.** On `Tensor` at `n = 1024`, `eigh`, `svd` and `schur` are at
  0.002-0.11 of LAPACK because eigenvector accumulation is still a host
  loop; a `2^17`-point FFT round trip is 14x behind pocketfft; `quantile`,
  `cov` and `lfilter` are at 0.05-0.11 of NumPy and SciPy. The surfaces
  that are one `elementwise` launch or one batched transform are ahead
  (`norm.cdf` 5.4x, `interp` 9.4x, `welch` 12x, on the processor named in
  [`performance.md`](performance.md)).
- **A drop-in NumPy or CuPy replacement.** No fancy indexing, no owned
  slicing, no dtype promotion (`astype` is explicit), and run-time-shaped
  tensors are CPU-only because a GPU launch needs the extent in the type.
- **Every SciPy operation at every conformer.** The `Tensor` tier is
  `dtype`-monomorphic by construction: a `DeviceBuffer` holds a machine
  scalar, and the conformers are Mojo structs. Conformers live on
  `Array[T, n]`, and `to_tensor` lowers `Plain` only. The two tiers are a
  real choice, not a staging area; [`architecture.md`](architecture.md)
  has the seam.
- **Reverse-mode autodiff or large parameter vectors.** `Gradient[T, n]` is
  forward-mode and its cost grows with `n`. A tape was prototyped, ran
  inside a Metal kernel, and lost to `Gradient` below about sixteen
  variables; [`parity.md`](parity.md) records the numbers.
- **Sound interval arithmetic, or an enclosure through every special
  function.** Bounds are computed in round-to-nearest; `inflate` is the
  opt-in widening. Monotone kernels enclose exactly and polynomial branches
  loosely, but an alternating rational approximation such as Lanczos'
  `gamma` yields `[nan, nan]` at `Interval`, because the sum's interval
  crosses zero before its `ln`. That is a property of interval arithmetic,
  and the example shows it rather than hiding it.
- **Sparse matrices, Krylov solvers, distributed execution.** A different
  library's job.
- **ROCm numbers.** ROCm is reached because `linalg.matmul` dispatches it
  and numax names no architecture; nothing on any page is an AMD
  measurement.
- **Stability.** Young and experimental, pinned to `mojo ==1.0.0` and
  `max ==26.5`; APIs may change.

## Where to read next

- [`architecture.md`](architecture.md): the trait, the two tiers, the
  fixed-iteration invariant, the tensor and GPU layer.
- [`parity.md`](parity.md): what MAX ships, what numax delegates, extends,
  or leaves out, and why.
- [`features.md`](features.md): the full public inventory.
- [`performance.md`](performance.md): every number, per processor, with
  the harness that produced it.
