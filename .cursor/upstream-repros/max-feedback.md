# MAX kernel library feedback

Missing kernels, wrong abstractions and awkward APIs in **MAX**, found
while building [numax](https://github.com/ehsanmok/numax), a numerical
computing library that delegates to MAX wherever MAX ships a kernel.
Companion document: [`mojo-feedback.md`](mojo-feedback.md) for the
language and standard library.

Each entry stands alone: the offending signature quoted inline, a repro
that does not import numax, the diagnostic as printed, what the behavior
should be, and what the gap cost. Every repro is a file in this
directory, runnable with the `pixi run repro` command shown beside it.
See [`README.md`](README.md) for setup, the severity legend and the full
index.

numax's standing rule is that re-implementing something MAX already ships
is a defect, so every absence below was confirmed across all four
top-level kernel roots (`linalg`, `nn`, `algorithm`, `layout`) rather
than the one the subsystem's name suggests.

**Pin: MAX 26.5, Mojo 1.0.0 (ed45d567).** Every diagnostic below was
produced by that toolchain on an Apple M3 Pro, except where an entry
names different hardware.

---

# 2. MAX kernel library

numax's standing rule is that re-implementing something MAX already ships
is a defect, so every item here was confirmed absent before it was
written. The searches covered all four top-level kernel roots (`linalg`,
`nn`, `algorithm`, `layout`) rather than the one the subsystem's name
suggests.

## 2.1 GEMV writes past the end of a destination whose row count is not a lane multiple

**Severity: crash.** The most severe item in this file.

**Environment: reproduced on x86-64 Linux (GitHub Actions
`ubuntu-latest`), not reproducible on the Apple M3 Pro of the preamble.**
That asymmetry is the point, and it is why this was expensive to find:
on Apple, `linalg.matmul` dispatches to Accelerate, which is correct, so
a macOS test suite stays green while the same call dies on x86. The
harness below was run locally at `m` in {1, 2, 3, 5, 6, 7} with float32
and reported no overwrite on any of them; it is included so the defect
can be confirmed on hardware that reaches MAX's own GEMV kernel.

`linalg.matmul` routes an `n == 1` product to a GEMV kernel that stores
its output in whole SIMD vectors with no masked tail. A destination with
`m` rows where `m % simd_width_of[dtype]() != 0` is therefore written
past its end -- a crash or a corrupted neighbor, not a wrong answer.

This harness detects the overwrite with a sentinel rather than waiting
for a fault, so it does not depend on where the allocation happens to sit
relative to a page boundary.

`pixi run repro 2.1-gemv-destination-overwrite.mojo` -- edit `m` and `k`
at the top to sweep shapes.

```mojo
from layout import Coord, TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu.host import DeviceContext
from std.sys.info import simd_width_of

comptime dtype = DType.float32
comptime m = 2       # rows of the destination
comptime k = 12      # contraction length
comptime guard = 8   # sentinel elements allocated after the destination
comptime sentinel = Scalar[dtype](-777.0)


def main() raises:
    var ctx = DeviceContext(api="cpu")
    comptime lanes = simd_width_of[dtype]()
    print("simd_width =", lanes, " m =", m, " m % lanes =", m % lanes)

    var a_buf = ctx.enqueue_create_buffer[dtype](m * k)
    var x_buf = ctx.enqueue_create_buffer[dtype](k)
    var y_buf = ctx.enqueue_create_buffer[dtype](m + guard)
    ctx.synchronize()

    var ap = a_buf.unsafe_ptr()
    var xp = x_buf.unsafe_ptr()
    var yp = y_buf.unsafe_ptr()

    for i in range(m * k):
        ap[unsafe_offset=i] = Scalar[dtype](1.0)
    for i in range(k):
        xp[unsafe_offset=i] = Scalar[dtype](1.0)
    for i in range(m + guard):
        yp[unsafe_offset=i] = sentinel

    # Hand MAX a destination that claims only the honest `m` rows.
    var a_t = TileTensor(ap, row_major(Coord(m, k)))
    var x_t = TileTensor(xp, row_major(Coord(k, 1)))
    var y_t = TileTensor(yp, row_major(Coord(m, 1)))

    matmul[target="cpu"](y_t, a_t, x_t, ctx)
    ctx.synchronize()

    var clobbered = 0
    for i in range(m, m + guard):
        if yp[unsafe_offset=i] != sentinel:
            clobbered += 1
    print("elements written past the destination:", clobbered)

    _ = a_buf^
    _ = x_buf^
    _ = y_buf^
```

**Expected.** `clobbered` is 0 for every `m`. The kernel should mask its
final store.

**Impact.** Two separate crashes in CI, each of which passed on every
developer machine before it was pushed. A matrix-vector product at
`m = 6, k = 4` was the first; a tensor contraction leaving one output
column was the second, where `tensordot` of a `(2, 3, 4)` against a
`(3, 4)` is `m = 2, k = 12, n = 1`. Both were architecture-dependent and
neither was a wrong answer, so the failure surfaced as an unexplained
segfault on one runner.

**Workaround.** Three call sites in numax now defend against this
individually. The matrix-vector path grows its matrix to a lane multiple
and zeros the phantom rows; the contraction path bypasses `matmul`
entirely for a single-column result and issues one elementwise dot
product instead; a one-column product inside the QR factorization is
padded to two columns through three GEMMs. The cheapest general fix for a
caller is to over-allocate the destination, hand MAX a view claiming only
the real rows, and let the tail land in slack -- but that is a workaround
every MAX caller has to know about, which is the real cost.

## 2.2 The only dense factorization MAX ships takes `LayoutTensor`, not `TileTensor`

**Severity: blocked.**

`pixi run repro 2.2-qr-layouttensor-only.mojo`

```mojo
from layout import Coord, TileTensor
from layout.tile_layout import row_major
from linalg.qr_factorization import qr_factorization
from max.gpu.host import DeviceContext

comptime dtype = DType.float64
comptime n = 4


def main() raises:
    var ctx = DeviceContext(api="cpu")
    var a_buf = ctx.enqueue_create_buffer[dtype](n * n)
    var tau_buf = ctx.enqueue_create_buffer[dtype](n)
    ctx.synchronize()

    var a = TileTensor(a_buf.unsafe_ptr(), row_major(Coord(n, n)))
    var tau = TileTensor(tau_buf.unsafe_ptr(), row_major(Coord(n)))

    qr_factorization(tau, a)
```

Observed:

```
error: invalid call to 'qr_factorization': value passed to 'sigma' cannot be
converted from 'TileTensor[DType.float64, Layout[*?, *?], origin_of(tau_buf)]'
to 'LayoutTensor[...]'
note: function declared here
def qr_factorization[dtype: DType, element_layout: Layout](sigma: LayoutTensor[...], A: LayoutTensor[...])
```

The quoted signature also shows no `target` parameter and no
`DeviceContext`, so it is CPU-only regardless of the tensor type.

**Expected.** `TileTensor` in and out, a `target: StaticString` parameter
and an optional `DeviceContext`, matching `linalg.matmul` and the rest of
the root.

**Impact.** numax's interop rule is `TileTensor`-only, so this API is
denied rather than bridged, and numax writes its own blocked Householder
QR. `linalg.math.outer_product_acc` is denied for the same reason, with
the additional problem of being accumulate-only. Separately: MAX ships no
Cholesky, no LU, no general solve, no triangular solve, no SVD and no
eigensolver at all, so numax implements all of them in MAX's idiom.

## 2.3 `matmul`'s `transpose_a` parameter exists but is not implemented

**Severity: friction.**

`pixi run repro 2.3-matmul-transpose-a.mojo`

```mojo
matmul[target="cpu", transpose_b=True](ct, at, bt, ctx)   # fine
matmul[target="cpu", transpose_a=True](ct, at, bt, ctx)   # not
```

Observed on the second line only:

```
note: call expansion failed with parameter value(s): ("transpose_a": True, ...)
max/kernels/src/linalg/matmul/__init__.mojo:152:9: note: constraint failed: transpose_a not yet supported
mojo: error: failed to run the pass manager
```

**Expected.** Implement it, since the parameter is already declared and
the intent is clearly there.

**Impact.** An `A^T B` product needs a materialized transpose, which is a
full pass over the data and an allocation. That product is the trailing
update of a blocked factorization, so it is on the hot path of every
factorization numax ships: Cholesky, LU, QR, and the three reductions
behind the eigensolvers and the SVD.

## 2.4 No symmetric or triangular BLAS-3

**Severity: blocked (performance).**

There is no `syrk`, `symm`, `trmm`, `trsm` or `potrf` on `TileTensor` in
any of the four kernel roots. The vendor libraries have them, but only
behind MAX's private cuBLAS and rocBLAS bindings, which are per-vendor
and therefore unusable by a library that refuses to name an
architecture.

**Impact, quantified.** A Cholesky factorization's trailing update is a
symmetric rank-k update. Expressed through general `matmul` it computes
the full square and discards half, so the factorization does twice the
necessary flops in its dominant term. numax restricts the update to the
lower block triangle to claw back part of that, but the remaining
distance needs the `syrk` shape itself. This is the single largest
remaining gap between numax's Cholesky and LAPACK's.

## 2.5 No forward FFT

**Severity: blocked.**

MAX ships `nn.irfft` and nothing else: inverse only, real only, last axis
only, and NVIDIA-only (it is a `_cufft` binding). There is no forward
transform, no complex transform, no DCT and no DST anywhere in the four
roots, and nothing at all on Metal or AMD.

**Impact.** numax implements a complete FFT engine -- fused bit-reversal
gather plus radix-4 stages in registers, Bluestein for non-power-of-two
lengths, real transforms, 2-D, DCT and DST. It is currently about 3x
behind pocketfft on a complex 2^20 transform and 4-5x elsewhere, which is
the honest cost of a downstream library doing this itself.

## 2.6 `nn.cumsum` has no device path

**Severity: blocked.**

`pixi run repro 2.6-cumsum-no-target.mojo`

```mojo
from nn.cumsum import cumsum
cumsum[target="gpu"](y, x, ctx)
```

Observed:

```
error: invalid call to 'cumsum': unexpected argument
    cumsum[target="gpu"](y, x, ctx)
                               ^~~
note: function declared here
def cumsum[dtype: DType, exclusive: Bool, reverse: Bool, *, axis: Int](output: TileTensor[...], input: TileTensor[...])
```

The signature takes no `target` and no `DeviceContext`. There is no
`cumsum_gpu` sibling, and no `cumprod` at any target.

**Expected.** A `target` parameter, as the rest of `nn` has, or a `_gpu`
sibling.

**Impact.** A cumulative sum of a device-resident array has to come back
to the host. numax delegates the CPU case and walks the host for
everything else, and implements `cumprod` entirely on the host.

## 2.7 Other absent kernels, confirmed

Each was searched for across all four roots before numax implemented it.
Listed compactly because the pattern is the same: MAX has no kernel, so a
downstream library writes one.

| Missing | What numax had to do |
|---|---|
| `linalg.transpose` has no working device path (every path it reaches is a host memcpy) | Custom elementwise gather for the GPU case |
| Quantile, selection, partition | Three-way quickselect |
| Covariance, correlation | Welford means plus a Gram matrix through `matmul` |
| Scatter-accumulate (`nn.gather_scatter` stores, never accumulates) | Host-side histograms |
| General device sort and argsort (only `nn.top_k`) | Host sorts |
| `kron`, `inner`, `matrix_power` | Elementwise or `matmul` |
| 1-D convolution (`nn.conv` is shaped for multi-channel NHWC; the GPU entry point is GPU-only and the CPU one needs a packed filter) | Direct elementwise dot products, plus an FFT route |
| Out-of-place broadcast (`TileTensor` broadcast is in-place only) | Stride arithmetic |
| Any optimizer, quadrature rule or ODE integrator | The whole of two subpackages |
| `nn.argmaxmin` handles the innermost axis only | Delegate innermost, host walk otherwise |
| float64 GPU matmul (the GEMV path uses `warp.shuffle` with no float64 case) | GPU benchmarks and examples are float32 only |
| About half of `std.math` is not GPU-linkable (`gamma`, `lgamma`, the Bessel family; float32 Bessel symbols are missing on Metal) | Own GPU-launchable approximations for every special function |

`linalg.grouped_matmul` deserves a specific note: it is the closest thing
to a batched primitive, but it handles ragged `M` only with static `N` and
`K`, and is GPU-only, so it cannot batch the differently-shaped block
updates a factorization issues.

## 2.8 No sparse linear algebra, and a misleading `spmv`

**Severity: blocked.** An entire problem domain, not one kernel.

There is no sparse matrix representation and no sparse kernel anywhere in
MAX. Searched across every root, in both the pinned release and `main`:
`csr`, `csc`, `coo`, `bsr`, `ell`, `sparse`, `spmv`, `spmm`, `csrmv`,
`csrmm`, `nnz`, `indptr`, `cusparse`, `rocsparse`, `hipsparse`. At the
26.5 pin, `strings` over the shipped `linalg` and `algorithm` packages
returns zero hits for `spmv`, `csrmv`, `csr_matvec`, `cusparse` and
`SparseMatrix`.

Three things look like sparse support and are not. Worth naming, because
each one costs a reader time:

- **`cublasSspmv` / `rocblas_sspmv` in the vendor FFI are dense.** The
  name is BLAS-2's *packed symmetric* matvec -- rocBLAS's own docstring
  says "A should contain an upper or lower triangular n by n packed
  symmetric matrix". Nothing in `linalg/` calls them, and there is no
  cuSPARSE or rocSPARSE binding beside the cuBLAS and rocBLAS ones.
- **CSR appears only as batching metadata.** `nn/moe.mojo` scans a
  histogram into "CSR offsets" for token-to-expert grouping,
  `nn/sampling` stores frequency penalties "in a CSR format", and the
  ragged attention path carries `seqlens_qo_indptr` / `pages_kv_indptr`.
  These are prefix sums over batch elements, not matrix structure.
- **"Sparse attention" is a sparse access *pattern*.** The sparse MLA
  kernels gather a subset of dense KV rows by index. There is no `y = A x`
  for a general sparse `A`.
- **The only real SpMV in the repository is a tutorial.**
  `max/kernels/examples/pmpp/chapter_17/` has `COOMatrix` and
  `spmv_coo_kernel`, which is teaching material under `examples/`, not a
  shipped kernel.

**Impact.** numax claims no sparse support and cannot, because the
foundation is a sparse matvec and there is nothing to delegate to. Writing
one means picking a storage format, implementing it per target, and
carrying it -- which is the "re-implementing MAX" failure mode in reverse:
the work is real but there is nothing to route to.

**Expected.** One sparse format (CSR is the obvious first) with matvec and
SpMM, or a cuSPARSE/rocSPARSE binding on the pattern the cuBLAS and
rocBLAS ones already establish. Even a single-target CPU CSR matvec would
give a downstream library something to build on.

## 2.9 No iterative solvers, and the pieces to write them are missing too

**Severity: blocked.**

No conjugate gradient, GMRES, BiCGSTAB, MINRES, Arnoldi, Lanczos, or
preconditioner (ILU, Jacobi) in any root. At the pin, `strings` over the
shipped packages returns zero for `gmres`, `bicgstab`,
`conjugate_gradient`, `krylov` and `arnoldi`. The `lanczos` and
`conjugate` hits that do exist are an image-resize filter and BLAS's
"conjugate of x".

This entry is separate from 2.8 because it would not be fixed by sparse
storage alone. A Krylov solver is a loop whose iteration count depends on
a residual, so it needs device-resident state that survives across
iterations -- and that is exactly what **3.1** below breaks. numax already
hit this in its optimizers, which keep their iterate on the host as an
ordinary list and rebuild a tensor per evaluation, paying a round trip per
iteration. A CG solver written on today's API would inherit that.

**Impact.** Both halves of "sparse and iterative" are out of scope for
numax, and they are the two things a PDE or large-scale optimization user
asks for first.

**Expected.** Sparse matvec (2.8), plus a context lifetime that survives
reassignment (3.1). Those two make Krylov methods ordinary library work
rather than a fight with the runtime.

---

# 3. MAX API shape

The kernel exists; the signature makes it hard to use well.

## 3.1 `DeviceContext` obtained from a buffer dies when the owning tensor is reassigned

**Severity: crash.**

A context handle acquired from a device buffer does not keep the
underlying device alive across a reassignment of the tensor it came from.
Reassigning the source tensor -- the ordinary shape of an iterative
solver, where each step produces a new state -- invalidates a context
captured before it, and the next enqueue on that handle crashes with no
diagnostic. Self-reassignment specifically is what triggers it; holding
the context while an unrelated tensor is replaced is fine.

**Impact.** This decided the architecture of numax's optimizers. An
iterative method wants to hold device state and update it in place; it
cannot, so `numax.optimize` keeps its iterate on the host as an ordinary
list and rebuilds a tensor per function evaluation. That is a host round
trip per iteration, in the one subsystem where iteration count is the
whole cost.

**Expected.** A context handle that keeps its device alive, or a
documented lifetime rule and a diagnostic when it is violated.

## 3.2 `target` is a compile-time parameter, so device selection cannot be a runtime decision

**Severity: friction.** This one shapes every public signature in numax.

Every MAX kernel takes `target: StaticString`, resolved at compile time.
A program that wants to use the GPU when one is present therefore cannot
ask at runtime -- `ctx.api()` returns a string, and a string is not a
parameter. The only way to branch is `comptime if`, which compiles both
paths, so a CPU-only build would contain GPU kernels for hardware it will
never see.

**Impact.** numax's entire public API carries a `gpu: Bool = False`
compile-time parameter to pass through to `target`: `gpu: Bool` appears
396 times across 45 of its modules. That parameter is visible to users,
who must thread `gpu=True`
through every call in a program; operators cannot spell it at all, so
`a + b` silently takes the host path on device-resident data while
`add[gpu=True](a, b)` does not. The convention is contagious: it appears
in every signature, every docstring and every example.

**Expected.** A runtime dispatch path, or a build-level target selection
so a library does not have to expose the choice in its own API.

## 3.3 `DeviceContext` cannot cross the `enqueue_function` boundary

**Severity: blocked.**

A kernel launched through `enqueue_function` cannot take a
`DeviceContext` as an argument, and cannot be `raises`. Multi-core CPU
execution needs the context; a GPU launch needs `enqueue_function`.

**Impact.** numax cannot unify its CPU-threaded and GPU paths under one
entry point. Its threaded-CPU primitive is a separate function with
`target` hardcoded to `"cpu"`, which is a permanent asymmetry in what is
otherwise a single `gpu: Bool` surface. Relatedly, panel kernels inside
the factorizations cannot use host threading in the phases that need a
barrier, so those phases run single-block.

## 3.4 `enqueue_function` rejects kernels whose layout carries a `where` clause

**Severity: friction.**

A generic launch helper parameterized over a kernel whose signature
constrains its layout fails with `no matching method in call to
'enqueue_function'`, with no indication that the layout constraint is the
reason.

**Impact.** numax has two launch policies instead of one. Kernels that
can go through `enqueue_function` do; the FFT, the statistical
distributions, and the whole NumPy-named surface go through
`max.algorithm.functional.elementwise` with a capturing closure instead.
Having both is not a design choice, and the reason is invisible from
either call site.

## 3.5 `rand_uniform` and `rand_normal` cannot take a capturing output function

**Severity: friction.**

Their `OutputFn` parameter must be `RegisterPassable`, so a capturing
closure is rejected; the seed arrives through a device pointer; and the
shape is graph-operator-oriented rather than array-oriented.

**Impact.** numax does not use them. It seeds a Philox generator per
element inside a single elementwise pass, which is the same underlying
generator without the output-function constraint.

## 3.6 `ArgMax` leaves the answer somewhere other than where it is named

**Severity: wrong.**

The winning index has to be read from `acc_indices[0]`. The
similarly-named `best_idx` field is stale on the tiled path, so reading
the obviously-named one gives a plausible wrong answer rather than an
error.

**Impact.** Found by a failing test after the natural spelling was used.
This is the kind of thing that survives code review.

## 3.7 An `imm` capture writes zeros on Metal

**Severity: wrong.**

A closure forwarded to `elementwise` that captures its operands with
`{imm xs}` produces all-zero output on Metal; capturing with `var`
instead works. No diagnostic either way.

**Impact.** Silent wrong answers on one backend, from a capture-mode
annotation that reads as the more conservative choice.

## 3.8 Two different `algorithm` roots, and kernel families split across modules

**Severity: friction (discoverability).** Cheap to fix, and it cost real
work.

Two traps, the same shape:

- `max.algorithm.functional` exports `elementwise` and little else, while
  the **top-level** `algorithm` root is a reduction library -- the
  `reduce_op` monoids driven by the `rowwise` CPU and GPU scaffolder.
  Reading the first as the second is how numax came to hand-write a
  reduction engine that MAX already shipped. The same distinction exists
  between `max.linalg` (nothing) and the top-level `linalg` (the GEMM
  family).
- A kernel family is not one module. `nn.pad` has no `target` and no
  `DeviceContext`, so reading it alone says padding is host-only -- but
  `nn.pad_gpu` sits beside it with the device path for the constant mode,
  spelled in raw pointers rather than `TileTensor`. A `_gpu` sibling or a
  differently spelled entry point is the common case, not the rare one.

**Impact.** One hand-written reduction engine, later deleted and
re-delegated. Repeated near-misses on padding, argmax and softmax.

**Expected.** A single index of kernels by operation name, saying which
root and module owns each and which targets it reaches. This is
documentation, not engineering.

## 3.9 `docs_check_imports` reports false negatives on nested kernel paths

**Severity: friction (tooling).**

The documentation MCP's import checker rejects valid nested Mojo kernel
paths, including `linalg.matmul` and `max.gpu.host`. `docs_get` on the
package page is reliable; the checker is not.

**Impact.** An agent or developer verifying an import against the docs
gets told a correct path does not exist, which is worse than no check.

## 3.10 Device storage is keyed on `DType`, so a user struct cannot be a tensor element

**Severity: blocked.** The largest ask in this file, and the only one that
is genuinely a design question rather than a defect.

`DeviceBuffer` and `TileTensor` are parameterized on `dtype: DType`, which
is a closed enum of machine scalars. There is no way to build a device
tensor whose element is a user struct, however simple that struct is and
however trivially it satisfies `TrivialRegisterPassable`.

`pixi run repro 3.10-struct-element-tensor.mojo` -- as shipped it runs the
structure-of-arrays workaround; uncomment either attempt to see the
rejection.

```mojo
@fieldwise_init
struct Dual(Copyable, Movable):
    """A value and its derivative. Two floats, no pointer, no allocation."""

    var value: Float32
    var deriv: Float32
```

Attempt 1, allocate a buffer of it:

```
error: invalid call to 'enqueue_create_buffer': 'enqueue_create_buffer'
parameter 'dtype' has 'DType' type, but value has type 'AnyStruct[Dual]'
    var buf = ctx.enqueue_create_buffer[Dual](16)
              ~~~~~~~~~~~~~~~~~~~~~~~~~ ^~~~
note: function declared here
def enqueue_create_buffer[dtype: DType](self, size: Int) -> DeviceBuffer[dtype]
```

Attempt 2, name a `TileTensor` over it:

```
error: 'TileTensor' parameter 'dtype' has 'DType' type, but value has type 'AnyStruct[Dual]'
    var t = TileTensor[Dual, type_of(row_major(Coord(4, 4)))]
                       ^~~~
note: 'TileTensor' declared here
struct TileTensor[mut: Bool, //, dtype: DType, LayoutType: TensorLayout,
  origin: Origin[mut=mut], *, Storage: TensorStorage = PointerStorage,
  address_space: AddressSpace = AddressSpace.GENERIC,
  linear_idx_type: DType = _get_index_type[LayoutType](address_space)]
```

### Why the existing extension point cannot absorb this

`TileTensor` already has a trait-based seam for storage, which makes this
look like a small change. It is not, and the reason is worth stating
precisely: **the seam is itself `DType`-keyed.**

At the 26.5 pin the parameter is `Storage: TensorStorage = PointerStorage`
(quoted above). On `main` it has been renamed to
`Engine: TensorEngine = DefaultEngine[element_width=1]`, and the trait
reads:

```mojo
trait TensorEngine:
    comptime element_size: Int = 1

    comptime StorageType[
        mut: Bool,
        //,
        dtype: DType,                      # <-- here
        origin: Origin[mut=mut],
        address_space: AddressSpace,
    ]: TrivialRegisterPassable

    @staticmethod
    def unsafe_ptr[...](
        storage: Self.StorageType[dtype, origin, address_space],
    ) raises -> Pointer[Scalar[dtype], origin, address_space=address_space]
```

Every operation the trait defines -- load, store, offset, distance,
reinterpret -- is expressed over `Scalar[dtype]`, so a conforming engine
can change *where* and *how* elements are addressed but not *what an
element is*. `DefaultEngine`'s handle is
`Pointer[SIMD[dtype, element_width], ...]`, which is the other half of the
point: `element_width` is **vectorization**, not composition. A logical
element may be a SIMD vector of one dtype; it may not be a pair of
different fields.

So the request is not "conform a new engine". It is to lift `dtype: DType`
to an element type in the storage trait and in `TileTensor`, which is why
this is a design conversation rather than a patch.

### MAX already pays this cost internally, three times

This is not only a downstream complaint. Three places in MAX's own tree
hand-roll the encoding that an element type would give them -- and the
third one does it by defeating the type system outright.

**`ArgMax`, in the reduction library.** Its element is conceptually a
`(value, index)` pair. Because a SIMD lane cannot hold a pair, it carries
two parallel registers:

```mojo
struct ArgMax[dtype: DType, W: Int = simd_width_of[dtype]()](ReduceOp):
    var best: Scalar[Self.dtype]
    var best_idx: Int
    var acc_values: SIMD[Self.dtype, Self.W]
    var acc_indices: SIMD[.int64, Self.W]
```

That is structure-of-arrays inside a single accumulator, and the
duplication it forces between `best_idx` and `acc_indices[0]` is exactly
the trap filed as 3.6 above. One bug in this document is a direct
consequence of this limitation in MAX's own code.

**Rotary embeddings, in `nn`.** Complex multiplication is the smallest
possible composite element, and `DType` has no complex variant at all --
the enum runs `bool`, `int`, `uint`, the integer widths, the `float8`
family, `bfloat16`, `float16`, `float32`, `float64`, and stops. So
`nn/fused_qk_rope.mojo` deinterleaves by hand, and carries a helper to
reconcile two incompatible conventions for where the halves live:

```
Deinterleaves the input into real and imaginary parts, multiplies them as
...
# In GGUF, weights are organized as real, imag, real, imag, real, imag, ...,
# while in safetensors, the data is stored as real, ..., real, imag, ..., imag.
# This function return the indices for the real and imaginary part.
```

A production ML kernel doing manual index arithmetic to fake a
two-field element, and a format-compatibility shim because two upstreams
picked different manual encodings, is the same problem numax has -- and it
would be solved by the same feature.

**Quantization, which stores structs in tensors by lying about the
dtype.** This is the strongest form of the argument, because here MAX does
not work *around* composite elements -- it uses them, and hides them from
the type system to do it. `Q4sym` is a real composite element:

```mojo
struct Q4sym[group_size: Int, float_dtype: DType = .float32](Defaultable):
    var scale: StaticTuple[UInt8, 2]
    var bits: StaticTuple[UInt8, SIMDLength(Self.group_size) // 2]
```

`pixi run repro 3.10b-quantization-struct-precedent.mojo` confirms it at
the pin -- `Q4sym[32]` is 18 bytes. `block_Q4_K` beside it is larger and
has four fields, two of them `Float16`.

But a tensor of them cannot be declared, so `matmul_Q4_K` takes
`b_tt: TileTensor[mut=False, .uint8, ...]` and recovers the structure with
a pointer bitcast:

```mojo
var base_block_ptr = blob_output_ptr.bitcast[
    Q4sym[Self.group_size, Self.float_dtype]
]()
```

The element type is erased to bytes and the layout travels by convention.
Every stride calculation (`bytes_per_group_int4` and friends) is manual,
nothing checks that the producer and the consumer agree, and the same
idiom repeats across `fp8_quantization.mojo`,
`block_scaled_quantization.mojo` and the `mxfp4_*` kernels.

So the request is not for a capability MAX lacks internally. It is for the
one MAX already relies on, with the type system allowed to express it.

**Expected.** An element-generic path -- storage and `TileTensor`
parameterized over `T: TrivialRegisterPassable & DevicePassable` rather
than over `DType` -- for the operations where element layout is all that
matters: allocation, copy, elementwise application, gather and scatter.
The `DType`-keyed path would stay exactly as it is for everything that
depends on knowing the scalar: tensor cores, vendor BLAS, dtype-specific
intrinsics. Those are the majority of MAX's kernels and none of them need
to change. A narrower version that would still help a great deal: add
complex to `DType`, which covers the single most common composite and is
a fraction of the work.

**Impact.** This is the single structural reason numax has two array types
instead of one, which is the most common complaint it gets. A trait-generic
conformer (a dual number for derivatives, a double-double for extra
precision, an interval for rigorous bounds) is an ordinary Mojo struct of
two or four floats. It can live in a register, it can be a kernel
argument, and it cannot be a tensor element. So numax carries a
register-resident array type for conformers and a device tensor type for
machine scalars, with explicit conversions between them, and every user
has to learn which one a given algorithm wants.

The workaround it ships is worth describing, because it is also the shape
the feature would take: a dual-number tensor is lowered to *two* float
tensors, a value and a derivative, and a gradient tensor to a value plus
one tensor per partial. That is a structure-of-arrays encoding done by
hand, at the boundary, by the user -- the same encoding `ArgMax` and
`fused_qk_rope` do by hand above. Doing it once, keyed on the element
type's fields, is the whole ask.

It also costs numax its FFT API: with no complex element, a spectrum is
returned as two tensors that are never unpacked, so `scipy.fft`'s
single-array return has no equivalent.

**Not asked for.** Arbitrary element types. Anything with a pointer, an
allocation or a non-trivial destructor is correctly out of scope; the
useful case is small trivially-copyable aggregates, which is exactly what
`TrivialRegisterPassable` already identifies.

---

# What would help most

Ranked by what it unlocks for a downstream numerical library, not by
implementation cost. Language-side asks are ranked separately in
[`mojo-feedback.md`](mojo-feedback.md).

1. **Mask the GEMV tail store** (2.1). A crash, architecture-dependent,
   with three separate workarounds in one library. Every MAX caller doing
   a matrix-vector product is exposed and most do not know it.
2. **Fix the `DeviceContext` lifetime** (3.1). Also a crash, also silent,
   and it dictated the architecture of an entire subpackage.
3. **Symmetric and triangular BLAS-3 on `TileTensor`** (2.4) and
   **`transpose_a`** (2.3). Together these are most of the remaining
   distance between a MAX-based factorization and LAPACK, and
   `transpose_a` is already a declared parameter.
4. **A runtime device-dispatch path** (3.2). Would remove a compile-time
   parameter from 45 modules' public signatures.
5. **Element-generic device storage** (3.10), or as a much cheaper first
   step, **a complex `DType`**. The largest item here, and the one MAX
   itself already pays for three times -- `ArgMax`'s parallel accumulator
   fields, `fused_qk_rope`'s manual deinterleaving, and the quantization
   kernels bitcasting structs through `uint8` tensors. A design
   conversation rather than a fix, and worth starting.
6. **One sparse format with a matvec** (2.8), which is the foundation
   under a whole problem domain MAX currently has nothing in, and with
   3.1 fixed also unlocks iterative solvers (2.9). A CPU CSR matvec alone
   would give downstream libraries something to route to.
7. **A kernel index by operation name** (3.8), saying which root owns each
   operation and which targets it reaches. Pure documentation, and the
   `max.algorithm`-vs-`algorithm` confusion cost this project a
   hand-written reduction engine that MAX already shipped. While you are
   there: `cublasSspmv` and `rocblas_sspmv` are dense packed-symmetric
   matvecs, and their names cost a reader real time when they are looking
   for sparse support (2.8).

## Filing notes

Two entries need hardware this project does not have, and say so in their
own environment lines rather than claiming a local reproduction: **2.1**
needs an x86-64 runner, because Apple builds route `matmul` to Accelerate
and never reach MAX's own GEMV kernel, and **3.7** needs Apple hardware.
Everything else reproduces on any machine with the pinned toolchain.

One API-churn note for whoever reads the 3.10 entry: the storage seam is
`Storage: TensorStorage = PointerStorage` at the 26.5 pin and
`Engine: TensorEngine = DefaultEngine[element_width=1]` on `main`. The
rename is not the subject of a complaint, but it does mean downstream code
naming `Storage=` will break on upgrade.
