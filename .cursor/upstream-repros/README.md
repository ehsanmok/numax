# Upstream feedback for Modular

Defects, gaps and API problems in **Mojo** and **MAX**, found while
building [numax](https://github.com/ehsanmok/numax) -- a numerical
computing library for Mojo built on MAX: special functions, dense linear
algebra, quadrature, ODE solvers, FFTs, distributions, statistics,
interpolation, signal processing, and a NumPy-named array surface. It is
89 test suites and 26 example programs over ten subpackages, it delegates
to MAX wherever MAX ships a kernel, and it is the kind of downstream
consumer that finds the edges of both the language and the kernel
library.

Two documents, split by who owns the fix:

| | |
|---|---|
| [`mojo-feedback.md`](mojo-feedback.md) | the language and standard library -- 8 entries |
| [`max-feedback.md`](max-feedback.md) | the kernel library and its APIs -- 19 entries |

`mojo-feedback.md` also carries a section, **"What these gaps prevent"**,
for two capabilities that are inexpressible rather than merely unwritten:
decompositions over run-time shapes, and reverse-mode autodiff. Sparse
matrices and Krylov solvers are the same kind of item on the kernel side.

Every entry is self-contained: a repro that does not import numax, the
diagnostic as it was actually printed, what the behavior should be, what
the gap cost, and the workaround shipped instead. The `.mojo` files in
this directory are that evidence, and each entry carries the command that
runs it.

## Running a repro

Everything goes through pixi, which pins the toolchain the diagnostics
came from. From the repository root:

```bash
pixi run repro 1.1b-conformance-where-rejected.mojo
```

**Most of these are expected to fail.** The diagnostic *is* the result,
so a nonzero exit usually means the repro worked. One exception:
`1.8-extension-self-recursion-hang.mojo` never terminates, which is its
finding -- interrupt it.

No repro imports numax -- a report that needed a third-party library to
demonstrate would not be worth filing -- so none of them need the `-I .`
that the rest of this project's tasks spell out.

## Environment

Unless an entry says otherwise, everything was reproduced on:

- **Mojo 1.0.0 (ed45d567)**, **MAX 26.5**, installed through pixi from
  the Modular conda channel
- **Apple M3 Pro, arm64, macOS 15**

Two provenance notes, so no claim is broader than its evidence. **ROCm is
entirely unmeasured** -- numax names no architecture, so its AMD coverage
is inherited from `linalg.matmul`'s own dispatch, which is a statement
about what runs and not about what it costs. And the two entries that
need hardware this project does not have
(`max-feedback.md` 2.1, x86-64; 3.7, a Metal device) say so in their own
environment lines instead of claiming a local reproduction.

## How to read an entry

Each carries **severity**, a **repro**, **observed**, **expected**,
**impact** and **workaround**. Severity means:

| | |
|---|---|
| **crash** | the compiler or program dies, or memory outside the object is written |
| **wrong** | a plausible answer that is silently incorrect |
| **blocked** | the thing cannot be expressed at all |
| **friction** | expressible, but the shape forces a worse program |

## Index

Mojo, in [`mojo-feedback.md`](mojo-feedback.md):

| Entry | Severity | Repro |
|---|---|---|
| 1.1 Extensions cannot carry a `where` clause | blocked | `1.1a`-`1.1f` (six files) |
| 1.2 The manual denies a shipped feature | friction | none (documentation) |
| 1.3 `enqueue_memset` on `DType.bool` | crash | `1.3-bool-memset-pass-manager.mojo` |
| 1.4 The `where` prover cannot fold `%` | blocked | `1.4-where-prover-modulo.mojo` |
| 1.5 Move-only tuples do not destructure | blocked | `1.5a`, `1.5b` |
| 1.6 `a > b` on a SIMD vector | friction | `1.6-simd-strict-inequality.mojo` |
| 1.7 float64 `exp`/`log`/`erf` accuracy | wrong | `1.7-stdmath-float64-accuracy.mojo` |
| 1.8 Extension method self-recursion hangs | crash | `1.8-extension-self-recursion-hang.mojo` |

MAX, in [`max-feedback.md`](max-feedback.md):

| Entry | Severity | Repro |
|---|---|---|
| 2.1 GEMV writes past its destination | crash | `2.1-gemv-destination-overwrite.mojo` (x86-64) |
| 2.2 The only factorization takes `LayoutTensor` | blocked | `2.2-qr-layouttensor-only.mojo` |
| 2.3 `transpose_a` declared, unimplemented | friction | `2.3-matmul-transpose-a.mojo` |
| 2.4 No symmetric or triangular BLAS-3 | blocked | none (absence) |
| 2.5 No forward FFT | blocked | none (absence) |
| 2.6 `nn.cumsum` has no device path | blocked | `2.6-cumsum-no-target.mojo` |
| 2.7 Other absent kernels | blocked | none (absence) |
| 2.8 No sparse linear algebra | blocked | none (absence) |
| 2.9 No iterative solvers | blocked | none (absence) |
| 3.1 `DeviceContext` lifetime | crash | none (needs a reassignment sequence) |
| 3.2 `target` cannot be chosen at runtime | friction | none (by inspection) |
| 3.3 `DeviceContext` cannot cross `enqueue_function` | blocked | none (by inspection) |
| 3.4 `enqueue_function` rejects layout-`where` kernels | friction | none (by inspection) |
| 3.5 `rand_uniform` rejects capturing output | friction | none (by inspection) |
| 3.6 `ArgMax` answer is not where it is named | wrong | none (by inspection) |
| 3.7 `imm` capture writes zeros | wrong | none (needs Metal) |
| 3.8 Two `algorithm` roots | friction | none (documentation) |
| 3.9 `docs_check_imports` false negatives | friction | none (tooling) |
| 3.10 Storage is keyed on `DType` | blocked | `3.10-struct-element-tensor.mojo`, `3.10b-quantization-struct-precedent.mojo` |

Four files in this directory are **not** failures, and are here because
the reports rest on them:

| File | Why |
|---|---|
| `1.1a-extension-works.mojo` | retroactive conformance *working* at the pin, which the manual says is impossible |
| `1.1f-constant-body-works.mojo` | an extension method body conforming, to show 1.1 is about evidence and not about bodies |
| `3.10b-quantization-struct-precedent.mojo` | MAX storing an 18-byte composite element in a `uint8` tensor via pointer bitcast -- the capability 3.10 asks for, already in use with the type erased |
| `disproved-simd-ne-nan.mojo` | a long-standing claim that **no longer reproduces**; `SIMD.ne` was believed ordered, but `nan != nan` returns `[True, True]`, matching NumPy. Withdrawn rather than filed, and recorded so nobody re-files it |

## Before citing any of this

Re-run it. Every diagnostic here is from one toolchain version, and the
withdrawn claim above had already gone stale by the time it was checked.
