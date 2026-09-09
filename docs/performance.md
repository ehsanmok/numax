# Performance

> Companion to [bench/README.md](../bench/README.md), which has the
> cross-language methodology, the per-bench descriptions, and the full
> results table. This page is the architectural framing of those
> numbers: what the kernel is doing, what the ceiling is, and what each
> measurement does and doesn't tell you. For the design intent behind
> the roofline work and the two benchmark bugs that faked a 40%
> codegen deficit, see [`bench/README.md`](../bench/README.md), which
> records both retractions.

## The kernel

The headline benchmark is one line:

```mojo
def gaussian[T: FloatLike](x: T) -> T:
    return (-(x * x)).exp()
```

Run across an n-element `TileTensor` of `float32` via
`numax.core.tensor.map[step=gaussian_step]`. It moves 8 bytes per element
(float32 read + float32 write) however it's written, so on a 150 GB/s
machine it's capped at ~18,750 M elem/s. **Most of the surface below is
competing for that bandwidth, not for arithmetic.** That's measured,
not assumed: an identity copy with no arithmetic runs at the same speed
as `gaussian` through the same walk, so the `exp` is free at every size
tested (see `bench/bench_gpu_roofline.mojo`).

## CPU paths

`numax` has two CPU walks:

- **`map`** -- single-thread, native SIMD width via
  `TileTensor.vectorize()`, plus a scalar tail for whatever doesn't
  divide evenly.
- **`map_threaded`** -- the same `step` handed to
  `max.algorithm.elementwise[target="cpu"]`, multi-threaded across CPU
  cores.

The serial path is what every example uses. The threaded path loses
below ~250K elements (dispatch costs more than the work) and wins 3-5x
above ~1M; on an Apple M3 Pro it also beats that machine's own GPU path
until about 16M elements (a crossover that depends on the GPU -- a
discrete card with far more bandwidth moves it down). One behavioral difference to know: the
threaded path flushes denormals to zero (MAX's worker-thread FP
environment, not a `numax` choice). See `bench/bench_elementwise.mojo`.

## GPU path

`map[gpu=True]` is the body of one GPU thread, launched via
`DeviceContext.enqueue_function` with one element per thread (the
default `width=1`; thread coarsening with `width>1` is supported but
buys nothing). That tuning has now been measured on
both backends and agrees: sweeping `width` in `{1,2,4,8}` against
`block_dim` in `{128,256,512,1024}` at 67M elements, `width=1` is
fastest or tied on Metal *and* on CUDA, with `width=8` a few percent
behind on both. On an Apple M3 Pro at ~150 GB/s peak bandwidth, the GPU
path reaches **84% of peak** (126 GB/s) at 67M elements; on an NVIDIA
A10G at ~600 GB/s it reaches **~83% of peak** (498 GB/s) at the same
size.

Two measurement shapes matter:

- **Per-call** -- synchronize inside every iteration (the latency of
  one launch-through-completion). Use this when you care about the
  cost of one dispatch, not steady-state throughput.
- **Amortized** -- enqueue all of them and synchronize once (steady-
  state throughput, dispatches pipelined). Use this for measuring the
  kernel itself, not the launch latency.

An earlier results table mixed these across libraries and flattered
PyTorch by ~1.6x; both columns are reported in `bench/README.md` now.

## The "no fusion pass" claim

A `FloatLike` kernel *is* a fused kernel, because composition happens
inside `step` before any tensor walk occurs. `numax` needs no fusion
pass to get the effect -- the user composes inside `step`, and the walk
is one pass over memory. That only holds if callers compose inside
`step` rather than chaining `map` calls, which is a documentation and
API-shape problem more than a codegen one. `bench/bench_fusion.mojo`
measures the cost of getting it wrong at 1.4-3x.

## Cross-language baseline

`bench/{numpy,mlx,torch,cupy,thermite}/` run the identical kernel, sizes,
input values, and warmup/timed-iteration counts as standalone
scripts, on the same Apple M3 Pro. M elem/s, higher is better. Every
one of these also runs on Linux/CUDA (`thermite` on AVX2 rather than
NEON, torch on CUDA rather than Metal; MLX is macOS-only); the tables
below are one machine's numbers, not a statement about which platforms
work.

**CPU-only:**

| n | numax `map` | numax `map_threaded` | thermite (NEON) | NumPy | MLX | torch eager | torch compile |
|---|---|---|---|---|---|---|---|
| 65,536 | 2,348 | 3,744 | 1,706 | 680 | 249 | 1,194 | 576 |
| 262,144 | 2,340 | 2,776 | 1,298 | 660 | 811 | 1,737 | 2,214 |
| 1,048,576 | 2,355 | 11,560 | 1,311 | 511 | 1,586 | 2,165 | 3,544 |
| 4,194,304 | 2,349 | 10,099 | 1,508 | 626 | 1,647 | 2,221 | 4,136 |
| 16,777,216 | 2,343 | 8,736 | 1,627 | 508 | 1,893 | 1,989 | 4,015 |
| 67,108,864 | 2,347 | 9,026 | 1,632 | 493 | 1,954 | 1,935 | 4,046 |

**GPU (same Metal device), per-call sync:**

| n | numax | MLX | torch eager | torch compile |
|---|---|---|---|---|
| 65,536 | 301 | 328 | 200 | 231 |
| 262,144 | 639 | 864 | 715 | 1,007 |
| 1,048,576 | 3,230 | 3,242 | 2,810 | 3,320 |
| 4,194,304 | 7,897 | 3,684 | 3,701 | 9,615 |
| 16,777,216 | 11,722 | 4,514 | 4,792 | 12,332 |
| 67,108,864 | 14,465 | 4,866 | 4,807 | 13,831 |

**GPU (same Metal device), amortized sync:**

| n | numax | MLX* | torch eager | torch compile |
|---|---|---|---|---|
| 65,536 | 1,280 | 899 | 888 | 799 |
| 262,144 | 4,064 | 2,772 | 2,580 | 4,800 |
| 1,048,576 | 13,206 | 2,458 | 4,612 | 9,119 |
| 4,194,304 | 13,902 | 3,776 | 4,423 | 13,080 |
| 16,777,216 | 15,086 | 3,896 | 5,082 | 15,695 |
| 67,108,864 | 15,755 | 3,860 | 4,809 | 14,380 |

\* MLX is lazy, so forcing only the last of ten enqueued ops would time
one kernel instead of ten -- the benchmark has to hold all ten outputs
live instead (ten 268MB buffers at 67M elements), and that memory
pressure makes MLX's amortized number *worse* than its own per-call
one. Read MLX's per-call column; the amortized one is a benchmark
artifact of MLX's execution model, not a speed measurement.

Reading this without treating it as a scoreboard:

- **`numax` and `torch.compile` on MPS are at parity on the GPU**, both
  pressed against the same bandwidth roofline: 15,755 vs. 14,380 M
  elem/s amortized at 67M (84% vs. 77% of peak), 14,465 vs. 13,831
  per-call. They trade wins by a few percent in both directions across
  the sweep, which given run-to-run spread reads as a tie.
- **`numax`'s threaded CPU walk is the fastest CPU number above ~1M**
  (8,700-11,600 vs. `torch.compile`'s 3,500-4,100), and it beats this
  machine's own GPU path until about 16M elements.
- **MLX's GPU path leads at every size up to 16M**, and `numax`
  crosses over only at 67M -- not a claim that `numax`'s GPU codegen is
  categorically ahead of MLX's.
- **NumPy is the slowest CPU baseline at every size**, consistent with
  it not doing the same kind of native SIMD-width dispatch the other
  three do.
- **thermite (Rust NEON)**, the crate this project's pattern was
  originally ported from, is behind `numax`'s CPU path at every size,
  but the two aren't running identical code (different `exp`
  approximations, a runtime ISA dispatch check on every call,
  in-place vs. separate-output-buffer) -- so the gap isn't attributed
  to codegen.

See `bench/README.md` for the two measurement bugs that inflated an
earlier version of this comparison (a benchmark artifact, not a
`numax` change) before drawing further conclusions from these numbers.

## Cross-language baseline, NVIDIA A10G

The same benchmark scripts, same kernel, same sizes, run on an NVIDIA
A10G (23 GB, 300 W, 600 GB/s spec bandwidth) with an AMD EPYC host. CuPy
14.2.0, NumPy 2.5.2, PyTorch 2.13.0+cu130, `thermite` 0.2 on its AVX2
backend. CuPy is CUDA-only, so this is the only table it appears in, the
mirror of MLX being macOS-only and absent here. M elem/s, higher is better;
every column below is from one session so the rows are comparable.

CuPy carries three columns: **eager** (`cupy.exp(-(x * x))`, three kernels),
**fused** (the same under `cupy.fuse()`, one kernel), and **kernel** (a
hand-written `cupy.ElementwiseKernel`). The last is the like-for-like
comparison against a `numax` `step`, which is one fused kernel already.

**GPU, amortized sync:**

| n | numax | CuPy kernel | CuPy fused | CuPy eager | torch compile | torch eager |
|---|---|---|---|---|---|---|
| 65,536 | 17,852 | 6,731 | 6,020 | 1,849 | 1,224 | 2,277 |
| 262,144 | 70,374 | 27,579 | 24,635 | 8,421 | 3,963 | 9,203 |
| 1,048,576 | 68,789 | 52,428 | 52,776 | 18,207 | 18,213 | 25,360 |
| 4,194,304 | 58,753 | 57,718 | 57,403 | 19,476 | 51,489 | 19,301 |
| 16,777,216 | 60,913 | 60,127 | 60,168 | 20,155 | 55,153 | 19,757 |
| 67,108,864 | **61,653** | 61,051 | 61,166 | 20,375 | 56,245 | 19,905 |

**GPU, per-call sync** (67M): numax 61,013, CuPy kernel 60,352, CuPy fused
60,213, CuPy eager 20,239, torch compile 53,670, torch eager 19,803.

**CPU:**

| n | numax `map_threaded` | numax `map` | torch compile | thermite (AVX2) | torch eager | NumPy |
|---|---|---|---|---|---|---|
| 65,536 | 2,228 | 1,732 | 770 | 1,349 | 630 | 283 |
| 1,048,576 | 10,797 | 1,812 | 8,059 | 1,366 | 953 | 401 |
| 16,777,216 | 13,167 | 1,720 | 1,619 | 1,332 | 380 | 307 |
| 67,108,864 | 8,733 | 1,367 | 1,880 | 1,329 | 468 | 316 |

What this machine says, and where it differs from the M3 Pro above:

- **numax is at parity with a hand-written CUDA kernel, and leads
  `torch.compile` by ~10%.** At 67M amortized: 61,653 against CuPy's
  `ElementwiseKernel` at 61,051 and `cupy.fuse()` at 61,166, with
  `torch.compile` at 56,245. The first three sit within 1% of each other at
  82% of the card's peak, which is the roofline — parity, not a win, and
  the point is that a generic `FloatLike` kernel gives up nothing to CUDA C
  written by hand for this expression. On Metal numax and `torch.compile`
  were a tie, so the 10% lead is this device's, not a general claim.
- **CuPy measures the fusion argument directly.** Its eager path is 20,375
  and its fused path 61,166, a ratio of 3.002 against the three passes the
  unfused expression makes. A numax kernel has no eager column to lose,
  because composition happens inside `step` before any tensor walk.
- **It is bandwidth-bound, not compute-bound.** `bench-roofline` puts the
  best configuration at **500.9 GB/s** — ~83% of the card's 600 GB/s spec —
  and an identity copy at 489.10 GB/s against the Gaussian's 489.16 GB/s at
  67M. The `exp` is free; what is left to win is the last ~17% of the memory
  path.
- **Kernel fusion is worth 1.99x on the GPU** at every size tested (1.991 /
  1.981 / 2.000 / 1.997 from 1M to 67M) — two chained `map` calls against one
  `map` with a composed `step`. On the CPU the same change is worth 1.26-1.58x.
- **numax's serial CPU walk matches hand-written Rust SIMD** (1,367 vs
  `thermite`'s 1,329 at 67M) and is 4.3x NumPy. Threaded, it is the fastest
  CPU number at 16M and 67M.
- **The threaded CPU path is noisy on this host** — repeat runs move by a
  factor of two at the same size, and 1M has read anywhere from 3,325 to
  10,797 M elem/s across runs. Read it as a range, not a point.
- **Launch overhead dominates below ~1M.** At 64K the same kernel measures
  61.1 GB/s with a sync inside every iteration against 162.8 GB/s amortized;
  by 67M the two agree (489.6 vs 492.2).


## Accuracy

`bench/accuracy/` (`pixi run accuracy`) measures max absolute, relative,
and ULP error per function per domain against mpmath at 50 decimal
digits. The references are checked in, so the harness runs in the
default Mojo environment with no Python.

The bounds hold. A&S 7.1.26's ~1.5e-7 for `default_erf_approx` measures
1.38e-07; A&S 17.3.36's ~2e-8 for `elliptic_e` measures 1.57e-08 (which
also confirms the `b4` coefficient recovered by fitting, after every
OCR'd copy of the source table turned out to be misdigitized); the Bessel
family lands at 1.6e-08 to 3.9e-08 absolute across both branches and
their blend; the orthogonal polynomial recurrences are exact to within
rounding, at 3 to 253 ULP.

One caveat governs the whole table, and the harness found it: Mojo's
`std.math` `exp`, `log`, and `erf` are not correctly rounded at float64
(`exp(1.0)` is wrong from the 13th significant digit, while `sin` and
`sqrt` are exact). Every `numax` function built on `exp`/`ln` inherits
that floor, so their measured float64 errors are an upper bound on their
own error rather than a measurement of it. At float32 -- the `dtype`
this library is normally used at -- the floor sits two orders of
magnitude below the representable resolution and cannot be observed.
See `bench/accuracy/README.md` for the full table and the defect
writeup.

## Cross the tiers past N (dense linalg)

`numax.linalg.array`'s `Array[T, n*n]` tier keeps its matrices in
registers -- a compile-time size that keeps them GPU-launchable (one matrix
per SIMD lane, callable from inside `map[gpu=True]`) and lets `T` be `Dual`
or `Compensated`, at the cost of both compile time and register pressure
growing with `n`. Past a certain `n`, the `Tensor` tier -- `numax.linalg`
itself -- wins on raw speed, because its cubic term is MAX's kernel.
`bench/bench_matmul.mojo` measures exactly where, for `matmul` (nanoseconds
per `n x n` product, lower is better; the batched column runs 4 independent
products per call, one per SIMD lane, divided by 4 to stay comparable):

| n | MAX | numax scalar | numax batched (per matrix) |
|---|---|---|---|
| 4 | 102 | 57 | 15 |
| 8 | 143 | 323 | 74 |
| 16 | 366 | 1,870 | 492 |
| 32 | 380 | 14,335 | 3,690 |
| 64 | 900 | 117,440 | 30,170 |

MAX wins from `n = 8` on for a single matrix, from `n = 16` on even
against the 4-wide batched form, and is ~130x ahead by `n = 64`. See
`bench/README.md`'s "Matmul: where MAX overtakes the generic loop" for
the full writeup.

Crossing is `to_tensor` and then the same function name from
`numax.linalg` rather than `numax.linalg.array`. What that gets differs by
operation:

- `matmul`, `matvec`, `batched_matmul` are MAX kernels outright, so the
  table above *is* their number.
- `dot`, `nrm2`, `asum`, `axpy` and `outer` are compositions of MAX
  primitives: the reductions drive `ReduceSum` under the `rowwise`
  scaffolder, the maps go through `elementwise`.
- `cholesky`, `lu_factor`, `qr_factor`, `solve`, `solve_triangular`,
  `cholesky_solve`, `inverse`, `det`, `norm` and `trace` are numax's,
  because MAX ships no factorization on `TileTensor` at any size. They are
  blocked and device-resident: the `O(n^3)` trailing update is a matrix
  product and lands in `linalg.matmul`, while the panel stays
  `O(n * block^2)`.
- `svd`, `eigh`, `eigvals`, `cond`, `pinv` and `tridiagonal_solve` are
  `Array`-only, deliberately -- `numax/linalg/__init__.mojo` has the
  reasoning. Past the crossover they are genuinely missing rather than one
  import away.

### The blocked factorizations, measured

`bench/bench_linalg.mojo` and `bench/bench_linalg_gpu.mojo`, against
`bench/scipy/linalg.py` (LAPACK), `bench/torch/linalg.py` and
`bench/cupy/linalg.py` (cuSOLVER). `float32` throughout, on both
processors, because `linalg.matmul` does not compile for GPU at `float64`
at all and a `float64` CPU table beside a `float32` GPU one would compare
two different computations.

The first row of each table is not a competitor, it is the **ceiling**:
`linalg.matmul` on the same size, which is what a blocked factorization's
cubic term actually calls. The fraction of that row a factorization
reaches is how much of its work went to MAX.

**CPU -- AMD EPYC 7R32, 16 cores / 32 threads, 64 MiB L3.** GFLOP/s at
`n = 1024`, from the standard LAPACK flop counts:

| op | numax | SciPy (LAPACK + OpenBLAS) | numax / SciPy |
|---|---|---|---|
| `matmul` (ceiling) | 656 | 833 | 0.79 |
| `cholesky` | 15.4 | 99.5 | 0.15 |
| `lu_factor` | 17.2 | 43.9 | 0.39 |
| `solve` | 17.0 | 52.6 | 0.32 |
| `qr_factor` | 3.9 at the default block, 8.3 at `block=4` | 12.4 | 0.31-0.67 |

**GPU -- NVIDIA A10G (Ampere).** Same counts, `n = 1024` except `qr`,
which is `n = 512`:

| op | numax | PyTorch (cuSOLVER) | CuPy (cuSOLVER) |
|---|---|---|---|
| `matmul` (ceiling) | 20,459 | 15,342 | 14,995 |
| `cholesky` | 51.0 | 595.0 | 349.9 |
| `lu_factor` | 30.6 | 282.9 | 266.8 |
| `solve` | 28.1 | 257.6 | 257.9 |
| `qr` (n=512) | 8.4 | 80.0 | 79.1 |

Three things to read out of those, in order of how much they matter:

1. **The GEMM is not the problem.** MAX's `matmul` reaches 79% of
   OpenBLAS on CPU and *beats* cuBLAS's FP32 path on the A10G -- 20.5
   against 15.3 TFLOP/s. It is also less accurate on the same product
   (max residual 8.4 against cuBLAS FP32's 3.4, where cuBLAS with TF32
   enabled gives 18.2 at 25.2 TFLOP/s), so MAX's `float32` GEMM sits
   between the two vendor paths on both axes. Either way, a factorization
   at 51 GFLOP/s on a device whose GEMM does 20,000 is not being held back
   by the multiply.
2. **The gap is the panel and the launch count.** Every block step is a
   single-block panel kernel -- one SM on the GPU, effectively serial on
   CPU -- followed by a host-side launch of the trailing GEMM. At
   `block = 32` and `n = 1024` that is 32 dependent launches around 32
   panels, and the panels are `O(n * block^2)` of work that no GEMM
   touches. cuSOLVER's advantage is a parallel panel, not a better GEMM.
   That is the next optimization, and it is a real one: 6-12x on the GPU.
3. **The best block size is small, and for QR it shrinks with `n`.**
   Measured at `n = 1024`: `cholesky` 22.99 ms at `block=32` against 29.66
   at 16 and 30.11 at 64; `lu_factor` 41.43 ms at 16 against 43.87 at 8 and
   58.31 at 32; `qr_factor` 168 ms at `block=4` against 203 at 8, 359 at 16
   and 1502 at 128. Bigger blocks make the panel quadratically more
   expensive faster than they make the GEMM more efficient, which is the
   signature of point 2. This sweep predates the panel work, and the
   defaults have moved since: `cholesky` and `lu_factor` are now 64 and 32
   on the host, 32 and 16 on a device. `qr_factor` stays at 16, which the
   EPYC sweep found right at `n <= 512` and about 2x off at `n = 1024` --
   an EPYC statement, since the M3 Pro sweep has since inverted it.

### BLAS-1 on `Tensor`

Bandwidth, not arithmetic, so GB/s over the traffic each operation must
move. At `n = 67M` (256 MiB, past this box's L3 either way):

| op | numax CPU | SciPy CPU | numax A10G | PyTorch A10G | CuPy A10G |
|---|---|---|---|---|---|
| `dot` | 70.6 | 27.9 | 298.9 | 503.9 | 244.9 |
| `nrm2` | 71.1 | 13.9 | 169.2 | 495.9 | 163.5 |
| `asum` | 67.4 | 48.5 | 169.1 | 161.2 | 163.5 |
| `axpy` | 25.3 | 78.3 | 342.4 | 475.8 | 292.8 |

The reductions on CPU beat OpenBLAS's own level-1 routines by 1.4-5x,
which is the clearest single payoff of routing them through MAX's
`rowwise` scaffolder: `sdot` and `snrm2` are one thread, `ReduceSum` under
`rowwise` is all 16 cores. On the GPU they land between CuPy and PyTorch.

`axpy` is the exception and the reason is its signature, not its kernel:
it returns a new vector rather than updating `y` in place, so every call
allocates and first-touches 256 MiB. Measured directly at `n = 16M`: the
whole call is 8.5 ms, the `elementwise` pass alone is 4.3 ms and the
zeroed allocation alone is 4.7 ms. Half the time is the allocation, which
is why OpenBLAS's in-place `saxpy` is ahead on CPU. Reuse the result
tensor if this is on a hot path.

Note also the `n = 16M` row of the raw tables: 16M `float32` is exactly
64 MiB, this box's L3, so the CPU reductions there read out of cache and
report 264-362 GB/s. That is a cache number and it is in the harness on
purpose, sitting next to a memory-resident one.

### The same sweep on an Apple M3 Pro

The linalg tables above are an EPYC host and an A10G. This one is the
**Apple M3 Pro** this library is developed on -- 12 cores (6P + 6E), an
18-core GPU, 36 GB unified, ~150 GB/s -- and it is a different machine, so
nothing here may be read across into the tables above. `float32`,
`n = 1024`, GFLOP/s from the same LAPACK counts.

Two things about the baselines matter before the numbers. SciPy here links
**Apple Accelerate**, not OpenBLAS, so LAPACK runs on the AMX coprocessor
and every CPU row is a harder target than the EPYC table's: Accelerate's
`cholesky` is 248 GFLOP/s where OpenBLAS's was 99.5. And on the GPU side
**MLX has no linalg on the device at all** -- `cholesky`, `lu_factor`,
`qr` and `solve` each refuse a GPU stream with "not yet supported on the
GPU" -- so PyTorch's MPS backend is the only Metal baseline there is, and
numax's factorizations being device-resident on Metal has no MLX
counterpart to be compared against.

**CPU -- Apple M3 Pro, SciPy 1.18.1 on Accelerate:**

| op | numax | SciPy (LAPACK + Accelerate) | numax / SciPy |
|---|---|---|---|
| `matmul` (ceiling)* | 1,475 | 1,393 | 1.06 |
| `cholesky` | 83.4 | 280.1 | 0.30 |
| `lu_factor` | 86.9 | 231.9 | 0.37 |
| `solve` | 80.3 | 162.8 | 0.49 |
| `qr_factor` | 37.6 | 52.0 | 0.72 |

\* Not a kernel comparison: both are Apple Accelerate's `cblas_sgemm`. See
the first bullet below.

**Metal -- the same machine's 18-core GPU, PyTorch 2.13.0 on MPS:**

| op | numax | PyTorch (MPS) | numax / PyTorch |
|---|---|---|---|
| `matmul` (ceiling) | 1,800 | 1,143 | **1.57** |
| `cholesky` | 65.2 | 125.0 | 0.52 |
| `lu_factor` | 18.8 | 51.5 | 0.36 |
| `solve` | 16.8 | 20.9 | 0.80 |

**BLAS-1, GB/s at `n = 67M`** (256 MiB, past any cache on this box):

| op | numax CPU | SciPy CPU | numax Metal | PyTorch MPS |
|---|---|---|---|---|
| `dot` | 113.8 | 102.3 | 115.3 | 121.8 |
| `nrm2` | 89.2 | 29.8 | 106.2 | 4.3 |
| `asum` | 116.0 | 72.1 | 106.2 | 39.0 |
| `axpy` | 91.5 | 109.6 | 58.0 | 116.8 |

What this machine says:

- **The CPU ceiling row is Accelerate, reached through MAX.** MAX's CPU
  `matmul` dispatches to Apple's `cblas_sgemm` whenever the target is macOS
  and every operand is `float32`
  (`linalg/matmul/cpu/apple_accelerate.mojo`), which is exactly this table.
  So numax and SciPy are calling the *same* GEMM here and the 1,475 against
  1,393 is call overhead, not a better multiply. The dtype sweep shows it
  plainly: at `float64`, where the gate does not fire and MAX uses its own
  kernel, MAX is 264 GFLOP/s against Accelerate's `dgemm` at 365 -- 0.72x,
  in line with the 0.79x of OpenBLAS on the EPYC. A 5.6x `float32`/`float64`
  ratio where the lane count alone predicts 2x is the signature.

  That is a better statement of the problem than a win would be: on this
  machine numax's factorizations and LAPACK's are built on the identical
  GEMM, so the whole `cholesky` gap of 46 against 248 belongs to the blocked
  algorithm around it and none of it to the multiply.
- **MAX's Metal GEMM does beat PyTorch's**, 1,800 against 1,143. The
  Accelerate gate is under `matmul/cpu/`, so the device path is MAX's own
  kernel and this one is a like-for-like comparison.
- **The BLAS-1 reductions beat Accelerate by 1.7-3.2x**, the same result
  the EPYC table reports against OpenBLAS and for the same reason:
  `ReduceSum` under MAX's `rowwise` scaffolder threads and `sdot`/`snrm2`
  do not. On Metal `nrm2` is 25x PyTorch's, which is a statement about
  PyTorch's MPS reduction rather than about numax.
- **`solve` on Metal is at 0.81 of PyTorch**, the closest any factorization
  comes to parity on either processor.
- **The factorizations otherwise trail, and the ceiling row is why they
  cannot be read as a fraction of it.** See the next section: a blocked
  factorization's trailing update is a rank-`block` GEMM, not a square one,
  and the two run at very different speeds.
- `cholesky` and `lu_factor` move by 5-10% between runs at `n = 1024` on
  this box. Read them as ranges, not points.
- **The host factorizations are 1.2-1.8x faster than when this page first
  carried them**, and the rows above are the current numbers. `cholesky`
  went 46.0 -> 83.4, `lu_factor` 67.1 -> 86.9, `solve` 65.9 -> 80.3, over
  six changes measured one at a time: `pack_block` copying at the SIMD
  width, `trsm_right_lower_t`'s inner dot vectorized, the trailing update
  restricted to the lower triangle, the panel solves handed to
  `parallelize` because `elementwise` was running them on one core, LU's
  panel made recursive, and the block defaults retuned after each. The
  `parallelize` one was worth more than the other five together. A seventh
  change then moved `qr_factor` 34.0 -> 37.6 on its own, by staging `C`
  through `pack_block` instead of a scalar `elementwise` walk. `axpy`
  moved separately, 70.2 -> 91.5 GB/s on the host and 23.9 -> 58.0 on
  Metal, by not zeroing a buffer it overwrites.
- Every block default was measured after the change that moved it, and
  each change moved it again -- `cholesky`'s host block went 32 -> 48 -> 64
  over three commits as successive terms got cheaper. The
  `bench_linalg.mojo` sweep is what found each, and it still runs, so the
  next change to a panel or a solve should re-run it rather than assume
  these survive.
- **QR's block sweep has inverted on this machine.** `block = 16` now
  beats `block = 4` at `n = 1024` -- 37.6 GFLOP/s against 26.8, where the
  EPYC sweep below had the small block ahead. The panel work is what moved
  it, so the "pass `block` explicitly for a large QR" advice in the EPYC
  section is an EPYC statement and should not be read across.
- **What is left, in the order the profile puts it.** Cholesky is bounded
  by its trailing GEMM at ~68% of a much shorter run, and with the flops
  already halved the rest of that distance is GEMM shape rather than waste.
  LU is still bounded by its panel even after the recursion. QR has had its
  staging copy widened and carries two remaining items, neither of them the
  cleanup they were filed as: `T` rebuilt on every `q()` and `solve()` is
  argued against by `numax/linalg/qr.mojo`'s own docstring (`nb^3 / 3`
  against the products it enables) and needs measuring before it is
  changed, and `V` packed twice per panel is structural -- `v` feeds
  `V Y` and `v_t` feeds `V^T C`, and `matmul` transposes `b` and never
  `a`, so one materialized orientation is unavoidable. The nine launches
  per block step are still nine.

### The ceiling row is not the ceiling a blocked factorization can reach

Every table above opens with `linalg.matmul` at `n x n x n` and invites
reading the gap to it as the panel's cost. That overstates the panel,
because a blocked factorization never issues a square GEMM: its trailing
update is rank-`block`. Measured through Accelerate on the M3 Pro, at
`n = 1024`:

| GEMM shape | GFLOP/s |
|---|---|
| `1024 x 1024 x 1024` -- the ceiling row as written | 1,355 |
| `1024 x 128 x 1024` | 668 |
| `1024 x 64 x 1024` | 470 |
| `1024 x 32 x 1024` -- `cholesky`'s default block | 248 |
| `1024 x 16 x 1024` -- `lu_factor`'s and `qr_factor`'s | 149 |

So a `block = 32` factorization gives up 5.4x against the advertised
ceiling before its panel costs anything, and a `block = 16` one 9x. The
same effect is worse on Metal, where rank-16 measures 168 GFLOP/s against
the square 2,394.

This sharpens the panel diagnosis rather than replacing it. The block
sweep finds its optimum at a *small* block even though each doubling of
`block` buys roughly 1.7x on the GEMM, which means the panel term has to
be steeper than a reading of the square ceiling alone would suggest. But
"reaches 12% of the ceiling" is not a claim the square row can support,
and the honest denominator for `cholesky` at `block = 32` is 248, not
1,355.

### What is not measured

**ROCm is not measured anywhere on this page.** It is reached the same way
Metal and CUDA are -- numax passes `target="gpu"` and `linalg.matmul`
dispatches Apple simdgroup, CDNA or RDNA underneath, with no
per-architecture code anywhere in numax -- so the *coverage* is inherited
from MAX's dispatch. That is a statement about what compiles and runs, not
about what it costs, and nothing here should be read as an AMD number.

Metal *is* measured now, in the M3 Pro section above; it was not before,
and the reason was partly that neither harness could reach it.
`bench-linalg-gpu` failed to build on Metal at all -- the factorization
and BLAS-1 halves together exceeded what Apple's compiler will put in one
metallib -- and the PyTorch linalg baseline was CUDA-only in four places.
Both are fixed, and BLAS-1 is now `bench-blas1-gpu`, its own file for that
reason.

## Bench tasks

```bash
pixi run bench          # CPU: numax.core.tensor.map vs. a hand-rolled raw-SIMD loop
pixi run bench-gpu     # CPU vs. GPU (map[gpu=True]) across a size sweep
pixi run bench-roofline # GPU: how much memory bandwidth map[gpu=True] reaches
pixi run bench-elementwise # CPU: serial vs. threaded at six sizes
pixi run bench-fusion   # CPU + GPU: composing inside step vs. chaining maps
pixi run bench-matmul   # CPU: the Array tier's matmul vs. MAX's linalg.matmul
pixi run bench-linalg   # CPU: the Tensor tier vs. the linalg.matmul ceiling
pixi run bench-linalg-gpu # the factorizations on a device (CUDA/Metal)
pixi run bench-blas1-gpu # BLAS-1 on a device; separate, see the Metal note above
pixi run bench-numpy    # cross-language: NumPy, CPU
pixi run bench-mlx      # cross-language: MLX, CPU + GPU (macOS only)
pixi run bench-torch    # cross-language: PyTorch (eager + compile), CPU + GPU
pixi run bench-cupy     # cross-language: CuPy, GPU (Linux/CUDA only)
pixi run -e bench-python bench-scipy-linalg # linalg baseline: LAPACK (OpenBLAS or Accelerate), CPU
pixi run -e bench-python bench-torch-linalg # linalg baseline: cuSOLVER on CUDA, MPS on Metal
pixi run -e bench-python bench-cupy-linalg  # linalg baseline: cuSOLVER, CUDA
pixi run bench-thermite # cross-language: Rust thermite, CPU (NEON/AVX2)
pixi run accuracy       # CPU: max error per function vs. checked-in mpmath refs
```

For methodology, how to run each, and the full results tables, see
[`bench/README.md`](../bench/README.md).
