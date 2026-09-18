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

The harness also found what governed the whole table until it was
fixed: Mojo's `std.math` `exp`, `log`, and `erf` are not correctly rounded
at float64 (`exp(1.0)` is wrong from the 13th significant digit, `log`
has a `2e-10` absolute floor, `erf` uses float32 coefficients, while
`sin`, `sqrt` and `erfc` are within a few ulp). `Plain` now takes those
three from `numax/core/libm.mojo` -- fdlibm's algorithms in SIMD form,
within one ulp -- and every function built on them dropped from `1e-9`
to `1e-15` relative in the same run: `erf` to 2.1e-16, `lgamma` to
2.8e-14, the incomplete gamma and beta to `1e-15`, the real-order Bessel
family to `1e-15`. See `bench/accuracy/README.md` for the full table and
the writeup.

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
| `matmul` (ceiling) | 776 | 833 | 0.93 |
| `cholesky` | 22.2 | 99.5 | 0.22 |
| `lu_factor` | 20.1 | 43.9 | 0.46 |
| `solve` | 20.4 | 52.6 | 0.39 |
| `qr_factor` | 4.4 at the default block, 9.6 at `block=4` | 12.4 | 0.36-0.77 |

**GPU -- NVIDIA A10G (Ampere).** Same counts, `n = 1024` except `qr`,
which is `n = 512`:

| op | numax | PyTorch (cuSOLVER) | CuPy (cuSOLVER) |
|---|---|---|---|
| `matmul` (ceiling) | 20,459 | 15,342 | 14,995 |
| `cholesky` | 44.6 | 595.0 | 349.9 |
| `lu_factor` | 30.6 | 282.9 | 266.8 |
| `solve` | 28.1 | 257.6 | 257.9 |
| `qr` (n=512) | 8.4 | 80.0 | 79.1 |

**Re-verified under `max-core 26.6` / `mojo 1.1` (`08a161e`), same EPYC
7R32 + A10G box.** Every CPU row above rose 14-44% from the session this
table first recorded -- `matmul`'s own ceiling went 656 to 776 GFLOP/s,
and `cholesky`, `lu_factor`, `solve` and `qr_factor` all improved by more
than that, so the panel side gained on top of the GEMM. The GPU table
holds for every op except `cholesky`, which dropped from 51.0 to
**44.6 GFLOP/s** (-12.5%, reproduced across two runs) while its own
ceiling (20,447-20,496 across two runs, unchanged from 20,459) and its
three siblings -- `lu_factor`, `solve`, `qr`, which share its
panel-then-GEMM shape -- all sit within a percent of their prior figures.
The regression is isolated to `cholesky`'s own panel or trailing update,
not the panel pattern generally, and has not been root-caused; a
`parallelize` signature change in the same bump
(`08a161e`'s commit body) is the leading suspect but that call is
host-only and `cholesky`'s GPU panel does not run through it, so this is
still open.

Three things to read out of those, in order of how much they matter:

1. **The GEMM is not the problem.** MAX's `matmul` reaches 79% (now 93%)
   of OpenBLAS on CPU and *beats* cuBLAS's FP32 path on the A10G -- 20.5
   against 15.3 TFLOP/s. It is also less accurate on the same product
   (max residual 8.4 against cuBLAS FP32's 3.4, where cuBLAS with TF32
   enabled gives 18.2 at 25.2 TFLOP/s), so MAX's `float32` GEMM sits
   between the two vendor paths on both axes. Either way, a factorization
   at 45 GFLOP/s on a device whose GEMM does 20,000 is not being held back
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
| `axpy` | 57.9 | 78.3 | 444.0 | 475.8 | 292.8 |

The reductions on CPU beat OpenBLAS's own level-1 routines by 1.4-5x,
which is the clearest single payoff of routing them through MAX's
`rowwise` scaffolder: `sdot` and `snrm2` are one thread, `ReduceSum` under
`rowwise` is all 16 cores. On the GPU they land between CuPy and PyTorch.

`axpy` is the exception and the reason is its signature, not its kernel:
it returns a new vector rather than updating `y` in place, so every call
allocates and first-touches 256 MiB. Re-verified under `max-core 26.6` /
`mojo 1.1`: `axpy` rose from 25.3 to 57.9 GB/s on the CPU and 342.4 to
444.0 on the A10G, while `dot`/`nrm2`/`asum` held within a couple percent
on both -- closing more than half its old gap to OpenBLAS's in-place
`saxpy`. The ms breakdown below predates this run and was not re-split;
the direction is consistent with the allocation path getting cheaper
rather than the reduction itself. Measured directly at `n = 16M`: the
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
  through `pack_block` instead of a scalar `elementwise` walk. An eighth
  -- the block step's nine launches cut to seven, and to six on the
  solve path -- moved nothing measurable: 39.8 ms before against 38.3
  and 40.5 in two runs after at `n = 1024`, and on Metal 39.3 -> 38.9 ms
  at `n = 512`, all inside the run-to-run band. `axpy`
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
  LU is still bounded by its panel even after the recursion. QR's launch
  count has been cut and turned out not to be the cost: `larft_panel`
  builds `T^T` directly instead of a pack transposing it, one
  `pack_reflectors` writes `V` and `V^T` together (the second
  materialized orientation is structural -- `matmul` transposes `b` and
  never `a` -- but the second launch was not), and a block spanning full
  rows is read in place rather than staged. Seven launches per step, six
  on the solve path, and the tables did not move, which is the
  measurement the backlog item was waiting for: at `n = 512` on Metal a
  QR is 32 steps, so the 64 launches removed were well under a
  millisecond of a 39 ms run, and the rest is the single-block panel.
  What remains for QR is that panel, and `T` rebuilt on every `q()` and
  `solve()`, which `numax/linalg/qr.mojo`'s own docstring argues against
  changing (`nb^3 / 3` against the products it enables) and which needs
  measuring before it is.

### Scaling with n

Every linalg table above this one is a single size. `n = 1024` was the
largest the harness had ever instantiated, so "numax is only useful for
small tensors" had no measurement against it either way. The sweep now
runs the four factorizations to **n = 4096** on the CPU and to **n = 2048**
on Metal, and the answer is the opposite of the critique: the ratio to
LAPACK **rises** with `n`, because a blocked factorization's trailing
update is `O(n^3)` in `linalg.matmul` while its panel is `O(n^2 block)` on
one core.

**CPU -- Apple M3 Pro, `float32`, numax `bench-linalg` against SciPy
1.18.1 on Accelerate (`bench-scipy-linalg`), best of four numax runs and
two SciPy runs, load average 2.3 to 5.3 with no other process above 1% of
a core.** Each cell is `numax GFLOP/s / LAPACK GFLOP/s`:

| `n` | `cholesky` | `lu_factor` | `solve` | `qr_factor` |
|---|---|---|---|---|
| 128 | 0.14 | 0.18 | 0.35 | 0.75 |
| 256 | 0.10 | 0.21 | 0.29 | 0.81 |
| 512 | 0.17 | 0.23 | 0.36 | 0.58 |
| 1,024 | 0.23 | 0.32 | 0.54 | 0.66 |
| 2,048 | 0.26 | 0.33 | 0.48 | 0.78 |
| 4,096 | **0.41** | **0.42** | 0.50 | 0.69 |

and the absolute numbers behind them (GFLOP/s, numax then LAPACK):

| `n` | `cholesky` | `lu_factor` | `solve` | `qr_factor` | `matmul` ceiling |
|---|---|---|---|---|---|
| 128 | 7.3 / 52.3 | 7.6 / 42.4 | 7.7 / 22.2 | 12.1 / 16.1 | 558 / 882 |
| 256 | 15.7 / 151.8 | 20.1 / 97.6 | 17.8 / 60.5 | 19.1 / 23.5 | 1,030 / 1,293 |
| 512 | 40.6 / 234.4 | 42.4 / 180.4 | 40.4 / 112.3 | 22.2 / 38.4 | 1,412 / 1,600 |
| 1,024 | 81.1 / 356.8 | 84.4 / 263.0 | 86.2 / 160.9 | 35.9 / 54.1 | 1,487 / 1,406 |
| 2,048 | 113.0 / 436.9 | 116.7 / 355.1 | 116.6 / 241.3 | 57.9 / 74.0 | 1,442 / 1,550 |
| 4,096 | 140.2 / 339.7 | 137.5 / 331.6 | 130.4 / 261.6 | 58.3 / 84.8 | 1,324 / 1,386 |

numax's own throughput rises by **19x** from `n = 128` to `n = 4096` on
`cholesky` (7.3 to 140.2 GFLOP/s) and 18x on `lu_factor`, while LAPACK's
rises 6.5x and 7.8x and then turns over at 4096. That turn is why the last
row's ratios jump: Accelerate loses ground at 4096 (`cholesky` 436.9 to
339.7) where numax does not, so part of the final gain is LAPACK falling
back rather than numax pulling forward. The honest summary is that the gap
narrows steadily from `n = 256` on and is roughly halved across the sweep.

The `n = 128` column is the one to distrust in both directions. LAPACK's
`cholesky` there measures 52.3 GFLOP/s against 151.8 at `n = 256` -- call
overhead, not arithmetic -- and numax's `qr_factor` ratio of 0.75 at 128
and 0.81 at 256 is the same effect on the other side. Nothing at `n = 128`
is measuring a factorization.

`qr_factor` is the row that does not improve. It sits between 0.58 and
0.81 with no trend, and at `n = 4096` it is the only routine still under
60 GFLOP/s. Its default `block = 16` is the reason the file's own
docstring gives -- the best block for QR *shrinks* with `n` while the
others hold -- and the `q()`/`solve()` rebuild of `T` is the rest.

**Metal -- the same M3 Pro's 18-core GPU, `float32`, `bench-linalg-gpu`
against PyTorch 2.13.0 on MPS (`bench-torch-linalg`).** A separate
processor and a separate baseline, so no cell here may be read against a
cell above. Each cell is `numax GFLOP/s / PyTorch-MPS GFLOP/s`:

| `n` | `cholesky` | `lu_factor` | `solve` | `qr_factor` |
|---|---|---|---|---|
| 256 | 0.41 | 0.38 | 0.43 | 0.72 |
| 512 | 0.43 | 0.35 | 0.55 | **1.48** |
| 1,024 | 0.54 | 0.35 | 0.81 | -- |
| 2,048 | 0.55 | 0.37 | **1.18** | -- |

and the absolute numbers (GFLOP/s, numax then PyTorch MPS):

| `n` | `cholesky` | `lu_factor` | `solve` | `matmul` ceiling |
|---|---|---|---|---|
| 256 | 3.3 / 8.1 | 1.2 / 3.1 | 1.0 / 2.4 | 137 / 144 |
| 512 | 15.2 / 35.2 | 4.8 / 13.9 | 4.3 / 7.8 | 734 / 612 |
| 1,024 | 65.0 / 120.7 | 18.9 / 53.5 | 16.9 / 21.0 | 1,867 / 1,813 |
| 2,048 | 182.2 / 329.5 | 66.1 / 176.3 | 59.7 / 50.4 | 2,986 / 4,936 |

The same shape, more steeply: numax's Metal `cholesky` goes from 3.3 to
182.2 GFLOP/s across three doublings, a **55x** rise, because the panel's
single-thread-block kernel and its per-step launch latency are a fixed
cost that `n^3` of GEMM eventually buries. `solve` passes PyTorch at
`n = 2048` and `qr_factor` is 1.48x ahead at 512.

**The two `qr_factor` cells are blank on purpose.** PyTorch's MPS
`linalg_qr` is limited to `min(m, n) <= 512` and silently falls back to
the CPU above it (it warns, and the warning is in the harness output), so
its `n = 1024` and `n = 2048` rows are CPU numbers. Putting them in a
Metal column would be exactly the CPU/GPU mixing this page forbids, so
they are left out rather than quietly compared.

### The 0.2 surfaces, measured

Everything the `Tensor` tier gained in 0.2 shipped with a `ponytail:` note
naming its ceiling and no number saying how far away that ceiling was.
These are the numbers: the same **Apple M3 Pro**, `float32`, numax
against SciPy 1.18.1 on Accelerate, one processor per row. The harnesses
are `bench_linalg.mojo`'s new spectral table and the three new files
`bench_signal.mojo`, `bench_interpolate.mojo`, `bench_stats.mojo`, each
with a `bench/scipy/` baseline printing the same columns from the same
data.

**Spectral, `n = 1024`** (ms per call; GFLOP/s from Golub and Van Loan's
counts, so the two columns are comparable to each other and to the
factorization table above, not to a flop counter). Re-measured after the
whole A-lane landed: `bench-linalg` against `bench-scipy-linalg`, **Apple
M3 Pro**, `float32`, best of four numax runs and two SciPy runs, load
average 2.3 to 5.3 with no other process above 1% of a core. A spectral
row's four samples span up to 1.3x -- more than any factorization row --
so the third digit is not a measurement:

| op | numax ms | numax GFLOP/s | LAPACK ms | LAPACK GFLOP/s | numax / LAPACK | was (0.2 draft) |
|---|---|---|---|---|---|---|
| `eigvalsh` | 142 | 10.1 | 39.8 | 35.9 | **0.28** | 0.11 |
| `eigh` | 199 | 48.5 | 88.8 | 108.9 | **0.45** | 0.015 |
| `svdvals` | 1,459 | 2.0 | 50.1 | 57.2 | 0.034 | 0.022 |
| `svd` | 1,583 | 14.9 | 89.1 | 265.0 | 0.056 | 0.002 |
| `eigvals` | 920 | 11.7 | 139.2 | 77.1 | **0.15** | 0.056 |
| `schur` | 1,056 | 25.4 | 165.3 | 162.4 | **0.16** | 0.021 |

The last column is the table this section shipped with, and every row
above it is the same call on the same machine, so the two columns are a
before and an after: **`eigh` is 30x faster than it was** (6,101 ms to
199), `svd` 26x (41,465 to 1,583), `schur` 6.9x, `eigvals` 2.7x,
`eigvalsh` 2.5x, `svdvals` 1.55x. What moved them is in the record below.

Against the targets the plan set for this work: `eigh` was to reach 0.25
of LAPACK and reaches 0.45; `eigvalsh` was to reach 0.2 and reaches 0.28;
`schur` was to improve with its ceiling stated and improved 7.5x.
**`svd` was to reach 0.08 and reaches 0.056, so that one is missed**, for
the reason the next paragraph gives.

**What each row is waiting on now**, which is no longer the band
iteration in any of them:

- **`svd` and `svdvals` are `gebrd`, and `gebrd` is two whole-matrix
  products per column.** The two rows are 1,583 and 1,459 ms and the
  reduction is about 1,450 of both, which is why a values-only run is
  barely cheaper than one that forms `U` and `V`: everything the vectors
  cost is roughly 120 ms. The `labrd` panel below took the per-column
  traffic from four passes over the matrix to two, and the two that are
  left -- `A v` and `A u` -- are the ceiling. They are `matvec` at
  whole-matrix shape, so the fix is not a wider panel; it is the
  two-stage dense-to-banded reduction, which is 0.3 work.
- **`eigvalsh` is `sytrd`, and `sytrd` is the same product once.** 142 ms
  at the default `block = 32`, and **74.8 ms at `block = 8`** -- a ratio
  of 0.53 to LAPACK -- because a `latrd` column costs `O(n * block)`
  whatever it is threaded on while the trailing GEMM a wider panel saves
  is a few milliseconds at any width. A values-only caller should pass
  `eigvalsh[..., block=8]`; 32 stays the default because `eigh` pays for
  the rotation window and `q()` out of the same number.
- **`eigh` is the one row where the reduction is no longer most of the
  run**: 199 ms against `eigvalsh`'s 142 at the same block, so the
  vectors -- the windowed rotation GEMMs plus `orgtr`'s panel walk --
  cost about 57 ms for `9n^3` of work. That is the shape the whole
  accumulation design was for.
- **`schur` and `eigvals` are the host Francis iteration.** `eigvals` is
  920 ms of which the `lahr2` reduction is about 146, so five sixths of
  the call is `_hqr` on the host; `schur` adds 136 ms of Schur vectors
  and its far-from-diagonal `T` update on top. Only multishift QR with
  aggressive early deflation moves that term, and it is 0.3 work, filed
  in `docs/parity.md` with these numbers.

**`float32` residuals are LAPACK's too**, and on three rows better.
numax's trace gap on `eigvalsh` at `n = 1024` is 0.0041 against
Accelerate's 0.375, its `eigvals` gap 0.106 against 0.833 and its `schur`
residual 0.0091 against 0.0188; `svd` is a tie at 0.0106 against 0.0100,
and the two numax loses are `eigh` at 1.5e-3 against 2.7e-4 and `svdvals`
at 5.5e-7 against 3.8e-8. All at the same precision, not a wider one: the
band iteration runs at the caller's `dtype` throughout (`_tql` is generic
over it and `sytrd` hands it `List[Scalar[dtype]]`), and there is no
`float64` promotion anywhere in these routines.

**One known correctness bug is not in this table, and is now guarded.**
Every row above is `gpu=False`. The same routines named with `gpu=True`
returned wrong answers on Metal -- `sytrd[gpu=True]` disagrees with the
host band at `n = 4` already, at every `block` including 1, so it predates
the `latrd` panel -- and `svd[gpu=True]` raised rather than converging.
Nothing in numax, its tests or its examples called that spelling, which is
why it went unnoticed. **`gpu=True` no longer compiles** on any spectral
routine or on `pinv`/`cond`/`matrix_rank`: a `comptime assert` in each body
refuses it and names the routine, so a wrong answer is no longer reachable.
There is still no device spectral table here, because a compile error is
not a measurement either -- fixing the path is Backlog 0.3. See "What is
not measured" below.

**How it got here.** The four changes that produced the after column,
each measured back to back against the commit before it on the same
machine, are kept as the record of which term each one moved.

`sytrd` through a `latrd` panel, and then `latrd_w`'s own `O(n j)`
arithmetic off the single thread block -- the `2j + 1` reductions one task
each and the `w` build one task per row, two `parallelize`s on the host
and two `elementwise`s on the accelerator. `n = 1024`, `float32`, twice
and taking the smaller, the machine not otherwise quiet:

| `block` | `sytrd` before | after | `eigvalsh` before | after | `eigh` before | after |
|---|---|---|---|---|---|---|
| 8 | 109.2 | 111.0 | 117.0 | 120.4 | 460.5 | 469.4 |
| 16 | 120.7 | 117.9 | 127.0 | 130.9 | 283.5 | 287.2 |
| 32 | 146.6 | 150.0 | 157.7 | 156.9 | 272.5 | 275.4 |
| 64 | 197.3 | 189.1 | 206.4 | 189.1 | 304.9 | 287.7 |

The slope there is the panel's own arithmetic and not a scheduling
artifact, and what the threading buys is the wide end, where the serial
term was large enough to beat the dispatch. Below `_LATRD_MIN_WORK` a
`parallelize` per column costs more than the work it spreads, so narrow
panels stay on the serial path; the threshold is an absolute work count,
so the same code threads a narrower panel as `n` grows.

`gebrd` through a `labrd` panel, the two rank-one updates per column
deferred into two GEMMs per panel. Same method, `n = 1024`:

| `block` | `gebrd` before | after | `svdvals` before | after | `svd` before | after |
|---|---|---|---|---|---|---|
| 16 | 2,361 | 1,650 | 2,348 | 1,693 | 2,913 | 2,227 |
| 32 | 2,355 | 1,740 | 2,392 | 1,733 | 2,654 | 2,059 |
| 64 | 2,419 | 1,769 | 2,371 | 1,783 | 2,646 | 2,002 |

About 27% off `gebrd` and `svdvals` and 23% off `svd`, and the reason is
traffic rather than flops: the unblocked reduction made four passes over
the matrix per column and the panel defers both subtracts, leaving two.
The sweep is nearly flat in `block` because `gebrd`'s panel corrections
are `O((m + n) j)` against two whole-matrix passes rather than one; 32
stays the default, between `gebrd`'s own best at 16 and `svd`'s at 64.

`hessenberg` through a `lahr2` panel, the panel's `V`, `T` and
`Y = A V T` accumulated per column and the two-sided update deferred into
one `transpose_b=True` GEMM on the right and `larfb`'s three on the left.
Measured on the `_general` fixture, alternating binaries, three rounds
each and taking the minimum, `n = 1024`, load average 4.8 to 8.8:

| | before | after |
|---|---|---|
| `hessenberg`, `block = 16` | 1,474 | 108 |
| `hessenberg`, `block = 32` | 1,461 | 146 |
| `hessenberg`, `block = 64` | 1,457 | 199 |
| `eigvals`, `block = 16` | 2,261 | 915 |
| `eigvals`, `block = 32` | 2,263 | 924 |
| `eigvals`, `block = 64` | 2,280 | 999 |
| `schur`, `block = 32` | 2,442 | 1,099 |

The `before` column repeats because `block` reached only `.q()` there and
neither `hessenberg` nor `eigvals` forms `Q`; the spread across its three
rows is the honest error bar on the column. A factor of ten on the
reduction, for the reason the `labrd` paragraph gives at a factor of four
thirds. 32 stays the default because `eigvals` is nearly flat across it
and because the same number is `.q()`'s panel width, which `schur` forms.

And the `2n x 2n` Golub-Kahan doubling is gone: `_bdsqr` chases the
bidiagonal itself and pushes a rotation into each of two batches, so
`U_B^T` and `V_B^T` are accumulated at width `n` instead of the
eigenvectors of a `2n` tridiagonal being de-interleaved afterwards. Same
method, `n = 1024`, load average 3.8 to 9.5:

| `block` | `svdvals` before | after | `svd` before | after |
|---|---|---|---|---|
| 16 | 1,621 | 1,615 | 2,034 | 1,686 |
| 32 | 1,644 | 1,648 | 1,949 | 1,768 |
| 64 | 1,704 | 1,691 | 1,949 | 1,752 |

Nine to seventeen per cent off `svd` and nothing off `svdvals`, and both
are the honest numbers: a values-only sweep pushes no rotation, so the
doubling only ever cost it tens of milliseconds of band arithmetic, while
the vectors were paying for stripes four times the area. Anyone reading
"four times fewer rotations" as a factor on the call should read the
`svdvals` row first.

**Convolution -- where `fftconvolve` overtakes `convolve`** (µs per call,
`full` mode; the two numax rows at each `(m, k)` agree to `1e-6`, and so
do SciPy's). Re-measured on a quiet machine after the engine change,
best of two runs each, **Apple M3 Pro**, `float32`, load average 3.2 to
6.0 with no other process above 1% of a core:

| `m` | `k` | numax direct | numax FFT | SciPy direct | SciPy FFT |
|---|---|---|---|---|---|
| 4,096 | 8 | 12.7 | 272 | 5.8 | 44.9 |
| 4,096 | 32 | 59.2 | 272 | 31.0 | 44.7 |
| 4,096 | 128 | 319 | 272 | 45.4 | 44.8 |
| 4,096 | 512 | 1,743 | 275 | 126 | 45.1 |
| 4,096 | 2,048 | 7,870 | 271 | 635 | 52.6 |
| 65,536 | 8 | 83.0 | 5,638 | 58.5 | 749 |
| 65,536 | 32 | 329 | 5,602 | 458 | 752 |
| 65,536 | 128 | 1,761 | 5,579 | 679 | 784 |
| 65,536 | 512 | 9,902 | 5,581 | 1,937 | 787 |
| 65,536 | 2,048 | 43,656 | 5,514 | 11,060 | 475 |

**The crossover, which is what this table is for**: numax's transform
route overtakes its direct one at about **`k = 110` at `m = 4,096`** and
**`k = 310` at `m = 65,536`**, against SciPy's `k ~ 128` and `k ~ 165` on
the same runs. So the acceptance target of `k <= 128` is met at the
smaller length and missed at the larger, and the reason is the row above:
numax's direct convolution is competitive (it *beats* SciPy's at
`m = 65,536, k = 32`, 329 µs against 458) while its transform is 5-12x
behind `pocketfft`, so the ratio that sets the crossover is the transform's.
`numax/signal/convolution.mojo`'s docstring carries these two numbers.

**The engine is fused and radix-4**, and the launch count is the whole of
the change: `1 + ceil((log2(n) - 6) / 2)` rather than `log2(n) + 1`, so 7
launches at `n = 2^18` where there were 18, and 1 at every `n <= 64`. One
kernel does the bit-reversal gather and the first six stages in 128
registers, each pair of stages after it is one radix-4 kernel over `n/4`
butterflies, and the inverse `1/n` rides on the last kernel's stores
instead of a pass of its own.

**Against `pocketfft`** (`bench-fft` against `bench-scipy-fft`, same
machine and session, best of two numax runs; numax's figure is the timed
call minus that size's `input` row, which is the buffer construction every
transform here consumes):

| `n` | numax `fft` | SciPy `fft` | ratio | numax `rfft` | SciPy `rfft` | ratio |
|---|---|---|---|---|---|---|
| `2^10` | 16.5 | 4.0 | 4.1 | 15.2 | 4.1 | 3.7 |
| `2^12` | 49.8 | 10.6 | 4.7 | 39.8 | 8.8 | 4.5 |
| `2^14` | 173 | 42.5 | 4.1 | 158 | 28.7 | 5.5 |
| `2^16` | 875 | 213 | 4.1 | 808 | 155 | 5.2 |
| `2^18` | 4,075 | 1,241 | 3.3 | 3,755 | 957 | 3.9 |
| `2^20` | 17,647 | 6,002 | **2.9** | 16,631 | 3,817 | 4.4 |

and `fft2` at `512 x 512` is 4,782 against 887, a ratio of 5.4.

**The honest summary is 3 to 5.5x behind, not "about 3x".** 2.9x is the
best cell in the table -- the complex `fft` at the largest size, where the
launch count is amortized over the most work -- and the real transforms
sit nearer 4-5x because `rfft` pays a pack/unpack pass that `pocketfft`
folds into its own butterflies. Before the fused engine the same
comparison was 14x, so the launch count was most of the old gap and is
not most of what is left. What is left is that one radix-4 launch still
writes its result to memory and reads it back, where `pocketfft` keeps a
cache-sized block in registers across every stage; a six-stage fused block
at the *top* of the transform as well as the bottom is the shape that
closes it, and it is not written.

**Filters and spectra, `n = 2^20`** (µs per call, best of two runs each,
same machine and session):

| op | numax | SciPy | numax / SciPy | was |
|---|---|---|---|---|
| `lfilter`, 32-tap FIR | 10,649 | 7,926 | 0.74 | 0.08 |
| `filtfilt`, Butterworth order 4 | 12,396 | 14,870 | **1.20** | 0.44 |
| `medfilt`, kernel 5 | 2,371 | 4,870 | **2.05** | 2.0 |
| `savgol_filter`, window 11 / order 3 | 2,401 | 4,951 | **2.06** | 1.9 |
| `welch`, `nperseg = 256` | 9,480 | 84,035 | **8.9** | 12.4 |

The two recurrences are the rows that moved. `lfilter` was a `Float64`
host loop over four full-length `List`s and 12x behind SciPy's C; a
`Scalar[dtype]` loop over the tensor's own mapped buffer is **9.3x
faster** and now 1.35x behind, which meets the acceptance target of 2x.
`filtfilt` runs its backward pass in place over the extended buffer
instead of materializing a reversed copy and is now **ahead** of SciPy.
The recurrence itself stays host-side and the module docstring says why:
a first-order recurrence has no parallel form that is worth its constant.

`welch` reads 8.9x rather than the 12.4x this table first published, and
the ratio moved because **SciPy got faster on this machine, not numax
slower** -- 84,035 µs here against the 83,839 the first table recorded,
while numax went 6,780 to 9,480. Both numax figures are `welch` through
one batched `rfft` over 8,191 segments; the difference is that the 6,780
was taken before the FFT engine was rewritten and on a machine whose state
is not recoverable. 8.9x is the number measured on a quiet machine with
both halves in one session, and it is the one to use.

**Interpolation** (µs per call, `n = 1024` knots, `m = 2^20` queries):

| op | numax | NumPy / SciPy | numax / SciPy |
|---|---|---|---|
| `interp` | 4,605 | 43,508 | **9.4** |
| `CubicSpline` evaluation | 5,105 | 19,763 | **3.9** |
| `CubicSpline` construction, `n = 1024` | 234 | 72.6 | 0.31 |
| `CubicSpline` construction, `n = 4096` | 428 | 142 | 0.33 |

Evaluation is the shape a `Tensor` tier exists for -- a vectorized
`searchsorted` and a gather, one launch each -- and is 4-9x ahead.
Construction is the not-a-knot tridiagonal system on the host, 3x behind
SciPy's `solve_banded`, at a cost that is a quarter of a millisecond and
independent of how many points are later evaluated.

**Statistics** (µs per call; `norm.cdf`, `histogram` and `quantile` at
`n = 2^24`, `cov`/`corrcoef` on `8 x 2^20`; best of two runs each, same
machine and session, load average 2.6 to 2.7):

| op | numax | NumPy / SciPy | numax / SciPy | was |
|---|---|---|---|---|
| `norm.cdf` | 34,392 (3.9 GB/s) | 217,048 (0.6 GB/s) | **6.3** | 5.4 |
| `histogram`, 64 bins | 70,012 | 75,181 | **1.07** | 1.05 |
| `quantile`, `q = 0.5` | 150,546 | 52,765 | 0.35 | 0.05 |
| `cov`, 8 variables | 4,133 | 13,502 | **3.27** | 0.11 |
| `corrcoef`, 8 variables | 4,012 | 13,557 | **3.38** | 0.11 |

`norm.cdf` is one `elementwise` through the tier-1 `erf` and beats
`scipy.stats` by 6x -- and is still at 3.9 GB/s on a machine whose memory
moves `~150`, so it is compute-bound on the `erf` polynomial rather than
bandwidth-bound: the same `elementwise` walk moves `medfilt` over `2^20`
points in 2.4 ms. `histogram` matches NumPy; both are host-side counts.

**`cov` is the largest single ratio this page records.** It was an
`O(rows^2 n)` `Float64` host loop at 124,783 µs and is now a Welford
`variance_axis` for the means, one `broadcast_op_axis` to centre, and one
`inner` for the Gram matrix with the `1/(n - ddof)` scaling folded into
the matmul epilogue: **30x faster** and 3.3x ahead of NumPy, because the
cubic term is `linalg.matmul` and NumPy's `cov` is not a GEMM.
`corrcoef` adds one `elementwise` over the outer product of the
diagonal's square roots and costs nothing more.

**`quantile` is the row that improved most and still loses.** The host
sort of a `List[Float64]` of the whole tensor is gone -- it partitions at
the one or two indices the method needs, at `dtype` rather than widened --
which is **7.3x** off the old figure, from 21x behind NumPy to 2.85x. It
misses the acceptance target of 2x, and the reason is not the selection
any more: at `n = 2^24` the call still moves the whole 64 MB tensor to the
host before partitioning, and 64 MB at the `to_host` path's rate is most
of the 150 ms. The device route exists for the comptime-shaped median
(`top_k[largest=False]` at `k = n//2 + 1` is fully device-resident) and
does not cover the general `q`. That download is the ceiling, and closing
it is a device selection network, not a better partition.

### The core surface, measured

At `b0c0def` every NumPy-named routine over `Tensor` -- `exp(a)`, `a + b`,
`a * 2`, the comparisons, `sum` -- was a `to_host()`, a scalar `for` loop
and a rebuild, whatever the tensor's memory or the caller's target. 0.2
routes all of them through `numax.core._drive`, which picks a serial SIMD
loop, `max.algorithm.elementwise[target="cpu"]` or
`elementwise[target="gpu"]` from one `gpu: Bool` and one size threshold.
These are the numbers.

Harness `bench/bench_core_surface.mojo` (`pixi run bench-core-surface`)
against `bench/numpy/core_surface.py`
(`pixi run -e bench-python bench-numpy-core-surface`), the same hashed data
on `[-3, 3)` and the same byte count per row in both. **Apple M3 Pro**, 6
performance and 6 efficiency cores, `float32`, NumPy 2.5.2. Each figure is
the best of five consecutive runs for the 0.2 column and the best of three
for the other two: the four elementwise rows allocate their destination
per call the way a caller's would, and that allocation occasionally hits a
slow path in the CPU allocator, which puts a 10-30x outlier in one run out
of five at `n <= 2^14` and leaves the rows that allocate nothing (the
crossover table below) reproducible to 1%. The `uptime` figure read 2.0 to
5.8 before each run, which is the decay tail of the run before it;
`ps -Ao pid,%cpu,comm` between runs showed nothing above 1% of one core.

**At `n = 2^24`** (µs per call; `B/elem` is the bytes each row moves per
element, so the GB/s columns are comparable only within a row):

| op | B/elem | `b0c0def` µs | 0.2 µs | 0.2 GB/s | speedup | NumPy µs | numax / NumPy |
|---|---|---|---|---|---|---|---|
| `exp(a)` | 8 | 55,745 | 1,624 | 82.6 | **34.3x** | 25,153 | **15.5** |
| `a + b` | 12 | 73,530 | 2,153 | 93.5 | **34.2x** | 5,271 | **2.4** |
| `a * 2` | 8 | 53,732 | 1,580 | 85.0 | **34.0x** | 4,641 | **2.9** |
| `greater(a, zeros)` | 9 | 63,940 | 1,397 | 108.1 | **45.8x** | 4,875 | **3.5** |
| `sum(a)` | 4 | 19,698 | 567 | 118.3 | **34.7x** | 2,000 | **3.5** |

The host walk was moving 2.4-3.4 GB/s at every size above `2^14` -- one
element per iteration, one round trip per call -- and the routed surface
moves 83-118, against a machine whose memory tops out near 150. `sum` at
118 GB/s is 79% of that. The acceptance question this table was built to
answer ("only useful for small tensors") is answered in the last column:
at `2^24` numax is ahead of NumPy on all five, and `greater` is ahead
while moving 9 bytes per element to NumPy's 5 -- NumPy's own `a > 0`
spelling, which broadcasts a scalar instead of naming a zero tensor, is
3,903 µs, still 2.8x the routed comparison.

**Do not read that last column as a kernel comparison.** Most of it is
core count: `elementwise[target="cpu"]` runs on all twelve, NumPy's
ufuncs are single-threaded. On a pure memory-bound add NumPy reaches 38.2
GB/s on one core where numax reaches 93.5 on twelve, so per core NumPy's
kernel is the better one and the ratio is a statement about the launch
policy, not about the arithmetic. The one row where numax's *serial*
kernel is genuinely faster is `exp`: the crossover table's serial walk
moves 21.3 GB/s through `std.math.exp`, four times NumPy's 5.3 for
`np.exp` at `float32`.

**At `n = 2^10`** the same five rows, where the policy deliberately does
*not* thread (µs per call):

| op | `b0c0def` | 0.2 | NumPy |
|---|---|---|---|
| `exp(a)` | 8.95 | 0.81 | 1.69 |
| `a + b` | 12.22 | 0.51 | 0.39 |
| `a * 2` | 8.81 | 0.51 | 0.51 |
| `greater(a, zeros)` | 11.60 | 2.79 | 0.58 |
| `sum(a)` | 3.47 | 3.56 | 0.51 |

Ten to twenty-four times faster than the host walk and within about 1.3x
of NumPy on the two arithmetic rows, which is the allocation and the
device gate. `sum` does not move at all at this size, because its
`b0c0def` spelling was already a MAX reduction rather than a host loop;
what 0.2 changed for it is the 2^24 row above. `greater` is the outlier
at 2.79 µs, and it is the one row whose five samples spanned 3.6x, so
treat it as "a few microseconds" rather than as a measurement.

**Where threading starts to pay, and why `_THREADED_FROM` did not move.**
The last table in the harness runs one `exp` body through
`numax.core.tensor.map` (a serial SIMD walk) and `map_threaded` (the same
walk through `elementwise[target="cpu"]`) over two buffers the caller
already owns, so neither row pays for an allocation and both are
reproducible to 1%:

| `n` | `map`, serial SIMD µs | `map_threaded` µs | threaded / serial |
|---|---|---|---|
| `2^12` | 1.53 | 1.59 | 0.96 |
| `2^14` | 6.19 | 6.22 | 1.00 |
| `2^15` | 12.31 | 12.33 | 1.00 |
| `2^16` | 24.73 | 16.28 | **1.52** |
| `2^18` | 97.55 | 71.52 | 1.36 |
| `2^20` | 390.31 | 79.69 | **4.90** |
| `2^24` | 6,298.6 | 1,581.5 | 3.98 |

The serial walk is flat at 2.65 G elem/s (21.3 GB/s) at every size from
`2^12` up, so the ratio column is entirely what threading adds. It adds
nothing through `2^15` and 1.52x at `2^16`, which puts the crossover
inside a single doubling of `_THREADED_FROM`'s provisional `1 << 16`.
**The constant stays where it was**; this is the measurement that was
owed for it, not a change to it.

Two things in that table are worth naming rather than smoothing. The
`2^18` row is slower *per element* than `2^20` -- 3.7 G elem/s against
13.2 -- and it is not noise: it reproduces across all five runs to
within 1.3x, and the routed `exp(a)` dips at the same size, so it is
`elementwise`'s CPU grain policy rather than anything numax does. And
the jump from 1.00 to 1.52 between `2^15` and `2^16` is sharp enough
that no smoother threshold would fit it better.

At `2^24` the same three spellings read 6,299 µs for `map`, 1,582 for
`map_threaded` and 1,614 for `exp(a)`, so the NumPy-named call costs **2%**
over the primitive it is built on -- the device gate, the rank-1 flatten
and the destination allocation together.

**The same surface on Metal**, `pixi run bench-core-surface-gpu`, same M3
Pro, `float32`, best of three. Separate table and separate processor: no
row here may be compared with a row above. Both sync shapes are reported
because they differ by 5x at small sizes -- **per-call** synchronizes
inside the timed region, so it is one launch through completion, and
**amortized** enqueues ten launches and synchronizes once:

| op | `n` | µs/call | GB/s | µs amortized | GB/s |
|---|---|---|---|---|---|
| `exp(a)` | `2^10` | 125.8 | 0.07 | 24.2 | 0.34 |
| `a + b` | `2^10` | 120.4 | 0.10 | 24.0 | 0.51 |
| `a * 2` | `2^10` | 119.5 | 0.07 | 23.9 | 0.34 |
| `greater(a, zeros)` | `2^10` | 114.8 | 0.08 | 24.0 | 0.38 |
| `sum(a)` | `2^10` | 302.0 | 0.01 | 290.9 | 0.01 |
| `exp(a)` | `2^20` | 289.0 | 29.0 | 138.2 | 60.7 |
| `a + b` | `2^20` | 345.7 | 36.4 | 149.4 | 84.2 |
| `a * 2` | `2^20` | 300.9 | 27.9 | 134.7 | 62.3 |
| `greater(a, zeros)` | `2^20` | 221.1 | 42.7 | 88.7 | 106.4 |
| `sum(a)` | `2^20` | 348.5 | 12.0 | 332.5 | 12.6 |
| `exp(a)` | `2^24` | 2,674 | 50.2 | 2,106 | 63.7 |
| `a + b` | `2^24` | 3,284 | 61.3 | 2,204 | 91.4 |
| `a * 2` | `2^24` | 2,672 | 50.2 | 2,087 | 64.3 |
| `greater(a, zeros)` | `2^24` | 1,911 | 79.0 | 1,387 | 108.9 |
| `sum(a)` | `2^24` | 1,017 | 66.0 | 1,033 | 65.0 |

Below about `2^16` the table is launch latency and nothing else: every
elementwise row costs the same 115-126 µs per call and the same 24 µs
amortized whatever `n` is, because the work is far under one dispatch.
`sum` is the exception that names itself -- it ends in a one-element
`to_host`, which orders against the launch whatever the caller does, so
its two columns are the same measurement twice.

The reading to take from the `2^24` rows is **not** that the GPU is
faster. On an M3 Pro the two processors share one memory controller, so a
memory-bound elementwise kernel reaches 64-109 GB/s amortized on the
device against 83-118 GB/s threaded on the CPU, and the CPU is ahead on
four rows of five. What the device path is for is that the data does not
move: a tensor built on a `DeviceContext` and operated on with `gpu=True`
never round-trips, which at `b0c0def` it did on every single call.

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

**There is no Metal spectral table, and the reason is a correctness bug
rather than a build limit.** The spectral routines all take a `gpu`
parameter, and naming it `True` returned wrong answers on Metal -- it is
now a compile error, so the diagnosis below is the record of what the
guard refuses rather than a live hazard. Measured
against the host path on the same matrices: `eigvalsh[gpu=True]` at
`n = 8` differs from `eigvalsh[gpu=False]` by 6.3 in an eigenvalue and
leaves a trace gap of 3.1 where the host path leaves 5e-6; at `n = 128`
the gap is 90. `svdvals[gpu=True]` reports a Frobenius identity off by
1e22 and `svd[gpu=True]` raises rather than converging. Narrowed one step,
`sytrd[gpu=True]` already disagrees with the host band at **`n = 4`**, and
it disagrees at `block = 1` -- the unblocked algorithm -- so the fault is
older than the `latrd` panel and is not the blocking.

Nothing in numax, its tests, its examples or its benches called that
spelling, which is why it survived: `examples-gpu-build` compiles no
spectral device kernel and CI has no GPU. That is what the `comptime
assert` in each body now closes -- an untested path that answered instead
of refusing. The harness rows for the table
were written and are what found this, and they are not committed, because
a benchmark that times a wrong answer publishes a number worse than none.
They belong in the commit that fixes the device path, where they are the
proof it is fixed. The factorization half of `bench-linalg-gpu` is
unaffected -- `cholesky[gpu=True]` at `n = 256` has a residual of 6e-5 --
so the upload path and the GEMM routing are fine and the fault is inside
the reduction.

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
pixi run bench-linalg   # CPU: the Tensor tier vs. the linalg.matmul ceiling, spectral included
pixi run bench-signal   # CPU: convolve vs. fftconvolve crossover, filters, welch
pixi run bench-fft      # CPU: fft/rfft/irfft/fft2, us per call and the launch count
pixi run bench-interpolate # CPU: interp and CubicSpline, evaluation and construction
pixi run bench-stats    # CPU: norm.cdf, histogram, quantile, cov/corrcoef
pixi run bench-core-surface # CPU: exp, a + b, a * 2, comparison, sum, and map vs. the named call
pixi run bench-linalg-gpu # the factorizations on a device (CUDA/Metal)
pixi run bench-blas1-gpu # BLAS-1 on a device; separate, see the Metal note above
pixi run bench-core-surface-gpu # the same core surface on a device, both sync shapes
pixi run bench-numpy    # cross-language: NumPy, CPU
pixi run bench-mlx      # cross-language: MLX, CPU + GPU (macOS only)
pixi run bench-torch    # cross-language: PyTorch (eager + compile), CPU + GPU
pixi run bench-cupy     # cross-language: CuPy, GPU (Linux/CUDA only)
pixi run -e bench-python bench-scipy-linalg # linalg baseline: LAPACK (OpenBLAS or Accelerate), CPU
pixi run -e bench-python bench-scipy-signal # signal baseline: scipy.signal, CPU
pixi run -e bench-python bench-scipy-fft    # transform baseline: scipy.fft (pocketfft), CPU
pixi run -e bench-python bench-scipy-interpolate # interpolation baseline: numpy.interp, scipy CubicSpline
pixi run -e bench-python bench-scipy-stats  # statistics baseline: scipy.stats.norm, numpy histogram/quantile/cov
pixi run -e bench-python bench-numpy-core-surface # core-surface baseline: numpy exp/add/scale/compare/sum
pixi run -e bench-python bench-torch-linalg # linalg baseline: cuSOLVER on CUDA, MPS on Metal
pixi run -e bench-python bench-cupy-linalg  # linalg baseline: cuSOLVER, CUDA
pixi run bench-thermite # cross-language: Rust thermite, CPU (NEON/AVX2)
pixi run accuracy       # CPU: max error per function vs. checked-in mpmath refs
```

For methodology, how to run each, and the full results tables, see
[`bench/README.md`](../bench/README.md).
