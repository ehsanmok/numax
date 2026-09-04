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

`bench/{numpy,mlx,torch,thermite}/` run the identical kernel, sizes,
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

The same six benchmark scripts, same kernel, same sizes, run on an NVIDIA
A10G (23 GB, 300 W, CUDA 13, 600 GB/s spec bandwidth) with an AMD EPYC host.
NumPy 2.5.2, PyTorch 2.13.0+cu130, `thermite` 0.2 on its AVX2 backend. MLX is
macOS-only and therefore absent. M elem/s, higher is better.

**GPU, amortized sync:**

| n | numax | torch compile | torch eager |
|---|---|---|---|
| 65,536 | 17,852 | 1,045 | 2,452 |
| 262,144 | 68,445 | 4,070 | 10,337 |
| 1,048,576 | 68,458 | 16,222 | 24,981 |
| 4,194,304 | 58,586 | 50,981 | 19,176 |
| 16,777,216 | 61,008 | 55,007 | 19,785 |
| 67,108,864 | **61,537** | 56,141 | 19,893 |

**GPU, per-call sync** (67M): numax 60,955, torch compile 53,607, torch eager
19,814.

**CPU:**

| n | numax `map_threaded` | numax `map` | torch compile | thermite (AVX2) | torch eager | NumPy |
|---|---|---|---|---|---|---|
| 65,536 | 2,139 | 1,836 | 848 | 1,324 | 547 | 387 |
| 1,048,576 | 3,325 | 1,790 | 8,259 | 1,365 | 1,013 | 391 |
| 16,777,216 | 13,569 | 1,729 | 1,702 | 1,325 | 382 | 303 |
| 67,108,864 | 8,527 | 1,394 | 1,833 | 1,326 | 401 | 313 |

What this machine says, and where it differs from the M3 Pro above:

- **numax's GPU path leads `torch.compile` by ~10% at 67M** (61,537 vs
  56,141 amortized; 60,955 vs 53,607 per-call) and is ~3x eager PyTorch.
  On Metal the two were a tie, so the lead is this device's, not a general
  claim.
- **It is bandwidth-bound, not compute-bound.** `bench-roofline` puts the
  best configuration at **500.9 GB/s** — ~83% of the card's 600 GB/s spec —
  and an identity copy at 489.10 GB/s against the Gaussian's 489.16 GB/s at
  67M. The `exp` is free; what is left to win is the last ~17% of the memory
  path.
- **Kernel fusion is worth 1.99x on the GPU** at every size tested (1.991 /
  1.981 / 2.000 / 1.997 from 1M to 67M) — two chained `map` calls against one
  `map` with a composed `step`. On the CPU the same change is worth 1.26-1.58x.
- **numax's serial CPU walk matches hand-written Rust SIMD** (1,394 vs
  `thermite`'s 1,326 at 67M) and is 4.5x NumPy. Threaded, it is the fastest
  CPU number at 16M and 67M.
- **The threaded CPU path is noisy on this host** — 3,325 M elem/s at 1M
  against 13,569 at 16M, and repeat runs move by a factor of two at the same
  size. Read it as a range, not a point.
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

## Use MAX past N (small-matrix linalg)

`numax/linalg/linalg.mojo`'s matrices are `Array[T, n*n]` in registers -- a
compile-time size that keeps them GPU-launchable (one matrix per SIMD
lane, callable from inside `map[gpu=True]`) and lets `T` be `Dual` or
`Compensated`, at the cost of both compile time and register pressure
growing with `n`. Past a certain `n`, MAX's own `TileTensor`-based,
`dtype`-monomorphic routines win on raw speed. `bench/bench_matmul.mojo`
measures exactly where, for `matmul` (nanoseconds per `n x n` product,
lower is better; the batched column runs 4 independent products per call,
one per SIMD lane, divided by 4 to stay comparable):

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

`matmul` and `qr_factorization` (Householder, in-place) are the only two
dense-linear-algebra primitives MAX itself ships -- there is no MAX
`lu`/`solve`/`det`/`trace`/`norm`/`inverse` at any size, generic or
otherwise (verified against `~/workspace/modular/max/kernels/src/linalg/`).
So for a large, plain-`dtype` system past the crossover above, the
practical move is not a drop-in MAX function call for every name in this
module -- it's building the large-matrix equivalent from
`linalg.qr_factorization` the way a LAPACK-style solver does
(`solve`/`det`/`inverse` all reduce to `R` and `Q^T b` once `A = QR`).
`cholesky`/`cholesky_solve`/`tridiagonal_solve` have no MAX equivalent to
route to at any size, since MAX ships neither a Cholesky factorization nor
a triangular solve. Every function's own docstring in `numax/linalg/linalg.mojo`
states which of these two cases it falls into.

## Bench tasks

```bash
pixi run bench          # CPU: numax.core.tensor.map vs. a hand-rolled raw-SIMD loop
pixi run bench-gpu     # CPU vs. GPU (map[gpu=True]) across a size sweep
pixi run bench-roofline # GPU: how much memory bandwidth map[gpu=True] reaches
pixi run bench-elementwise # CPU: serial vs. threaded at six sizes
pixi run bench-fusion   # CPU + GPU: composing inside step vs. chaining maps
pixi run bench-matmul   # CPU: numax.linalg.matmul vs. MAX's linalg.matmul
pixi run bench-numpy    # cross-language: NumPy, CPU
pixi run bench-mlx      # cross-language: MLX, CPU + GPU (macOS only)
pixi run bench-torch    # cross-language: PyTorch (eager + compile), CPU + GPU
pixi run bench-thermite # cross-language: Rust thermite, CPU (NEON/AVX2)
pixi run accuracy       # CPU: max error per function vs. checked-in mpmath refs
```

For methodology, how to run each, and the full results tables, see
[`bench/README.md`](../bench/README.md).
