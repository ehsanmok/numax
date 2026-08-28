# Performance

> Companion to [bench/README.md](../bench/README.md), which has the
> cross-language methodology, the per-bench descriptions, and the full
> results table. This page is the architectural framing of those
> numbers: what the kernel is doing, what the ceiling is, and what each
> measurement does and doesn't tell you. For the design intent behind
> the roofline work and the two benchmark bugs that faked a 40%
> codegen deficit, see
> [`.cursor/rules/findings.mdc`](../.cursor/rules/findings.mdc).

## The kernel

The headline benchmark is one line:

```mojo
def gaussian[T: FloatLike](x: T) -> T:
    return (-(x * x)).exp()
```

Run across an n-element `TileTensor` of `float32` via
`numax.tensor.map[step=gaussian_step]`. It moves 8 bytes per element
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
above ~1M; on an Apple M3 Pro it also beats the machine's own GPU path
until about 16M elements. One behavioral difference to know: the
threaded path flushes denormals to zero (MAX's worker-thread FP
environment, not a `numax` choice). See `bench/bench_elementwise.mojo`.

## GPU path

`map[gpu=True]` is the body of one GPU thread, launched via
`DeviceContext.enqueue_function` with one element per thread (the
default `width=1`, which is what measured fastest on Metal; thread
coarsening with `width>1` is supported but buys nothing on this
hardware, see `findings.mdc`). On an Apple M3 Pro at ~150 GB/s peak
bandwidth, the GPU path reaches **84% of peak** (126 GB/s) at 67M
elements.

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
scripts. The libraries that matter as comparison points:

- **PyTorch** (`torch.compile` on MPS, the same Metal device): at
  matched methodology, `numax`'s GPU path and `torch.compile`'s are at
  parity at 67M elements -- both pressed against the same roofline,
  within a few percent of each other, which given run-to-run spread is
  a tie rather than a ranking.
- **MLX** (Apple MLX, CPU and the same Metal GPU): passes `numax`'s GPU
  path at smaller sizes; `numax` crosses over at the largest tested.
- **thermite** (Rust NEON): the original Rust crate `numax`'s pattern
  was ported from. `numax`'s CPU path is ahead at every size, but the
  two implementations aren't running identical code paths.
- **NumPy**: the slowest CPU baseline, consistent with not doing the
  same kind of native SIMD-width dispatch the other three do.

The honest reading is in `bench/README.md`, not the headlines.

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

`numax/linalg.mojo`'s matrices are `Array[T, n*n]` in registers -- a
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
`max.linalg.qr_factorization` the way a LAPACK-style solver does
(`solve`/`det`/`inverse` all reduce to `R` and `Q^T b` once `A = QR`).
`cholesky`/`cholesky_solve`/`tridiagonal_solve` have no MAX equivalent to
route to at any size, since MAX ships neither a Cholesky factorization nor
a triangular solve. Every function's own docstring in `numax/linalg.mojo`
states which of these two cases it falls into.

## Bench tasks

```bash
pixi run bench          # CPU: numax.tensor.map vs. a hand-rolled raw-SIMD loop
pixi run bench-gpu     # CPU vs. GPU (map[gpu=True]) across a size sweep
pixi run bench-roofline # GPU: how much memory bandwidth map[gpu=True] reaches
pixi run bench-elementwise # CPU: serial vs. threaded at six sizes
pixi run bench-fusion   # CPU + GPU: composing inside step vs. chaining maps
pixi run bench-matmul   # CPU: numax.linalg.matmul vs. max.linalg.matmul
pixi run bench-numpy    # cross-language: NumPy, CPU
pixi run bench-mlx      # cross-language: Apple MLX, CPU + GPU
pixi run bench-torch    # cross-language: PyTorch (eager + compile), CPU + GPU
pixi run bench-thermite # cross-language: Rust thermite, CPU (NEON)
pixi run accuracy       # CPU: max error per function vs. checked-in mpmath refs
```

For methodology, how to run each, and the full results tables, see
[`bench/README.md`](../bench/README.md).
