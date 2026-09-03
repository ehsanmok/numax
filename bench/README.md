# Benchmarks

Two kinds of benchmark live here:

- `bench_tensor_map.mojo` / `bench_tensor_map_gpu.mojo` (`pixi run bench`,
  `pixi run bench-gpu`) -- `numax` measured against itself: does its
  trait-and-`TileTensor` layer cost anything over a hand-rolled raw-`SIMD`
  loop (CPU), and where does its GPU path (`map[gpu=True]`) actually earn
  back its own dispatch overhead over the CPU path. See the top-level
  `README.md`'s Benchmarks section for those two.
- `bench_fusion.mojo` (`pixi run bench-fusion`) -- two chained `map` calls
  against one `map` with a composed `step`, on CPU and GPU. See the fusion
  section below.
- `bench_elementwise.mojo` (`pixi run bench-elementwise`) -- `numax`'s
  serial `map` against its threaded `map_threaded`, which is
  `max.algorithm.elementwise` underneath. See the threading section below.
- `bench_matmul.mojo` (`pixi run bench-matmul`) -- `numax.linalg.matmul`
  against `max.linalg.matmul`, to find the size where MAX's blocked,
  threaded kernel overtakes the generic triple loop. See the matmul
  section below.
- `bench_gpu_roofline.mojo` (`pixi run bench-roofline`) -- not a comparison
  at all, a diagnosis: how much of the device's memory bandwidth
  `map[gpu=True]` actually reaches, whether the ceiling is the memory path
  or `std.math.exp` (an identity-copy kernel separates the two), and what
  per-thread `width` and `block_dim` are worth. This is what established
  that the numbers below are bandwidth-bound rather than compute-bound, and
  what caught both measurement bugs described under Results.
- `numpy/`, `mlx/`, `torch/`, `thermite/` (this page) -- `numax` measured
  against libraries someone would actually reach for today, on the same
  machine, for the same kernel.

Every baseline here is a real dependency of its own pixi feature/environment
(`bench-python` for NumPy/MLX/PyTorch, `bench-rust` for the Rust toolchain
`thermite` needs) — `pixi run <task>` installs whatever that baseline needs
and runs it, the same as any other task in this repo. Nothing here needs a
manually-managed virtualenv or a system-wide `cargo`.

## The kernel and methodology

Every benchmark runs the same thing:

```
gaussian(x) = exp(-x^2)
```

over `float32` arrays of `n` elements, at six sizes (`2^16` through `2^26`,
64K to ~67M elements), input `x[i] = i * 0.0001 - 50.0`. Each measurement is
3 warmup calls followed by 10 timed calls, wall-clock time around the whole
operation (dispatch/launch through completion, not an isolated kernel-only
time) — what a caller actually experiences, not a best case. Every script
also checks its own output against an `f64` reference and warns if the max
error exceeds `1e-5`; none did on the run below.

For the GPU baselines there is one more methodological choice, and it turns
out to matter more than anything else on this page: whether the device
synchronize happens inside each timed iteration or once after all ten. Every
GPU baseline reports both — see Results.

This keeps the comparison to one narrow, well-defined kernel on purpose —
it's a statement about elementwise-`exp` throughput on this machine, not a
general verdict on any of these libraries.

## The baselines, and why these four

- **NumPy** (`numpy/gaussian.py`, `pixi run bench-numpy`) — the default
  reference point for anyone doing numerical work in Python. CPU-only;
  there's nothing to compare against `numax`'s GPU path here, which is what
  MLX and PyTorch are for.
- **MLX** (`mlx/gaussian.py`, `pixi run bench-mlx`) — the closest match to
  what `numax`'s own GPU path is doing: one array library, one API, a
  `stream=mx.cpu`/`stream=mx.gpu` switch, running on the same GPU `numax`
  targets via `DeviceContext`. Metal on macOS; on Linux it needs an
  explicit CUDA backend wheel, which `pixi.toml` selects per platform.
- **PyTorch** (`torch/gaussian.py`, `pixi run bench-torch`) — eager and
  `torch.compile`'d, on CPU and on whichever GPU torch can see (CUDA, else
  MPS — the same device as above). `torch.compile`
  traces and JIT-compiles the kernel into a fused device program, which is
  worth a lot over *eager* PyTorch on this kernel (roughly 3x on MPS,
  consistent with three elementwise dispatches becoming one) and roughly
  nothing over `numax`, which was already one fused kernel — see Results.
- **Rust `thermite` 0.2** (`thermite/`, `pixi run bench-thermite`) — not an
  arbitrary Rust SIMD crate: it's the library `numax`'s composable-numeric-type
  pattern was originally ported from (see the top-level README's
  introduction). Its NEON backend is "Complete. Mandatory on the
  architecture" for `aarch64`, so this runs genuinely vectorized code on
  this machine, not a scalar fallback.
- **`numax` itself**, CPU (`map[gpu=False]`) and GPU (`map[gpu=True]`), via
  `bench_tensor_map_gpu.mojo` (`pixi run bench-gpu`), plus its threaded CPU
  walk (`map_threaded`) from `bench_elementwise.mojo`.

## How to run each

```bash
pixi run bench-gpu       # numax, CPU + GPU sweep (both sync shapes)
pixi run bench-roofline  # numax, bandwidth diagnosis (needs a GPU)
pixi run bench-numpy     # NumPy, CPU
pixi run bench-mlx       # MLX, CPU + GPU (macOS only)
pixi run bench-torch     # PyTorch, eager + compile, CPU + CUDA/MPS
pixi run bench-thermite  # Rust thermite, CPU (NEON or AVX2)
```

Each `pixi run bench-*` task above resolves and installs whatever that
baseline needs the first time it's run (a Python interpreter plus
NumPy/MLX/PyTorch for the first three, a Rust toolchain for the last) into
its own pixi environment, cached in `.pixi/` like every other environment
in this repo -- there's no separate setup step. `bench-mlx` is declared only
on the macOS target, so it does not appear at all on Linux: MLX's Linux
backend is an opt-in wheel whose name pins a CUDA major version, and a pin
that is only correct on one machine is worse than no baseline.

## Results

Measured on an Apple M3 Pro (12 CPU cores, 36GB unified memory, ~150 GB/s
memory bandwidth), macOS 25.5.0, `mojo`/`max` per `pixi.toml`, NumPy 2.5.2,
MLX (Metal GPU device), PyTorch 2.13.0, `thermite` 0.2.1 (NEON backend). All
throughput in millions of elements/sec — higher is better.

These are one machine's numbers. Every benchmark here also runs on
Linux/x86_64 with a CUDA GPU (`thermite` picks its AVX2 backend, PyTorch its
CUDA one; MLX is macOS-only); the figures differ, the conclusions below about
*where the time goes* are the ones worth carrying across hardware, and
where they have been re-checked on CUDA that is said explicitly.

### Read the roofline first

This kernel moves 8 bytes per element (one float32 read, one write) no
matter how it is written, so **at large `n` every library here is competing
for the same memory bandwidth, not for arithmetic**. That puts a hard
ceiling on the whole table: 150 GB/s ÷ 8 bytes = **18,750 M elem/s**.
Anything within a few percent of that is at the hardware limit, and a
"1.1x win" up there is a different claim from a 1.1x win at 64K.

Confirmed rather than assumed: `bench_gpu_roofline.mojo` runs an identity
copy (no arithmetic at all) through the same walk and gets the same
throughput as `gaussian` does — 118 vs 120 GB/s at 67M. The `exp` is free;
the memory traffic is everything.

### Two GPU columns, and why mixing them was wrong

An earlier version of this file claimed "every GPU number includes the same
host round-trip (dispatch + synchronize)". That was **false**, and it
flattered PyTorch by roughly 1.6x at the top end: `numax`'s benchmark
synchronized inside each timed iteration while `torch/gaussian.py`
synchronized once after all ten, so one was measuring per-call latency and
the other pipelined throughput. Those are both legitimate questions, so
every GPU baseline now reports both:

- **per-call** — synchronize inside each iteration. One full
  launch-through-completion round trip, i.e. what a caller pays for a
  single call.
- **amortized** — enqueue all iterations, synchronize once. Dispatches
  pipeline, so this is steady-state throughput for back-to-back work.

A second, separate measurement bug was found at the same time and fixed in
`bench_tensor_map_gpu.mojo`: it accumulated the CPU and GPU timings *inside
one loop*, so a ~28 ms bandwidth-saturating CPU pass ran between every timed
GPU launch, on the same unified memory. Splitting them into separate loops
(and warming each measurement) moved the reported 67M GPU figure from ~9,400
to ~15,700 M elem/s **without changing a line of `numax`**. The old
`numax` GPU column was measuring interference.

**CPU-only:**

`numax` gets two columns because it has two CPU walks: serial `map`
(`map[gpu=False]`, one thread, SIMD-vectorized) and `map_threaded`
(`max.algorithm.elementwise`, multi-threaded). The serial one is what this
table used to report, which undersold it by 3-5x at large `n`.

| n | numax `map` | numax `map_threaded` | thermite (NEON) | NumPy | MLX | torch eager | torch compile |
|---|---|---|---|---|---|---|---|
| 65,536 | 2,348 | 3,744 | 1,706 | 680 | 249 | 1,194 | 576 |
| 262,144 | 2,340 | 2,776 | 1,298 | 660 | 811 | 1,737 | 2,214 |
| 1,048,576 | 2,355 | 11,560 | 1,311 | 511 | 1,586 | 2,165 | 3,544 |
| 4,194,304 | 2,349 | 10,099 | 1,508 | 626 | 1,647 | 2,221 | 4,136 |
| 16,777,216 | 2,343 | 8,736 | 1,627 | 508 | 1,893 | 1,989 | 4,015 |
| 67,108,864 | 2,347 | 9,026 | 1,632 | 493 | 1,954 | 1,935 | 4,046 |

`map_threaded` is the median of three runs and is genuinely noisy —
individual runs at 1M ranged 7,955-12,307. Serial `map` varies by under 2%.

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

\* MLX's amortized column is not comparable at large `n`, for a reason
specific to MLX rather than to MLX's speed: MLX is lazy, so an unreferenced
graph node is never evaluated, and forcing only the last of ten would time
one kernel instead of ten. The benchmark therefore has to hold all ten
outputs live — ten 268MB buffers at 67M — and that memory pressure makes its
amortized number *worse* than its per-call one. Read MLX's per-call column.

Reading this honestly, not as an `numax`-wins scoreboard:

- **At matched methodology, `numax` and `torch.compile` on MPS are at
  parity on the GPU, both pinned against the bandwidth roofline.** At 67M
  amortized, `numax` reaches 15,755 M elem/s (126 GB/s, **84% of peak**)
  and `torch.compile` 14,380 (115 GB/s, 77%); per-call the two are 14,465
  and 13,831. They trade wins by a few percent in both directions across
  the sweep, which given the run-to-run spread is better described as a tie
  than as a ranking. The earlier "`torch.compile` is fastest by a wide
  margin" conclusion did not survive fixing the two measurement bugs above.
- **The kernel-fusion theory for that old gap was wrong, and is worth
  retracting explicitly.** It was inferred from the shape of the numbers:
  Inductor fusing `exp(-(x*x))` into one Metal kernel instead of three
  dispatches. But `numax`'s `gaussian_step` was *already* one fused kernel,
  so fusion could never have explained an `numax`-to-compile gap. It does
  plausibly explain eager-to-compile (4,809 vs 14,380 amortized at 67M is
  close to a 3-passes-to-1 ratio), which is where the inference belongs.
- **`numax`'s threaded CPU walk is the fastest CPU number here above ~1M**
  (8,700-11,600 vs `torch.compile`'s 3,500-4,100), and it beats this
  machine's own GPU path per-call until about 4M elements. For a single
  elementwise pass on an M3 Pro, threading the CPU is the better first move
  and the GPU only pays off at scale.
- **Serial `numax` CPU is flat across every size** (~2,350 M elem/s), which
  says more about Mojo's NEON codegen and `std.math.exp` than anything
  `numax`-specific — `pixi run bench` already showed `numax`'s layer adds
  ~0.2% over the same raw loop.
- **`numax` CPU outperforms `thermite`** (roughly 1.4-1.8x serial, more when
  threaded), which is worth being skeptical of rather than triumphant about:
  the two aren't running identical code paths (`thermite`'s run is strictly
  in-place, `dispatch_dyn!`'s runtime ISA check happens on every timed call,
  and `std.math.exp` vs. `thermite`'s `TranscendentalMath::exp` may not be
  the same approximation or accuracy trade). Attributing the gap would need
  a same-ISA, same-approximation microbenchmark of `exp` alone.
- **GPU dispatch overhead dominates below ~1M for everyone.** Every library
  here is under 3,400 M elem/s at 1M per-call and under 1,300 at 64K — a
  fixed cost of a few hundred microseconds that no amount of kernel quality
  recovers. `numax`'s GPU path only overtakes its own serial CPU path above
  ~1M, and its threaded CPU path above ~4M.
- **NumPy is the slowest CPU number throughout**, consistent with it not
  doing the same kind of native SIMD-width dispatch the other four do.

Rerun all five on a different machine before trusting these numbers to
generalize — this is one data point, on one chip, for one kernel.

Run-to-run spread matters for reading the GPU tables specifically, since the
parity claim above rests on differences of a few percent. Repeat runs of
`numax` at 67M moved between 14,081-14,465 per-call and 15,074-15,755
amortized, i.e. ±3%, which is the same order as its margin over
`torch.compile` — hence "tie" rather than "wins". Below 1M the per-call
numbers are far worse: 64K moved by up to 2x between runs on the same
machine, so treat that row as an order of magnitude, not a measurement.

## Fusion: chaining two `map`s vs composing inside `step`

`pixi run bench-fusion`, same machine. Both sides compute
`2.5 * exp(-x^2)`: **chained** runs `map(gaussian)` then `map(scale)` (two
passes, two dispatches), **fused** runs one `map` whose `step` is
`scale(gaussian(x))`. Millions of elements/sec.

CPU, via `map_threaded`:

| n | chained | fused | speedup |
|---|---|---|---|
| 1,048,576 | 3,865 | 11,335 | 2.93x |
| 4,194,304 | 4,205 | 10,968 | 2.61x |
| 16,777,216 | 3,979 | 8,170 | 2.05x |
| 67,108,864 | 4,180 | 7,886 | 1.89x |

GPU, via `map[gpu=True]`:

| n | chained | fused | speedup |
|---|---|---|---|
| 1,048,576 | 1,941 | 5,533 | 2.85x |
| 4,194,304 | 4,035 | 12,253 | 3.04x |
| 16,777,216 | 7,239 | 13,936 | 1.92x |
| 67,108,864 | 7,306 | 13,973 | 1.91x |

Identical results (`max |diff| = 0.0` throughout), 2-3x apart in speed. So
"compose inside `step`, don't chain `map` calls" is not a style preference,
it's the difference between one pass and two — and it costs nothing to
follow, because composing `FloatLike` values is just writing the
expression.

Two cautions on reading the GPU column. First, this benchmark enqueues all
timed iterations and synchronizes once at the end, so per-call host
round-trip is amortized; the cross-language table above pays a synchronize
per call, and the two sets of numbers are therefore not comparable. Second,
this is the *most* favourable possible fusion case — two cheap elementwise
ops where the second does almost no arithmetic, so the chained version is
nearly pure extra memory traffic.

What it does support is the structural claim: `numax` needs no fusion pass
to get this, because a `FloatLike` kernel composes before any tensor walk
begins. What it does not support is a claim of parity with Inductor, which
fuses code the user wrote as separate operations — `numax` requires the
caller to write the composition. `max.algorithm.dual_elementwise` was
evaluated as a way to fuse two *independent* elementwise ops into a single
launch, and is unusable here: unlike `elementwise`, it has only the
compile-time-parameter form, so passing a closure fails with "failed to
infer parameter `__origins__`", and a non-capturing function has no way to
reach the tensors (its signature passes only a coordinate). It also
wouldn't address this case, since it fuses independent ops rather than
chained ones.

## Threading: `map` vs `map_threaded`

`pixi run bench-elementwise`, same machine and same `gaussian` kernel.
`numax.core.tensor.map` walks the tensor on one thread at native SIMD width;
`numax.core.tensor.map_threaded` hands the same `step` to
`max.algorithm.elementwise[target="cpu"]`, which vectorizes *and*
distributes across cores. Millions of elements/sec, higher is better.

| n | map | map_threaded | speedup |
|---|---|---|---|
| 65,536 | 2,199 | 1,424 | 0.65x |
| 262,144 | 2,267 | 3,037 | 1.34x |
| 1,048,576 | 2,342 | 12,206 | 5.21x |
| 4,194,304 | 2,323 | 10,538 | 4.54x |
| 16,777,216 | 2,334 | 8,818 | 3.78x |
| 67,108,864 | 2,312 | 7,804 | 3.38x |

The threaded path costs more than it saves below about 250K elements, then
wins by 3-5x. Two things stand out beyond the obvious:

- **Threaded CPU beats this machine's GPU until about 16M elements.**
  Against the GPU table above, `map_threaded` at 1M elements (12,206) is
  ~5.6x `numax`'s own Metal path at the same size (2,173), and stays ahead
  through 4M. Only at 16M+ does the GPU pull in front. For a single
  elementwise pass on an M3 Pro, threading the CPU is the better first
  move — the GPU's advantage needs a lot of elements to pay off its
  dispatch.
- **The threaded path flushes denormals to zero.** Results below the
  smallest normal `float32` (~1.2e-38) come back as exactly `0` where the
  serial walk returns the denormal — the `max |diff|` column is exactly
  that threshold at every size where the input sweep reaches it. Verified
  directly: at `x = -9.3639`, serial gives `8.32e-39` and threaded gives
  `0.0`. Everything at or above the threshold is bit-identical. This is
  MAX's worker-thread FP environment, not anything `numax` chooses, and
  it's documented on `map_threaded` rather than papered over.

`map_threaded` is a separate function rather than a `threads: Bool`
parameter on `map`, because `elementwise` needs a `DeviceContext` and
`map`'s signature is shared with the GPU kernel body that
`enqueue_function` launches — an argument that cannot cross that boundary.
Same reasoning that keeps `reduce_block_gpu` separate from `reduce`.

## Matmul: where MAX overtakes the generic loop

`pixi run bench-matmul`, same machine. `max.linalg.matmul` (blocked,
vectorized, threaded, `TileTensor` of raw `float32`) against
`numax.linalg.matmul` (naive triple loop over `Array[T, n*n]`, generic in
`T: FloatLike`). Nanoseconds per `n x n` product, lower is better. The
batched column runs `Plain[dtype, 4]`, so one call does four independent
products — one per SIMD lane — and the time is divided by four to stay
comparable.

| n | MAX | numax scalar | numax batched (per matrix) |
|---|---|---|---|
| 4 | 102 | 57 | 15 |
| 8 | 143 | 323 | 74 |
| 16 | 366 | 1,870 | 492 |
| 32 | 380 | 14,335 | 3,690 |
| 64 | 900 | 117,440 | 30,170 |

Both implementations agreed to the last bit at every size (the benchmark
checks this, and prints `max |diff| = 0.0`), so this is purely a speed
comparison of two correct routines.

The crossover is early and sharp: MAX wins from `n = 8` on for a single
matrix, from `n = 16` on even against the 4-wide batched version, and by
`n = 64` it is ~130x ahead of the scalar loop. That is the expected shape —
blocking and threading are worth exactly nothing at `n = 4`, where MAX's
per-call overhead (~100ns) is most of its time, and worth two orders of
magnitude by `n = 64`.

So `numax.linalg.matmul` was **not** replaced by a call into MAX, and the
reason isn't performance:

- The two take different arguments and cannot substitute for each other.
  MAX needs a `TileTensor` backed by device or host memory; `numax`'s
  matrix is an `Array[T, n*n]` living in registers inside a per-element
  kernel, which is what makes it callable from inside `map[gpu=True]` at
  all. Adapting one to the other means materializing a buffer per matrix,
  which costs more than the multiply at these sizes.
- MAX's is monomorphic in `dtype`, so it cannot carry a `Dual`, a
  `Compensated`, or a `Complex`. Differentiating a matmul is the entire
  reason `numax`'s exists.

The practical guidance, now in `numax/linalg/linalg.mojo`'s docstring as well: if
your entries are plain `dtype` and your matrix is bigger than about 8x8,
call `max.linalg.matmul` directly — `numax` deliberately does not wrap it,
since a pass-through adding no capability would just be a second name for
the same function.
