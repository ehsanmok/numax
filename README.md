<p align="center">
  <img src="logo.png" alt="numax" width="480" height="240">
</p>

<p align="center">
  <a href="https://github.com/ehsanmok/numax/actions/workflows/ci.yml"><img src="https://github.com/ehsanmok/numax/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/ehsanmok/numax/actions/workflows/docs.yaml"><img src="https://github.com/ehsanmok/numax/actions/workflows/docs.yaml/badge.svg" alt="Docs"></a>
  <a href="https://mojolang.org"><img src="https://img.shields.io/badge/Mojo-1.0.0-orange" alt="Mojo 1.0.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code-Apache%202.0%20with%20LLVM%20Exceptions-yellow.svg" alt="Code: Apache 2.0 with LLVM Exceptions"></a>
  <a href="https://www.modular.com/legal/community"><img src="https://img.shields.io/badge/MAX%20binaries-Modular%20Community%20License-blue.svg" alt="MAX binaries: Modular Community License"></a>
</p>

<p align="center"><em>NumPy and SciPy's ground, in Mojo, on MAX — where one kernel means several things and runs on any device.</em></p>

## What νMAX is

A numerical computing library for [Mojo](https://mojolang.org): special
functions, linear algebra, quadrature, ODE solvers, FFTs, distributions, and a
NumPy-named array surface, built on [MAX](https://max.modular.com/docs/)'s
`TileTensor` and kernels.

```mojo
from numax import Dual, FloatLike, f32

def g[T: FloatLike](x: T) -> T:              # written once, no `dtype`
    return (-(x * x)).exp()

def main():
    print(g(f32(0.5)).v)                     # 0.7788008  -- just the value
    var d = g(Dual[f32].seed(0.5))           # derivative seeded to 1
    print(d.value.v, d.deriv.v)              # 0.7788008 -0.7788008
```

That is forward-mode automatic differentiation — $g(x)$ and
$g'(x) = -2xe^{-x^2}$ from one call, by the chain rule built into `Dual`'s
arithmetic. It works on every function in the library.

## Why νMAX

- **One kernel, several meanings.** Every function is written once against the
  `FloatLike` trait. The type you call it with decides what comes back: a value
  (`Plain`), a derivative (`Dual`), a full gradient (`Gradient`), extra
  precision (`Compensated`), exact base-10 fixed point (`Decimal`), a complex
  result (`Complex`), an interval bound (`Interval`). They nest, so autodiff,
  precision and complex arithmetic compose instead of each needing its own copy
  of every kernel.
- **One tensor, every device.** `Tensor` owns a MAX `DeviceBuffer`, so the
  `DeviceContext` you pass a factory decides host or device memory. Nothing else
  changes, and `.view()` yields the `TileTensor` every MAX kernel takes.
- **NumPy and SciPy's ground.** Special functions, dense linear algebra,
  quadrature, ODE solvers, FFTs, distributions, statistics, interpolation,
  signal and a NumPy-named array surface — plus `.npy` read/write, so a
  program ported from NumPy can ingest the files it already has and hand
  results back the same way. Full inventory in
  [`docs/features.md`](docs/features.md).
- **Fast, and measured, per processor.** On an A10G's GPU, 60,955 M elem/s
  against `torch.compile`'s 53,607, at ~83% of the card's bandwidth spec. On
  its CPU, 0.998x a hand-written raw-SIMD loop and 4.5x NumPy. CPU and GPU
  numbers are never mixed into one comparison — see
  [`docs/performance.md`](docs/performance.md).
- **Accurate on purpose.** Every approximation documents an error bound, and
  `pixi run accuracy` checks it against checked-in mpmath references at 50
  digits.

Young and experimental — APIs may change.

## Install

```toml
[workspace]
channels = ["https://conda.modular.com/max", "conda-forge"]
preview = ["pixi-build"]

[dependencies]
numax = { git = "https://github.com/ehsanmok/numax.git", tag = "<latest-release>" }
```

Requires [pixi](https://pixi.sh); `mojo` and `max` come in transitively.

Working inside a clone rather than depending on the package? Every `mojo`
invocation needs this directory on the import path: `pixi run mojo -I .
examples/basic/npy_interop.mojo`, or `pixi run run examples/basic/npy_interop.mojo`,
which supplies the `-I .` for you. The named tasks (`pixi run example-npy-interop`)
already do.

## Getting started

### One import

`numax.prelude` is the surface most programs want, in one line:

```mojo
from numax.prelude import *

def main() raises:
    var xs = linspace[DType.float64, 5](0.0, 1.0)   # no DeviceContext to name
    print(sqrt(xs))                                 # [0.0, 0.5, 0.7071, 0.866, 1.0]
    print(mean(xs), median(xs))                     # 0.5 0.5
    numpy.save(xs, "grid.npy")                      # numpy.load opens it
```

It deliberately leaves out the handful of names that would shadow a Mojo
builtin — `sum`, `min`, `max`, `abs`, `all`, `any`, `round` — so a star
import cannot break `min(1, 2)` in your own file. Those stay one explicit
import away (`from numax.stats import sum`). `from numax import ...` is the
full flat surface, and `from numax.linalg import ...` is one subsystem.

About `Plain`, and whether it costs anything: `Plain[dtype, width]` wraps a `SIMD[dtype, width]`: Mojo declares conformance
where a type is defined, so a bare `SIMD` cannot conform to `FloatLike` and
`Plain` is what lets the hardware type participate. It is the baseline, not a
niche — every kernel runs at `Plain` unless you ask for something else, and it
compiles away: `pixi run bench` measures `map` over a `Plain` kernel at
**0.998x a hand-written raw-SIMD loop**, `max |raw - numax| = 0.0`. `.v`
unwraps a `Plain`; `Dual` exposes `.value`/`.deriv`.

> **Run it:** `pixi run example-gaussian` ·
> [`examples/basic/gaussian.mojo`](examples/basic/gaussian.mojo)

### Precision, for free

Sum a million nearly-equal `float32` values and the running total stops seeing
the next one. Swap the type, not the algorithm:

```mojo
var plain_var = variance(plain_list).v         # float32 accumulation
var comp_var = variance(comp_list).value       # ~double precision, same code
```

`Compensated` carries the rounding error ordinary arithmetic discards.

> **Run it:** `pixi run example-statistics` ·
> [`examples/intermediate/statistics.mojo`](examples/intermediate/statistics.mojo)

### The same kernel, on the GPU

Only the context and the walk differ:

```mojo
comptime T = Tensor[DType.float32, 1024]

var cpu = DeviceContext(api="cpu")
var xs = linspace[DType.float32, 1024](cpu, -2.0, 2.0)
var ys = T(cpu)
map[step=gaussian_step, width=8](xs.view(), ys.view())

var gpu = DeviceContext()
var gxs = linspace[DType.float32, 1024](gpu, -2.0, 2.0)
var gys = T(gpu)
gpu.enqueue_function[map[LayoutType = T.LayoutType, step=gaussian_step, gpu=True]](
    gxs.view(), gys.view(), grid_dim=4, block_dim=256
)
```

Every conformer is built from plain `SIMD` fields with no pointers, so `Dual`
and `Compensated` kernels launch on a GPU thread unmodified too.

> **Run it:** `pixi run example-unified-tensor-gpu` (needs a GPU) ·
> [`examples/advanced/unified_tensor_gpu.mojo`](examples/advanced/unified_tensor_gpu.mojo)

### Exact gradients into an optimizer

The objective is an ordinary `FloatLike` kernel, so BFGS evaluates it at
`Gradient` and gets every $\partial f / \partial x_i$ *exactly*:

```mojo
def rosenbrock[U: FloatLike](v: Array[U, 2]) -> U:
    var a = U.one() + (-v[0])
    var b = v[1] + (-(v[0] * v[0]))
    return a * a + U.constant(100.0) * b * b

var minimized = bfgs[2, rosenbrock](start)     # no `jac` argument
```

A central difference cannot beat about $\varepsilon^{2/3}$ relative accuracy:
truncation falls as $O(h^2)$ while cancellation grows as $O(\varepsilon/h)$.
AD has neither term — the example sweeps the step and prints both curves, best
finite difference ~5e-10 against AD at exactly 0.

> **Run it:** `pixi run example-optimize` ·
> [`examples/advanced/optimize.mojo`](examples/advanced/optimize.mojo)

### A Hessian, by nesting types

```mojo
comptime G = Gradient[Dual[P], 2]      # gradient of a dual number
```

`Dual` inside itself is a second derivative; `Gradient` over `Dual` is the
full Hessian $H_{ij} = \partial^2 f / \partial x_i \partial x_j$ and
Hessian-vector products. Neither type contains second-order mathematics.

> **Run it:** `pixi run example-hessian` ·
> [`examples/basic/hessian.mojo`](examples/basic/hessian.mojo)

### Keep pulling the thread

| You want | Call it at | Example |
|---|---|---|
| Holomorphic derivatives | `Complex[Dual[Plain]]` | `pixi run example-complex` |
| Guaranteed bounds | `Interval[Plain]` | [`numax/core/interval.mojo`](numax/core/interval.mojo) |
| Exact decimals (`0.1 + 0.2 == 0.3`) | `Decimal[width, scale]` | [`numax/core/decimal.mojo`](numax/core/decimal.mojo) |
| Every special function differentiated | `Dual` | `pixi run example-special-functions` |
| A 2-D wave packet, differentiated by its own width | `Dual` | `pixi run example-wave-packet` |
| Interference fringes, measured, and how they move with the geometry | `Dual` | `pixi run example-interference` |
| 1024 ODE trajectories, one GPU thread each | `Dual` for sensitivities | `pixi run example-ode` |

Full index: [`examples/README.md`](examples/README.md). `pixi run examples-cpu`
skips the GPU ones; `pixi run examples` includes them. All seven conformers,
with what each one returns and where it lives:
[`docs/features.md`](docs/features.md#the-trait-and-its-conformers).

## Performance

Same kernel everywhere — $g(x) = e^{-x^2}$ over `float32`, 67M elements, wall
clock from dispatch through completion. M elem/s, higher is better. CPU and GPU
are measured and reported separately: every row below compares against
implementations running on the same processor.

**GPU, 67M elements, one sync per call:**

| Device | numax | torch.compile | torch eager | MLX |
|---|---|---|---|---|
| NVIDIA A10G (CUDA 13) | **60,955** | 53,607 | 19,814 | — |
| Apple M3 Pro (Metal) | **14,465** | 13,831 | 4,807 | 4,866 |

Amortizing the sync over ten launches instead: 61,537 vs. `torch.compile`'s
56,141 on the A10G, 15,755 vs. 14,380 on the M3 Pro. MLX is macOS-only, hence
the gap in the first row.

**CPU, 67M elements:**

| Host | numax `map_threaded` | numax `map` (1 thread) | torch.compile | torch eager | Rust `thermite` | NumPy |
|---|---|---|---|---|---|---|
| AMD EPYC (the A10G's host) | **8,527** | 1,394 | 1,833 | 401 | 1,326 (AVX2) | 313 |
| Apple M3 Pro | **9,026** | 2,347 | 4,046 | 1,935 | 1,632 (NEON) | 493 |

MLX's CPU path measures 1,954 on the M3 Pro (it has no CUDA build, so no EPYC
row).

Reading these:

- **The GPU work is bandwidth-bound, not compute-bound.** The best
  configuration on the A10G runs at **500.9 GB/s — ~83% of the card's 600 GB/s
  spec** — and an identity copy measures 489.10 GB/s against the Gaussian's
  489.16, so the `exp` is free.
- **Fusing two passes into one composed `step` is worth 1.99x** on the GPU at
  every size tested, and 1.26-1.58x on the CPU.
- **The serial CPU walk matches hand-written Rust SIMD** (1,394 vs.
  `thermite`'s 1,326 on the EPYC host) and is 4.5x NumPy — the `FloatLike`
  abstraction costs 0.998x a raw-SIMD loop, not a factor.
- **The threaded CPU path is noisy** — repeat runs on the EPYC host move by a
  factor of two at the same size. Read it as a range, not a point.
- On Metal, numax and `torch.compile` are a tie within run-to-run spread; the
  ~10% A10G lead is that device's, not a general claim.

The GPU path is written against `DeviceContext`, not a backend — the same
source produces both device rows. Full sweeps from 64K to 67M, both sync
shapes, and the methodology: [`docs/performance.md`](docs/performance.md),
[`bench/README.md`](bench/README.md).

## Accuracy

Every approximation documents an error bound, and `pixi run accuracy` checks it
against checked-in [mpmath](https://mpmath.org/) references at 50 digits
(`erf`'s A&S 7.1.26 bound of ~1.5e-7 measures 1.38e-07). One caveat: Mojo's
`std.math` `exp`/`log`/`erf` are not correctly rounded at `float64`, so every
function built on them inherits that floor — invisible at `float32`. Details:
[`bench/accuracy/README.md`](bench/accuracy/README.md).

## Testing

```bash
pixi run tests           # 42 suites, 638 tests
pixi run examples-cpu    # every example that does not need a GPU
pixi run bench           # map vs. a hand-rolled raw-SIMD loop
pixi run accuracy        # max error per function vs. mpmath references
```

`tests` and `examples-cpu` run in CI on macOS and Linux; GPU examples and
benchmarks are a local check.

## Documentation

- [`docs/features.md`](docs/features.md) — the complete public surface, by
  subpackage.
- [`examples/README.md`](examples/README.md) — every example, one line each.
- [`docs/architecture.md`](docs/architecture.md) — the trait, the
  fixed-iteration invariant, the tensor/GPU layer, the package layout.
- [`docs/parity.md`](docs/parity.md) — what numax absorbs, routes to MAX, or
  leaves out.
- [`docs/performance.md`](docs/performance.md) — the full performance writeup.
- **API reference** — <https://ehsanmok.github.io/numax/>, built on every push
  to `main`. Locally: `pixi run -e dev docs`.

## License

The source in this repository is licensed under
[Apache 2.0 (with LLVM exceptions)](LICENSE). MAX is distributed as prebuilt
binaries and container images, licensed separately under the
[Modular Community License](https://www.modular.com/legal/community). The
applicable license is determined by the artifact you are using, not by how you
obtained it.
