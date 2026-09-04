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

<p align="center"><em>NumPy and SciPy's ground, in Mojo, on MAX. One kernel means several things, and runs on any device.</em></p>

## What νMAX is

A numerical computing library for [Mojo](https://mojolang.org): special
functions, linear algebra, quadrature, ODE solvers, FFTs, distributions, and a
NumPy-named array surface, built on [MAX](https://max.modular.com/docs/)'s
`TileTensor` and kernels.

```mojo
from numax import Dual, FloatLike, Plain, f32

def g[T: FloatLike](x: T) -> T:              # written once, no `dtype`
    return (-(x * x)).exp()

def main():
    print(g(Plain[f32](0.5)).v)              # 0.7788008  -- just the value
    var d = g(Dual[Plain[f32]].seed(0.5))    # derivative seeded to 1
    print(d.value.v, d.deriv.v)              # 0.7788008 -0.7788008
```

That is forward-mode automatic differentiation: $g(x)$ and
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
  signal and a NumPy-named array surface, plus `.npy` read/write, so a
  program ported from NumPy can ingest the files it already has and hand
  results back the same way. Full inventory in
  [`docs/features.md`](docs/features.md).
- **Fast, and measured, per processor.** On an A10G's GPU, 60,955 M elem/s
  against `torch.compile`'s 53,607, at ~83% of the card's bandwidth spec. On
  its CPU, 0.998x a hand-written raw-SIMD loop and 4.5x NumPy. CPU and GPU
  numbers are never mixed into one comparison; see
  [`docs/performance.md`](docs/performance.md).
- **Accurate on purpose.** Every approximation documents an error bound, and
  `pixi run accuracy` checks it against checked-in mpmath references at 50
  digits.

Young and experimental, so APIs may change.

## Install

```toml
[workspace]
channels = ["https://conda.modular.com/max", "conda-forge"]
preview = ["pixi-build"]

[dependencies]
numax = { git = "https://github.com/ehsanmok/numax.git", tag = "<latest-release>" }
```

Requires [pixi](https://pixi.sh); `mojo` and `max` come in transitively.

Inside a clone, rather than depending on the package, every `mojo`
invocation needs this directory on the import path: `pixi run mojo -I .
examples/basic/npy_interop.mojo`, or `pixi run run examples/basic/npy_interop.mojo`,
which supplies the `-I .` for you. The named tasks (`pixi run example-npy-interop`)
already do.

## Getting started

### Coming from NumPy and SciPy

Same programs, side by side. Shapes are compile-time parameters in the square
brackets, the `DeviceContext` is the last argument of every factory and
optional, and `f64` is `DType.float64` spelled short.

<table>
<tr><th width="50%">NumPy / SciPy</th><th width="50%">νMAX</th></tr>
<tr><td>

```python
import numpy as np

xs = np.linspace(0.0, 1.0, 5)
print(np.sqrt(xs))
print(xs.mean(), xs.std())
np.save("grid.npy", xs)
```

</td><td>

```mojo
from numax.prelude import *

var xs = linspace[5](0.0, 1.0)
print(sqrt(xs))
print(mean(xs), stddev(xs))
numpy.save(xs, "grid.npy")
```

</td></tr>

<tr><td>

```python
import numpy as np

A = np.eye(3)
b = np.ones(3)
print(np.linalg.solve(A, b))
print(np.linalg.det(A))
print(np.linalg.norm(A))
```

</td><td>

```mojo
comptime P = Plain[f64]

var A = to_array[P](eye[3]())
var b = Array[P, 3](fill=P.constant(1.0))
print(solve[P, 3](A, b)[0].v)
print(det[P, 3](A).v)
print(norm[P, 3](A).v)
```

</td></tr>

<tr><td>

```python
from scipy import special, integrate

f = lambda x: np.exp(-x * x)
print(integrate.fixed_quad(f, 0, 1, n=16)[0])
print(special.gamma(5.0), special.erf(1.0))

g = lambda t, y: -y
print(integrate.solve_ivp(g, [0, 0.1], [1.0]).y[0, -1])
```

</td><td>

```mojo
def f[U: FloatLike](x: U) -> U:
    return (-(x * x)).exp()

def g[U: FloatLike](t: U, y: U) -> U:
    return -y

print(gauss_legendre[P, f, 16](P.constant(0.0), P.one()).v)
print(gamma(P.constant(5.0)).v, erf(P.one()).v)
print(rk4[P, g](P.constant(0.0), P.one(), P.constant(0.1)).v)
```

</td></tr>

<tr><td>

```python
import numpy as np
from scipy import stats

xs = np.linspace(0.0, 1.0, 5)
print(xs.sum())
print(stats.norm.cdf(1.96))
print(np.sort(xs), np.argsort(xs))
```

</td><td>

```mojo
from numax.stats import norm, sum

var xs = linspace[5](0.0, 1.0)
print(sum(xs))
print(norm.cdf(P.constant(1.96), P.constant(0.0), P.one()).v)
print(sort(xs))
print(argsort(xs))
```

</td></tr>
</table>

One case has no left-hand column. SciPy differentiates by finite
difference; here the derivative falls out of the same kernel, exactly, because
the *type* carries it:

<table>
<tr><th width="50%">SciPy (approximate)</th><th width="50%">νMAX (exact)</th></tr>
<tr><td>

```python
from scipy.optimize import approx_fprime

f = lambda x: np.exp(-x * x)
print(f(0.5), approx_fprime([0.5], f)[0])
# 0.7788007830714049 -0.7788008010...
#                            ^ noise
```

</td><td>

```mojo
def f[U: FloatLike](x: U) -> U:
    return (-(x * x)).exp()

var d = f(Dual[P].seed(0.5))
print(d.value.v, d.deriv.v)
# 0.7788007830714049 -0.7788007830714049
```

</td></tr>
</table>

`f` was never written for derivatives. It is written against `FloatLike`, and
`Dual` is one of the types that satisfies it. So are `Gradient` (every
$\partial f/\partial x_i$ at once), `Compensated` (~double the precision),
`Complex`, `Interval` and `Decimal`, and they nest.

### Cheatsheet

| NumPy / SciPy | νMAX | Note |
|---|---|---|
| `np.zeros((2, 3))` | `zeros[f64, 2, 3]()` | shape is a compile-time parameter |
| `np.linspace(0, 1, 5)` | `linspace[5](0, 1)` | count first, `dtype` defaults to `f64` |
| `np.linspace(0, 1, 5, dtype=np.float32)` | `linspace[5, f32](0, 1)` | |
| `np.arange(5)`, `np.eye(3)` | `arange[5]()`, `eye[3]()` | `arange` takes a count, not a `stop` |
| `np.zeros_like(a)` | `zeros_like(a)` | derived shapes inherit `a`'s device |
| `a.reshape(2, 3)` | `reshape[f64, 6, 2, 3](a)` | source and target extents both named |
| `a + b`, `np.exp(a)`, `np.sort(a)` | `a + b`, `exp(a)`, `sort(a)` | |
| `a.astype(np.float32)` | `astype[f32](a)` | explicit: there is no dtype promotion |
| `a.sum()`, `a.mean()`, `np.var(a)` | `sum(a)`, `mean(a)`, `variance(a)` | `sum`/`min`/`max` are outside the prelude |
| `np.linalg.solve(A, b)` | `solve[P, n](A, b)` | over `Array[T, n*n]`, so it differentiates |
| `np.linalg.cholesky/qr/svd/eigh` | `cholesky`, `qr`, `svd`, `eigh` | |
| `scipy.special.gamma/erf/j0` | `gamma`, `erf`, `j0` | every one documents an error bound |
| `scipy.integrate.fixed_quad` | `gauss_legendre[T, f, n]` | fixed nodes, GPU-launchable |
| `scipy.integrate.quad` | `quad[f](a, b)` | adaptive, host-only, `Float64` bounds |
| `scipy.integrate.solve_ivp` | `solve_ivp`, or `rk4` for fixed steps | |
| `scipy.interpolate.CubicSpline` | `CubicSpline[T, n]` | built once, `__call__` evaluates |
| `scipy.optimize.brentq` / `minimize` | `brentq[f](a, b)` / `bfgs[n, f](x0)` | `bfgs` needs no `jac` |
| `scipy.optimize.approx_fprime` | evaluate at `Dual` / `Gradient` | exact, not a difference quotient |
| `np.fft.fft`, `np.fft.rfft` | `fft[T, log2n]`, `rfft` | length is `2^log2n`, over `Array[Complex[T], n]` |
| `np.save` / `np.load` | `numpy.save` / `numpy.load` | real `.npy`, readable by NumPy |
| `stats.norm.cdf(x)` | `norm.cdf(x, mu, sigma)` | nine distributions, parameters explicit |
| `np.random.default_rng(0)` | `Generator(seed=0)` | or `seed(0)` for the global stream |

Two names to know, because they have no NumPy counterpart: `Tensor[dtype, *dims]`
owns storage on a device and is what the array surface returns, while
`Array[T, n]` is a register-resident value generic over the `FloatLike`
conformer, and is what `linalg`, `signal` and `interpolate` take.
`to_array[T]` lifts, `to_tensor` lowers.

### One import

`numax.prelude` is the surface most programs want, in one line:

```mojo
from numax.prelude import *

def main() raises:
    var xs = linspace[5](0.0, 1.0)   # no DeviceContext to name
    print(sqrt(xs))                  # [0.0, 0.5, 0.7071, 0.866, 1.0]
    print(mean(xs), median(xs))      # 0.5 0.5
    numpy.save(xs, "grid.npy")       # numpy.load opens it
```

It deliberately leaves out the nine names that would shadow a Mojo
builtin (`sum`, `prod`, `min`, `max`, `abs`, `all`, `any`, `round`,
`copysign`), so a star import cannot break `min(1, 2)` in your own file.
Those stay one explicit import away (`from numax.stats import sum`). `from numax import ...` is the
full flat surface, and `from numax.linalg import ...` is one subsystem.

`f32`/`f64` are short for `DType.float32`/`.float64` and nothing else, so the
same name works wherever a dtype belongs, across both layers:
`linspace[5, f64](...)` and `Tensor[f64, 4, 4](ctx)` on the tensor side,
`Plain[f64]` and `Dual[Plain[f64]]` on the kernel side.

`Plain[dtype, width]` wraps a `SIMD[dtype, width]`. Mojo declares conformance
where a type is defined, so a bare `SIMD` cannot conform to `FloatLike`, and
`Plain` is what lets the hardware type participate. It is the baseline: every
kernel runs at `Plain` unless you ask for something else, and it compiles away.
`pixi run bench` measures `map` over a `Plain` kernel at **0.998x a
hand-written raw-SIMD loop**, `max |raw - numax| = 0.0`. `width` defaults to 1,
so `Plain[f64]` is the scalar. `.v` unwraps a `Plain`; `Dual` exposes
`.value`/`.deriv`.

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
comptime T = Tensor[f32, 1024]

var cpu = DeviceContext(api="cpu")
var xs = linspace[1024, f32](-2.0, 2.0, ctx=cpu)
var ys = T(cpu)
map[step=gaussian_step, width=8](xs.view(), ys.view())

var gpu = DeviceContext()
var gxs = linspace[1024, f32](-2.0, 2.0, ctx=gpu)
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
    var a = U.one() - v[0]
    var b = v[1] - v[0] * v[0]
    return a * a + U.constant(100.0) * b * b

var minimized = bfgs[2, rosenbrock](start)     # no `jac` argument
```

A central difference cannot beat about $\varepsilon^{2/3}$ relative accuracy:
truncation falls as $O(h^2)$ while cancellation grows as $O(\varepsilon/h)$.
AD has neither term. The example sweeps the step and prints both curves: best
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

Same kernel everywhere: $g(x) = e^{-x^2}$ over `float32`, 67M elements, wall
clock from dispatch through completion. M elem/s, higher is better. CPU and GPU
are measured and reported separately: every row below compares against
implementations running on the same processor.

**GPU, 67M elements, one sync per call:**

| Device | numax | torch.compile | torch eager | MLX |
|---|---|---|---|---|
| NVIDIA A10G (CUDA 13) | **60,955** | 53,607 | 19,814 | n/a |
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

- **The GPU work is bandwidth-bound, not compute-bound.** The best
  configuration on the A10G runs at **500.9 GB/s, ~83% of the card's 600 GB/s
  spec**, and an identity copy measures 489.10 GB/s against the Gaussian's
  489.16, so the `exp` is free.
- **Fusing two passes into one composed `step` is worth 1.99x** on the GPU at
  every size tested, and 1.26-1.58x on the CPU.
- **The serial CPU walk matches hand-written Rust SIMD** (1,394 vs.
  `thermite`'s 1,326 on the EPYC host) and is 4.5x NumPy. The `FloatLike`
  abstraction costs 0.998x a raw-SIMD loop.
- **The threaded CPU path is noisy.** Repeat runs on the EPYC host move by a
  factor of two at the same size, so read it as a range rather than a point.
- On Metal, numax and `torch.compile` are a tie within run-to-run spread; the
  ~10% A10G lead is that device's, not a general claim.

The GPU path is written against `DeviceContext` rather than a backend, so the
same source produces both device rows. Full sweeps from 64K to 67M, both sync
shapes, and the methodology: [`docs/performance.md`](docs/performance.md),
[`bench/README.md`](bench/README.md).

## Accuracy

Every approximation documents an error bound, and `pixi run accuracy` checks it
against checked-in [mpmath](https://mpmath.org/) references at 50 digits
(`erf`'s A&S 7.1.26 bound of ~1.5e-7 measures 1.38e-07). One caveat: Mojo's
`std.math` `exp`/`log`/`erf` are not correctly rounded at `float64`, so every
function built on them inherits that floor, which is invisible at `float32`.
Details: [`bench/accuracy/README.md`](bench/accuracy/README.md).

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

- [`docs/features.md`](docs/features.md): the complete public surface, by
  subpackage.
- [`examples/README.md`](examples/README.md): every example, one line each.
- [`docs/architecture.md`](docs/architecture.md): the trait, the
  fixed-iteration invariant, the tensor/GPU layer, the package layout.
- [`docs/parity.md`](docs/parity.md): what numax absorbs, routes to MAX, or
  leaves out.
- [`docs/performance.md`](docs/performance.md): the full performance writeup.
- **API reference**: <https://ehsanmok.github.io/numax/>, built on every push
  to `main`. Locally: `pixi run -e dev docs`.

## License

The source in this repository is licensed under
[Apache 2.0 (with LLVM exceptions)](LICENSE). MAX is distributed as prebuilt
binaries and container images, licensed separately under the
[Modular Community License](https://www.modular.com/legal/community). The
applicable license is determined by the artifact you are using, not by how you
obtained it.
