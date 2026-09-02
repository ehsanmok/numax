<h1 align="center">numax</h1>

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

**One kernel, several meanings.** Every function is written once against the
`FloatLike` trait. The type you call it with decides what comes back: a value,
a derivative, extra precision, a complex result, an interval bound.

**One tensor, every device.** `Tensor` owns a MAX `DeviceBuffer`, so the
`DeviceContext` you pass a factory decides host or device memory. Nothing else
changes, and `.view()` yields the `TileTensor` every MAX kernel takes.

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

## Getting started

Write $g(x) = e^{-x^2}$ once. Call it twice, and the second call hands back
its derivative:

```mojo
from numax import Dual, FloatLike, Plain

def g[T: FloatLike](x: T) -> T:                # written once, no `dtype`
    return (-(x * x)).exp()

def main():
    comptime P = Plain[DType.float32, 1]       # ordinary SIMD, as a FloatLike

    print(g(P(0.5)).v)                         # 0.7788008    -- just the value
    var d = g(Dual[P](P(0.5), P(1.0)))         # seed the derivative with 1
    print(d.value.v, d.deriv.v)                # 0.7788008 -0.7788008
```

That is forward-mode automatic differentiation — $g(x)$ and
$g'(x) = -2xe^{-x^2}$ from one call, by the chain rule built into `Dual`'s
arithmetic. It works on every function in the library.

<details>
<summary><strong>Why <code>Plain</code>, and does it cost anything?</strong></summary>

`Plain[dtype, width]` wraps a `SIMD[dtype, width]`. Mojo declares conformance
where a type is defined, so a bare `SIMD` cannot conform to `FloatLike` and
`Plain` is what lets the hardware type participate. It is the baseline, not a
niche: every kernel runs at `Plain` unless you ask for something else.

It compiles away. `pixi run bench` measures `map` over a `Plain` kernel at
**0.998x a hand-written raw-SIMD loop**, `max |raw - numax| = 0.0`.

`.v` unwraps a `Plain`; `Dual` exposes `.value`/`.deriv`.
</details>

> **Run it:** `pixi run example-gaussian` ·
> [`examples/basic/gaussian.mojo`](examples/basic/gaussian.mojo)

### 1. Precision, for free

Sum a million nearly-equal `float32` values and the running total stops seeing
the next one. Swap the type, not the algorithm:

```mojo
var plain_var = variance(plain_list).v         # float32 accumulation
var comp_var = variance(comp_list).value       # ~double precision, same code
```

`Compensated` carries the rounding error ordinary arithmetic discards.

> **Run it:** `pixi run example-statistics` ·
> [`examples/intermediate/statistics.mojo`](examples/intermediate/statistics.mojo)

### 2. The same kernel, on the GPU

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

### 3. Any rank, not just vectors

Rank is a compile-time variadic, and `map`/`reduce` coalesce any contiguous
row-major tensor before running your kernel:

```mojo
var a = full[dtype, 2, 3, 4](ctx, 0.5)      # rank-3, 24 elements
map[step=gaussian_step, width=4](a.view(), b.view())
print(sum(a.view()), mean(a.view()))        # reductions over every element

reduce_rows[combine = add_combine[dtype]](m.view(), rows.view(), 0)      # per row
reduce_axis[combine = add_combine[dtype], axis=0](m.view(), cols.view(), 0)
```

> **Where rank stops.** `numax.statistics` reduces over every element and has
> no `axis=` yet — axis-wise folding is `numax.tensor.reduce_axis`/
> `reduce_rows`. `numax.sorting` flattens. `transpose` is 2-D,
> `concatenate`/`split`/`stack` rank-1, `reshape` targets rank 2 or 3. The
> SciPy-shaped algorithms are fixed-size `Array` kernels: `linalg` is
> matrices, `fft2` a square transform, `rk4_system` an `n`-component state,
> while `quad`, `solve_ivp`, the splines and the distributions are 1-D. There
> is no broadcasting yet.

### 4. Exact gradients into an optimizer

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

### 5. A Hessian, by nesting types

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
| Guaranteed bounds | `Interval[Plain]` | [`numax/interval.mojo`](numax/interval.mojo) |
| Exact decimals (`0.1 + 0.2 == 0.3`) | `Decimal[width, scale]` | [`numax/decimal.mojo`](numax/decimal.mojo) |
| Every special function differentiated | `Dual` | `pixi run example-special-functions` |
| 1024 ODE trajectories, one GPU thread each | `Dual` for sensitivities | `pixi run example-ode` |

Full index: [`examples/README.md`](examples/README.md). `pixi run examples-cpu`
skips the GPU ones; `pixi run examples` includes them.

## The trait and its conformers

`FloatLike` is small: `+ * / -`, `exp`, `ln`, `sqrt`, `erf`, `erfc`, `sin`,
`cos`, `abs`, `copysign`, `floor`, `ceil`, `trunc`, `one()`, `constant()`.

| Type | What you get |
|---|---|
| `Plain[dtype, width]` | Ordinary `SIMD`, at hardware speed |
| `Dual[Inner]` | Forward-mode autodiff; nests for second derivatives |
| `Compensated[dtype, width]` | A value carried as $a + b$ with $\lvert b \rvert \ll \lvert a \rvert$ — roughly double `dtype`'s precision |
| `Decimal[width, scale]` | Exact base-10 fixed point |
| `Complex[Inner]` | Complex over any conformer; `Complex[Dual[...]]` differentiates holomorphically |
| `Gradient[Inner, n_vars]` | Full gradient from one call; nests with `Dual` for Hessians |
| `Interval[Inner]` | An enclosure of $\\{f(x) : x \in [\ell, u]\\}$ |

They nest (`Complex[Dual[Plain[...]]]`, `Gradient[Dual[...], n]`), so autodiff,
precision and complex arithmetic compose instead of each needing its own copy
of every kernel. Design rationale: [`docs/architecture.md`](docs/architecture.md).

## What's in the box

| Module | Contents |
|---|---|
| `special`, `erf`, `gamma`, `beta`, `bessel`, `lambertw`, `elliptic`, `legendre`, `orthopoly` | $\Gamma$, $B$, $\mathrm{erf}$, $J_n$/$Y_n$, $W$, $K$/$E$, orthogonal polynomials, activations |
| `solve`, `quadrature`, `ode` | `newton`/`halley`/`bisection`, Gauss-Legendre/Simpson/trapezoid, `rk4`/`dopri5` — fixed iteration counts |
| `integrate`, `optimize` | `quad`, `quad_vec`, `solve_ivp`; `newton_tol`, `brentq`, `bfgs` — iterate to a tolerance |
| `linalg` | `cholesky`, `lu`, `qr`, `eigh`, `svd`, `solve`, `inverse`, `pinv`, `det`, `trace`, `cond`, norms, `dot`/`nrm2`/`outer`, `matmul`, `tridiagonal_solve` |
| `fft`, `signal` | `fft`/`ifft`, `rfft`/`irfft`, `fft2`, `fftfreq`, `fftshift`; `convolve`, `correlate`, windows |
| `interp`, `distributions` | Horner, cubic splines, Chebyshev; pdf/cdf/quantile for nine families |
| `array` | Creation, `*_like`, reshape/ravel/transpose/squeeze/stack/concatenate/split, `diag`, `tri`/`tril`/`triu`, `vander`, `meshgrid`, `flip`, `vstack`/`hstack` |
| `ops`, `elementwise`, `logic` | Arithmetic and operators on `Tensor`, `astype`; the elementwise math surface; comparisons, `isnan` family, logical ops, `allclose` |
| `statistics`, `sorting` | `sum`/`mean`/`median`/`mode`/`argmax`…; `sort`, `argsort`, `searchsorted`, `unique`, `nonzero`, `extract`, `select` |
| `io`, `random`, `constants` | Binary `save`/`load`, `print_tensor`; `uniform`/`normal`/`exponential`/`randint`/`randbool`/`seed`; `pi`, `e` |

`linalg`'s matrices are small and compile-time-sized — registers, not heap.
That is what makes `cholesky` differentiable at `Dual` and GPU-launchable, and
why MAX's own `linalg` is the right call past roughly 8x8.

[`docs/parity.md`](docs/parity.md) records what numax absorbs, what it routes
to MAX, and what it leaves out — plus the surveyed MAX surface those decisions
rest on. Worth knowing: MAX ships one matrix decomposition and no forward FFT.

## Tensors, devices, and the two tiers

`numax.tensor.map`/`reduce` drive a `FloatLike` kernel over a `TileTensor`,
CPU or GPU, chosen by one `gpu: Bool` parameter. `map_threaded` spreads it
across cores via `max.algorithm.elementwise`; `reduce_axis`/`broadcast_op_axis`
fold and broadcast along any axis.

Shapes may be comptime or runtime: `map`/`reduce` have two overloads under one
name, picked by `where` clauses that are exact negations. The static path can
be launched on a GPU; the runtime one is CPU-only, since a launch needs the
extent in the type. There is no second tensor type — both are `TileTensor`.

`Tensor` adds ownership only. Reads go through `to_host()`/`copy_from_host()`,
never `DeviceBuffer.unsafe_ptr()`, which on CUDA returns a *device* pointer
that segfaults a host read.

**Tier 1** is everything with a fixed iteration count and no per-lane
branching, and therefore launchable inside a GPU thread: the special
functions, the algorithms layer, `linalg`. **Tier 2** is `optimize`,
`integrate`, `sorting`, `logic`, `elementwise` and `ops`: `Plain`-only,
host-side, free to loop or branch on data. Every module docstring declares its
tier, and tier 1 never calls tier 2.

## Performance and accuracy

Same kernel everywhere — $g(x) = e^{-x^2}$ over `float32`, 67M elements, wall
clock around dispatch through completion. M elem/s, higher is better.

**NVIDIA A10G (CUDA 13):**

| | numax GPU | torch.compile (CUDA) | torch eager (CUDA) | numax CPU (threaded) | Rust `thermite` (AVX2) | NumPy |
|---|---|---|---|---|---|---|
| M elem/s | **61,537** | 56,141 | 19,893 | 8,527 | 1,326 | 313 |

**Apple M3 Pro (Metal):**

| | numax GPU | torch.compile (Metal) | MLX (GPU) | numax CPU (threaded) | Rust `thermite` (NEON) | NumPy |
|---|---|---|---|---|---|---|
| M elem/s | **15,755** | 14,380 | 4,866 | 9,026 | 1,632 | 493 |

numax leads `torch.compile` by ~10% on the A10G and ties it on Metal, at
**500.9 GB/s — ~83% of the A10G's 600 GB/s spec**. The work is
bandwidth-bound, not compute-bound: an identity copy runs at 489.10 GB/s
against the Gaussian's 489.16, so the `exp` is free. Fusing two passes into
one composed `step` is worth **1.99x** on the GPU at every size. On the CPU
the serial walk matches hand-written Rust SIMD (1,394 vs 1,326) and is 4.5x
NumPy.

The GPU path is written against `DeviceContext`, not a backend — the same
source produces both rows above. Full sweeps, both sync shapes, and the
methodology: [`docs/performance.md`](docs/performance.md),
[`bench/README.md`](bench/README.md).

Every approximation documents an error bound, and `pixi run accuracy` checks it
against checked-in [mpmath](https://mpmath.org/) references at 50 digits
(`erf`'s A&S 7.1.26 bound of ~1.5e-7 measures 1.38e-07). One caveat: Mojo's
`std.math` `exp`/`log`/`erf` are not correctly rounded at `float64`, so every
function built on them inherits that floor — invisible at `float32`. Details:
[`bench/accuracy/README.md`](bench/accuracy/README.md).

## Testing

```bash
pixi run tests           # 40 suites, 628 tests
pixi run examples-cpu    # every example that does not need a GPU
pixi run bench           # map vs. a hand-rolled raw-SIMD loop
pixi run accuracy        # max error per function vs. mpmath references
```

`tests` and `examples-cpu` run in CI on macOS and Linux; GPU examples and
benchmarks are a local check.

## Documentation

- [`examples/README.md`](examples/README.md) — every example, one line each.
- [`docs/architecture.md`](docs/architecture.md) — the trait, the
  fixed-iteration invariant, the tensor/GPU layer.
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
