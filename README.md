<h1 align="center">numax</h1>

<p align="center">
  <a href="https://github.com/ehsanmok/numax/actions/workflows/ci.yml"><img src="https://github.com/ehsanmok/numax/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/ehsanmok/numax/actions/workflows/docs.yaml"><img src="https://github.com/ehsanmok/numax/actions/workflows/docs.yaml/badge.svg" alt="Docs"></a>
  <a href="https://mojolang.org"><img src="https://img.shields.io/badge/Mojo-1.0.0-orange" alt="Mojo 1.0.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0%20with%20LLVM%20Exceptions-yellow.svg" alt="License: Apache 2.0 with LLVM Exceptions"></a>
</p>

<p align="center"><em>NumPy and SciPy's ground, in Mojo, on MAX — where one kernel means several things and runs on any device.</em></p>

## What $\nu$MAX is

$\nu$MAX is a numerical computing library for
[Mojo](https://mojolang.org): special functions, linear algebra, quadrature,
ODE solvers, FFTs, distributions, and a NumPy-named array/statistics/random
surface, built on [MAX](https://max.modular.com/docs/)'s `TileTensor` and
kernels.

Two ideas hold it together.

**One kernel, several meanings.** Every function is written once against a
trait, `FloatLike`, instead of once per numeric type. The type you call it
with decides what you get back — a value, a derivative, extra precision, a
complex result, an interval bound.

**One tensor, every device.** `numax.array.Tensor` owns a MAX `DeviceBuffer`,
so the `DeviceContext` you hand a factory decides whether it lives in host
memory or on an accelerator. Nothing else changes: same type, same shape
parameters, same `.view()` handing back the `TileTensor` every MAX kernel
already takes.

This is a young, experimental library — APIs may change.

## Install

```toml
[workspace]
channels = ["https://conda.modular.com/max", "conda-forge"]
preview = ["pixi-build"]

[dependencies]
numax = { git = "https://github.com/ehsanmok/numax.git", tag = "<latest-release>" }
```

```bash
pixi install
```

Requires [pixi](https://pixi.sh). Pin to a released tag for reproducible
builds; `mojo` and `max` come in as transitive dependencies, and `numax` ships
as source.

## Getting started

Write the Gaussian, $g(x) = e^{-x^2}$, once. Call it twice, and the second
call hands back its derivative:

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

The second line is forward-mode automatic differentiation: $g(x)$ and
$g'(x) = -2xe^{-x^2}$ from one call, by the chain rule built into `Dual`'s
arithmetic. No second formula, no tape, no autodiff framework — and it works
on every function in this library, not only this one.

> **Run it:** `pixi run example-gaussian` — the same kernel at `Plain`,
> `Dual` and `Compensated`, checked against a reference ·
> [`examples/basic/gaussian.mojo`](examples/basic/gaussian.mojo)

<details>
<summary><strong>Why <code>Plain</code>, and does it cost anything?</strong></summary>

`Plain[dtype, width]` is a one-field wrapper holding a
`SIMD[dtype, width]`. It exists because Mojo conformance is declared where a
type is defined — you cannot retroactively make a type you don't own conform
to a new trait ([Mojo manual, *Traits → Things to
know*](https://docs.modular.com/mojo/manual/traits/)) — so a bare `SIMD`
cannot conform to `FloatLike` and `Plain` is what lets the hardware type
participate. `Plain` is the *baseline* instantiation, not a niche one: every
kernel in numax runs at `Plain` unless you ask for something else.

It compiles away. `pixi run bench` drives `map` over a `Plain` kernel against
a hand-written raw-SIMD loop over 1M `float32` elements: **0.998x the raw
loop, `max |raw - numax| = 0.0`** — the wrapper and the trait dispatch cost
nothing measurable.

`.v` unwraps a `Plain` back to `SIMD`; `Dual` exposes `.value`/`.deriv`, and
each conformer names its own parts.
</details>

---

### 1. Precision, for free

Sum a million nearly-equal `float32` values and the running total stops being
able to see the next one. Swap the type instead of the algorithm:

```mojo
from numax import Compensated
from numax.statistics import variance

var plain_var = variance(plain_list).v         # float32 accumulation
var comp_var = variance(comp_list).value       # ~double precision, same code
```

`Compensated` carries the rounding error ordinary arithmetic throws away, so
a long summation keeps roughly double `float32`'s precision — without
`float64`, and without a second implementation of `variance`.

> **Run it:** `pixi run example-statistics` ·
> [`examples/intermediate/statistics.mojo`](examples/intermediate/statistics.mojo)

### 2. The same kernel, on the GPU

One tensor type, two devices. The only differences below are which context
gets created and whether the walk is a host loop or a launch:

```mojo
from max.gpu.host import DeviceContext
from numax.array import Tensor, linspace
from numax.tensor import map

comptime T = Tensor[DType.float32, 1024]

def gaussian_step[w: Int](x: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    return gaussian(Plain[DType.float32, w](x)).v

# CPU
var cpu = DeviceContext(api="cpu")
var xs = linspace[DType.float32, 1024](cpu, -2.0, 2.0)
var ys = T(cpu)
map[step=gaussian_step, width=8](xs.view(), ys.view())

# GPU -- same tensor type, same kernel, same view type
var gpu = DeviceContext()
var gxs = linspace[DType.float32, 1024](gpu, -2.0, 2.0)
var gys = T(gpu)
gpu.enqueue_function[map[LayoutType = T.LayoutType, step=gaussian_step, gpu=True]](
    gxs.view(), gys.view(), grid_dim=4, block_dim=256
)
```

Every `FloatLike` conformer is built from plain `SIMD` fields with no pointers
or allocations, so `Dual` and `Compensated` kernels launch on a GPU thread
unmodified too — the derivative and the extra-precision versions of the code
above run on-device with no changes.

> **Run it:** `pixi run example-unified-tensor-gpu` (needs a GPU) ·
> [`examples/advanced/unified_tensor_gpu.mojo`](examples/advanced/unified_tensor_gpu.mojo)

### 3. Any rank, not just vectors

`Tensor`'s shape is a compile-time variadic, so rank is just more parameters,
and `map`/`reduce` coalesce any contiguous row-major tensor to a flat walk
before running your kernel:

```mojo
from numax.statistics import mean, sum
from numax.tensor import add_combine, map, reduce_axis, reduce_rows

var a = full[dtype, 2, 3, 4](ctx, 0.5)      # rank-3, 24 elements
var b = Tensor[dtype, 2, 3, 4](ctx)
map[step=gaussian_step, width=4](a.view(), b.view())   # same kernel, rank-3

print(sum(a.view()), mean(a.view()))        # reductions over every element

var m = full[dtype, 3, 4](ctx, 2.0)         # rank-2
reduce_rows[combine = add_combine[dtype]](m.view(), rows.view(), 0)      # per row
reduce_axis[combine = add_combine[dtype], axis=0](m.view(), cols.view(), 0)  # per column
```

The optimizer is multivariate for the same reason — `bfgs[n_vars, f]` takes
any `n_vars`, and section 4's two-variable Rosenbrock is just the classic test
case, not a limit.

> **Where rank stops, plainly.** `numax.statistics` reduces over *every*
> element and has no `axis=` keyword yet — axis-wise folding lives in
> `numax.tensor.reduce_axis`/`reduce_rows`. `numax.sorting` flattens (NumPy's
> `axis=None`). Manipulation has documented rank limits: `transpose` is 2-D,
> `concatenate`/`split`/`stack` are rank-1, `reshape` targets rank 2 or 3.
> And the SciPy-shaped algorithms are small fixed-size `Array` kernels rather
> than tensor ops: `linalg` is matrices, `fft2` is a square 2-D transform,
> `rk4_system` integrates an `n`-component state — but `quad`, `solve_ivp`,
> the splines, `signal.convolve` and the distributions are one-dimensional.
> There is no broadcasting yet.

### 4. Exact gradients, straight into an optimizer

Optimizers normally estimate gradients with finite differences. numax's
objective is an ordinary `FloatLike` kernel, so BFGS evaluates it at
`Gradient` and gets every $\partial f / \partial x_i$ *exactly*. Minimizing
Rosenbrock, $f(x, y) = (1 - x)^2 + 100(y - x^2)^2$:

```mojo
from std.collections import Array
from numax import FloatLike
from numax.optimize import bfgs

def rosenbrock[U: FloatLike](v: Array[U, 2]) -> U:
    var a = U.one() + (-v[0])
    var b = v[1] + (-(v[0] * v[0]))
    return a * a + U.constant(100.0) * b * b

var start = Array[Float64, 2](fill=0)
start[0] = -1.2
start[1] = 1.0
var minimized = bfgs[2, rosenbrock](start)     # no `jac` argument
```

A central difference,
$\frac{f(x+h) - f(x-h)}{2h}$, cannot beat about $\varepsilon^{2/3}$ relative
accuracy no matter how you pick $h$: truncation error falls as $O(h^2)$ while
cancellation error grows as $O(\varepsilon/h)$, and the two meet at
$h \sim \varepsilon^{1/3}$. Forward-mode AD has neither term. The example sweeps the
step size and prints both error curves: the best finite difference lands at
~5e-10, AD at exactly 0.

> **Run it:** `pixi run example-optimize` ·
> [`examples/advanced/optimize.mojo`](examples/advanced/optimize.mojo)

### 5. A Hessian, by nesting types

The conformers compose, and second-order information falls out of that
composition rather than out of second-order code:

```mojo
from numax import Dual, Gradient, Plain

comptime P = Plain[DType.float32, 1]
comptime G = Gradient[Dual[P], 2]      # gradient of a dual number
```

`Dual` inside itself is a second derivative. `Gradient` over `Dual` is the
full Hessian $H_{ij} = \partial^2 f / \partial x_i \partial x_j$, and
Hessian-vector products $Hv$ with it. Neither type contains a line of
second-order mathematics.

> **Run it:** `pixi run example-hessian` ·
> [`examples/basic/hessian.mojo`](examples/basic/hessian.mojo)

### Keep pulling the thread

| You want | Call it at | Example |
|---|---|---|
| Holomorphic derivatives | `Complex[Dual[Plain]]` | `pixi run example-complex` |
| Guaranteed bounds instead of a value | `Interval[Plain]` | [`numax/interval.mojo`](numax/interval.mojo) |
| Exact decimal arithmetic (`0.1 + 0.2 == 0.3`) | `Decimal[width, scale]` | [`numax/decimal.mojo`](numax/decimal.mojo) |
| Every special function differentiated | `Dual` | `pixi run example-special-functions` |
| Row-wise softmax on CPU and GPU | — | `pixi run example-softmax` |
| 1024 ODE trajectories, one GPU thread each | `Dual` for sensitivities | `pixi run example-ode` |

Full index with one line per file: [`examples/README.md`](examples/README.md).
`pixi run examples-cpu` runs everything that doesn't need a GPU;
`pixi run examples` includes the GPU ones.

## The trait and its conformers

`FloatLike` is deliberately small — just enough arithmetic to build the
library on: `+ * / -`, `exp`, `ln`, `sqrt`, `erf`, `erfc`, `sin`, `cos`,
`abs`, `copysign`, `floor`, `ceil`, `trunc`, plus `one()` and `constant()`.

| Type | What you get |
|---|---|
| `Plain[dtype, width]` | Ordinary `SIMD[dtype, width]`, at hardware speed. |
| `Dual[Inner: FloatLike]` | Forward-mode autodiff: value + derivative. Nests for second derivatives. |
| `Compensated[dtype, width]` | Double-double arithmetic — a value carried as $a + b$ with $\lvert b \rvert \ll \lvert a \rvert$, so roughly double `dtype`'s precision with no `float64`. |
| `Decimal[width, scale]` | Exact base-10 fixed-point, so `0.1 + 0.2 == 0.3` exactly. |
| `Complex[Inner: FloatLike]` | A complex number over any other conformer — `Complex[Dual[...]]` differentiates holomorphically. |
| `Gradient[Inner: FloatLike, n_vars: Int]` | Full gradient vector from one call; nests with `Dual` for Hessians. |
| `Interval[Inner: FloatLike]` | Bounds instead of a value: run a kernel over $[\ell, u]$, get an enclosure of $\{f(x) : x \in [\ell, u]\}$. |

Each nests inside the others (`Complex[Dual[Plain[...]]]`,
`Gradient[Dual[...], n]`, ...), so autodiff, precision, and complex arithmetic
compose instead of each needing its own copy of every kernel. Documented
limitations live in each conformer's module docstring;
[`docs/architecture.md`](docs/architecture.md) has the design rationale.

## What's in the box

**Special functions** — one file per family, each written once against
`FloatLike`. $\Gamma$, $B$, $\mathrm{erf}$, $J_n$/$Y_n$, $W$, $K$/$E$, and the
classical orthogonal polynomials:

| Module | Functions |
|---|---|
| `numax.special` | `gaussian`, `sigmoid`, `swish`, `tanh`, `relu`, `leaky_relu`, `gelu`, `softmax` |
| `numax.erf` | `erf`, `erfc` |
| `numax.gamma` | `gamma`, `lgamma`, `gammainc`, `gammaincc`, `digamma` |
| `numax.beta` | `beta`, `betainc`, `betaincc` |
| `numax.legendre` / `numax.orthopoly` | `legendre_p`, `hermite_h`, `laguerre_l`, `chebyshev_t`, `chebyshev_u` |
| `numax.bessel` | `bessel_j0`, `bessel_j1`, `bessel_y0`, `bessel_y1` |
| `numax.lambertw` | `lambertw`, `lambertw_m1` |
| `numax.elliptic` | `elliptic_k`, `elliptic_e` |

`gamma`/`lgamma` are valid for any `x` except the non-positive integers (via
reflection); Bessel and Lambert W cover both real branches. Where MAX's or
Mojo's own accelerator math is GPU-compatible and at least as accurate
(`erf`, `sin`/`cos`), `Plain` calls it directly; where it is CPU-only libm
(`gamma`, Bessel), numax keeps its own GPU-launchable implementation.

**Algorithms** — also `FloatLike`-generic, so a solver or distribution
inherits autodiff and extra precision with no second implementation:

| Module | Functions |
|---|---|
| `numax.solve` | `newton`, `halley`, `bisection` — no derivative to supply; each evaluates `f` at `Dual` internally |
| `numax.quadrature` | `gauss_legendre`, `simpson`, `trapezoid` — fixed nodes |
| `numax.integrate` | `quad`, `quad_vec` (adaptive quadrature), `solve_ivp` (adaptive ODE) |
| `numax.linalg` | `cholesky`, `lu`, `qr`, `eigh`, `svd`, `solve`, `inverse`, `pinv`, `det`, `trace`, `cond`, the `norm_*` family, `dot`/`nrm2`/`outer`, `matmul`, `matvec`, `tridiagonal_solve` |
| `numax.fft` | `fft`/`ifft`, `rfft`/`irfft`, `fft2`/`ifft2`, `fftfreq`/`rfftfreq`, `fftshift`, `circular_convolve` |
| `numax.signal` | `convolve`, `convolve_same`, `correlate`, `hann`/`hamming`/`blackman`, `apply_window` |
| `numax.interp` | `horner`, natural cubic splines, `chebyshev_fit`/`chebyshev_eval` |
| `numax.distributions` | pdf/cdf/quantile for normal, exponential, gamma, chi-square, beta, Student-t, F, Poisson, binomial |
| `numax.ode` | `rk4`, `rk4_system`, Dormand-Prince `dopri5` — fixed step |
| `numax.optimize` | `newton_tol`, `brentq`, `bfgs` — iterate to a tolerance |

```mojo
def cos_minus_x[U: FloatLike](x: U) -> U:
    return x.cos() + (-x)

var root = newton[f=cos_minus_x](Plain[dtype, width](1.0))   # no f' needed
```

`numax.linalg`'s matrices are small and compile-time-sized (registers, not
heap) — that is what makes `cholesky` differentiable at `Dual` (useful for a
Gaussian process or Kalman filter) and GPU-launchable, but it is the wrong
tool for a large system: use MAX's own `linalg` past roughly 8x8, as each
docstring notes.

**Array, statistics, I/O, random** — a NumPy-named surface over MAX's
`TileTensor`, added only where MAX ships no equivalent:

| Module | Functions |
|---|---|
| `numax.array` | `zeros`/`ones`/`full`/`empty`/`eye`/`linspace`/`logspace`/`arange`/`*_like`, `reshape`/`ravel`/`transpose`/`squeeze`/`stack`/`concatenate`/`split` |
| `numax.statistics` | `sum`/`prod`/`min`/`max`/`mean`/`median`/`mode`/`argmax`/`argmin`; `variance`/`stddev`/`cumsum` also work over any `FloatLike` |
| `numax.io` | binary `save`/`load`, `print_tensor` |
| `numax.random` | `uniform`, `normal`, `exponential`, `seed` |
| `numax.sorting` | `sort`, `argsort`, `searchsorted`, `unique`, `count_nonzero`, `any_nonzero`/`all_nonzero`, `nonzero`, `extract` (boolean masking), `select` |

[`docs/parity.md`](docs/parity.md) is the full disposition table: what numax
absorbs, what it routes to MAX, what it leaves out — and the surveyed MAX API
those decisions rest on. Worth knowing up front: MAX ships one matrix
decomposition and no forward FFT, so numax's own kernels carry more of the
mathematics than "a layer over MAX" suggests.

## Tensors, devices, and the two tiers

`numax.tensor.map`/`reduce` drive a `FloatLike` kernel over a `TileTensor`,
CPU or GPU, picked with one `gpu: Bool` compile-time parameter.
`map_threaded` spreads the same kernel across CPU cores via
`max.algorithm.elementwise`; `reduce_axis`/`broadcast_op_axis` fold and
broadcast along any axis of any rank.

Shapes can be known at compile time or at runtime. `map` and `reduce` each
have two overloads under one name, picked by `where` clauses that are exact
negations: a comptime `row_major[n]()` tensor takes the static path (and can
be launched on GPU), a runtime `row_major(Coord(n))` tensor takes the runtime
path (CPU only — a launch needs the extent in the type). There is no second
tensor type; both are `TileTensor`, and the distinction lives in the layout
where MAX already put it.

Rank is a compile-time variadic on `Tensor[dtype, *dims]`, and `map`/`reduce`
accept any rank by coalescing to a flat walk first, so the elementwise and
reduction layer is genuinely n-dimensional. The SciPy-shaped algorithms above
it are not: they are small, fixed-size, register-resident `Array` kernels, so
`linalg` works on matrices, `fft2` on a square 2-D transform, `rk4_system` on
an `n`-component state, and `quad`/`solve_ivp`/`interp`/`signal`/the
distributions on one dimension. Reductions fold over every element —
`axis=`-style folding is `numax.tensor.reduce_axis`/`reduce_rows`, not a
keyword on `numax.statistics` — and there is no broadcasting yet.

`Tensor` adds ownership and nothing else: a bare `TileTensor` is a view — a
pointer plus a layout — and dangles the moment the function that built it
returns. Reads go through `to_host()`/`copy_from_host()`, never
`DeviceBuffer.unsafe_ptr()`, which on CUDA hands back a *device* pointer that
segfaults a host read.

**Tier 1** is everything with a fixed iteration count and no per-lane
branching, which is therefore launchable inside a GPU thread unmodified —
the special functions, the algorithms layer, `numax.linalg`. That is what the
`FloatLike` spine buys.

**Tier 2** is `numax.optimize`, `numax.integrate` and `numax.sorting`:
`Plain`-only, host-side, free to loop until they converge. Root finding to a
tolerance, adaptive quadrature, pivoting and comparison sorts need that, and
pretending otherwise would either exclude them or quietly weaken the tier-1
guarantee. Every module and function docstring declares its tier, and tier 1
never calls tier 2.

## Performance and accuracy

Measured on an Apple M3 Pro (~150 GB/s memory bandwidth) on
$g(x) = e^{-x^2}$ over `float32`, which is memory-bandwidth-bound at
every size tested — an identity copy runs at the same speed, so the `exp` is
free. At 67M elements:

| | numax CPU (`map_threaded`) | numax GPU (amortized) | NumPy (CPU) | MLX (GPU, per-call) | torch.compile (GPU, amortized) |
|---|---|---|---|---|---|
| M elem/s | 9,026 | 15,755 | 493 | 4,866 | 14,380 |

numax's GPU path reaches 84% of that hardware's bandwidth ceiling, on par with
`torch.compile` on the same Metal device. The GPU path is written against
`DeviceContext` rather than any one backend: on an NVIDIA A10G (CUDA 13) the
same code measures 500.9 GB/s at its best block size (62.6 G elem/s,
amortized) — ~83% of that card's 600 GB/s spec — and kernel fusion there is
worth a 1.99x speedup at every size tested. Full sweeps,
both CPU walks, and every cross-language baseline (NumPy, MLX, PyTorch, Rust
`thermite`) are in [`docs/performance.md`](docs/performance.md) and
[`bench/README.md`](bench/README.md).

Every approximation documents an error bound from the literature it came from,
and `pixi run accuracy` checks this implementation delivers it against
[mpmath](https://mpmath.org/) references at 50 decimal digits — checked in, so
the harness needs no Python at run time. The bounds hold (`erf`'s A&S 7.1.26
bound of ~1.5e-7 measures 1.38e-07). One caveat applies library-wide: Mojo's
`std.math` `exp`/`log`/`erf` are not correctly rounded at `float64`, so every
numax function built on them inherits that floor at `float64` — invisible at
the `float32` this library normally runs at. Full table and the defect
writeup: [`bench/accuracy/README.md`](bench/accuracy/README.md).

## Testing and benchmarks

```bash
pixi run tests           # 36 suites, 566 tests
pixi run bench           # numax.tensor.map vs. a hand-rolled raw-SIMD loop
pixi run bench-gpu       # CPU vs. GPU across a size sweep
pixi run bench-roofline  # bandwidth vs. arithmetic, block size, sync accounting
pixi run bench-numpy     # cross-language: NumPy
pixi run bench-mlx       # cross-language: MLX (CPU + GPU; Metal or CUDA)
pixi run bench-torch     # cross-language: PyTorch (eager + compile, CPU + GPU)
pixi run bench-thermite  # cross-language: Rust thermite (NEON or AVX2)
pixi run accuracy        # max error per function vs. checked-in mpmath refs
```

`pixi run tests` and `pixi run examples-cpu` run in CI on every push, on macOS
and Linux; GPU examples and benchmarks are a local check, since no
GitHub-hosted runner has a GPU.

## License

The source code in this repository is licensed under
[Apache 2.0 (with LLVM exceptions)](LICENSE). MAX is distributed as prebuilt
binaries and container images, which are licensed separately under the
[Modular Community License](https://www.modular.com/legal/community). The
applicable license is determined by the artifact you are using, not by how you
obtained it.

numax depends on MAX at build and runtime, so using this library means using
those prebuilt MAX artifacts under their own terms.
