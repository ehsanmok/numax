<h1 align="center">numax</h1>

<p align="center">
  <a href="https://github.com/ehsanmok/numax/actions/workflows/ci.yml"><img src="https://github.com/ehsanmok/numax/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/ehsanmok/numax/actions/workflows/docs.yaml"><img src="https://github.com/ehsanmok/numax/actions/workflows/docs.yaml/badge.svg" alt="Docs"></a>
  <a href="https://mojolang.org"><img src="https://img.shields.io/badge/Mojo-1.0.0-orange" alt="Mojo 1.0.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

<p align="center"><em>A numerical computing library for Mojo, built on MAX.</em></p>

`numax` covers the ground NumPy and SciPy cover — special functions, linear
algebra, quadrature, ODE solvers, distributions, an array/statistics/random
surface — but every kernel is written once, against a trait, instead of once
per numeric type:

```mojo
def gaussian[T: FloatLike](x: T) -> T:
    return (-(x * x)).exp()
```

- `gaussian(Plain(x))` — ordinary `SIMD[dtype, width]`, at hardware speed.
- `gaussian(Dual(x, seed))` — the value **and** its derivative, via forward-mode
  automatic differentiation. No second formula, no separate autodiff framework.
- `gaussian(Compensated(x, 0))` — the value at roughly double `dtype`'s
  precision, recovered from rounding error ordinary arithmetic would discard.

Same function body, no edits between them. The same holds for every special
function, solver, and distribution below: swap the type, get forward-mode AD,
extra precision, complex arithmetic, or interval bounds for free. This is a
young, experimental library — APIs may change.

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
builds; `mojo` and `max` install as transitive dependencies, and `numax` is
distributed as source, so `from numax import ...` resolves with no `-I <path>`
flag needed.

## Quick start

```mojo
from numax import Compensated, Complex, Decimal, Dual, FloatLike, Gradient, Plain
from numax import gaussian, sigmoid, swish, tanh
from numax import erf, gamma, bessel_j0, lambertw, elliptic_k
```

## The trait and its conformers

`FloatLike` is deliberately small — just enough arithmetic to build the
library on:

```mojo
trait FloatLike(Copyable, Movable, Deinitable):
    @staticmethod
    def one() -> Self: ...
    def __add__(self, rhs: Self) -> Self: ...
    def __mul__(self, rhs: Self) -> Self: ...
    def __neg__(self) -> Self: ...
    def __truediv__(self, rhs: Self) -> Self: ...
    def exp(self) -> Self: ...
    def ln(self) -> Self: ...
    def sqrt(self) -> Self: ...
    def erf(self) -> Self: ...
    def erfc(self) -> Self: ...
    def sin(self) -> Self: ...
    def cos(self) -> Self: ...
    @staticmethod
    def constant(v: Float64) -> Self: ...
    def abs(self) -> Self: ...
    def copysign(self, sign_source: Self) -> Self: ...
    def floor(self) -> Self: ...
    def ceil(self) -> Self: ...
    def trunc(self) -> Self: ...
```

| Type | What you get |
|---|---|
| `Plain[dtype, width]` | Ordinary `SIMD[dtype, width]`, at hardware speed. |
| `Dual[Inner: FloatLike]` | Forward-mode autodiff: value + derivative, via the chain rule. Nests for second derivatives. |
| `Compensated[dtype, width]` | Double-double arithmetic — roughly double `dtype`'s precision, no `float64` required. |
| `Decimal[width, scale]` | Exact base-10 fixed-point, so `0.1 + 0.2 == 0.3` exactly. |
| `Complex[Inner: FloatLike]` | A complex number over any other conformer — `Complex[Dual[...]]` differentiates holomorphically. |
| `Gradient[Inner: FloatLike, n_vars: Int]` | Full gradient vector from one call; nests with `Dual` for Hessians. |
| `Interval[Inner: FloatLike]` | Bounds instead of a value: run a kernel over `[lo, hi]`, get the range it can produce. |

Each conformer nests inside the others (`Complex[Dual[Plain[...]]]`,
`Gradient[Dual[...], n]`, ...), so autodiff, precision, and complex arithmetic
compose rather than each needing their own version of every kernel. Details
and documented limitations live in each conformer's module docstring; see
[`docs/architecture.md`](docs/architecture.md) for the design rationale.

## Special functions

One file per family, each written once against `FloatLike`:

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
(`erf`, `sin`/`cos`), `Plain` calls it directly; where it's CPU-only libm
(`gamma`, Bessel), `numax` keeps its own GPU-launchable implementation
instead. See `examples/intermediate/special_functions.mojo` for all of the
above differentiated via `Dual` with no extra code.

## Algorithms

Also written against `FloatLike`, so a solver, integrator, or distribution
inherits autodiff and extra precision with no second implementation:

| Module | Functions |
|---|---|
| `numax.solve` | `newton`, `halley`, `bisection` — no derivative to supply; each evaluates `f` at `Dual` internally |
| `numax.quadrature` | `gauss_legendre`, `simpson`, `trapezoid` — fixed nodes |
| `numax.integrate` | `quad`, `quad_vec` — adaptive, subdividing to a tolerance |
| `numax.linalg` | `cholesky`, `lu`, `qr`, `solve`, `inverse`, `det`, `trace`, `norm_frobenius`/`norm_1`/`norm_inf`, `matmul`, `matvec`, `tridiagonal_solve` — small, compile-time-sized, differentiable |
| `numax.fft` | `fft`, `ifft`, `circular_convolve`, over `Complex[Inner]` |
| `numax.interp` | `horner`, natural cubic splines, `chebyshev_fit`/`chebyshev_eval` |
| `numax.distributions` | pdf/cdf/quantile for normal, exponential, gamma, chi-square, beta, Student-t, F, Poisson, binomial |
| `numax.ode` | `rk4`, `rk4_system`, Dormand-Prince `dopri5` |
| `numax.optimize` | `newton_tol`, `brentq`, `bfgs` — iterate to a tolerance; BFGS takes an **exact** gradient from `Gradient`, not a finite difference |

```mojo
def cos_minus_x[U: FloatLike](x: U) -> U:
    return x.cos() + (-x)

var root = newton[f=cos_minus_x](Plain[dtype, width](1.0))  # no f' needed
```

`numax.linalg`'s matrices are small and compile-time-sized (registers, not
heap) — that's what makes `cholesky` differentiable at `Dual` (useful for a
Gaussian process or Kalman filter) and GPU-launchable, but it's the wrong
tool for a large system; use `max.linalg` past roughly 8x8, as each
function's docstring notes. Every iteration count in this layer is fixed
rather than tolerance-driven, so every kernel here also runs unmodified
inside a GPU thread — see [`docs/architecture.md`](docs/architecture.md) for
why (SIMD lanes can't branch independently, so adaptive convergence isn't
expressible).

## Array, statistics, I/O, and random

A NumPy-named surface layered over MAX's `TileTensor`, added only where MAX
ships no equivalent:

| Module | Functions |
|---|---|
| `numax.array` | `zeros`/`ones`/`full`/`empty`/`eye`/`linspace`/`logspace`/`arange`/`*_like`, `reshape`/`ravel`/`transpose`/`squeeze`/`stack`/`concatenate`/`split` |
| `numax.statistics` | `sum`/`prod`/`min`/`max`/`mean`/`median`/`mode`/`argmax`/`argmin`; `variance`/`stddev`/`cumsum` also work over any `FloatLike` |
| `numax.io` | binary `save`/`load`, `print_tensor` |
| `numax.random` | `uniform`, `normal`, `exponential`, `seed` |

Calling `variance`/`stddev`/`cumsum` at `Compensated` instead of `Plain`
recovers precision a long summation would otherwise lose — the one place
this surface and the composable-type trait meet.

[`docs/parity.md`](docs/parity.md) is the full disposition table: what numax
absorbs, what it routes to MAX, what it leaves out (`sort`/`argsort` as
`FloatLike` kernels, for the same fixed-iteration reason above) — and the
surveyed MAX API surface those decisions rest on. Worth knowing up front:
MAX ships one matrix decomposition and no forward FFT, so numax's own
kernels carry more of the mathematics than the "layer over MAX" framing
suggests.

### Two tiers

Everything above is **tier 1**: a fixed iteration count, no branching per
SIMD lane, and therefore launchable inside a GPU thread unmodified. That is
what the `FloatLike` spine buys.

`numax.optimize` and `numax.integrate` are **tier 2**: `Plain`-only,
host-side, and free to loop until they converge. Root finding to a
tolerance, adaptive quadrature and pivoting need that, and pretending
otherwise would either exclude them or quietly weaken the tier-1
guarantee. The tier is declared in every module
and function docstring, and tier 1 never calls tier 2.

The objective function is unaffected — it stays an ordinary `FloatLike`
kernel, which is exactly why `bfgs` can evaluate it at `Gradient` and get
every partial derivative exactly:

```mojo
def rosenbrock[U: FloatLike](v: Array[U, 2]) -> U:
    var a = U.one() + (-v[0])
    var b = v[1] + (-(v[0] * v[0]))
    return a * a + U.constant(100.0) * b * b

var result = bfgs[2, rosenbrock](start)   # exact gradient, no `jac` argument
```

A central difference cannot beat about `eps**(2/3)` relative accuracy no
matter how the step is picked — truncation and cancellation error pull in
opposite directions. Forward-mode AD has neither term.
`examples/advanced/optimize.mojo` sweeps the step size and prints both
error curves: the best finite difference lands at ~5e-10, AD at exactly 0.

## Tensors and GPU

`numax.tensor.map`/`reduce` drive a `FloatLike` kernel over MAX's
`TileTensor` — CPU or GPU, picked with one `gpu: Bool` compile-time parameter:

```mojo
from numax.tensor import map

def gaussian_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(Plain[dtype, w](x)).v

map[width = simd_width_of[dtype](), step=gaussian_step](xs, ys)          # CPU
ctx.enqueue_function[map[step=gaussian_step, gpu=True]](xs, ys, ...)    # GPU
```

`map_threaded` distributes the same kernel across CPU cores via
`max.algorithm.elementwise`; `reduce_axis`/`broadcast_op_axis` fold and
broadcast along any axis of any rank.

Shapes can be known at compile time or at runtime. `map` and `reduce` each
have two overloads under the same name, picked by `where` clauses that are
exact negations of each other: a comptime `row_major[n]()` tensor takes the
static path (and can be launched on GPU), a runtime `row_major(Coord(n))`
tensor takes the runtime path (CPU only, since a launch needs the extent in
the type). There is no second tensor type — both are `TileTensor`, which is
what every MAX kernel accepts, and the distinction lives in the layout where
MAX already put it.

Every `FloatLike` conformer here is
built from plain `SIMD` fields with no pointers or allocations of its own, so
every kernel — including `Dual` and `Compensated` — runs inside a GPU thread
with no changes. See [`docs/architecture.md`](docs/architecture.md) for the
full tensor-layer design and `numax/tensor.mojo`'s module docstring for the
API details.

## Performance and accuracy

Measured on an Apple M3 Pro (~150 GB/s memory bandwidth) on
`gaussian(x) = exp(-x^2)` over `float32`, which is memory-bandwidth-bound at
every size tested (an identity copy runs at the same speed, so the `exp` is
free). The numbers below are that machine's; the library itself is not
Apple-specific, and the same suites run on Linux/CUDA. At 67M elements:

| | numax CPU (`map_threaded`) | numax GPU (amortized) | NumPy (CPU) | MLX (GPU, per-call) | torch.compile (GPU, amortized) |
|---|---|---|---|---|---|
| M elem/s | 9,026 | 15,755 | 493 | 4,866 | 14,380 |

`numax`'s GPU path reaches 84% of the hardware bandwidth ceiling, on par
with `torch.compile` on the same Metal device. The GPU path is written
against `DeviceContext` rather than any one backend and is verified on CUDA
as well; `pixi run bench-roofline` reports percent-of-peak for whichever
device it finds. Full sweeps across six sizes,
both CPU walks, and every cross-language baseline (NumPy, MLX, PyTorch,
Rust `thermite`) are in [`docs/performance.md`](docs/performance.md) and
[`bench/README.md`](bench/README.md).

Every approximation documents an error bound from the literature it came
from; `pixi run accuracy` checks that this implementation delivers it,
against [mpmath](https://mpmath.org/) references at 50 decimal digits,
checked in so the harness needs no Python at run time. The bounds hold
(e.g. `erf`'s A&S 7.1.26 bound of ~1.5e-7 measures 1.38e-07). One caveat
applies library-wide: Mojo's `std.math` `exp`/`log`/`erf` are not correctly
rounded at `float64`, so every `numax` function built on them inherits that
floor at `float64` — invisible at the `float32` this library normally runs
at. Full table and the defect writeup: [`bench/accuracy/README.md`](bench/accuracy/README.md).

## Examples

Tiered by complexity under `examples/basic/`, `examples/intermediate/`, and
`examples/advanced/` — see [`examples/README.md`](examples/README.md) for the
full index. `pixi run examples-cpu` runs everything that doesn't need a GPU;
`pixi run examples` includes the GPU ones (needs a real GPU -- Metal or
CUDA, whichever `DeviceContext` finds).

## Testing and benchmarks

```bash
pixi run tests           # full suite -- see pixi.toml for individual suites
pixi run bench           # numax.tensor.map vs. a hand-rolled raw-SIMD loop
pixi run bench-gpu       # CPU vs. GPU across a size sweep
pixi run bench-numpy     # cross-language: NumPy
pixi run bench-mlx       # cross-language: MLX (CPU + GPU; Metal or CUDA)
pixi run bench-torch     # cross-language: PyTorch (eager + compile, CPU + GPU)
pixi run bench-thermite  # cross-language: Rust thermite (NEON or AVX2)
pixi run accuracy        # max error per function vs. checked-in mpmath refs
```

`pixi run tests` and `pixi run examples-cpu` run in CI on every push; GPU
examples and benchmarks are a local-only check (no GPU on the CI runner).

## Documentation

- **This README** — the tour: what's here, quick start, performance and
  accuracy headlines.
- **[`docs/architecture.md`](docs/architecture.md)** — the design: the
  trait, the fixed-iteration invariant, the tensor/GPU layer, the parity
  surface.
- **[`docs/performance.md`](docs/performance.md)** — the full performance
  writeup, with [`bench/README.md`](bench/README.md) for methodology.
- **API reference** — generated by [mojodoc](https://github.com/ehsanmok/mojodoc)
  on every push to `main`, published to <https://ehsanmok.github.io/numax/>.
  Locally: `pixi run -e dev docs` (opens in a browser) or `docs-build` (writes
  `target/doc/`).

## License

[MIT](LICENSE)
