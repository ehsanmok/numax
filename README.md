# ember

[![CI](https://github.com/ehsanmok/ember/actions/workflows/ci.yml/badge.svg)](https://github.com/ehsanmok/ember/actions/workflows/ci.yml)
[![Mojo 1.0.0](https://img.shields.io/badge/Mojo-1.0.0-orange)](https://docs.modular.com/mojo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ember lets you write a numeric kernel once, against a trait, and get three
different things out of it depending on what you call it with:

```mojo
def gaussian[T: FloatLike](x: T) -> T:
    return (-(x * x)).exp()
```

- `gaussian(Plain(x))` — ordinary `SIMD[dtype, width]`, at hardware speed.
- `gaussian(Dual(x, seed))` — the value **and** its derivative, via forward-mode
  automatic differentiation. No second formula to write or keep in sync.
- `gaussian(Compensated(x, 0))` — the value at roughly double `dtype`'s
  precision, recovered from the rounding error ordinary arithmetic would
  otherwise discard.

Same function body, no edits between them. `ember` ships a growing library of
kernels built the same way — activations (`gaussian`, `sigmoid`, `swish`,
`tanh`, `relu`, `gelu`, `softmax`) and special functions (`erf`, `gamma`,
`bessel_j0`, `lambertw`, ...) — each one differentiable and extra-precise for
free.

Mojo's `SIMD[dtype, width]` is already a native, portable vector type, so
`ember` doesn't try to add another cross-ISA vector abstraction on top of it.
What it adds is the composable numeric-type layer: swap `Plain` for `Dual` or
`Compensated` and the kernel adapts with no changes of its own.

This is a young, experimental library. APIs may change.

## Install

Ember targets Mojo `1.0.0` and is managed with [pixi](https://pixi.sh). Clone
this repo, then:

```bash
pixi install
pixi run examples-cpu   # run the CPU-only examples
pixi run tests          # run the full test suite
pixi run bench          # compare ember.tensor.map_simd against raw SIMD
```

`ember` isn't published anywhere yet, so from another project the simplest
path is to check it out alongside your code and pass `-I <path-to-ember>` (or
`-I .` if you're running from this directory) so the compiler can resolve the
package:

```mojo
from ember import Compensated, Decimal, Dual, FloatLike, Plain, gaussian, sigmoid, swish, tanh
from ember import erf, gamma, bessel_j0, lambertw  # special functions, see below
```

```bash
mojo -I /path/to/ember your_script.mojo
```

## The trait

`FloatLike` is deliberately small — just what `special`'s kernels need, and
no more:

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
    @staticmethod
    def constant(v: Float64) -> Self: ...
    def abs(self) -> Self: ...
    def copysign(self, sign_source: Self) -> Self: ...
```

Adding an operation here is a real cost: every conforming type has to
implement it. Grow the trait only when a kernel genuinely needs the
operation.

## The three types

| Type | Fields | What instantiating a kernel with it gets you |
|---|---|---|
| `Plain[dtype, width]` | `v` | A thin wrapper around `SIMD[dtype, width]` — Mojo can't retroactively make `SIMD` conform to `FloatLike`, so this is the bridge. |
| `Dual[Inner: FloatLike]` | `value`, `deriv` | Forward-mode automatic differentiation. Seed `deriv=Inner.one()` and the kernel's return value carries its derivative alongside its value, propagated by the chain rule through every operation. `Inner` is any `FloatLike` — usually `Plain[dtype, width]`, giving the classic single-direction dual number — but `Dual` itself conforms to `FloatLike`, so `Dual[Dual[Plain[dtype, width]]]` nests: seeded correctly, it carries a value, first derivative, *and* second derivative through one kernel call (see `test_dual.mojo`'s nested-`Dual` tests). Still single-variable — no `Dual<V, N>` gradient across multiple inputs. |
| `Compensated[dtype, width]` | `value`, `error` | Double-double arithmetic. Every operation is an error-free transformation (`two_sum`/`two_prod`, built on `fma`) that recovers the rounding error `dtype` alone would discard and folds it into `error`. `value + error` (as `Float64`, say) lands far closer to a true result than `value` alone — see `test_compensated.mojo` for the exact margins. |
| `Decimal[width, scale]` | `raw` | Base-10 fixed-point arithmetic: `raw` is the value times `10^scale`, stored as an exact integer, so `0.1 + 0.2` is exactly `0.3` — never `0.30000000000000004`. A different problem than `Compensated` solves (base-10 exactness vs. extra *binary* significand bits), so it's a fourth conformer next to it, not a replacement. `exp()`/`ln()` use fixed-iteration series/Newton's method for the same GPU-motivated reason `ember.gamma`'s kernels do — see `ember/decimal.mojo`'s docstring for its scoped magnitude/precision trade-offs. |

## Special functions

Every special function is written once against `FloatLike`, one file per
family (matching the project's `plain.mojo`/`dual.mojo`/`compensated.mojo`
convention):

**`ember.special`** — activations:

- `gaussian(x)` — the unnormalized Gaussian bump, `exp(-x^2)`.
- `sigmoid(x)` — the logistic function, `1 / (1 + exp(-x))`.
- `swish(x)` — `x * sigmoid(x)` (SiLU).
- `tanh(x)` — hyperbolic tangent, via `(exp(2x) - 1) / (exp(2x) + 1)`.
- `relu(x)` — `max(x, 0)`, via `(x + |x|) / 2` so it's branch-free.
- `leaky_relu(x, alpha)` — `x` for `x >= 0`, `alpha * x` otherwise, the same
  branch-free shape as `relu`.
- `gelu(x)` — the GELU activation's `tanh` approximation, reusing `tanh`
  above rather than `erf` directly.

Every one of them differentiates and gains precision for free, by virtue of
being written against `FloatLike` rather than against `Plain` or `SIMD`
directly — see `examples/activations.mojo`.

`softmax` is the one exception in `ember.special`, and isn't
`FloatLike`-generic: computing `softmax(xs)[i]` needs the whole row `xs`
belongs to (its max, for numerical stability, and its sum of exponentials),
not just `xs[i]`, so it can't be a single-element kernel that `map` drives.
It's a small CPU-side orchestration of `ember.tensor`'s `reduce_rows` and
`broadcast_op_rows` instead — see the next section and
`examples/softmax.mojo`.

**`ember.erf`** — `erf(x)` / `erfc(x)`, the error function and its
complement, via the Abramowitz & Stegun 7.1.26 rational approximation (max
absolute error ~1.5e-7), extended to negative `x` with `copysign`.

**`ember.gamma`** — `gamma(x)` / `lgamma(x)`, via a 9-term Lanczos
approximation (`x > 0` — no reflection formula, since that needs `sin`,
which isn't in `FloatLike`); `gammainc(a, x)` / `gammaincc(a, x)`, the
regularized lower/upper incomplete gamma functions, via a fixed 100-term
series (most accurate when `x` isn't much larger than `a`; no
continued-fraction branch for the opposite regime).

**`ember.bessel`** — `bessel_j0(x)`, the order-zero Bessel function of the
first kind, via the Abramowitz & Stegun 9.4.1 polynomial (valid for
`|x| <= 3`, max absolute error ~5e-8; no asymptotic branch for `|x| > 3`
yet).

**`ember.lambertw`** — `lambertw(x)`, the principal branch of the Lambert W
function (`x >= 0`), via Halley's method seeded from `ln(1 + x)`.

`gamma`/`gammainc`/`lambertw` all use a **fixed** number of terms or
iterations rather than a data-dependent convergence check — a GPU thread
can't branch per-lane on "has this series converged yet" the way a scalar
loop could, so each trades adaptive precision for a bounded, uniform amount
of work. See `examples/special_functions.mojo`, which also exercises their
derivatives via `Dual` (`Gamma'(x) = Gamma(x) * digamma(x)` and
`dW/dx = W(x) / (x*(1+W(x)))` fall out with no extra code).

## Tensors

`ember.tensor` drives a `FloatLike` kernel over MAX's `TileTensor` — the same
tensor type `TileTensor` uses for both CPU- and GPU-resident data — instead
of a hand-rolled pointer loop:

```mojo
from ember.tensor import map  # CPU-backed TileTensor, one fixed width
from ember.tensor import map_simd  # CPU-backed TileTensor, native SIMD width + scalar tail
from ember.tensor import map_gpu  # DeviceBuffer-backed TileTensor, launched via enqueue_function
```

`map` and `map_gpu` take the same three ingredients: `kernel` (a
`FloatLike`-generic function, e.g. `gaussian[Plain[dtype, 1]]`), `wrap` (raw
`SIMD` in, `T` out), and `unwrap` (`T` in, raw `SIMD` out — the caller's
choice of which field of `T` is "the answer", since `Dual` and `Compensated`
carry two). `map` walks a CPU tensor with a loop at one fixed `width`;
`map_gpu` is the body of one GPU thread, launched once per element via
`DeviceContext.enqueue_function` — GPU parallelism comes from thread count,
not per-thread SIMD registers, so it stays one-element-per-thread rather than
vectorizing.

`map_simd` is the CPU entry point that gets real SIMD width without the
caller doing the arithmetic: give it one `step` function generic over its own
width (`def[w: Int](SIMD[dtype, w]) thin -> SIMD[dtype, w]`, typically `wrap`
→ kernel → `unwrap` composed inline), and a `width` (usually
`simd_width_of[dtype]()`), and it walks the tensor in non-overlapping
`width`-wide groups via `TileTensor.vectorize()`, then finishes off whatever
doesn't divide evenly with a scalar (`width=1`) tail loop using the same
`step`:

```mojo
def gaussian_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(Plain[dtype, w](x)).v

map_simd[width = simd_width_of[dtype](), step=gaussian_step](xs, ys)
```

`ember.tensor` also has `reduce`/`reduce_block_gpu` (fold a whole tensor down
to one value, e.g. a sum or max) and the 2D row-wise pair
`reduce_rows`/`reduce_rows_gpu` + `broadcast_op_rows`/`broadcast_op_rows_gpu`
(fold each row to one value; combine a 2D tensor with a per-row value) --
see `examples/softmax.mojo`, which is what these two exist for.

## Examples

- `examples/gaussian.mojo` — the `Plain`/`Dual`/`Compensated` three-ways demo
  over 4096 points, walked with `ember.tensor.map_simd` at the CPU's native
  SIMD width, with a max-error check against a `Float64` reference.
- `examples/activations.mojo` — `sigmoid`, `swish`, `tanh`, `relu`,
  `leaky_relu`, and `gelu` differentiated via `Dual`, checked against their
  closed-form derivatives.
- `examples/gaussian_gpu.mojo` — the same `gaussian` kernel and `Plain`/
  `Dual`/`Compensated` types as `examples/gaussian.mojo`, run on the GPU via
  `ember.tensor.map_gpu` and `DeviceContext`. All three are built entirely
  from `SIMD` lanes already on the device side, so no code changes at all
  versus the CPU example.
- `examples/softmax.mojo` — row-wise softmax over a 2D `TileTensor`, on CPU
  via `ember.special.softmax` and on GPU via the same
  `reduce_rows_gpu`/`broadcast_op_rows_gpu` primitives hand-launched as a
  four-kernel sequence, checked against each other and against each row
  summing to 1.
- `examples/special_functions.mojo` — `gamma`/`lgamma`, `gammainc`,
  `bessel_j0`, and `lambertw`, run over `Dual` to get derivatives
  (`Gamma'(x) = Gamma(x)*digamma(x)`, `dW/dx = W(x)/(x*(1+W(x)))`) with no
  extra code in either kernel.

## GPU

A GPU kernel in Mojo is a plain function; the `DevicePassable` trait only
constrains values crossing the host/device boundary as a kernel's arguments
(buffers, pointers, scalars, and `TileTensor` itself), not whatever a kernel
builds from them internally. `Plain` and `Dual` hold nothing but `SIMD`
fields with no pointers or allocations of their own, so neither needs any
change to run inside a kernel body — `examples/gaussian_gpu.mojo` imports the
exact same `gaussian`, `Plain`, and `Dual` the CPU example does, and drives
them with `ember.tensor.map_gpu` instead of `map`.

`Compensated` used to be the exception, for a mundane reason rather than a
fundamental one: its `exp()` built its coefficient table as an
`Array[Float64, ...]` at runtime, and Apple's Metal backend has no float64
support at all, so that kernel failed to compile for GPU regardless of
`Self.dtype`. It's fixed now — each coefficient is split into a `dtype`-
native hi/lo pair at compile time (see `_split_f64` in
`ember/compensated.mojo`), so the only float64 arithmetic left happens in
the Mojo compiler itself, never in generated device code.

## What's not here (yet)

- **Runtime ISA dispatch.** Mojo compiles for one target rather than carrying
  every backend in a single binary and picking at runtime, so `ember` doesn't
  attempt that kind of dispatch.
- **A wider special-function library.** `gamma`/`gammainc` are scoped to
  `x > 0`, `bessel_j0` to `|x| <= 3`, and `lambertw` to `x >= 0` -- reflection
  formulas, the asymptotic Bessel regime, other branches/orders, and
  elliptic integrals aren't implemented yet (see `.cursor/rules/strategy.mdc`
  for why each is scoped the way it is).
- **Multi-variable duals / complex numbers.** `Dual[Inner]` nests for
  higher-order *single-variable* derivatives (`Dual[Dual[Plain[...]]]` gets a
  second derivative), but there's no multi-input `Dual<V, N>` gradient or
  `Complex<V>` yet.
- **Multi-dimensional `ember.tensor` walks.** `map`/`map_simd`/`map_gpu` are
  rank-1 — flatten a multi-dimensional `TileTensor` with `.coalesce()` first
  if its storage is contiguous.

## Testing

```bash
pixi run test-dual
pixi run test-compensated
pixi run test-special
pixi run test-tensor
pixi run test-tensor-reduce
pixi run test-activations
pixi run test-erf
pixi run test-gamma
pixi run test-bessel
pixi run test-lambertw
pixi run test-decimal
```

Each test file uses Mojo's `TestSuite` for automatic discovery — see
`tests/` for the full suite, including the numerical margins that back the
precision claims above. `pixi run tests` runs all of them in sequence, and
is what CI runs on every push (see `.github/workflows/ci.yml`) alongside the
CPU examples (`pixi run examples-cpu`) — the GPU examples need real Metal
hardware, so they're a local-only check for now.

## Benchmarks

```bash
pixi run bench
```

`benchmarks/bench_tensor_map.mojo` runs the same `gaussian(x) = exp(-x^2)`
sweep two ways — a hand-rolled `std.algorithm.functional.vectorize` loop over
raw `SIMD`, and `ember.tensor.map_simd` over `Plain` — and reports both
timings side by side. The trait-and-`TileTensor` layer costs nothing
measurable over the raw loop it's standing in for; that's the whole basis for
"swap `Plain` for `Dual` or `Compensated` for free."

## License

MIT.
