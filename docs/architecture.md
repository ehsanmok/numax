# Architecture

> Companion to the top-level [README](../README.md). For the design intent
> behind the parity surface and the rename from `ember` to `numax`, see
> [`.cursor/rules/design-v0.1.mdc`](../.cursor/rules/design-v0.1.mdc) and
> [`.cursor/rules/parity.mdc`](../.cursor/rules/parity.mdc). For the
> per-library design history (Tracks A-E: special functions, conformers,
> algorithms, linalg/FFT, performance layer), see
> [`.cursor/rules/strategy.mdc`](../.cursor/rules/strategy.mdc).

`numax` is built on two co-equal axes:

1. **Composable type layer** -- one kernel written against `FloatLike`
   produces plain SIMD, autodiff, extra precision, or complex arithmetic
   depending on the conformer you instantiate it with.
2. **NumPy/SciPy parity on Mojo, MAX-first** -- MAX's existing
   infrastructure (`TileTensor`, `max.linalg`, `max.algorithm`,
   `max.random`) is the substrate for array-shaped work; `numax` adds
   the composable-type layer and the NumPy-named entry surface over it.

## The trait: `FloatLike`

The trait every kernel in `numax` is written against lives in
[`numax/numeric.mojo`](../numax/numeric.mojo). It's deliberately small --
enough to build a vectorized special-function library on, and no more.
Every method added is one more thing every future conformer must
implement.

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
```

The trait grew only when a kernel genuinely needed a new operation
(`sqrt` was held out for a long time on the "one call site isn't reason
enough" rule until the call sites multiplied AND the `exp(0.5*ln(x))`
workaround turned out to be inaccurate, not merely verbose -- see
`findings.mdc`).

## The conformers

| Type | What you get |
|---|---|
| `Plain[dtype, width]` | Thin wrapper around `SIMD[dtype, width]` -- the bridge that lets ordinary SIMD conform to `FloatLike`. |
| `Dual[Inner: FloatLike]` | Forward-mode AD: value paired with derivative, propagated by the chain rule. Nests (`Dual[Dual[Plain[...]]]`) for second derivatives. |
| `Compensated[dtype, width]` | Double-double arithmetic -- `value` + `error`, recovering rounding error `dtype` alone would discard. |
| `Decimal[width, scale]` | Exact base-10 fixed-point: `0.1 + 0.2 == 0.3` exactly. Different problem than `Compensated`, not a replacement. |
| `Complex[Inner: FloatLike]` | Complex number over any other conformer -- `Complex[Dual[Plain[...]]]` differentiates holomorphically for free. |
| `Gradient[Inner: FloatLike, n_vars: Int]` | `Dual`'s multi-input counterpart: full gradient vector from one call. Nests with `Dual` for Hessian columns. |
| `Interval[Inner: FloatLike]` | Bounds instead of a value: run a kernel over `[lo, hi]` and get the range. Not a rigorous enclosure (no directed rounding on SIMD/GPU); `inflate` is the manual widening. |

Each conformer's module docstring carries its design rationale and
documented limitations; `tests/` has the numerical margins.

## The fixed-iteration invariant

Stated once in `strategy.mdc` and inherited everywhere: **no kernel in
`numax` runs a data-dependent number of iterations, and no kernel
branches per lane.** A `Self` may hold a SIMD vector whose lanes disagree
about which branch they want or whether a series has converged, and there
is no `select`-like primitive on `FloatLike` to resolve that per lane.
So a fixed amount of uniform work is done instead, and per-lane selection
is an arithmetic `0`/`1` blend built from `copysign` (see
`max_of`/`min_of`/`ge_indicator`/`blend` in `numax/numeric.mojo`).

The cost is accuracy at the extremes of a domain (documented per
function); the benefit is that **every kernel here is launchable inside
a GPU thread unmodified**. Adaptive-tolerance variants are deliberately
out of scope, not missing.

## The `numax.tensor` layer

[`numax/tensor.mojo`](../numax/tensor.mojo) drives a `FloatLike` kernel
across MAX's `TileTensor` -- the same tensor type used for both CPU- and
GPU-resident data -- instead of a hand-rolled pointer loop. CPU or GPU
is picked with one `gpu: Bool` compile-time parameter rather than two
differently-named functions.

```mojo
from numax.tensor import map

def gaussian_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(Plain[dtype, w](x)).v

# CPU: native SIMD width via TileTensor.vectorize(), scalar tail for the rest.
map[width = simd_width_of[dtype](), step=gaussian_step](xs, ys)

# GPU: the body of one thread, launched once per element.
ctx.enqueue_function[map[step=gaussian_step, gpu=True]](
    xs, ys, grid_dim=..., block_dim=...
)
```

Primitives: `map` (unary + binary), `reduce`, `reduce_block_gpu`,
`reduce_rows`, `broadcast_op_rows`, `reduce_axis` (any rank, any axis),
`broadcast_op_axis`. `map_threaded` distributes the same `step` across
CPU cores via `max.algorithm.elementwise[target="cpu"]`.

`step`/`combine` are plain, non-capturing functions passed as
compile-time parameters on both the CPU and GPU paths -- required for
the GPU path to work through `DeviceContext.enqueue_function`, and kept
the same shape on CPU so `map` has one signature instead of two.

## GPU

A GPU kernel in Mojo is a plain function; `DevicePassable` only
constrains values crossing the host/device boundary as kernel arguments,
not what a kernel builds internally. Every `FloatLike` conformer here
is built entirely from `SIMD` fields with no pointers or allocations of
its own, so all of them -- including `Compensated`, once its `exp()`
coefficients moved to compile-time `dtype`-native constants instead of
a runtime `float64` table -- run inside a kernel body unchanged.

`examples/gaussian_gpu.mojo` imports the exact same `gaussian`, `Plain`,
and `Dual` the CPU example does and drives them with `map[gpu=True]`
instead of `map[gpu=False]`. **No `FloatLike` kernel needed an edit to
become GPU-launchable.** That's the benefit the fixed-iteration invariant
buys.

## What's not here yet (Track F)

`numax` does not yet ship a NumPy-named array surface
(`zeros`/`ones`/`arange`/`reshape`/...), a statistics module
(`mean`/`var`/`std`), `numax.io`, or `numax.random`. The dispositions for
those -- which to absorb, which to route to MAX, which to leave out --
are in [`.cursor/rules/parity.mdc`](../.cursor/rules/parity.mdc) and the
phased roadmap in
[`.cursor/rules/design-v0.1.mdc`](../.cursor/rules/design-v0.1.mdc)
points to them. The spine (Tracks A-E: special functions, conformers,
algorithms, linalg/FFT, performance layer) is what's shipped today.
