# Architecture

> Companion to the top-level [README](../README.md). For what numax absorbs
> from NumPy/SciPy and what it deliberately leaves out -- with the surveyed
> MAX API surface those decisions rest on -- see
> [`parity.md`](parity.md).

`numax` is built on two co-equal axes:

1. **Composable type layer** -- one kernel written against `FloatLike`
   produces plain SIMD, autodiff, extra precision, or complex arithmetic
   depending on the conformer you instantiate it with.
2. **NumPy/SciPy parity on Mojo, MAX-first** -- MAX's existing
   infrastructure (`TileTensor`, the top-level `linalg` and `nn` roots,
   `max.algorithm`, `max.gpu`) is the substrate for array-shaped work;
   `numax` adds
   the composable-type layer and the NumPy-named entry surface over it.

## Package layout

Subpackages follow NumPy/SciPy naming, so a NumPy or SciPy import has an
obvious counterpart. Each one re-exports its public surface, and
`numax/__init__.mojo` re-exports all of them, so `from numax import ...`
reaches everything flat and `from numax.linalg import ...` reaches one
subsystem.

```
numax/
  core/         FloatLike + conformers (plain, dual, gradient, compensated,
                decimal, interval, complex), the tensor engine (tensor,
                array), and the elementwise/ops/logic/sorting surface
  linalg/       factorizations, solves, norms, eigenvalues
  optimize/     minimization (optimize) and scalar root finding (solve)
  integrate/    quadrature, fixed-step and adaptive ODE, quad/solve_ivp
  interpolate/  polynomial and spline interpolation
  special/      erf, gamma, beta, bessel, elliptic, lambertw, orthogonal
                polynomials, activations
  stats/        descriptive statistics, distributions, random sampling
  fft/          discrete Fourier transforms over Complex
  signal/       convolution, correlation, windows
  io/           the NMX1 binary format and tensor printing
```

Dependencies run one way: `core` depends on nothing else in `numax`, every
other subpackage depends on `core`, and the few cross-subpackage edges are
deliberate (`stats` uses `special`'s incomplete gamma and beta,
`integrate` uses `special`'s Legendre roots and `optimize`'s Newton solver,
`interpolate` uses `linalg`'s tridiagonal solve).

## The trait: `FloatLike`

The trait every kernel in `numax` is written against lives in
[`numax/core/numeric.mojo`](../numax/core/numeric.mojo). It's deliberately small --
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
    def __sub__(self, rhs: Self) -> Self:     # the one with a body:
        return self + (-rhs)                  # `self + (-rhs)`
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

The trait grew only when a kernel genuinely needed a new operation
(`sqrt` was held out for a long time on the "one call site isn't reason
enough" rule until the call sites multiplied AND the `exp(0.5*ln(x))`
workaround turned out to be inaccurate, not merely verbose).

`__sub__` is the exception that costs nothing: it ships with a body rather
than `...`, so it is inherited rather than implemented and no conformer
grew a method. It replaced the `a + (-b)` spelling at 191 call sites, one of
them `Interval.width`, whose docstring said `hi - lo` while its body could
not. `__pow__` and comparisons stay out -- powers genuinely want
`exp(k*ln(x))` here, and per-lane comparison is what the branchless
`max_of`/`min_of`/`ge_indicator`/`blend` helpers exist to avoid.

### Two tiers

The invariant above is what makes every kernel here GPU-launchable, and it
is absolute for `FloatLike`-generic code. It also rules out a large part of
SciPy: root finding to a tolerance, adaptive quadrature, adaptive ODE step
control, pivoting, and sorting all need to branch on data or to iterate
until something converges.

Rather than erode the rule one exception at a time, numax states two tiers
and declares which one every function is in:

| | types | iteration | guarantee |
|---|---|---|---|
| **Tier 1** | `FloatLike`-generic | fixed count, no per-lane branching | launches inside a GPU thread unmodified |
| **Tier 2** | `Plain`-only, host | may loop to a tolerance, may branch on data | none about GPU |

Everything shipping today is tier 1 -- the special functions, `rk4`,
`dopri5` (fixed-step), `gauss_legendre`, `lambertw`'s fixed 20 Halley
iterations, `Compensated.ln()`'s 3 fixed Newton refinements, `gammainc`'s
fixed 100-term series, `numax.optimize`'s fixed-iteration `newton`/`halley`/
`bisection`. `numax.stats.median` and `mode` are the nearest thing to
an exception, and they are not one: they reach MAX's own sort, which has
data-dependent control flow, from `Plain`-only code. The rule governs what
numax writes *inside the trait*.

Three rules govern the boundary. The tier is **declared**, in the module
docstring and in the function's own, so a reader at a call site never has
to audit a body to learn whether it can be launched. **Tier 1 never calls
tier 2** -- the dependency runs one way, because a tier-1 kernel that
reaches a convergence loop silently loses the property that justifies the
whole spine. And where both make sense, **both ship, cross-referenced**:
a fixed-iteration `newton` and a converge-to-tolerance one are siblings,
not replacements.


## The conformers

| Type | What you get |
|---|---|
| `Plain[dtype, width]` | Thin wrapper around `SIMD[dtype, width]` -- the bridge that lets ordinary SIMD conform to `FloatLike`. |
| `Dual[Inner: FloatLike]` | Forward-mode AD: value paired with derivative, propagated by the chain rule. Nests (`Dual[Dual[Plain[...]]]`) for second derivatives. |
| `Compensated[dtype, width]` | Double-double arithmetic -- `value` + `error`, recovering rounding error `dtype` alone would discard. |
| `Decimal[width, scale]` | Exact base-10 fixed-point: `0.1 + 0.2 == 0.3` exactly. Different problem than `Compensated`, not a replacement. |
| `Complex[Inner: FloatLike]` | Complex number over any other conformer -- `Complex[Dual[Plain[...]]]` differentiates holomorphically for free. |
| `Gradient[Inner: FloatLike, n_vars: Int]` | `Dual`'s multi-input counterpart: full gradient vector from one call. Nests with `Dual` for Hessian columns. |
| `Interval[Inner: FloatLike]` | Bounds instead of a value: run a kernel over `[lo, hi]` and get the range. Not a rigorous enclosure (no directed rounding on SIMD/GPU); `inflate` is the manual widening. `sin`/`cos` detect an enclosed peak or trough via `floor` and return a tight bound rather than the trivial `[-1, 1]`. |

Each conformer's module docstring carries its design rationale and
documented limitations; `tests/` has the numerical margins.

## The fixed-iteration invariant

Inherited everywhere: **no `FloatLike`-generic kernel in `numax` runs a
data-dependent number of iterations, and none branches per lane.** A `Self` may hold a SIMD vector whose lanes disagree
about which branch they want or whether a series has converged, and there
is no `select`-like primitive on `FloatLike` to resolve that per lane.
So a fixed amount of uniform work is done instead, and per-lane selection
is an arithmetic `0`/`1` blend built from `copysign` (see
`max_of`/`min_of`/`ge_indicator`/`blend` in `numax/core/numeric.mojo`).

The cost is accuracy at the extremes of a domain (documented per
function); the benefit is that **every kernel here is launchable inside
a GPU thread unmodified**. Adaptive-tolerance variants are deliberately
out of scope, not missing.

## The `numax.core.tensor` layer

[`numax/core/tensor.mojo`](../numax/core/tensor.mojo) drives a `FloatLike` kernel
across MAX's `TileTensor` -- the same tensor type used for both CPU- and
GPU-resident data -- instead of a hand-rolled pointer loop. CPU or GPU
is picked with one `gpu: Bool` compile-time parameter rather than two
differently-named functions.

```mojo
from numax.core.tensor import map

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

## The NumPy/SciPy parity surface

Modules that fill NumPy/SciPy-shaped gaps, each picked because MAX ships no
usable equivalent. [`parity.md`](parity.md) has the disposition table and the
survey of what MAX does ship.

- **`numax.core.array`** — creation and manipulation over a `Tensor` that owns its
  storage, because a bare `TileTensor` is a view and dangles once the function
  that built it returns. The storage is a MAX `DeviceBuffer`, so the
  `DeviceContext` decides host or device memory and `.view()` yields the same
  `TileTensor` either way; the context is the last argument of every factory
  and optional, so a host-only program never names one. Element access is
  `a[i]` flat or `a[r, c]` on a rank-2 tensor, and in bulk goes through
  `to_host`/`copy_from_host`: on CUDA `unsafe_ptr()` returns a device pointer
  and a host read segfaults. `Tensor` is `Writable`, so `print(a)` works.
  The view seam is cheap downward and a copy upward: `.view()` hands out a
  pointer plus a layout, `Tensor.from_view` copies, and no constructor takes
  a `TileTensor` at all, because a view owns nothing to adopt.
  `to_array`/`to_tensor` are the seam to the `Array[T, n]` half of the
  library.
- **`numax.core.ops`, `numax.core.elementwise`, `numax.core.logic`** — arithmetic and
  operators on `Tensor`, the elementwise math surface, and comparisons
  returning `Tensor[DType.bool]`. `Plain`-only, tier 2.
- **`numax.stats`** — whole-tensor reductions, every one taking a `Tensor`,
  with `argmax`/`argmin` routed to `nn.argmaxmin`, and the distributions as
  nine `scipy.stats`-shaped namespaces (`norm.cdf`, `chi2.ppf`, ...). `mean`/`variance`/`stddev`/`cumsum` also have a
  `FloatLike`-generic form over `List[T]`, so calling them at `Compensated`
  recovers precision a long summation loses at `Plain` — the one place this
  surface and the composable-type spine meet.
- **`numax.linalg`** — small dense linear algebra, `FloatLike`-generic. The
  point is differentiability, not speed: MAX's `matmul` and `qr_factorization`
  are monomorphic in a raw `dtype`, so no `Dual` passes through them.
- **`numax.io`, `numax.stats.random`** — `nmx.save`/`nmx.load`, a binary format
  of numax's own since MAX ships no array I/O, plus `numpy.save`/`numpy.load`
  for `.npy` interchange, so a program ported from NumPy can ingest the files
  it already has; and sampling over `std.random` on the host, with `Generator`
  for a named stream and no `Random[FloatLike]` conformer, because RNG is not
  differentiable.

## Static and runtime shapes

`numax.core.tensor`'s walks originally required `all_dims_known`, because they
flatten with `TileTensor.coalesce()` and `coalesce()` is itself constrained
to statically-shaped storage. That is what a GPU launch needs: a kernel
reaching `enqueue_function` must have its entire type resolved before the
launch, and a runtime extent is not part of the type.

It is also why nothing could express a reshape to a computed shape, a
boolean mask, or any operation whose output extent depends on input
*values*. `row_major(Coord(3, 4))` builds exactly that tensor --
`is_row_major=True`, `all_dims_known=False` -- and `coalesce()` rejects it.

`map` (unary and binary) and `reduce` therefore each have two overloads
under the same name, selected by `where` clauses that are exact negations
of each other:

| | shape | flattening | GPU |
|---|---|---|---|
| static overload | `row_major[n]()`, comptime | `coalesce()` | yes, via `gpu=True` |
| runtime overload | `row_major(Coord(n))`, runtime | rank-1 layout over the same pointer | no |

The runtime overload flattens by *construction* rather than by
`coalesce()`: a row-major tensor's elements are already contiguous, so a
rank-1 `row_major(Coord(n))` layout over the same pointer addresses the
same memory in the same order. That is all `coalesce()` does for a
row-major input -- it just insists on proving the shape at compile time
first.

**There is no second tensor type.** Both overloads take a `TileTensor`,
which is the type every MAX kernel accepts; a parallel numax array type
would have to be converted at every boundary into MAX. The
compile-time/runtime distinction lives in the *layout*, where MAX already
put it, not in a numax-owned wrapper.
