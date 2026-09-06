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

A numerical computing library built on
[MAX](https://max.modular.com/docs/): special functions, linear algebra,
quadrature, ODE solvers, FFTs, distributions, and a NumPy-named array surface,
written in [Mojo](https://mojolang.org) against MAX's `TileTensor` and kernel
infrastructure. Data lives in a MAX `DeviceBuffer`, so the `DeviceContext` you
pass decides host or device: the same kernel, any accelerator, unmodified.

Here is what that buys you. Take the
[quantum harmonic oscillator](https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator),
whose stationary states solve the time-independent Schrödinger equation

$$-\tfrac{1}{2}\psi'' + \tfrac{1}{2}\omega^2 x^2 \psi = E\psi,$$

in units where $\hbar = m = 1$, giving a ground-state energy $E_0 = \omega/2$.
Discretized on a grid of $n$ points, the left-hand side is a symmetric matrix
and $E_0$ is its lowest eigenvalue, so the whole physics problem is one call to
`eigh`:

```mojo
from max.gpu.host import DeviceContext
from numax.core.numeric import min_of
from numax.core.tensor import map
from numax.prelude import *

comptime P = Plain[f64]
comptime n = 24                                     # grid points over [-4, 4]
comptime dx = 8.0 / (n - 1)
comptime kinetic = 1.0 / (dx * dx)
comptime Sweep = Shaped[f64, 256].LayoutType

# Written once against `FloatLike`: no dtype, no device, no derivative rule.
def ground_energy[T: FloatLike](w: T) -> T:
    var H = zeros[T, n * n]()
    var x = T.constant(-4.0)
    for i in range(n):
        H[i * n + i] = T.constant(kinetic) + w * w * x * x / T.constant(2.0)
        if i + 1 < n:
            H[i * n + i + 1] = T.constant(-0.5 * kinetic)
            H[(i + 1) * n + i] = T.constant(-0.5 * kinetic)
        x = x + T.constant(dx)

    var energies = eigh[T, n](H)[0].copy()          # eigh leaves them unsorted
    var lowest = energies[0].copy()
    for i in range(1, n):
        lowest = min_of(lowest, energies[i])        # branchless, so Dual sees through it
    return lowest^

def detuning[T: FloatLike](w: T) -> T:
    return ground_energy(w) - T.one()

def step[w: Int](ws: SIMD[f64, w]) -> SIMD[f64, w]:
    return ground_energy(Plain[f64, w](ws)).v

def main() raises:
    print(ground_energy(P.constant(1.0)))           # 0.4962  the grid's ground state
    print(ground_energy(Dual[P].seed(1.0)).deriv)   # 0.4923  dE0/dw, no perturbation theory
    print(newton[f=detuning](P.constant(1.0)))      # 2.0317  the w that puts E0 at 1

    var gpu = DeviceContext()                       # CUDA or Metal, whichever is there
    var ws = linspace[256, f64](0.5, 2.0, ctx=gpu)  # a Tensor, in device memory
    var es = zeros[f64, 256](gpu)
    gpu.enqueue_function[map[LayoutType=Sweep, step=step, gpu=True]](
        ws.view(), es.view(), grid_dim=4, block_dim=64
    )                                               # 256 wells, one eigensolve per thread
    gpu.synchronize()
    print(es.to_host()[255])                        # 0.9846  bit-identical to the CPU path
```

One function, four answers. At `Plain` it is the ground-state energy. At `Dual`
it also returns $dE_0/d\omega$, which is the Hellmann-Feynman theorem arriving
without perturbation theory being written anywhere. Handed to `newton` it runs
backwards, finding the frequency that puts the ground state at a chosen energy
from a derivative the caller never supplied. Handed to `map`, the entire 24x24
eigensolve runs inside a single GPU thread, 256 wells at once, and matches the
host path bit for bit.

The percent or so between these numbers and the continuum ($E_0 = \omega/2$, so
$0.5$ and $2.0$) is the 24-point grid, not the library. The derivative is exact
for the discretized operator, and inherits that same offset.
[`examples/advanced/quantum_well.mojo`](examples/advanced/quantum_well.mojo)
runs the whole thing, CPU and GPU side by side.

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
  changes, and `.view()` yields the `TileTensor` every MAX kernel takes. Its
  shape lives in its layout type, so `Shaped[f32, 2, 3]` with the extents
  compiled in and `Dynamic[f32, 2]` with the extents supplied at run time are
  one type, not two. `Array[T, n]` is the register-resident half that carries
  the algorithms; [the section below](#start-with-tensor-cross-to-array-for-algorithms)
  is the map between them.
- **NumPy and SciPy's ground.** Special functions, dense linear algebra,
  quadrature, ODE solvers, FFTs, distributions, statistics, interpolation,
  signal and a NumPy-named array surface, plus `.npy` read/write, so a
  program ported from NumPy can ingest the files it already has and hand
  results back the same way. Full inventory in
  [`docs/features.md`](docs/features.md).
- **Fast, and measured, per processor.** On an A10G's GPU, 61,013 M elem/s
  against a hand-written CUDA kernel's 60,352 and `torch.compile`'s 53,670,
  at ~82% of the card's bandwidth spec. On its CPU, 0.998x a hand-written
  raw-SIMD loop and 4.3x NumPy. CPU and GPU
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
numax = { git = "https://github.com/ehsanmok/numax.git", tag = "v0.1.0" }
```

Requires [pixi](https://pixi.sh); `mojo` and `max` come in transitively.

Inside a clone, rather than depending on the package, every `mojo`
invocation needs this directory on the import path: `pixi run mojo -I .
examples/basic/npy_interop.mojo`, or `pixi run run examples/basic/npy_interop.mojo`,
which supplies the `-I .` for you. The named tasks (`pixi run example-npy-interop`)
already do.

## Getting started

### Start with `Tensor`, cross to `Array` for algorithms

`Tensor` is the array you use. It owns its memory, its shape lives in its
type, and the `DeviceContext` you hand a factory decides whether that memory
sits on the host or on a GPU.

```mojo
from numax.prelude import *
from numax.stats import sum                         # shadows a builtin, so not in the prelude

def main() raises:
    var a = ones[f64, 2, 3]()                       # extents compiled in
    print(a + a)                                    # [[2.0, 2.0, 2.0],
                                                    #  [2.0, 2.0, 2.0]]
    print(mean(a), sum[axis=1](a))                  # 1.0 [3.0, 3.0]

    var rows = 3                                    # a count computed at run time
    var b = zeros_dyn[f64, 2](rows, 3)              # same type, extents as arguments

    var on_gpu = zeros[f32, 1024](ctx=DeviceContext())   # same code, device memory
```

`Shaped[f64, 2, 3]` and `Dynamic[f64, 2]` are one struct at two layouts. The
first has its extents in the type, which is what a GPU launch and the
algorithm layer both need; the second carries them as values, which is what
makes a shape read from a file expressible at all. `.dynamic()` and
`.static_view[2, 3]()` move between the two without copying, and the second
raises if the extents it asserts are not the ones there.

Everything NumPy has a name for works on `Tensor`: creation, elementwise
math, reductions along an axis, sorting, masking, reshaping, `.npy` files.

`Array[T, n]` is the other array type, and it is not a smaller `Tensor`. It
is a value in registers whose element count is part of its type, generic over
the `FloatLike` conformer rather than over a `DType`. That is what makes a
Cholesky differentiate at `Dual` and run inside one GPU thread, one matrix
per SIMD lane, so it is what every algorithm takes: `linalg`, `fft`,
`signal`, `interpolate`, and the fixed-step kernels in `optimize` and
`integrate`. NumPy gets away with one array type because it needs neither
property.

`TileTensor` is MAX's borrowed view, a pointer and a layout that own nothing.
`.view()` hands one to a kernel and that is the only place it appears; you do
not build one yourself.

| | Owns its memory | Shape | Where you meet it |
|---|---|---|---|
| `Tensor` (`Shaped`, `Dynamic`) | yes, a MAX `DeviceBuffer` | in the layout type, compile time or run time per dimension | every NumPy-named call |
| `TileTensor` | no, it borrows | from the tensor it views | `.view()`, at a kernel boundary |
| `Array[T, n]` | it *is* the value, in registers | `n` at compile time | every SciPy-named algorithm |

Crossing is explicit in both directions:

```mojo
comptime P = Plain[f64]

var m = eye[3, f64]()                  # Tensor, 3x3
var lifted = to_array[P](m)            # Array[P, 9], row-major
var chol = cholesky[P, 3](lifted)      # algorithms live here
var back = to_tensor[f64, 3, 3](chol)  # Tensor again

var direct = solve[P, 3](eye[P, 3](), ones[P, 3]())   # no tensor to lift
```

So `Tensor` alone does not cover everything, and the boundary has two limits
worth knowing early. A `Dynamic` tensor cannot lift until it names a shape,
since `n` is a compile-time parameter. And `n`
unrolls: a 64x64 Cholesky takes about a minute to compile, a 128x128 one does
not finish, so the algorithm layer suits the small fixed-size problems that
appear inside a kernel rather than a large matrix read off disk. Large
run-time-sized decompositions are the gap this version leaves open.

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

var A = eye[P, 3]()
var b = ones[P, 3]()
print(solve[P, 3](A, b)[0])
print(det[P, 3](A))
print(norm[P, 3](A))
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

print(gauss_legendre[P, f, 16](P.constant(0.0), P.one()))
print(gamma(P.constant(5.0)), erf(P.one()))
print(rk4[P, g](P.constant(0.0), P.one(), P.constant(0.1)))
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
print(norm.cdf(P.constant(1.96), P.constant(0.0), P.one()))
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
print(d)
# Dual(0.7788007830714049, -0.7788007830714049)
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
| `np.zeros((2, 3))` | `zeros[f64, 2, 3]()` | extents in the type, checked at compile time |
| `np.zeros((r, c))` for computed `r`, `c` | `zeros_dyn[f64, 2](r, c)` | extents as ordinary arguments |
| `np.linspace(0, 1, 5)` | `linspace[5](0, 1)` | count first, `dtype` defaults to `f64` |
| `np.linspace(0, 1, 5, dtype=np.float32)` | `linspace[5, f32](0, 1)` | |
| `np.arange(5)`, `np.eye(3)` | `arange[5]()`, `eye[3]()` | `arange` takes a count, not a `stop` |
| `np.zeros_like(a)` | `zeros_like(a)` | derived shapes inherit `a`'s device |
| `np.eye(3)` to hand to `linalg` | `eye[P, 3]()` | same names at the conformer layer, returning `Array` |
| `a.reshape(2, 3)` | `reshape[f64, 6, 2, 3](a)`, or `reshape_dyn[rank=2](a, r, c)` | the second takes a shape you computed |
| `a[1:3, :]`, `np.broadcast_to(a, (2, 3))` | `slice(a, [1, 0], [3, cols])`, `broadcast_to[rank=2](a, 2, 3)` | both copy rather than returning a view |
| `a[a > 0]`, `np.take(a, idx)` | `extract(greater(a, zeros_like(a)), a)`, `take(a, idx)` | the result is sized by the data, so it comes back `Dynamic` |
| `a + b`, `np.exp(a)`, `np.sort(a)` | `a + b`, `exp(a)`, `sort(a)` | |
| `a.astype(np.float32)` | `astype[f32](a)` | explicit: there is no dtype promotion |
| `a.sum()`, `a.mean()`, `np.var(a)` | `sum(a)`, `mean(a)`, `variance(a)` | `sum`/`min`/`max` are outside the prelude |
| `a.sum(axis=1)`, `a.mean(axis=1)` | `sum[axis=1](a)`, `mean[axis=1](a)` | same name as the whole-tensor form; one axis drops, the rest survive |
| `np.linalg.solve(A, b)` | `solve[P, n](A, b)` | over `Array[T, n*n]`, so it differentiates |
| `np.linalg.cholesky/qr/svd/eigh` | `cholesky`, `qr`, `svd`, `eigh` | |
| `np.linalg.eigvals(A)`, `np.linalg.lstsq(A, b)` | `eigvals`, `lstsq` | no symmetry assumed; `lstsq` factors instead of forming the normal equations |
| `scipy.linalg.lu_factor` / `lu_solve` | `lu_factor(A).solve(b)` | partial pivoting, so it survives a zero pivot |
| `scipy.special.gamma/erf/j0` | `gamma`, `erf`, `j0` | every one documents an error bound |
| `scipy.integrate.fixed_quad` | `gauss_legendre[T, f, n]` | fixed nodes, GPU-launchable |
| `scipy.integrate.quad` | `quad[f](a, b)` | adaptive, host-only, `Float64` bounds |
| `scipy.integrate.solve_ivp` | `solve_ivp`, or `rk4` for fixed steps | |
| `solve_ivp(method="BDF")` | `solve_ivp_stiff` | implicit, so the step size follows accuracy rather than stability |
| `scipy.interpolate.CubicSpline` | `CubicSpline[T, n]` | built once, `__call__` evaluates |
| `scipy.optimize.brentq` / `minimize` | `brentq[f](a, b)` / `bfgs[n, f](x0)` | `bfgs` needs no `jac` |
| `minimize(method="Nelder-Mead")` | `nelder_mead[n, f](x0)` | compares values only, for objectives with kinks |
| `scipy.optimize.least_squares` / `curve_fit` | `least_squares`, `curve_fit` | Jacobian from `Gradient`, so it is exact |
| `scipy.optimize.approx_fprime` | evaluate at `Dual` / `Gradient` | exact, not a difference quotient |
| `np.fft.fft`, `np.fft.rfft` | `fft[T, log2n]`, `rfft` | length is `2^log2n`, over `Array[Complex[T], n]` |
| `scipy.signal.lfilter` / `firwin` | `lfilter`, `firwin` | a recursion `convolve` cannot express, and taps to run through it |
| `np.save` / `np.load` | `numpy.save` / `numpy.load` | real `.npy`, readable by NumPy |
| `stats.norm.cdf(x)` | `norm.cdf(x, mu, sigma)` | nine distributions, parameters explicit |
| `np.random.default_rng(0)` | `Generator(seed=0)` | or `seed(0)` for the global stream |

Rows that read `[f64, ...]` return a `Tensor` and rows that read `[P, ...]`
return an `Array`, which is the split the section above lays out.
`zeros`/`ones`/`full`/`eye` are spelled the same on both sides, so the first
parameter is what picks: a `DType` builds a tensor, a conformer builds an
array.

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
`linspace[5, f64](...)` and `Shaped[f64, 4, 4](ctx)` on the tensor side,
`Plain[f64]` and `Dual[Plain[f64]]` on the kernel side. Every MAX dtype has
one: `f16`, `bf16`, the five `f8e*` variants, `i8` through `u64`, and `bool`.
The kernel layer needs a floating-point dtype, so the integer names belong to
tensors, where storage and comparison results live.

`Plain[dtype, width]` wraps a `SIMD[dtype, width]`. Mojo declares conformance
where a type is defined, so a bare `SIMD` cannot conform to `FloatLike`, and
`Plain` is what lets the hardware type participate. It is the baseline: every
kernel runs at `Plain` unless you ask for something else, and it compiles away.
`pixi run bench` measures `map` over a `Plain` kernel at **0.998x a
hand-written raw-SIMD loop**, `max |raw - numax| = 0.0`. `width` defaults to 1,
so `Plain[f64]` is the scalar. Every conformer prints, so `print(x)` is
enough to see a result; `.v` reaches the raw `SIMD` when that is what you
want, and `Dual` exposes `.value`/`.deriv`.

> **Run it:** `pixi run example-gaussian` ·
> [`examples/basic/gaussian.mojo`](examples/basic/gaussian.mojo)

### Precision, for free

Sum a million nearly-equal `float32` values and the running total stops seeing
the next one. Swap the type, not the algorithm:

```mojo
var plain_var = variance(plain_list)           # float32 accumulation
var comp_var = variance(comp_list).value       # ~double precision, same code
```

`Compensated` carries the rounding error ordinary arithmetic discards.

> **Run it:** `pixi run example-statistics` ·
> [`examples/intermediate/statistics.mojo`](examples/intermediate/statistics.mojo)

### The same kernel, on the GPU

Only the context and the walk differ:

```mojo
comptime T = Shaped[f32, 1024]

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

The objective is an ordinary `FloatLike` kernel taking the two-variable point
as an `Array` (a value in registers, per the map above), so BFGS evaluates it
at `Gradient` and gets every $\partial f / \partial x_i$ *exactly*:

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

## The two tiers

Some algorithms can run inside a GPU thread and some cannot, so every module
says which it is rather than leaving you to read the body.

Tier 1 runs a fixed number of iterations and never branches per lane, which is
what makes it launchable on a GPU and callable at any conformer. That covers
the conformers themselves, the tensor engine, `special`, `linalg`,
`interpolate`, `fft`, `signal`, and the fixed-step half of `optimize` and
`integrate`. Per-lane choices are arithmetic blends built from `copysign`
rather than `if`, because the lanes of one SIMD value can disagree about which
branch they want.

Tier 2 is free to loop until it converges and to branch on the data it sees. It
is `Plain`-only and host-side: `ops`, `elementwise`, `logic`, `sorting`, `io`,
the tensor reductions in `stats`, the converge-to-tolerance minimizers in
`optimize`, and the adaptive `quad`/`solve_ivp` in `integrate`.

Tier 1 never calls tier 2, so a kernel you can launch stays launchable. Where
both make sense the library ships both: `newton` at a fixed iteration count and
`newton_tol` to a tolerance are siblings, and each docstring names the other.

## Performance

Same kernel everywhere: $g(x) = e^{-x^2}$ over `float32`, 67M elements, wall
clock from dispatch through completion. M elem/s, higher is better. CPU and GPU
are measured and reported separately: every row below compares against
implementations running on the same processor.

**GPU, 67M elements, one sync per call:**

| Device | numax | CuPy kernel | torch.compile | torch eager | MLX |
|---|---|---|---|---|---|
| NVIDIA A10G (CUDA 12.8) | **61,013** | 60,352 | 53,670 | 19,803 | n/a |
| Apple M3 Pro (Metal) | **14,465** | n/a | 13,831 | 4,807 | 4,866 |

Amortizing the sync over ten launches instead: 61,653 for numax against
61,051 for CuPy and 56,245 for `torch.compile` on the A10G, 15,755 vs. 14,380
on the M3 Pro. CuPy is CUDA-only and MLX macOS-only, hence the two gaps.

CuPy's column is a hand-written `cupy.ElementwiseKernel` — CUDA C for this
one expression. numax is at parity with it, from a kernel that names no
device and no dtype, both pinned at 82% of the card's bandwidth. CuPy's
*eager* path measures 20,375, the 3x cost of leaving `exp(-(x*x))` unfused;
numax has no such column, because a `FloatLike` kernel is already fused
before the walk begins.

**CPU, 67M elements:**

| Host | numax `map_threaded` | numax `map` (1 thread) | torch.compile | torch eager | Rust `thermite` | NumPy |
|---|---|---|---|---|---|---|
| AMD EPYC (the A10G's host) | **8,733** | 1,367 | 1,880 | 468 | 1,329 (AVX2) | 316 |
| Apple M3 Pro | **9,026** | 2,347 | 4,046 | 1,935 | 1,632 (NEON) | 493 |

MLX's CPU path measures 1,954 on the M3 Pro (it has no CUDA build, so no EPYC
row).

- **The GPU work is bandwidth-bound, not compute-bound.** The best
  configuration on the A10G runs at **500.9 GB/s, ~83% of the card's 600 GB/s
  spec**, and an identity copy measures 489.10 GB/s against the Gaussian's
  489.16, so the `exp` is free.
- **Fusing two passes into one composed `step` is worth 1.99x** on the GPU at
  every size tested, and 1.26-1.58x on the CPU.
- **The serial CPU walk matches hand-written Rust SIMD** (1,367 vs.
  `thermite`'s 1,329 on the EPYC host) and is 4.3x NumPy. The `FloatLike`
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
