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
[MAX](https://max.modular.com/docs/) -- what NumPy and SciPy provide, on MAX's
tensors: special functions, dense linear algebra with its spectral
decompositions, optimization, quadrature and ODE solvers, interpolation, FFTs,
signal processing, distributions and statistics, and a NumPy-named array
surface, written in [Mojo](https://mojolang.org) against MAX's `TileTensor` and
kernel infrastructure. Data lives in a MAX `DeviceBuffer`, so the
`DeviceContext` you pass decides host or device: the same kernel, any
accelerator, unmodified.

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
comptime Sweep = Static[f64, 256].LayoutType

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
  shape lives in its layout type, so `Static[f32, 2, 3]` with the extents
  compiled in and `Dynamic[f32, 2]` with the extents supplied at run time are
  one type, not two. `Array[T, n]` is the register-resident half that carries
  the algorithms; [the section below](#start-with-tensor-cross-to-array-for-algorithms)
  is the map between them.
- **NumPy and SciPy's ground.** Special functions at arbitrary order, dense
  linear algebra through the spectral decompositions -- `eigh`, `svd`,
  `schur` and the matrix functions on the Schur form -- optimization with
  bounds, quadrature and ODE solvers, interpolation, FFTs at any length,
  signal processing from filter design to the spectral estimators, the nine
  `scipy.stats` distributions over whole tensors, the statistics surface from
  `quantile` and `histogram` through the hypothesis tests, and a NumPy-named
  array surface, plus `.npy` read/write, so a program ported from NumPy can
  ingest the files it already has and hand results back the same way. Full
  inventory in [`docs/features.md`](docs/features.md).
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

`Static[f64, 2, 3]` and `Dynamic[f64, 2]` are one struct at two layouts. The
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
per SIMD lane.

**Every algorithmic subpackage now has both tiers**, one per import, sharing
one set of names: `numax.linalg`, `numax.optimize`, `numax.integrate`,
`numax.interpolate`, `numax.fft` and `numax.signal` are the `Tensor` tier and
go through MAX, and `numax.linalg.array`, `numax.optimize.array` and their
siblings are the differentiable register-resident one. Pick by what you need
rather than by what exists: a device-resident matrix or a long recording
wants the `Tensor` tier, a derivative through the algorithm wants the `Array`
tier, and `to_tensor`/`to_array` cross between them. NumPy gets away with one
array type because it needs neither property.

`TileTensor` is MAX's borrowed view, a pointer and a layout that own nothing.
`.view()` hands one to a kernel and that is the only place it appears; you do
not build one yourself.

| | Owns its memory | Shape | Where you meet it |
|---|---|---|---|
| `Tensor` (`Static`, `Dynamic`) | yes, a MAX `DeviceBuffer` | in the layout type, compile time or run time per dimension | every NumPy-named call |
| `TileTensor` | no, it borrows | from the tensor it views | `.view()`, at a kernel boundary |
| `Array[T, n]` | it *is* the value, in registers | `n` at compile time | a SciPy-named algorithm you want to differentiate, or to run per SIMD lane |

Crossing is explicit in both directions:

```mojo
comptime P = Plain[f64]

var m = eye[3, f64]()                  # Tensor, 3x3
var lifted = to_array[P](m)            # Array[P, 9], row-major
var chol = cholesky[P, 3](lifted)      # algorithms live here
var back = to_tensor[f64, 3, 3](chol)  # Tensor again

var direct = solve[P, 3](eye[P, 3](), ones[P, 3]())   # no tensor to lift
```

The crossing has two limits worth knowing early, and both are about the
`Array` tier rather than about `Tensor`. A `Dynamic` tensor cannot lift until
it names a shape, since `n` is a compile-time parameter. And `n` unrolls: a
64x64 Cholesky takes about a minute to compile and a 128x128 one does not
finish, so that tier suits the small fixed-size problems that appear inside a
kernel rather than a large matrix read off disk.

Which is what the `Tensor` tier is for, and it no longer stops at the
factorizations: `eigh`, `svd`, `schur` and the matrix functions on the Schur
form all run there, blocked and device-resident. What that tier does not do
is differentiate, because a `Tensor` is monomorphic in a `DType` and a
conformer is a struct -- so the two tiers are a real choice and not a
staging area. Run-time *extents* for a decomposition are the gap this version
still leaves open: a `Dynamic` tensor has to name its shape first.

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
| `np.pad(a, (before, after), mode)` | `pad[f64, n, before, after, mode](a)` | the widths are compile-time because the padded extent is part of the return type; `gpu=True` is the constant mode only, and the `where` clause makes the other modes a compile error rather than a silent host fallback |
| `a[a > 0]`, `np.take(a, idx)` | `extract(greater(a, zeros_like(a)), a)`, `take(a, idx)` | the result is sized by the data, so it comes back `Dynamic` |
| `a + b`, `np.exp(a)`, `np.sort(a)` | `a + b`, `exp(a)`, `sort(a)` | |
| `np.argpartition(a, -k)`, `torch.topk` | `top_k[f64, n, k](a)` | `(values, indices)`, `largest=False` for the smallest. A whole delegation to `nn.top_k`, so `gpu=True` runs MAX's device kernel with no host copy in either direction |
| `a.astype(np.float32)` | `astype[f32](a)` | explicit: there is no dtype promotion |
| `a.sum()`, `a.mean()`, `np.var(a)` | `sum(a)`, `mean(a)`, `variance(a)` | `sum`/`min`/`max` are outside the prelude |
| `a.sum(axis=1)`, `a.mean(axis=1)` | `sum[axis=1](a)`, `mean[axis=1](a)` | same name as the whole-tensor form; one axis drops, the rest survive |
| `np.linalg.solve(A, b)` | `solve(A, b)`, or `solve[P, n](A, b)` from `numax.linalg.array` | the first is blocked pivoted LU with its trailing update in MAX's GEMM; the second differentiates |
| `np.linalg.cholesky/qr/svd/eigh` | `cholesky`, `qr_factor`, `svd`, `eigh` | all four over `Tensor`, and all four in `numax.linalg.array` too. The `Tensor` ones are blocked with their cubic term in MAX's GEMM; the `Array` ones are register-resident and differentiate |
| `np.linalg.eigvals(A)`, `np.linalg.lstsq(A, b)` | `eigvals`, `lstsq`, or `qr_factor(A).solve(b)` | `eigvals` assumes no symmetry and returns `(re, im)` as two tensors, since a `dtype`-monomorphic tensor holds no complex number; every least-squares route factors instead of forming the normal equations |
| `scipy.linalg.lu_factor` / `lu_solve` | `lu_factor(A).solve(b)` | partial pivoting, so it survives a zero pivot |
| `np.kron`, `np.linalg.matrix_power`, `np.inner` | `kron`, `matrix_power[dtype, n, p]`, `inner` | `inner` is `a @ b.T` without materializing the transpose |
| `np.linalg.slogdet` | `slogdet(A)` | `(sign, ln\|det\|)`, for the ordinary matrices whose determinant overflows |
| `np.linalg.norm(v, ord)` | `norm[dtype, n, ord](v)` | a vector overload beside the matrix one; `ord=0` is `count_nonzero` |
| `np.linalg.matrix_rank`, `eigvalsh`, `svdvals` | same names | over `Tensor`, `eigvalsh` is ascending and `svdvals` descending; the `numax.linalg.array` forms are unsorted and `matrix_rank` there returns a count per SIMD lane |
| `np.linalg.pinv`, `np.linalg.cond` | `pinv`, `cond` | both one SVD and a few lines on top of it; `pinv` takes NumPy's `rcond` |
| `scipy.linalg.schur`, `hessenberg` | `schur`, `hessenberg` | real Schur form, so a complex pair stays a `2 x 2` block rather than splitting into eigenvalues that are not there |
| `np.tensordot`, `np.cross` | `tensordot[axes=k]`, `cross` | `tensordot` reshapes to a matrix and hands the contraction to MAX's GEMM; `einsum` stays out |
| `np.linalg.tensorsolve`, `tensorinv` | `tensorsolve`, `tensorinv` | the rank-n shapes read as a square matrix, which is what NumPy's do |
| `scipy.linalg.toeplitz` / `circulant` / `companion` / `block_diag` | same names | plus `hankel`, `hilbert`, `khatri_rao`, `convolution_matrix`, `pascal`, `invpascal`, `hadamard`, `helmert`, `fiedler`, `fiedler_companion`, `leslie`. `dft` and `invhilbert` stay out, on grounds [`docs/parity.md`](docs/parity.md) records |
| `scipy.linalg.solve_banded` / `solveh_banded` / `solve_toeplitz` | same names | SciPy's diagonal-ordered `ab` storage verbatim; host-side by declaration |
| `scipy.linalg.solve_circulant` | `solve_circulant` | three FFTs and a division, so power-of-two `n` |
| `scipy.linalg.expm` | `expm(A)` | and `expm[T, n, squarings]` over `Array`, which differentiates at `Dual` |
| `scipy.linalg.sqrtm`, `logm`, `funm` | `sqrtm`, `logm`, `funm[f=...]`, `cosm`, `sinm`, `fractional_matrix_power` | over `Tensor` these are the general case: the Bjorck-Hammarling and block Parlett recurrences on the real Schur form, so repeated and defective eigenvalues are handled rather than dividing by zero. `funm`'s `f` is any `FloatLike` kernel, which is where the two halves of the library meet. `sqrtm[P, n]` from `numax.linalg.array` is the SPD-only one that differentiates |
| `scipy.special.gamma/erf/j0` | `gamma`, `erf`, `j0` | every one documents an error bound, checked by `pixi run accuracy` |
| `scipy.special.jv/yv/iv/kv`, `airy`, `struve` | same names, plus `spherical_jn`/`spherical_yn`, `ive`/`kve` | arbitrary real order, not just the integer-order `j0`..`y1` |
| `scipy.special.erfinv`, `zeta`, `expi`, `sici`, `fresnel`, `owens_t` | same names, plus `exp1`/`expn`, `hyp1f1`/`hyp2f1`, `poch`, `factorial`/`comb`/`perm` | tier 1 throughout, so every one compiles into a `map[gpu=True]` body |
| `scipy.special.logsumexp` | `logsumexp` | over `Tensor`, driving MAX's own `OnlineLogSumExp` monoid through the `rowwise` scaffolder |
| `scipy.special.xlogy`, `rel_entr`, `kl_div` | same names, plus `entr`, `xlog1py`, `logit` | the information-theoretic set, each with the limit at zero defined the way SciPy defines it |
| `scipy.integrate.trapezoid` / `simpson` | `trapezoid(y, dx=...)`, `simpson`, `cumulative_trapezoid` | SciPy's signature: these integrate **samples** in a `Tensor`. The function-taking forms are `numax.integrate.array`, where they differentiate |
| `scipy.integrate.fixed_quad` | `gauss_legendre[T, f, n]` | fixed nodes, GPU-launchable |
| `scipy.integrate.quad` | `quad[f](a, b)` | adaptive, host-only, `Float64` bounds |
| `scipy.integrate.solve_ivp` | `solve_ivp`, or `rk4` for fixed steps | |
| `solve_ivp(method="BDF")` | `solve_ivp_stiff` | implicit, so the step size follows accuracy rather than stability |
| `scipy.interpolate.CubicSpline` | `CubicSpline[dtype, n]` | built once, `__call__` evaluates, `[nu=k]` differentiates and `.integrate` integrates. Knots need not be uniform, and `bc_type` is SciPy's `not-a-knot`/`natural`/`clamped` |
| `scipy.interpolate.PchipInterpolator` / `Akima1DInterpolator` | same names, plus `CubicHermiteSpline` | the shape-preserving siblings, same evaluation surface |
| `np.interp`, `scipy.interpolate.RegularGridInterpolator` | `interp`, `RegularGridInterpolator` | `interp` is a vectorized `searchsorted` and a gather, one launch each |
| `scipy.optimize.root_scalar` / `minimize` | `root_scalar[f](bracket=(a, b))` / `minimize[n, f](x0)` | over `Array`, no `jac`, `fprime` or `fprime2` anywhere — every derivative comes from `Dual` or `Gradient` |
| `minimize` over a large vector | `minimize[dtype, n, f, jac](x0)` from `numax.optimize` | the `Tensor` tier, which takes `jac` because a tensor cannot hold a `Gradient`. `bfgs`, `l-bfgs`, `cg` and `powell`; limited memory is the one to reach for when a dense inverse Hessian will not fit |
| `minimize(method="L-BFGS-B", bounds=...)` | `minimize[...](x0, lower, upper)` | projected gradient on the free set; `powell` needs no `jac` at all |
| `scipy.optimize.nnls`, `lsq_linear` | `nnls(A, b)`, `lsq_linear(A, b, lo, hi)` | one box-constrained QP, the normal equations formed on the device and the active set settled on the host |
| `scipy.optimize.minimize_scalar` | `minimize_scalar[f]()` | `brent`, `golden`, `bounded`; a bracket is a direction, bounds are a constraint |
| `minimize(method="Nelder-Mead")` / `"CG"` | `minimize[n, f, method="nelder-mead"]` / `method="cg"` | SciPy's own method spelling; `bfgs`, `cg` and `nelder_mead` are also callable by name |
| `scipy.optimize.root` | `root[n, f](x0)` over `Array`, `root[dtype, n, f, jac](x0)` over `Tensor` | `newton` and `lm`; check `residual_norm`, not only `converged`, since a system with no root still has points where `\|\|F\|\|` stops falling |
| `scipy.optimize.least_squares` / `curve_fit` | `least_squares`, `curve_fit` | Jacobian from `Gradient`, so it is exact |
| `scipy.optimize.approx_fprime` | evaluate at `Dual` / `Gradient` | exact, not a difference quotient |
| `np.fft.fft`, `np.fft.rfft`, `np.fft.irfft` | `fft`, `rfft`, `irfft` | **any length** over `Tensor`: radix-2 at a power of two, Bluestein's chirp-z otherwise. `numax.fft` is the `Tensor` tier, a real/imaginary pair across `log2(n) + 1` device stages; `numax.fft.array` is `Array[Complex[T], n]`, differentiates, and stays power-of-two |
| `np.fft.fft2` / `rfft2` / `fftshift` | `fft2`, `ifft2`, `rfft2`, `fftshift`, `ifftshift`, `next_fast_len` | rectangular, one axis at a time, device-resident between them |
| `scipy.fft.dct` / `dst` | `dct`, `idct`, `dst`, `idst` | types I through IV, each a real projection of one complex DFT |
| `scipy.signal.convolve` / `correlate` / `fftconvolve` | same names | `full`/`same`/`valid`. The direct form is one launch of dot products; which route is faster depends on the kernel length and [`docs/performance.md`](docs/performance.md) measures the crossover rather than guessing |
| `scipy.signal.lfilter` / `filtfilt` / `sosfilt` | same names, plus `lfilter_zi` | recurrences, so host-side by declaration: sample `k` needs sample `k - 1`, which leaves neither a GEMM nor independent lanes |
| `scipy.signal.firwin` / `butter` / `freqz` | same names | `firwin` covers every band shape; `butter` is IIR design, which used to be out of scope here and is a recorded reversal rather than a quiet addition |
| `scipy.signal.medfilt` / `savgol_filter` / `detrend` / `resample` | same names | a window per lane, so these are the filters that go to a device unchanged |
| `scipy.signal.get_window` and the window factories | `hann`, `hamming`, `blackman`, `bartlett`, `kaiser`, `boxcar`, `get_window` | SciPy's symmetric and periodic forms both |
| `scipy.signal.welch` / `spectrogram` / `stft` / `hilbert` | same names from `numax.signal`, plus `periodogram`, `find_peaks` | each one batched transform over framed input |
| `np.save` / `np.load` | `numpy.save` / `numpy.load` | real `.npy`, readable by NumPy |
| `stats.norm.cdf(x)` | `norm.cdf(x, mu, sigma)` | nine distributions, parameters explicit, all eight `scipy.stats` methods each (`pdf`/`logpdf`, `cdf`/`logcdf`, `sf`/`logsf`, `ppf`, `isf`). **`x` may be a whole `Tensor`**, which is one `elementwise` launch and runs on a device with `gpu=True` |
| `np.quantile`, `np.percentile`, `np.median` | `quantile`, `percentile`, `median` | all thirteen NumPy `method=` interpolations, plus the `nan*` forms and `iqr` |
| `np.histogram`, `np.bincount`, `np.digitize` | same names, plus `histogram2d`, `histogramdd` | NumPy's edge rules, weights and `density` included |
| `np.cov`, `np.corrcoef` | `cov`, `corrcoef` | plus `pearsonr`, `spearmanr`, `kendalltau`, `linregress`, `rankdata`, `zscore` |
| `scipy.stats.ttest_ind`, `chisquare`, `ks_1samp` | same names, plus `ttest_1samp`/`ttest_rel`, `f_oneway`, `mannwhitneyu` | each a statistic and a tail of `t`/`chi2`/`f`/`norm`, returning `statistic` and `pvalue` |
| `scipy.stats.describe`, `skew`, `kurtosis` | same names, plus `sem`, `gmean`, `hmean`, `entropy`, `trim_mean` | |
| `np.random.default_rng(0)` | `Generator(seed=0)` | or `seed(0)` for the global stream |

Every row above is runnable: `pixi run example-scipy-surface` prints the
`scipy.optimize` and `scipy.linalg` entry points side by side with the SciPy
they mirror ([`scipy_surface.mojo`](examples/intermediate/scipy_surface.mojo)).


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
full flat surface, and `from numax.linalg import ...` is one subsystem. The
`Array` tier of `linalg` is outside the prelude for the same reason: it
shares its names with the `Tensor` tier, so it is
`from numax.linalg.array import cholesky` when that is the one you want.

`f32`/`f64` are short for `DType.float32`/`.float64` and nothing else, so the
same name works wherever a dtype belongs, across both layers:
`linspace[5, f64](...)` and `Static[f64, 4, 4](ctx)` on the tensor side,
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
comptime T = Static[f32, 1024]

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

var minimized = minimize[2, rosenbrock](start)  # no `jac` argument
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
the conformers themselves, the tensor engine, all of `special`, and the
`Array` tiers -- `linalg.array`, `interpolate.array`, `fft.array`,
`signal.array`, and the fixed-step half of `optimize.array` and
`integrate.array`. Per-lane choices are arithmetic blends built from `copysign`
rather than `if`, because the lanes of one SIMD value can disagree about which
branch they want.

Tier 2 is free to loop until it converges and to branch on the data it sees. It
is `Plain`-only and host-side: `ops`, `elementwise`, `logic`, `sorting`, `io`,
the tensor reductions and the whole statistics surface in `stats`, the
converge-to-tolerance minimizers in `optimize`, the adaptive
`quad`/`solve_ivp` in `integrate`, and the `Tensor` tier of `linalg` --
including the spectral decompositions, whose reduction to band form is
blocked and device-resident but whose sweep over that band is a host loop
that deflates on a test of the data.

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

**Where this version is slow, stated rather than omitted.** Everything that
is one `elementwise` launch or one batched transform is ahead of SciPy on the
same processor -- `norm.cdf` 5.4x, `interp` 9.4x, `welch` 12x, `medfilt` and
`savgol_filter` about 2x. Everything that is still a host loop is behind it
by roughly what a scalar loop costs against C: `quantile` 0.05x, `lfilter`
0.08x, `cov` 0.11x. The spectral decompositions are the sharpest case, at
0.002-0.11 of LAPACK at `n = 1024`, because the reduction to band form is
blocked and device-resident but accumulating the eigenvectors is `O(n^3)` of
scalar Givens rotations on the host. Every one of those numbers, and what
would close each gap, is in
[`docs/performance.md`](docs/performance.md).

**Dense linalg is a separate measurement, on separate hardware** (EPYC 7R32
host, A10G device), `float32` because MAX's `matmul` does not compile for GPU
at `float64`. GFLOP/s at `n = 1024`, higher is better:

| | `matmul` (the ceiling) | `cholesky` | `lu_factor` | `solve` |
|---|---|---|---|---|
| numax, CPU | 656 | 15.4 | 17.2 | 17.0 |
| SciPy (LAPACK + OpenBLAS), CPU | 833 | 99.5 | 43.9 | 52.6 |
| numax, A10G | 20,459 | 51.0 | 30.6 | 28.1 |
| PyTorch (cuSOLVER), A10G | 15,342 | 595.0 | 282.9 | 257.6 |

The same four on an **Apple M3 Pro** and its 18-core Metal GPU, against
Accelerate and PyTorch's MPS backend -- a different machine, so it is a
separate table and not a column of the one above:

| | `matmul` (the ceiling) | `cholesky` | `lu_factor` | `solve` |
|---|---|---|---|---|
| numax, CPU | 1,475* | 83.4 | 86.9 | 80.3 |
| SciPy (LAPACK + Accelerate), CPU | 1,393* | 280.1 | 231.9 | 162.8 |
| numax, Metal | **1,800** | 65.2 | 19.0 | 16.9 |
| PyTorch (MPS), Metal | 1,143 | 125.0 | 51.5 | 20.9 |

\* Both CPU ceiling entries are the same kernel -- Apple Accelerate's
`cblas_sgemm`, which MAX dispatches to on macOS at `float32`.

Read those as one claim and one gap. The `matmul` row is the claim: numax's
factorizations put their whole `O(n^3)` term through `linalg.matmul`, and
MAX's GEMM is at 79% of OpenBLAS on the EPYC, *ahead* of cuBLAS's FP32 path
on the A10G, and ahead of PyTorch's Metal kernel on the M3 Pro -- 1,800
against 1,143. **The M3 Pro's CPU row is not a kernel comparison**: MAX
dispatches to Apple's `cblas_sgemm` on macOS at `float32`, so numax and SciPy
are calling the same GEMM there and the 6% is call overhead. That makes the
gap statement sharper rather than weaker -- on that machine the multiply
underneath both is identical, so every bit of the factorization gap is the
blocked algorithm around it.

The factorization rows are that gap. Each block step is a single-block panel
kernel plus a host launch, so cuSOLVER's 6-12x is a parallel panel. Note also
that the ceiling row is a *square* GEMM while a blocked factorization only
ever issues a rank-`block` one, which on this machine runs 5-9x slower -- so
the fraction of the ceiling a factorization reaches overstates how much it is
leaving behind. [`docs/performance.md`](docs/performance.md) has both
measurements.

BLAS-1 goes the other way: `dot` and `nrm2` on `Tensor` beat OpenBLAS's own
`sdot`/`snrm2` by 2.5-5x on the EPYC and Accelerate's by 1.7-3.2x on the M3
Pro, because `ReduceSum` under MAX's `rowwise` scaffolder threads and they
do not. On Metal, **MLX has no GPU linalg at all** -- it refuses `cholesky`,
`lu_factor`, `qr` and `solve` on a GPU stream -- so numax's device-resident
factorizations have no MLX counterpart there. ROCm is reached by the same
`target="gpu"` with no per-architecture code in numax, and is
**unmeasured**.

## Accuracy

Every approximation documents an error bound, and `pixi run accuracy` checks it
against checked-in [mpmath](https://mpmath.org/) references at 50 digits
(`erf`'s A&S 7.1.26 bound of ~1.5e-7 measures 1.38e-07). At `float64` the
bounds run 1e-15 to 1e-13 across the table. Getting there meant not using
Mojo's `std.math` at that width: its `exp`, `log` and `erf` are off by 1e5,
9e6 and 2e8 ulp respectively, so `Plain` calls numax's own fdlibm versions in
`numax/core/libm.mojo` instead, and every function built on them stopped
inheriting that floor. Details:
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
