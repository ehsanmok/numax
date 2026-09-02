# Examples

Runnable examples, tiered by complexity. Each is a single Mojo file
invoked with `mojo -I .`:

```bash
pixi run example-gaussian            # or any single example -- see pixi.toml
pixi run examples-cpu                # every CPU example, in sequence
pixi run examples                    # CPU + GPU examples (needs a GPU)
```

## basic/

Single-feature entry points -- one concept per file.

| File | What it shows |
|---|---|
| [gaussian.mojo](basic/gaussian.mojo) | `Plain` / `Dual` / `Compensated` three ways, via `numax.tensor.map` at native SIMD width, checked against a reference. |
| [array_creation.mojo](basic/array_creation.mojo) | `numax.array`'s NumPy-named creation (`zeros`/`ones`/`full`/`eye`/`linspace`) and manipulation (`transpose`/`squeeze`/`stack`) surface over `TileTensor`. |
| [activations.mojo](basic/activations.mojo) | Activations differentiated via `Dual`, checked against their closed-form derivatives. |
| [complex.mojo](basic/complex.mojo) | `Complex[Plain]` arithmetic, plus `Complex[Dual[Plain]]` differentiating `z^2` holomorphically. |
| [gradient.mojo](basic/gradient.mojo) | `Gradient[Plain, 2]` recovering both partial derivatives of a two-variable kernel from one call. |
| [hessian.mojo](basic/hessian.mojo) | `Gradient[Dual[Plain], 2]` producing a full Hessian (and Hessian-vector products) purely by nesting, with no second-order code in either type. |

## intermediate/

Multi-feature compositions -- more than one concept per file.

| File | What it shows |
|---|---|
| [softmax.mojo](intermediate/softmax.mojo) | Row-wise softmax on CPU and GPU, checked against each other and against each row summing to 1. |
| [quadrature.mojo](intermediate/quadrature.mojo) | Root-finding from `f` alone, 8-point Gauss-Legendre against a 64-point uniform grid, and differentiating through an integral. |
| [special_functions.mojo](intermediate/special_functions.mojo) | Every special function (erf, gamma, Bessel, Lambert W, elliptic integrals, orthogonal polynomials) differentiated via `Dual`. |
| [statistics.mojo](intermediate/statistics.mojo) | `numax.statistics`'s `Plain`-only `TileTensor` surface (`sum`/`median`/`argmax`/...), plus `variance`/`stddev` at `Compensated` recovering precision a long summation would lose at `Plain`. |
| [wave_packet.mojo](intermediate/wave_packet.mojo) | A 2-D Gaussian wave packet on a `meshgrid`: the density along a cut, the carrier recovered by counting sign changes of `Re psi`, and the packet's spread differentiated with respect to its own width -- `d<x^2>/d(sigma)` from the same kernel that computes `<x^2>`. |
| [interference.mojo](intermediate/interference.mojo) | Two-source interference over a coordinate grid in one `map`, with the fringe maxima and their spacing measured down the far column, plus the far-field phase differentiated with respect to the slit separation. |
| [random_ensemble.mojo](intermediate/random_ensemble.mojo) | `numax.random.uniform` drawing an ODE ensemble's initial conditions on CPU, versus `std.random.philox.Random` drawing them independently per GPU thread inside a `map[gpu=True]` kernel. Needs a GPU. |

## advanced/

GPU + ensembles -- needs the actual hardware.

| File | What it shows |
|---|---|
| [gaussian_gpu.mojo](advanced/gaussian_gpu.mojo) | The same kernel and types as `basic/gaussian.mojo` run on GPU via `numax.tensor.map[gpu=True]`, no code changes versus the CPU example. |
| [ode.mojo](advanced/ode.mojo) | 1024 ODE trajectories, one GPU thread each, with solution sensitivities from the same integrator. |
| [unified_tensor_gpu.mojo](advanced/unified_tensor_gpu.mojo) | One `numax.array.Tensor` type on both devices: the `DeviceContext` handed to the factory decides host or accelerator, `.view()` is the same `TileTensor` either way, and the two results agree to within one float32 ulp. |

## GPU note

`gaussian_gpu.mojo`, `softmax.mojo`, `ode.mojo`, `random_ensemble.mojo` and
`unified_tensor_gpu.mojo` launch GPU kernels alongside their CPU paths.
They need a real GPU -- Metal or CUDA, whichever `DeviceContext` finds --
and no GitHub-hosted runner has either, so they're a local-only check;
`pixi run examples-cpu` skips them, `pixi run examples` includes them.
