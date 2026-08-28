# Examples

Runnable examples, tiered by complexity. Each is a single Mojo file
invoked with `mojo -I .`:

```bash
pixi run example-gaussian            # or any single example -- see pixi.toml
pixi run examples-cpu                # every CPU example, in sequence
pixi run examples                    # CPU + GPU examples (needs Metal hardware)
```

## basic/

Single-feature entry points -- one concept per file.

| File | What it shows |
|---|---|
| [gaussian.mojo](basic/gaussian.mojo) | `Plain` / `Dual` / `Compensated` three ways, via `numax.tensor.map` at native SIMD width, checked against a reference. |
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

## advanced/

GPU + ensembles -- needs the actual hardware.

| File | What it shows |
|---|---|
| [gaussian_gpu.mojo](advanced/gaussian_gpu.mojo) | The same kernel and types as `basic/gaussian.mojo` run on GPU via `numax.tensor.map[gpu=True]`, no code changes versus the CPU example. |
| [ode.mojo](advanced/ode.mojo) | 1024 ODE trajectories, one GPU thread each, with solution sensitivities from the same integrator. |

## GPU note

`gaussian_gpu.mojo`, `softmax.mojo`, and `ode.mojo` launch GPU kernels
alongside their CPU paths. They need real Metal hardware, so they're a
local-only check; `pixi run examples-cpu` skips them, `pixi run examples`
includes them.
