# Quickstart

From a clean project to printed answers: a tensor, a derivative, a GPU
call. Every block below is a complete program; `pixi run doc-check`
compiles all of them.

## Install

Add numax to a pixi project:

```toml
[workspace]
channels = ["https://conda.modular.com/max", "conda-forge"]
preview = ["pixi-build"]

[dependencies]
numax = { git = "https://github.com/ehsanmok/numax.git" }
```

Pin a `tag = "..."` for a release. Inside a clone of numax itself, every
`mojo` call needs the repository on the import path: `pixi run mojo -I .
file.mojo`, or `pixi run run file.mojo`, which adds `-I .` for you.

## A tensor

```mojo
from numax.prelude import *


def main() raises:
    var x = linspace[5, DType.float64](0.0, 1.0)
    print(x)
    var y = exp(x) * 2.0 + 1.0
    print(y)
    var a = Static[DType.float64, 2, 2]([4.0, 2.0, 2.0, 3.0])
    print(cholesky(a))
    print(a @ a)
    var mask = x > 0.5
    print(mask)
```

`Static[dtype, *dims]` is a tensor whose shape is part of its type; the
factories (`linspace`, `zeros`, `eye`, ...) and the constructor take the
device last and default to the host. Operators, `@` and comparisons work
as in NumPy.

## A derivative

The same function, evaluated at a `Dual`, returns its value and its
derivative -- no second function, no finite difference.

```mojo
from numax import Dual, FloatLike, Plain

comptime P = Plain[DType.float64]


def f[T: FloatLike](x: T) -> T:
    return x * x.sin()


def main():
    var at = Dual[P].seed(2.0)
    var result = f(at)
    print("f(2)  =", result.value)
    print("f'(2) =", result.deriv)
```

## On a GPU

A tensor lives where its `DeviceContext` says. Routines take
`gpu: Bool = False`; operators follow the tensor.

```mojo
from max.gpu.host import DeviceContext
from numax.prelude import *


def main() raises:
    var gpu = DeviceContext()
    var x = linspace[1024, DType.float32](0.0, 1.0, gpu)
    var y = exp[gpu=True](x)
    var z = y * 2.0
    print(z.on_host(), z.to_host()[1023])
```

Use `float32` on a device: Metal has no `double`, and on CUDA everything
routed through `matmul` (the factorizations, `solve`, `eigh`, `svd`) is
`float32` only at the 26.6 pin. A `float64` device tensor falls back to the
host with a one-line notice.

## Next

`examples/` runs the rest of the library end to end, `docs/features.md`
lists every public name, and `llms.txt` is the one-page guide to the
traps.
