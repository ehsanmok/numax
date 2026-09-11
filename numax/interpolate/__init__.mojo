"""numax.interpolate: interpolation and polynomial evaluation.

```mojo
from numax.interpolate import interp, horner
```

Two tiers, one import each, the same split `numax.linalg`, `numax.fft`,
`numax.optimize` and `numax.integrate` make:

| Import | Holds | Good for |
| --- | --- | --- |
| `numax.interpolate` | `Tensor`, `Plain`-only, tier 2 | a device buffer of samples queried at a tensor of points: `interp` (NumPy's linear lookup) and `horner` over a tensor of points |
| `numax.interpolate.array` | `Array[T, n]` and `FloatLike`, tier 1 | a handful of knots in registers: `horner`, the natural `CubicSpline` and the `Chebyshev` fit of a `FloatLike` function, all of which differentiate at `Dual` and run per SIMD lane inside a kernel |

This surface is the `Tensor` one. What separates the tiers is the interval
search: over a tensor a query bisects the knots, a data-dependent branch
that a per-lane `FloatLike` kernel cannot make, which is why the `Array`
spline scans every interval and blends instead.

MAX has no interpolation at arbitrary points -- its `nn.resize_*` kernels
resample a whole image onto a fixed grid by a scale factor -- so both tiers
are numax's own; `numax/interpolate/interp.mojo` records the gate.
"""

from .interp import horner, interp
