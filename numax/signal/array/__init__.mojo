"""`numax.signal` over `Array[T, n]` and `FloatLike` scalars: the
register-resident, `FloatLike`-generic tier.

```mojo
from numax.signal.array import convolve, correlate, hann, lfilter, firwin
```

**Opt-in, and one import away rather than in the flat surface.** The names
here are the same names `numax.signal` exports over `Tensor`, so a caller
gets one tier per import and the surface never resolves an overload by a
type the reader has to look up. `numax.signal`, `numax.prelude` and `numax`
itself export the `Tensor` tier; this subpackage is the other one, the same
split `numax.linalg.array`, `numax.fft.array`, `numax.optimize.array`,
`numax.integrate.array` and `numax.interpolate.array` make.

| Module | Holds |
| --- | --- |
| `signal` | `convolve` (`full`/`same`), `correlate`, `hann`/`hamming`/`blackman`/`apply_window`, `lfilter`, the lowpass `firwin` |

## Why this tier exists at all

Every routine is a direct sum over comptime-sized `Array`s with no
data-dependent branch, so it is **tier 1**: a convolution at `Dual`
differentiates with respect to the taps or the signal, a filter runs per
SIMD lane inside a GPU thread, and the windows are compile-time tables.
It is sized for the sequences that live inside a per-element kernel -- a
handful of taps against a short frame -- not for a recording.

## Where the two tiers part

Length, and the algorithm it justifies. Over a `Tensor`, `convolve` is one
`elementwise` launch of `m + k - 1` dot products and `fftconvolve` goes
through `numax.fft` when the kernel is long; the recursive filters run
their recurrence on the host, since a recurrence has no GEMM to feed; and
the spectral estimators frame a signal and transform every frame at once.
None of that fits in registers, and none of it differentiates.

## The cost of the split, stated plainly

Mojo deprecates importing one name from two modules, so a file that wants
both tiers of a name has to alias one of them:

```mojo
from numax.signal import convolve                    # Tensor
from numax.signal.array import convolve as convolve_a  # Array
```

That is the price of a flat surface that means exactly one thing.
"""

from .signal import (
    apply_window,
    blackman,
    convolve,
    correlate,
    firwin,
    full,
    hamming,
    hann,
    lfilter,
    same,
)
