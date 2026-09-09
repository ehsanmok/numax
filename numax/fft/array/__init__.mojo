"""`numax.fft` over `Array[Complex[T], n]`: the register-resident,
`FloatLike`-generic tier.

```mojo
from numax.fft.array import fft, ifft, rfft, fftfreq, fftshift
```

**Opt-in, and one import away rather than in the flat surface.** The names
here are the same names `numax.fft` exports over `Tensor`, so a caller gets
one tier per import and the surface never resolves an overload by a type the
reader has to look up. `numax.fft`, `numax.prelude` and `numax` itself
export the `Tensor` tier; this subpackage is the other one. It is the same
split `numax.linalg.array` makes, for the same reasons.

| Module | Holds |
| --- | --- |
| `fft` | `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`, `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`, `circular_convolve` |

## Why this tier exists at all

It is not a fallback for small transforms, though it is faster there. It is
the only tier that **differentiates**: the butterfly is built from `Complex`
arithmetic and `Complex` is built from `FloatLike` arithmetic, so `fft` over
`Complex[Dual[Plain]]` returns the transform *and* its derivative with
respect to whatever the input was seeded on, with no adjoint rule written
anywhere. It is also the only tier that **runs per SIMD lane inside a GPU
kernel body** -- the data lives in registers, so a 64-point transform can
sit inside a `map[gpu=True]` step, one transform per lane, which is a shape
the `Tensor` tier cannot express at all.

**Tier 1** throughout: the size is a compile-time parameter, so every loop
bound is known and nothing branches per lane.

## Where the two tiers part

Size is the whole of it. This tier holds its data in an
`Array[Complex[Inner], n]`, a register/stack object whose twiddle table is
built by a `comptime for` over `n/2` entries, so compile time and register
pressure both grow with `n`. It is sized for the transforms that appear
*inside* a per-element kernel -- 16, 64, 256 points -- not for a
four-million-point spectrogram. `numax.fft` over `Tensor` is the one for
that, and it is `Plain`-only in exchange.

## The cost of the split, stated plainly

Mojo deprecates importing one name from two modules, so a file that wants
both tiers of a name has to alias one of them:

```mojo
from numax.fft import fft                  # Tensor
from numax.fft.array import fft as fft_a   # Array
```

That is the price of a flat surface that means exactly one thing. Files
wanting only one tier -- which is nearly all of them -- pay nothing.
"""

from .fft import (
    circular_convolve,
    fft,
    fft2,
    fftfreq,
    fftshift,
    ifft,
    ifft2,
    ifftshift,
    irfft,
    rfft,
    rfftfreq,
)
