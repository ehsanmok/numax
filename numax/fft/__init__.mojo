"""numax.fft: discrete Fourier transforms.

```mojo
from numax.fft import fft, ifft, rfft, irfft, fft2, rfft2, fftfreq, fftshift
```

Radix-2 Cooley-Tukey at a power of two, Bluestein's chirp-z at every other
length over `Tensor`. MAX ships **no forward transform at all** -- its only one is `nn.irfft`, inverse-only,
last-axis-only and NVIDIA-only over the private `_cufft` -- so this is an
**extend** rather than a delegation, written in MAX's idiom with `gpu: Bool`
selecting the target and the data device-resident between stages.

Two tiers, one import each, the same split `numax.linalg` makes:

| Import | Holds | Good for |
| --- | --- | --- |
| `numax.fft` | `Tensor`, `Plain`-only, tier 2 | the large transform: `log2(n) + 1` device launches, no host round trip |
| `numax.fft.array` | `Array[Complex[T], n]`, `FloatLike`-generic, tier 1 | the small one: register-resident, differentiates at `Dual`, runs per SIMD lane inside a kernel body |

This surface is the `Tensor` one, and it carries `fft`/`ifft`,
`rfft`/`irfft`, `fft2`/`ifft2`/`rfft2` (rectangular, where the `Array`
tier's is square), `fftshift`/`ifftshift` at rank 1 and 2,
`fftfreq`/`rfftfreq`, `next_fast_len` -- the next power of two, the length
this engine is fast at -- and the `Spectrum` pair they travel in.
`circular_convolve` is `Array`-tier only; over `Tensor` the same identity is
`numax.signal`'s to spell.
"""

from .fft import (
    Spectrum,
    fft,
    fft2,
    fftfreq,
    fftshift,
    ifft,
    ifft2,
    ifftshift,
    irfft,
    next_fast_len,
    rfft,
    rfft2,
    rfftfreq,
)
