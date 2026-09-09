"""numax.fft: discrete Fourier transforms.

```mojo
from numax.fft import fft, ifft, rfft, fftfreq, rfftfreq
```

Radix-2 Cooley-Tukey, power-of-two by construction. MAX ships **no forward
transform at all** -- its only one is `nn.irfft`, inverse-only,
last-axis-only and NVIDIA-only over the private `_cufft` -- so this is an
**extend** rather than a delegation, written in MAX's idiom with `gpu: Bool`
selecting the target and the data device-resident between stages.

Two tiers, one import each, the same split `numax.linalg` makes:

| Import | Holds | Good for |
| --- | --- | --- |
| `numax.fft` | `Tensor`, `Plain`-only, tier 2 | the large transform: `log2(n) + 1` device launches, no host round trip |
| `numax.fft.array` | `Array[Complex[T], n]`, `FloatLike`-generic, tier 1 | the small one: register-resident, differentiates at `Dual`, runs per SIMD lane inside a kernel body |

This surface is the `Tensor` one, and it carries `fft`/`ifft`, `rfft`,
`fftfreq`/`rfftfreq` and the `Spectrum` pair they travel in. `fft2`,
`irfft`, `fftshift`/`ifftshift` and `circular_convolve` are `Array`-tier
only so far.
"""

from .fft import Spectrum, fft, fftfreq, ifft, rfft, rfftfreq
