"""numax.signal: convolution, correlation, windows, filtering and spectral
estimation.

```mojo
from numax.signal import convolve, fftconvolve, get_window, filtfilt, savgol_filter
```

Two tiers, one import each, the same split `numax.linalg`, `numax.fft`,
`numax.optimize`, `numax.integrate` and `numax.interpolate` make:

| Import | Holds | Good for |
| --- | --- | --- |
| `numax.signal` | `Tensor`, `Plain`-only, tier 2 | a recording: `convolve`/`correlate` in `full`/`same`/`valid` as one launch of dot products, `fftconvolve` through `numax.fft`, and the window factories `boxcar`/`hann`/`hamming`/`blackman`/`bartlett`/`kaiser`/`get_window` in SciPy's symmetric and periodic forms; the filters `lfilter`/`lfilter_zi`/`filtfilt`/`sosfilt` (host recurrences), `medfilt`, `detrend`, `savgol_filter`, `resample` and the multiband `firwin` |
| `numax.signal.array` | `Array[T, n]` and `FloatLike`, tier 1 | a frame inside a kernel: the direct `convolve`/`correlate`, `lfilter`, the lowpass `firwin`, and the cosine windows as compile-time tables, all differentiating at `Dual` |

This surface is the `Tensor` one. `apply_window` has no `Tensor` spelling
because `numax.core.ops.multiply` already is one.

MAX ships the neural-network convolution (`nn.conv`, NHWC, channels and
filters, GPU-only with a pack-the-filter CPU sibling) and no
signal-processing one; `numax/signal/convolution.mojo` records why the
1-D direct sum is written here rather than routed there, and
`docs/parity.md` carries the disposition.
"""

from .convolution import convolve, correlate, fftconvolve, full, same, valid
from .windows import (
    bartlett,
    blackman,
    boxcar,
    get_window,
    hamming,
    hann,
    kaiser,
)
from .filters import (
    detrend,
    filtfilt,
    firwin,
    lfilter,
    lfilter_zi,
    medfilt,
    resample,
    savgol_filter,
    sosfilt,
)
