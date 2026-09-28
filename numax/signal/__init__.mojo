"""numax.signal: convolution, correlation, windows, filtering and spectral
estimation.

```mojo
from numax.signal import convolve, fftconvolve, get_window, filtfilt, savgol_filter
```

Two tiers under one import. A name the two tiers share is one function
with two overloads, picked by the argument -- a `Tensor` takes the device
tier, an `Array[T, n]` of `FloatLike` values the register tier -- and the
register tier's own names are exported here too:

| Tier | Holds | Good for |
| --- | --- | --- |
| `Tensor`, `Plain`-only, tier 2 | a recording: `convolve`/`correlate` in `MODE_FULL`/`MODE_SAME`/`MODE_VALID` as one launch of dot products, `fftconvolve` through `numax.fft`, and the window factories `boxcar`/`hann`/`hamming`/`blackman`/`bartlett`/`kaiser`/`tukey`/`gaussian`/`flattop`/`nuttall`/`chebwin`/`get_window` in SciPy's symmetric and periodic forms; the filters `lfilter`/`lfilter_zi`/`filtfilt`/`sosfilt`/`sosfilt_zi`/`sosfiltfilt` (host recurrences at `dtype`, in place over the tensor's own host mapping), `medfilt`, `detrend`, `savgol_filter`, `resample` and the multiband `firwin`; the spectral estimators `periodogram`/`welch`/`spectrogram`/`stft` as one batched transform each, `hilbert`, `find_peaks`, and the IIR design family -- `butter`/`cheby1`/`cheby2`/`ellip` in all four band shapes behind one `iirfilter` front door -- with `freqz`, `output=` for `(b, a)`, zeros-poles-gain or second-order sections, and the conversions `tf2zpk`/`zpk2sos`/`tf2sos`/`sos2tf`/`sosfreqz` |
| `Array[T, n]` and `FloatLike`, tier 1 | a frame inside a kernel: the direct `convolve`/`correlate`, `lfilter`, the lowpass `firwin`, and the cosine windows as compile-time tables, all differentiating at `Dual` |

`apply_window` has no `Tensor` spelling
because `numax.core.ops.multiply` already is one.

MAX ships the neural-network convolution (`nn.conv`, NHWC, channels and
filters, GPU-only with a pack-the-filter CPU sibling) and no
signal-processing one; `numax/signal/convolution.mojo` records why the
1-D direct sum is written here rather than routed there, and
`docs/parity.md` carries the disposition.
"""

from .convolution import (
    convolve,
    correlate,
    fftconvolve,
    MODE_FULL,
    oaconvolve,
    MODE_SAME,
    MODE_VALID,
)
from .windows import (
    bartlett,
    blackman,
    boxcar,
    chebwin,
    flattop,
    gaussian,
    get_window,
    hamming,
    hann,
    kaiser,
    nuttall,
    tukey,
)
from .filters import (
    decimate,
    detrend,
    filtfilt,
    firwin,
    lfilter,
    lfilter_zi,
    medfilt,
    resample,
    savgol_filter,
    sosfilt,
    sosfilt_zi,
    sosfiltfilt,
)
from .spectral import (
    STFT,
    CrossSpectrum,
    Periodogram,
    Spectrogram,
    coherence,
    csd,
    hilbert,
    istft,
    periodogram,
    spectrogram,
    stft,
    welch,
)
from .peaks import find_peaks, peak_prominences
from .design import (
    OUTPUT_BA,
    OUTPUT_SOS,
    OUTPUT_ZPK,
    FrequencyResponse,
    TransferFunction,
    ZerosPolesGain,
    butter,
    cheby1,
    cheby2,
    ellip,
    freqz,
    iirfilter,
    sos2tf,
    sosfreqz,
    tf2sos,
    tf2zpk,
    zpk2sos,
    zpk2tf,
)
from ._array import (
    apply_window,
)
